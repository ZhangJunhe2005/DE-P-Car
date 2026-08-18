import pytest
import torch

from dep_car.core.state_contract import STATE_NORMALIZATION_SCALE
from dep_car.model.dep_car_net import DEPCarNetV1
from dep_car.model.symmetry import (
    mirror_candidate_values,
    mirror_depth,
    mirror_lidar_bev,
    mirror_scores,
    mirror_trajectory,
    mirror_vehicle_state,
)
from dep_car.training.stages import configure_training_stage, parameter_partitions
from dep_car.training.losses import kinematic_violation_mask


def inputs(batch=2):
    torch.manual_seed(17)
    depth = torch.rand(batch, 2, 96, 160)
    depth[:, 1] = (depth[:, 1] > 0.15).float()
    lidar = torch.rand(batch, 6, 160, 160)
    state = torch.zeros(batch, 9)
    state[:, 4] = 2.0
    state[:, 5] = torch.linspace(-0.3, 0.3, batch)
    state[:, 7] = 1.0
    gear = torch.tensor(([1, -1] * ((batch + 1) // 2))[:batch])
    return depth, lidar, state, gear


def test_formal_network_contract_and_no_gear_prediction():
    model = DEPCarNetV1().eval()
    output = model(*inputs())
    assert output.raw_residuals.shape == (2, 15, 4)
    assert output.residuals.shape == (2, 15, 4)
    assert output.controls.shape == (2, 15, 4)
    assert output.trajectories.shape == (2, 15, 11, 6)
    assert output.scores.shape == (2, 15)
    assert all(torch.isfinite(value).all() for value in output)
    assert not hasattr(output, "gear")
    assert model.predicts_gear is False
    assert not any("gear_head" in name or "gear_logits" in name for name, _ in model.named_modules())
    assert hasattr(model, "speed_embedding") and hasattr(model, "steering_embedding")
    assert not hasattr(model, "spatial_projection")


def test_amp_keeps_executable_rollout_in_fp32_and_preserves_gradients():
    torch.manual_seed(23)
    model = DEPCarNetV1().train()
    state = torch.zeros(2, 9)
    state[:, 0] = torch.tensor([0.4, -0.3])
    state[:, 2] = torch.tensor([0.15, -0.20])
    gear = torch.tensor([1, -1])
    raw = torch.randn(2, 15, 4, requires_grad=True)

    reference = model._rollout_fp32(state, gear, raw)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        mixed = model._rollout_fp32(state, gear, raw)

    for reference_value, mixed_value in zip(reference, mixed):
        assert mixed_value.dtype == torch.float32
        torch.testing.assert_close(mixed_value, reference_value, atol=0.0, rtol=0.0)
    violation = kinematic_violation_mask(
        mixed.trajectory,
        gear,
        lateral_acceleration_limit_mps2=1.0e6,
    )
    assert not bool(violation.any())
    mixed.trajectory[..., 1:4].square().mean().backward()
    assert raw.grad is not None
    assert torch.isfinite(raw.grad).all()
    assert float(raw.grad.abs().sum()) > 0.0


def test_state_normalization_uses_the_versioned_physical_scale():
    scale = torch.tensor(STATE_NORMALIZATION_SCALE)
    torch.testing.assert_close(
        DEPCarNetV1.normalize_state(scale), torch.ones(9), atol=0.0, rtol=0.0
    )


def test_modality_mask_is_effective_and_never_allows_sensorless_planning():
    model = DEPCarNetV1().eval()
    depth, lidar, state, gear = inputs(1)
    with torch.no_grad():
        depth_only_a = model(depth, lidar, state, gear, torch.tensor([[1.0, 0.0]]))
        depth_only_b = model(depth, lidar + 1000.0, state, gear, torch.tensor([[1.0, 0.0]]))
        lidar_only_a = model(depth, lidar, state, gear, torch.tensor([[0.0, 1.0]]))
        lidar_only_b = model(depth + 1000.0, lidar, state, gear, torch.tensor([[0.0, 1.0]]))
    for lhs, rhs in zip(depth_only_a, depth_only_b):
        torch.testing.assert_close(lhs, rhs, atol=1.0e-6, rtol=0.0)
    for lhs, rhs in zip(lidar_only_a, lidar_only_b):
        torch.testing.assert_close(lhs, rhs, atol=1.0e-6, rtol=0.0)
    with pytest.raises(ValueError, match="at least one"):
        model(depth, lidar, state, gear, torch.zeros(1, 2))
    with pytest.raises(ValueError, match="-1 or \\+1"):
        model(depth, lidar, state, torch.zeros(1, dtype=torch.long))


def test_missing_rows_never_enter_encoder_batchnorm_statistics():
    reference = DEPCarNetV1().train()
    mixed = DEPCarNetV1().train()
    mixed.load_state_dict(reference.state_dict())
    depth, lidar, state, gear = inputs(2)

    reference(depth[:1], lidar[:1], state[:1], gear[:1], torch.ones(1, 2))
    mixed(
        depth,
        lidar,
        state,
        gear,
        torch.tensor([[1.0, 1.0], [0.0, 1.0]]),
    )

    reference_depth_buffers = dict(reference.depth_encoder.named_buffers())
    mixed_depth_buffers = dict(mixed.depth_encoder.named_buffers())
    assert reference_depth_buffers.keys() == mixed_depth_buffers.keys()
    for name in reference_depth_buffers:
        torch.testing.assert_close(
            reference_depth_buffers[name], mixed_depth_buffers[name], atol=0.0, rtol=0.0
        )


def test_fully_inactive_encoder_is_not_executed():
    model = DEPCarNetV1().train()
    depth, lidar, state, gear = inputs(1)
    calls = []
    handle = model.lidar_encoder.register_forward_hook(
        lambda *_arguments: calls.append(True)
    )
    try:
        model(depth, lidar, state, gear, torch.tensor([[1.0, 0.0]]))
    finally:
        handle.remove()
    assert calls == []


def test_depth_pixel_validity_is_not_collapsed_into_far_depth():
    model = DEPCarNetV1().eval()
    depth = torch.ones(1, 2, 96, 160)
    valid = depth.clone()
    invalid = depth.clone()
    invalid[:, 1] = 0.0
    with torch.no_grad():
        valid_feature = model.depth_encoder(valid)
        invalid_feature = model.depth_encoder(invalid)
    assert float((valid_feature - invalid_feature).abs().max()) > 1.0e-5


@pytest.mark.parametrize("gear_value", (-1, 1))
def test_network_has_exact_group_averaged_reflection_equivariance(gear_value):
    model = DEPCarNetV1().eval()
    depth, lidar, state, _ = inputs(1)
    gear = torch.tensor([gear_value])
    with torch.no_grad():
        original = model(depth, lidar, state, gear)
        reflected = model(
            mirror_depth(depth), mirror_lidar_bev(lidar), mirror_vehicle_state(state), gear
        )
    torch.testing.assert_close(
        original.raw_residuals,
        mirror_candidate_values(reflected.raw_residuals, steering_channels=(0, 1)),
        atol=2.0e-6,
        rtol=1.0e-5,
    )
    torch.testing.assert_close(
        original.controls,
        mirror_candidate_values(reflected.controls, steering_channels=(0, 1)),
        atol=2.0e-6,
        rtol=1.0e-5,
    )
    torch.testing.assert_close(
        original.trajectories, mirror_trajectory(reflected.trajectories), atol=2.0e-6, rtol=1.0e-5
    )
    torch.testing.assert_close(original.scores, mirror_scores(reflected.scores), atol=2.0e-6, rtol=1.0e-5)


def test_candidate_and_score_parameter_ownership_is_disjoint():
    model = DEPCarNetV1()
    groups = parameter_partitions(model)
    assert not ({id(value) for value in groups["candidate"]} & {id(value) for value in groups["score"]})
    candidate = configure_training_stage(model, "candidate_capacity")
    assert candidate["candidate_trainable"] > 0 and candidate["score_trainable"] == 0
    score = configure_training_stage(model, "score_calibration")
    assert score["candidate_trainable"] == 0 and score["score_trainable"] > 0


def test_score_geometry_uses_bounded_residuals_for_extreme_logits():
    model = DEPCarNetV1().eval()
    depth, lidar, state, gear = inputs(1)
    captured = []
    handle = model.score_geometry_encoder.register_forward_pre_hook(
        lambda _module, arguments: captured.append(arguments[0].detach().clone())
    )
    original_forward = model.candidate_head.forward
    try:
        model.candidate_head.forward = lambda features: features.new_full(
            (*features.shape[:-1], 4), 1.0e6
        )
        with torch.no_grad():
            output = model(depth, lidar, state, gear)
    finally:
        model.candidate_head.forward = original_forward
        handle.remove()
    assert captured and all(torch.isfinite(value).all() for value in captured)
    assert max(float(value.abs().max()) for value in captured) < 3.0
    assert torch.isfinite(output.scores).all()
