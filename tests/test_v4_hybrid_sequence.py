import torch

from dep_car.model.dep_car_net import DEPCarNetConfig
from dep_car.model.dep_car_net_v3 import DEPCarNetV3
from dep_car.model.dep_car_net_v4 import DEPCarNetV4
from dep_car.model.dep_car_net_v42 import DEPCarNetV42
from dep_car.model.hybrid_sequence_rollout import HybridSequenceAckermannRolloutV4
from dep_car.training.losses_v4 import (
    best_of_k_sequence_gear_loss,
    sequence_target_tokens,
)
from dep_car.training.losses_v41 import (
    balanced_binary_loss,
    hierarchy_margin_loss,
)


def test_v4_rollout_executes_complete_multileg_sequence_and_stops():
    rollout = HybridSequenceAckermannRolloutV4()
    state = torch.zeros(1, 9)
    state[:, 0] = 0.4
    current = torch.tensor([1])
    raw = torch.zeros(1, 15, 6, 4)
    logits = torch.full((1, 15, 6, 3), -4.0)
    logits[..., 0] = 4.0
    for action, token in enumerate((1, 2, 1)):
        logits[:, 0, action, 0] = -4.0
        logits[:, 0, action, token] = 5.0
    output = rollout(state, current, raw, logits)

    assert output.trajectory.shape == (1, 15, 31, 6)
    assert output.action_gears[0, 0].tolist() == [1, -1, 1, 0, 0, 0]
    assert output.action_mask[0, 0].tolist() == [True, True, True, False, False, False]
    assert output.shift_required[0, 0].tolist() == [False, True, True, False, False, False]
    assert float(output.transition_duration[0, 0, 1]) > 0.25
    assert torch.all(output.trajectory[..., 1:, 0] > output.trajectory[..., :-1, 0])
    assert torch.all(
        output.motion_gears.to(output.trajectory) * output.trajectory[..., 4]
        >= -1.0e-6
    )


def test_v4_network_outputs_fifteen_complete_sequences_without_base_authority():
    base = DEPCarNetV3(
        config=DEPCarNetConfig(enforce_reflection_equivariance=False)
    )
    model = DEPCarNetV4(base_model=base).eval()
    depth = torch.zeros(1, 2, 96, 160)
    depth[:, 1] = 1.0
    lidar = torch.zeros(1, 6, 160, 160)
    state = torch.zeros(1, 9)
    current = torch.zeros(1, dtype=torch.long)
    history = torch.zeros(1, 6)
    route = torch.zeros(1, 8, 3)
    route[:, :, 0] = torch.linspace(0.2, 2.0, 8)
    route_mask = torch.ones(1, 8, dtype=torch.bool)

    with torch.inference_mode():
        output = model(
            depth, lidar, state, current, history, route, route_mask
        )
    assert output.gear_logits.shape == (1, 15, 6, 3)
    assert output.controls.shape == (1, 15, 6, 4)
    assert output.trajectories.shape == (1, 15, 31, 6)
    assert output.scores.shape == (1, 15)
    assert torch.all(torch.isfinite(output.trajectories))
    assert all(not parameter.requires_grad for parameter in model.base_model.parameters())
    assert not hasattr(model, "gear_selector")


def test_v4_best_of_k_gear_loss_supports_multiple_reverse_legs():
    target_gears = torch.tensor([[1, -1, 1, -1, 1, 0]])
    target_mask = target_gears != 0
    target = sequence_target_tokens(target_gears, target_mask)
    assert target.tolist() == [[1, 2, 1, 2, 1, 0]]
    logits = torch.full((1, 15, 6, 3), -5.0)
    logits[:, :, :, 0] = 0.0
    for action, token in enumerate(target[0]):
        logits[0, 7, action, int(token)] = 6.0
    loss, per_candidate = best_of_k_sequence_gear_loss(
        logits, target_gears, target_mask
    )
    assert float(loss) < 0.01
    assert int(per_candidate.argmin(dim=1)[0]) == 7


def test_v41_hierarchy_prefers_viable_over_merely_safe_candidate():
    preferred = torch.tensor([[False, True, False]])
    good = torch.tensor([[1.0, 0.2, 1.2]], requires_grad=True)
    bad = torch.tensor([[0.1, 0.8, 1.2]], requires_grad=True)
    assert float(hierarchy_margin_loss(good, preferred, 0.35)) == 0.0
    loss = hierarchy_margin_loss(bad, preferred, 0.35)
    assert float(loss) > 1.0
    loss.backward()
    assert bad.grad is not None and torch.all(torch.isfinite(bad.grad))


def test_v41_balanced_binary_calibration_penalizes_both_classes():
    logits = torch.zeros(1, 10, requires_grad=True)
    target = torch.tensor([[True] * 9 + [False]])
    loss = balanced_binary_loss(logits, target)
    loss.backward()
    assert float(loss) > 0.0
    assert logits.grad[0, 0] < 0.0
    assert logits.grad[0, -1] > 0.0


def test_v41_hierarchy_is_finite_without_a_positive_or_negative_pair():
    scores = torch.zeros(2, 15, requires_grad=True)
    preferred = torch.stack((
        torch.zeros(15, dtype=torch.bool),
        torch.ones(15, dtype=torch.bool),
    ))
    loss = hierarchy_margin_loss(scores, preferred, 0.35)
    assert float(loss) == 0.0
    assert torch.isfinite(loss)
    loss.backward()
    assert scores.grad is not None and torch.all(torch.isfinite(scores.grad))


def test_v42_execution_score_penalizes_low_viability_without_new_state():
    base = DEPCarNetV4(
        base_model=DEPCarNetV3(
            config=DEPCarNetConfig(enforce_reflection_equivariance=False)
        )
    )
    calibrated = DEPCarNetV42(
        base_model=DEPCarNetV3(
            config=DEPCarNetConfig(enforce_reflection_equivariance=False)
        )
    )
    assert set(base.state_dict()) == set(calibrated.state_dict())
    assert calibrated.viability_risk_weight == 8.0
    assert calibrated.safety_risk_weight == 0.0
    assert calibrated.requires_mandatory_hard_veto is True
