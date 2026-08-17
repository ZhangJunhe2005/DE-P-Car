import torch

from dep_car.model.dep_car_net import DEPCarNetConfig, DEPCarNetV1
from dep_car.training.losses import DEPCarObjectiveV1
from dep_car.training.stages import configure_training_stage, parameter_partitions


def batch(batch_size=2):
    torch.manual_seed(23)
    depth = torch.rand(batch_size, 2, 96, 160)
    depth[:, 1] = 1.0
    lidar = torch.rand(batch_size, 6, 160, 160)
    state = torch.zeros(batch_size, 9)
    state[:, 4] = torch.tensor([2.0, -1.0][:batch_size])
    state[:, 5] = torch.tensor([0.2, -0.2][:batch_size])
    state[:, 7] = 1.0
    gear = torch.tensor([1, -1][:batch_size])
    route = torch.zeros(batch_size, 12, 3)
    for index in range(batch_size):
        route[index, :, 0] = torch.linspace(0.0, 2.0 * float(gear[index]), 12)
        route[index, :, 1] = state[index, 5] * torch.linspace(0.0, 1.0, 12)
    route_mask = torch.ones(batch_size, 12, dtype=torch.bool)
    distance = torch.full((batch_size, 1, 160, 160), 8.0)
    return depth, lidar, state, gear, route, route_mask, distance


def test_candidate_objective_reaches_both_encoders_state_queries_and_head_but_not_score():
    model = DEPCarNetV1()
    model.eval()
    groups = parameter_partitions(model)
    configure_training_stage(model, "candidate_capacity")
    depth, lidar, state, gear, route, route_mask, distance = batch()
    output = model(depth, lidar, state, gear)
    losses = DEPCarObjectiveV1()(
        output,
        distance_field=distance,
        route=route,
        route_mask=route_mask,
        requested_gear=gear,
        stage="candidate_capacity",
    )
    losses["total"].backward()
    active_prefixes = (
        "depth_encoder",
        "lidar_encoder",
        "state_encoder",
        "speed_embedding",
        "steering_embedding",
        "candidate_head",
    )
    for prefix in active_prefixes:
        gradients = [
            parameter.grad
            for name, parameter in model.named_parameters()
            if name.startswith(prefix) and parameter.requires_grad and parameter.grad is not None
        ]
        assert gradients, prefix
        assert all(torch.isfinite(value).all() for value in gradients)
        assert sum(float(value.abs().sum()) for value in gradients) > 1.0e-8
    assert all(parameter.grad is None for parameter in groups["score"])


def test_score_stage_updates_only_independent_score_partition():
    model = DEPCarNetV1()
    model.eval()
    groups = parameter_partitions(model)
    configure_training_stage(model, "score_calibration")
    depth, lidar, state, gear, route, route_mask, distance = batch()
    output = model(depth, lidar, state, gear)
    losses = DEPCarObjectiveV1()(
        output,
        distance_field=distance,
        route=route,
        route_mask=route_mask,
        requested_gear=gear,
        stage="score_calibration",
    )
    losses["total"].backward()
    assert all(parameter.grad is None for parameter in groups["candidate"])
    score_gradients = [parameter.grad for parameter in groups["score"] if parameter.grad is not None]
    assert score_gradients and all(torch.isfinite(value).all() for value in score_gradients)
    assert sum(float(value.abs().sum()) for value in score_gradients) > 1.0e-8


def test_tiny_nonzero_candidate_and_score_targets_can_be_overfit():
    torch.manual_seed(101)
    model = DEPCarNetV1(DEPCarNetConfig(enforce_reflection_equivariance=False)).eval()
    # This diagnostic isolates the newly initialized logical query/head path;
    # P5 remains responsible for training on the differentiable objective.
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(
            name.startswith(("speed_embedding", "steering_embedding", "candidate_tower", "candidate_head", "score_"))
        )
    depth, lidar, state, gear, *_ = batch()
    target_residual = torch.linspace(-0.25, 0.25, 15)[None, :, None].expand(2, -1, 4).clone()
    target_residual[..., 1] *= -0.5
    target_residual[..., 2] *= 0.25
    target_score = torch.linspace(0.1, 1.5, 15)[None, :].expand(2, -1)
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=8.0e-3)
    initial = None
    for _ in range(120):
        output = model(depth, lidar, state, gear)
        loss = (output.raw_residuals - target_residual).square().mean()
        loss = loss + (output.scores - target_score).square().mean()
        initial = float(loss.detach()) if initial is None else initial
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert float(loss.detach()) < initial * 0.01
    assert float((output.raw_residuals - target_residual).square().mean()) < 1.0e-4
    assert float((output.scores - target_score).square().mean()) < 2.0e-3
