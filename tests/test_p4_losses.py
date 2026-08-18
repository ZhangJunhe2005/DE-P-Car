from types import SimpleNamespace

import numpy as np
import pytest
import torch

from dep_car.core.occupancy import (
    FootprintConfig,
    OccupancyGrid2D,
    SWEPT_SUBSTEPS_PER_SEGMENT,
    densify_trajectory_se2,
)
from dep_car.model.ackermann_rollout import AckermannRolloutV1
from dep_car.model.symmetry import mirror_route, mirror_scores, mirror_trajectory
from dep_car.training.p4_dataset import _bev_distance_field, _signed_distance_field
from dep_car.training.losses import (
    DEPCarLossConfig,
    DEPCarLossWeights,
    DEPCarObjectiveV1,
    _aggregate_swept_safety_penalty,
    _densify_trajectories_se2,
    candidate_diversity_loss,
    comfort_loss,
    kinematic_loss,
    kinematic_violation_components,
    kinematic_violation_mask,
    route_guidance_loss,
    score_ranking_loss,
    swept_footprint_clearance,
    swept_footprint_loss,
    swept_map_footprint_clearance,
)


def straight_trajectories(batch=1, candidates=15, lateral=0.0, reverse=False):
    steps = 11
    output = torch.zeros(batch, candidates, steps, 6)
    output[..., 0] = torch.linspace(0.0, 1.0, steps)
    direction = -1.0 if reverse else 1.0
    output[..., 1] = direction * torch.linspace(0.0, 1.0, steps)
    output[..., 2] = lateral
    output[..., 4] = direction * 0.5
    return output


def point_distance_field(x, y, *, extent=2.0, size=80):
    resolution = 2.0 * extent / size
    coordinates = torch.linspace(-extent + 0.5 * resolution, extent - 0.5 * resolution, size)
    yy, xx = torch.meshgrid(coordinates, coordinates, indexing="ij")
    return torch.sqrt((xx - x) ** 2 + (yy - y) ** 2)[None, None]


def signed_field_from_occupancy(occupancy, resolution):
    return torch.from_numpy(
        _signed_distance_field(~np.asarray(occupancy, dtype=bool), resolution)
    )[None, None]


def test_swept_footprint_empty_collision_and_monotonic_clearance():
    trajectory = straight_trajectories(candidates=1)
    empty_loss, empty_clearance = swept_footprint_loss(
        trajectory, torch.full((1, 1, 80, 80), 8.0), extent_m=2.0
    )
    collision_loss, collision_clearance = swept_footprint_loss(
        trajectory, point_distance_field(0.5, 0.0), extent_m=2.0
    )
    near_loss, _ = swept_footprint_loss(
        trajectory, torch.full((1, 1, 80, 80), 0.20), extent_m=2.0
    )
    far_loss, _ = swept_footprint_loss(
        trajectory, torch.full((1, 1, 80, 80), 0.60), extent_m=2.0
    )
    assert float(empty_loss.max()) < 1.0e-8
    assert float(empty_clearance.min()) > 0.0
    assert float(collision_loss.min()) > 0.05
    assert float(collision_clearance.min()) < 0.0
    assert float(near_loss.min()) > float(far_loss.max())


def test_training_and_runtime_swept_footprint_agree_on_collision_sign():
    resolution, extent, size = 0.05, 2.0, 80
    occupancy = np.zeros((size, size), dtype=np.int8)
    obstacle = np.floor((np.asarray([0.5, 0.0]) + extent) / resolution).astype(int)
    occupancy[obstacle[1], obstacle[0]] = 100
    grid = OccupancyGrid2D(occupancy, resolution=resolution, origin=(-extent, -extent))
    trajectory = straight_trajectories(candidates=1)[0, 0].numpy()
    runtime_safe, _ = grid.swept_footprint_clearance(trajectory, FootprintConfig())
    distance = signed_field_from_occupancy(occupancy >= 50, resolution)
    training_clearance = swept_footprint_clearance(
        torch.from_numpy(trajectory)[None, None], distance, extent_m=extent
    )
    assert runtime_safe is False
    assert float(training_clearance.min()) < 0.0


def test_training_and_runtime_reject_same_grazing_obstacle_at_point_one_meter_grid():
    """Protect the one-cell diagonal inflation used at the safety boundary."""

    resolution, extent, size = 0.10, 2.0, 40
    occupancy = np.zeros((size, size), dtype=np.int8)
    obstacle = np.floor((np.asarray([0.5, 0.4]) + extent) / resolution).astype(int)
    occupancy[obstacle[1], obstacle[0]] = 100
    grid = OccupancyGrid2D(occupancy, resolution=resolution, origin=(-extent, -extent))
    trajectory = straight_trajectories(candidates=1)[0, 0].numpy()

    runtime_safe, _ = grid.swept_footprint_clearance(trajectory, FootprintConfig())
    distance = signed_field_from_occupancy(occupancy >= 50, resolution)
    training_clearance = swept_footprint_clearance(
        torch.from_numpy(trajectory)[None, None], distance, extent_m=extent
    )

    assert runtime_safe is False
    assert float(training_clearance.min()) < 0.0


def test_numpy_and_torch_continuous_sweep_are_identical_and_differentiable():
    source = np.asarray(
        (
            (0.0, -0.4, 0.1, np.deg2rad(179.0), -0.2, 0.1),
            (1.0, 0.4, -0.1, np.deg2rad(-179.0), -0.4, -0.1),
        ),
        dtype=np.float64,
    )
    expected = densify_trajectory_se2(source)
    torch_source = torch.tensor(source, dtype=torch.double)[None, None]
    torch_source.requires_grad_(True)
    actual = _densify_trajectories_se2(torch_source)[0, 0]
    assert actual.shape[0] == SWEPT_SUBSTEPS_PER_SEGMENT + 1
    torch.testing.assert_close(actual, torch.from_numpy(expected), atol=1.0e-12, rtol=0.0)
    actual[:, 1:4].square().sum().backward()
    assert torch_source.grad is not None
    assert torch.isfinite(torch_source.grad).all()


def test_sixteen_substeps_cover_the_maximum_frozen_rollout_envelope():
    """Every outer footprint centre advances by less than half a 5 cm diagonal."""

    rollout = AckermannRolloutV1()
    state = torch.zeros(1, 9)
    state[:, 0] = rollout.config.forward_speed_limit_mps
    state[:, 2] = rollout.config.steering_limit_rad
    saturated_residuals = torch.full((1, 15, 4), 20.0)
    source = rollout(
        state, torch.ones(1, dtype=torch.long), saturated_residuals
    ).trajectory
    dense = _densify_trajectories_se2(source)

    assert source.shape == (1, 15, 11, 6)
    assert dense.shape == (1, 15, 161, 6)
    assert float(source[..., 0].max()) == pytest.approx(1.25)
    assert float(source[..., 4].abs().max()) == pytest.approx(2.5)
    assert float(source[..., 5].abs().max()) == pytest.approx(
        rollout.config.steering_limit_rad
    )

    offsets = dense.new_tensor(FootprintConfig().longitudinal_offsets)
    yaw = dense[..., 3]
    heading = torch.stack((torch.cos(yaw), torch.sin(yaw)), dim=-1)
    centers = (
        dense[..., None, 1:3]
        + heading[..., None, :] * offsets[None, None, None, :, None]
    )
    center_step = torch.linalg.vector_norm(
        centers[..., 1:, :, :] - centers[..., :-1, :, :], dim=-1
    )
    finest_runtime_half_diagonal_m = 0.5 * np.sqrt(2.0) * 0.05
    assert float(center_step.max()) < finest_runtime_half_diagonal_m


def test_local_global_and_runtime_sweep_reject_obstacle_between_source_rows():
    resolution, extent, size = 0.05, 2.0, 80
    occupancy = np.zeros((size, size), dtype=np.int8)
    obstacle = np.floor((np.asarray([0.0, 0.0]) + extent) / resolution).astype(int)
    occupancy[obstacle[1], obstacle[0]] = 100
    field = signed_field_from_occupancy(occupancy >= 50, resolution)
    footprint = FootprintConfig(
        length=0.01, width=0.01, safety_margin=0.0, circle_count=1
    )
    trajectory = torch.zeros(1, 1, 2, 6)
    trajectory[..., 0] = torch.tensor([0.0, 1.0])
    trajectory[..., 1] = torch.tensor([-0.5, 0.5])
    trajectory.requires_grad_(True)

    local = swept_footprint_clearance(
        trajectory, field, extent_m=extent, footprint=footprint
    )
    mapped = swept_map_footprint_clearance(
        trajectory,
        field,
        torch.tensor([resolution]),
        torch.tensor([[-extent, -extent]]),
        torch.eye(4)[None],
        footprint=footprint,
    )
    grid = OccupancyGrid2D(
        occupancy, resolution=resolution, origin=(-extent, -extent)
    )
    runtime_safe, _ = grid.swept_footprint_clearance(
        trajectory.detach()[0, 0].numpy(), footprint
    )

    assert not runtime_safe
    assert float(local.min()) < 0.0
    assert float(mapped.min()) < 0.0
    torch.testing.assert_close(local, mapped, atol=2.0e-6, rtol=1.0e-5)
    local.sum().backward()
    assert trajectory.grad is not None and torch.isfinite(trajectory.grad).all()


def test_body_to_map_sdf_landmark_uses_ros_lower_left_coordinates():
    trajectory = straight_trajectories(candidates=1)
    local_field = point_distance_field(0.5, 0.0, extent=2.0, size=80)
    local = swept_footprint_clearance(trajectory, local_field, extent_m=2.0)
    mapped = swept_map_footprint_clearance(
        trajectory,
        local_field,
        torch.tensor([0.05]),
        torch.tensor([[-2.0, -2.0]]),
        torch.eye(4)[None],
    )
    torch.testing.assert_close(local, mapped, atol=2.0e-5, rtol=1.0e-4)


def test_local_and_global_signed_sdf_sampling_have_exact_body_frame_parity():
    resolution, extent, size = 0.10, 2.0, 40
    bev = np.zeros((6, size, size), dtype=np.float32)
    bev[5] = 1.0
    bev[0, 18:25, 27:31] = 1.0
    local_field = torch.from_numpy(_bev_distance_field(bev, resolution))[None, None]
    expected = signed_field_from_occupancy(bev[0] >= 0.5, resolution)
    torch.testing.assert_close(local_field, expected, atol=0.0, rtol=0.0)

    trajectory = straight_trajectories(candidates=1, lateral=0.2)
    local = swept_footprint_clearance(
        trajectory, local_field, extent_m=extent
    )
    mapped = swept_map_footprint_clearance(
        trajectory,
        local_field,
        torch.tensor([resolution]),
        torch.tensor([[-extent, -extent]]),
        torch.eye(4)[None],
    )
    torch.testing.assert_close(local, mapped, atol=2.0e-6, rtol=1.0e-5)


def test_signed_sdf_gives_a_directional_exit_gradient_inside_asymmetric_obstacle():
    resolution, extent, size = 0.10, 4.0, 80
    occupancy = np.zeros((size, size), dtype=bool)
    # The candidate is much closer to the right face than to the other faces.
    occupancy[20:60, 20:55] = True
    field = signed_field_from_occupancy(occupancy, resolution)
    trajectory = torch.zeros(1, 1, 11, 6)
    trajectory[..., 0] = torch.linspace(0.0, 1.0, 11)
    trajectory[..., 1] = -extent + (48.5 * resolution)
    trajectory[..., 2] = -extent + (40.5 * resolution)
    trajectory.requires_grad_(True)

    loss, minimum_clearance = swept_footprint_loss(
        trajectory, field, extent_m=extent
    )
    loss.sum().backward()

    assert float(minimum_clearance) < 0.0
    # dL/dx < 0 means gradient descent increases x, toward the nearest free face.
    assert float(trajectory.grad[..., 1].sum()) < -1.0e-5
    assert float(trajectory.grad[..., 1:3].abs().sum()) > 1.0e-5


def test_worst_and_cvar_barrier_prevent_single_contact_mean_dilution():
    clearance = torch.ones(1, 1, 11, 5)
    clearance[0, 0, 4, 2] = -0.10
    aggregate, minimum = _aggregate_swept_safety_penalty(clearance)
    elementwise = torch.nn.functional.softplus(-clearance / 0.04) * 0.04
    mean_only = elementwise.mean(dim=(-1, -2))
    worst = elementwise.amax(dim=(-1, -2))

    assert minimum.item() == pytest.approx(-0.10)
    assert aggregate.item() >= 0.50 * worst.item()
    assert aggregate.item() > 10.0 * mean_only.item()


def test_route_guidance_rewards_forward_and_reverse_progress_without_pi_bias():
    mask = torch.ones(1, 11, dtype=torch.bool)
    for reverse in (False, True):
        matching = straight_trajectories(candidates=1, reverse=reverse)
        shifted = straight_trajectories(candidates=1, lateral=0.8, reverse=reverse)
        route = torch.zeros(1, 11, 3)
        route[..., 0] = matching[:, 0, :, 1]
        matching_cost = route_guidance_loss(matching, route, mask)
        shifted_cost = route_guidance_loss(shifted, route, mask)
        assert float(matching_cost) < float(shifted_cost)
        assert float(matching_cost) < 0.05


def test_kinematic_and_comfort_penalize_violations_and_oscillation():
    legal = straight_trajectories(candidates=1)
    illegal = legal.clone()
    illegal[..., 4] = 3.5
    illegal[..., 5] = 0.8
    gear = torch.ones(1, dtype=torch.long)
    assert float(kinematic_loss(legal, gear)) < 1.0e-8
    assert float(kinematic_loss(illegal, gear)) > 0.1
    smooth = legal.clone()
    oscillating = legal.clone()
    oscillating[..., 5] = torch.tensor([0.0, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, 0.0])
    assert float(comfort_loss(smooth)) < 1.0e-8
    assert float(comfort_loss(oscillating)) > 1.0


def test_all_candidate_kinematic_margin_penalizes_non_top_k_violation():
    trajectories = straight_trajectories()
    trajectories[:, -1, :, 4] = 2.4  # hard-legal, outside the 10% operating margin
    output = SimpleNamespace(
        trajectories=trajectories,
        residuals=torch.zeros(1, 15, 4),
        scores=torch.zeros(1, 15),
    )
    route = torch.zeros(1, 11, 3)
    route[..., 0] = torch.linspace(0.0, 1.0, 11)
    arguments = dict(
        distance_field=torch.full((1, 1, 160, 160), 8.0),
        route=route,
        route_mask=torch.ones(1, 11, dtype=torch.bool),
        requested_gear=torch.ones(1, dtype=torch.long),
        stage="candidate_capacity",
    )
    enabled = DEPCarObjectiveV1()(output, **arguments)
    disabled_config = DEPCarLossConfig(
        weights=DEPCarLossWeights(kinematic_all=0.0)
    )
    disabled = DEPCarObjectiveV1(disabled_config)(output, **arguments)

    assert not bool(enabled["kinematic_violation"].any())
    assert float(enabled["kinematic_per_candidate"][0, -1]) > 0.0
    assert float(enabled["candidate"] - disabled["candidate"]) == pytest.approx(
        float(enabled["kinematic"] * 2.0), rel=1.0e-5, abs=1.0e-7
    )


def test_kinematic_acceleration_and_veto_are_relative_to_requested_gear():
    time = torch.tensor([0.0, 0.1])
    cases = (
        (-1, (0.0, -0.14), False),
        (-1, (-0.5, -0.31), False),
        (-1, (0.0, -0.16), True),
        (-1, (-0.5, -0.29), True),
        (1, (0.0, 0.14), False),
        (1, (0.5, 0.31), False),
        (1, (0.0, 0.16), True),
        (1, (0.5, 0.29), True),
    )
    for gear_value, speeds, expected_violation in cases:
        trajectory = torch.zeros(1, 1, 2, 6)
        trajectory[..., 0] = time
        trajectory[..., 4] = torch.tensor(speeds)
        gear = torch.tensor([gear_value])
        violation = kinematic_violation_mask(trajectory, gear)
        assert bool(violation.item()) is expected_violation
        loss = kinematic_loss(trajectory, gear)
        assert (float(loss) > 0.0) is expected_violation


def test_kinematic_veto_exposes_constraint_breakdown():
    trajectory = straight_trajectories(candidates=1)
    trajectory[..., 5] = 0.8
    components = kinematic_violation_components(
        trajectory, torch.ones(1, dtype=torch.long)
    )
    assert set(components) == {
        "opposite_motion",
        "speed_limit",
        "steering_limit",
        "steering_rate",
        "acceleration",
        "deceleration",
        "lateral_acceleration",
    }
    assert bool(components["steering_limit"].item())
    assert not bool(components["speed_limit"].item())


def test_diversity_detects_candidate_collapse():
    rollout = AckermannRolloutV1()
    for gear_value in (-1, 1):
        gear = torch.tensor([gear_value])
        canonical = rollout(
            torch.zeros(1, 9), gear, torch.zeros(1, 15, 4)
        ).trajectory
        collapsed = canonical[:, :1].expand(-1, 15, -1, -1).clone()
        canonical_loss = candidate_diversity_loss(canonical, gear)
        collapsed_loss = candidate_diversity_loss(collapsed, gear)
        assert float(canonical_loss) < 2.0e-4
        assert float(collapsed_loss) > 1.0e-2


def test_lower_is_better_score_loss_and_all_infeasible_is_finite():
    target = torch.tensor([[0.1, 1.0, 2.0] + [3.0] * 12])
    good_scores = target.clone()
    bad_scores = target.flip(-1)
    infeasible = torch.zeros_like(target, dtype=torch.bool)
    good = score_ranking_loss(good_scores, target, feasible=infeasible)
    bad = score_ranking_loss(bad_scores, target, feasible=infeasible)
    assert torch.isfinite(good).all()
    assert float(good) < float(bad)


def test_safety_and_guidance_are_reflection_invariant_and_differentiable():
    trajectory = AckermannRolloutV1()(
        torch.zeros(1, 9), torch.ones(1, dtype=torch.long), torch.randn(1, 15, 4) * 0.1
    ).trajectory.detach().requires_grad_(True)
    field = point_distance_field(0.7, 0.3, extent=2.0, size=80)
    route = torch.zeros(1, 11, 3)
    route[..., 0] = torch.linspace(0.0, 1.0, 11)
    route[..., 1] = 0.2
    mask = torch.ones(1, 11, dtype=torch.bool)
    safe, _ = swept_footprint_loss(trajectory, field, extent_m=2.0)
    guide = route_guidance_loss(trajectory, route, mask)
    mirrored_safe, _ = swept_footprint_loss(
        mirror_trajectory(trajectory), field.flip(-2), extent_m=2.0
    )
    mirrored_guide = route_guidance_loss(
        mirror_trajectory(trajectory), mirror_route(route), mask
    )
    torch.testing.assert_close(safe, mirror_scores(mirrored_safe), atol=2.0e-5, rtol=1.0e-4)
    torch.testing.assert_close(guide, mirror_scores(mirrored_guide), atol=2.0e-6, rtol=1.0e-5)
    (safe.mean() + guide.mean()).backward()
    assert trajectory.grad is not None and torch.isfinite(trajectory.grad).all()
    assert float(trajectory.grad.abs().sum()) > 0.0
