import math

import numpy as np

from dep_car.core.occupancy import OccupancyGrid2D
from dep_car.core.planner import DeterministicPlanner, PlanningResult
from dep_car.core.types import Candidate, Gear, VehicleState
from dep_car.runtime.arrival import ArrivalState, GoalArrivalController
from dep_car.runtime.maneuver import CommittedManeuver, ManeuverState
from dep_car.runtime.p6_contract import build_p6_runtime_contract
from dep_car.runtime.online_sync import StampedHistory, interpolated, nearest
from dep_car.runtime.preprocessing import (
    build_policy_state,
    normalize_depth_metric,
    normalize_lidar_bev,
)
from dep_car.runtime.route_guidance import (
    apply_runtime_route_preference,
    corner_speed_limit,
    monotonic_route_index,
    route_turn_angle,
    segment_is_visible,
    visible_corridor_subgoal,
)
from dep_car.runtime.safety import (
    evaluate_learned_candidate_bank,
    kinematic_veto_reason,
)
from dep_car.runtime.start_robustness import audit_start_robustness


def trajectory(speed=0.1, steering=0.0, x_end=0.1):
    time = np.linspace(0.0, 1.0, 11)
    return np.column_stack(
        (
            time,
            np.linspace(0.0, x_end, 11),
            np.zeros(11),
            np.zeros(11),
            np.linspace(0.0, speed, 11),
            np.full(11, steering),
        )
    )


def candidate(candidate_id, *, score=0.0, speed=0.1, x_end=0.1, gear=Gear.FORWARD):
    return Candidate(
        candidate_id=candidate_id,
        speed_anchor=speed,
        steering_anchor=0.0,
        duration=1.0,
        trajectory=trajectory(speed=speed, x_end=x_end),
        gear=gear,
        learned_score=score,
    )


def test_online_depth_preprocessing_matches_network_shape_and_invalid_fill():
    depth = np.full((480, 640), 5.0, dtype=np.float32)
    depth[0, 0] = np.nan
    output = normalize_depth_metric(depth)
    assert output.shape == (2, 96, 160)
    assert output.dtype == np.float32
    assert np.all((output >= 0.0) & (output <= 1.0))
    assert np.any(output[0] == 0.5)


def test_online_bev_height_normalization_preserves_channel_contract():
    raw = np.zeros((6, 160, 160), dtype=np.float32)
    raw[0, 10, 20] = 1.0
    raw[1, 10, 20] = 0.5
    raw[2, 10, 20] = 0.05
    raw[3, 10, 20] = 1.30
    raw[4, 10, 20] = 0.25
    raw[5, 10, 20] = 1.0
    output = normalize_lidar_bev(raw)
    assert output.shape == raw.shape
    assert output[2, 10, 20] == 0.0
    assert output[3, 10, 20] == 1.0
    assert output[4, 10, 20] == 0.25


def test_policy_state_uses_stopped_boundary_for_opposite_requested_gear():
    state = build_policy_state(
        signed_speed=0.4,
        longitudinal_acceleration=0.2,
        steering=0.1,
        yaw_rate=0.2,
        subgoal_body=(-1.0, 0.0),
        heading_error=math.pi,
        reference_curvature=0.0,
        requested_gear=-1,
        current_gear=1,
    )
    assert state.shape == (9,)
    assert state[0] == 0.0
    assert state[1] == 0.0
    assert state[4] == -1.0
    assert abs(state[6]) < 1.0e-6
    assert state[7] == -1.0


def test_kinematic_veto_accepts_reverse_and_rejects_opposite_motion():
    reverse = candidate(0, speed=-0.1, x_end=-0.1, gear=Gear.REVERSE)
    assert kinematic_veto_reason(reverse) == ""
    reverse.trajectory[:, 4] = 0.1
    assert kinematic_veto_reason(reverse) == "kinematic_opposite_motion"


def test_learned_bank_applies_hard_veto_before_score_ranking():
    grid = OccupancyGrid2D(np.zeros((200, 200), dtype=np.int16), 0.1, (-10.0, -10.0))
    candidates = [candidate(index, score=float(index + 1)) for index in range(15)]
    candidates[0].learned_score = 0.0
    candidates[0].trajectory[:, 1] = 20.0  # Outside map: must never be restored by score.
    candidates[1].learned_score = 0.1
    result = evaluate_learned_candidate_bank(candidates, (1.0, 0.0), grid)
    assert result.executable
    assert not result.candidates[0].feasible
    assert result.candidates[0].veto_reason == "static_footprint_collision"
    assert result.selected.candidate_id == 1


def test_p6_runtime_contract_covers_ros_policy_and_safety_boundary():
    contract = build_p6_runtime_contract()
    assert contract["schema"] == "DEPCarP6RuntimeImplementationV1"
    assert "ros/dep_car_local_planner/scripts/policy_node.py" in contract["files"]
    assert "ros/dep_car_local_planner/scripts/local_planner_node.py" in contract["files"]
    assert len(contract["aggregate_sha256"]) == 64


def test_online_sync_uses_anchor_nearest_and_bracketed_sources():
    history = StampedHistory(4)
    for stamp, value in ((0.96, [0.0]), (1.00, [2.0]), (1.04, [4.0])):
        history.append(stamp, np.asarray(value))
    value, distance = nearest(history.snapshot(), 1.03, 0.05)
    assert np.allclose(value, [4.0])
    assert abs(distance - 0.01) < 1.0e-9
    value, distance = interpolated(history.snapshot(), 1.02, 0.03)
    assert np.allclose(value, [3.0])
    assert abs(distance - 0.02) < 1.0e-9


def test_online_sync_rejects_unbracketed_or_distant_sources_and_resets_clock():
    history = StampedHistory(3)
    history.append(1.0, [1.0])
    history.append(1.1, [2.0])
    assert interpolated(history.snapshot(), 1.05, 0.04) is None
    assert nearest(history.snapshot(), 1.3, 0.05) is None
    history.append(0.2, [3.0])
    assert len(history.snapshot()) == 1
    assert history.snapshot()[0][0] == 0.2


def test_open_p6_start_passes_position_and_heading_perturbation_gate():
    grid = OccupancyGrid2D(
        np.zeros((200, 200), dtype=np.int16), 0.1, (-10.0, -10.0)
    )
    evidence = audit_start_robustness(grid, (0.0, 0.0, 0.0))
    assert evidence["status"] == "PASS"
    assert evidence["perturbations"] == 27
    assert evidence["minimum_safe_ackermann_primitives"] == 10


def test_goal_arrival_slows_then_brakes_and_latches_hold():
    controller = GoalArrivalController()
    far = controller.update(2.0, 0.0, 0.8)
    near = controller.update(0.30, 0.0, 0.5)
    braking = controller.update(0.20, 0.1, 0.4)
    held = controller.update(0.21, 0.1, 0.02)
    drifted = controller.update(0.60, 1.0, 0.0)

    assert far.speed_limit_mps is None
    assert 0.1 <= near.speed_limit_mps < 0.8
    assert braking.state == ArrivalState.ACTIVE_BRAKING
    assert held.state == ArrivalState.HOLD
    assert drifted.state == ArrivalState.HOLD


def test_committed_maneuver_uses_longer_ninety_degree_leg_and_alternates():
    maneuver = CommittedManeuver()
    assert maneuver.begin(
        Gear.REVERSE, (0.0, 0.0), 1.0, (0.0, 2.0), 0.5 * math.pi
    )
    assert maneuver.target_distance_m > 1.0
    assert maneuver.body_subgoal()[0] < -1.0
    assert maneuver.body_subgoal()[1] < 0.0

    maneuver.observe((0.6, 0.0), 2.0)
    maneuver.observe((1.2, 0.0), 3.0)
    assert maneuver.state == ManeuverState.SETTLING
    assert not maneuver.finish_if_stopped(0.10)
    assert maneuver.finish_if_stopped(0.02)
    assert maneuver.recovery_gear_order(Gear.REVERSE) == (
        Gear.FORWARD,
        Gear.REVERSE,
    )


def test_terminal_alignment_keeps_a_strong_directional_turn_reference():
    maneuver = CommittedManeuver()
    proposed = maneuver.proposed_subgoal(
        Gear.REVERSE,
        (1.0, 0.0),
        math.pi,
        purpose="terminal_alignment",
    )
    assert proposed == (-1.15, -0.90)
    assert maneuver.begin(
        Gear.REVERSE,
        (0.0, 0.0),
        0.0,
        (1.0, 0.0),
        math.pi,
        purpose="terminal_alignment",
    )
    assert maneuver.body_subgoal() == (-1.15, -0.90)


def test_shortened_primitives_recover_space_hidden_by_full_horizon():
    data = np.zeros((200, 200), dtype=np.int16)
    origin = np.asarray((-10.0, -10.0))
    wall_x = int(np.floor((0.75 - origin[0]) / 0.1))
    data[:, wall_x : wall_x + 2] = 100
    grid = OccupancyGrid2D(data, 0.1, tuple(origin))
    planner = DeterministicPlanner()

    full = planner.plan(VehicleState(), (2.0, 0.0), grid)
    shortened = planner.plan(
        VehicleState(),
        (0.5, 0.0),
        grid,
        spatial_scales=(1.0, 0.75, 0.50, 0.35, 0.25),
    )

    assert not full.executable
    assert full.blocked_by_static
    assert shortened.executable
    assert shortened.retime_factor < 1.0


def test_reference_steering_breaks_equal_endpoint_ties_toward_route_control():
    grid = OccupancyGrid2D(
        np.zeros((200, 200), dtype=np.int16), 0.1, (-10.0, -10.0)
    )
    planner = DeterministicPlanner()
    result = planner.plan(
        VehicleState(),
        (0.0, 0.0),
        grid,
        requested_gear=Gear.FORWARD,
        target_heading=0.0,
        target_steering=0.52,
    )
    assert result.executable
    assert result.selected.steering_anchor > 0.0


def wall_corner_grid():
    data = np.zeros((120, 120), dtype=np.int16)
    origin = np.asarray((-3.0, -3.0))
    # A compact block interrupts the diagonal chord while leaving the L-shaped
    # route itself wider than the runtime circular visibility envelope.
    x0 = int(np.floor((0.35 - origin[0]) / 0.05))
    x1 = int(np.ceil((0.55 - origin[0]) / 0.05))
    y0 = int(np.floor((0.45 - origin[1]) / 0.05))
    y1 = int(np.ceil((0.65 - origin[1]) / 0.05))
    data[y0:y1, x0:x1] = 100
    return OccupancyGrid2D(data, 0.05, tuple(origin))


def test_visible_corridor_subgoal_cannot_jump_through_a_blind_corner():
    grid = wall_corner_grid()
    route = [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, math.pi / 2.0),
        (1.0, 1.0, math.pi / 2.0),
    ]
    index, travelled, _, reason = visible_corridor_subgoal(
        route, 0, (0.0, 0.0), grid, 4.0
    )
    assert index == 1
    assert travelled == 1.0
    assert reason == "visibility_boundary"
    assert not segment_is_visible(grid, (0.0, 0.0), (1.0, 1.0))


def test_monotonic_route_index_rejects_a_closer_branch_across_a_wall():
    data = np.zeros((120, 120), dtype=np.int16)
    origin = np.asarray((-3.0, -3.0))
    y0 = int(np.floor((0.30 - origin[1]) / 0.05))
    y1 = int(np.ceil((0.40 - origin[1]) / 0.05))
    x0 = int(np.floor((-0.30 - origin[0]) / 0.05))
    x1 = int(np.ceil((0.30 - origin[0]) / 0.05))
    data[y0:y1, x0:x1] = 100
    grid = OccupancyGrid2D(data, 0.05, tuple(origin))
    route = [
        (0.80, 0.0, 0.0),
        (1.00, 0.0, math.pi / 2.0),
        (1.00, 0.70, math.pi),
        (0.00, 0.70, math.pi),
    ]
    index, distance = monotonic_route_index(
        route, 0, (0.0, 0.0), grid=grid, maximum_search=4
    )
    assert index == 0
    assert abs(distance - 0.80) < 1.0e-9


def test_runtime_route_preference_rejects_short_corner_cutting():
    grid = OccupancyGrid2D(
        np.zeros((200, 200), dtype=np.int16), 0.05, (-5.0, -5.0)
    )
    time = np.asarray([0.0, 0.5, 1.0])
    cut = Candidate(
        0,
        1.0,
        0.0,
        1.0,
        np.column_stack(
            (time, [0.0, 0.5, 1.0], [0.0, 0.5, 1.0], [0.0] * 3, [0.1] * 3, [0.0] * 3)
        ),
    )
    follows = Candidate(
        1,
        1.0,
        0.0,
        1.0,
        np.column_stack(
            (time, [0.0, 1.0, 1.0], [0.0, 0.0, 1.0], [0.0] * 3, [0.1] * 3, [0.0] * 3)
        ),
    )
    result = PlanningResult(cut, [cut, follows], 1.0, False, False, 1)
    apply_runtime_route_preference(
        result,
        np.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]),
        grid,
    )
    assert result.selected.candidate_id == 1


def test_corner_speed_limit_only_caps_a_material_upcoming_turn():
    straight = np.asarray([[0.0, 0.0], [0.5, 0.0], [1.2, 0.0]])
    corner = np.asarray([[0.0, 0.0], [0.5, 0.0], [0.5, 1.0]])
    assert corner_speed_limit(route_turn_angle(straight)) is None
    assert corner_speed_limit(route_turn_angle(corner)) < 0.35
