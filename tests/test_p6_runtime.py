import math

import numpy as np
import pytest

from dep_car.core.occupancy import OccupancyGrid2D
from dep_car.core.planner import DeterministicPlanner, PlanningResult
from dep_car.core.types import Candidate, Gear, VehicleState
from dep_car.runtime.arrival import ArrivalState, GoalArrivalController
from dep_car.runtime.maneuver import (
    CommittedManeuver,
    ManeuverState,
    MeasuredPoseReplanGate,
    RouteRecoveryReplanGate,
)
from dep_car.runtime.forward_preference import (
    ForwardPreferenceState,
    ForwardPreferenceSupervisor,
    corridor_direction_body,
    navigation_authority_reference,
    route_requires_far_revalidation,
    terminal_capture_route_authorized,
)
from dep_car.runtime.p6_contract import build_p6_runtime_contract
from dep_car.runtime.online_sync import (
    StampedHistory,
    interpolated,
    nearest,
    newest_synchronized_anchor,
)
from dep_car.runtime.occupancy import (
    RuntimeOccupancyGrid2D,
    ego_unknown_clearance_mask,
)
from dep_car.runtime.preprocessing import (
    build_policy_state,
    normalize_depth_metric,
    normalize_lidar_bev,
)
from dep_car.runtime.route_guidance import (
    apply_corner_clearance_preference,
    apply_runtime_route_preference,
    corner_severity,
    corner_speed_limit,
    monotonic_route_index,
    monotonic_route_reference_body,
    route_reference_body,
    route_turn_angle,
    segment_is_visible,
    visible_corridor_subgoal,
)
from dep_car.runtime.safety import (
    evaluate_hybrid_sequence_candidate_bank,
    evaluate_learned_candidate_bank,
    hybrid_sequence_kinematic_veto_reason,
    kinematic_veto_reason,
)
from dep_car.runtime.hybrid_sequence import JointGearHistoryTracker
from dep_car.runtime.hybrid_execution import (
    HybridFirstActionLatch,
    align_trajectory_between_chassis_frames,
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


def hybrid_candidate(candidate_id, score=0.0):
    time = np.linspace(0.0, 3.0, 31)
    value = Candidate(
        candidate_id=candidate_id,
        speed_anchor=0.1,
        steering_anchor=0.0,
        duration=0.5,
        trajectory=np.column_stack(
            (
                time,
                0.1 * time,
                np.zeros(31),
                np.zeros(31),
                np.full(31, 0.1),
                np.zeros(31),
            )
        ),
        gear=Gear.FORWARD,
        learned_score=score,
    )
    value.action_gears = np.ones(6, dtype=np.int8)
    value.action_mask = np.ones(6, dtype=bool)
    value.action_durations = np.full(6, 0.5)
    value.shift_required = np.zeros(6, dtype=bool)
    value.transition_duration = np.zeros(6)
    value.motion_gears = np.ones(31, dtype=np.int8)
    return value


def test_hybrid_sequence_hard_veto_covers_complete_31_row_trajectory():
    grid = OccupancyGrid2D(
        np.zeros((200, 200), dtype=np.int16), 0.1, (-10.0, -10.0)
    )
    candidates = [hybrid_candidate(index, float(index)) for index in range(15)]
    candidates[0].trajectory[-1, 1] = 20.0
    result = evaluate_hybrid_sequence_candidate_bank(
        candidates, (1.0, 0.0), grid
    )
    assert result.executable
    assert result.selected.candidate_id == 1
    assert result.candidates[0].veto_reason == "static_footprint_collision"


def test_hybrid_sequence_rejects_hidden_opposite_action_motion():
    value = hybrid_candidate(0)
    value.action_gears[2] = -1
    assert (
        hybrid_sequence_kinematic_veto_reason(value)
        == "hybrid_action_opposite_motion"
    )


def test_online_joint_gear_history_uses_measured_motion_and_retains_stop_gear():
    tracker = JointGearHistoryTracker()
    current, history = tracker.observe(1.0, 0.4)
    assert current == 1 and history[0] == 1
    current, history = tracker.observe(1.5, 0.0)
    assert current == 0 and history[0] == 1
    current, history = tracker.observe(2.0, -0.3, recovery_mode=True)
    assert current == -1 and history[0] == -1
    assert history[4] == 1 and history[5] == 1


def test_hybrid_first_action_latch_preserves_learned_gear_not_old_candidate():
    reverse = candidate(0, score=0.0, speed=-0.1, x_end=-0.1, gear=Gear.REVERSE)
    forward = candidate(1, score=1.0, gear=Gear.FORWARD)
    reverse.action_durations = np.full(6, 0.5)
    forward.action_durations = np.full(6, 0.5)
    initial = PlanningResult(
        selected=reverse,
        candidates=[reverse, forward],
        retime_factor=1.0,
        blocked_by_static=False,
        blocked_by_dynamic=False,
        generation=1,
    )
    latch = HybridFirstActionLatch()
    selected, reason = latch.select(initial, 0.0)
    assert selected.selected.gear == Gear.REVERSE
    assert reason == "new_model_sequence"
    assert latch.observe_drive_authorized(1.0, Gear.REVERSE)

    # A fresh score can choose a new trajectory, but it cannot cancel the
    # learned reverse action before its own duration has elapsed.
    forward.learned_score = 0.0
    reverse.learned_score = 2.0
    refreshed = PlanningResult(
        selected=forward,
        candidates=[reverse, forward],
        retime_factor=1.0,
        blocked_by_static=False,
        blocked_by_dynamic=False,
        generation=2,
    )
    selected, reason = latch.select(refreshed, 1.2)
    assert selected.selected.gear == Gear.REVERSE
    assert selected.selected.candidate_id == reverse.candidate_id
    assert reason == "committed_sequence_action_0"

    selected, reason = latch.select(refreshed, 1.6)
    assert selected.selected.gear == Gear.FORWARD
    assert reason == "new_model_sequence"


def test_hybrid_first_action_latch_never_overrides_current_hard_veto():
    reverse = candidate(0, score=0.0, speed=-0.1, x_end=-0.1, gear=Gear.REVERSE)
    forward = candidate(1, score=1.0, gear=Gear.FORWARD)
    reverse.action_durations = np.full(6, 1.0)
    forward.action_durations = np.full(6, 1.0)
    result = PlanningResult(
        selected=reverse,
        candidates=[reverse, forward],
        retime_factor=1.0,
        blocked_by_static=False,
        blocked_by_dynamic=False,
        generation=1,
    )
    latch = HybridFirstActionLatch()
    latch.select(result, 0.0)
    latch.observe_drive_authorized(0.0, Gear.REVERSE)
    reverse.feasible = False
    result.selected = forward
    selected, reason = latch.select(result, 0.1)
    assert not selected.executable
    assert reason == "committed_sequence_action_unavailable"
    # It never executes the vetoed reverse trajectory.  If that committed
    # gear stays unavailable for the bounded safety wait, a fresh complete
    # model sequence may acquire control.
    selected, reason = latch.select(result, 1.0)
    assert selected.selected.gear == Gear.FORWARD
    assert reason == "new_model_sequence"


def test_hybrid_sequence_latch_advances_model_prefix_across_replans():
    reverse = candidate(0, score=0.0, speed=-0.1, x_end=-0.1, gear=Gear.REVERSE)
    forward = candidate(1, score=1.0, gear=Gear.FORWARD)
    reverse.action_gears = np.asarray([-1, 1, -1, 0, 0, 0], dtype=np.int8)
    reverse.action_mask = np.asarray([1, 1, 1, 0, 0, 0], dtype=bool)
    reverse.action_durations = np.asarray([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    forward.action_gears = np.asarray([1, -1, 1, 0, 0, 0], dtype=np.int8)
    forward.action_mask = np.asarray([1, 1, 1, 0, 0, 0], dtype=bool)
    forward.action_durations = np.asarray([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    result = PlanningResult(
        selected=reverse,
        candidates=[reverse, forward],
        retime_factor=1.0,
        blocked_by_static=False,
        blocked_by_dynamic=False,
        generation=1,
    )
    latch = HybridFirstActionLatch()
    selected, reason = latch.select(result, 0.0)
    assert reason == "new_model_sequence"
    assert latch.locked_sequence == (-1, 1, -1)
    assert latch.observe_drive_authorized(0.0, Gear.REVERSE)

    # Even though receding-horizon ranking still prefers a new reverse-first
    # plan, the next action of the already selected model sequence is forward.
    selected, reason = latch.select(result, 0.6)
    assert selected.selected.gear == Gear.FORWARD
    assert reason == "committed_sequence_action_1"
    assert latch.action_index == 1


def test_hybrid_motion_veto_allows_only_sub_deadband_shift_residual():
    value = hybrid_candidate(0)
    value.trajectory[0, 4] = -0.02
    value.motion_gears[0] = 1
    assert hybrid_sequence_kinematic_veto_reason(value) == ""
    value.trajectory[0, 4] = -0.04
    assert hybrid_sequence_kinematic_veto_reason(value) == "hybrid_opposite_motion"


def test_delayed_hybrid_trajectory_is_rigidly_aligned_to_current_chassis():
    value = trajectory(speed=0.2, x_end=1.0)
    value[:, 2] = np.linspace(0.0, 0.3, len(value))
    value[:, 3] = 0.2
    aligned = align_trajectory_between_chassis_frames(
        value,
        anchor_pose=(2.0, 3.0, math.pi / 2.0),
        current_pose=(2.0, 3.5, math.pi / 2.0),
    )
    # The vehicle moved 0.5 m along its old +x while CUDA was evaluating.
    assert np.allclose(aligned[:, 1], value[:, 1] - 0.5, atol=1.0e-9)
    assert np.allclose(aligned[:, 2], value[:, 2], atol=1.0e-9)
    assert np.allclose(aligned[:, 3], value[:, 3], atol=1.0e-9)
    assert np.allclose(aligned[:, (0, 4, 5)], value[:, (0, 4, 5)])
    # The source candidate remains immutable to the caller.
    assert value[0, 1] == 0.0


def test_p6_runtime_contract_covers_ros_policy_and_safety_boundary():
    contract = build_p6_runtime_contract()
    assert contract["schema"] == "DEPCarP6RuntimeImplementationV1"
    assert "ros/dep_car_local_planner/scripts/policy_node.py" in contract["files"]
    assert "ros/dep_car_local_planner/scripts/local_planner_node.py" in contract["files"]
    assert len(contract["aggregate_sha256"]) == 64


def test_lidar_unknown_self_clearance_contains_runtime_five_circle_footprint():
    resolution = 0.05
    size = 80
    coordinates = (np.arange(size, dtype=np.float64) + 0.5 - size // 2) * resolution
    clear_mask = ego_unknown_clearance_mask(
        coordinates,
        coordinates,
        resolution=resolution,
        blind_radius=0.45,
    )
    data = np.full((size, size), -1, dtype=np.int16)
    data[clear_mask] = 0
    grid = RuntimeOccupancyGrid2D(
        data,
        resolution,
        (-0.5 * size * resolution, -0.5 * size * resolution),
    )
    stationary = np.asarray([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    safe, clearance = grid.swept_footprint_clearance(stationary)
    assert safe
    assert clearance > 0.0


def test_lidar_self_clearance_never_erases_an_observed_obstacle():
    resolution = 0.05
    size = 80
    coordinates = (np.arange(size, dtype=np.float64) + 0.5 - size // 2) * resolution
    clear_mask = ego_unknown_clearance_mask(
        coordinates,
        coordinates,
        resolution=resolution,
        blind_radius=0.45,
    )
    data = np.full((size, size), -1, dtype=np.int16)
    obstacle_cell = (size // 2, size // 2 + 5)
    data[obstacle_cell] = 100
    data[clear_mask & (data < 0)] = 0
    assert data[obstacle_cell] == 100
    grid = RuntimeOccupancyGrid2D(
        data,
        resolution,
        (-0.5 * size * resolution, -0.5 * size * resolution),
    )
    stationary = np.asarray([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    safe, _ = grid.swept_footprint_clearance(stationary)
    assert not safe


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


def test_online_sync_falls_back_to_newest_fully_bracketed_anchor():
    anchors = ((1.00, "old"), (1.10, "latest"))
    state = ((0.98, np.array([0.0])), (1.02, np.array([2.0])))
    depth = ((0.99, "depth-old"), (1.09, "depth-new"))
    synchronized = newest_synchronized_anchor(
        anchors,
        {"state": (state, 0.03)},
        {"depth": (depth, 0.03)},
    )
    stamp, anchor, matches = synchronized
    assert stamp == 1.00
    assert anchor == "old"
    np.testing.assert_allclose(matches["state"][0], [1.0])
    assert matches["depth"][0] == "depth-old"
    assert abs(matches["depth"][1] - 0.01) < 1.0e-9


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


def test_position_only_arrival_ignores_rviz_arrow_heading():
    controller = GoalArrivalController()
    braking = controller.update(
        0.20,
        math.pi,
        0.4,
        heading_required=False,
    )
    held = controller.update(
        0.20,
        math.pi,
        0.01,
        heading_required=False,
    )
    assert braking.state == ArrivalState.ACTIVE_BRAKING
    assert held.state == ArrivalState.HOLD


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
    assert not maneuver.finish_if_stopped(0.02, 0.20)
    assert maneuver.finish_if_stopped(0.02)
    assert maneuver.recovery_gear_order(Gear.REVERSE) == (
        Gear.FORWARD,
        Gear.REVERSE,
    )


def test_exhausted_maneuver_releases_transaction_authority():
    maneuver = CommittedManeuver()
    for index in range(maneuver.config.maximum_legs):
        assert maneuver.begin(
            Gear.FORWARD,
            (float(index), 0.0),
            float(index),
            (1.0, 0.2),
            0.2,
        )
        maneuver.settle("test")
        assert maneuver.finish_if_stopped(0.0)
    assert maneuver.exhausted
    assert not maneuver.active
    assert not maneuver.begin(
        Gear.REVERSE, (0.0, 0.0), 10.0, (-1.0, -0.2), -0.2
    )


def test_forward_restoration_budget_can_be_renewed_once_by_caller():
    maneuver = CommittedManeuver()
    for index in range(maneuver.config.maximum_legs):
        assert maneuver.begin(
            Gear.FORWARD if index % 2 else Gear.REVERSE,
            (float(index), 0.0),
            float(index),
            (-1.0, 0.2),
            2.4,
            purpose="forward_restoration",
            turn_sign_hint=1.0,
        )
        maneuver.settle("test")
        assert maneuver.finish_if_stopped(0.0)
    retained_sign = maneuver.turn_sign
    retained_gear = maneuver.last_completed_gear
    assert maneuver.exhausted
    assert maneuver.renew_leg_budget()
    assert maneuver.leg_count == 0
    assert maneuver.turn_sign == retained_sign
    assert maneuver.last_completed_gear == retained_gear
    assert maneuver.begin(
        Gear.REVERSE,
        (0.0, 0.0),
        20.0,
        (-1.0, 0.2),
        2.4,
        purpose="forward_restoration",
        turn_sign_hint=1.0,
    )


def test_forward_restoration_latches_corridor_turn_side_across_gear_changes():
    maneuver = CommittedManeuver()
    # The far end of a winding route can be on the opposite lateral side from
    # its immediate direction.  The local corridor bearing must own the turn.
    proposed = maneuver.proposed_subgoal(
        Gear.FORWARD,
        (-1.0, -2.0),
        2.4,
        purpose="forward_restoration",
        turn_sign_hint=2.4,
    )
    assert proposed[0] == 0.85
    assert proposed[1] > 0.0
    assert maneuver.begin(
        Gear.FORWARD,
        (0.0, 0.0),
        0.0,
        (-1.0, -2.0),
        2.4,
        purpose="forward_restoration",
        turn_sign_hint=2.4,
    )
    assert maneuver.target_distance_m == 0.85
    maneuver.settle("test_leg")
    assert maneuver.finish_if_stopped(0.0)

    # Replanning may wrap the instantaneous bearing to the other sign.  A
    # committed multi-point turn must keep rotating in its original direction.
    proposed = maneuver.proposed_subgoal(
        Gear.REVERSE,
        (-1.0, 2.0),
        -2.8,
        purpose="forward_restoration",
        turn_sign_hint=-2.8,
    )
    assert proposed[1] < 0.0
    assert maneuver.begin(
        Gear.REVERSE,
        (0.0, 0.0),
        1.0,
        (-1.0, 2.0),
        -2.8,
        purpose="forward_restoration",
        turn_sign_hint=-2.8,
    )
    assert maneuver.turn_sign == 1.0
    assert maneuver.body_subgoal()[1] < 0.0


def test_forward_exit_keeps_committed_turn_side_after_reverse_leg():
    maneuver = CommittedManeuver()
    assert maneuver.begin(
        Gear.REVERSE,
        (0.0, 0.0),
        0.0,
        (-1.0, 0.1),
        3.10,
        purpose="forward_restoration",
        turn_sign_hint=3.10,
    )
    assert maneuver.turn_sign > 0.0
    maneuver.settle("target_distance_reached")
    assert maneuver.finish_if_stopped(0.0)
    assert maneuver.last_completed_reason == "target_distance_reached"

    proposed = maneuver.proposed_subgoal(
        Gear.FORWARD,
        (1.0, -0.6),
        -0.60,
        purpose="forward_restoration",
        turn_sign_hint=-0.60,
    )
    assert proposed[0] > 0.0
    assert proposed[1] > 0.0
    assert abs(proposed[1]) <= maneuver.config.lateral_offset_m
    assert maneuver.begin(
        Gear.FORWARD,
        (0.0, 0.0),
        1.0,
        (1.0, -0.6),
        -0.60,
        purpose="forward_restoration",
        turn_sign_hint=-0.60,
    )
    assert maneuver.turn_sign > 0.0
    assert maneuver.body_subgoal()[1] > 0.0

    maneuver.settle("target_distance_reached")
    assert maneuver.finish_if_stopped(0.0)
    proposed = maneuver.proposed_subgoal(
        Gear.REVERSE,
        (-1.0, 0.6),
        2.50,
        purpose="forward_restoration",
        turn_sign_hint=2.50,
    )
    assert proposed[1] < 0.0
    assert maneuver.begin(
        Gear.REVERSE,
        (0.0, 0.0),
        2.0,
        (-1.0, 0.6),
        2.50,
        purpose="forward_restoration",
        turn_sign_hint=2.50,
    )
    assert maneuver.turn_sign > 0.0
    maneuver.settle("target_distance_reached")
    assert maneuver.finish_if_stopped(0.0)

    # A dense frozen route may wrap its instantaneous bearing after every
    # parking leg.  Never reinterpret that wrap as permission to reverse the
    # already committed turn side.
    proposed = maneuver.proposed_subgoal(
        Gear.FORWARD,
        (1.0, 0.6),
        0.60,
        purpose="forward_restoration",
        turn_sign_hint=0.60,
    )
    assert proposed[1] > 0.0


def test_interleg_revalidation_preserves_turnaround_authority():
    supervisor = ForwardPreferenceSupervisor()
    front = np.asarray([[0.0, 0.0], [0.5, 0.2], [1.0, 0.3]])

    supervisor.request_route_revalidation()
    supervisor.approve_continuation_route()
    decision = supervisor.update(
        front,
        turnaround_feasible=True,
        forward_capture_feasible=True,
        forward_exit_verified=False,
        turnaround_start_authorized=False,
    )
    assert decision.start_turnaround
    assert decision.state == ForwardPreferenceState.TURNAROUND_PENDING

    # Completion requires a measured full forward leg plus the configured
    # number of stable forward probes; route refresh alone cannot end it.
    supervisor.request_route_revalidation()
    supervisor.approve_continuation_route()
    decisions = [
        supervisor.update(
            front,
            turnaround_feasible=True,
            forward_capture_feasible=True,
            forward_exit_verified=True,
            turnaround_start_authorized=False,
        )
        for _ in range(supervisor.config.forward_confirmation_cycles)
    ]
    assert decisions[-1].reason == "forward_corridor_reacquired"


def test_planner_hard_filters_opposite_turn_during_committed_maneuver():
    planner = DeterministicPlanner()
    occupancy = OccupancyGrid2D(
        np.zeros((160, 160), dtype=np.int8),
        resolution=0.1,
        origin=(-8.0, -8.0),
    )
    result = planner.plan(
        VehicleState(),
        (1.0, 0.35),
        occupancy,
        requested_gear=Gear.FORWARD,
        required_yaw_direction=1.0,
        minimum_yaw_progress_rad=0.035,
    )
    assert result.executable
    assert result.selected.trajectory[-1, 3] >= 0.035
    rejected = [
        item
        for item in result.candidates
        if item.veto_reason == "required_yaw_progress_not_met"
    ]
    assert rejected
    assert all(item.trajectory[-1, 3] < 0.035 for item in rejected)


def test_measured_pose_replan_gate_prevents_stationary_retry_loop():
    gate = MeasuredPoseReplanGate(0.25)
    assert gate.authorize((1.0, 2.0))
    assert not gate.authorize((1.10, 2.10))
    assert gate.authorize((1.26, 2.0))
    gate.reset()
    assert gate.authorize((1.10, 2.10))


def test_route_recovery_gate_waits_for_new_route_or_measured_displacement():
    gate = RouteRecoveryReplanGate(0.25)
    route = ("FAR_ATTEMPTABLE_NAVIGATION", "route-a", 7, 11)
    gate.block(route, (1.0, 2.0))

    assert gate.held(route, (1.10, 2.10))
    assert not gate.held(("FAR_ATTEMPTABLE_NAVIGATION", "route-a", 8, 12), (1.0, 2.0))

    gate.block(route, (1.0, 2.0))
    assert not gate.held(route, (1.26, 2.0))


def test_committed_maneuver_pauses_distance_and_timeout_during_gear_shift():
    maneuver = CommittedManeuver()
    assert maneuver.begin(
        Gear.REVERSE, (0.0, 0.0), 1.0, (-1.0, -0.3), -0.4
    )
    maneuver.hold_for_drive_authorization((0.20, 0.0), 9.0)

    assert maneuver.travelled_m == 0.0
    assert maneuver.state == ManeuverState.DRIVE_LEG
    maneuver.observe((0.30, 0.0), 9.1)
    assert math.isclose(maneuver.travelled_m, 0.10)
    assert maneuver.state == ManeuverState.DRIVE_LEG


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


def test_memory_backtrack_can_exit_soft_margin_but_never_physical_overlap():
    resolution = 0.05
    data = np.zeros((160, 160), dtype=np.int16)
    origin = np.asarray((-4.0, -4.0))
    # The wall is inside the normal 12 cm soft margin at t=0, while still
    # leaving the physical Urban Car footprint clear.  Normal planning must
    # remain fail-closed; explicit breadcrumb backtracking may only choose a
    # reverse primitive whose signed margin improves.
    wall_x = int(np.floor((0.475 - origin[0]) / resolution))
    data[:, wall_x] = 100
    grid = RuntimeOccupancyGrid2D(data, resolution, tuple(origin))
    planner = DeterministicPlanner()

    normal = planner.plan(
        VehicleState(),
        (-1.0, 0.0),
        grid,
        requested_gear=Gear.REVERSE,
        spatial_scales=(1.0, 0.75, 0.50, 0.35, 0.25),
    )
    egress = planner.plan(
        VehicleState(),
        (-1.0, 0.0),
        grid,
        requested_gear=Gear.REVERSE,
        spatial_scales=(1.0, 0.75, 0.50, 0.35, 0.25),
        allow_static_margin_egress=True,
    )

    assert not normal.executable
    assert normal.blocked_by_static
    assert egress.executable
    assert egress.selected.gear == Gear.REVERSE
    strict = grid.swept_footprint_signed_clearance_profile(
        egress.selected.trajectory
    )
    assert strict[0] <= 0.0
    assert strict[-1] >= strict[0] + 0.02


def test_memory_margin_egress_cannot_restore_a_physical_collision():
    resolution = 0.05
    data = np.zeros((160, 160), dtype=np.int16)
    origin = np.asarray((-4.0, -4.0))
    wall_x = int(np.floor((0.30 - origin[0]) / resolution))
    data[:, wall_x] = 100
    grid = RuntimeOccupancyGrid2D(data, resolution, tuple(origin))

    result = DeterministicPlanner().plan(
        VehicleState(),
        (-1.0, 0.0),
        grid,
        requested_gear=Gear.REVERSE,
        spatial_scales=(1.0, 0.75, 0.50, 0.35, 0.25),
        allow_static_margin_egress=True,
    )

    assert not result.executable
    assert result.blocked_by_static


def test_accumulated_map_allows_unexplored_extent_but_keeps_known_wall_veto():
    resolution = 0.10
    origin = (-2.0, -2.0)
    trajectory_at_frontier = np.asarray([[0.0, 1.90, 0.0, 0.0]])

    unexplored = RuntimeOccupancyGrid2D(
        np.zeros((40, 40), dtype=np.int16),
        resolution,
        origin,
        unknown_is_occupied=False,
    )
    strict = unexplored.swept_footprint_signed_clearance_profile(
        trajectory_at_frontier
    )
    exploratory = unexplored.swept_footprint_signed_clearance_profile(
        trajectory_at_frontier,
        outside_is_occupied=False,
    )
    assert np.isneginf(strict[0])
    assert np.isposinf(exploratory[0])

    known_wall = np.zeros((40, 40), dtype=np.int16)
    known_wall[:, 37] = 100
    accumulated = RuntimeOccupancyGrid2D(
        known_wall,
        resolution,
        origin,
        unknown_is_occupied=False,
    )
    clearance = accumulated.swept_footprint_signed_clearance_profile(
        trajectory_at_frontier,
        outside_is_occupied=False,
    )
    assert clearance[0] < 0.0


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


def test_frozen_dense_far_route_skips_its_already_driven_prefix():
    route = np.column_stack((np.linspace(0.0, 5.0, 101), np.zeros(101)))
    pose = (2.0, 0.0, 0.0)
    stale = route_reference_body(route, pose, 0, horizon_m=0.75)
    reference, index = monotonic_route_reference_body(
        route, pose, 0, horizon_m=0.75, maximum_search=60
    )

    assert corridor_direction_body(stale, 0.75)[0] > 3.0
    assert index >= 39
    assert abs(corridor_direction_body(reference, 0.75)[0]) < 1.0e-9

    # Reverse-space creation must not roll the frozen cursor back to an old
    # point and manufacture another rear-route transaction.
    reference, next_index = monotonic_route_reference_body(
        route, (1.8, 0.0, 0.0), index, horizon_m=0.75, maximum_search=60
    )
    assert next_index >= index
    assert abs(corridor_direction_body(reference, 0.75)[0]) < 1.0e-9


def test_terminal_capture_requires_explicit_far_or_known_terminal_authority():
    assert terminal_capture_route_authorized("FAR_KNOWN_VISIBILITY")
    assert terminal_capture_route_authorized("FAR_ATTEMPTABLE_NAVIGATION")
    assert terminal_capture_route_authorized("KNOWN_TERMINAL_DIRECT")
    assert not terminal_capture_route_authorized("EXPLORED_TOPOLOGY")
    assert not terminal_capture_route_authorized("LOCAL_SAFE_EXPLORATION")


def test_only_actual_far_route_revisions_require_far_revalidation():
    assert route_requires_far_revalidation("FAR_KNOWN_VISIBILITY")
    assert route_requires_far_revalidation("FAR_ATTEMPTABLE_NAVIGATION")
    assert not route_requires_far_revalidation("EXPLORED_TOPOLOGY")
    assert not route_requires_far_revalidation("LOCAL_SAFE_EXPLORATION")
    assert not route_requires_far_revalidation("BOUNDARY_FOLLOW_LEFT")


def test_rear_explored_topology_can_enter_bounded_turnaround_without_far():
    supervisor = ForwardPreferenceSupervisor()
    forward = np.asarray([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]])
    rear = np.asarray([[0.0, 0.0], [-0.5, -0.1], [-1.0, -0.2]])
    supervisor.update(forward, turnaround_feasible=True)
    decision = supervisor.update(rear, turnaround_feasible=True)
    assert decision.state == ForwardPreferenceState.ROUTE_REVALIDATION
    assert not route_requires_far_revalidation("EXPLORED_TOPOLOGY")

    supervisor.approve_revalidated_route()
    decisions = [
        supervisor.update(rear, turnaround_feasible=True)
        for _ in range(supervisor.config.behind_confirmation_cycles)
    ]
    assert decisions[-1].state == ForwardPreferenceState.TURNAROUND_PENDING
    assert decisions[-1].start_turnaround
    assert not terminal_capture_route_authorized("BOUNDARY_FOLLOW_RIGHT")


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


def test_corner_speed_limit_allows_safe_cruise_on_a_gentle_bend():
    limit = corner_speed_limit(0.25)
    assert 1.5 < limit < 2.0
    assert corner_speed_limit(0.5 * math.pi) == pytest.approx(0.26)


def test_corner_soft_clearance_prefers_outer_candidate_without_hard_blocking():
    class EndpointClearance:
        @staticmethod
        def swept_footprint_clearance(trajectory):
            # The inner candidate approaches the synthetic wall corner while
            # the wider candidate keeps useful swept-footprint margin.
            clearance = 0.04 if trajectory[-1, 2] > 0.5 else 0.24
            return True, clearance

    time = np.asarray([0.0, 0.5, 1.0])
    inner = Candidate(
        0,
        0.2,
        0.4,
        1.0,
        np.column_stack(
            (time, [0.0, 0.3, 0.6], [0.0, 0.4, 0.8], [0.0] * 3, [0.2] * 3, [0.4] * 3)
        ),
        learned_score=0.0,
    )
    outer = Candidate(
        1,
        0.2,
        0.2,
        1.0,
        np.column_stack(
            (time, [0.0, 0.4, 0.8], [0.0, 0.1, 0.2], [0.0] * 3, [0.2] * 3, [0.2] * 3)
        ),
        learned_score=0.25,
    )
    result = PlanningResult(inner, [inner, outer], 1.0, False, False, 1)
    corner = np.asarray([[0.0, 0.0], [0.5, 0.0], [0.5, 1.0]])
    apply_corner_clearance_preference(
        result,
        corner,
        EndpointClearance(),
        learned_score_base=True,
    )
    assert result.selected.candidate_id == 1
    assert inner.feasible and outer.feasible
    assert inner.guidance_cost > outer.guidance_cost


def test_corner_soft_clearance_survives_short_local_route_handoff():
    class EndpointClearance:
        @staticmethod
        def swept_footprint_clearance(trajectory):
            clearance = 0.05 if trajectory[-1, 2] > 0.5 else 0.25
            return True, clearance

    time = np.asarray([0.0, 0.5, 1.0])
    inner = Candidate(
        0,
        0.2,
        0.4,
        1.0,
        np.column_stack(
            (
                time,
                [0.0, 0.3, 0.6],
                [0.0, 0.4, 0.8],
                [0.0, 0.22, 0.44],
                [0.2] * 3,
                [0.4] * 3,
            )
        ),
        learned_score=0.0,
    )
    outer = Candidate(
        1,
        0.2,
        0.2,
        1.0,
        np.column_stack(
            (
                time,
                [0.0, 0.4, 0.8],
                [0.0, 0.1, 0.2],
                [0.0, 0.16, 0.32],
                [0.2] * 3,
                [0.2] * 3,
            )
        ),
        learned_score=0.20,
    )
    result = PlanningResult(inner, [inner, outer], 1.0, False, False, 1)
    short_local_reference = np.asarray(
        [[0.0, 0.0], [0.6, 0.0], [1.2, 0.20]]
    )
    assert corner_severity(short_local_reference) == 0.0
    apply_corner_clearance_preference(
        result,
        short_local_reference,
        EndpointClearance(),
        learned_score_base=True,
    )
    assert result.corner_soft_route_severity == 0.0
    assert result.corner_soft_candidate_severity > 0.0
    assert result.corner_soft_applied
    assert result.selected.candidate_id == 1
    assert inner.feasible and outer.feasible


def test_corner_soft_clearance_is_inactive_on_straight_route():
    straight = np.asarray([[0.0, 0.0], [0.5, 0.0], [1.2, 0.0]])
    corner = np.asarray([[0.0, 0.0], [0.5, 0.0], [0.5, 1.0]])
    assert corner_severity(straight) == 0.0
    assert corner_severity(corner) == 1.0


def test_forward_preference_leaves_ninety_degree_corner_to_local_planner():
    supervisor = ForwardPreferenceSupervisor()
    corner = np.asarray([[0.0, 0.0], [0.5, 0.0], [0.5, 1.0]])
    bearing, _ = corridor_direction_body(corner)
    decision = supervisor.update(corner, turnaround_feasible=True)
    assert abs(bearing) < supervisor.config.behind_bearing_rad
    assert decision.state == ForwardPreferenceState.FORWARD_CRUISE
    assert decision.requested_gear == Gear.FORWARD
    assert not decision.start_turnaround


def test_local_exploration_turnaround_uses_mission_direction_not_passed_carrot():
    short_carrot = np.asarray(((0.0, 0.0), (-0.8, 0.0)), dtype=float)
    first = navigation_authority_reference(
        short_carrot,
        (-8.0, 0.0),
        "LOCAL_SAFE_EXPLORATION",
        horizon_m=2.5,
    )
    # After one parking-style leg the old local point can be behind/right,
    # but the still-distant mission direction is already in front/left.  The
    # direction authority must follow the mission and must not trigger a new
    # turnaround to chase the disposable point.
    passed_carrot = np.asarray(((0.0, 0.0), (-0.4, -0.2)), dtype=float)
    after_turn = navigation_authority_reference(
        passed_carrot,
        (7.0, 0.5),
        "LOCAL_SAFE_EXPLORATION",
        horizon_m=2.5,
    )
    far_route = navigation_authority_reference(
        passed_carrot,
        (7.0, 0.5),
        "FAR_KNOWN_VISIBILITY",
        horizon_m=2.5,
    )

    assert math.isclose(
        abs(corridor_direction_body(first)[0]), math.pi, abs_tol=1.0e-9
    )
    assert abs(corridor_direction_body(after_turn)[0]) < 0.1
    np.testing.assert_allclose(far_route, passed_carrot)


def test_forward_preference_turns_before_long_reverse_when_space_is_open():
    supervisor = ForwardPreferenceSupervisor()
    behind = np.asarray([[0.0, 0.0], [-0.5, 0.0], [-1.0, 0.2]])
    decisions = [
        supervisor.update(behind, turnaround_feasible=True) for _ in range(3)
    ]
    assert all(
        decision.state == ForwardPreferenceState.TURNAROUND_CONFIRM
        and decision.requested_gear == Gear.NEUTRAL
        for decision in decisions[:2]
    )
    assert decisions[-1].state == ForwardPreferenceState.TURNAROUND_PENDING
    assert decisions[-1].start_turnaround
    assert decisions[-1].requested_gear == Gear.FORWARD
    assert decisions[-1].reverse_escape_m == 0.0


def test_forward_preference_does_not_shift_when_safe_forward_arc_can_capture_route():
    supervisor = ForwardPreferenceSupervisor()
    angle = math.radians(125.0)
    direction = np.asarray([math.cos(angle), math.sin(angle)])
    behind = np.vstack((np.zeros(2), 0.5 * direction, direction))

    for _ in range(8):
        decision = supervisor.update(
            behind,
            turnaround_feasible=True,
            forward_capture_feasible=True,
            route_requested_gear=Gear.FORWARD,
        )
        assert decision.state == ForwardPreferenceState.FORWARD_CRUISE
        assert decision.requested_gear == Gear.FORWARD
        assert not decision.start_turnaround
        assert decision.reason == "safe_forward_course_capture"


def test_forward_preference_commits_turnaround_for_strongly_rearward_route():
    supervisor = ForwardPreferenceSupervisor()
    angle = math.radians(168.0)
    direction = np.asarray([math.cos(angle), math.sin(angle)])
    strongly_behind = np.vstack((np.zeros(2), 0.5 * direction, direction))

    decisions = [
        supervisor.update(
            strongly_behind,
            turnaround_feasible=True,
            forward_capture_feasible=True,
            route_requested_gear=Gear.FORWARD,
        )
        for _ in range(3)
    ]
    assert decisions[-1].state == ForwardPreferenceState.TURNAROUND_PENDING
    assert decisions[-1].start_turnaround
    assert decisions[-1].reason == "safe_local_turnaround_available"


def test_forward_preference_revalidates_abrupt_mid_goal_route_reversal():
    supervisor = ForwardPreferenceSupervisor()
    forward = np.asarray([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]])
    rear = np.asarray([[0.0, 0.0], [-0.5, 0.0], [-1.0, 0.0]])
    supervisor.update(forward, turnaround_feasible=True)

    decision = supervisor.update(rear, turnaround_feasible=True)
    assert decision.state == ForwardPreferenceState.ROUTE_REVALIDATION
    assert decision.requested_gear == Gear.NEUTRAL
    assert not decision.start_turnaround
    assert decision.reason == "abrupt_rear_route_revalidation"

    supervisor.approve_revalidated_route()
    decisions = [
        supervisor.update(rear, turnaround_feasible=True) for _ in range(3)
    ]
    assert decisions[-1].start_turnaround


def test_turnaround_rearm_blocks_source_flap_until_forward_progress():
    supervisor = ForwardPreferenceSupervisor()
    rear = np.asarray([[0.0, 0.0], [-0.5, 0.0], [-1.0, 0.0]])
    decisions = [
        supervisor.update(
            rear,
            turnaround_feasible=True,
            turnaround_start_authorized=False,
        )
        for _ in range(supervisor.config.behind_confirmation_cycles + 2)
    ]
    assert all(not decision.start_turnaround for decision in decisions)
    assert decisions[-1].requested_gear == Gear.NEUTRAL
    assert any(
        decision.reason == "turnaround_rearm_forward_progress_pending"
        for decision in decisions
    )


def test_forward_route_evidence_survives_global_wait_reset():
    supervisor = ForwardPreferenceSupervisor()
    forward = np.asarray([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]])
    rear = np.asarray([[0.0, 0.0], [-0.5, 0.0], [-1.0, 0.0]])
    supervisor.update(forward, turnaround_feasible=True)
    supervisor.reset(preserve_forward_evidence=True)
    decision = supervisor.update(rear, turnaround_feasible=True)
    assert decision.reason == "abrupt_rear_route_revalidation"


def test_forward_preference_revalidates_instead_of_reversing_after_diverged_capture():
    supervisor = ForwardPreferenceSupervisor()
    moderate_angle = math.radians(125.0)
    moderate_direction = np.asarray(
        [math.cos(moderate_angle), math.sin(moderate_angle)]
    )
    moderate = np.vstack((np.zeros(2), 0.5 * moderate_direction, moderate_direction))
    decision = supervisor.update(
        moderate,
        turnaround_feasible=True,
        forward_capture_feasible=True,
    )
    assert decision.reason == "safe_forward_course_capture"
    assert decision.requested_gear == Gear.FORWARD

    # A course-capture that turns a previously moderate corridor almost
    # antiparallel is stale route evidence, not reverse authority.
    diverged_angle = math.radians(168.0)
    diverged_direction = np.asarray(
        [math.cos(diverged_angle), math.sin(diverged_angle)]
    )
    diverged = np.vstack((np.zeros(2), 0.5 * diverged_direction, diverged_direction))
    decision = supervisor.update(
        diverged,
        turnaround_feasible=True,
        forward_capture_feasible=False,
    )
    assert decision.state == ForwardPreferenceState.ROUTE_REVALIDATION
    assert decision.requested_gear == Gear.NEUTRAL
    assert not decision.start_turnaround
    assert decision.reason == "forward_course_capture_route_revalidation"

    held = supervisor.update(
        diverged,
        turnaround_feasible=True,
        forward_capture_feasible=True,
    )
    assert held.state == ForwardPreferenceState.ROUTE_REVALIDATION
    assert held.requested_gear == Gear.NEUTRAL


def test_committed_reverse_leg_counts_toward_escape_distance():
    supervisor = ForwardPreferenceSupervisor()
    behind = np.asarray([[0.0, 0.0], [-0.5, 0.0], [-1.0, 0.2]])
    for _ in range(3):
        decision = supervisor.update(behind, turnaround_feasible=False)
    assert decision.state == ForwardPreferenceState.REVERSE_ESCAPE

    supervisor.observe_committed_motion(0.42, Gear.REVERSE)
    decision = supervisor.update(behind, turnaround_feasible=True)
    assert decision.start_turnaround
    assert decision.reverse_escape_m >= 0.42


def test_forward_preference_keeps_recovery_latched_across_threshold_band():
    supervisor = ForwardPreferenceSupervisor()
    behind = np.asarray([[0.0, 0.0], [-0.5, 0.0], [-1.0, 0.2]])
    for _ in range(3):
        decision = supervisor.update(
            behind,
            turnaround_feasible=True,
            route_requested_gear=Gear.REVERSE,
        )
    assert decision.state == ForwardPreferenceState.TURNAROUND_PENDING

    # 108.9 degrees reproduces the logged replan boundary: it is below the
    # 110-degree entry threshold but must not cancel an active turnaround.
    ambiguous_end = np.asarray([math.cos(1.901), math.sin(1.901)])
    ambiguous = np.vstack((np.zeros(2), 0.5 * ambiguous_end, ambiguous_end))
    decision = supervisor.update(
        ambiguous,
        turnaround_feasible=True,
        route_requested_gear=Gear.REVERSE,
    )
    assert decision.state == ForwardPreferenceState.TURNAROUND_PENDING
    assert decision.start_turnaround
    assert decision.requested_gear == Gear.FORWARD


def test_forward_reacquisition_requires_near_corridor_forward_hint():
    supervisor = ForwardPreferenceSupervisor()
    front = np.asarray([[0.0, 0.0], [0.4, 0.4], [0.8, 0.8]])
    assert not supervisor.forward_corridor_reacquired(
        front, route_requested_gear=Gear.REVERSE
    )
    assert supervisor.forward_corridor_reacquired(
        front, route_requested_gear=Gear.FORWARD
    )

    behind = np.asarray([[0.0, 0.0], [-0.5, 0.0], [-1.0, 0.2]])
    for _ in range(3):
        supervisor.update(
            behind,
            turnaround_feasible=True,
            route_requested_gear=Gear.REVERSE,
        )
    decision = supervisor.update(
        front,
        turnaround_feasible=True,
        route_requested_gear=Gear.REVERSE,
    )
    assert decision.state == ForwardPreferenceState.TURNAROUND_PENDING
    decision = supervisor.update(
        front,
        turnaround_feasible=False,
        forward_capture_feasible=True,
        forward_exit_verified=True,
        route_requested_gear=Gear.FORWARD,
    )
    assert decision.state == ForwardPreferenceState.TURNAROUND_VERIFY
    assert decision.requested_gear == Gear.NEUTRAL
    decision = supervisor.update(
        front,
        turnaround_feasible=False,
        forward_capture_feasible=True,
        forward_exit_verified=True,
        route_requested_gear=Gear.FORWARD,
    )
    assert decision.state == ForwardPreferenceState.TURNAROUND_VERIFY
    decision = supervisor.update(
        front,
        turnaround_feasible=False,
        forward_capture_feasible=True,
        forward_exit_verified=True,
        route_requested_gear=Gear.FORWARD,
    )
    assert decision.state == ForwardPreferenceState.FORWARD_CRUISE
    assert decision.reason == "forward_corridor_reacquired"


def test_reverse_or_space_exhausted_leg_cannot_finish_turnaround_transaction():
    supervisor = ForwardPreferenceSupervisor()
    behind = np.asarray([[0.0, 0.0], [-0.5, 0.0], [-1.0, 0.2]])
    front = np.asarray([[0.0, 0.0], [0.4, 0.2], [0.9, 0.3]])
    for _ in range(3):
        supervisor.update(
            behind,
            turnaround_feasible=True,
            route_requested_gear=Gear.REVERSE,
        )

    after_reverse = supervisor.update(
        front,
        turnaround_feasible=True,
        forward_capture_feasible=True,
        forward_exit_verified=False,
        route_requested_gear=Gear.FORWARD,
    )
    assert after_reverse.state == ForwardPreferenceState.TURNAROUND_PENDING
    assert after_reverse.start_turnaround

    after_exhausted_forward = supervisor.update(
        front,
        turnaround_feasible=True,
        forward_capture_feasible=True,
        forward_exit_verified=False,
        route_requested_gear=Gear.FORWARD,
    )
    assert after_exhausted_forward.state == ForwardPreferenceState.TURNAROUND_PENDING
    assert after_exhausted_forward.start_turnaround


def test_aligned_reverse_space_creation_preserves_achieved_heading():
    maneuver = CommittedManeuver()
    proposed = maneuver.proposed_subgoal(
        Gear.REVERSE,
        (-1.0, -0.2),
        math.radians(35.0),
        purpose="forward_restoration",
        turn_sign_hint=-1.0,
    )
    assert proposed[0] < 0.0
    assert math.isclose(proposed[1], 0.0, abs_tol=1.0e-9)

    # Steering is restored gradually outside the alignment deadband and is
    # only allowed to reach the normal parking target for a truly rearward
    # route.  This prevents a discontinuity around a 90-degree bearing.
    partial = maneuver.proposed_subgoal(
        Gear.REVERSE,
        (-1.0, -0.2),
        math.radians(75.0),
        purpose="forward_restoration",
        turn_sign_hint=-1.0,
    )
    rearward = maneuver.proposed_subgoal(
        Gear.REVERSE,
        (-1.0, -0.2),
        math.radians(120.0),
        purpose="forward_restoration",
        turn_sign_hint=-1.0,
    )
    assert 0.0 < abs(partial[1]) < maneuver.config.lateral_offset_m
    assert math.isclose(
        abs(rearward[1]), maneuver.config.lateral_offset_m, abs_tol=1.0e-9
    )


def test_planning_result_reports_selected_geometric_horizon():
    grid = OccupancyGrid2D(
        np.zeros((240, 240), dtype=np.int8), 0.1, (-12.0, -12.0)
    )
    planner = DeterministicPlanner()
    result = planner.plan(
        VehicleState(0.0, 0.0, 0.0, 0.0, 0.0),
        (1.0, 0.0),
        grid,
        requested_gear=Gear.FORWARD,
        spatial_scales=(0.75,),
    )
    assert result.executable
    assert math.isclose(result.spatial_scale, 0.75)


def test_online_sync_contract_needs_state_rate_above_thirty_hz():
    anchor = 0.025
    thirty_hz = ((0.0, np.asarray([0.0])), (1.0 / 30.0, np.asarray([1.0])))
    hundred_hz = ((0.02, np.asarray([0.0])), (0.03, np.asarray([1.0])))
    assert interpolated(thirty_hz, anchor, 0.02) is None
    value, distance = interpolated(hundred_hz, anchor, 0.02)
    np.testing.assert_allclose(value, [0.5])
    assert distance < 0.02


def test_forward_preference_reverses_only_until_turnaround_becomes_safe():
    supervisor = ForwardPreferenceSupervisor()
    behind = np.asarray([[0.0, 0.0], [-0.5, 0.0], [-1.0, -0.2]])
    for _ in range(3):
        decision = supervisor.update(behind, turnaround_feasible=False)
    assert decision.state == ForwardPreferenceState.REVERSE_ESCAPE
    assert decision.requested_gear == Gear.REVERSE
    decision = supervisor.update(
        behind, turnaround_feasible=True, progress_m=0.35
    )
    assert decision.state == ForwardPreferenceState.TURNAROUND_PENDING
    assert decision.start_turnaround


def test_forward_preference_never_authorizes_unbounded_reverse():
    supervisor = ForwardPreferenceSupervisor()
    behind = np.asarray([[0.0, 0.0], [-1.0, 0.0]])
    for _ in range(3):
        decision = supervisor.update(behind, turnaround_feasible=False)
    for _ in range(4):
        decision = supervisor.update(
            behind, turnaround_feasible=False, progress_m=0.8
        )
    assert decision.state == ForwardPreferenceState.REVERSE_ESCAPE_EXHAUSTED
    assert decision.requested_gear == Gear.NEUTRAL
