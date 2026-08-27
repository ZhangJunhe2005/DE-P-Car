import math

import numpy as np

from dep_car.runtime.navigation_memory import (
    ackermann_arc_trajectory,
    BoundaryFollowMode,
    BoundaryFollowSupervisor,
    BreadcrumbTrail,
    DeadEndRecoverySupervisor,
    MemoryNavigationState,
    MonotonicRouteProgress,
    ProgressiveMapAcquisitionGate,
    TopologicalMemory,
    TopologyEdgeState,
    local_space_features,
    resample_polyline,
    select_reactive_heading,
    select_dead_end_egress_site,
)
from dep_car.runtime.wheel_odometry import (
    AckermannWheelOdometry,
    PlanarTransformRevisionTracker,
    compose_planar_pose,
)


def test_rolling_route_progress_never_chases_a_passed_carrot():
    cursor = MonotonicRouteProgress(
        minimum_lookahead_m=1.0,
        maximum_lookahead_m=2.5,
    )
    cursor.bind(
        [(0.0, 0.0), (2.0, 0.0), (4.0, 0.0)],
        (0.0, 0.0),
        route_id="far-1",
        revision=1,
        source="FAR",
    )
    first = cursor.observe((1.8, 0.0))
    second = cursor.observe((2.8, 0.0))
    reverse_noise = cursor.observe((2.3, 0.0))
    assert second.progress_m > first.progress_m
    assert reverse_noise.progress_m == second.progress_m
    assert reverse_noise.carrot_m >= reverse_noise.progress_m
    assert reverse_noise.carrot_xy[0] >= 2.8


def test_rolling_route_progress_cannot_teleport_at_self_intersection():
    cursor = MonotonicRouteProgress(maximum_projection_advance_m=0.6)
    cursor.bind(
        [
            (0.0, 0.0),
            (2.0, 0.0),
            (2.0, 2.0),
            (0.0, 2.0),
            (0.0, 0.0),
            (-2.0, 0.0),
        ],
        (0.0, 0.0),
        route_id="topology-loop",
        revision=1,
        source="EXPLORED_TOPOLOGY",
    )
    observation = cursor.observe((0.0, 0.0))
    assert observation.progress_m <= 0.6 + 1.0e-9
    assert observation.skipped_vertices == 0


def test_rolling_carrot_waits_for_vehicle_then_advances_one_bounded_step():
    cursor = MonotonicRouteProgress(
        minimum_lookahead_m=0.9,
        maximum_lookahead_m=1.4,
        carrot_capture_radius_m=0.4,
        maximum_carrot_advance_m=0.7,
        maximum_carrot_distance_m=1.5,
    )
    cursor.bind(
        [(0.0, 0.0), (6.0, 0.0)],
        (0.0, 0.0),
        route_id="far-latched",
        revision=1,
        source="FAR",
    )
    first = cursor.observe((0.0, 0.0))
    small_motion = cursor.observe((0.3, 0.0))
    not_yet_captured = cursor.observe((0.99, 0.0))
    captured = cursor.observe((1.01, 0.0))

    assert first.carrot_m == small_motion.carrot_m
    assert small_motion.carrot_m == not_yet_captured.carrot_m
    assert not small_motion.carrot_advanced
    assert small_motion.carrot_hold_reason == "waiting_for_vehicle_capture"
    assert captured.carrot_advanced
    assert captured.carrot_m - not_yet_captured.carrot_m <= 0.7 + 1.0e-9
    assert captured.carrot_distance_m <= 1.5 + 1.0e-9


def test_route_handoff_projects_the_latched_carrot_without_skipping_ahead():
    cursor = MonotonicRouteProgress(
        minimum_lookahead_m=0.9,
        maximum_lookahead_m=1.4,
    )
    cursor.bind(
        [(0.0, 0.0), (5.0, 0.0)],
        (0.0, 0.0),
        route_id="far-old",
        revision=1,
        source="FAR",
    )
    old = cursor.observe((0.2, 0.0))
    candidate = [(0.15, 0.03), (3.0, 0.03), (5.0, 0.2)]
    decision = cursor.preview_handoff(
        candidate,
        (0.2, 0.0),
        maximum_entry_deviation_m=0.5,
        maximum_direction_change_rad=math.radians(60.0),
    )
    assert decision.accepted
    assert decision.candidate_carrot_m is not None
    cursor.bind(
        candidate,
        (0.2, 0.0),
        route_id="far-new",
        revision=2,
        source="FAR",
        initial_progress_m=decision.candidate_progress_m,
        initial_carrot_m=decision.candidate_carrot_m,
    )
    renewed = cursor.observe((0.2, 0.0))

    assert not renewed.carrot_advanced
    assert renewed.carrot_hold_reason == "waiting_for_vehicle_capture"
    assert np.linalg.norm(
        np.asarray(renewed.carrot_xy) - np.asarray(old.carrot_xy)
    ) <= 0.05


def test_route_handoff_matches_progress_without_deforming_candidate():
    cursor = MonotonicRouteProgress()
    cursor.bind(
        [(0.0, 0.0), (2.0, 0.0), (4.0, 0.0)],
        (0.0, 0.0),
        route_id="far-old",
        revision=1,
        source="FAR",
    )
    cursor.observe((1.0, 0.0))
    candidate = np.asarray(
        [(0.9, 0.05), (2.5, 0.05), (4.5, 0.4)], dtype=float
    )
    original = candidate.copy()
    decision = cursor.preview_handoff(
        candidate,
        (1.0, 0.0),
        maximum_entry_deviation_m=0.5,
        maximum_direction_change_rad=math.radians(60.0),
    )
    assert decision.accepted
    cursor.bind(
        candidate,
        (1.0, 0.0),
        route_id="far-new",
        revision=2,
        source="FAR",
        initial_progress_m=decision.candidate_progress_m,
    )
    np.testing.assert_allclose(cursor.path, original)


def test_route_handoff_rejects_unattached_or_opposite_candidate():
    cursor = MonotonicRouteProgress()
    cursor.bind(
        [(0.0, 0.0), (3.0, 0.0)],
        (0.0, 0.0),
        route_id="far-old",
        revision=1,
        source="FAR",
    )
    cursor.observe((1.0, 0.0))
    disconnected = cursor.preview_handoff(
        [(5.0, 5.0), (7.0, 5.0)],
        (1.0, 0.0),
        maximum_entry_deviation_m=0.9,
        maximum_direction_change_rad=math.radians(60.0),
    )
    assert not disconnected.accepted
    assert disconnected.reason == "candidate_has_no_local_attachment"
    opposite = cursor.preview_handoff(
        [(1.0, 0.0), (-2.0, 0.0)],
        (1.0, 0.0),
        maximum_entry_deviation_m=0.9,
        maximum_direction_change_rad=math.radians(60.0),
    )
    assert not opposite.accepted
    assert opposite.reason == "candidate_tangent_discontinuous"


def test_route_handoff_rejects_a_delayed_fold_behind_the_incumbent():
    cursor = MonotonicRouteProgress(curvature_window_m=1.5)
    cursor.bind(
        [(0.0, 0.0), (5.0, 0.0)],
        (0.0, 0.0),
        route_id="far-stable",
        revision=1,
        source="FAR",
    )
    cursor.observe((1.0, 0.0))
    # It initially agrees with the incumbent, then folds behind within the
    # local execution horizon.  Attachment-only comparison used to accept it.
    candidate = [
        (1.0, 0.0),
        (1.35, 0.0),
        (1.0, 0.35),
        (0.0, 0.35),
    ]
    decision = cursor.preview_handoff(
        candidate,
        (1.0, 0.0),
        maximum_entry_deviation_m=0.9,
        maximum_direction_change_rad=math.radians(60.0),
    )

    assert not decision.accepted
    assert decision.reason == "candidate_tangent_discontinuous"
    assert decision.direction_change_rad > math.radians(60.0)


def test_wheel_odometry_is_signed_and_obeys_ackermann_curvature():
    odometry = AckermannWheelOdometry(wheel_radius=0.1, wheelbase=0.5)
    odometry.update(0.0, 10.0, 10.0, 0.0)
    straight = odometry.update(0.1, 10.0, 10.0, 0.0)
    assert math.isclose(straight.x, 0.1, abs_tol=1.0e-9)
    assert math.isclose(straight.yaw, 0.0, abs_tol=1.0e-9)
    turning = odometry.update(0.2, 10.0, 10.0, math.atan(0.5))
    assert turning.yaw_rate > 0.0
    reverse = odometry.update(0.3, -10.0, -10.0, math.atan(0.5))
    assert reverse.speed < 0.0
    assert reverse.yaw_rate < 0.0


def test_progressive_map_acquisition_rearms_only_after_scan_and_map_growth():
    gate = ProgressiveMapAcquisitionGate(
        chunk_distance_m=0.8,
        settle_time_s=0.6,
        minimum_known_cell_gain=20,
        maximum_accumulated_m=2.4,
    )
    assert gate.observe((0.0, 0.0), 1000, 0.0).motion_authorized
    active = gate.observe((0.5, 0.0), 1010, 1.0)
    assert active.motion_authorized and math.isclose(active.remaining_m, 0.3)
    settle = gate.observe((0.8, 0.0), 1010, 2.0)
    assert not settle.motion_authorized
    assert settle.reason == "mapping_pulse_scan_settle"
    still_settling = gate.observe((0.8, 0.0), 1030, 2.5)
    assert not still_settling.motion_authorized
    rearmed = gate.observe((0.8, 0.0), 1030, 2.7)
    assert rearmed.motion_authorized
    assert rearmed.reason == "mapping_pulse_rearmed_after_map_growth"
    assert rearmed.cycle == 1


def test_progressive_map_acquisition_holds_when_motion_reveals_no_new_space():
    gate = ProgressiveMapAcquisitionGate(
        chunk_distance_m=0.8,
        settle_time_s=0.5,
        minimum_known_cell_gain=20,
        maximum_accumulated_m=2.4,
    )
    gate.observe((0.0, 0.0), 1000, 0.0)
    gate.observe((0.8, 0.0), 1005, 1.0)
    decision = gate.observe((0.8, 0.0), 1010, 1.6)
    assert not decision.motion_authorized
    assert decision.reason == "mapping_pulse_no_new_observed_space"


def test_wheel_odometry_suppresses_stopped_steering_joint_noise():
    odometry = AckermannWheelOdometry(
        wheel_radius=0.1,
        wheelbase=0.5,
        speed_deadband_mps=0.02,
    )
    odometry.update(0.0, 0.10, 0.10, 0.55)
    stopped = odometry.update(0.1, 0.10, 0.10, 0.55)
    assert stopped.speed == 0.0
    assert stopped.yaw_rate == 0.0
    assert stopped.x == 0.0
    assert stopped.y == 0.0
    assert stopped.yaw == 0.0


def test_map_odometry_pose_composes_slam_correction_without_touching_twist():
    x, y, yaw = compose_planar_pose(
        (1.0, 2.0, math.pi / 2.0), (3.0, 0.0, math.pi / 2.0)
    )
    assert math.isclose(x, 1.0, abs_tol=1.0e-9)
    assert math.isclose(y, 5.0, abs_tol=1.0e-9)
    assert math.isclose(abs(yaw), math.pi, abs_tol=1.0e-9)


def test_map_odometry_correction_is_emitted_once_per_tf_revision():
    tracker = PlanarTransformRevisionTracker()
    initial = tracker.observe((12, 100), (1.0, 2.0, 0.2))
    assert initial is not None
    assert initial.translation_delta == 0.0
    assert initial.yaw_delta == 0.0
    # The latest-TF lookup may be repeated by many odometry callbacks.  Even
    # if a caller presents different values, an unchanged TF stamp is not a
    # new SLAM authority revision.
    assert tracker.observe((12, 100), (9.0, 9.0, 2.0)) is None
    update = tracker.observe((12, 200), (1.3, 2.4, -3.0))
    assert update is not None
    assert math.isclose(update.translation_delta, 0.5, abs_tol=1.0e-9)
    assert math.isclose(
        update.yaw_delta,
        math.atan2(math.sin(-3.2), math.cos(-3.2)),
        abs_tol=1.0e-9,
    )


def test_breadcrumb_reverse_corridor_follows_certified_history():
    trail = BreadcrumbTrail(spacing_m=0.19)
    for index in range(8):
        trail.record(index * 0.2, 0.0, 0.0, index, turnaround=index == 2)
    corridor, target = trail.older_corridor(1.4, 0.0, target_distance_m=0.8)
    assert target is not None and target < 7
    assert corridor[0].x > corridor[-1].x
    assert trail.most_recent_recovery_site(6) == 2
    sampled = resample_polyline([(point.x, point.y) for point in corridor], 20)
    assert sampled.shape == (20, 2)
    assert np.all(np.diff(sampled[:, 0]) <= 1.0e-9)


def test_signed_breadcrumb_replay_inverts_each_constant_gear_transaction():
    trail = BreadcrumbTrail(spacing_m=0.19)
    trail.record(0.0, 0.0, 0.0, 0.0, motion_direction=0, force=True)
    trail.record(0.2, 0.0, 0.0, 1.0, motion_direction=1)
    trail.record(0.4, 0.0, 0.0, 2.0, motion_direction=1)
    # The vehicle then reversed to a lateral recovery pose.
    trail.record(0.3, 0.2, 0.2, 3.0, motion_direction=-1)
    trail.record(0.2, 0.4, 0.3, 4.0, motion_direction=-1)

    corridor, target, replay_direction = trail.reverse_replay_corridor(
        0.2, 0.4, start_index=4, target_distance_m=2.0
    )
    assert replay_direction == 1
    assert target == 2
    assert [(point.x, point.y) for point in corridor[-2:]] == [
        (0.3, 0.2),
        (0.4, 0.0),
    ]
    corridor, target, replay_direction = trail.reverse_replay_corridor(
        0.4, 0.0, start_index=2, target_distance_m=2.0
    )
    assert replay_direction == -1
    assert target == 0


def test_monotonic_breadcrumb_window_does_not_jump_across_self_intersection():
    trail = BreadcrumbTrail(spacing_m=0.05)
    for index, (x, y) in enumerate(
        [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.0, 0.0)]
    ):
        trail.record(x, y, 0.0, index, force=True)
    # The newest and oldest points coincide.  A bounded replay cursor must
    # select the new lap, not teleport to history index zero.
    assert trail.nearest_index(0.0, 0.0) == 0
    assert trail.nearest_index_window(
        0.0, 0.0, lower_index=2, upper_index=4
    ) == 4


def test_recovery_site_must_be_far_enough_back_along_certified_history():
    trail = BreadcrumbTrail(spacing_m=0.19)
    for index in range(11):
        trail.record(
            index * 0.2,
            0.0,
            0.0,
            index,
            turnaround=index in (2, 8),
        )
    assert trail.most_recent_recovery_site(9) == 8
    assert trail.most_recent_recovery_site(
        9, from_index=10, minimum_path_distance_m=1.0
    ) == 2


def test_dead_end_egress_prefers_a_real_external_recovery_anchor():
    trail = BreadcrumbTrail(spacing_m=0.19)
    for index in range(21):
        trail.record(
            index * 0.2,
            0.0,
            0.0,
            index,
            junction=index == 10,
        )
    site, kind, distance = select_dead_end_egress_site(
        trail,
        20,
        minimum_distance_m=1.5,
        maximum_distance_m=8.0,
    )
    assert site == 10
    assert kind == "RECOVERY_ANCHOR"
    assert math.isclose(distance, 2.0)


def test_dead_end_egress_without_junction_uses_bounded_oldest_connector():
    trail = BreadcrumbTrail(spacing_m=0.19)
    for index in range(31):
        trail.record(index * 0.2, 0.0, 0.0, index)
    site, kind, distance = select_dead_end_egress_site(
        trail,
        30,
        minimum_distance_m=1.5,
        maximum_distance_m=3.0,
    )
    assert site == 15
    assert kind == "CERTIFIED_TRAIL_LIMIT"
    assert math.isclose(distance, 3.0)


def test_dynamic_wait_never_becomes_dead_end_but_static_blockage_does():
    recovery = DeadEndRecoverySupervisor(confirmation_s=1.0, minimum_trail_points=4)
    recovery.reset(active=True)
    assert recovery.update_goal_seek(
        0.0, progress_m=0.0, static_blocked=True, dynamic_blocked=True,
        rear_clear=True, trail_points=10,
    ) == MemoryNavigationState.GOAL_SEEK
    recovery.update_goal_seek(
        1.0, progress_m=0.0, static_blocked=True, dynamic_blocked=False,
        rear_clear=True, trail_points=10,
    )
    assert recovery.update_goal_seek(
        2.1, progress_m=0.0, static_blocked=True, dynamic_blocked=False,
        rear_clear=True, trail_points=10,
    ) == MemoryNavigationState.BACKTRACK_REVERSE
    recovery.complete_goal()
    assert recovery.state == MemoryNavigationState.GOAL_REACHED


def test_visibility_backend_can_revoke_breadcrumb_motion_authority():
    recovery = DeadEndRecoverySupervisor(
        confirmation_s=0.5,
        minimum_trail_points=4,
        backtrack_enabled=False,
    )
    recovery.reset(active=True)
    states = []
    for stamp in (0.0, 0.6, 1.2):
        state = recovery.update_goal_seek(
            stamp,
            progress_m=0.0,
            static_blocked=True,
            dynamic_blocked=False,
            rear_clear=True,
            trail_points=30,
        )
        states.append(state)
    assert states == [
        MemoryNavigationState.GOAL_SEEK,
        MemoryNavigationState.SUSPECT_DEAD_END,
        MemoryNavigationState.SUSPECT_DEAD_END,
    ]
    with np.testing.assert_raises_regex(
        RuntimeError, "breadcrumb motion authority is disabled"
    ):
        recovery.force_backtrack()

    # A separate, explicitly bounded FAR egress remains possible after the
    # node has independently confirmed repeated static hard blocks.
    recovery.force_certified_egress()
    assert recovery.state == MemoryNavigationState.FAR_DEAD_END_EGRESS
    recovery.complete_certified_egress()
    assert recovery.state == MemoryNavigationState.GOAL_SEEK


def test_committed_turnaround_cannot_be_preempted_as_a_dead_end():
    recovery = DeadEndRecoverySupervisor(confirmation_s=0.5, minimum_trail_points=4)
    recovery.reset(active=True)
    for stamp in (0.0, 1.0, 2.0):
        state = recovery.update_goal_seek(
            stamp,
            progress_m=0.0,
            static_blocked=True,
            dynamic_blocked=False,
            rear_clear=True,
            trail_points=10,
            maneuver_active=True,
        )
        assert state == MemoryNavigationState.GOAL_SEEK
    assert recovery.blocked_since is None


def test_failed_resume_returns_to_older_breadcrumb_backtracking():
    recovery = DeadEndRecoverySupervisor()
    recovery.reset(active=True)
    recovery.state = MemoryNavigationState.BACKTRACK_REVERSE
    recovery.begin_resume()
    recovery.resume_failed()
    assert recovery.state == MemoryNavigationState.BACKTRACK_REVERSE


def test_successful_escape_returns_to_same_goal_and_prunes_abandoned_suffix():
    trail = BreadcrumbTrail(spacing_m=0.19)
    for index in range(8):
        trail.record(index * 0.2, 0.0, 0.0, index)
    trail.truncate_after(3)
    assert len(trail.points) == 4
    assert math.isclose(trail.points[-1].x, 0.6)

    recovery = DeadEndRecoverySupervisor()
    recovery.reset(active=True)
    recovery.state = MemoryNavigationState.BACKTRACK_REVERSE
    recovery.begin_resume()
    recovery.escaped()
    recovery.continue_goal_seek()
    assert recovery.state == MemoryNavigationState.GOAL_SEEK


def test_local_space_features_distinguish_corridor_and_open_site():
    grid = np.zeros((80, 80), dtype=np.int8)
    origin = (-2.0, -2.0)
    open_site = local_space_features(grid, 0.05, origin)
    assert open_site["junction"] and open_site["turnaround"]
    corridor = grid.copy()
    corridor[:, :34] = 100
    corridor[:, 46:] = 100
    narrow = local_space_features(corridor, 0.05, origin)
    assert not narrow["junction"]


def test_front_goal_cannot_choose_open_rear_as_obstacle_avoidance_shortcut():
    decision = select_reactive_heading(
        0.0,
        [
            (0.0, 0.35),
            (math.pi / 2.0, 1.40),
            (-math.pi / 2.0, 0.50),
            (math.pi, 3.00),
        ],
        safe_clearance_m=0.80,
    )
    assert decision.sector == "FORWARD_EXPLORATION"
    assert math.isclose(decision.angle, math.pi / 2.0)
    assert not decision.blocked


def test_fully_blocked_front_sector_defers_to_dead_end_supervisor():
    decision = select_reactive_heading(
        0.15,
        [(0.0, 0.35), (1.2, 0.55), (-1.2, 0.45), (math.pi, 3.0)],
        safe_clearance_m=0.80,
    )
    assert decision.sector == "FORWARD_EXPLORATION"
    assert abs(decision.angle) <= 1.65
    assert decision.blocked


def test_rear_goal_produces_one_explicit_turnaround_sector_request():
    decision = select_reactive_heading(
        math.pi - 0.10,
        [(0.0, 3.0), (2.4, 1.0), (math.pi - 0.10, 1.8), (-2.7, 1.1)],
        safe_clearance_m=0.80,
    )
    assert decision.sector == "TURNAROUND"
    assert abs(decision.angle) > 2.10
    assert not decision.blocked


def test_free_sector_hysteresis_requires_a_real_clearance_advantage():
    decision = select_reactive_heading(
        0.0,
        [(-0.8, 1.50), (0.8, 1.58)],
        previous_heading=-0.8,
        safe_clearance_m=0.80,
    )
    assert math.isclose(decision.angle, -0.8)


def test_boundary_follow_latches_the_safer_side_until_release_contract():
    boundary = BoundaryFollowSupervisor(
        enter_clearance_m=0.9,
        release_clearance_m=1.3,
        leave_progress_m=0.4,
        leave_confirmation_s=0.5,
    )
    entered = boundary.update(
        0.0,
        (0.0, 0.0),
        5.0,
        0.5,
        left_score=1.4,
        right_score=0.9,
    )
    assert entered.entered and entered.active
    assert entered.mode == BoundaryFollowMode.FOLLOW_LEFT
    # A transiently more attractive opposite side cannot flip the latch.
    held = boundary.update(
        0.2,
        (0.2, 0.0),
        4.8,
        0.6,
        left_score=0.2,
        right_score=3.0,
    )
    assert held.mode == BoundaryFollowMode.FOLLOW_LEFT
    assert boundary.heading_utility(-0.8) < boundary.heading_utility(0.2)


def test_boundary_follow_requires_progress_and_confirmed_direct_clearance_to_leave():
    boundary = BoundaryFollowSupervisor(
        enter_clearance_m=0.9,
        release_clearance_m=1.3,
        leave_progress_m=0.4,
        leave_confirmation_s=0.5,
    )
    boundary.update(
        0.0, (0.0, 0.0), 5.0, 0.5, left_score=1.0, right_score=0.8
    )
    no_progress = boundary.update(
        1.0, (0.5, 0.0), 4.8, 1.5, left_score=1.0, right_score=0.8
    )
    assert no_progress.active
    confirming = boundary.update(
        2.0, (1.0, 0.0), 4.5, 1.5, left_score=1.0, right_score=0.8
    )
    assert confirming.active
    released = boundary.update(
        2.6, (1.2, 0.0), 4.4, 1.5, left_score=1.0, right_score=0.8
    )
    assert released.left_boundary and not released.active
    assert boundary.mode == BoundaryFollowMode.DIRECT


def test_boundary_follow_detects_a_closed_loop_at_the_hit_region():
    boundary = BoundaryFollowSupervisor(
        enter_clearance_m=0.9,
        release_clearance_m=1.3,
        leave_progress_m=0.4,
        leave_confirmation_s=0.5,
        loop_radius_m=0.3,
        minimum_loop_travel_m=3.0,
        maximum_boundary_travel_m=10.0,
    )
    boundary.update(
        0.0, (0.0, 0.0), 5.0, 0.5, left_score=1.0, right_score=0.8
    )
    for stamp, point in enumerate(
        ((1.0, 0.0), (1.0, 1.0), (0.0, 1.0)), start=1
    ):
        decision = boundary.update(
            stamp, point, 4.9, 0.7, left_score=1.0, right_score=0.8
        )
        assert not decision.loop_detected
    loop = boundary.update(
        4.0, (0.1, 0.0), 4.9, 0.7, left_score=1.0, right_score=0.8
    )
    assert loop.loop_detected and not loop.active
    assert loop.reason == "returned_to_boundary_hit_region"


def test_failed_boundary_side_is_not_repeated_near_the_same_hit_for_one_goal():
    boundary = BoundaryFollowSupervisor(
        enter_clearance_m=0.9,
        release_clearance_m=1.3,
        leave_progress_m=0.4,
        leave_confirmation_s=0.5,
        failure_radius_m=0.8,
    )
    first = boundary.update(
        0.0, (0.0, 0.0), 5.0, 0.5, left_score=2.0, right_score=1.0
    )
    assert first.mode == BoundaryFollowMode.FOLLOW_LEFT
    remembered = boundary.remember_current_failure()
    assert remembered is not None and remembered.side == 1
    boundary.reset(clear_failures=False)

    retried = boundary.update(
        1.0, (0.2, 0.0), 5.1, 0.5, left_score=3.0, right_score=1.0
    )
    assert retried.mode == BoundaryFollowMode.FOLLOW_RIGHT
    assert len(boundary.failed_sides) == 1

    boundary.reset(clear_failures=True)
    fresh_goal = boundary.update(
        2.0, (0.2, 0.0), 5.1, 0.5, left_score=3.0, right_score=1.0
    )
    assert fresh_goal.mode == BoundaryFollowMode.FOLLOW_LEFT


def test_certified_boundary_loop_can_force_breadcrumb_backtracking():
    recovery = DeadEndRecoverySupervisor()
    recovery.reset(active=True)
    recovery.force_backtrack()
    assert recovery.state == MemoryNavigationState.BACKTRACK_REVERSE


def test_ackermann_arc_trajectory_has_consistent_endpoint_and_yaw():
    left = ackermann_arc_trajectory(math.pi / 2.0, 2.0, count=21)
    right = ackermann_arc_trajectory(-math.pi / 2.0, 2.0, count=21)
    assert left.shape == (21, 4)
    assert np.all(np.diff(left[:, 0]) > 0.0)
    assert math.isclose(left[-1, 3], math.pi / 2.0)
    assert math.isclose(right[-1, 3], -math.pi / 2.0)
    assert math.isclose(left[-1, 1], right[-1, 1])
    assert math.isclose(left[-1, 2], -right[-1, 2])


def test_breadcrumb_resume_target_uses_path_progress_not_net_displacement():
    trail = BreadcrumbTrail(spacing_m=0.19)
    for index in range(11):
        trail.record(index * 0.2, 0.0, 0.0, index)
    assert trail.older_index_at_distance(10, 1.0) == 5


def build_branching_topology():
    topology = TopologicalMemory(node_spacing_m=0.5, merge_radius_m=0.25)
    topology.record(0.0, 0.0, 0.0, 0.0, force=True)
    topology.record(1.0, 0.0, 0.0, 1.0, junction=True, force=True)
    topology.record(2.0, 0.0, 0.0, 2.0, force=True)
    topology.record(1.0, 0.0, math.pi, 3.0, junction=True, force=True)
    topology.record(1.0, 1.0, math.pi / 2.0, 4.0, force=True)
    return topology


def test_topology_remembers_failed_branch_and_routes_through_alternative_edge():
    topology = build_branching_topology()
    branch = topology.mark_failed_branch(
        [(1.0, 0.0), (2.0, 0.0)], goal_xy=(2.0, 1.0), stamp=5.0
    )
    assert branch is not None
    assert topology.summary()["failed_branches"] == 1
    assert topology.summary()["edge_states"][TopologyEdgeState.DEAD_END.value] == 1
    path = topology.guidance_path((1.0, 0.0), (1.0, 1.2), stamp=6.0)
    assert len(path) >= 2
    assert path[-1][1] > 0.5

    failed_edge = topology.edges[topology._edge_key(1, 2)]
    # The terminal may leave through the same physical edge, but the entry
    # may not traverse it toward the remembered dead end again.
    assert topology._transition_available(
        failed_edge, 2, 1, (1.0, 2.0), stamp=6.0
    )
    assert not topology._transition_available(
        failed_edge, 1, 2, (1.0, 2.0), stamp=6.0
    )


def test_failed_branch_is_directional_and_goal_aware():
    topology = build_branching_topology()
    topology.mark_failed_branch(
        [(1.0, 0.0), (2.0, 0.0)], goal_xy=(2.0, 1.0), stamp=5.0
    )
    assert topology.polyline_enters_failed_branch(
        [(1.0, 0.0), (1.5, 0.0), (2.0, 0.0)],
        goal_xy=(1.0, 2.0),
        stamp=6.0,
    )
    assert not topology.polyline_enters_failed_branch(
        [(1.0, 0.0), (0.5, 0.0), (0.0, 0.0)],
        goal_xy=(1.0, 2.0),
        stamp=6.0,
    )
    branch = topology.failed_branches[0]
    assert branch.egress_points()[0] == branch.terminal
    assert branch.egress_points()[-1] == branch.entry
    assert math.isclose(branch.length_m, 1.0)
    assert not topology.polyline_enters_failed_branch(
        [(1.0, 0.0), (1.5, 0.0), (2.0, 0.0)],
        goal_xy=(2.0, 0.0),
        stamp=6.0,
    )


def test_breadcrumb_retains_slam_pose_without_replacing_continuous_odom():
    trail = BreadcrumbTrail(spacing_m=0.1)
    trail.record(
        1.0,
        2.0,
        0.2,
        3.0,
        map_pose=(4.0, 5.0, 0.3),
        map_revision=7,
        force=True,
    )
    point = trail.points[0]
    assert (point.x, point.y) == (1.0, 2.0)
    assert (point.map_x, point.map_y) == (4.0, 5.0)
    assert point.map_revision == 7


def test_repeated_dead_end_extends_taboo_upstream_and_marks_new_edges():
    topology = TopologicalMemory(
        node_spacing_m=0.4,
        merge_radius_m=0.25,
        failure_buffer_m=0.35,
        goal_branch_allowance_m=0.30,
    )
    for index in range(6):
        topology.record(index * 0.4, 0.0, 0.0, index, force=True)
    first = topology.mark_failed_branch(
        [(0.8, 0.0), (1.2, 0.0), (1.6, 0.0), (2.0, 0.0)],
        goal_xy=(2.0, 1.0),
        stamp=6.0,
    )
    repeated = topology.mark_failed_branch(
        [(0.4, 0.05), (0.8, 0.03), (1.2, 0.02), (1.6, 0.01), (2.0, 0.0)],
        goal_xy=(2.0, 1.0),
        stamp=7.0,
    )
    assert repeated is first
    assert repeated.failures == 2
    assert repeated.entry[0] < 0.5
    assert len(topology.failed_branches) == 1
    assert topology.edges[(1, 2)].state == TopologyEdgeState.DEAD_END


def test_goal_near_but_not_at_dead_end_terminal_does_not_reopen_branch():
    topology = TopologicalMemory(goal_branch_allowance_m=0.30)
    topology.record(0.0, 0.0, 0.0, 0.0, force=True)
    topology.record(1.0, 0.0, 0.0, 1.0, force=True)
    topology.mark_failed_branch(
        [(0.0, 0.0), (1.0, 0.0)], goal_xy=(1.0, 0.5), stamp=2.0
    )
    assert topology.polyline_enters_failed_branch(
        [(0.0, 0.0), (0.5, 0.0), (1.0, 0.0)],
        goal_xy=(1.0, 0.31),
        stamp=3.0,
    )
    assert not topology.polyline_enters_failed_branch(
        [(0.0, 0.0), (0.5, 0.0), (1.0, 0.0)],
        goal_xy=(1.0, 0.20),
        stamp=3.0,
    )


def test_heading_utility_prefers_accumulated_map_clearance_when_paths_are_safe():
    decision = select_reactive_heading(
        0.0,
        [(-0.5, 2.0), (0.5, 2.0)],
        safe_clearance_m=0.8,
        angle_utilities=[(-0.5, 0.0), (0.5, 0.6)],
    )
    assert math.isclose(decision.angle, 0.5)


def test_goal_preemption_reanchors_topology_instead_of_creating_phantom_edge():
    topology = TopologicalMemory(node_spacing_m=0.5, merge_radius_m=0.25)
    origin = topology.record(0.0, 0.0, 0.0, 0.0, force=True)
    terminal = topology.record(2.0, 0.0, 0.0, 1.0, force=True)
    assert topology.last_node_id == terminal
    assert topology.reanchor(0.05, 0.0, maximum_distance_m=0.5) == origin
    topology.record(0.0, 1.0, math.pi / 2.0, 2.0, force=True)
    assert topology._edge_key(origin, topology.last_node_id) in topology.edges
    assert topology._edge_key(terminal, topology.last_node_id) not in topology.edges


def test_topology_visits_count_arrivals_not_control_ticks():
    topology = TopologicalMemory(node_spacing_m=0.5, merge_radius_m=0.25)
    origin = topology.record(0.0, 0.0, 0.0, 0.0, force=True)
    topology.record(0.05, 0.0, 0.0, 0.1, force=True)
    topology.record(0.10, 0.0, 0.0, 0.2, force=True)
    assert topology.nodes[origin].visits == 1
    other = topology.record(1.0, 0.0, 0.0, 1.0, force=True)
    topology.record(0.0, 0.0, math.pi, 2.0, force=True)
    assert topology.nodes[origin].visits == 2
    assert topology.nodes[other].visits == 1
