import numpy as np

from dep_car.runtime.far_visibility import (
    DynamicVisibilityPlanner,
    VisibilityEdge,
    VisibilityNode,
    VisibilityPlan,
    VisibilityRouteAcquisitionGate,
    measured_pose_revalidation_authorized,
    polyline_prefix,
    route_initial_bearing,
    transient_route_lease_authorized,
)


def test_route_initial_bearing_exposes_rolling_handoff_reversal():
    forward = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)]
    reverse = [(0.0, 0.0), (-1.0, 0.0), (-2.0, 0.0)]
    assert np.isclose(route_initial_bearing(forward), 0.0)
    assert np.isclose(abs(route_initial_bearing(reverse)), np.pi)


def test_polyline_prefix_keeps_corner_before_interpolated_carrot():
    prefix = polyline_prefix(
        ((0.0, 0.0), (1.0, 0.0), (1.0, 2.0)), 1.5
    )
    assert np.allclose(prefix, ((0.0, 0.0), (1.0, 0.0), (1.0, 0.5)))


def test_measured_pose_revalidation_uses_only_observed_safe_prefix():
    plan = VisibilityPlan(
        status="PASS",
        mode="ATTEMPTABLE_VISIBILITY",
        path=((0.0, 0.0), (1.0, 0.0), (5.0, 2.0)),
        nodes=(),
        edges=(),
        path_cost=8.0,
        known_edges=1,
        attemptable_edges=1,
        reason="attempt_unknown_route",
        path_length=6.0,
        path_unknown_fraction=0.13,
    )
    assert measured_pose_revalidation_authorized(
        plan,
        observed_prefix_clear=True,
        maximum_attemptable_unknown_fraction=0.20,
    )
    assert not measured_pose_revalidation_authorized(
        plan,
        observed_prefix_clear=False,
        maximum_attemptable_unknown_fraction=0.20,
    )
    too_unknown = VisibilityPlan(
        **{
            **plan.__dict__,
            "path_unknown_fraction": 0.25,
        }
    )
    assert not measured_pose_revalidation_authorized(
        too_unknown,
        observed_prefix_clear=True,
        maximum_attemptable_unknown_fraction=0.20,
    )


def test_transient_route_lease_requires_same_goal_safe_prefix_and_finite_age():
    contract = dict(
        previous_route_motion_authorized=True,
        same_goal=True,
        local_prefix_clear=True,
        local_prefix_length_m=1.8,
        minimum_prefix_length_m=0.7,
        dropout_age_s=0.9,
        grace_s=4.0,
    )
    assert transient_route_lease_authorized(**contract)
    assert not transient_route_lease_authorized(
        **{**contract, "same_goal": False}
    )
    assert not transient_route_lease_authorized(
        **{**contract, "local_prefix_clear": False}
    )
    assert not transient_route_lease_authorized(
        **{**contract, "local_prefix_length_m": 0.4}
    )
    assert not transient_route_lease_authorized(
        **{**contract, "dropout_age_s": 4.01}
    )


def test_known_visibility_route_goes_around_an_observed_wall():
    grid = np.zeros((220, 220), dtype=np.int16)
    resolution = 0.05
    wall_column = int(5.0 / resolution)
    grid[: int(8.0 / resolution) + 1, wall_column - 1 : wall_column + 2] = 100
    planner = DynamicVisibilityPlanner(maximum_nodes=100)

    result = planner.plan(grid, resolution, (0.0, 0.0), (2.0, 4.0), (8.0, 4.0))

    assert result.status == "PASS"
    assert result.mode == "KNOWN_VISIBILITY"
    assert len(result.path) >= 3
    assert max(point[1] for point in result.path) > 8.0
    assert result.known_edges == len(result.edges)


def test_unknown_empty_space_produces_an_attemptable_direct_route():
    grid = np.full((100, 100), -1, dtype=np.int16)
    planner = DynamicVisibilityPlanner()

    result = planner.plan(grid, 0.10, (0.0, 0.0), (1.0, 1.0), (7.0, 7.0))

    assert result.status == "PASS"
    assert result.mode == "ATTEMPTABLE_VISIBILITY"
    assert result.path == ((1.0, 1.0), (7.0, 7.0))
    assert result.attemptable_edges == 1
    assert result.path_unknown_fraction == 1.0
    assert result.path_length > 8.0


def test_observed_closed_box_disconnects_the_goal():
    grid = np.zeros((160, 160), dtype=np.int16)
    resolution = 0.05
    first, last = int(3.0 / resolution), int(5.0 / resolution)
    grid[first:last + 1, first - 1 : first + 2] = 100
    grid[first:last + 1, last - 1 : last + 2] = 100
    grid[first - 1 : first + 2, first:last + 1] = 100
    grid[last - 1 : last + 2, first:last + 1] = 100
    planner = DynamicVisibilityPlanner(maximum_nodes=100)

    result = planner.plan(grid, resolution, (0.0, 0.0), (1.0, 4.0), (4.0, 4.0))

    assert result.status == "NO_ROUTE"
    assert result.path == ()


def test_failed_branch_virtual_obstacle_changes_visibility_route():
    grid = np.zeros((120, 120), dtype=np.int16)
    planner = DynamicVisibilityPlanner(maximum_nodes=80)
    branch = [(2.8, 0.5), (2.8, 4.8)]

    result = planner.plan(
        grid,
        0.05,
        (0.0, 0.0),
        (1.0, 2.5),
        (5.0, 2.5),
        blocked_polylines=[branch],
        failure_buffer_m=0.20,
    )

    assert result.status == "PASS"
    assert len(result.path) >= 3
    assert any(point[1] < 0.5 or point[1] > 4.8 for point in result.path[1:-1])


def test_directed_failed_branch_allows_exit_but_rejects_reentry():
    planner = DynamicVisibilityPlanner(maximum_nodes=40)
    branch = [(1.0, 2.0), (4.0, 2.0)]

    assert planner._transition_enters_failed_branch(
        (0.5, 2.0), (3.5, 2.0), [branch], 0.25
    )
    assert not planner._transition_enters_failed_branch(
        (3.5, 2.0), (0.5, 2.0), [branch], 0.25
    )


def test_vehicle_inside_directed_failed_branch_keeps_visibility_egress():
    grid = np.zeros((120, 160), dtype=np.int16)
    planner = DynamicVisibilityPlanner(
        maximum_nodes=40,
        start_heading_weight_m=0.0,
        reverse_start_penalty_m=0.0,
    )
    branch = [(2.0, 3.0), (6.0, 3.0)]

    egress = planner.plan(
        grid,
        0.05,
        (0.0, 0.0),
        (5.5, 3.0),
        (0.5, 3.0),
        directed_failed_branches=[branch],
        failure_buffer_m=0.30,
    )
    reentry = planner.plan(
        grid,
        0.05,
        (0.0, 0.0),
        (0.5, 3.0),
        (5.5, 3.0),
        directed_failed_branches=[branch],
        failure_buffer_m=0.30,
    )

    assert egress.status == "PASS"
    assert egress.path == ((5.5, 3.0), (0.5, 3.0))
    assert reentry.status == "NO_ROUTE"


def test_committed_route_is_retained_until_new_occupied_space_crosses_it():
    grid = np.zeros((80, 120), dtype=np.int16)
    planner = DynamicVisibilityPlanner(inflation_radius_m=0.20)
    route = [(1.0, 2.0), (5.0, 2.0)]

    assert planner.path_is_traversable(route, grid, 0.05, (0.0, 0.0))

    # A newly observed wall revokes the old route; unrelated map growth does
    # not force a different topological direction.
    grid[:, 59:62] = 100
    assert not planner.path_is_traversable(route, grid, 0.05, (0.0, 0.0))


def test_dead_end_egress_uses_only_the_current_map_safe_prefix():
    grid = np.zeros((80, 120), dtype=np.int16)
    planner = DynamicVisibilityPlanner(inflation_radius_m=0.20)
    route = [(1.0, 2.0), (2.0, 2.0), (4.0, 2.0)]
    grid[:, 59:62] = 100

    prefix = planner.longest_traversable_prefix(
        route,
        grid,
        0.05,
        (0.0, 0.0),
        maximum_unknown_fraction=0.05,
    )

    assert prefix == ((1.0, 2.0), (2.0, 2.0))


def test_dead_end_egress_can_leave_conservative_inflation_overlap():
    grid = np.zeros((80, 100), dtype=np.int16)
    grid[:, 19:21] = 100
    planner = DynamicVisibilityPlanner(inflation_radius_m=0.20)

    prefix = planner.longest_margin_egress_prefix(
        [(1.16, 2.0), (1.35, 2.0), (1.60, 2.0)],
        grid,
        0.05,
        (0.0, 0.0),
    )

    assert len(prefix) == 3


def test_dead_end_egress_cannot_move_deeper_or_cross_a_real_wall():
    grid = np.zeros((80, 100), dtype=np.int16)
    grid[:, 19:21] = 100
    planner = DynamicVisibilityPlanner(inflation_radius_m=0.20)

    assert not planner.longest_margin_egress_prefix(
        [(1.16, 2.0), (1.10, 2.0)], grid, 0.05, (0.0, 0.0)
    )
    assert not planner.longest_margin_egress_prefix(
        [(1.16, 2.0), (1.00, 2.0), (0.80, 2.0)],
        grid,
        0.05,
        (0.0, 0.0),
    )


def test_known_terminal_handoff_rejects_unknown_or_occupied_segment():
    planner = DynamicVisibilityPlanner(inflation_radius_m=0.20)
    route = [(1.0, 2.0), (2.5, 2.0)]
    grid = np.zeros((80, 100), dtype=np.int16)

    assert planner.path_is_traversable(
        route,
        grid,
        0.05,
        (0.0, 0.0),
        maximum_unknown_fraction=0.02,
    )
    grid[:, 35:45] = -1
    assert not planner.path_is_traversable(
        route,
        grid,
        0.05,
        (0.0, 0.0),
        maximum_unknown_fraction=0.02,
    )
    grid[:, 35:45] = 0
    grid[:, 39:42] = 100
    assert not planner.path_is_traversable(
        route,
        grid,
        0.05,
        (0.0, 0.0),
        maximum_unknown_fraction=0.02,
    )


def test_visibility_module_contains_no_dense_grid_astar_contract():
    import inspect
    import dep_car.runtime.far_visibility as module

    source = inspect.getsource(module).lower()
    assert "astar" not in source
    assert "a_star" not in source
    assert "heapq" in source
    assert "findcontours" in source


def test_attemptable_route_needs_distinct_stable_slam_observations():
    gate = VisibilityRouteAcquisitionGate(
        minimum_confirmations=2, minimum_stable_s=0.5
    )
    plan = VisibilityPlan(
        status="PASS",
        mode="ATTEMPTABLE_VISIBILITY",
        path=((0.0, 0.0), (1.0, 0.1), (3.0, 0.2)),
        nodes=(),
        edges=(),
        path_cost=3.2,
        known_edges=0,
        attemptable_edges=2,
        reason="attempt_unknown_route",
    )

    first = gate.update(plan, stamp=0.0, map_revision=1)
    # Reprocessing the same observation cannot confirm itself.  A later SLAM
    # publication receives a new observation sequence even when the occupancy
    # bytes are unchanged, so it can confirm temporal route stability without
    # an expensive graph rebuild.
    duplicate = gate.update(plan, stamp=0.6, map_revision=1)
    accepted = gate.update(plan, stamp=0.7, map_revision=2)

    assert not first.accepted and first.confirmations == 1
    assert not duplicate.accepted and duplicate.confirmations == 1
    assert accepted.accepted and accepted.confirmations == 2


def test_known_visibility_route_needs_distinct_content_stability():
    gate = VisibilityRouteAcquisitionGate(minimum_confirmations=3)
    plan = VisibilityPlan(
        status="PASS",
        mode="KNOWN_VISIBILITY",
        path=((0.0, 0.0), (2.0, 0.0)),
        nodes=(),
        edges=(),
        path_cost=2.0,
        known_edges=1,
        attemptable_edges=0,
        reason="known_route",
    )
    first = gate.update(plan, stamp=0.0, map_revision=1)
    duplicate = gate.update(plan, stamp=0.7, map_revision=1)
    second = gate.update(plan, stamp=0.8, map_revision=2)
    accepted = gate.update(plan, stamp=1.0, map_revision=3)

    assert not first.accepted
    assert duplicate.confirmations == 1
    assert not second.accepted
    assert accepted.accepted
    assert accepted.reason == "stable_known_visibility_route"


def test_new_measured_sensor_pose_confirms_route_on_unchanged_map():
    gate = VisibilityRouteAcquisitionGate(
        minimum_confirmations=2,
        minimum_stable_s=0.10,
        minimum_observer_displacement_m=0.25,
    )
    plan = VisibilityPlan(
        status="PASS",
        mode="KNOWN_VISIBILITY",
        path=((0.0, 0.0), (2.0, 0.0)),
        nodes=(),
        edges=(),
        path_cost=2.0,
        known_edges=1,
        attemptable_edges=0,
        reason="known_route",
    )

    first = gate.update(
        plan, stamp=0.0, map_revision=4, observer_position=(0.0, 0.0)
    )
    stationary = gate.update(
        plan, stamp=0.2, map_revision=4, observer_position=(0.1, 0.0)
    )
    moved = gate.update(
        plan, stamp=0.3, map_revision=4, observer_position=(0.30, 0.0)
    )

    assert first.confirmations == 1 and not first.motion_authorized
    assert stationary.confirmations == 1 and not stationary.motion_authorized
    assert moved.confirmations == 2 and moved.motion_authorized
    assert moved.accepted


def test_no_route_revokes_stale_route_acquisition_authority():
    gate = VisibilityRouteAcquisitionGate(minimum_confirmations=2)
    known = VisibilityPlan(
        status="PASS",
        mode="KNOWN_VISIBILITY",
        path=((0.0, 0.0), (2.0, 0.0)),
        nodes=(),
        edges=(),
        path_cost=2.0,
        known_edges=1,
        attemptable_edges=0,
        reason="known_route",
    )
    disconnected = VisibilityPlan(
        status="NO_ROUTE",
        mode="NONE",
        path=(),
        nodes=(),
        edges=(),
        path_cost=None,
        known_edges=0,
        attemptable_edges=0,
        reason="visibility_graph_disconnected",
    )

    gate.update(known, stamp=0.0, map_revision=1)
    assert gate.update(known, stamp=0.8, map_revision=2).accepted
    decision = gate.update(disconnected, stamp=1.0, map_revision=3)

    assert not decision.accepted
    assert decision.confirmations == 0
    assert not gate.accepted
    assert gate.confirmations == 0
    assert gate.last_bearing is None


def test_materially_different_attemptable_route_must_reacquire_authority():
    gate = VisibilityRouteAcquisitionGate(
        minimum_confirmations=2,
        minimum_stable_s=0.01,
        maximum_bearing_change_rad=0.4,
    )
    forward = VisibilityPlan(
        status="PASS",
        mode="ATTEMPTABLE_VISIBILITY",
        path=((0.0, 0.0), (2.0, 0.0)),
        nodes=(),
        edges=(),
        path_cost=2.0,
        known_edges=0,
        attemptable_edges=1,
        reason="attempt_unknown_route",
    )
    opposite_side = VisibilityPlan(
        status="PASS",
        mode="ATTEMPTABLE_VISIBILITY",
        path=((0.0, 0.0), (0.0, 2.0)),
        nodes=(),
        edges=(),
        path_cost=2.0,
        known_edges=0,
        attemptable_edges=1,
        reason="attempt_unknown_route",
    )

    gate.update(forward, stamp=0.0, map_revision=1)
    assert gate.update(forward, stamp=0.1, map_revision=2).accepted
    replacement = gate.update(opposite_side, stamp=0.2, map_revision=3)

    assert not replacement.accepted
    assert replacement.confirmations == 1


def test_unknown_high_detour_cannot_steer_before_it_is_observed():
    gate = VisibilityRouteAcquisitionGate(
        minimum_confirmations=2,
        minimum_stable_s=0.5,
        maximum_attemptable_detour_ratio=2.0,
    )
    plan = VisibilityPlan(
        status="PASS",
        mode="ATTEMPTABLE_VISIBILITY",
        path=((0.0, 0.0), (-2.0, 0.0), (2.0, 0.0)),
        nodes=(),
        edges=(),
        path_cost=6.0,
        known_edges=0,
        attemptable_edges=2,
        reason="attempt_unknown_route",
    )

    first = gate.update(plan, stamp=0.0, map_revision=1)
    second = gate.update(plan, stamp=1.0, map_revision=2)

    assert not first.accepted and not first.motion_authorized
    assert not second.accepted and not second.motion_authorized
    assert second.confirmations == 2
    assert second.reason == "high_detour_route_needs_more_observed_space"


def test_mostly_observed_maze_detour_is_accepted_after_extra_stability():
    gate = VisibilityRouteAcquisitionGate(
        minimum_confirmations=2,
        minimum_stable_s=0.5,
        maximum_attemptable_detour_ratio=2.0,
        maximum_high_detour_unknown_fraction=0.20,
        high_detour_extra_confirmations=2,
        high_detour_minimum_stable_s=1.2,
    )
    plan = VisibilityPlan(
        status="PASS",
        mode="ATTEMPTABLE_VISIBILITY",
        path=((0.0, 0.0), (-2.0, 0.0), (2.0, 0.0)),
        nodes=(),
        edges=(),
        path_cost=6.1,
        known_edges=0,
        attemptable_edges=2,
        reason="attempt_unknown_route",
        path_length=6.0,
        path_unknown_fraction=0.10,
    )

    decisions = [
        gate.update(plan, stamp=stamp, map_revision=revision)
        for revision, stamp in enumerate((0.0, 0.4, 0.8, 1.3), start=1)
    ]

    assert not any(decision.accepted for decision in decisions[:-1])
    assert decisions[2].motion_authorized
    assert decisions[-1].accepted
    assert decisions[-1].confirmations == 4
    assert decisions[-1].reason == "stable_observed_high_detour_route"


def test_visibility_search_softly_prefers_forward_compatible_first_edge():
    planner = DynamicVisibilityPlanner(
        start_heading_weight_m=1.2, reverse_start_penalty_m=2.0
    )
    nodes = (
        VisibilityNode(0, 0.0, 0.0, "START"),
        VisibilityNode(1, 2.0, 0.0, "GOAL"),
        VisibilityNode(2, -0.5, 0.0, "REAR_SHORTCUT"),
        VisibilityNode(3, 0.5, 0.5, "FORWARD_ARC"),
    )
    edges = (
        VisibilityEdge(0, 2, 0.5, 0.0),
        VisibilityEdge(2, 1, 1.0, 0.0),
        VisibilityEdge(0, 3, 0.71, 0.0),
        VisibilityEdge(3, 1, 1.58, 0.0),
    )

    point_robot_ids, _ = planner._search(
        nodes,
        edges,
        known_only=True,
        unknown_cost_weight=1.0,
        start_yaw=None,
    )
    ackermann_ids, _ = planner._search(
        nodes,
        edges,
        known_only=True,
        unknown_cost_weight=1.0,
        start_yaw=0.0,
    )

    assert point_robot_ids == [0, 2, 1]
    assert ackermann_ids == [0, 3, 1]
