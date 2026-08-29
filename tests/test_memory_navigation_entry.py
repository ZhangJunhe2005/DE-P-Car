import hashlib
import json
from pathlib import Path
import sys

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import run_memory_navigation as runner


def test_v43_reproduction_scenario_passes_frozen_map_and_start_preflight():
    config = yaml.safe_load(
        (ROOT / "dep_car/config/p6_memory_navigation_v43_shadow.yaml").read_text(
            encoding="utf-8"
        )
    )
    manifest, manifest_path = runner.load_scenario_manifest(config)
    assert manifest_path == (ROOT / "data/p6_static/reproduction_manifest.json")
    report = runner.scenario_artifact_preflight(manifest["scenarios"][0])
    assert report["status"] == "PASS"
    assert report["start_robustness_passed"] is True
    assert report["errors"] == []


def test_scenario_preflight_rejects_hash_change_and_nonrobust_start():
    config = yaml.safe_load(
        (ROOT / "dep_car/config/p6_memory_navigation_v43_shadow.yaml").read_text(
            encoding="utf-8"
        )
    )
    manifest, _ = runner.load_scenario_manifest(config)
    scenario = json.loads(json.dumps(manifest["scenarios"][0]))
    scenario["world_sha256"] = "0" * 64
    scenario["start_robustness"]["status"] = "FAIL"
    report = runner.scenario_artifact_preflight(scenario)
    assert report["status"] == "FAIL"
    assert "world SHA-256 mismatch" in " ".join(report["errors"])
    assert "start robustness evidence is not PASS" in report["errors"]


def test_full_manifest_identity_is_self_consistent():
    path = ROOT / "data/p6_static/scenario_manifest.json"
    if not path.is_file():
        pytest.skip("full generated P6 corpus is intentionally not bundled")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    content = dict(manifest)
    expected = content.pop("scenario_manifest_sha256")
    observed = hashlib.sha256(
        json.dumps(content, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert observed == expected


def test_memory_launch_is_dual_backend_and_excludes_frozen_map_authority():
    urban = (ROOT / "ros/dep_car_bringup/launch/urban_sim.launch").read_text(
        encoding="utf-8"
    )
    memory = (
        ROOT / "ros/dep_car_bringup/launch/p6_memory_static.launch"
    ).read_text(encoding="utf-8")
    assert "navigation_backend" in urban
    assert "arg('navigation_backend') == 'astar'" in urban
    assert "arg('navigation_backend') == 'memory'" in urban
    assert '<arg name="map_yaml" value=""/>' in memory
    assert '<arg name="navigation_backend" value="memory"/>' in memory
    assert '<arg name="odometry_topic" value="/dep_car/map_odometry"/>' in memory
    assert '<arg name="policy_odometry_topic" value="/odometry/filtered"/>' in memory


def test_memory_authority_sources_never_subscribe_to_gazebo_truth():
    package = ROOT / "ros/dep_car_memory_navigation"
    for path in package.rglob("*"):
        if path.is_file():
            assert "/base_pose_ground_truth" not in path.read_text(
                encoding="utf-8", errors="ignore"
            )
    ekf = yaml.safe_load((package / "config/ekf.yaml").read_text(encoding="utf-8"))
    assert ekf["odom0"] == "/dep_car/wheel_odom"
    assert ekf["imu0"] == "/imu/data"
    assert ekf["world_frame"] == "odom"
    assert ekf["frequency"] == 100
    # Wheel integration carries only the frozen x/y reference and signed
    # longitudinal speed.  Gazebo IMU yaw + yaw rate are the sole heading
    # authority, so steering noise cannot fight a second integrated yaw.
    assert ekf["odom0_config"] == [
        True, True, False,
        False, False, False,
        True, False, False,
        False, False, False,
        False, False, False,
    ]
    assert ekf["imu0_config"] == [
        False, False, False,
        False, False, True,
        False, False, False,
        False, False, True,
        False, False, False,
    ]
    assert ekf["imu0_relative"] is False


def test_fixed_and_interactive_entries_bind_frozen_world_but_not_map_yaml():
    config = yaml.safe_load(
        (ROOT / "dep_car/config/p6_memory_navigation.yaml").read_text(
            encoding="utf-8"
        )
    )
    _, scenario = runner.load_scenario(config, config["fixed_default_scenario"])

    class Args:
        stage = "fixed"
        policy_mode = "shadow"
        headless = False
        goal_x = None
        goal_y = None
        goal_yaw = None

    launch, reset, publish, goal = runner.command(config, scenario, Args(), 11431)
    joined = " ".join(launch)
    assert "p6_memory_static.launch" in joined
    assert "map_yaml:=" not in joined
    assert "world:=" in joined
    assert reset[-6:] == [
        "--x", str(scenario["start"][0]),
        "--y", str(scenario["start"][1]),
        "--yaw", str(scenario["start"][2]),
    ]
    assert goal == config["fixed_recovery_probe"]["goal"]
    assert config["fixed_recovery_probe"]["expected_terminal_state"] == "AUTOMATIC_GOAL_RESUME"
    assert "publish_memory_goal.py" in publish
    assert publish[-2:] == ["--frame", "odom"]


def test_episode_replay_uses_effective_cli_overridden_goal():
    command = runner.episode_automated_goal_command(
        [6.457, -5.388, 0.0], 30.0, ROOT / "reports/unused.json"
    )
    goal_index = command.index("--goal")
    assert command[goal_index + 1 : goal_index + 4] == ["6.457", "-5.388", "0.0"]
    assert command[command.index("--frame") + 1] == "odom"


def test_regression_goal_preflight_rejects_stale_map_frame_and_wall_goal():
    config = yaml.safe_load(
        (ROOT / "dep_car/config/p6_memory_navigation_v43_shadow.yaml").read_text(
            encoding="utf-8"
        )
    )
    _, scenario = runner.load_scenario(
        config, config["interactive_default_scenario"]
    )
    invalid = runner.regression_goal_preflight(
        {
            "coordinate_frame": "map",
            "goals": [[6.536, -4.331, 0.0]],
        },
        scenario,
    )
    assert invalid["status"] == "FAIL"
    assert invalid["reason"] == "INVALID_REPLAY_GOAL"
    assert invalid["goals"][0]["minimum_static_clearance_m"] == 0.0


def test_v43_regression_goals_are_stable_odom_points_with_static_clearance():
    config = yaml.safe_load(
        (ROOT / "dep_car/config/p6_memory_navigation_v43_shadow.yaml").read_text(
            encoding="utf-8"
        )
    )
    _, scenario = runner.load_scenario(
        config, config["interactive_default_scenario"]
    )
    for sequence in config["regression_sequences"].values():
        result = runner.regression_goal_preflight(
            sequence,
            scenario,
            sequence["minimum_static_goal_clearance_m"],
        )
        assert result["status"] == "PASS"
        command = runner.automated_goal_command(
            sequence["goals"], 10.0, frame=sequence["coordinate_frame"]
        )
        assert command[command.index("--frame") + 1] == "odom"


def test_local_route_command_has_explicit_memory_control_modes():
    message = (ROOT / "ros/dep_car_msgs/msg/LocalRouteCommand.msg").read_text(
        encoding="utf-8"
    )
    assert "NAVIGATION_MEMORY_BACKTRACK=2" in message
    assert "NAVIGATION_MEMORY_RESUME=3" in message
    assert "NAVIGATION_FAR_DEAD_END_EGRESS=4" in message
    assert "uint8 navigation_mode" in message
    assert "string route_id" in message
    assert "string route_source" in message
    assert "uint32 route_revision" in message
    assert "uint64 authority_epoch" in message
    assert "bool rolling_target_latched" in message


def test_logged_regression_is_input_only_and_multi_seed_manifest_remains_available():
    config = yaml.safe_load(
        (ROOT / "dep_car/config/p6_memory_navigation.yaml").read_text(
            encoding="utf-8"
        )
    )
    sequence = config["regression_sequences"]["logged_t_junction_turnaround"]
    assert sequence["scenario_id"] == config["interactive_default_scenario"]
    assert len(sequence["goals"]) == 2
    latest_sequence = config["regression_sequences"][
        "latest_interactive_turnaround_20260821"
    ]
    assert latest_sequence["scenario_id"] == config["interactive_default_scenario"]
    assert len(latest_sequence["goals"]) == 2
    runtime = (
        ROOT / "ros/dep_car_memory_navigation/scripts/navigation_memory_node.py"
    ).read_text(encoding="utf-8")
    assert "logged_t_junction_turnaround" not in runtime
    assert "6.536" not in runtime
    full_manifest = ROOT / "data/p6_static/scenario_manifest.json"
    if not full_manifest.is_file():
        pytest.skip("full generated P6 corpus is intentionally not bundled")
    manifest = json.loads(full_manifest.read_text(encoding="utf-8"))
    development_maps = {
        row["map_uuid"] for row in manifest["scenarios"] if row["cohort"] == "development"
    }
    holdout_maps = {
        row["map_uuid"] for row in manifest["scenarios"] if row["cohort"] == "holdout"
    }
    assert len(development_maps) > 1
    assert holdout_maps
    assert development_maps.isdisjoint(holdout_maps)


def test_bounded_matrix_prioritizes_distinct_reproducible_map_seeds():
    config = yaml.safe_load(
        (ROOT / "dep_car/config/p6_memory_navigation.yaml").read_text(
            encoding="utf-8"
        )
    )
    full_manifest = ROOT / "data/p6_static/scenario_manifest.json"
    if not full_manifest.is_file():
        pytest.skip("full generated P6 corpus is intentionally not bundled")
    manifest = json.loads(full_manifest.read_text(encoding="utf-8"))

    class Args:
        cohort = "development"
        maximum_scenarios = 8
        selection_seed = 12345
        config = ROOT / "dep_car/config/p6_memory_navigation.yaml"
        policy_mode = "shadow"
        goal_timeout = 90.0

    first = runner.matrix_commands(config, manifest, Args())
    second = runner.matrix_commands(config, manifest, Args())
    assert [row[0] for row in first] == [row[0] for row in second]
    selected_map_seeds = [row[2]["map_seed"] for row in first]
    assert len(selected_map_seeds) == len(set(selected_map_seeds))


def test_matrix_children_preserve_scenario_manifest_override():
    config = yaml.safe_load(
        (ROOT / "dep_car/config/p6_memory_navigation_v43_shadow.yaml").read_text(
            encoding="utf-8"
        )
    )
    manifest_path = ROOT / "data/p6_static/scenario_manifest.json"
    if not manifest_path.is_file():
        pytest.skip("full generated P6 corpus is intentionally not bundled")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    class Args:
        cohort = "development"
        maximum_scenarios = 1
        selection_seed = 12345
        config = ROOT / "dep_car/config/p6_memory_navigation_v43_shadow.yaml"
        scenario_manifest = Path("data/p6_static/scenario_manifest.json")
        policy_mode = "shadow"
        goal_timeout = 90.0

    _, command, _ = runner.matrix_commands(config, manifest, Args())[0]
    index = command.index("--scenario-manifest")
    assert command[index + 1] == str(manifest_path.resolve())


def test_m5_uses_accumulated_slam_map_without_reintroducing_dense_global_astar():
    source = (
        ROOT / "ros/dep_car_memory_navigation/scripts/navigation_memory_node.py"
    ).read_text(encoding="utf-8")
    runtime = (
        ROOT / "dep_car/src/dep_car/runtime/navigation_memory.py"
    ).read_text(encoding="utf-8")
    config = (
        ROOT / "ros/dep_car_memory_navigation/config/navigation_memory.yaml"
    ).read_text(encoding="utf-8")
    launch = (
        ROOT / "ros/dep_car_bringup/launch/p6_memory_static.launch"
    ).read_text(encoding="utf-8")
    assert "accumulated_map_topic: /map" in config
    assert "unknown_is_occupied=False" in source
    assert "class TopologicalMemory" in runtime
    assert "mark_failed_branch" in runtime
    assert '"uses_full_map_search": False' in source
    assert "hybrid_astar" not in launch


def test_m5_exposes_navigation_memory_and_progress_bounded_recovery_in_rviz():
    source = (
        ROOT / "ros/dep_car_memory_navigation/scripts/navigation_memory_node.py"
    ).read_text(encoding="utf-8")
    rviz = (
        ROOT / "ros/dep_car_bringup/rviz/dep_car.rviz"
    ).read_text(encoding="utf-8")
    replay = (
        ROOT / "ros/dep_car_memory_navigation/scripts/replay_memory_goals.py"
    ).read_text(encoding="utf-8")
    assert "/dep_car/navigation_memory_markers" in source
    assert "Navigation Memory" in rviz
    assert "resume_target_index" in source
    assert "resume_maximum_travel" in source
    assert "preempted_state" in source
    assert "DEPCarMemoryGoalReplayV3" in replay


def test_m5_7_is_stateful_boundary_following_without_dense_grid_search():
    runtime = (
        ROOT / "dep_car/src/dep_car/runtime/navigation_memory.py"
    ).read_text(encoding="utf-8")
    source = (
        ROOT / "ros/dep_car_memory_navigation/scripts/navigation_memory_node.py"
    ).read_text(encoding="utf-8")
    config = (
        ROOT / "ros/dep_car_memory_navigation/config/navigation_memory.yaml"
    ).read_text(encoding="utf-8")
    assert "class BoundaryFollowSupervisor" in runtime
    assert "class BoundarySideFailure" in runtime
    assert "returned_to_boundary_hit_region" in runtime
    assert "progressing_direct_corridor_confirmed" in runtime
    assert "BOUNDARY_FOLLOW_LEFT" in source
    assert "boundary_loop_reason" in source
    assert "remember_current_failure" in source
    assert "boundary_leave_progress_m" in config
    assert "boundary_failure_radius_m" in config
    assert "uses_full_map_search" in source
    assert "hybrid_astar" not in (
        ROOT / "ros/dep_car_bringup/launch/p6_memory_static.launch"
    ).read_text(encoding="utf-8")


def test_each_goal_reanchors_recovery_stack_but_keeps_sparse_memory():
    source = (
        ROOT / "ros/dep_car_memory_navigation/scripts/navigation_memory_node.py"
    ).read_text(encoding="utf-8")
    goal_handler = source.split("def on_goal", 1)[1].split("def lookup_pose", 1)[0]
    assert "self.trail.clear()" in goal_handler
    assert "self.topology.reanchor" in goal_handler
    assert "self.topology = TopologicalMemory" not in goal_handler


def test_m6_routes_on_dynamic_polygon_visibility_graph_without_grid_astar():
    runtime = (
        ROOT / "dep_car/src/dep_car/runtime/far_visibility.py"
    ).read_text(encoding="utf-8")
    source = (
        ROOT / "ros/dep_car_memory_navigation/scripts/navigation_memory_node.py"
    ).read_text(encoding="utf-8")
    launch = (
        ROOT / "ros/dep_car_bringup/launch/p6_memory_static.launch"
    ).read_text(encoding="utf-8")
    assert "class DynamicVisibilityPlanner" in runtime
    assert "findContours" in runtime
    assert "KNOWN_VISIBILITY" in runtime
    assert "ATTEMPTABLE_VISIBILITY" in runtime
    assert "astar" not in runtime.lower()
    assert "FAR_KNOWN_VISIBILITY" in source
    assert "visibility_corridor_body" in source
    assert '"uses_dynamic_visibility_graph": True' in source
    assert "hybrid_astar" not in launch


def test_m6_uses_one_bounded_slam_correction_and_reports_rviz_drift_evidence():
    source = (
        ROOT / "ros/dep_car_memory_navigation/scripts/map_odometry_node.py"
    ).read_text(encoding="utf-8")
    replay = (
        ROOT / "ros/dep_car_memory_navigation/scripts/replay_memory_goals.py"
    ).read_text(encoding="utf-8")
    assert "maximum_transform_skew_s" in source
    assert "Mixing an" in source
    assert "rejected stale SLAM TF skew" in source
    assert "/dep_car/map_odom_correction" in source
    assert "DEPCarMapOdomCorrectionV1" in source
    assert "PlanarTransformRevisionTracker" in source
    assert "if revision is not None:" in source
    assert "maximum_map_odom_translation_correction_m" in replay
    assert "time_aligned_map_odom_rate" in replay
    assert "maximum_stationary_excursion" in replay


def test_m6_far_route_owns_recovery_and_memory_markers_use_one_rigid_tf():
    source = (
        ROOT / "ros/dep_car_memory_navigation/scripts/navigation_memory_node.py"
    ).read_text(encoding="utf-8")
    config = (
        ROOT / "ros/dep_car_memory_navigation/config/navigation_memory.yaml"
    ).read_text(encoding="utf-8")
    assert "breadcrumb_motion_authority: false" in config
    assert 'backtrack_enabled=self.breadcrumb_motion_authority' in source
    assert '"topology_anchor_and_closed_loop_far_dead_end_egress"' in source
    assert "force_certified_egress" in source
    assert "certified_far_egress=True" in source
    assert "FAR_DEAD_END_EGRESS" in source
    assert "directed_failed_branches" in source
    assert "validate_dead_end_egress_route" in source
    assert "live_dead_end_egress_reanchor" in source
    assert "FAILED_BRANCH_EXIT_LOCK" in source
    assert "failed_branch_exit_lock_waiting_for_branch_safe_far_route" in source
    assert "EGRESS_REANCHOR_EXHAUSTED_CURRENT_MAP" in source
    assert "dead_end_escape_lookahead_m" in config
    assert "failed_branch_exit_lock_progress_m" in config
    assert "far_static_replans_before_egress" in config
    assert '"confirmed_local_static_block"' in source
    assert '"slam_map_odom_correction"' in source
    correction_handler = source.split(
        "def apply_pending_map_correction", 1
    )[1].split("def on_goal", 1)[0]
    assert "slam_map_odom_revalidation" in correction_handler
    assert "invalidate_visibility_route" not in correction_handler
    assert "map_correction_replan_minimum_period_s" in config
    assert "last_significant_map_correction" in source
    assert "FAR_ATTEMPTABLE_NAVIGATION" in source
    assert "LOCAL_SAFE_EXPLORATION" in source
    assert '"EXPLORED_TOPOLOGY_ROUTE"' in source
    assert "explored_topology_corridor_body" in source
    assert "no_route_rolling_local_exploration" in source
    assert "recent_far_authority_dropout_local_continuation" in source
    assert "explored_topology_motion_authority: false" in config
    assert "TOPOLOGY_MOTION_AUTHORITY_DISABLED" in source
    assert "visibility_terminal_direct_handoff_radius_m" in config
    assert "KNOWN_TERMINAL_DIRECT" in source
    assert "allow_direct_goal=False" in source
    invalidation = source.split(
        "def invalidate_visibility_route", 1
    )[1].split("def sync_visibility_maneuver_transaction", 1)[0]
    assert "visibility_route_acquisition_started_stamp = None" not in invalidation
    local = (
        ROOT / "ros/dep_car_local_planner/scripts/local_planner_node.py"
    ).read_text(encoding="utf-8")
    launch = (
        ROOT / "ros/dep_car_local_planner/launch/local_planner.launch"
    ).read_text(encoding="utf-8")
    assert "breadcrumb_backtrack or far_dead_end_egress" in local
    assert '"far_dead_end_egress_realign"' in local
    assert "egress_bidirectional_safety_exhausted" in source
    assert "bidirectional_hard_safety_exhausted=True" in source
    assert "start_authority_transaction" in local
    assert "maneuver_retry_observation_hold" in local
    assert "maneuver_retry_observation_hold_s" in launch
    marker_source = source.split("def publish_memory_markers", 1)[1].split(
        "def progress", 1
    )[0]
    assert 'marker.header.frame_id = "odom"' in marker_source
    assert "marker.frame_locked = True" in marker_source
    assert 'marker.header.frame_id = "map"' in marker_source
    assert 'marker.ns = "map_driven_history"' in marker_source
    assert 'points.ns = "map_breadcrumb_points"' in marker_source
    assert "memory_breadcrumbs_odom_control" not in marker_source
    assert 'open_edges = base_marker' not in marker_source
    assert "lookup_transform" not in marker_source


def test_static_local_block_promotes_only_an_observed_far_recovery_prefix():
    memory = (
        ROOT / "ros/dep_car_memory_navigation/scripts/navigation_memory_node.py"
    ).read_text(encoding="utf-8")

    certification = memory.split(
        "def far_recovery_prefix_authorized", 1
    )[1].split("def bind_far_recovery_candidate", 1)[0]
    assert 'plan.status != "PASS"' in certification
    assert "polyline_prefix" in certification
    assert "maximum_unknown_fraction=0.02" in certification
    assert "polyline_enters_failed_branch" in certification

    blocked = memory.split(
        "blocked_guidance_source = str(self.last_guidance_source)", 1
    )[1].split("previous_state = self.recovery.state", 1)[0]
    assert '"LOCAL_SAFE_EXPLORATION"' in blocked
    assert "visibility_static_recovery_pending = True" in blocked
    assert "and not static_recovery_newly_armed" in blocked

    candidate = memory.split(
        "static_recovery_motion_authorized = bool(", 1
    )[1].split("bootstrap_motion_authorized", 1)[0]
    assert "visibility_static_recovery_pending" in candidate
    assert "far_recovery_prefix_authorized" in candidate
    assert "visibility_no_route_static_evidence_timeout" in candidate
    assert "observed_far_detour_after_local_hard_block" in memory


def test_far_dead_end_live_anchor_is_latched_and_recovery_stop_is_recoverable():
    memory = (
        ROOT / "ros/dep_car_memory_navigation/scripts/navigation_memory_node.py"
    ).read_text(encoding="utf-8")
    local = (
        ROOT / "ros/dep_car_local_planner/scripts/local_planner_node.py"
    ).read_text(encoding="utf-8")
    config = (
        ROOT / "ros/dep_car_memory_navigation/config/navigation_memory.yaml"
    ).read_text(encoding="utf-8")

    reanchor = memory.split(
        "def live_dead_end_egress_reanchor", 1
    )[1].split("def failed_branch_by_id", 1)[0]
    assert "dead_end_escape_live_target_index" in reanchor
    assert "Latched live FAR dead-end egress anchor" in reanchor
    assert reanchor.count("dead_end_escape_route_revision += 1") == 1
    assert "dead_end_escape_live_target_capture" in reanchor

    backtrack = memory.split("def backtrack_route", 1)[1].split(
        "def resume_route", 1
    )[0]
    assert "EGRESS_REANCHOR_REVALIDATION_HOLD" in backtrack
    assert "dead_end_escape_connector_unavailable_since" in backtrack
    assert "dead_end_escape_divergence_confirmation" in backtrack
    assert "EGRESS_REANCHOR_WAIT" in memory
    assert '"EGRESS_REANCHOR_WAIT"' in local
    assert "dead_end_escape_live_target_capture_m" in config

    safe_stop = memory.split(
        "elif self.recovery.state == MemoryNavigationState.SAFE_STOP", 1
    )[1]
    assert "branch_safe_far_route_recovered_egress_stop" in safe_stop
    assert "require_failed_branch_avoidance=True" in safe_stop
    assert "RECOVERED_BY_BRANCH_SAFE_FAR_ROUTE" in safe_stop


def test_m6_dead_end_egress_lifecycle_is_in_replay_reports():
    replay = (
        ROOT / "ros/dep_car_memory_navigation/scripts/replay_memory_goals.py"
    ).read_text(encoding="utf-8")
    assert "far_dead_end_egress_transactions" in replay
    assert "far_dead_end_egress_completion_reasons" in replay
    assert "maximum_far_dead_end_egress_target_distance_m" in replay
    assert "maximum_far_dead_end_egress_cross_track_error_m" in replay
    assert "maximum_far_dead_end_egress_map_reanchors" in replay


def test_exhausted_local_turnaround_revalidates_before_reporting_failure():
    local = (
        ROOT / "ros/dep_car_local_planner/scripts/local_planner_node.py"
    ).read_text(encoding="utf-8")
    branch = local.split(
        'self.stop("forward_restoration_budget_revalidation")', 1
    )[1].split("return", 1)[0]
    assert "forward_restoration_budget_replan_pending" in local
    assert "executable_override=False" in branch
    assert "blocked_by_static_override=False" in branch
    assert 'self.stop("forward_restoration_budget_exhausted")' in branch


def test_far_dead_end_egress_uses_signed_connector_not_rear_ray_gate():
    memory = (
        ROOT / "ros/dep_car_memory_navigation/scripts/navigation_memory_node.py"
    ).read_text(encoding="utf-8")
    launch = (
        ROOT / "ros/dep_car_local_planner/launch/local_planner.launch"
    ).read_text(encoding="utf-8")
    branch = memory.split("request_certified_far_egress = bool(", 1)[1].split(
        ")\n                    if failed_side", 1
    )[0]
    assert "certified_trail_m" in branch
    assert 'features["rear"]' not in branch
    assert "signed, closed-loop exit" in memory


def test_m6_far_handoff_is_atomic_and_frozen_during_local_turnaround():
    memory = (
        ROOT / "ros/dep_car_memory_navigation/scripts/navigation_memory_node.py"
    ).read_text(encoding="utf-8")
    local = (
        ROOT / "ros/dep_car_local_planner/scripts/local_planner_node.py"
    ).read_text(encoding="utf-8")
    launch = (
        ROOT / "ros/dep_car_local_planner/launch/local_planner.launch"
    ).read_text(encoding="utf-8")

    assert "sync_visibility_maneuver_transaction" in memory
    assert "visibility_maneuver_transaction_active" in memory
    assert "Deferred visibility replan" in memory
    assert "maximum_deviation=math.inf" in memory
    assert "path_is_traversable" in memory
    assert "visibility_route_direction_authority" in memory
    assert "FAR or a physically driven topology corridor already selected" in memory
    assert "commit_route_transaction_locked" in local
    assert ") and not self.maneuver.active:" in local
    assert "committed local Ackermann leg owns motion" in memory
    assert "pending_route_commands" in local
    assert "pending_routes" in local
    assert "Buffered unmatched FAR route transaction halves" in local
    assert "Queued complete route transaction until committed maneuver" in local
    assert "deferred_route_transaction" in local
    assert "route_transaction_stamp_tolerance" in launch
    assert "visibility_route_direction_authority: true" in (
        ROOT / "ros/dep_car_memory_navigation/config/navigation_memory.yaml"
    ).read_text(encoding="utf-8")


def test_course_capture_revalidation_is_released_by_a_fresh_far_transaction():
    local = (
        ROOT / "ros/dep_car_local_planner/scripts/local_planner_node.py"
    ).read_text(encoding="utf-8")
    memory = (
        ROOT / "ros/dep_car_memory_navigation/scripts/navigation_memory_node.py"
    ).read_text(encoding="utf-8")
    launch = (
        ROOT / "ros/dep_car_local_planner/launch/local_planner.launch"
    ).read_text(encoding="utf-8")

    # FAR can publish PASS plus a new synchronized transaction without an
    # observable MAPPING_WAIT edge.  The transaction stamp is therefore the
    # primary acknowledgement for the local revalidation barrier.
    assert "fresh_course_capture_reanchor" in local
    assert "command_stamp > self.forward_capture_replan_after_stamp" in local
    assert "and next_is_far" in local
    assert "Accepted fresh FAR route transaction" in local
    assert "self.forward_capture_replan_requested = True" in local

    # An intentional revalidation stop must not be promoted into a dead-end
    # and breadcrumb reverse by the mission-level memory authority.
    assert "local_route_revalidation_hold" in memory
    transaction_sync = memory.split(
        "local_maneuver_reported = bool(", 1
    )[1].split("self.sync_route_turnaround_transaction", 1)[0]
    assert "and not local_route_revalidation_hold" in transaction_sync
    assert "and not self.visibility_course_revalidation_pending" in (
        transaction_sync
    )
    assert "self.visibility_course_revalidation_pending = True" in memory
    assert "Published FAR route answer for local course revalidation" in memory
    assert "measured_pose_route_revalidation" in memory
    assert '"measured_pose_observed_prefix_far_revalidation"' in memory
    assert "polyline_prefix" in memory
    assert "maximum_unknown_fraction=0.02" in memory
    assert "self.visibility_course_revalidation_pending" in memory
    assert "and plan_age >= self.visibility_replan_period" in memory
    assert "Retained FAR route during course revalidation" in local
    assert "if not fresh_course_capture_reanchor:" in local
    assert "route_requires_far_revalidation" in local
    assert "Resolved rearward non-FAR route" in local
    assert "forward_course_revalidation_timeout_s" in launch
    assert "forward_course_revalidation_fallback" in local
    assert "forward_course_revalidation_fallback" in memory
    assert "stale_fallback_turnaround_replaced_by_far" in local
    assert '"EXPLORED_TOPOLOGY", "LOCAL_SAFE_EXPLORATION"' in local
    assert "Cancelled stale fallback turnaround" in local
    assert "stable_far_forward_exit_available" in local
    assert "forward_probe = None" in local
    assert '"forward_restoration_budget_exhausted"' in local
    assert '"forward_restoration_budget_exhausted"' in memory
    assert "forward-restoration scheduling budget exhausted" in memory
    static_gate = memory.split("static_blocked = bool(", 1)[1].split(
        "request_certified_far_egress", 1
    )[0]
    assert "not local_route_revalidation_hold" in static_gate


def test_m6_rolling_carrot_is_monotonic_and_rear_topology_has_one_transaction():
    runtime = (
        ROOT / "dep_car/src/dep_car/runtime/navigation_memory.py"
    ).read_text(encoding="utf-8")
    memory = (
        ROOT / "ros/dep_car_memory_navigation/scripts/navigation_memory_node.py"
    ).read_text(encoding="utf-8")
    local = (
        ROOT / "ros/dep_car_local_planner/scripts/local_planner_node.py"
    ).read_text(encoding="utf-8")
    config = (
        ROOT / "ros/dep_car_memory_navigation/config/navigation_memory.yaml"
    ).read_text(encoding="utf-8")

    assert "class MonotonicRouteProgress" in runtime
    assert "self.progress_m = max(self.progress_m, projected)" in runtime
    assert "candidate_tangent_discontinuous" in runtime
    assert "candidate_has_no_local_attachment" in runtime
    assert "rolling_route_minimum_lookahead_m" in config
    assert "rolling_route_maximum_lookahead_m" in config
    assert "visibility_cursor_path" in memory
    assert "active_rolling_route" in memory
    assert "TOPOLOGY_REAR_SUPPRESSED" in memory
    assert "sync_route_turnaround_transaction" in memory
    assert "topology_last_turnaround_route_id" in memory
    assert "route_turnaround_transaction" in memory
    assert "local_turnaround_transaction_id" in local
    # The handoff selects an attachment arclength.  There is deliberately no
    # affine fit, rotation or translation of FAR path geometry.
    handoff = runtime.split("def preview_handoff", 1)[1].split(
        "def bind", 1
    )[0]
    assert "linalg" not in handoff or "np.linalg.norm" in handoff
    assert "rotation" not in handoff


def test_far_no_route_cannot_fabricate_or_double_count_static_evidence():
    memory = (
        ROOT / "ros/dep_car_memory_navigation/scripts/navigation_memory_node.py"
    ).read_text(encoding="utf-8")

    blocked_gate = memory.split(
        "if (\n                static_blocked", 1
    )[1].split("previous_state = self.recovery.state", 1)[0]
    assert "and local_static_blocked" in blocked_gate
    assert "self.visibility_last_local_static_block_stamp = -math.inf" in blocked_gate

    route_hold = memory.split("if body is None:", 1)[1].split(
        "decision = self.last_heading_decision", 1
    )[0]
    assert "static_evidence_created=False" in route_hold
    assert "self.visibility_last_local_static_block_stamp = float" not in route_hold


def test_m6_rolling_subgoal_is_capture_gated_and_distance_bounded():
    runtime = (
        ROOT / "dep_car/src/dep_car/runtime/navigation_memory.py"
    ).read_text(encoding="utf-8")
    memory = (
        ROOT / "ros/dep_car_memory_navigation/scripts/navigation_memory_node.py"
    ).read_text(encoding="utf-8")
    config = (
        ROOT / "ros/dep_car_memory_navigation/config/navigation_memory.yaml"
    ).read_text(encoding="utf-8")

    assert "carrot_capture_radius_m" in runtime
    assert "maximum_carrot_advance_m" in runtime
    assert "maximum_carrot_distance_m" in runtime
    assert "waiting_for_vehicle_capture" in runtime
    assert "candidate_carrot_m" in runtime
    assert "rolling_target_world" in memory
    assert "rolling_target_latched=True" in memory
    assert "rolling_route_carrot_capture_radius_m" in config
    assert "rolling_route_maximum_carrot_advance_m" in config
    assert "rolling_route_maximum_carrot_distance_m" in config


def test_m6_far_handoff_and_local_turnaround_have_serial_authority():
    message = (
        ROOT / "ros/dep_car_msgs/msg/LocalRouteCommand.msg"
    ).read_text(encoding="utf-8")
    memory = (
        ROOT / "ros/dep_car_memory_navigation/scripts/navigation_memory_node.py"
    ).read_text(encoding="utf-8")
    local = (
        ROOT / "ros/dep_car_local_planner/scripts/local_planner_node.py"
    ).read_text(encoding="utf-8")
    preference = (
        ROOT / "dep_car/src/dep_car/runtime/forward_preference.py"
    ).read_text(encoding="utf-8")

    assert "uint64 authority_epoch" in message
    assert "string route_source" in message
    assert "FAR keeps planning in parallel" in memory
    active_branch = memory.split("if local_maneuver_active:", 1)[1].split(
        "goal_distance =", 1
    )[0]
    assert "self.update_visibility_plan(" in active_branch
    assert "deferred_control_handoff=True" in active_branch
    assert "request_route_revalidation" in local
    assert "Accepted atomic navigation authority handoff" in local
    assert "status_matches_command" in local
    assert "navigation_authority_reference" in local
    assert "LOCAL_SAFE_EXPLORATION" in preference
    assert "mission_goal_body" in preference


def test_committed_recovery_finishes_gear_shift_before_candidate_safety_check():
    local = (
        ROOT / "ros/dep_car_local_planner/scripts/local_planner_node.py"
    ).read_text(encoding="utf-8")
    active = local.split("if self.maneuver.active:", 1)[1].split(
        "requested_gear = Gear.require_drive", 1
    )[0]

    shift = active.index("shift_decision = self.gear_supervisor.update")
    observe = active.index("self.maneuver.observe(position, now)")
    safety_plan = active.index("maneuver_result, _ = self.plan_context")
    assert shift < observe < safety_plan
    assert "hold_for_drive_authorization" in active
    assert "static_recovery_far_replan_hold" in local


def test_no_route_uses_full_lidar_safe_local_ackermann_authority():
    memory = (
        ROOT / "ros/dep_car_memory_navigation/scripts/navigation_memory_node.py"
    ).read_text(encoding="utf-8")
    local = (
        ROOT / "ros/dep_car_local_planner/scripts/local_planner_node.py"
    ).read_text(encoding="utf-8")
    config = (
        ROOT / "ros/dep_car_memory_navigation/config/navigation_memory.yaml"
    ).read_text(encoding="utf-8")

    assert "FAR_ATTEMPTABLE_NAVIGATION" in memory
    assert "LOCAL_SAFE_EXPLORATION" in memory
    assert "no_route_rolling_local_exploration" in memory
    assert "self.visibility_active_route_motion_authorized" in memory
    assert "map_acquisition_forward_only" in memory
    route_mode = memory.split("route_transaction_mode =", 1)[1].split(
        "self.publish_route", 1
    )[0]
    assert '"LOCAL_SAFE_EXPLORATION"' in route_mode
    assert "LocalRouteCommand.NAVIGATION_MEMORY_GOAL" in route_mode
    assert "currently observed LiDAR-free swept footprints" in config


def test_node_capped_far_uses_process_local_background_dense_expansion():
    memory = (
        ROOT / "ros/dep_car_memory_navigation/scripts/navigation_memory_node.py"
    ).read_text(encoding="utf-8")
    runtime = (
        ROOT / "dep_car/src/dep_car/runtime/far_visibility.py"
    ).read_text(encoding="utf-8")
    config = (
        ROOT / "ros/dep_car_memory_navigation/config/navigation_memory.yaml"
    ).read_text(encoding="utf-8")

    assert "visibility_dense_replan_initial_nodes: 192" in config
    assert "visibility_dense_replan_maximum_nodes: 320" in config
    assert "visibility_dense_replan_node_step: 32" in config
    assert "visibility_dense_replan_time_budget_s: 2.5" in config
    assert "plan_progressive" in runtime
    assert "ProgressiveVisibilitySession" in runtime
    assert "PARTIAL_ATTEMPTABLE" in runtime
    assert "Only pairs touching a newly admitted vertex" in runtime
    assert "and plan.node_limit_hit" in memory
    assert "and not plan.start_inside_inflation" in memory
    assert "and not plan.goal_inside_inflation" in memory
    assert 'name="dep_car_far_cached_backend"' in memory
    assert 'name="dep_car_far_dense_%d"' not in memory
    assert memory.count("threading.Thread(") == 1
    assert "visibility_dense_replan_worker_slice_s: 0.15" in config
    assert "daemon=True" in memory
    assert "FAR_DENSE_REPLAN_PENDING" in memory
    assert "visibility_dense_replan_session" in memory
    assert "FAR_PARTIAL_ATTEMPTABLE" in memory
    assert "partial_frontier_authority" in memory
    assert "information_gain_partial_frontier" in runtime
    assert "visibility_trajectory_points_map" in memory
    assert 'trajectory_points=request["trajectory_points"]' in memory
    assert "visibility_trajectory_bridge_enabled: true" in config
    assert "visibility_trajectory_bridge_maximum_nodes: 64" in config
    assert 'plan.mode == "PARTIAL_ATTEMPTABLE"' in memory
    assert "not dense_replan_pending" in memory
    assert "LocalRouteCommand.NAVIGATION_CONNECTIVITY" in memory
    assert '"cross_episode_route_cache": False' in memory
    assert '"map_identity_input": False' in memory
    assert "planning_request_epoch" in memory
    assert "occupancy_snapshot_version" in memory


def test_ordinary_far_replan_uses_one_latest_only_cached_backend():
    memory = (
        ROOT / "ros/dep_car_memory_navigation/scripts/navigation_memory_node.py"
    ).read_text(encoding="utf-8")
    config = (
        ROOT / "ros/dep_car_memory_navigation/config/navigation_memory.yaml"
    ).read_text(encoding="utf-8")
    runtime = (
        ROOT / "dep_car/src/dep_car/runtime/far_visibility.py"
    ).read_text(encoding="utf-8")

    update_body = memory.split("def update_visibility_plan", 1)[1].split(
        "def far_recovery_prefix_authorized", 1
    )[0]
    assert "self.visibility.plan(" not in update_body
    assert 'name="dep_car_far_cached_backend"' in memory
    assert memory.count('name="dep_car_far_cached_backend"') == 1
    assert "self.visibility_query_request = proposal" in memory
    assert "self.visibility_query_latest_replacements += 1" in memory
    assert 'first["directed_failed_branches"]' in memory
    assert 'first["trajectory_points"] == second["trajectory_points"]' in memory
    assert "newest_outstanding_sequence" in memory
    assert "self.visibility_query_superseded_discards += 1" in memory
    assert "SPARSE_CACHED" in update_body
    assert "cached_backend_pending" in update_body
    assert "latest-grid route validation" in memory
    assert '"single_planner_authority": True' in memory
    assert '"cross_episode_route_cache": False' in memory
    assert '"map_identity_input": False' in memory
    assert "visibility_graph_cache_capacity: 1" in config
    assert "compile_snapshot" in runtime
    assert "plan_compiled" in runtime
    assert "never a map/scenario identifier" in runtime

    safety_check = memory.index("map_content_changed = bool(")
    maneuver_return = memory.index(
        "and not self.visibility_route_suffix_invalidated\n        ):"
    )
    assert safety_check < maneuver_return
    assert "latest_map_invalidated_route_suffix" in memory
    assert "longest_traversable_prefix" in memory


def test_far_async_handoff_cannot_starve_or_restore_an_invalidated_suffix():
    memory = (
        ROOT / "ros/dep_car_memory_navigation/scripts/navigation_memory_node.py"
    ).read_text(encoding="utf-8")

    result_body = memory.split(
        "def take_visibility_query_result", 1
    )[1].split("def start_dense_visibility_replan", 1)[0]
    assert "matching.sort(" in result_body
    assert "for document in matching:" in result_body
    assert 'plan.status == "PASS"' in result_body
    assert "newest_current_negative" in result_body
    assert "newest_outstanding_sequence <= negative_sequence" in result_body
    # A newer in-flight pose request must not directly reject a completed,
    # latest-grid-certified PASS.
    assert "newest_outstanding_sequence > int(document" not in result_body

    update_body = memory.split("def update_visibility_plan", 1)[1].split(
        "def topology_guidance", 1
    )[0]
    invalidation = update_body.index(
        '"latest_map_invalidated_route_suffix"'
    )
    cursor_reset = update_body.rfind("self.reset_visibility_cursor()", 0, invalidation)
    assert cursor_reset >= 0
    assert "self.visibility_route_suffix_invalidated = True" in update_body
    assert "and not self.visibility_route_suffix_invalidated" in update_body
    assert "or self.visibility_route_suffix_invalidated" in update_body
    assert '"latest_map_safe_prefix_lease_expired"' in update_body
    assert "visibility_goal_transaction_is_current" in update_body
    assert "expected_goal_epoch" in update_body


def test_observed_occupied_rviz_goal_is_rejected_before_local_exploration():
    memory = (
        ROOT / "ros/dep_car_memory_navigation/scripts/navigation_memory_node.py"
    ).read_text(encoding="utf-8")

    assert "def goal_endpoint_evidence" in memory
    assert '"GOAL_IN_OBSERVED_OBSTACLE"' in memory
    validation = memory.split(
        "goal_endpoint = self.goal_endpoint_evidence", 1
    )[1].split("stamp = odom.header.stamp.to_sec()", 1)[0]
    assert 'self.last_guidance_source = "INVALID_GOAL"' in validation
    assert '"INVALID_GOAL"' in validation
    assert "local_exploration_authorized=False" in validation
    assert validation.rstrip().endswith("return")


def test_m6_far_route_renewal_retains_only_a_safe_certified_suffix():
    memory = (
        ROOT / "ros/dep_car_memory_navigation/scripts/navigation_memory_node.py"
    ).read_text(encoding="utf-8")
    config = (
        ROOT / "ros/dep_car_memory_navigation/config/navigation_memory.yaml"
    ).read_text(encoding="utf-8")
    far_runtime = (
        ROOT / "dep_car/src/dep_car/runtime/far_visibility.py"
    ).read_text(encoding="utf-8")

    assert "visibility_route_renewal_period_s" in config
    assert "visibility_route_renewal_distance_m" in config
    assert "visibility_route_continuity_minimum_m" in config
    assert "visibility_route_dropout_grace_s" in config
    assert "visibility_route_lease_prefix_m" in config
    assert "visibility_no_route_static_evidence_timeout_s" in config
    assert "duplicate_map_updates_skipped" in memory
    assert "zlib.crc32" in memory
    assert "accumulated_map_observation_revision" in memory
    assert "FAR stable candidate bootstrap promotion" in memory
    assert "visibility_initial_exploration_minimum_m" in config
    assert "visibility_initial_exploration_maximum_duration_s" in config
    assert "observe_initial_local_exploration" in memory
    assert "visibility_mapping_session_established" in memory
    assert '"online_mapping_session_already_established"' in memory
    goal_reset = memory.split(
        "self.visibility_initial_exploration_distance_m = 0.0", 2
    )[2].split("self.visibility_maneuver_transaction_active", 1)[0]
    assert "self.visibility_mapping_session_established" in goal_reset
    assert "same cached grid cannot turn one speculative" in memory
    assert "unconfirmed_route_rolling_local_exploration" in memory
    assert "visibility_route_replacement_maximum_direction_change_rad" in config
    assert "route_renewal_due" in memory
    assert "attemptable_refresh_due" in memory
    assert "acquisition_refresh_due" in memory
    assert "observer_position=current_xy" in memory
    assert "visibility_route_minimum_observer_displacement_m" in config
    assert "retaining_safe_active_route_" in memory
    assert "retaining_leased_active_route_safe_local_prefix" in memory
    assert "transient_route_lease_authorized" in memory
    assert "measured_pose_revalidation_retaining_active_route" in memory
    assert "route_lease_refresh_due" in memory
    assert "transient_route_lease_refresh" in memory
    assert "replacement_direction_discontinuous" in memory
    assert "goal_connected_incumbent_retention_reason" in memory
    assert "partial_candidate_cannot_preempt_goal_route" in far_runtime
    assert "DEPCarP6V431ConnectedRouteTransactionV1" in memory
    assert "PENDING_CONNECTED" in memory
    assert "connected_known_route_transaction_promoted" in memory
    assert "visibility_connected_candidate_suppressed_weaker" in memory
    assert "partial_frontier_authority_reason" in memory
    assert "visibility_partial_minimum_goal_progress_m" in config
    assert "visibility_partial_minimum_information_gain" in config
    assert "visibility_partial_maximum_information_detour_m" in config
    assert "DISCARDED_STALE_SNAPSHOT" in memory
    assert "visibility_dense_replan_maximum_start_drift_m" in config
    assert "previous_route_safe" in memory
    assert "self.visibility.path_is_traversable(" in memory
    assert "visibility_active_route_accepted" in memory
    assert "visibility_active_route_motion_authorized" in memory
    assert "not acquisition.motion_authorized" in memory
    assert "preserve_acquisition=preserve_acquisition" in memory
    assert "A local recovery manoeuvre may start while FAR has no" in memory
    assert "route_unavailable_after_recent_static_block" in memory
    assert "hard_route_hold" in memory
    assert "absence of a graph route is not physical" in memory


def test_m6_replan_wait_preserves_local_route_transaction_barrier():
    local = (
        ROOT / "ros/dep_car_local_planner/scripts/local_planner_node.py"
    ).read_text(encoding="utf-8")
    assert "if not self.forward_capture_replan_requested:" in local
    assert "approve_revalidated_route" in local
    assert "abrupt_rear_route_change" in local


def test_m6_locks_upstream_reference_but_keeps_native_dep_car_adapter():
    fetch = (ROOT / "scripts/fetch_far_planner_upstream.sh").read_text(
        encoding="utf-8"
    )
    lock = (ROOT / "third_party.lock.yaml").read_text(encoding="utf-8")
    assert "https://github.com/MichaelFYang/far_planner.git" in fetch
    assert "2799b6964c141cacd1c32a14b19bc7abffbe0e52" in fetch
    assert "far_planner:" in lock
    assert "independent OccupancyGrid adapter" in lock


def test_interrupted_episode_cannot_leave_stale_acceptance_report(tmp_path):
    report = tmp_path / "episode.json"
    report.write_text('{"status":"PASS"}\n', encoding="utf-8")

    archived = runner.archive_previous_report(report)

    assert archived is not None
    assert not report.exists()
    assert archived.is_file()
    assert archived.parent.name == "history"
