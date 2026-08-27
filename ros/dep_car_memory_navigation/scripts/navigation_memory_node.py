#!/usr/bin/env python3
"""Online-SLAM visibility routing with DE-P execution and memory recovery."""

import json
import math
import threading
import time
import zlib
from dataclasses import replace

import numpy as np
import rospy
import tf2_ros
from dep_car.core.types import Gear
from dep_car.core.occupancy import FootprintConfig
from dep_car.runtime.far_visibility import (
    DynamicVisibilityPlanner,
    VisibilityRouteAcquisitionGate,
    goal_route_direction_continuity_hold,
    goal_connected_incumbent_retention_reason,
    locally_certified_route_motion,
    measured_pose_revalidation_authorized,
    partial_frontier_authority_reason,
    polyline_prefix,
    transient_route_lease_authorized,
    visibility_plan_is_goal_connected,
)
from dep_car.runtime.navigation_memory import (
    ackermann_arc_trajectory,
    BoundaryFollowSupervisor,
    BreadcrumbTrail,
    DeadEndRecoverySupervisor,
    MemoryNavigationState,
    MonotonicRouteProgress,
    ReactiveHeadingDecision,
    TopologicalMemory,
    TopologyEdgeState,
    local_space_features,
    ray_clearance,
    resample_polyline,
    select_reactive_heading,
    select_dead_end_egress_site,
    wrap_angle,
)
from dep_car.runtime.occupancy import RuntimeOccupancyGrid2D
from dep_car_msgs.msg import (
    AckermannRoute,
    CandidateArray,
    LocalRouteCommand,
    PlannerState,
    RoutePoint,
)
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Point
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray


def yaw_from_quaternion(quaternion):
    return math.atan2(
        2.0 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y),
        1.0 - 2.0 * (quaternion.y * quaternion.y + quaternion.z * quaternion.z),
    )


class NavigationMemoryNode:
    def __init__(self):
        self.lock = threading.Lock()
        self.odom = None
        self.grid = None
        self.accumulated_grid = None
        self.accumulated_occupancy = None
        self.accumulated_map_revision = 0
        # Content revisions drive expensive graph rebuilds and route evidence.
        # Observation revisions remain diagnostic only: an identical grid is
        # not allowed to confirm its own speculative FAR side choice.
        self.accumulated_map_observation_revision = 0
        self.accumulated_known_cells = 0
        self.accumulated_map_signature = None
        self.accumulated_map_duplicate_updates = 0
        # These are process-local concurrency tokens, not map identities.  They
        # are reset on node startup and are never used to retrieve a route from
        # another map, scenario or episode.
        self.visibility_planning_request_epoch = 0
        self.visibility_dense_replan_sequence = 0
        self.visibility_dense_replan_pending = False
        self.visibility_dense_replan_result = None
        self.visibility_dense_replan_snapshot_version = -1
        self.visibility_dense_replan_request_epoch = -1
        self.visibility_dense_replan_goal_key = None
        self.visibility_dense_replan_started_stamp = -math.inf
        self.visibility_dense_replan_last_duration_ms = None
        self.visibility_dense_replan_last_error = None
        self.visibility_dense_replan_last_status = "IDLE"
        self.visibility_dense_replan_last_attempt_snapshot = -1
        self.visibility_dense_replan_last_completed_stamp = -math.inf
        self.visibility_dense_replan_session = None
        self.visibility_dense_replan_session_snapshot_version = -1
        self.visibility_dense_replan_session_request_epoch = -1
        self.visibility_dense_replan_session_goal_key = None
        self.visibility_dense_replan_session_start_key = None
        self.goal = None
        self.local_state = None
        self.local_candidates = None
        self.last_progress_pose = None
        self.last_progress_stamp = None
        self.backtrack_start_index = None
        self.backtrack_cursor_index = None
        self.backtrack_site_index = None
        self.backtrack_start_xy = None
        self.resume_start_xy = None
        self.resume_target_index = None
        self.resume_fallback_target_xy = None
        self.resume_started_stamp = None
        self.resume_travelled_m = 0.0
        self.resume_last_xy = None
        self.resume_blocked_since = None
        self.backtrack_started_stamp = None
        self.backtrack_blocked_since = None
        self.generation = 0
        self.last_status_signature = None
        # Stored in map coordinates.  A body-frame angle from the previous
        # control tick is not a valid continuity reference after the car yaws.
        self.previous_goal_heading = None
        self.last_heading_decision = None
        self.last_boundary_decision = None
        self.last_mission_goal_bearing = None
        self.last_guidance_source = "DIRECT_GOAL"
        self.last_topology_path = []
        self.committed_topology_path = []
        self.committed_topology_failed_branches = 0
        self.visibility_plan = None
        self.visibility_path_map = []
        self.visibility_remaining_path_map = []
        self.visibility_route_revision = 0
        self.visibility_route_cursor = None
        self.visibility_last_handoff = None
        self.visibility_last_carrot_map = None
        self.topology_route_revision = 0
        self.topology_route_cursor = None
        self.topology_last_carrot_odom = None
        # A rear topology corridor may start one multi-leg transaction.  It
        # cannot repeatedly reacquire gearbox authority for the same route
        # after that transaction has already restored forward motion.
        self.topology_rear_authority_issued = False
        self.topology_last_turnaround_route_id = ""
        self.topology_turnaround_completion_progress_m = None
        self.topology_turnaround_completion_xy = None
        self.route_turnaround_transaction_active = False
        self.route_turnaround_transaction_id = 0
        self.route_turnaround_transaction_sequence = 0
        self.route_turnaround_source = "NONE"
        self.route_turnaround_trigger = "NONE"
        self.route_turnaround_outcome = "NONE"
        self.route_turnaround_start_trail_index = None
        # ``visibility_plan`` is the route currently handed to DE-P.  FAR may
        # compute a newer candidate before it is stable enough to replace that
        # route, so candidate and active-route authority must not share one
        # mutable slot.
        self.visibility_active_route_accepted = False
        self.visibility_active_route_motion_authorized = False
        self.visibility_last_candidate_plan = None
        self.visibility_last_candidate_retained_active = False
        # P6 V4.3.1 process-local connected-route transaction.  A dense solve
        # may discover a complete maze detour between two ordinary sparse
        # planning ticks.  Keep that current-goal candidate until it is either
        # promoted or genuinely invalidated; a weaker PARTIAL/NO_ROUTE result
        # is not allowed to erase it.  Nothing here is serialized or keyed by
        # map/scenario identity.
        self.visibility_connected_candidate_plan = None
        self.visibility_connected_candidate_goal_key = None
        self.visibility_connected_candidate_revision = -1
        self.visibility_connected_candidate_discovered_stamp = -math.inf
        self.visibility_connected_candidate_status = "EMPTY"
        self.visibility_connected_candidate_reason = "none_discovered"
        self.visibility_connected_candidate_route_id = ""
        self.visibility_connected_candidate_suppressed_weaker = 0
        self.visibility_connected_candidate_promotions = 0
        self.visibility_route_dropout_started_stamp = None
        self.visibility_route_lease_active = False
        self.visibility_route_lease_reason = "inactive"
        self.visibility_route_lease_prefix_length_m = 0.0
        # Keep only a short-lived *direction* from the latest authorized FAR
        # corridor.  Unlike an odom-frame topology polyline, this map-frame
        # unit vector cannot accumulate into a second, stale driving route.
        # It bridges a transient FAR solve dropout while live DE-P collision
        # checks remain authoritative.
        self.visibility_last_authorized_direction_map = None
        self.visibility_last_authorized_direction_stamp = -math.inf
        self.visibility_goal_key = None
        self.visibility_planned_revision = -1
        self.visibility_planned_failed_branches = -1
        self.visibility_last_plan_stamp = -math.inf
        self.visibility_last_plan_duration_ms = None
        self.visibility_replan_count = 0
        self.visibility_last_replan_reason = "initial"
        self.visibility_static_blocked_since = None
        self.visibility_last_local_static_block_stamp = -math.inf
        self.visibility_no_route_since = None
        self.visibility_last_blocked_replan_stamp = -math.inf
        self.visibility_static_replan_failures = 0
        # A locally explored direction which is rejected by the swept-
        # footprint hard veto may hand authority to one *observed prefix* of
        # the next FAR candidate.  This is deliberately a one-shot recovery
        # bridge: if that FAR route is itself hard-blocked it cannot promote
        # itself again and normal certified egress takes over.
        self.visibility_static_recovery_pending = False
        self.visibility_static_recovery_handoffs = 0
        self.far_emergency_egress_active = False
        self.dead_end_escape_sequence = 0
        self.dead_end_escape_id = 0
        self.dead_end_escape_branch_id = None
        self.dead_end_escape_route_revision = 0
        self.dead_end_escape_site_kind = "NONE"
        self.dead_end_escape_target_distance_m = 0.0
        self.dead_end_escape_cross_track_error_m = 0.0
        self.dead_end_escape_last_route_map = []
        self.dead_end_escape_started_map_correction_generation = 0
        self.dead_end_escape_completion_reason = "NONE"
        self.dead_end_escape_diverged_since = None
        self.dead_end_escape_connector_unavailable_since = None
        self.dead_end_escape_live_reanchors = 0
        self.dead_end_escape_live_target_index = None
        self.failed_branch_exit_lock_branch_id = None
        self.failed_branch_exit_lock_origin_odom = None
        self.failed_branch_exit_lock_started_stamp = None
        self.failed_branch_exit_lock_progress_m = 0.0
        # A FAR route and the local multi-gear Ackermann manoeuvre form one
        # transaction.  Once a turnaround starts, keep its route suffix fixed
        # until the local planner reports that every committed leg is done.
        self.visibility_maneuver_transaction_active = False
        self.visibility_maneuver_path_map = []
        self.visibility_deferred_replan_reasons = []
        self.visibility_course_revalidation_pending = False
        self.visibility_route_validation_revision = -1
        self.visibility_route_validation_stamp = -math.inf
        self.visibility_route_validation_passed = False
        self.map_correction_generation = 0
        self.applied_map_correction_generation = 0
        self.last_map_correction = None
        self.last_significant_map_correction = None
        self.last_map_correction_replan_stamp = -math.inf
        self.last_map_pose = None
        self.last_odom_pose = None
        self.best_goal_distance = None
        self.last_goal_improvement_stamp = None
        self.goal_stall_confirmation = float(
            rospy.get_param("~goal_stall_confirmation_s", 3.0)
        )

        self.visibility = DynamicVisibilityPlanner(
            inflation_radius_m=float(
                rospy.get_param("~visibility_inflation_radius_m", 0.38)
            ),
            contour_simplification_m=float(
                rospy.get_param("~visibility_contour_simplification_m", 0.22)
            ),
            vertex_offset_m=float(
                rospy.get_param("~visibility_vertex_offset_m", 0.18)
            ),
            node_separation_m=float(
                rospy.get_param("~visibility_node_separation_m", 0.28)
            ),
            maximum_nodes=int(
                rospy.get_param("~visibility_maximum_nodes", 160)
            ),
            maximum_edge_length_m=float(
                rospy.get_param("~visibility_maximum_edge_length_m", 12.0)
            ),
            unknown_cost_weight=float(
                rospy.get_param("~visibility_unknown_cost_weight", 1.25)
            ),
            start_heading_weight_m=float(
                rospy.get_param("~visibility_start_heading_weight_m", 1.20)
            ),
            reverse_start_penalty_m=float(
                rospy.get_param("~visibility_reverse_start_penalty_m", 2.00)
            ),
        )
        self.visibility_replan_period = float(
            rospy.get_param("~visibility_replan_period_s", 0.60)
        )
        self.visibility_trajectory_bridge_enabled = bool(
            rospy.get_param("~visibility_trajectory_bridge_enabled", True)
        )
        self.visibility_trajectory_bridge_maximum_nodes = int(
            rospy.get_param(
                "~visibility_trajectory_bridge_maximum_nodes", 64
            )
        )
        if self.visibility_trajectory_bridge_maximum_nodes < 0:
            raise ValueError(
                "visibility trajectory bridge node limit cannot be negative"
            )
        self.visibility_dense_replan_enabled = bool(
            rospy.get_param("~visibility_dense_replan_enabled", True)
        )
        self.visibility_dense_replan_initial_nodes = int(
            rospy.get_param("~visibility_dense_replan_initial_nodes", 192)
        )
        self.visibility_dense_replan_maximum_nodes = int(
            rospy.get_param("~visibility_dense_replan_maximum_nodes", 320)
        )
        self.visibility_dense_replan_node_step = int(
            rospy.get_param("~visibility_dense_replan_node_step", 32)
        )
        self.visibility_dense_replan_time_budget = float(
            rospy.get_param("~visibility_dense_replan_time_budget_s", 2.5)
        )
        self.visibility_dense_replan_retry_period = float(
            rospy.get_param("~visibility_dense_replan_retry_period_s", 1.5)
        )
        self.visibility_dense_replan_maximum_start_drift = float(
            rospy.get_param(
                "~visibility_dense_replan_maximum_start_drift_m", 0.45
            )
        )
        if self.visibility_dense_replan_maximum_start_drift <= 0.0:
            raise ValueError("dense FAR maximum start drift must be positive")
        if not (
            self.visibility.maximum_nodes
            < self.visibility_dense_replan_initial_nodes
            <= self.visibility_dense_replan_maximum_nodes
        ):
            raise ValueError(
                "dense FAR node limits must exceed the ordinary graph limit"
            )
        self.visibility_route_renewal_period = float(
            rospy.get_param("~visibility_route_renewal_period_s", 0.35)
        )
        self.visibility_lookahead = float(
            rospy.get_param("~visibility_lookahead_m", 1.50)
        )
        self.rolling_route_minimum_lookahead = float(
            rospy.get_param("~rolling_route_minimum_lookahead_m", 1.00)
        )
        self.rolling_route_maximum_lookahead = float(
            rospy.get_param("~rolling_route_maximum_lookahead_m", 2.50)
        )
        self.rolling_route_curvature_window = float(
            rospy.get_param("~rolling_route_curvature_window_m", 1.50)
        )
        self.rolling_route_maximum_projection_advance = float(
            rospy.get_param(
                "~rolling_route_maximum_projection_advance_m", 1.25
            )
        )
        self.rolling_route_projection_rollback_tolerance = float(
            rospy.get_param(
                "~rolling_route_projection_rollback_tolerance_m", 0.08
            )
        )
        self.rolling_route_carrot_capture_radius = float(
            rospy.get_param("~rolling_route_carrot_capture_radius_m", 0.40)
        )
        self.rolling_route_maximum_carrot_advance = float(
            rospy.get_param("~rolling_route_maximum_carrot_advance_m", 0.70)
        )
        self.rolling_route_maximum_carrot_distance = float(
            rospy.get_param("~rolling_route_maximum_carrot_distance_m", 1.50)
        )
        self.visibility_route_renewal_distance = float(
            rospy.get_param("~visibility_route_renewal_distance_m", 3.00)
        )
        self.visibility_route_continuity_minimum = float(
            rospy.get_param("~visibility_route_continuity_minimum_m", 0.70)
        )
        self.visibility_route_dropout_grace = float(
            rospy.get_param("~visibility_route_dropout_grace_s", 4.00)
        )
        # Sparse topology is semantic memory (visited/failed branches), not
        # metric control authority.  Its odom history can drift over a long
        # mission, so normal navigation must not turn it back into a precise
        # trajectory unless a legacy experiment explicitly opts in.
        self.explored_topology_motion_authority = bool(
            rospy.get_param("~explored_topology_motion_authority", False)
        )
        self.visibility_route_lease_prefix = float(
            rospy.get_param("~visibility_route_lease_prefix_m", 2.00)
        )
        self.visibility_no_route_static_evidence_timeout = float(
            rospy.get_param(
                "~visibility_no_route_static_evidence_timeout_s", 8.0
            )
        )
        self.visibility_maximum_deviation = float(
            rospy.get_param("~visibility_maximum_deviation_m", 0.90)
        )
        self.visibility_partial_minimum_goal_progress = float(
            rospy.get_param(
                "~visibility_partial_minimum_goal_progress_m", 0.05
            )
        )
        self.visibility_partial_minimum_information_gain = float(
            rospy.get_param(
                "~visibility_partial_minimum_information_gain", 0.05
            )
        )
        self.visibility_partial_maximum_information_detour = float(
            rospy.get_param(
                "~visibility_partial_maximum_information_detour_m", 0.50
            )
        )
        if min(
            self.visibility_partial_minimum_goal_progress,
            self.visibility_partial_minimum_information_gain,
            self.visibility_partial_maximum_information_detour,
        ) < 0.0:
            raise ValueError("partial FAR authority thresholds cannot be negative")
        self.visibility_route_direction_authority = bool(
            rospy.get_param("~visibility_route_direction_authority", True)
        )
        self.visibility_route_acquisition = VisibilityRouteAcquisitionGate(
            minimum_confirmations=int(
                rospy.get_param("~visibility_route_minimum_confirmations", 2)
            ),
            minimum_stable_s=float(
                rospy.get_param("~visibility_route_minimum_stable_s", 0.60)
            ),
            maximum_bearing_change_rad=float(
                rospy.get_param(
                    "~visibility_route_maximum_bearing_change_rad",
                    math.radians(30.0),
                )
            ),
            maximum_relative_cost_change=float(
                rospy.get_param(
                    "~visibility_route_maximum_relative_cost_change", 0.35
                )
            ),
            maximum_attemptable_detour_ratio=float(
                rospy.get_param(
                    "~visibility_route_maximum_attemptable_detour_ratio", 2.25
                )
            ),
            maximum_high_detour_unknown_fraction=float(
                rospy.get_param(
                    "~visibility_route_maximum_high_detour_unknown_fraction",
                    0.20,
                )
            ),
            high_detour_extra_confirmations=int(
                rospy.get_param(
                    "~visibility_route_high_detour_extra_confirmations", 2
                )
            ),
            high_detour_minimum_stable_s=float(
                rospy.get_param(
                    "~visibility_route_high_detour_minimum_stable_s", 1.20
                )
            ),
            lookahead_m=self.visibility_lookahead,
            minimum_observer_displacement_m=float(
                rospy.get_param(
                    "~visibility_route_minimum_observer_displacement_m",
                    0.25,
                )
            ),
        )
        self.visibility_route_replacement_maximum_direction_change = float(
            rospy.get_param(
                "~visibility_route_replacement_maximum_direction_change_rad",
                math.radians(60.0),
            )
        )
        self.visibility_terminal_direct_handoff_radius = float(
            rospy.get_param(
                "~visibility_terminal_direct_handoff_radius_m", 1.80
            )
        )
        self.visibility_route_acquisition_started_stamp = None
        self.visibility_route_acquisition_reason = "waiting_for_visibility_route"
        # Only a fresh online-mapping session begins with a short, continuously
        # safe local exploration phase.  Once wheel motion has established the
        # SLAM/topology session, later RViz goals must not repeat this blind
        # one-metre drive while an already stable FAR route is available.
        self.visibility_mapping_session_established = False
        self.visibility_initial_exploration_distance_m = 0.0
        self.visibility_initial_exploration_last_xy = None
        self.visibility_initial_exploration_started_stamp = None
        self.visibility_initial_exploration_complete = False
        self.visibility_initial_exploration_reason = "waiting_for_goal_motion"
        self.visibility_initial_exploration_minimum = float(
            rospy.get_param("~visibility_initial_exploration_minimum_m", 1.00)
        )
        self.visibility_initial_exploration_maximum_duration = float(
            rospy.get_param(
                "~visibility_initial_exploration_maximum_duration_s", 6.0
            )
        )
        # FAR/visibility routing owns ordinary route recovery in M6.
        # Breadcrumbs cannot select a gear unless a legacy experiment opts in
        # or the separately bounded, repeated-static-block egress fires.
        self.breadcrumb_motion_authority = bool(
            rospy.get_param("~breadcrumb_motion_authority", False)
        )
        # Breadcrumb motion remains forbidden during ordinary FAR navigation.
        # It is re-enabled only as a short, certified egress after repeated
        # local hard blocks and failed FAR replans.
        self.far_static_egress_enabled = bool(
            rospy.get_param("~far_static_egress_enabled", True)
        )
        self.far_static_replans_before_egress = int(
            rospy.get_param("~far_static_replans_before_egress", 2)
        )
        self.far_static_egress_distance = float(
            rospy.get_param("~far_static_egress_distance_m", 0.80)
        )
        self.far_static_egress_maximum = float(
            rospy.get_param("~far_static_egress_maximum_m", 1.50)
        )
        self.far_static_egress_block_confirmation = float(
            rospy.get_param(
                "~far_static_egress_block_confirmation_s", 1.0
            )
        )
        self.dead_end_escape_minimum_distance = float(
            rospy.get_param("~dead_end_escape_minimum_distance_m", 1.50)
        )
        self.dead_end_escape_lookahead = float(
            rospy.get_param("~dead_end_escape_lookahead_m", 0.90)
        )
        self.dead_end_escape_maximum_cross_track = float(
            rospy.get_param("~dead_end_escape_maximum_cross_track_m", 0.65)
        )
        self.dead_end_escape_divergence_confirmation = float(
            rospy.get_param(
                "~dead_end_escape_divergence_confirmation_s", 1.0
            )
        )
        self.dead_end_escape_maximum_unknown_fraction = float(
            rospy.get_param(
                "~dead_end_escape_maximum_unknown_fraction", 0.08
            )
        )
        self.dead_end_escape_live_target_capture = float(
            rospy.get_param(
                "~dead_end_escape_live_target_capture_m", 0.30
            )
        )
        self.failed_branch_exit_lock_progress = float(
            rospy.get_param("~failed_branch_exit_lock_progress_m", 0.75)
        )
        self.map_correction_replan_translation = float(
            rospy.get_param("~map_correction_replan_translation_m", 0.08)
        )
        self.map_correction_replan_yaw = float(
            rospy.get_param("~map_correction_replan_yaw_rad", 0.04)
        )
        self.map_correction_replan_minimum_period = float(
            rospy.get_param("~map_correction_replan_minimum_period_s", 1.0)
        )

        self.trail = BreadcrumbTrail(
            spacing_m=float(rospy.get_param("~breadcrumb_spacing_m", 0.20)),
            heading_spacing_rad=float(
                rospy.get_param("~breadcrumb_heading_spacing_rad", math.radians(10.0))
            ),
            maximum_length_m=float(
                rospy.get_param("~breadcrumb_maximum_length_m", 25.0)
            ),
        )
        self.topology = TopologicalMemory(
            node_spacing_m=float(rospy.get_param("~topology_node_spacing_m", 0.75)),
            merge_radius_m=float(rospy.get_param("~topology_merge_radius_m", 0.40)),
            failure_buffer_m=float(rospy.get_param("~failed_branch_buffer_m", 0.55)),
            goal_branch_allowance_m=float(
                rospy.get_param("~failed_branch_goal_allowance_m", 0.90)
            ),
            maximum_nodes=int(rospy.get_param("~topology_maximum_nodes", 1200)),
        )
        self.recovery = DeadEndRecoverySupervisor(
            confirmation_s=float(rospy.get_param("~dead_end_confirmation_s", 1.8)),
            minimum_trail_points=int(
                rospy.get_param("~dead_end_minimum_trail_points", 4)
            ),
            backtrack_enabled=self.breadcrumb_motion_authority,
        )
        self.boundary = BoundaryFollowSupervisor(
            enter_clearance_m=float(
                rospy.get_param("~boundary_enter_clearance_m", 1.10)
            ),
            release_clearance_m=float(
                rospy.get_param("~boundary_release_clearance_m", 1.45)
            ),
            leave_progress_m=float(
                rospy.get_param("~boundary_leave_progress_m", 0.45)
            ),
            leave_confirmation_s=float(
                rospy.get_param("~boundary_leave_confirmation_s", 0.80)
            ),
            loop_radius_m=float(
                rospy.get_param("~boundary_loop_radius_m", 0.70)
            ),
            minimum_loop_travel_m=float(
                rospy.get_param("~boundary_minimum_loop_travel_m", 4.0)
            ),
            maximum_boundary_travel_m=float(
                rospy.get_param("~boundary_maximum_travel_m", 14.0)
            ),
            failure_radius_m=float(
                rospy.get_param("~boundary_failure_radius_m", 1.00)
            ),
            maximum_side_failures=int(
                rospy.get_param("~boundary_maximum_side_failures", 32)
            ),
        )
        self.rear_clearance = float(rospy.get_param("~rear_clearance_m", 0.70))
        self.backtrack_chunk = float(rospy.get_param("~backtrack_chunk_m", 2.50))
        self.backtrack_maximum = float(rospy.get_param("~backtrack_maximum_m", 8.0))
        self.minimum_backtrack_before_resume = float(
            rospy.get_param("~minimum_backtrack_before_resume_m", 1.50)
        )
        self.resume_failure_confirmation = float(
            rospy.get_param("~resume_failure_confirmation_s", 1.50)
        )
        self.resume_distance = float(
            rospy.get_param("~resume_forward_distance_m", 1.00)
        )
        self.resume_maximum_travel = float(
            rospy.get_param("~resume_maximum_travel_m", 3.00)
        )
        self.resume_timeout = float(rospy.get_param("~resume_timeout_s", 25.0))
        self.route_horizon = float(rospy.get_param("~local_route_horizon_m", 2.50))
        self.route_minimum = float(rospy.get_param("~local_route_minimum_m", 0.45))
        self.standoff = float(rospy.get_param("~obstacle_standoff_m", 0.35))
        self.route_points = int(rospy.get_param("~route_point_count", 40))
        self.goal_tolerance = float(
            rospy.get_param("~goal_position_tolerance_m", 0.25)
        )
        self.goal_settled_speed = float(
            rospy.get_param("~goal_settled_speed_mps", 0.04)
        )
        self.dead_end_goal_exclusion = float(
            rospy.get_param("~dead_end_goal_exclusion_radius_m", 0.80)
        )
        self.topology_path_capture_radius = float(
            rospy.get_param("~topology_path_capture_radius_m", 0.50)
        )
        self.topology_path_maximum_deviation = float(
            rospy.get_param("~topology_path_maximum_deviation_m", 1.50)
        )
        self.topology_rear_authority_bearing = float(
            rospy.get_param(
                "~topology_rear_authority_bearing_rad",
                math.radians(110.0),
            )
        )
        self.topology_turnaround_rearm_progress = float(
            rospy.get_param(
                "~topology_turnaround_rearm_progress_m", 1.50
            )
        )
        if min(
            self.rear_clearance,
            self.backtrack_chunk,
            self.backtrack_maximum,
            self.minimum_backtrack_before_resume,
            self.resume_failure_confirmation,
            self.resume_distance,
            self.resume_maximum_travel,
            self.resume_timeout,
            self.route_horizon,
            self.route_minimum,
            self.dead_end_goal_exclusion,
            self.topology_path_capture_radius,
            self.topology_path_maximum_deviation,
            self.visibility_replan_period,
            self.visibility_route_renewal_period,
            self.visibility_lookahead,
            self.rolling_route_minimum_lookahead,
            self.rolling_route_maximum_lookahead,
            self.rolling_route_curvature_window,
            self.rolling_route_maximum_projection_advance,
            self.rolling_route_projection_rollback_tolerance,
            self.visibility_route_renewal_distance,
            self.visibility_route_continuity_minimum,
            self.visibility_route_dropout_grace,
            self.visibility_route_lease_prefix,
            self.visibility_no_route_static_evidence_timeout,
            self.visibility_maximum_deviation,
            self.visibility_initial_exploration_minimum,
            self.visibility_initial_exploration_maximum_duration,
            self.visibility_route_replacement_maximum_direction_change,
            self.visibility_terminal_direct_handoff_radius,
            self.far_static_egress_distance,
            self.far_static_egress_maximum,
            self.far_static_egress_block_confirmation,
            self.dead_end_escape_minimum_distance,
            self.dead_end_escape_lookahead,
            self.dead_end_escape_maximum_cross_track,
            self.dead_end_escape_divergence_confirmation,
            self.failed_branch_exit_lock_progress,
            self.map_correction_replan_translation,
            self.map_correction_replan_yaw,
            self.map_correction_replan_minimum_period,
            self.topology_turnaround_rearm_progress,
        ) <= 0.0:
            raise ValueError("memory navigation distances must be positive")
        if not 0.0 <= self.dead_end_escape_maximum_unknown_fraction <= 1.0:
            raise ValueError("dead-end egress unknown fraction must be in [0, 1]")
        if (
            self.rolling_route_minimum_lookahead
            > self.rolling_route_maximum_lookahead
        ):
            raise ValueError("rolling route lookahead limits are invalid")
        if not math.pi / 2.0 < self.topology_rear_authority_bearing < math.pi:
            raise ValueError("topology rear bearing must be between 90 and 180 degrees")
        if self.far_static_replans_before_egress < 1:
            raise ValueError("FAR static egress needs at least one failed replan")

        route_cursor_options = {
            "minimum_lookahead_m": self.rolling_route_minimum_lookahead,
            "maximum_lookahead_m": self.rolling_route_maximum_lookahead,
            "curvature_window_m": self.rolling_route_curvature_window,
            "maximum_projection_advance_m": (
                self.rolling_route_maximum_projection_advance
            ),
            "projection_rollback_tolerance_m": (
                self.rolling_route_projection_rollback_tolerance
            ),
            "carrot_capture_radius_m": (
                self.rolling_route_carrot_capture_radius
            ),
            "maximum_carrot_advance_m": (
                self.rolling_route_maximum_carrot_advance
            ),
            "maximum_carrot_distance_m": (
                self.rolling_route_maximum_carrot_distance
            ),
        }
        self.visibility_route_cursor = MonotonicRouteProgress(
            **route_cursor_options
        )
        self.topology_route_cursor = MonotonicRouteProgress(
            **route_cursor_options
        )

        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(30.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.path_pub = rospy.Publisher("/dep_car/global_path", Path, queue_size=1, latch=True)
        self.route_pub = rospy.Publisher(
            "/dep_car/global_route", AckermannRoute, queue_size=1, latch=True
        )
        self.subgoal_pub = rospy.Publisher(
            "/dep_car/local_subgoal", PoseStamped, queue_size=1, latch=True
        )
        self.command_pub = rospy.Publisher(
            "/dep_car/local_route_command", LocalRouteCommand, queue_size=1, latch=True
        )
        self.status_pub = rospy.Publisher(
            "/dep_car/global_planner_status", String, queue_size=1, latch=True
        )
        self.marker_pub = rospy.Publisher(
            "/dep_car/global_planner_status_marker", Marker, queue_size=1, latch=True
        )
        self.memory_marker_pub = rospy.Publisher(
            "/dep_car/navigation_memory_markers", MarkerArray, queue_size=1, latch=True
        )
        self.visibility_path_pub = rospy.Publisher(
            "/dep_car/visibility_path", Path, queue_size=1, latch=True
        )
        self.visibility_marker_pub = rospy.Publisher(
            "/dep_car/visibility_graph_markers", MarkerArray, queue_size=1, latch=True
        )
        rospy.Subscriber(
            rospy.get_param("~odometry_topic", "/odometry/filtered"),
            Odometry,
            self.on_odom,
            queue_size=10,
        )
        rospy.Subscriber(
            rospy.get_param("~local_costmap_topic", "/dep_car/local_costmap"),
            OccupancyGrid,
            self.on_grid,
            queue_size=1,
        )
        rospy.Subscriber(
            rospy.get_param("~accumulated_map_topic", "/map"),
            OccupancyGrid,
            self.on_accumulated_map,
            queue_size=1,
        )
        rospy.Subscriber(
            rospy.get_param("~goal_topic", "/move_base_simple/goal"),
            PoseStamped,
            self.on_goal,
            queue_size=1,
        )
        rospy.Subscriber(
            "/dep_car/planner_state", PlannerState, self.on_local_state, queue_size=5
        )
        rospy.Subscriber(
            "/dep_car/candidates", CandidateArray, self.on_candidates, queue_size=5
        )
        rospy.Subscriber(
            "/dep_car/map_odom_correction",
            String,
            self.on_map_odom_correction,
            queue_size=10,
        )
        rospy.Subscriber(
            "/dep_car/replan_request",
            String,
            self.on_replan_request,
            queue_size=5,
        )
        rate = float(rospy.get_param("~control_rate", 5.0))
        self.timer = rospy.Timer(rospy.Duration(1.0 / rate), self.update)
        self.publish_status("MAPPING_WAIT", "waiting for wheel/IMU odometry, scan and SLAM TF")

    def on_odom(self, message):
        with self.lock:
            self.odom = message

    def on_replan_request(self, message):
        """Re-anchor FAR after local course capture rejects its corridor."""

        reason = str(message.data).strip() or "local_measured_pose_request"
        if reason == "forward_course_revalidation_fallback":
            with self.lock:
                self.visibility_course_revalidation_pending = False
            rospy.logwarn(
                "Released FAR course-revalidation handshake to bounded "
                "topology/local fallback"
            )
            return
        if reason not in (
            "forward_course_capture_diverged",
            "abrupt_rear_route_change",
            "forward_restoration_budget_exhausted",
        ):
            return
        # Course capture reports geometric deviation from the local tube, not
        # evidence that the already selected topological side became invalid.
        # Preserve its acquisition history so a measured-pose rebuild can be
        # handed off atomically on the first consistent result.  A genuinely
        # abrupt rear-route change still starts a fresh acquisition.
        preserve_acquisition = reason in (
            "forward_course_capture_diverged",
            "forward_restoration_budget_exhausted",
        )
        with self.lock:
            # This request can only be emitted between committed local legs;
            # the active-leg branch returns before reaching the revalidation
            # state.  Release the frozen suffix here so the request cannot be
            # deferred behind the very transaction which is waiting for its
            # answer (the forward_course_capture_revalidation_hold deadlock).
            if self.visibility_maneuver_transaction_active:
                self.visibility_maneuver_transaction_active = False
                self.visibility_maneuver_path_map = []
                self.visibility_deferred_replan_reasons = []
            # Keep an explicit handshake bit in addition to the planner-state
            # detail string.  The replan-request callback can run before the
            # next PlannerState callback, and without this bit a timer tick
            # could re-latch the just-released manoeuvre transaction.
            self.visibility_course_revalidation_pending = True
            retain_measured_pose_anchor = bool(
                preserve_acquisition
                and self.visibility_plan is not None
                and self.visibility_plan.status == "PASS"
                and self.visibility_active_route_motion_authorized
                and len(self.visibility_path_map) >= 2
            )
            if retain_measured_pose_anchor:
                # Revalidation asks FAR for a route rebuilt from the measured
                # pose; it is not proof that the previously accepted corridor
                # vanished.  Keep that corridor and its rolling target visible
                # until update_visibility_plan either installs the replacement,
                # certifies a short local lease, or observes a genuinely
                # occupied prefix.  Making the active slot empty here caused a
                # one-frame NO_ROUTE result to hand control to unrelated local
                # exploration in the middle of a dead-end turnaround.
                self.visibility_last_plan_stamp = -math.inf
                self.visibility_route_validation_revision = -1
                self.visibility_route_validation_stamp = -math.inf
                self.visibility_route_validation_passed = False
                self.visibility_last_replan_reason = reason
                self.visibility_route_acquisition_reason = (
                    "measured_pose_revalidation_retaining_active_route"
                )
                accepted = True
            else:
                accepted = self.invalidate_visibility_route(
                    reason,
                    force=True,
                    preserve_acquisition=preserve_acquisition,
                )
            if accepted and not preserve_acquisition:
                self.visibility_route_acquisition_started_stamp = None
        if accepted:
            rospy.loginfo(
                "Accepted local measured-pose FAR re-anchor reason=%s "
                "retained_active=%s",
                reason,
                retain_measured_pose_anchor,
            )

    def on_grid(self, message):
        values = np.asarray(message.data, dtype=np.int16).reshape(
            message.info.height, message.info.width
        )
        with self.lock:
            self.grid = (
                values,
                float(message.info.resolution),
                (message.info.origin.position.x, message.info.origin.position.y),
            )

    def on_accumulated_map(self, message):
        values = np.asarray(message.data, dtype=np.int16).reshape(
            message.info.height, message.info.width
        )
        resolution = float(message.info.resolution)
        origin = (message.info.origin.position.x, message.info.origin.position.y)
        # Slam Toolbox republishes the map even when its contents did not
        # change.  Treating every publication as a graph revision caused a
        # complete O(N^2) visibility rebuild while stationary.  CRC32 is only
        # a change detector; route safety continues to use the full grid.
        signature = (
            tuple(values.shape),
            round(resolution, 9),
            round(float(origin[0]), 6),
            round(float(origin[1]), 6),
            int(zlib.crc32(values.tobytes(order="C")) & 0xFFFFFFFF),
        )
        with self.lock:
            self.accumulated_map_observation_revision += 1
            if signature == self.accumulated_map_signature:
                self.accumulated_map_duplicate_updates += 1
                return
        # Unknown SLAM cells may still be explored when the live LiDAR proves
        # them free.  Known occupied cells, including currently occluded wall
        # corners, remain a persistent navigation constraint.  Construct the
        # distance field only for a genuinely new map revision.
        occupancy = RuntimeOccupancyGrid2D(
            values,
            resolution,
            origin,
            unknown_is_occupied=False,
        )
        with self.lock:
            self.accumulated_grid = (
                values,
                resolution,
                origin,
                message.header.stamp.to_sec(),
            )
            self.accumulated_occupancy = occupancy
            self.accumulated_known_cells = int(np.count_nonzero(values >= 0))
            self.accumulated_map_signature = signature
            self.accumulated_map_revision += 1

    def goal_endpoint_evidence(self, goal):
        """Classify an RViz goal against observed occupancy.

        Unknown or not-yet-covered cells remain legal exploration targets. A
        cell positively observed as occupied is different: no visibility-graph
        density can connect the vehicle centre to it. Detect that condition
        before local exploration so an invalid click cannot masquerade as a
        missing FAR route.
        """

        with self.lock:
            accumulated_grid = self.accumulated_grid
            occupancy = self.accumulated_occupancy
        point = (
            float(goal.pose.position.x),
            float(goal.pose.position.y),
        )
        evidence = {
            "goal_x": point[0],
            "goal_y": point[1],
            "cell_value": None,
            "observed_occupied": False,
            "clearance_m": None,
            "reason": "MAP_NOT_READY",
        }
        if accumulated_grid is None:
            return evidence
        values, resolution, origin, _ = accumulated_grid
        column = int(math.floor((point[0] - float(origin[0])) / resolution))
        row = int(math.floor((point[1] - float(origin[1])) / resolution))
        if not (0 <= row < values.shape[0] and 0 <= column < values.shape[1]):
            evidence["reason"] = "OUTSIDE_CURRENT_SLAM_EXTENT"
            return evidence
        cell_value = int(values[row, column])
        evidence["cell_value"] = cell_value
        if occupancy is not None:
            evidence["clearance_m"] = float(occupancy.point_clearance(point))
        if cell_value >= int(self.visibility.occupied_threshold):
            evidence["observed_occupied"] = True
            evidence["reason"] = "GOAL_IN_OBSERVED_OBSTACLE"
        elif cell_value < 0:
            evidence["reason"] = "GOAL_UNOBSERVED"
        else:
            evidence["reason"] = "GOAL_OBSERVED_FREE"
        return evidence

    def dense_visibility_replan_is_pending(self):
        with self.lock:
            return bool(self.visibility_dense_replan_pending)

    def start_dense_visibility_replan(
        self,
        values,
        resolution,
        origin,
        start_xy,
        goal_xy,
        start_yaw,
        snapshot_version,
        directed_failed_branches,
        trajectory_points,
        stamp,
    ):
        """Start one bounded dense solve for the current request epoch.

        ``snapshot_version`` is only a same-process freshness token for this
        occupancy array.  It is neither a map UUID nor a cache key, and the
        resulting route is never reused by a later node run or scenario.
        """

        if not self.visibility_dense_replan_enabled:
            return False
        goal_key = (round(float(goal_xy[0]), 3), round(float(goal_xy[1]), 3))
        start_key = (
            round(float(start_xy[0]), 3),
            round(float(start_xy[1]), 3),
            round(float(start_yaw), 3),
        )
        with self.lock:
            if self.visibility_dense_replan_pending:
                return False
            request_epoch = self.visibility_planning_request_epoch
            resume_session = None
            if (
                self.visibility_dense_replan_session is not None
                and self.visibility_dense_replan_session_snapshot_version
                == int(snapshot_version)
                and self.visibility_dense_replan_session_request_epoch
                == int(request_epoch)
                and self.visibility_dense_replan_session_goal_key == goal_key
                and math.hypot(
                    self.visibility_dense_replan_session.start[0]
                    - float(start_xy[0]),
                    self.visibility_dense_replan_session.start[1]
                    - float(start_xy[1]),
                ) <= 0.05
                and abs(
                    wrap_angle(
                        float(self.visibility_dense_replan_session.start_yaw)
                        - float(start_yaw)
                    )
                ) <= math.radians(3.0)
            ):
                resume_session = self.visibility_dense_replan_session
                if resume_session.complete:
                    self.visibility_dense_replan_last_status = (
                        "EXHAUSTED_CURRENT_REQUEST"
                    )
                    return False
            if (
                resume_session is None
                and int(snapshot_version)
                == self.visibility_dense_replan_last_attempt_snapshot
                and float(stamp) - self.visibility_dense_replan_last_completed_stamp
                < self.visibility_dense_replan_retry_period
            ):
                return False
            self.visibility_dense_replan_sequence += 1
            sequence = self.visibility_dense_replan_sequence
            self.visibility_dense_replan_session = None
            self.visibility_dense_replan_pending = True
            self.visibility_dense_replan_result = None
            self.visibility_dense_replan_snapshot_version = int(snapshot_version)
            self.visibility_dense_replan_request_epoch = int(request_epoch)
            self.visibility_dense_replan_goal_key = goal_key
            self.visibility_dense_replan_started_stamp = float(stamp)
            self.visibility_dense_replan_last_attempt_snapshot = int(
                snapshot_version
            )
            self.visibility_dense_replan_last_error = None
            self.visibility_dense_replan_last_status = "PENDING"

        # The OccupancyGrid callback replaces its array rather than mutating it,
        # but make this ownership explicit so the worker always sees one coherent
        # online snapshot while SLAM continues publishing newer revisions.
        snapshot_values = np.asarray(values, dtype=np.int16).copy()
        branch_snapshot = tuple(
            tuple((float(point[0]), float(point[1])) for point in branch)
            for branch in directed_failed_branches
        )
        trajectory_snapshot = tuple(
            (float(point[0]), float(point[1])) for point in trajectory_points
        )

        def worker():
            started = time.perf_counter()
            result = None
            error = None
            try:
                result, updated_session = self.visibility.plan_progressive(
                    snapshot_values,
                    float(resolution),
                    (float(origin[0]), float(origin[1])),
                    (float(start_xy[0]), float(start_xy[1])),
                    (float(goal_xy[0]), float(goal_xy[1])),
                    initial_maximum_nodes=(
                        self.visibility_dense_replan_initial_nodes
                    ),
                    maximum_nodes=self.visibility_dense_replan_maximum_nodes,
                    node_step=self.visibility_dense_replan_node_step,
                    time_budget_s=self.visibility_dense_replan_time_budget,
                    directed_failed_branches=branch_snapshot,
                    trajectory_points=trajectory_snapshot,
                    maximum_trajectory_vertices=(
                        self.visibility_trajectory_bridge_maximum_nodes
                    ),
                    failure_buffer_m=self.topology.failure_buffer_m,
                    start_yaw=float(start_yaw),
                    session=resume_session,
                    return_session=True,
                )
            except Exception as exc:  # fail closed; ordinary FAR remains alive
                error = "%s: %s" % (type(exc).__name__, exc)
                updated_session = None
            duration_ms = 1000.0 * (time.perf_counter() - started)
            completed_stamp = rospy.Time.now().to_sec()
            with self.lock:
                if (
                    sequence != self.visibility_dense_replan_sequence
                    or request_epoch != self.visibility_planning_request_epoch
                ):
                    return
                self.visibility_dense_replan_result = {
                    "plan": result,
                    "error": error,
                    "sequence": int(sequence),
                    "request_epoch": int(request_epoch),
                    "snapshot_version": int(snapshot_version),
                    "goal_key": goal_key,
                    "start_key": start_key,
                    "session": updated_session,
                    "duration_ms": float(duration_ms),
                }
                self.visibility_dense_replan_last_duration_ms = float(duration_ms)
                self.visibility_dense_replan_last_error = error
                self.visibility_dense_replan_last_completed_stamp = float(
                    completed_stamp
                )
                self.visibility_dense_replan_last_status = (
                    "ERROR"
                    if error is not None
                    else "READY_%s" % result.status
                )

        threading.Thread(
            target=worker,
            name="dep_car_far_dense_%d" % sequence,
            daemon=True,
        ).start()
        rospy.logwarn(
            "Started background dense FAR expansion request_epoch=%d "
            "snapshot_version=%d nodes=%d..%d step=%d resumed=%s",
            request_epoch,
            int(snapshot_version),
            self.visibility_dense_replan_initial_nodes,
            self.visibility_dense_replan_maximum_nodes,
            self.visibility_dense_replan_node_step,
            resume_session is not None,
        )
        return True

    def take_dense_visibility_replan(
        self,
        goal_key,
        current_xy,
        current_snapshot_version,
        values,
        resolution,
        origin,
    ):
        """Consume a completed result only for the current transient request."""

        with self.lock:
            document = self.visibility_dense_replan_result
            if document is None:
                return None
            self.visibility_dense_replan_result = None
            self.visibility_dense_replan_pending = False
            current_epoch = self.visibility_planning_request_epoch
        expected_goal = (
            round(float(goal_key[0]), 3), round(float(goal_key[1]), 3)
        )
        if (
            document["request_epoch"] != current_epoch
            or document["goal_key"] != expected_goal
        ):
            return None
        if document["error"] is not None:
            rospy.logerr(
                "Background dense FAR expansion failed: %s", document["error"]
            )
            return None
        plan = document["plan"]
        same_snapshot = bool(
            int(document["snapshot_version"])
            == int(current_snapshot_version)
        )
        start_drift = math.hypot(
            float(current_xy[0]) - float(document["start_key"][0]),
            float(current_xy[1]) - float(document["start_key"][1]),
        )
        maximum_unknown_fraction = (
            0.02 if plan.mode == "PARTIAL_ATTEMPTABLE" else 1.0
        )
        current_route_clear = bool(
            plan is not None
            and plan.status == "PASS"
            and len(plan.path) >= 2
            and self.visibility.path_is_traversable(
                plan.path,
                values,
                resolution,
                origin,
                maximum_unknown_fraction=maximum_unknown_fraction,
            )
        )
        result_fresh = bool(
            start_drift <= self.visibility_dense_replan_maximum_start_drift
            and (same_snapshot or current_route_clear)
        )
        if same_snapshot and result_fresh:
            with self.lock:
                self.visibility_dense_replan_session = document["session"]
                self.visibility_dense_replan_session_snapshot_version = int(
                    document["snapshot_version"]
                )
                self.visibility_dense_replan_session_request_epoch = int(
                    document["request_epoch"]
                )
                self.visibility_dense_replan_session_goal_key = document[
                    "goal_key"
                ]
                self.visibility_dense_replan_session_start_key = document[
                    "start_key"
                ]
        rospy.logwarn(
            "Completed background dense FAR expansion status=%s "
            "snapshot_version=%d current_snapshot_version=%d nodes=%d/%d "
            "cap_hit=%s stages=%d start_degree=%d goal_degree=%d "
            "disconnect=%s duration_ms=%.1f complete=%s same_snapshot=%s "
            "start_drift=%.3f current_route_clear=%s consumed=%s",
            plan.status,
            int(document["snapshot_version"]),
            int(self.accumulated_map_revision),
            int(plan.candidate_vertices_selected),
            int(plan.candidate_vertices_total),
            bool(plan.node_limit_hit),
            int(plan.progressive_stages),
            int(plan.start_degree),
            int(plan.goal_degree),
            plan.disconnect_class,
            float(document["duration_ms"]),
            bool(plan.progressive_complete),
            same_snapshot,
            float(start_drift),
            current_route_clear,
            result_fresh,
        )
        if not result_fresh:
            self.visibility_dense_replan_last_status = (
                "DISCARDED_STALE_SNAPSHOT"
            )
            return None
        return plan

    def on_local_state(self, message):
        with self.lock:
            self.local_state = message

    def on_candidates(self, message):
        with self.lock:
            self.local_candidates = message

    def on_map_odom_correction(self, message):
        """Queue meaningful SLAM corrections for coherent route revalidation.

        The odom-frame graph is intentionally continuous.  A SLAM loop
        correction changes only its rigid placement in ``map``.  The route is
        checked against the newest map revision without being discarded merely
        because the transform changed.
        """

        try:
            document = json.loads(message.data)
            if document.get("schema") != "DEPCarMapOdomCorrectionV1":
                return
            translation = abs(float(document.get("translation_delta_m", 0.0)))
            yaw = abs(float(document.get("yaw_delta_rad", 0.0)))
        except (TypeError, ValueError, json.JSONDecodeError):
            return
        significant = bool(
            translation >= self.map_correction_replan_translation
            or yaw >= self.map_correction_replan_yaw
        )
        with self.lock:
            self.last_map_correction = dict(document)
            if significant:
                self.last_significant_map_correction = dict(document)
                self.map_correction_generation += 1

    def invalidate_visibility_route(
        self,
        reason,
        *,
        force=False,
        preserve_acquisition=False,
    ):
        """Discard route-local state without changing the mission goal."""

        reason = str(reason)
        if self.visibility_maneuver_transaction_active and not force:
            if reason not in self.visibility_deferred_replan_reasons:
                self.visibility_deferred_replan_reasons.append(reason)
                rospy.loginfo(
                    "Deferred visibility replan reason=%s until committed "
                    "Ackermann manoeuvre completes",
                    reason,
                )
            self.visibility_last_replan_reason = "deferred_" + reason
            return False

        self.visibility_plan = None
        self.reset_visibility_cursor()
        self.visibility_active_route_accepted = False
        self.visibility_active_route_motion_authorized = False
        self.visibility_last_candidate_plan = None
        self.visibility_last_candidate_retained_active = False
        self.clear_connected_visibility_candidate(
            "route_invalidated_" + reason
        )
        self.visibility_route_dropout_started_stamp = None
        self.visibility_route_lease_active = False
        self.visibility_route_lease_reason = "route_invalidated_" + reason
        self.visibility_route_lease_prefix_length_m = 0.0
        self.visibility_planned_revision = -1
        self.visibility_planned_failed_branches = -1
        self.visibility_last_plan_stamp = -math.inf
        self.visibility_last_replan_reason = reason
        self.visibility_replan_count += 1
        self.visibility_route_validation_revision = -1
        self.visibility_route_validation_stamp = -math.inf
        self.visibility_route_validation_passed = False
        if not preserve_acquisition:
            self.visibility_route_acquisition.reset()
        # A measured-pose graph rebuild can preserve evidence for the same
        # topological direction.  The gate itself revokes that evidence if the
        # replacement bearing or cost is materially different.
        self.visibility_route_acquisition_reason = "route_invalidated_" + reason
        return True

    def sync_visibility_maneuver_transaction(self, active, map_pose, stamp):
        """Latch one FAR route across a local forward/reverse transaction."""

        active = bool(active)
        if active and not self.visibility_maneuver_transaction_active:
            snapshot = list(
                self.visibility_remaining_path_map
                or self.visibility_cursor_path(map_pose[:2])
            )
            if len(snapshot) < 2 and self.visibility_plan is not None:
                snapshot = list(self.visibility_plan.path)
            route_eligible = bool(
                self.visibility_active_route_motion_authorized
                and self.visibility_plan is not None
                and self.visibility_plan.status == "PASS"
                and len(snapshot) >= 2
            )
            if route_eligible:
                # Ignore deviation while taking the finite space-creation
                # legs.  The hard safety layer still certifies every driven
                # primitive; this only prevents a new topological direction
                # from pulling against a half-finished turnaround.
                committed = self.trim_visibility_path(
                    map_pose[:2], snapshot, maximum_deviation=math.inf
                )
                self.visibility_maneuver_path_map = committed or snapshot
            else:
                # A local recovery manoeuvre may start while FAR has no
                # route.  It must not create an empty FAR transaction that
                # defers later route updates until that manoeuvre ends.
                self.visibility_maneuver_path_map = []
                return
            self.visibility_maneuver_transaction_active = True
            rospy.loginfo(
                "Latched FAR route handoff for committed Ackermann manoeuvre "
                "points=%d",
                len(self.visibility_maneuver_path_map),
            )
            return
        if not active and self.visibility_maneuver_transaction_active:
            pending = list(self.visibility_deferred_replan_reasons)
            committed_snapshot = list(
                self.visibility_maneuver_path_map or self.visibility_path_map
            )
            self.visibility_maneuver_transaction_active = False
            self.visibility_maneuver_path_map = []
            self.visibility_deferred_replan_reasons = []
            non_map_pending = [
                reason for reason in pending
                if reason != "slam_map_odom_correction"
            ]
            # Re-anchor the frozen suffix at the final measured pose.  Keep
            # it only when its look-ahead remains in the forward/lateral
            # sector.  A rearward projection is precisely the stale-route
            # condition that caused an immediate second reverse leg, so that
            # case must go through a fresh FAR acquisition instead.
            reanchored = self.trim_visibility_path(
                map_pose[:2], committed_snapshot, maximum_deviation=math.inf
            )
            target = self.point_along_polyline(reanchored, 0.75)
            post_bearing = math.pi
            if target is not None:
                post_bearing = wrap_angle(
                    math.atan2(target[1] - map_pose[1], target[0] - map_pose[0])
                    - map_pose[2]
                )
            route_action = "retained_forward_suffix"
            if (
                non_map_pending
                or len(reanchored) < 2
                or abs(post_bearing) >= math.radians(110.0)
            ):
                reason = "post_maneuver_pose_reanchor"
                if non_map_pending:
                    reason += "+" + "+".join(non_map_pending)
                self.invalidate_visibility_route(reason, force=True)
                self.visibility_route_acquisition_started_stamp = None
                route_action = "fresh_far_reanchor"
            else:
                source = (
                    self.visibility_route_cursor.source
                    if self.visibility_route_cursor.active
                    else "FAR_POST_MANEUVER"
                )
                self.bind_visibility_route(
                    reanchored,
                    map_pose[:2],
                    self.visibility_goal_key,
                    source,
                )
                self.visibility_last_plan_stamp = float(stamp)
            rospy.loginfo(
                "Released FAR route handoff after committed Ackermann "
                "manoeuvre duration_end=%.3f deferred=%s route_action=%s "
                "post_bearing=%.3frad",
                float(stamp),
                ",".join(pending) if pending else "none",
                route_action,
                post_bearing,
            )

    def sync_route_turnaround_transaction(
        self, local_state, active, odom_pose, stamp
    ):
        """Track one multi-leg turnaround independently of route refreshes."""

        purpose = (
            "" if local_state is None else str(local_state.maneuver_purpose)
        )
        turnaround_active = bool(active and purpose == "forward_restoration")
        if turnaround_active and not self.route_turnaround_transaction_active:
            self.route_turnaround_transaction_sequence += 1
            self.route_turnaround_transaction_id = (
                self.route_turnaround_transaction_sequence
            )
            self.route_turnaround_transaction_active = True
            self.route_turnaround_source = str(self.last_guidance_source)
            self.route_turnaround_trigger = (
                "NEW_GOAL_OR_REAR_ROUTE"
                if self.route_turnaround_source
                in (
                    "EXPLORED_TOPOLOGY",
                    "FAR_KNOWN_VISIBILITY",
                    "FAR_ATTEMPTABLE_VISIBILITY",
                    "FAR_ATTEMPTABLE_NAVIGATION",
                    "FAR_PARTIAL_ATTEMPTABLE",
                )
                else "LOCAL_FORWARD_RESTORATION"
            )
            self.route_turnaround_outcome = "ACTIVE"
            self.route_turnaround_start_trail_index = (
                len(self.trail.points) - 1 if self.trail.points else None
            )
            if (
                self.route_turnaround_source == "EXPLORED_TOPOLOGY"
                and self.topology_route_cursor.active
            ):
                self.topology_rear_authority_issued = True
                self.topology_last_turnaround_route_id = (
                    self.topology_route_cursor.route_id
                )
            rospy.loginfo(
                "Started route turnaround transaction id=%d source=%s "
                "trigger=%s route_id=%s",
                self.route_turnaround_transaction_id,
                self.route_turnaround_source,
                self.route_turnaround_trigger,
                (
                    self.topology_route_cursor.route_id
                    if self.route_turnaround_source == "EXPLORED_TOPOLOGY"
                    and self.topology_route_cursor.active
                    else self.visibility_route_cursor.route_id
                    if self.visibility_route_cursor.active
                    else "none"
                ),
            )
            return
        if not turnaround_active and self.route_turnaround_transaction_active:
            completed_id = self.route_turnaround_transaction_id
            completed_source = self.route_turnaround_source
            local_detail = (
                "" if local_state is None else str(local_state.detail).lower()
            )
            failed = "turnaround_budget_exhausted" in local_detail
            if completed_source == "EXPLORED_TOPOLOGY":
                self.topology_turnaround_completion_xy = tuple(
                    np.asarray(odom_pose[:2], dtype=float)
                )
                self.topology_turnaround_completion_progress_m = (
                    self.topology_route_cursor.progress_m
                    if self.topology_route_cursor.active
                    else None
                )
            self.route_turnaround_transaction_active = False
            self.route_turnaround_outcome = (
                "FAILED_BUDGET_EXHAUSTED" if failed else "COMPLETED"
            )
            # Parking-style legs are deliberately excluded from the ordinary
            # ingress stack.  Add only the final measured pose as a neutral
            # transaction boundary, so future dead-end evidence cannot label
            # all of the local forward/reverse oscillations as one branch.
            if local_state is not None:
                features = (
                    self.local_features(self.grid)
                    if self.grid is not None
                    else {"junction": False, "turnaround": False}
                )
                self.record_breadcrumb(
                    odom_pose,
                    stamp,
                    features,
                    motion_direction=0,
                    record_topology=False,
                    force=True,
                )
            self.route_turnaround_start_trail_index = None
            rospy.loginfo(
                "%s route turnaround transaction id=%d source=%s "
                "duration_end=%.3f",
                "Failed" if failed else "Completed",
                completed_id,
                completed_source,
                float(stamp),
            )

    def apply_pending_map_correction(self, stamp):
        with self.lock:
            generation = self.map_correction_generation
            correction = self.last_significant_map_correction
        if generation == self.applied_map_correction_generation:
            return False
        if (
            float(stamp) - self.last_map_correction_replan_stamp
            < self.map_correction_replan_minimum_period
        ):
            return False
        self.applied_map_correction_generation = generation
        self.last_map_correction_replan_stamp = float(stamp)
        # ``map`` is the stable coordinate authority for both the SLAM grid
        # and the FAR path.  A scan-matching change in map->odom moves the
        # measured vehicle pose inside that frame; it does not by itself make
        # every map-frame route obsolete.  Revalidate the committed suffix on
        # the next map revision and replan only if it crosses newly occupied
        # space or the corrected pose is genuinely too far from it.  The old
        # behaviour destroyed a valid route on consecutive 0.1--0.3 m scan
        # corrections and repeatedly turned a course error into a gear shift.
        self.visibility_route_validation_revision = -1
        self.visibility_route_validation_stamp = -math.inf
        self.visibility_last_replan_reason = "slam_map_odom_revalidation"
        if self.recovery.state == MemoryNavigationState.FAR_DEAD_END_EGRESS:
            # Keep the historical entry anchor in map, replace only the live
            # first pose and revalidate the remaining connector.  Incrementing
            # the revision makes this correction visible without rolling the
            # monotonic egress cursor back to an older breadcrumb.
            self.dead_end_escape_route_revision += 1
            rospy.loginfo(
                "Re-anchoring FAR dead-end egress escape_id=%d revision=%d "
                "on coherent SLAM correction",
                self.dead_end_escape_id,
                self.dead_end_escape_route_revision,
            )
        if correction is not None:
            rospy.loginfo(
                "Revalidating committed FAR route after coherent SLAM "
                "correction translation=%.3fm yaw=%.3frad",
                abs(float(correction.get("translation_delta_m", 0.0))),
                abs(float(correction.get("yaw_delta_rad", 0.0))),
            )
        return True

    def on_goal(self, message):
        if message.header.frame_id.lstrip("/") not in ("", "map"):
            rospy.logerr("Memory navigation accepts mission goals in map frame only")
            self.publish_status("INVALID_GOAL", "goal frame must be map")
            return
        with self.lock:
            preempted_state = self.recovery.state.value
            self.visibility_planning_request_epoch += 1
            # Invalidate, rather than reuse, any background result belonging to
            # the previous RViz goal.  The worker is daemonized and may finish,
            # but its request epoch prevents it from publishing into this goal.
            self.visibility_dense_replan_pending = False
            self.visibility_dense_replan_result = None
            self.visibility_dense_replan_request_epoch = -1
            self.visibility_dense_replan_goal_key = None
            self.visibility_dense_replan_snapshot_version = -1
            self.visibility_dense_replan_last_status = "NEW_GOAL"
            self.visibility_dense_replan_last_attempt_snapshot = -1
            self.visibility_dense_replan_session = None
            self.visibility_dense_replan_session_snapshot_version = -1
            self.visibility_dense_replan_session_request_epoch = -1
            self.visibility_dense_replan_session_goal_key = None
            self.visibility_dense_replan_session_start_key = None
            reanchored_node = None
            current_odom_pose = None
            if self.odom is not None:
                current_odom_pose = self.odom_pose(self.odom)
                reanchored_node = self.topology.reanchor(
                    *current_odom_pose[:2], maximum_distance_m=1.5
                )
                # Breadcrumbs are a transaction-local escape stack.  Keeping
                # the previous mission's stack made a blockage after goal B
                # reverse through the entire route used for goal A.  Sparse
                # topology and failed branches remain persistent; only the
                # reversible ingress stack is re-anchored at the new goal.
                self.trail.clear()
                self.trail.record(
                    *current_odom_pose,
                    self.odom.header.stamp.to_sec(),
                    motion_direction=0,
                    map_pose=self.last_map_pose,
                    map_revision=self.accumulated_map_revision,
                    force=True,
                )
            else:
                self.trail.clear()
            self.goal = message
            self.local_state = None
            self.local_candidates = None
            self.last_progress_pose = None
            self.last_progress_stamp = None
            self.backtrack_start_index = None
            self.backtrack_cursor_index = None
            self.backtrack_site_index = None
            self.backtrack_start_xy = None
            self.resume_start_xy = None
            self.resume_target_index = None
            self.resume_fallback_target_xy = None
            self.resume_started_stamp = None
            self.resume_travelled_m = 0.0
            self.resume_last_xy = None
            self.resume_blocked_since = None
            self.backtrack_started_stamp = None
            self.backtrack_blocked_since = None
            self.previous_goal_heading = None
            self.last_heading_decision = None
            self.last_boundary_decision = None
            # Side failures are scoped to the position goal that produced
            # them; a new RViz goal starts a fresh obstacle transaction.
            self.boundary.reset(clear_failures=True)
            self.committed_topology_path = []
            self.committed_topology_failed_branches = len(
                self.topology.failed_branches
            )
            self.topology_route_cursor.reset()
            self.topology_route_revision = 0
            self.topology_last_carrot_odom = None
            self.topology_rear_authority_issued = False
            self.topology_last_turnaround_route_id = ""
            self.topology_turnaround_completion_progress_m = None
            self.topology_turnaround_completion_xy = None
            self.route_turnaround_transaction_active = False
            self.route_turnaround_transaction_id = 0
            self.route_turnaround_source = "NONE"
            self.route_turnaround_trigger = "NONE"
            self.route_turnaround_outcome = "NONE"
            self.route_turnaround_start_trail_index = None
            self.visibility_plan = None
            self.reset_visibility_cursor()
            self.visibility_route_revision = 0
            self.visibility_last_handoff = None
            self.visibility_active_route_accepted = False
            self.visibility_active_route_motion_authorized = False
            self.visibility_last_candidate_plan = None
            self.visibility_last_candidate_retained_active = False
            self.clear_connected_visibility_candidate("new_goal")
            self.visibility_connected_candidate_suppressed_weaker = 0
            self.visibility_connected_candidate_promotions = 0
            self.visibility_route_dropout_started_stamp = None
            self.visibility_route_lease_active = False
            self.visibility_route_lease_reason = "new_goal"
            self.visibility_route_lease_prefix_length_m = 0.0
            self.visibility_last_authorized_direction_map = None
            self.visibility_last_authorized_direction_stamp = -math.inf
            self.visibility_goal_key = None
            self.visibility_planned_revision = -1
            self.visibility_planned_failed_branches = -1
            self.visibility_last_plan_stamp = -math.inf
            self.visibility_last_plan_duration_ms = None
            self.visibility_replan_count = 0
            self.visibility_last_replan_reason = "new_goal"
            self.visibility_static_blocked_since = None
            self.visibility_last_local_static_block_stamp = -math.inf
            self.visibility_no_route_since = None
            self.visibility_last_blocked_replan_stamp = -math.inf
            self.visibility_static_replan_failures = 0
            self.visibility_static_recovery_pending = False
            self.visibility_static_recovery_handoffs = 0
            self.far_emergency_egress_active = False
            self.dead_end_escape_id = 0
            self.dead_end_escape_branch_id = None
            self.dead_end_escape_route_revision = 0
            self.dead_end_escape_site_kind = "NONE"
            self.dead_end_escape_target_distance_m = 0.0
            self.dead_end_escape_cross_track_error_m = 0.0
            self.dead_end_escape_last_route_map = []
            self.dead_end_escape_started_map_correction_generation = (
                self.map_correction_generation
            )
            self.dead_end_escape_completion_reason = "NONE"
            self.dead_end_escape_diverged_since = None
            self.dead_end_escape_connector_unavailable_since = None
            self.dead_end_escape_live_reanchors = 0
            self.dead_end_escape_live_target_index = None
            self.failed_branch_exit_lock_branch_id = None
            self.failed_branch_exit_lock_origin_odom = None
            self.failed_branch_exit_lock_started_stamp = None
            self.failed_branch_exit_lock_progress_m = 0.0
            self.visibility_route_acquisition.reset()
            self.visibility_route_acquisition_started_stamp = None
            self.visibility_route_acquisition_reason = (
                "waiting_for_visibility_route"
            )
            self.visibility_initial_exploration_distance_m = 0.0
            self.visibility_initial_exploration_last_xy = (
                None
                if current_odom_pose is None
                else np.asarray(current_odom_pose[:2], dtype=float)
            )
            self.visibility_initial_exploration_started_stamp = None
            self.visibility_initial_exploration_complete = bool(
                self.visibility_mapping_session_established
            )
            self.visibility_initial_exploration_reason = (
                "online_mapping_session_already_established"
                if self.visibility_mapping_session_established
                else "new_mapping_session_local_exploration"
            )
            self.visibility_maneuver_transaction_active = False
            self.visibility_maneuver_path_map = []
            self.visibility_deferred_replan_reasons = []
            self.visibility_course_revalidation_pending = False
            self.visibility_route_validation_revision = -1
            self.visibility_route_validation_stamp = -math.inf
            self.visibility_route_validation_passed = False
            self.applied_map_correction_generation = self.map_correction_generation
            self.last_map_correction_replan_stamp = -math.inf
            self.best_goal_distance = None
            self.last_goal_improvement_stamp = None
            self.recovery.reset(active=True)
        self.publish_status(
            "RECEIVED",
            "new position goal; final heading is unconstrained",
            preempted_state=preempted_state,
            topology_reanchored_node=reanchored_node,
            retained_failed_branches=len(self.topology.failed_branches),
        )

    def lookup_pose(self):
        transform = self.tf_buffer.lookup_transform(
            "map", "dummy", rospy.Time(0), rospy.Duration(0.05)
        ).transform
        return (
            float(transform.translation.x),
            float(transform.translation.y),
            yaw_from_quaternion(transform.rotation),
        )

    def odom_to_map(self, points):
        transform = self.tf_buffer.lookup_transform(
            "map", "odom", rospy.Time(0), rospy.Duration(0.05)
        ).transform
        yaw = yaw_from_quaternion(transform.rotation)
        cosine, sine = math.cos(yaw), math.sin(yaw)
        values = np.asarray(points, dtype=float)
        return np.column_stack((
            transform.translation.x + cosine * values[:, 0] - sine * values[:, 1],
            transform.translation.y + sine * values[:, 0] + cosine * values[:, 1],
        ))

    def map_to_odom(self, points):
        transform = self.tf_buffer.lookup_transform(
            "map", "odom", rospy.Time(0), rospy.Duration(0.05)
        ).transform
        yaw = yaw_from_quaternion(transform.rotation)
        cosine, sine = math.cos(yaw), math.sin(yaw)
        values = np.asarray(points, dtype=float)
        translated = values - np.asarray(
            (transform.translation.x, transform.translation.y), dtype=float
        )
        return np.column_stack((
            cosine * translated[:, 0] + sine * translated[:, 1],
            -sine * translated[:, 0] + cosine * translated[:, 1],
        ))

    def visibility_trajectory_points_map(self):
        """Project persistent driven topology into the current SLAM frame.

        This is the 2-D counterpart of upstream FAR's trajectory/inter-nav
        vertices.  Topology is recorded from actual wheel/IMU odometry and is
        never a map- or scenario-specific route cache.  Reprojecting the
        complete odom geometry with one current rigid ``map<-odom`` transform
        avoids mixing historical SLAM corrections into a saw-tooth planning
        graph; every candidate edge is still rechecked on current occupancy by
        :class:`DynamicVisibilityPlanner`.
        """

        if (
            not self.visibility_trajectory_bridge_enabled
            or not self.topology.nodes
            or self.visibility_trajectory_bridge_maximum_nodes == 0
        ):
            return ()
        ordered = [
            self.topology.nodes[node_id]
            for node_id in sorted(self.topology.nodes)
        ]
        odom_points = np.asarray(
            [(node.x, node.y) for node in ordered], dtype=float
        )
        if not len(odom_points):
            return ()
        map_points = self.odom_to_map(odom_points)
        return tuple(
            (float(point[0]), float(point[1])) for point in map_points
        )

    @staticmethod
    def odom_pose(message):
        return (
            float(message.pose.pose.position.x),
            float(message.pose.pose.position.y),
            yaw_from_quaternion(message.pose.pose.orientation),
        )

    def publish_status(self, state, detail, **evidence):
        topology = self.topology.summary()
        topology_observation = self.topology_route_cursor.last_observation
        if topology_observation is not None:
            topology["rolling_route"] = {
                "route_id": topology_observation.route_id,
                "route_revision": int(topology_observation.revision),
                "source": topology_observation.source,
                "progress_m": float(topology_observation.progress_m),
                "total_m": float(topology_observation.total_m),
                "carrot_m": float(topology_observation.carrot_m),
                "carrot_xy": list(topology_observation.carrot_xy),
                "lookahead_m": float(topology_observation.lookahead_m),
                "route_tangent_rad": float(
                    topology_observation.tangent_bearing_rad
                ),
                "cross_track_error_m": float(
                    topology_observation.deviation_m
                ),
                "passed_vertices": int(
                    topology_observation.skipped_vertices
                ),
                "carrot_distance_m": float(
                    topology_observation.carrot_distance_m
                ),
                "carrot_advanced": bool(
                    topology_observation.carrot_advanced
                ),
                "carrot_hold_reason": (
                    topology_observation.carrot_hold_reason
                ),
            }
        boundary = {
            "mode": self.boundary.mode.value,
            "active": bool(self.boundary.active),
            "side": int(self.boundary.side),
            "travelled_m": float(self.boundary.travelled_m),
            "hit_goal_distance_m": self.boundary.hit_goal_distance_m,
            "best_goal_distance_m": self.boundary.best_goal_distance_m,
            "failed_side_count": len(self.boundary.failed_sides),
        }
        if self.last_boundary_decision is not None:
            boundary.update({
                "direct_clearance_m": float(
                    self.last_boundary_decision.direct_clearance_m
                ),
                "reason": self.last_boundary_decision.reason,
            })
        visibility = {
            "available": self.visibility_plan is not None,
            "path_points": len(self.visibility_path_map),
            "planning_request_epoch": int(
                self.visibility_planning_request_epoch
            ),
            "occupancy_snapshot_version": int(
                self.visibility_planned_revision
            ),
            "planning_time_ms": self.visibility_last_plan_duration_ms,
            "duplicate_map_updates_skipped": int(
                self.accumulated_map_duplicate_updates
            ),
            "replan_count": int(self.visibility_replan_count),
            "last_replan_reason": self.visibility_last_replan_reason,
            "validated_map_revision": int(
                self.visibility_route_validation_revision
            ),
            "latest_validation_passed": bool(
                self.visibility_route_validation_passed
            ),
            "maneuver_transaction_active": bool(
                self.visibility_maneuver_transaction_active
            ),
            "maneuver_transaction_points": len(
                self.visibility_maneuver_path_map
            ),
            "deferred_replan_reasons": list(
                self.visibility_deferred_replan_reasons
            ),
            "route_acquisition_accepted": bool(
                self.visibility_active_route_accepted
            ),
            "route_motion_authorized": bool(
                self.visibility_active_route_motion_authorized
            ),
            "route_acquisition_confirmations": int(
                self.visibility_route_acquisition.confirmations
            ),
            "route_acquisition_reason": self.visibility_route_acquisition_reason,
            "initial_local_exploration_distance_m": float(
                self.visibility_initial_exploration_distance_m
            ),
            "initial_local_exploration_minimum_m": float(
                self.visibility_initial_exploration_minimum
            ),
            "initial_local_exploration_complete": bool(
                self.visibility_initial_exploration_complete
            ),
            "initial_local_exploration_reason": (
                self.visibility_initial_exploration_reason
            ),
            "static_replan_failures": int(
                self.visibility_static_replan_failures
            ),
            "static_recovery_far_handoff_pending": bool(
                self.visibility_static_recovery_pending
            ),
            "static_recovery_far_handoffs": int(
                self.visibility_static_recovery_handoffs
            ),
            "no_route_since": self.visibility_no_route_since,
            "last_local_static_block_stamp": (
                None
                if not math.isfinite(
                    self.visibility_last_local_static_block_stamp
                )
                else self.visibility_last_local_static_block_stamp
            ),
            "certified_static_egress_active": bool(
                self.far_emergency_egress_active
            ),
            "latest_candidate_retained_active_route": bool(
                self.visibility_last_candidate_retained_active
            ),
            "transient_route_lease_active": bool(
                self.visibility_route_lease_active
            ),
            "transient_route_lease_reason": self.visibility_route_lease_reason,
            "transient_route_lease_prefix_length_m": float(
                self.visibility_route_lease_prefix_length_m
            ),
            "transient_route_dropout_age_s": (
                None
                if self.visibility_route_dropout_started_stamp is None
                else max(
                    0.0,
                    rospy.Time.now().to_sec()
                    - self.visibility_route_dropout_started_stamp,
                )
            ),
            "connected_route_transaction": {
                "schema": "DEPCarP6V431ConnectedRouteTransactionV1",
                "status": self.visibility_connected_candidate_status,
                "reason": self.visibility_connected_candidate_reason,
                "route_id": (
                    self.visibility_connected_candidate_route_id or None
                ),
                "goal_key": self.visibility_connected_candidate_goal_key,
                "source_content_revision": int(
                    self.visibility_connected_candidate_revision
                ),
                "discovered_stamp": (
                    None
                    if not math.isfinite(
                        self.visibility_connected_candidate_discovered_stamp
                    )
                    else float(
                        self.visibility_connected_candidate_discovered_stamp
                    )
                ),
                "suppressed_weaker_candidates": int(
                    self.visibility_connected_candidate_suppressed_weaker
                ),
                "promotions": int(
                    self.visibility_connected_candidate_promotions
                ),
                "cross_episode_route_cache": False,
                "map_identity_input": False,
            },
            "dense_replan": {
                "enabled": bool(self.visibility_dense_replan_enabled),
                "pending": bool(self.visibility_dense_replan_pending),
                "status": self.visibility_dense_replan_last_status,
                "request_epoch": int(
                    self.visibility_dense_replan_request_epoch
                ),
                "occupancy_snapshot_version": int(
                    self.visibility_dense_replan_snapshot_version
                ),
                "initial_nodes": int(
                    self.visibility_dense_replan_initial_nodes
                ),
                "maximum_nodes": int(
                    self.visibility_dense_replan_maximum_nodes
                ),
                "node_step": int(self.visibility_dense_replan_node_step),
                "time_budget_s": float(
                    self.visibility_dense_replan_time_budget
                ),
                "last_duration_ms": (
                    self.visibility_dense_replan_last_duration_ms
                ),
                "last_error": self.visibility_dense_replan_last_error,
                "resume_available": bool(
                    self.visibility_dense_replan_session is not None
                    and not self.visibility_dense_replan_session.complete
                ),
                "completed_stages": (
                    None
                    if self.visibility_dense_replan_session is None
                    else int(
                        self.visibility_dense_replan_session.next_stage_index
                    )
                ),
                "cross_episode_route_cache": False,
                "map_identity_input": False,
            },
        }
        visibility_observation = self.visibility_route_cursor.last_observation
        if visibility_observation is not None:
            visibility["rolling_route"] = {
                "route_id": visibility_observation.route_id,
                "route_revision": int(visibility_observation.revision),
                "source": visibility_observation.source,
                "progress_m": float(visibility_observation.progress_m),
                "total_m": float(visibility_observation.total_m),
                "carrot_m": float(visibility_observation.carrot_m),
                "carrot_xy": list(visibility_observation.carrot_xy),
                "lookahead_m": float(visibility_observation.lookahead_m),
                "route_tangent_rad": float(
                    visibility_observation.tangent_bearing_rad
                ),
                "cross_track_error_m": float(
                    visibility_observation.deviation_m
                ),
                "passed_vertices": int(
                    visibility_observation.skipped_vertices
                ),
                "carrot_distance_m": float(
                    visibility_observation.carrot_distance_m
                ),
                "carrot_advanced": bool(
                    visibility_observation.carrot_advanced
                ),
                "carrot_hold_reason": (
                    visibility_observation.carrot_hold_reason
                ),
            }
        if self.visibility_last_handoff is not None:
            visibility["latest_route_handoff"] = {
                "accepted": bool(self.visibility_last_handoff.accepted),
                "reason": self.visibility_last_handoff.reason,
                "entry_distance_m": float(
                    self.visibility_last_handoff.entry_distance_m
                ),
                "direction_change_rad": (
                    None
                    if self.visibility_last_handoff.direction_change_rad is None
                    else float(
                        self.visibility_last_handoff.direction_change_rad
                    )
                ),
            }
        if self.visibility_plan is not None:
            visibility.update({
                "status": self.visibility_plan.status,
                "mode": self.visibility_plan.mode,
                "reason": self.visibility_plan.reason,
                "nodes": len(self.visibility_plan.nodes),
                "edges": len(self.visibility_plan.edges),
                "known_edges": self.visibility_plan.known_edges,
                "attemptable_edges": self.visibility_plan.attemptable_edges,
                "path_cost": self.visibility_plan.path_cost,
                "path_length": self.visibility_plan.path_length,
                "path_unknown_fraction": (
                    self.visibility_plan.path_unknown_fraction
                ),
                "candidate_vertices_total": int(
                    self.visibility_plan.candidate_vertices_total
                ),
                "candidate_vertices_selected": int(
                    self.visibility_plan.candidate_vertices_selected
                ),
                "node_limit_hit": bool(self.visibility_plan.node_limit_hit),
                "planning_node_limit": int(
                    self.visibility_plan.planning_node_limit
                ),
                "start_degree": int(self.visibility_plan.start_degree),
                "goal_degree": int(self.visibility_plan.goal_degree),
                "connected_components": int(
                    self.visibility_plan.connected_components
                ),
                "start_component_size": int(
                    self.visibility_plan.start_component_size
                ),
                "goal_component_size": int(
                    self.visibility_plan.goal_component_size
                ),
                "start_clearance_m": self.visibility_plan.start_clearance_m,
                "goal_clearance_m": self.visibility_plan.goal_clearance_m,
                "disconnect_class": self.visibility_plan.disconnect_class,
                "progressive_stages": int(
                    self.visibility_plan.progressive_stages
                ),
            })
        if self.visibility_last_candidate_plan is not None:
            visibility["latest_candidate"] = {
                "status": self.visibility_last_candidate_plan.status,
                "mode": self.visibility_last_candidate_plan.mode,
                "reason": self.visibility_last_candidate_plan.reason,
                "path_points": len(self.visibility_last_candidate_plan.path),
                "candidate_vertices_total": int(
                    self.visibility_last_candidate_plan.candidate_vertices_total
                ),
                "candidate_vertices_selected": int(
                    self.visibility_last_candidate_plan.candidate_vertices_selected
                ),
                "node_limit_hit": bool(
                    self.visibility_last_candidate_plan.node_limit_hit
                ),
                "start_degree": int(
                    self.visibility_last_candidate_plan.start_degree
                ),
                "goal_degree": int(
                    self.visibility_last_candidate_plan.goal_degree
                ),
                "disconnect_class": (
                    self.visibility_last_candidate_plan.disconnect_class
                ),
            }
        document = {
            "backend": "online_far_visibility_memory",
            "state": str(state),
            "detail": str(detail),
            "breadcrumb_count": len(self.trail.points),
            "topology": topology,
            "guidance_source": self.last_guidance_source,
            "topology_path_points": len(self.last_topology_path),
            "accumulated_map_available": self.accumulated_grid is not None,
            "uses_full_map_search": False,
            "uses_dynamic_visibility_graph": True,
            "visibility_route_direction_authority": bool(
                self.visibility_route_direction_authority
            ),
            "breadcrumb_motion_authority": bool(
                self.breadcrumb_motion_authority
            ),
            "breadcrumb_role": (
                "legacy_motion_authority"
                if self.breadcrumb_motion_authority
                else (
                    "topology_anchor_and_closed_loop_far_dead_end_egress"
                    if self.far_static_egress_enabled
                    else "diagnostics_and_topology_only"
                )
            ),
            "dead_end_escape": {
                "escape_id": int(self.dead_end_escape_id),
                "active": bool(
                    self.recovery.state
                    == MemoryNavigationState.FAR_DEAD_END_EGRESS
                ),
                "authority_priority": (
                    "HARD_VETO>FAR_DEAD_END_EGRESS>LOCAL_TURNAROUND>"
                    "FAR_ROUTE>TOPOLOGY>LOCAL_EXPLORATION"
                ),
                "failed_branch_id": self.dead_end_escape_branch_id,
                "site_kind": self.dead_end_escape_site_kind,
                "target_distance_m": float(
                    self.dead_end_escape_target_distance_m
                ),
                "cross_track_error_m": float(
                    self.dead_end_escape_cross_track_error_m
                ),
                "route_revision": int(
                    self.dead_end_escape_route_revision
                ),
                "map_reanchors": max(
                    0,
                    self.map_correction_generation
                    - self.dead_end_escape_started_map_correction_generation,
                ),
                "live_connector_reanchors": int(
                    self.dead_end_escape_live_reanchors
                ),
                "live_connector_target_index": (
                    self.dead_end_escape_live_target_index
                ),
                "connector_unavailable_since": (
                    self.dead_end_escape_connector_unavailable_since
                ),
                "failed_branch_exit_lock": {
                    "active": bool(
                        self.failed_branch_exit_lock_branch_id is not None
                    ),
                    "branch_id": self.failed_branch_exit_lock_branch_id,
                    "progress_m": float(
                        self.failed_branch_exit_lock_progress_m
                    ),
                    "required_progress_m": float(
                        self.failed_branch_exit_lock_progress
                    ),
                },
                "completion_reason": self.dead_end_escape_completion_reason,
                "background_far_ready": bool(
                    self.visibility_active_route_motion_authorized
                    and self.visibility_plan is not None
                    and self.visibility_plan.status == "PASS"
                ),
            },
            "boundary_follow": boundary,
            "visibility_graph": visibility,
            "route_turnaround_transaction": {
                "id": int(self.route_turnaround_transaction_id),
                "active": bool(self.route_turnaround_transaction_active),
                "source": self.route_turnaround_source,
                "trigger": self.route_turnaround_trigger,
                "outcome": self.route_turnaround_outcome,
                "topology_rear_authority_consumed": bool(
                    self.topology_rear_authority_issued
                ),
                "topology_route_id": (
                    self.topology_route_cursor.route_id
                    if self.topology_route_cursor.active
                    else ""
                ),
                "last_topology_turnaround_route_id": (
                    self.topology_last_turnaround_route_id
                ),
            },
        }
        active_observation = (
            topology_observation
            if self.last_guidance_source == "EXPLORED_TOPOLOGY"
            else visibility_observation
        )
        if active_observation is not None:
            document["active_rolling_route"] = {
                "route_id": active_observation.route_id,
                "route_revision": int(active_observation.revision),
                "source": active_observation.source,
                "progress_m": float(active_observation.progress_m),
                "carrot_m": float(active_observation.carrot_m),
                "carrot_xy": list(active_observation.carrot_xy),
                "lookahead_m": float(active_observation.lookahead_m),
                "route_tangent_rad": float(
                    active_observation.tangent_bearing_rad
                ),
                "carrot_distance_m": float(
                    active_observation.carrot_distance_m
                ),
                "carrot_advanced": bool(
                    active_observation.carrot_advanced
                ),
                "carrot_hold_reason": (
                    active_observation.carrot_hold_reason
                ),
            }
        if self.last_map_correction is not None:
            correction = (
                self.last_significant_map_correction
                or self.last_map_correction
            )
            document["map_odom_correction"] = {
                "generation": int(self.map_correction_generation),
                "applied_generation": int(
                    self.applied_map_correction_generation
                ),
                "translation_delta_m": float(
                    correction.get("translation_delta_m", 0.0)
                ),
                "yaw_delta_rad": float(
                    correction.get("yaw_delta_rad", 0.0)
                ),
                "transform_skew_s": float(
                    correction.get("transform_skew_s", 0.0)
                ),
            }
        document.update(evidence)
        # Evidence such as live goal distance and breadcrumb count can change
        # every control tick.  Log lifecycle transitions only; the latched ROS
        # status message still carries the complete current evidence.
        signature = (str(state), str(detail))
        if signature != self.last_status_signature:
            self.last_status_signature = signature
            if state in ("SAFE_STOP", "INVALID_GOAL"):
                rospy.logwarn("Memory navigation %s: %s", state, detail)
            else:
                rospy.loginfo("Memory navigation %s: %s", state, detail)
        self.status_pub.publish(String(data=json.dumps(document, sort_keys=True)))
        marker = Marker()
        marker.header.stamp = rospy.Time.now()
        marker.header.frame_id = "dummy"
        marker.ns = "dep_car_global_planner_status"
        marker.id = 0
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        marker.pose.position.z = 0.85
        marker.pose.orientation.w = 1.0
        marker.scale.z = 0.16
        marker.color.a = 1.0
        marker.color.r, marker.color.g, marker.color.b = (
            (1.0, 0.2, 0.1) if state == "SAFE_STOP" else (0.2, 0.9, 1.0)
        )
        marker.text = "MEMORY %s [%s]\n%s" % (
            state,
            self.boundary.mode.value,
            detail,
        )
        self.marker_pub.publish(marker)
        self.publish_memory_markers()

    @staticmethod
    def marker_point(values):
        point = Point()
        point.x = float(values[0])
        point.y = float(values[1])
        point.z = 0.06
        return point

    def publish_memory_markers(self):
        markers = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        markers.markers.append(clear)
        stamp = rospy.Time.now()

        def base_marker(marker_id, namespace, marker_type, scale):
            marker = Marker()
            marker.header.stamp = stamp
            # Keep every historical primitive in its native continuous frame
            # and let RViz apply one frame-locked map<-odom transform.  Baking
            # a new transform into every point at 5 Hz made a SLAM correction
            # appear as a trail deformation or a one-frame offset.
            marker.header.frame_id = "odom"
            marker.frame_locked = True
            marker.ns = namespace
            marker.id = marker_id
            marker.type = marker_type
            marker.action = Marker.ADD
            marker.pose.orientation.w = 1.0
            marker.scale.x = scale
            marker.scale.y = scale
            marker.scale.z = scale
            marker.color.a = 0.95
            return marker

        if len(self.trail.points) >= 2:
            map_observations = [
                point
                for point in self.trail.points
                if point.map_x is not None and point.map_y is not None
            ]
            if len(map_observations) >= 2:
                # This is the user-facing driven history.  Every sample was
                # stamped in the SLAM map frame when observed, so later
                # map->odom corrections must not rigidly rotate the whole
                # displayed trail away from already mapped walls.
                marker = Marker()
                marker.header.stamp = stamp
                marker.header.frame_id = "map"
                marker.frame_locked = False
                marker.ns = "memory_breadcrumbs"
                marker.id = 1
                marker.type = Marker.LINE_STRIP
                marker.action = Marker.ADD
                marker.pose.orientation.w = 1.0
                marker.scale.x = 0.055
                marker.color.a = 0.98
                marker.color.r, marker.color.g, marker.color.b = 0.1, 0.45, 1.0
                marker.points = [
                    self.marker_point((point.map_x, point.map_y))
                    for point in map_observations
                ]
                markers.markers.append(marker)

                # Retain the native odom replay stack as a subdued diagnostic
                # channel.  It is the correct continuous control geometry and
                # is expected to move under a real SLAM correction.
                control = base_marker(
                    12, "memory_breadcrumbs_odom_control", Marker.LINE_STRIP, 0.025
                )
                control.color.a = 0.35
                control.color.r, control.color.g, control.color.b = 0.55, 0.65, 0.80
                control.points = [
                    self.marker_point((point.x, point.y))
                    for point in self.trail.points
                ]
                markers.markers.append(control)
            else:
                marker = base_marker(
                    1, "memory_breadcrumbs", Marker.LINE_STRIP, 0.045
                )
                marker.color.r, marker.color.g, marker.color.b = 0.1, 0.45, 1.0
                marker.points = [
                    self.marker_point((point.x, point.y))
                    for point in self.trail.points
                ]
                markers.markers.append(marker)

        open_edges = base_marker(2, "memory_topology", Marker.LINE_LIST, 0.055)
        open_edges.color.r, open_edges.color.g, open_edges.color.b = 0.1, 0.9, 0.25
        failed_edges = base_marker(3, "memory_topology", Marker.LINE_LIST, 0.085)
        failed_edges.color.r, failed_edges.color.g, failed_edges.color.b = 1.0, 0.15, 0.05
        for edge in self.topology.edges.values():
            first, second = self.topology.nodes.get(edge.first), self.topology.nodes.get(edge.second)
            if first is None or second is None:
                continue
            target = open_edges if edge.state == TopologyEdgeState.OPEN else failed_edges
            target.points.extend((
                self.marker_point((first.x, first.y)),
                self.marker_point((second.x, second.y)),
            ))
        if open_edges.points:
            markers.markers.append(open_edges)
        if failed_edges.points:
            markers.markers.append(failed_edges)

        if self.topology.nodes:
            nodes = base_marker(4, "memory_topology_nodes", Marker.SPHERE_LIST, 0.11)
            nodes.color.r, nodes.color.g, nodes.color.b = 0.25, 1.0, 0.85
            nodes.points = [
                self.marker_point((node.x, node.y))
                for node in self.topology.nodes.values()
            ]
            markers.markers.append(nodes)

        marker_id = 20
        for branch in self.topology.failed_branches:
            failed = base_marker(marker_id, "memory_failed_branches", Marker.LINE_STRIP, 0.12)
            failed.color.r, failed.color.g, failed.color.b = 1.0, 0.05, 0.05
            failed.points = [self.marker_point(point) for point in branch.points]
            markers.markers.append(failed)
            direction = base_marker(
                1000 + marker_id,
                "memory_failed_branch_ingress_direction",
                Marker.ARROW,
                0.08,
            )
            direction.scale.y = 0.18
            direction.scale.z = 0.22
            direction.color.r, direction.color.g, direction.color.b = (
                1.0, 0.15, 0.05
            )
            direction.points = [
                self.marker_point(branch.entry),
                self.marker_point(branch.terminal),
            ]
            markers.markers.append(direction)
            marker_id += 1

        if self.backtrack_site_index is not None and self.trail.points:
            index = min(
                max(0, self.backtrack_site_index), len(self.trail.points) - 1
            )
            site = self.trail.points[index]
            target = base_marker(
                10, "far_dead_end_egress_anchor", Marker.SPHERE, 0.28
            )
            target.color.r, target.color.g, target.color.b = 0.1, 1.0, 0.85
            target.pose.position = self.marker_point((site.x, site.y))
            markers.markers.append(target)

        if len(self.last_topology_path) >= 2:
            selected = base_marker(5, "memory_selected_topology", Marker.LINE_STRIP, 0.10)
            selected.color.r, selected.color.g, selected.color.b = 1.0, 0.85, 0.05
            selected.points = [
                self.marker_point(point) for point in self.last_topology_path
            ]
            markers.markers.append(selected)

        if self.resume_target_index is not None and self.trail.points:
            index = min(max(0, self.resume_target_index), len(self.trail.points) - 1)
            point = (
                self.trail.points[index].x,
                self.trail.points[index].y,
            )
            target = base_marker(6, "memory_recovery_target", Marker.SPHERE, 0.24)
            target.color.r, target.color.g, target.color.b = 1.0, 0.2, 1.0
            target.pose.position = self.marker_point(point)
            markers.markers.append(target)
        elif self.resume_fallback_target_xy is not None:
            point = self.resume_fallback_target_xy
            target = base_marker(6, "memory_recovery_target", Marker.SPHERE, 0.24)
            target.color.r, target.color.g, target.color.b = 1.0, 0.2, 1.0
            target.pose.position = self.marker_point(point)
            markers.markers.append(target)

        if self.boundary.active and self.boundary.hit_xy is not None:
            point = self.boundary.hit_xy
            hit = base_marker(7, "memory_boundary_hit", Marker.SPHERE, 0.22)
            hit.color.r, hit.color.g, hit.color.b = 1.0, 0.55, 0.0
            hit.pose.position = self.marker_point(point)
            markers.markers.append(hit)

        if self.topology_last_carrot_odom is not None:
            carrot = base_marker(
                8, "topology_rolling_carrot", Marker.SPHERE, 0.23
            )
            carrot.color.r, carrot.color.g, carrot.color.b = 1.0, 0.75, 0.0
            carrot.pose.position = self.marker_point(
                self.topology_last_carrot_odom
            )
            markers.markers.append(carrot)

        if self.visibility_last_carrot_map is not None:
            # The FAR route and this ephemeral carrot are native map-frame
            # geometry.  Unlike odom breadcrumbs they should follow the
            # latest SLAM correction rather than retain an old transform.
            carrot = Marker()
            carrot.header.stamp = stamp
            carrot.header.frame_id = "map"
            carrot.ns = "far_rolling_carrot"
            carrot.id = 9
            carrot.type = Marker.SPHERE
            carrot.action = Marker.ADD
            carrot.pose.orientation.w = 1.0
            carrot.pose.position = self.marker_point(
                self.visibility_last_carrot_map
            )
            carrot.scale.x = carrot.scale.y = carrot.scale.z = 0.25
            carrot.color.a = 0.98
            carrot.color.r, carrot.color.g, carrot.color.b = 0.2, 1.0, 0.95
            markers.markers.append(carrot)

        if len(self.dead_end_escape_last_route_map) >= 2:
            route = Marker()
            route.header.stamp = stamp
            route.header.frame_id = "map"
            route.frame_locked = True
            route.ns = "far_dead_end_egress_route"
            route.id = 11
            route.type = Marker.LINE_STRIP
            route.action = Marker.ADD
            route.pose.orientation.w = 1.0
            route.scale.x = 0.11
            route.color.a = 0.98
            route.color.r, route.color.g, route.color.b = 0.05, 0.95, 1.0
            route.points = [
                self.marker_point(point)
                for point in self.dead_end_escape_last_route_map
            ]
            markers.markers.append(route)

        self.memory_marker_pub.publish(markers)

    def progress(self, odom_pose, stamp):
        xy = np.asarray(odom_pose[:2], dtype=float)
        if self.last_progress_pose is None:
            output = 0.0
        else:
            output = float(np.linalg.norm(xy - self.last_progress_pose))
        self.last_progress_pose = xy
        self.last_progress_stamp = stamp
        return output

    def local_features(self, grid):
        values, resolution, origin = grid
        return local_space_features(values, resolution, origin)

    def record_breadcrumb(
        self,
        odom_pose,
        stamp,
        features,
        *,
        motion_direction=1,
        record_topology=True,
        force=False,
    ):
        recorded = self.trail.record(
            *odom_pose,
            stamp,
            junction=bool(features["junction"]),
            turnaround=bool(features["turnaround"]),
            motion_direction=int(motion_direction),
            map_pose=self.last_map_pose,
            map_revision=self.accumulated_map_revision,
            force=bool(force),
        )
        if (recorded and record_topology) or not self.topology.nodes:
            self.topology.record(
                *odom_pose,
                stamp,
                junction=bool(features["junction"]),
                turnaround=bool(features["turnaround"]),
                force=not self.topology.nodes,
            )

    def body_goal(self, map_pose, goal):
        dx = goal.pose.position.x - map_pose[0]
        dy = goal.pose.position.y - map_pose[1]
        cosine, sine = math.cos(map_pose[2]), math.sin(map_pose[2])
        return np.asarray((cosine * dx + sine * dy, -sine * dx + cosine * dy))

    def remember_authorized_far_direction(self, map_pose, target, stamp):
        """Remember a bounded map-frame direction, never an old metric path."""

        delta = np.asarray(target, dtype=float) - np.asarray(
            map_pose[:2], dtype=float
        )
        norm = float(np.linalg.norm(delta))
        if norm <= 1.0e-6:
            return
        self.visibility_last_authorized_direction_map = tuple(delta / norm)
        self.visibility_last_authorized_direction_stamp = float(stamp)

    def recent_far_direction_guidance(self, map_pose, stamp):
        """Return live local guidance through a brief FAR solver dropout.

        This deliberately stores no waypoint and grants no reverse/gear
        authority.  A direction older than the route-dropout grace is simply
        discarded; the normal sensor-closed-loop exploration policy then
        takes over while FAR continues replanning.
        """

        direction = self.visibility_last_authorized_direction_map
        age = float(stamp) - self.visibility_last_authorized_direction_stamp
        if direction is None or age < 0.0 or age > self.visibility_route_dropout_grace:
            return None
        dx, dy = float(direction[0]), float(direction[1])
        cosine, sine = math.cos(map_pose[2]), math.sin(map_pose[2])
        bearing = math.atan2(-sine * dx + cosine * dy, cosine * dx + sine * dy)
        # A transient global dropout is not permission to initiate a reverse
        # or turnaround transaction.  Clamp the remembered tangent to the
        # forward/lateral sector; hard veto still rejects an obstructed arc.
        bearing = min(1.45, max(-1.45, bearing))
        return np.asarray((
            math.cos(bearing) * self.route_horizon,
            math.sin(bearing) * self.route_horizon,
        ))

    @staticmethod
    def body_trajectory_to_frame(trajectory, pose):
        values = np.asarray(trajectory, dtype=float).copy()
        cosine, sine = math.cos(pose[2]), math.sin(pose[2])
        x, y = values[:, 1].copy(), values[:, 2].copy()
        values[:, 1] = pose[0] + cosine * x - sine * y
        values[:, 2] = pose[1] + sine * x + cosine * y
        values[:, 3] = np.asarray([wrap_angle(pose[2] + yaw) for yaw in values[:, 3]])
        return values

    def history_topology_guidance(
        self, map_pose, odom_pose, goal, stamp, *, allow_direct_goal=True
    ):
        goal_odom = self.map_to_odom(
            [(goal.pose.position.x, goal.pose.position.y)]
        )[0]
        current_xy = np.asarray(odom_pose[:2], dtype=float)
        failed_count = len(self.topology.failed_branches)
        if failed_count != self.committed_topology_failed_branches:
            self.topology_route_cursor.reset()
            self.committed_topology_path = []

        path = []
        if self.topology_route_cursor.active:
            path = self.topology_route_cursor.remaining_path(
                current_xy,
                maximum_deviation_m=self.topology_path_maximum_deviation,
            )
            observation = self.topology_route_cursor.last_observation
            if (
                observation is not None
                and observation.total_m - observation.progress_m
                <= self.topology_path_capture_radius
            ):
                path = []

        if len(path) < 2:
            candidate = self.topology.guidance_path(
                odom_pose[:2], goal_odom, stamp=stamp
            )
            if len(candidate) >= 2:
                route_id = self.route_identity(
                    "EXPLORED_TOPOLOGY",
                    candidate,
                    (round(goal_odom[0], 3), round(goal_odom[1], 3)),
                )
                self.topology_route_revision += 1
                self.topology_route_cursor.bind(
                    candidate,
                    current_xy,
                    route_id=route_id,
                    revision=self.topology_route_revision,
                    source="EXPLORED_TOPOLOGY",
                )
                path = self.topology_route_cursor.remaining_path(
                    current_xy,
                    maximum_deviation_m=self.topology_path_maximum_deviation,
                )
            else:
                self.topology_route_cursor.reset()
                path = []
            self.committed_topology_failed_branches = failed_count

        self.committed_topology_path = (
            [tuple(point) for point in self.topology_route_cursor.path]
            if self.topology_route_cursor.active
            else []
        )
        self.last_topology_path = list(path)
        if len(path) < 2 or not self.topology_route_cursor.active:
            if not allow_direct_goal:
                self.last_guidance_source = "NO_EXPLORED_TOPOLOGY"
                return None, goal_odom
            self.last_guidance_source = "DIRECT_GOAL"
            return np.asarray(
                self.body_goal(map_pose, goal), dtype=float
            ), goal_odom
        observation = self.topology_route_cursor.last_observation
        lookahead = np.asarray(observation.carrot_xy, dtype=float)
        self.topology_last_carrot_odom = tuple(lookahead)
        target_map = self.odom_to_map([lookahead])[0]
        dx, dy = target_map[0] - map_pose[0], target_map[1] - map_pose[1]
        cosine, sine = math.cos(map_pose[2]), math.sin(map_pose[2])
        body = np.asarray(
            (cosine * dx + sine * dy, -sine * dx + cosine * dy),
            dtype=float,
        )
        bearing = math.atan2(body[1], body[0])
        route_id = self.topology_route_cursor.route_id

        # Rear topology is connectivity evidence, not a repeatedly sampled
        # transmission command.  Once its multi-leg turnaround completes,
        # the same sparse route cannot launch another "first" reverse.  A
        # genuinely different route may acquire authority only after the car
        # has made measurable forward displacement from the previous site.
        same_consumed_route = bool(
            self.topology_last_turnaround_route_id
            and route_id == self.topology_last_turnaround_route_id
        )
        different_route_rearmed = bool(
            self.topology_last_turnaround_route_id
            and route_id != self.topology_last_turnaround_route_id
            and self.topology_turnaround_completion_xy is not None
            and np.linalg.norm(
                current_xy
                - np.asarray(self.topology_turnaround_completion_xy, dtype=float)
            ) >= self.topology_turnaround_rearm_progress
        )
        if different_route_rearmed:
            self.topology_rear_authority_issued = False
            self.topology_last_turnaround_route_id = ""
            self.topology_turnaround_completion_progress_m = None
            self.topology_turnaround_completion_xy = None
            same_consumed_route = False
        if (
            abs(bearing) >= self.topology_rear_authority_bearing
            and (
                same_consumed_route
                or (
                    self.topology_rear_authority_issued
                    and not self.route_turnaround_transaction_active
                )
            )
        ):
            self.last_guidance_source = "TOPOLOGY_REAR_SUPPRESSED"
            return None, goal_odom
        self.last_guidance_source = "EXPLORED_TOPOLOGY"
        return body, goal_odom

    @staticmethod
    def project_to_polyline(point, path):
        values = np.asarray(path, dtype=float)
        query = np.asarray(point, dtype=float)
        if len(values) == 0:
            return math.inf, 0, query
        if len(values) == 1:
            return float(np.linalg.norm(query - values[0])), 0, values[0]
        best = (math.inf, 0, values[0])
        for index, (first, second) in enumerate(zip(values[:-1], values[1:])):
            delta = second - first
            denominator = float(np.dot(delta, delta))
            ratio = (
                0.0
                if denominator <= 1.0e-12
                else min(1.0, max(0.0, float(np.dot(query - first, delta)) / denominator))
            )
            projection = first + ratio * delta
            distance = float(np.linalg.norm(query - projection))
            if distance < best[0]:
                best = (distance, index, projection)
        return best

    @staticmethod
    def point_along_polyline(path, distance_m):
        values = np.asarray(path, dtype=float)
        if len(values) == 0:
            return None
        remaining = max(0.0, float(distance_m))
        for first, second in zip(values[:-1], values[1:]):
            length = float(np.linalg.norm(second - first))
            if length <= 1.0e-9:
                continue
            if remaining <= length:
                return first + (remaining / length) * (second - first)
            remaining -= length
        return values[-1]

    @staticmethod
    def polyline_length(path):
        values = np.asarray(path, dtype=float)
        if len(values) < 2:
            return 0.0
        return float(np.linalg.norm(np.diff(values, axis=0), axis=1).sum())

    @staticmethod
    def route_identity(source, path, goal_key=None):
        """Return a stable identity without changing route coordinates."""

        values = np.asarray(path, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 2 or len(values) < 2:
            return "%s:invalid" % str(source).lower()
        quantized = np.round(values, 3)
        checksum = zlib.crc32(quantized.tobytes())
        if goal_key is not None:
            checksum = zlib.crc32(
                repr(tuple(goal_key)).encode("utf-8"), checksum
            )
        return "%s:%08x" % (str(source).lower(), checksum & 0xFFFFFFFF)

    def clear_connected_visibility_candidate(self, reason):
        """Clear only the process-local candidate, never mission/map state."""

        self.visibility_connected_candidate_plan = None
        self.visibility_connected_candidate_goal_key = None
        self.visibility_connected_candidate_revision = -1
        self.visibility_connected_candidate_discovered_stamp = -math.inf
        self.visibility_connected_candidate_status = "EMPTY"
        self.visibility_connected_candidate_reason = str(reason)
        self.visibility_connected_candidate_route_id = ""

    def offer_connected_visibility_candidate(
        self, plan, goal_key, revision, stamp
    ):
        """Escrow a complete current-goal FAR result before generic gating."""

        if not visibility_plan_is_goal_connected(plan):
            return False
        active_goal_route = bool(
            self.visibility_active_route_motion_authorized
            and visibility_plan_is_goal_connected(self.visibility_plan)
            and self.visibility_goal_key == goal_key
        )
        if active_goal_route:
            return False
        route_id = self.route_identity(
            "FAR_CONNECTED_CANDIDATE", plan.path, goal_key
        )
        is_new = route_id != self.visibility_connected_candidate_route_id
        self.visibility_connected_candidate_plan = plan
        self.visibility_connected_candidate_goal_key = goal_key
        self.visibility_connected_candidate_revision = int(revision)
        self.visibility_connected_candidate_discovered_stamp = float(stamp)
        self.visibility_connected_candidate_status = "PENDING_CONNECTED"
        self.visibility_connected_candidate_reason = (
            "dense_goal_connected_candidate_discovered"
        )
        self.visibility_connected_candidate_route_id = route_id
        if is_new:
            rospy.logwarn(
                "Escrowed goal-connected FAR candidate route_id=%s mode=%s "
                "points=%d length=%s revision=%d",
                route_id,
                plan.mode,
                len(plan.path),
                "none"
                if plan.path_length is None
                else "%.3f" % float(plan.path_length),
                int(revision),
            )
        return True

    def revalidate_connected_visibility_candidate(
        self, goal_key, current_xy, values, resolution, origin
    ):
        """Return a live-pose candidate without mutating its global geometry."""

        plan = self.visibility_connected_candidate_plan
        if plan is None:
            return None
        if self.visibility_connected_candidate_status not in (
            "PENDING_CONNECTED",
            "PENDING_CONNECTOR",
        ):
            return None
        if self.visibility_connected_candidate_goal_key != goal_key:
            self.clear_connected_visibility_candidate("mission_goal_changed")
            return None
        # A pending route is solved from a recent measured pose.  Reproject the
        # live pose to it and include that connector in every occupancy check;
        # the stored route itself remains unchanged for audit and monotonic
        # handoff purposes.
        connected_path = self.trim_visibility_path(
            current_xy,
            plan.path,
            maximum_deviation=self.visibility_maximum_deviation,
        )
        if len(connected_path) < 2:
            self.visibility_connected_candidate_status = "PENDING_CONNECTOR"
            self.visibility_connected_candidate_reason = (
                "measured_pose_outside_connected_route_connector"
            )
            return None
        prefix = polyline_prefix(
            connected_path,
            max(
                self.visibility_lookahead,
                self.rolling_route_maximum_carrot_distance,
            ),
        )
        prefix_clear = bool(
            len(prefix) >= 2
            and self.visibility.path_is_traversable(
                prefix,
                values,
                resolution,
                origin,
                maximum_unknown_fraction=0.02,
            )
        )
        if not prefix_clear:
            self.visibility_connected_candidate_status = "PENDING_CONNECTOR"
            self.visibility_connected_candidate_reason = (
                "connected_route_live_prefix_blocked"
            )
            return None
        if plan.mode == "KNOWN_VISIBILITY" and not self.visibility.path_is_traversable(
            connected_path,
            values,
            resolution,
            origin,
            maximum_unknown_fraction=0.02,
        ):
            # A fully observed route is invalidated only by real newest-map
            # occupancy, never by a weaker capped graph returning NO_ROUTE.
            self.visibility_connected_candidate_status = "INVALID_CONNECTED"
            self.visibility_connected_candidate_reason = (
                "new_occupied_crossing_on_connected_route"
            )
            self.visibility_connected_candidate_plan = None
            return None
        path_length = self.polyline_length(connected_path)
        self.visibility_connected_candidate_status = "PENDING_CONNECTED"
        self.visibility_connected_candidate_reason = (
            "connected_route_revalidated_from_measured_pose"
        )
        return replace(
            plan,
            path=tuple(connected_path),
            path_length=float(path_length),
        )

    def visibility_cursor_path(self, current_xy, *, maximum_deviation=None):
        if not self.visibility_route_cursor.active:
            return []
        limit = (
            self.visibility_maximum_deviation
            if maximum_deviation is None
            else float(maximum_deviation)
        )
        remaining = self.visibility_route_cursor.remaining_path(
            current_xy, maximum_deviation_m=limit
        )
        self.visibility_remaining_path_map = list(remaining)
        observation = self.visibility_route_cursor.last_observation
        self.visibility_last_carrot_map = (
            None if observation is None else observation.carrot_xy
        )
        return remaining

    def bind_visibility_route(self, path, current_xy, goal_key, source):
        self.visibility_route_revision += 1
        route_id = self.route_identity(source, path, goal_key)
        handoff = self.visibility_route_cursor.preview_handoff(
            path,
            current_xy,
            maximum_entry_deviation_m=self.visibility_maximum_deviation,
            maximum_direction_change_rad=(
                self.visibility_route_replacement_maximum_direction_change
            ),
        )
        initial_progress = (
            handoff.candidate_progress_m if handoff.accepted else None
        )
        self.visibility_route_cursor.bind(
            path,
            current_xy,
            route_id=route_id,
            revision=self.visibility_route_revision,
            source=str(source),
            initial_progress_m=initial_progress,
            initial_carrot_m=(
                handoff.candidate_carrot_m if handoff.accepted else None
            ),
        )
        self.visibility_path_map = [tuple(point) for point in path]
        self.visibility_last_handoff = handoff
        return self.visibility_cursor_path(current_xy)

    def reset_visibility_cursor(self):
        self.visibility_route_cursor.reset()
        self.visibility_path_map = []
        self.visibility_remaining_path_map = []
        self.visibility_last_carrot_map = None

    def trim_visibility_path(self, current_xy, path, *, maximum_deviation=None):
        values = np.asarray(path, dtype=float)
        if len(values) < 2:
            return []
        distance, segment, projection = self.project_to_polyline(current_xy, values)
        deviation_limit = (
            self.visibility_maximum_deviation
            if maximum_deviation is None
            else float(maximum_deviation)
        )
        if distance > deviation_limit:
            return []
        output = [tuple(np.asarray(current_xy, dtype=float)), tuple(projection)]
        output.extend(tuple(point) for point in values[segment + 1 :])
        deduplicated = [output[0]]
        for point in output[1:]:
            if math.hypot(
                point[0] - deduplicated[-1][0], point[1] - deduplicated[-1][1]
            ) > 1.0e-4:
                deduplicated.append(point)
        return deduplicated

    def publish_visibility_plan(self, plan):
        stamp = rospy.Time.now()
        path = Path()
        path.header.stamp = stamp
        path.header.frame_id = "map"
        for point in plan.path:
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x = float(point[0])
            pose.pose.position.y = float(point[1])
            pose.pose.orientation.w = 1.0
            path.poses.append(pose)
        self.visibility_path_pub.publish(path)

        markers = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        markers.markers.append(clear)

        edges = Marker()
        edges.header = path.header
        edges.ns = "far_visibility_edges"
        edges.id = 0
        edges.type = Marker.LINE_LIST
        edges.action = Marker.ADD
        edges.pose.orientation.w = 1.0
        edges.scale.x = 0.025
        edges.color.a = 0.30
        edges.color.r, edges.color.g, edges.color.b = 0.1, 0.85, 1.0
        by_id = {node.node_id: node for node in plan.nodes}
        for edge in plan.edges:
            for node_id in (edge.first, edge.second):
                node = by_id[node_id]
                edges.points.append(self.marker_point((node.x, node.y)))
        if edges.points:
            markers.markers.append(edges)

        selected = Marker()
        selected.header = path.header
        selected.ns = "far_visibility_selected"
        selected.id = 1
        selected.type = Marker.LINE_STRIP
        selected.action = Marker.ADD
        selected.pose.orientation.w = 1.0
        selected.scale.x = 0.11
        selected.color.a = 0.95
        if plan.mode == "KNOWN_VISIBILITY":
            selected.color.r, selected.color.g, selected.color.b = 0.1, 1.0, 0.3
        else:
            selected.color.r, selected.color.g, selected.color.b = 0.1, 0.5, 1.0
        selected.points = [self.marker_point(point) for point in plan.path]
        if selected.points:
            markers.markers.append(selected)
        self.visibility_marker_pub.publish(markers)

    def observe_initial_local_exploration(
        self, odom_pose, signed_speed_mps, stamp
    ):
        """Measure real current-goal motion before FAR receives authority.

        Distance is integrated in the continuous odom frame and only while the
        wheels report meaningful motion.  This prevents SLAM corrections or
        stationary pose noise from satisfying the exploration contract.
        """

        current = np.asarray(odom_pose[:2], dtype=float)
        if self.visibility_initial_exploration_started_stamp is None:
            self.visibility_initial_exploration_started_stamp = float(stamp)
        previous = self.visibility_initial_exploration_last_xy
        self.visibility_initial_exploration_last_xy = current.copy()
        if self.visibility_initial_exploration_complete or previous is None:
            return
        displacement = float(np.linalg.norm(current - previous))
        if abs(float(signed_speed_mps)) >= 0.03 and displacement <= 0.50:
            self.visibility_initial_exploration_distance_m += displacement
        if (
            self.visibility_initial_exploration_distance_m
            >= self.visibility_initial_exploration_minimum
        ):
            self.visibility_initial_exploration_complete = True
            self.visibility_mapping_session_established = True
            self.visibility_initial_exploration_reason = (
                "local_exploration_distance_reached"
            )
            rospy.loginfo(
                "Initial local exploration completed distance=%.3fm",
                self.visibility_initial_exploration_distance_m,
            )

    def far_bootstrap_motion_authorized(self, plan, acquisition, stamp):
        """Apply the local-first handoff contract to one stable FAR route."""

        if not acquisition.motion_authorized:
            return False
        if self.visibility_initial_exploration_complete:
            return True
        age = (
            0.0
            if self.visibility_initial_exploration_started_stamp is None
            else max(
                0.0,
                float(stamp)
                - float(self.visibility_initial_exploration_started_stamp),
            )
        )
        # A bounded fail-open prevents a short cul-de-sac in front of the car
        # from making the distance gate itself a liveness deadlock.  It applies
        # only to a fully known route which has already passed the distinct-map
        # stability gate; speculative/unknown routes never use the timeout.
        if (
            age >= self.visibility_initial_exploration_maximum_duration
            and acquisition.accepted
            and plan is not None
            and plan.mode == "KNOWN_VISIBILITY"
        ):
            self.visibility_initial_exploration_complete = True
            self.visibility_mapping_session_established = True
            self.visibility_initial_exploration_reason = (
                "stable_known_route_after_bounded_local_exploration"
            )
            rospy.loginfo(
                "Initial local exploration handed off to a stable known FAR "
                "route after %.3fs and %.3fm",
                age,
                self.visibility_initial_exploration_distance_m,
            )
            return True
        self.visibility_initial_exploration_reason = (
            "local_exploration_before_far_handoff"
        )
        return False

    def far_recovery_prefix_authorized(
        self,
        plan,
        map_pose,
        goal,
        stamp,
        *,
        require_failed_branch_avoidance=False,
    ):
        """Certify only the immediately executable prefix of a FAR detour.

        Global unknown-space coverage is useful evidence while choosing an
        ordinary route, but it must not create a liveness deadlock after the
        local swept-footprint authority has already proved that its current
        direction is blocked.  In that recovery case FAR supplies the
        topological side and only a fully observed, inflated-map-clear prefix
        gains motion authority.  DE-P and the live hard veto remain
        authoritative for every driven primitive.
        """

        if plan is None or plan.status != "PASS" or len(plan.path) < 2:
            return False
        with self.lock:
            accumulated = self.accumulated_grid
        if accumulated is None:
            return False
        values, resolution, origin, _ = accumulated
        prefix = polyline_prefix(
            plan.path,
            max(
                self.visibility_lookahead,
                self.rolling_route_maximum_carrot_distance,
            ),
        )
        if not (
            len(prefix) >= 2
            and self.visibility.path_is_traversable(
                prefix,
                values,
                resolution,
                origin,
                maximum_unknown_fraction=0.02,
            )
        ):
            return False
        if require_failed_branch_avoidance:
            goal_odom = self.map_to_odom([
                (goal.pose.position.x, goal.pose.position.y)
            ])[0]
            if self.topology.polyline_enters_failed_branch(
                self.map_to_odom(plan.path),
                goal_xy=goal_odom,
                stamp=float(stamp),
            ):
                return False
        return True

    def bind_far_recovery_candidate(self, plan, map_pose, goal, reason):
        """Atomically hand one locally certified FAR detour to DE-P."""

        goal_key = (
            round(float(goal.pose.position.x), 3),
            round(float(goal.pose.position.y), 3),
        )
        self.visibility_plan = plan
        self.visibility_goal_key = goal_key
        self.visibility_active_route_accepted = False
        self.visibility_active_route_motion_authorized = True
        self.visibility_route_validation_passed = True
        self.visibility_route_acquisition_reason = str(reason)
        self.visibility_route_dropout_started_stamp = None
        self.visibility_route_lease_active = False
        self.visibility_route_lease_reason = "static_recovery_far_handoff"
        self.visibility_route_lease_prefix_length_m = 0.0
        trimmed = self.bind_visibility_route(
            plan.path,
            map_pose[:2],
            goal_key,
            "FAR_STATIC_RECOVERY",
        )
        if len(trimmed) < 2:
            self.visibility_active_route_motion_authorized = False
            return False
        self.visibility_static_recovery_pending = False
        self.visibility_static_recovery_handoffs += 1
        self.visibility_static_replan_failures = 0
        self.visibility_static_blocked_since = None
        self.last_guidance_source = "FAR_ATTEMPTABLE_NAVIGATION"
        rospy.logwarn(
            "Promoted observed FAR recovery prefix after local hard block "
            "reason=%s points=%d unknown=%.3f handoff=%d",
            reason,
            len(plan.path),
            float(plan.path_unknown_fraction),
            self.visibility_static_recovery_handoffs,
        )
        return True

    def update_visibility_plan(self, map_pose, goal, stamp):
        with self.lock:
            grid = self.accumulated_grid
            revision = self.accumulated_map_revision
            observation_revision = self.accumulated_map_observation_revision
        if grid is None:
            return None, []
        values, resolution, origin, _ = grid
        goal_xy = (float(goal.pose.position.x), float(goal.pose.position.y))
        goal_key = (round(goal_xy[0], 3), round(goal_xy[1], 3))
        failed_count = len(self.topology.failed_branches)
        current_xy = map_pose[:2]
        dense_candidate_plan = self.take_dense_visibility_replan(
            goal_key,
            current_xy,
            revision,
            values,
            resolution,
            origin,
        )
        dense_goal_connected = visibility_plan_is_goal_connected(
            dense_candidate_plan
        )
        if dense_goal_connected:
            self.offer_connected_visibility_candidate(
                dense_candidate_plan,
                goal_key,
                revision,
                stamp,
            )
        connected_candidate_plan = (
            self.revalidate_connected_visibility_candidate(
                goal_key,
                current_xy,
                values,
                resolution,
                origin,
            )
        )
        if self.visibility_route_acquisition_started_stamp is None:
            self.visibility_route_acquisition_started_stamp = float(stamp)

        # Do not let online-map growth, a SLAM correction, or the intentional
        # reverse offset of a parking-style turn replace the route halfway
        # through that turn.  Local continuous hard safety remains live.
        if (
            self.visibility_maneuver_transaction_active
            and self.visibility_plan is not None
            and self.visibility_active_route_motion_authorized
        ):
            committed_path = (
                self.visibility_maneuver_path_map
                or self.visibility_path_map
                or list(self.visibility_plan.path)
            )
            committed = self.trim_visibility_path(
                current_xy,
                committed_path,
                maximum_deviation=math.inf,
            )
            if committed:
                return self.visibility_plan, committed

        trimmed = (
            self.visibility_cursor_path(current_xy)
            if self.visibility_route_cursor.active
            else self.trim_visibility_path(
                current_xy, self.visibility_path_map
            )
        )
        remaining_route_m = self.polyline_length(trimmed)
        plan_age = float(stamp) - float(self.visibility_last_plan_stamp)
        route_renewal_due = bool(
            self.visibility_plan is not None
            and self.visibility_plan.status == "PASS"
            and self.visibility_active_route_motion_authorized
            and remaining_route_m <= self.visibility_route_renewal_distance
            and plan_age >= self.visibility_route_renewal_period
        )
        map_route_invalidated = False
        map_revision_due = bool(
            revision != self.visibility_planned_revision
            and plan_age >= self.visibility_replan_period
        )
        if (
            map_revision_due
            and self.visibility_plan is not None
            and self.visibility_plan.status == "PASS"
            and len(trimmed) >= 2
        ):
            if self.visibility_route_validation_revision != revision:
                self.visibility_route_validation_passed = bool(
                    self.visibility.path_is_traversable(
                        trimmed, values, resolution, origin
                    )
                )
                self.visibility_route_validation_revision = revision
                self.visibility_route_validation_stamp = stamp
            map_route_invalidated = not self.visibility_route_validation_passed
        attemptable_refresh_due = bool(
            map_revision_due
            and self.visibility_plan is not None
            and self.visibility_plan.status == "PASS"
            and self.visibility_plan.mode in (
                "ATTEMPTABLE_VISIBILITY",
                "PARTIAL_ATTEMPTABLE",
            )
        )
        acquisition_refresh_due = bool(
            self.visibility_plan is not None
            and self.visibility_plan.status == "PASS"
            and not self.visibility_active_route_motion_authorized
            and plan_age >= self.visibility_replan_period
        )
        route_lease_refresh_due = bool(
            self.visibility_route_lease_active
            and plan_age >= self.visibility_replan_period
        )
        replan = bool(
            dense_candidate_plan is not None
            or
            connected_candidate_plan is not None
            or
            self.visibility_plan is None
            or self.visibility_goal_key != goal_key
            or self.visibility_planned_failed_branches != failed_count
            or (
                self.visibility_course_revalidation_pending
                and plan_age >= self.visibility_replan_period
            )
            or (
                not trimmed
                and (
                    self.visibility_active_route_motion_authorized
                    or plan_age >= self.visibility_replan_period
                )
            )
            or (
                map_revision_due
                and (
                    self.visibility_plan.status != "PASS"
                    or map_route_invalidated
                    or attemptable_refresh_due
                )
            )
            or route_renewal_due
            or acquisition_refresh_due
            # A lease is finite even if SLAM publishes byte-identical maps.
            # Keep asking FAR for a replacement and re-evaluating its age;
            # otherwise a stationary duplicate-map period could accidentally
            # turn the four-second bridge into permanent stale authority.
            or route_lease_refresh_due
        )
        if (
            replan
            and dense_candidate_plan is None
            and self.dense_visibility_replan_is_pending()
        ):
            # The dense worker owns the expensive graph expansion.  Keep the
            # timer callback responsive and leave any separately certified
            # active prefix untouched.  With no old route, topology_guidance
            # issues at most the forward-only online-mapping probe below.
            self.visibility_last_replan_reason = "dense_replan_pending"
            return self.visibility_plan, trimmed
        if replan:
            previous_plan = self.visibility_plan
            previous_path = list(trimmed)
            previous_goal_key = self.visibility_goal_key
            previous_route_accepted = bool(
                self.visibility_active_route_accepted
            )
            previous_route_motion_authorized = bool(
                self.visibility_active_route_motion_authorized
            )
            goal_odom = self.map_to_odom([goal_xy])[0]
            directed_failed_branches = []
            for branch in self.topology.failed_branches:
                if self.topology._branch_relevant(branch, goal_odom, stamp):
                    directed_failed_branches.append(
                        self.odom_to_map(branch.points)
                    )
            # Upstream FAR retains actual driven positions as trajectory /
            # inter-navigation vertices in its dynamic global graph.  Reuse
            # only that online odom topology, coherently reprojected into the
            # current SLAM frame; no map identity or old route is consulted.
            trajectory_points = self.visibility_trajectory_points_map()
            if dense_candidate_plan is not None:
                plan = dense_candidate_plan
                candidate_origin = "DENSE"
                self.visibility_last_plan_duration_ms = (
                    self.visibility_dense_replan_last_duration_ms
                )
                if (
                    not plan.progressive_complete
                    and plan.node_limit_hit
                    and not plan.start_inside_inflation
                    and not plan.goal_inside_inflation
                ):
                    # Resume the retained graph/edge transaction immediately.
                    # start_dense_visibility_replan validates the current
                    # request epoch, occupancy revision and measured start pose
                    # before handing the session back to the worker.
                    self.start_dense_visibility_replan(
                        values,
                        resolution,
                        origin,
                        current_xy,
                        goal_xy,
                        float(map_pose[2]),
                        revision,
                        directed_failed_branches,
                        trajectory_points,
                        stamp,
                    )
            else:
                planning_started = time.perf_counter()
                plan = self.visibility.plan(
                    values,
                    resolution,
                    origin,
                    current_xy,
                    goal_xy,
                    directed_failed_branches=directed_failed_branches,
                    trajectory_points=trajectory_points,
                    maximum_trajectory_vertices=(
                        self.visibility_trajectory_bridge_maximum_nodes
                    ),
                    failure_buffer_m=self.topology.failure_buffer_m,
                    start_yaw=float(map_pose[2]),
                )
                candidate_origin = "SPARSE"
                self.visibility_last_plan_duration_ms = 1000.0 * (
                    time.perf_counter() - planning_started
                )
                if (
                    plan.status in ("NO_ROUTE", "PASS")
                    and plan.node_limit_hit
                    and (
                        plan.status == "NO_ROUTE"
                        or plan.mode == "PARTIAL_ATTEMPTABLE"
                    )
                    and not plan.start_inside_inflation
                    and not plan.goal_inside_inflation
                ):
                    self.start_dense_visibility_replan(
                        values,
                        resolution,
                        origin,
                        current_xy,
                        goal_xy,
                        float(map_pose[2]),
                        revision,
                        directed_failed_branches,
                        trajectory_points,
                        stamp,
                    )
            raw_candidate_plan = plan
            connected_candidate_selected = False
            if connected_candidate_plan is not None and (
                dense_goal_connected
                or not visibility_plan_is_goal_connected(plan)
            ):
                # The dense result is a complete current-goal route.  Preserve
                # it across the gap before generic acquisition and across any
                # subsequent weaker sparse PARTIAL/NO_ROUTE rebuild.
                if not visibility_plan_is_goal_connected(raw_candidate_plan):
                    self.visibility_connected_candidate_suppressed_weaker += 1
                plan = connected_candidate_plan
                connected_candidate_selected = True
                candidate_origin = (
                    "DENSE_CONNECTED"
                    if dense_goal_connected
                    else "CONNECTED_ESCROW"
                )
            self.visibility_last_candidate_plan = plan
            self.visibility_goal_key = goal_key
            self.visibility_planned_revision = revision
            self.visibility_planned_failed_branches = failed_count
            self.visibility_last_plan_stamp = stamp
            acquisition = self.visibility_route_acquisition.update(
                plan,
                stamp=float(stamp),
                # Stability requires new occupancy content.  Replaying the
                # same cached grid cannot turn one speculative side choice
                # into independent evidence.
                map_revision=int(revision),
                # Once local exploration moves the sensor to a genuinely new
                # measured pose, the rebuilt visibility route is new
                # geometric evidence even when the OccupancyGrid bytes are
                # unchanged.  Stationary duplicate publications still do not
                # confirm themselves.
                observer_position=current_xy,
            )
            revalidation_prefix = polyline_prefix(
                plan.path if plan is not None else (),
                max(
                    self.visibility_lookahead,
                    self.rolling_route_maximum_carrot_distance,
                ),
            )
            revalidation_prefix_clear = bool(
                len(revalidation_prefix) >= 2
                and self.visibility.path_is_traversable(
                    revalidation_prefix,
                    values,
                    resolution,
                    origin,
                    maximum_unknown_fraction=0.02,
                )
            )
            partial_frontier_path_clear = bool(
                plan is not None
                and plan.status == "PASS"
                and plan.mode == "PARTIAL_ATTEMPTABLE"
                and len(plan.path) >= 2
                and self.visibility.path_is_traversable(
                    plan.path,
                    values,
                    resolution,
                    origin,
                    maximum_unknown_fraction=0.02,
                )
            )
            connected_candidate_available = bool(
                self.visibility_connected_candidate_plan is not None
                and self.visibility_connected_candidate_goal_key == goal_key
                and self.visibility_connected_candidate_status
                in ("PENDING_CONNECTED", "PENDING_CONNECTOR")
            )
            measured_pose_route_revalidation = bool(
                self.visibility_course_revalidation_pending
                and measured_pose_revalidation_authorized(
                    plan,
                    observed_prefix_clear=revalidation_prefix_clear,
                    maximum_attemptable_unknown_fraction=(
                        self.visibility_route_acquisition
                        .maximum_high_detour_unknown_fraction
                    ),
                )
            )
            static_recovery_motion_authorized = bool(
                self.visibility_static_recovery_pending
                and float(stamp) - self.visibility_last_local_static_block_stamp
                <= self.visibility_no_route_static_evidence_timeout
                and self.far_recovery_prefix_authorized(
                    plan,
                    map_pose,
                    goal,
                    stamp,
                )
            )
            bootstrap_motion_authorized = self.far_bootstrap_motion_authorized(
                plan, acquisition, stamp
            )
            partial_frontier_authority = partial_frontier_authority_reason(
                plan,
                path_clear=partial_frontier_path_clear,
                connected_candidate_available=connected_candidate_available,
                explicit_egress=bool(
                    self.visibility_static_recovery_pending
                    or self.recovery.state
                    == MemoryNavigationState.FAR_DEAD_END_EGRESS
                ),
                minimum_goal_progress_m=(
                    self.visibility_partial_minimum_goal_progress
                ),
                minimum_information_gain=(
                    self.visibility_partial_minimum_information_gain
                ),
                maximum_information_detour_m=(
                    self.visibility_partial_maximum_information_detour
                ),
            )
            partial_frontier_motion_authorized = bool(
                partial_frontier_authority is not None
            )
            connected_known_immediate_authorized = bool(
                connected_candidate_selected
                and plan is not None
                and plan.mode == "KNOWN_VISIBILITY"
                and float(plan.path_unknown_fraction) <= 0.02
                and revalidation_prefix_clear
                and self.visibility.path_is_traversable(
                    plan.path,
                    values,
                    resolution,
                    origin,
                    maximum_unknown_fraction=0.02,
                )
            )
            candidate_motion_authorized = (
                bootstrap_motion_authorized
                and locally_certified_route_motion(
                    plan,
                    acquisition,
                    revalidation_prefix_clear,
                )
                # During an inter-leg revalidation the vehicle is stationary,
                # so requiring a second distinct SLAM-content revision can
                # never reveal more route evidence.  FAR has rebuilt the path
                # from the measured pose; only its fully observed, inflated-
                # occupancy-clear execution prefix is authorized here, while
                # DE-P/hard-veto still certifies every primitive.  Keep that
                # route under FAR authority so its rolling carrot is handed
                # back immediately and the multi-leg turn can continue.
                or measured_pose_route_revalidation
                # A dense complete route is already a solution on the current
                # immutable occupancy snapshot.  Once its live connector,
                # complete known geometry and local prefix are revalidated,
                # waiting for an identical second map publication adds no
                # safety evidence and lets the next capped PARTIAL erase the
                # only global detour.  Promote this current-goal transaction
                # atomically; DE-P/hard-veto still checks every primitive.
                or connected_known_immediate_authorized
                # A partial FAR path is not a claim that the distant goal is
                # connected.  It is a bounded path inside START's current
                # visibility component, and the complete execution prefix was
                # certified on the live inflated map by the dense planner and
                # again above.  It can therefore move the sensor to a useful
                # frontier while denser expansion continues in the background.
                or partial_frontier_motion_authorized
                # Local exploration has already been rejected by continuous
                # swept-footprint safety.  A FAR detour with a completely
                # observed, inflated-map-clear execution prefix must now own
                # the recovery direction even when the *remote* portion of
                # that detour is still unknown/high-cost.
                or static_recovery_motion_authorized
            )
            # A parking-style turn may deliberately carry the vehicle farther
            # from the old polyline than the ordinary tracking-deviation
            # threshold.  Recover its monotonic suffix without allowing route
            # progress to roll backward, then validate only the next bounded
            # execution prefix on the newest occupancy map.  A connector from
            # the measured pose to that suffix is included in this check.
            previous_lease_path = list(previous_path)
            if (
                previous_route_motion_authorized
                and previous_plan is not None
                and previous_plan.status == "PASS"
            ):
                if self.visibility_route_cursor.active:
                    relaxed = self.visibility_cursor_path(
                        current_xy, maximum_deviation=math.inf
                    )
                else:
                    relaxed = self.trim_visibility_path(
                        current_xy,
                        self.visibility_path_map or previous_plan.path,
                        maximum_deviation=math.inf,
                    )
                if len(relaxed) >= 2:
                    previous_lease_path = list(relaxed)
            lease_prefix = polyline_prefix(
                previous_lease_path,
                self.visibility_route_lease_prefix,
            )
            lease_prefix_length = self.polyline_length(lease_prefix)
            lease_prefix_clear = bool(
                len(lease_prefix) >= 2
                and self.visibility.path_is_traversable(
                    lease_prefix,
                    values,
                    resolution,
                    origin,
                )
            )
            previous_route_safe = bool(
                previous_route_motion_authorized
                and previous_plan is not None
                and previous_plan.status == "PASS"
                and len(previous_path) >= 2
                and self.polyline_length(previous_path)
                >= self.visibility_route_continuity_minimum
                and self.visibility.path_is_traversable(
                    previous_path, values, resolution, origin
                )
            )
            previous_goal_connected = visibility_plan_is_goal_connected(
                previous_plan
            )
            handoff = (
                self.visibility_route_cursor.preview_handoff(
                    plan.path,
                    current_xy,
                    maximum_entry_deviation_m=(
                        self.visibility_maximum_deviation
                    ),
                    maximum_direction_change_rad=(
                        self.visibility_route_replacement_maximum_direction_change
                    ),
                )
                if self.visibility_route_cursor.active
                and plan.status == "PASS"
                and len(plan.path) >= 2
                else None
            )
            self.visibility_last_handoff = handoff
            known_continuous_far_renewal = bool(
                previous_route_motion_authorized
                and plan is not None
                and plan.status == "PASS"
                and plan.mode == "KNOWN_VISIBILITY"
                and len(plan.path) >= 2
                and float(plan.path_unknown_fraction) <= 0.02
                and handoff is not None
                and handoff.accepted
            )
            if known_continuous_far_renewal:
                # A fully observed, direction-continuous rolling replacement
                # must not drop a previously authorized FAR mission back to
                # topology/local terminal capture merely because its renewed
                # cost changed enough to restart the generic acquisition gate.
                candidate_motion_authorized = True
            replacement_direction_change = (
                None if handoff is None else handoff.direction_change_rad
            )
            replacement_direction_discontinuous = bool(
                handoff is not None
                and goal_route_direction_continuity_hold(
                    previous_plan=previous_plan,
                    previous_route_safe=previous_route_safe,
                    lease_prefix_clear=lease_prefix_clear,
                    handoff_accepted=handoff.accepted,
                )
            )
            incumbent_retention_reason = (
                goal_connected_incumbent_retention_reason(
                    previous_route_motion_authorized=(
                        previous_route_motion_authorized
                    ),
                    same_goal=previous_goal_key == goal_key,
                    previous_mode=(
                        "NONE" if previous_plan is None else previous_plan.mode
                    ),
                    previous_route_globally_traversable=previous_route_safe,
                    candidate_mode=("NONE" if plan is None else plan.mode),
                    candidate_motion_authorized=candidate_motion_authorized,
                    handoff_accepted=bool(
                        handoff is not None and handoff.accepted
                    ),
                )
            )
            replacement_ready = bool(
                candidate_motion_authorized
                and not replacement_direction_discontinuous
                and incumbent_retention_reason is None
            )
            if incumbent_retention_reason is not None:
                # Persistent-path momentum: this full goal route remains valid
                # on the newest map.  A partial, unsettled or direction-
                # discontinuous rebuild remains diagnostic and may keep
                # replanning, but cannot change route ID, cursor or authority.
                self.visibility_route_dropout_started_stamp = None
                self.visibility_route_lease_active = False
                self.visibility_route_lease_reason = (
                    incumbent_retention_reason
                )
                self.visibility_route_lease_prefix_length_m = float(
                    lease_prefix_length
                )
            elif replacement_ready:
                self.visibility_route_dropout_started_stamp = None
                self.visibility_route_lease_active = False
                self.visibility_route_lease_reason = "replacement_ready"
                self.visibility_route_lease_prefix_length_m = 0.0
            elif (
                previous_route_motion_authorized
                and previous_plan is not None
                and previous_plan.status == "PASS"
                and previous_goal_key == goal_key
            ):
                if self.visibility_route_dropout_started_stamp is None:
                    self.visibility_route_dropout_started_stamp = float(stamp)
                dropout_age = max(
                    0.0,
                    float(stamp) - self.visibility_route_dropout_started_stamp,
                )
                self.visibility_route_lease_active = (
                    transient_route_lease_authorized(
                        previous_route_motion_authorized=(
                            previous_route_motion_authorized
                        ),
                        same_goal=previous_goal_key == goal_key,
                        local_prefix_clear=lease_prefix_clear,
                        local_prefix_length_m=lease_prefix_length,
                        minimum_prefix_length_m=(
                            self.visibility_route_continuity_minimum
                        ),
                        dropout_age_s=dropout_age,
                        grace_s=self.visibility_route_dropout_grace,
                    )
                )
                self.visibility_route_lease_prefix_length_m = float(
                    lease_prefix_length
                )
                self.visibility_route_lease_reason = (
                    "safe_local_prefix_during_candidate_dropout"
                    if self.visibility_route_lease_active
                    else "unsafe_or_expired_local_prefix"
                )
            else:
                self.visibility_route_dropout_started_stamp = None
                self.visibility_route_lease_active = False
                self.visibility_route_lease_reason = "no_previous_authority"
                self.visibility_route_lease_prefix_length_m = 0.0
            retain_active = bool(
                incumbent_retention_reason is not None
                or
                (
                    previous_route_safe
                    and (
                        previous_goal_connected
                        or not connected_candidate_selected
                    )
                    and (
                        not candidate_motion_authorized
                        or replacement_direction_discontinuous
                    )
                )
                or self.visibility_route_lease_active
            )
            self.visibility_last_candidate_retained_active = retain_active
            if retain_active:
                # Planning is rolling: a transient disconnected or still-
                # settling result is a candidate failure, not permission to
                # erase a separately certified, still traversable suffix.
                self.visibility_plan = previous_plan
                self.visibility_active_route_accepted = previous_route_accepted
                self.visibility_active_route_motion_authorized = True
                self.visibility_route_validation_passed = True
                self.visibility_route_acquisition_reason = (
                    incumbent_retention_reason
                    if incumbent_retention_reason is not None
                    else "retaining_leased_active_route_safe_local_prefix"
                    if self.visibility_route_lease_active
                    and not previous_route_safe
                    else "retaining_safe_active_route_direction_continuity"
                    if replacement_direction_discontinuous
                    else "retaining_safe_active_route_" + acquisition.reason
                )
                trimmed = (
                    previous_lease_path
                    if self.visibility_route_lease_active
                    else self.visibility_cursor_path(current_xy)
                )
            else:
                self.visibility_plan = plan
                self.visibility_active_route_accepted = bool(
                    acquisition.accepted
                    or connected_known_immediate_authorized
                )
                self.visibility_active_route_motion_authorized = bool(
                    candidate_motion_authorized
                )
                self.visibility_route_validation_passed = bool(
                    plan.status == "PASS"
                )
                self.visibility_route_acquisition_reason = (
                    "observed_far_detour_after_local_hard_block"
                    if static_recovery_motion_authorized
                    else "connected_known_route_transaction_promoted"
                    if connected_known_immediate_authorized
                    else partial_frontier_authority
                    if partial_frontier_motion_authorized
                    else "measured_pose_observed_prefix_far_revalidation"
                    if measured_pose_route_revalidation
                    else "known_continuous_far_renewal"
                    if known_continuous_far_renewal
                    else acquisition.reason
                    if candidate_motion_authorized
                    or not acquisition.motion_authorized
                    else self.visibility_initial_exploration_reason
                )
                if (
                    plan.status == "PASS"
                    and len(plan.path) >= 2
                    and candidate_motion_authorized
                ):
                    trimmed = self.bind_visibility_route(
                        plan.path,
                        current_xy,
                        goal_key,
                        "FAR_%s" % plan.mode,
                    )
                    if connected_candidate_selected:
                        self.visibility_connected_candidate_status = (
                            "ACTIVE_CONNECTED"
                        )
                        self.visibility_connected_candidate_reason = (
                            self.visibility_route_acquisition_reason
                        )
                        self.visibility_connected_candidate_promotions += 1
                else:
                    self.reset_visibility_cursor()
                    self.visibility_path_map = list(plan.path)
                    trimmed = self.trim_visibility_path(
                        current_xy, plan.path
                    )
            if (
                static_recovery_motion_authorized
                and self.visibility_active_route_motion_authorized
                and self.visibility_plan is plan
            ):
                self.visibility_static_recovery_pending = False
                self.visibility_static_recovery_handoffs += 1
                self.visibility_static_replan_failures = 0
                self.visibility_static_blocked_since = None
                self.visibility_initial_exploration_complete = True
                self.visibility_mapping_session_established = True
                self.visibility_initial_exploration_reason = (
                    "local_hard_block_far_recovery_handoff"
                )
                rospy.logwarn(
                    "Promoted observed FAR detour after local hard block "
                    "points=%d unknown=%.3f handoff=%d",
                    len(plan.path),
                    float(plan.path_unknown_fraction),
                    self.visibility_static_recovery_handoffs,
                )
            if (
                not self.visibility_active_route_motion_authorized
                and self.visibility_plan is not None
                and self.visibility_plan.status == "NO_ROUTE"
            ):
                if self.visibility_no_route_since is None:
                    self.visibility_no_route_since = float(stamp)
            else:
                self.visibility_no_route_since = None
            self.visibility_route_validation_revision = revision
            self.visibility_route_validation_stamp = stamp
            rospy.loginfo(
                "FAR route candidate status=%s mode=%s accepted=%s "
                "motion_authorized=%s gate_motion_authorized=%s "
                "confirmations=%d reason=%s planning_ms=%.1f "
                "points=%d cost=%s length=%s "
                "unknown=%.3f retained_active=%s lease_active=%s "
                "lease_prefix=%.3f direction_change=%s incumbent_hold=%s "
                "active_points=%d "
                "graph_nodes=%d/%d trajectory_nodes=%d/%d cap_hit=%s start_degree=%d "
                "goal_degree=%d components=%d disconnect=%s stages=%d "
                "progressive_complete=%s partial_progress=%s",
                plan.status,
                plan.mode,
                acquisition.accepted,
                candidate_motion_authorized,
                acquisition.motion_authorized,
                acquisition.confirmations,
                acquisition.reason,
                float(self.visibility_last_plan_duration_ms or 0.0),
                len(plan.path),
                "none" if plan.path_cost is None else "%.3f" % plan.path_cost,
                "none" if plan.path_length is None else "%.3f" % plan.path_length,
                float(plan.path_unknown_fraction),
                retain_active,
                self.visibility_route_lease_active,
                float(self.visibility_route_lease_prefix_length_m),
                (
                    "none"
                    if replacement_direction_change is None
                    else "%.3f" % replacement_direction_change
                ),
                (
                    "none"
                    if incumbent_retention_reason is None
                    else incumbent_retention_reason
                ),
                len(self.visibility_path_map),
                int(plan.candidate_vertices_selected),
                int(plan.candidate_vertices_total),
                int(plan.trajectory_vertices_selected),
                int(plan.trajectory_vertices_total),
                bool(plan.node_limit_hit),
                int(plan.start_degree),
                int(plan.goal_degree),
                int(plan.connected_components),
                plan.disconnect_class,
                int(plan.progressive_stages),
                bool(plan.progressive_complete),
                (
                    "none"
                    if plan.partial_goal_progress_m is None
                    else "%.3f" % plan.partial_goal_progress_m
                ),
            )
            rospy.loginfo(
                "P6 V4.3.1 connected route transaction status=%s reason=%s "
                "route_id=%s candidate_origin=%s selected=%s "
                "suppressed_weaker=%d promotions=%d",
                self.visibility_connected_candidate_status,
                self.visibility_connected_candidate_reason,
                self.visibility_connected_candidate_route_id or "none",
                candidate_origin,
                connected_candidate_selected,
                int(self.visibility_connected_candidate_suppressed_weaker),
                int(self.visibility_connected_candidate_promotions),
            )
            if map_route_invalidated:
                self.visibility_last_replan_reason = "new_occupied_route_crossing"
            elif route_renewal_due:
                self.visibility_last_replan_reason = "rolling_route_renewal"
            elif attemptable_refresh_due:
                self.visibility_last_replan_reason = "attemptable_map_refresh"
            elif acquisition_refresh_due:
                self.visibility_last_replan_reason = "route_acquisition_refresh"
            elif route_lease_refresh_due:
                self.visibility_last_replan_reason = "transient_route_lease_refresh"
            elif not previous_path:
                self.visibility_last_replan_reason = "route_exhausted"
            else:
                self.visibility_last_replan_reason = "route_contract_change"
            self.visibility_replan_count += 1
            # Visual route authority follows the route actually handed to
            # DE-P.  The rejected candidate remains available in status JSON.
            self.publish_visibility_plan(self.visibility_plan)
        elif (
            self.visibility_plan is not None
            and self.visibility_plan.status == "PASS"
            and len(trimmed) >= 2
            and not self.visibility_active_route_motion_authorized
            and self.visibility_route_acquisition.motion_authorized
        ):
            # Route evidence is unchanged, but real local motion may just have
            # completed the bootstrap distance.  Promote the already-stable
            # candidate without manufacturing another confirmation from an
            # identical OccupancyGrid publication.
            acquisition = self.visibility_route_acquisition
            motion_authorized = self.far_bootstrap_motion_authorized(
                self.visibility_plan, acquisition, stamp
            )
            self.visibility_active_route_accepted = bool(
                acquisition.accepted
            )
            self.visibility_active_route_motion_authorized = bool(
                motion_authorized
            )
            self.visibility_route_acquisition_reason = (
                acquisition.reason
                if motion_authorized
                else self.visibility_initial_exploration_reason
            )
            if motion_authorized and not self.visibility_route_cursor.active:
                trimmed = self.bind_visibility_route(
                    self.visibility_plan.path,
                    current_xy,
                    goal_key,
                    "FAR_%s" % self.visibility_plan.mode,
                )
            rospy.loginfo(
                "FAR stable candidate bootstrap promotion accepted=%s "
                "motion_authorized=%s confirmations=%d reason=%s "
                "content_revision=%d exploration=%.3fm",
                acquisition.accepted,
                motion_authorized,
                acquisition.confirmations,
                self.visibility_route_acquisition_reason,
                int(revision),
                self.visibility_initial_exploration_distance_m,
            )
        return self.visibility_plan, trimmed

    def topology_guidance(self, map_pose, odom_pose, goal, stamp):
        plan, path = self.update_visibility_plan(map_pose, goal, stamp)
        goal_odom = self.map_to_odom(
            [(goal.pose.position.x, goal.pose.position.y)]
        )[0]
        if self.failed_branch_exit_lock_branch_id is not None:
            exit_guidance = self.failed_branch_exit_lock_guidance(
                map_pose, odom_pose, goal_odom, plan, path, stamp
            )
            if exit_guidance is not None:
                return exit_guidance, goal_odom
        if (
            plan is not None
            and plan.status == "PASS"
            and len(path) >= 2
            and self.visibility_active_route_motion_authorized
        ):
            # FAR's attemptable route is itself the mechanism which reveals
            # the next view.  Do not stop after a fixed pulse and wait for
            # global unknown-space statistics to improve: behind a wall they
            # cannot improve without moving to the selected contour corner.
            # Immediate motion remains certified by DE-P, the live LiDAR BEV
            # and the swept-footprint hard veto.
            observation = self.visibility_route_cursor.last_observation
            target = (
                np.asarray(observation.carrot_xy, dtype=float)
                if observation is not None
                else self.point_along_polyline(path, self.visibility_lookahead)
            )
            dx, dy = target[0] - map_pose[0], target[1] - map_pose[1]
            cosine, sine = math.cos(map_pose[2]), math.sin(map_pose[2])
            self.last_guidance_source = (
                "FAR_KNOWN_VISIBILITY"
                if self.visibility_active_route_accepted
                and plan.mode == "KNOWN_VISIBILITY"
                else "FAR_PARTIAL_ATTEMPTABLE"
                if plan.mode == "PARTIAL_ATTEMPTABLE"
                else (
                    "FAR_ATTEMPTABLE_VISIBILITY"
                    if self.visibility_active_route_accepted
                    else "FAR_ATTEMPTABLE_NAVIGATION"
                )
            )
            self.remember_authorized_far_direction(map_pose, target, stamp)
            return np.asarray((cosine * dx + sine * dy, -sine * dx + cosine * dy)), goal_odom

        # Close to the goal, contour inflation can place the START or GOAL
        # graph vertex on an occupied quantisation halo and transiently make
        # FAR report NO_ROUTE.  Permit local arrival only when the complete
        # map-frame segment is already observed and footprint-clear.  This is
        # not the old direct-goal fallback: a wall or unknown cell keeps the
        # vehicle stopped and cannot be bypassed by Euclidean proximity.
        goal_xy = (float(goal.pose.position.x), float(goal.pose.position.y))
        goal_distance = math.hypot(
            goal_xy[0] - map_pose[0], goal_xy[1] - map_pose[1]
        )
        if goal_distance <= self.visibility_terminal_direct_handoff_radius:
            with self.lock:
                accumulated_grid = self.accumulated_grid
            if accumulated_grid is not None:
                values, resolution, origin, _ = accumulated_grid
                terminal_clear = self.visibility.path_is_traversable(
                    (map_pose[:2], goal_xy),
                    values,
                    resolution,
                    origin,
                    maximum_unknown_fraction=0.02,
                )
                if terminal_clear:
                    self.last_guidance_source = "KNOWN_TERMINAL_DIRECT"
                    return np.asarray(
                        self.body_goal(map_pose, goal), dtype=float
                    ), goal_odom
        if not self.visibility_initial_exploration_complete:
            # FAR keeps planning in the background, but the first viewpoint
            # change belongs to the local planner.  This also guarantees that
            # an unreliable initial graph cannot pull the car down a remote
            # branch simply because the same sparse map was republished.
            goal_body = np.asarray(self.body_goal(map_pose, goal), dtype=float)
            goal_norm = float(np.linalg.norm(goal_body))
            if goal_norm > 1.0e-6:
                self.visibility_route_acquisition_reason = (
                    self.visibility_initial_exploration_reason
                )
                self.last_guidance_source = "LOCAL_SAFE_EXPLORATION"
                return (
                    goal_body * min(1.0, self.route_horizon / goal_norm),
                    goal_odom,
                )
        # An authorized FAR route may disappear for one solve while an
        # obstacle contour is rebuilt.  Preserve only its recent map-frame
        # tangent, not its old polyline and not a drifting odom topology path.
        # The resulting candidate is regenerated from live sensors at every
        # tick, so this bridge cannot accumulate metric trajectory error or
        # acquire reverse/turnaround authority.
        recent_far_guidance = self.recent_far_direction_guidance(
            map_pose, stamp
        )
        if recent_far_guidance is not None:
            self.topology_route_cursor.reset()
            self.committed_topology_path = []
            self.last_topology_path = []
            self.visibility_route_acquisition_reason = (
                "recent_far_authority_dropout_local_continuation"
            )
            self.last_guidance_source = "LOCAL_SAFE_EXPLORATION"
            return recent_far_guidance, goal_odom

        if self.dense_visibility_replan_is_pending():
            # The one-metre initial observation transaction above is the only
            # new motion authorized solely to help this background solve.  Do
            # not manufacture repeated local turns or reversals while graph
            # expansion is still pending.
            self.visibility_route_acquisition_reason = (
                "background_dense_visibility_expansion"
            )
            self.last_guidance_source = "FAR_DENSE_REPLAN_PENDING"
            return np.zeros(2, dtype=float), goal_odom

        # Sparse topology remains useful as semantic evidence for FAR and
        # failed-branch suppression, but its accumulated odom geometry is not
        # a trustworthy long-horizon driving trajectory.  Keep the former
        # behaviour behind an explicit legacy opt-in only.
        if plan is None or plan.status != "PASS":
            self.reset_visibility_cursor()
        if self.explored_topology_motion_authority:
            history_guidance, goal_odom = self.history_topology_guidance(
                map_pose,
                odom_pose,
                goal,
                stamp,
                allow_direct_goal=False,
            )
            if history_guidance is not None:
                return history_guidance, goal_odom
        else:
            self.topology_route_cursor.reset()
            self.committed_topology_path = []
            self.last_topology_path = []
            self.topology_last_carrot_odom = None
            self.last_guidance_source = "TOPOLOGY_MOTION_AUTHORITY_DISABLED"
        if self.last_guidance_source == "TOPOLOGY_REAR_SUPPRESSED":
            # Do not translate a stale rear topology node into another gear
            # reversal.  Keep mapping with a forward/side, goal-biased local
            # heading while FAR rebuilds the graph from the new viewpoint.
            # The reactive selector and hard veto below still own collision
            # safety; this vector is direction guidance only.
            goal_body = np.asarray(self.body_goal(map_pose, goal), dtype=float)
            goal_bearing = math.atan2(goal_body[1], goal_body[0])
            exploratory_bearing = min(1.45, max(-1.45, goal_bearing))
            self.visibility_route_acquisition_reason = (
                "consumed_topology_rear_route_rolling_local_exploration"
            )
            self.last_guidance_source = "LOCAL_SAFE_EXPLORATION"
            return np.asarray((
                math.cos(exploratory_bearing) * self.route_horizon,
                math.sin(exploratory_bearing) * self.route_horizon,
            )), goal_odom

        # FAR can legitimately return NO_ROUTE while the graph has not yet
        # reached around an occluding wall.  Stopping until a known-cell count
        # increases is a liveness deadlock: a stationary LiDAR cannot acquire
        # a new viewpoint.  Continue with rolling local exploration instead.
        # This is not blind Euclidean driving: the reactive selector below
        # evaluates the full five-circle swept body on the live LiDAR grid,
        # persistent known walls and failed-branch memory; the DE-P hard veto
        # remains the final authority.  A blocked local field therefore enters
        # static recovery, while a merely disconnected global graph keeps
        # moving and lets SLAM/FAR recover online.
        goal_body = np.asarray(self.body_goal(map_pose, goal), dtype=float)
        goal_norm = float(np.linalg.norm(goal_body))
        if goal_norm > 1.0e-6:
            self.visibility_route_acquisition_reason = (
                "no_route_rolling_local_exploration"
                if plan is None or plan.status != "PASS"
                else "unconfirmed_route_rolling_local_exploration"
            )
            self.last_guidance_source = "LOCAL_SAFE_EXPLORATION"
            return (
                goal_body * min(1.0, self.route_horizon / goal_norm),
                goal_odom,
            )

        self.last_guidance_source = "FAR_ACQUISITION_HOLD"
        return np.zeros(2, dtype=float), goal_odom

    @staticmethod
    def safe_length_from_profile(profile, horizon):
        blocked = np.flatnonzero(np.asarray(profile) <= 0.0)
        if not len(blocked):
            return float(horizon)
        return float(horizon) * max(0, int(blocked[0]) - 1) / max(1, len(profile) - 1)

    def choose_local_heading(self, goal, grid, map_pose, odom_pose, stamp):
        goal_body = np.asarray(self.body_goal(map_pose, goal), dtype=float)
        self.last_mission_goal_bearing = math.atan2(goal_body[1], goal_body[0])
        guidance_body, goal_odom = self.topology_guidance(
            map_pose, odom_pose, goal, stamp
        )
        if self.last_guidance_source in (
            "FAR_ACQUISITION_HOLD",
            "FAR_DENSE_REPLAN_PENDING",
        ):
            self.last_heading_decision = None
            self.last_boundary_decision = None
            return None, 0.0
        desired = math.atan2(guidance_body[1], guidance_body[0])
        distance = max(float(np.linalg.norm(guidance_body)), self.route_minimum)
        if (
            self.visibility_route_direction_authority
            and self.last_guidance_source in (
                "FAR_KNOWN_VISIBILITY",
                "FAR_ATTEMPTABLE_VISIBILITY",
                "FAR_ATTEMPTABLE_NAVIGATION",
                "FAR_PARTIAL_ATTEMPTABLE",
                "EXPLORED_TOPOLOGY",
            )
        ):
            # FAR or a physically driven topology corridor already selected
            # the topological side of the obstacle.  The Bug/TangentBug layer
            # remains a fallback only when neither route exists; it must not
            # pull a half-finished turnaround away from a certified corridor.
            # DE-P keeps full trajectory freedom inside the route tube and all
            # candidates remain subject to continuous hard safety vetoes.
            self.boundary.reset(clear_failures=False)
            self.last_boundary_decision = self.boundary.update(
                stamp,
                odom_pose[:2],
                float(np.linalg.norm(goal_body)),
                self.boundary.release_clearance_m,
            )
            decision = ReactiveHeadingDecision(
                angle=float(desired),
                clearance=float(self.route_horizon),
                goal_bearing=float(desired),
                sector=(
                    "EXPLORED_TOPOLOGY_ROUTE"
                    if self.last_guidance_source == "EXPLORED_TOPOLOGY"
                    else "FAR_ROUTE"
                ),
                blocked=False,
            )
            self.previous_goal_heading = wrap_angle(map_pose[2] + desired)
            self.last_heading_decision = decision
            return desired, min(distance, self.route_horizon)

        values, resolution, origin = grid
        candidates = [wrap_angle(desired + offset) for offset in np.linspace(-1.2, 1.2, 25)]
        # Sample the complete forward/side field too.  This is a local
        # free-sector selector, not a graph search over the SLAM map.
        candidates.extend(np.linspace(-1.65, 1.65, 35))
        candidates.extend((0.0, math.pi / 2.0, -math.pi / 2.0, math.pi))
        occupancy = RuntimeOccupancyGrid2D(values, resolution, origin)
        angle_clearances = []
        angle_utilities = []
        seen = set()
        for raw_angle in candidates:
            angle = wrap_angle(raw_angle)
            key = round(angle, 6)
            if key in seen:
                continue
            seen.add(key)
            if abs(angle) <= 1.65:
                # Select free sectors with the same five-circle swept body
                # contract as the hard veto.  A centre ray can pass a wall
                # corner while the rear quarter clips it, which previously
                # caused repeated choose/veto/backtrack cycles.
                trajectory = ackermann_arc_trajectory(
                    angle, self.route_horizon, count=24
                )
                profile = occupancy.swept_footprint_signed_clearance_profile(
                    trajectory, FootprintConfig()
                )
                clearance = self.safe_length_from_profile(profile, self.route_horizon)
                persistent_minimum = math.inf
                with self.lock:
                    accumulated = self.accumulated_occupancy
                if accumulated is not None:
                    map_trajectory = self.body_trajectory_to_frame(trajectory, map_pose)
                    persistent_profile = accumulated.swept_footprint_signed_clearance_profile(
                        map_trajectory,
                        FootprintConfig(),
                        outside_is_occupied=False,
                    )
                    clearance = min(
                        clearance,
                        self.safe_length_from_profile(
                            persistent_profile, self.route_horizon
                        ),
                    )
                    finite = persistent_profile[np.isfinite(persistent_profile)]
                    if len(finite):
                        persistent_minimum = float(np.min(finite))
                odom_trajectory = self.body_trajectory_to_frame(trajectory, odom_pose)
                if self.topology.polyline_enters_failed_branch(
                    odom_trajectory[:, 1:3], goal_xy=goal_odom, stamp=stamp
                ):
                    clearance = 0.0
                    topology_utility = -4.0
                else:
                    topology_utility = 0.0
                # The accumulated map contributes a soft clearance preference
                # in addition to its hard known-wall veto.  This pushes a turn
                # toward the outside of a remembered corner without dictating
                # an exact grid path to the local planner.
                clearance_utility = (
                    0.0
                    if not math.isfinite(persistent_minimum)
                    else 0.45 * min(1.0, max(0.0, persistent_minimum))
                )
                angle_utilities.append(
                    (angle, topology_utility + clearance_utility)
                )
            else:
                # Rear headings are transaction requests.  The local planner
                # performs the exact forward/reverse probe before committing.
                clearance = ray_clearance(
                    values, resolution, origin, angle, maximum_m=self.route_horizon
                )
                angle_utilities.append((angle, 0.0))
            angle_clearances.append((angle, min(clearance, self.route_horizon)))
        previous_body = (
            None
            if self.previous_goal_heading is None
            else wrap_angle(self.previous_goal_heading - map_pose[2])
        )
        safe_clearance = self.standoff + self.route_minimum
        forward_rows = [
            (angle, clearance)
            for angle, clearance in angle_clearances
            if abs(angle) <= 1.65
        ]
        _, direct_clearance = min(
            forward_rows,
            key=lambda row: abs(wrap_angle(row[0] - desired)),
        )
        utility_by_angle = {
            round(wrap_angle(angle), 6): float(value)
            for angle, value in angle_utilities
        }

        def side_score(side):
            # Reserve a complete Ackermann tangent before latching a wall
            # side.  A merely centreline-safe ray can be too short to start
            # the turn and immediately hand control to static recovery.
            side_minimum = max(
                safe_clearance, self.boundary.enter_clearance_m
            )
            rows = [
                (angle, clearance)
                for angle, clearance in forward_rows
                if side * angle >= 0.15 and clearance >= side_minimum
            ]
            if not rows:
                return None
            return max(
                clearance
                - 0.60 * abs(wrap_angle(angle - desired))
                + utility_by_angle.get(round(wrap_angle(angle), 6), 0.0)
                for angle, clearance in rows
            )

        mission_distance = float(np.linalg.norm(goal_body))
        boundary_direct_clearance = direct_clearance
        left_score, right_score = side_score(1), side_score(-1)
        if abs(desired) > 1.65 and not self.boundary.active:
            # A genuinely rearward route is a turnaround transaction, not an
            # obstacle boundary hit.
            boundary_direct_clearance = max(
                self.boundary.release_clearance_m, self.route_horizon
            )
            left_score = right_score = None
        elif abs(desired) > 1.65:
            boundary_direct_clearance = 0.0
        boundary_decision = self.boundary.update(
            stamp,
            odom_pose[:2],
            mission_distance,
            boundary_direct_clearance,
            left_score=left_score,
            right_score=right_score,
        )
        self.last_boundary_decision = boundary_decision
        if boundary_decision.left_boundary:
            # Do not let the old tangent-heading hysteresis resist a newly
            # certified direct route.
            previous_body = None
            self.previous_goal_heading = None
        if boundary_decision.active:
            self.last_guidance_source = (
                "BOUNDARY_FOLLOW_LEFT"
                if boundary_decision.side > 0
                else "BOUNDARY_FOLLOW_RIGHT"
            )
            adjusted_clearances = []
            adjusted_utilities = []
            for angle, clearance in angle_clearances:
                boundary_utility = self.boundary.heading_utility(angle)
                if (
                    abs(angle) <= 1.65
                    and boundary_decision.side * angle < -0.40
                ):
                    # A strong side reversal would cross the obstacle-facing
                    # normal and silently swap wall sides.  Treat it as
                    # unavailable; near-straight corrections remain allowed.
                    clearance = min(clearance, 0.5 * safe_clearance)
                adjusted_clearances.append((angle, clearance))
                adjusted_utilities.append((
                    angle,
                    utility_by_angle.get(round(wrap_angle(angle), 6), 0.0)
                    + boundary_utility,
                ))
            angle_clearances = adjusted_clearances
            angle_utilities = adjusted_utilities
        decision = select_reactive_heading(
            desired,
            angle_clearances,
            previous_heading=previous_body,
            safe_clearance_m=safe_clearance,
            angle_utilities=angle_utilities,
        )
        # Do not latch an uncertified ray in a fully blocked sector.  The
        # breadcrumb supervisor should own that transition.
        if not decision.blocked:
            self.previous_goal_heading = wrap_angle(map_pose[2] + decision.angle)
        self.last_heading_decision = decision
        length = min(
            distance,
            self.route_horizon,
            max(0.0, decision.clearance - self.standoff),
        )
        return decision.angle, max(self.route_minimum, length)

    def visibility_corridor_body(self, map_pose):
        trimmed = self.visibility_cursor_path(map_pose[:2])
        if len(trimmed) < 2:
            return None
        clipped = [np.asarray(trimmed[0], dtype=float)]
        remaining = self.route_horizon
        for first, second in zip(trimmed[:-1], trimmed[1:]):
            first = np.asarray(first, dtype=float)
            second = np.asarray(second, dtype=float)
            length = float(np.linalg.norm(second - first))
            if length <= 1.0e-9:
                continue
            if length <= remaining:
                clipped.append(second)
                remaining -= length
            else:
                clipped.append(first + (remaining / length) * (second - first))
                remaining = 0.0
            if remaining <= 1.0e-6:
                break
        if len(clipped) < 2:
            return None
        values = resample_polyline(clipped, self.route_points)
        dx = values[:, 0] - map_pose[0]
        dy = values[:, 1] - map_pose[1]
        cosine, sine = math.cos(map_pose[2]), math.sin(map_pose[2])
        return np.column_stack((cosine * dx + sine * dy, -sine * dx + cosine * dy))

    def explored_topology_corridor_body(self, map_pose, odom_pose):
        """Return the driven sparse path as a local route-tube transaction."""

        path = [np.asarray(odom_pose[:2], dtype=float)]
        for point in self.last_topology_path:
            value = np.asarray(point, dtype=float)
            if np.linalg.norm(value - path[-1]) > 1.0e-4:
                path.append(value)
        if len(path) < 2:
            return None
        clipped = [path[0]]
        remaining = self.route_horizon
        for first, second in zip(path[:-1], path[1:]):
            length = float(np.linalg.norm(second - first))
            if length <= 1.0e-9:
                continue
            if length <= remaining:
                clipped.append(second)
                remaining -= length
            else:
                clipped.append(first + (remaining / length) * (second - first))
                remaining = 0.0
            if remaining <= 1.0e-6:
                break
        if len(clipped) < 2:
            return None
        points_map = np.asarray(self.odom_to_map(clipped), dtype=float)
        values = resample_polyline(points_map, self.route_points)
        dx = values[:, 0] - map_pose[0]
        dy = values[:, 1] - map_pose[1]
        cosine, sine = math.cos(map_pose[2]), math.sin(map_pose[2])
        return np.column_stack((
            cosine * dx + sine * dy,
            -sine * dx + cosine * dy,
        ))

    def forward_corridor_body(self, goal, grid, map_pose, odom_pose, stamp):
        angle, length = self.choose_local_heading(
            goal, grid, map_pose, odom_pose, stamp
        )
        if angle is None:
            return None
        if (
            self.last_guidance_source in (
                "FAR_KNOWN_VISIBILITY",
                "FAR_ATTEMPTABLE_VISIBILITY",
                "FAR_ATTEMPTABLE_NAVIGATION",
                "FAR_PARTIAL_ATTEMPTABLE",
                "EXPLORED_TOPOLOGY",
            )
            and self.last_boundary_decision is not None
            and not self.last_boundary_decision.active
            and not self.last_heading_decision.blocked
        ):
            corridor = (
                self.explored_topology_corridor_body(map_pose, odom_pose)
                if self.last_guidance_source == "EXPLORED_TOPOLOGY"
                else self.visibility_corridor_body(map_pose)
            )
            if corridor is not None:
                return corridor
        if abs(angle) > 2.1:
            # A route behind the car is a direction request.  The local
            # planner's Ackermann forward-restoration state machine decides
            # how to turn around; this node does not synthesize a fake arc.
            t = np.linspace(0.0, length, self.route_points)
            return np.column_stack((np.cos(angle) * t, np.sin(angle) * t))
        return ackermann_arc_trajectory(
            angle, length, count=self.route_points
        )[:, 1:3]

    @staticmethod
    def body_to_map(points, map_pose):
        values = np.asarray(points, dtype=float)
        cosine, sine = math.cos(map_pose[2]), math.sin(map_pose[2])
        return np.column_stack((
            map_pose[0] + cosine * values[:, 0] - sine * values[:, 1],
            map_pose[1] + sine * values[:, 0] + cosine * values[:, 1],
        ))

    def publish_route(
        self,
        world_points,
        gear,
        navigation_mode,
        state,
        detail,
        *,
        rolling_target_world=None,
        route_id=None,
        route_source=None,
        route_revision=None,
        **evidence
    ):
        values = resample_polyline(world_points, self.route_points)
        stamp = rospy.Time.now()
        header_frame = "map"
        path = Path()
        path.header.stamp = stamp
        path.header.frame_id = header_frame
        route = AckermannRoute()
        route.header = path.header
        for index, point in enumerate(values):
            if index + 1 < len(values):
                delta = values[index + 1] - point
            else:
                delta = point - values[index - 1]
            yaw = math.atan2(delta[1], delta[0])
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x = float(point[0])
            pose.pose.position.y = float(point[1])
            pose.pose.orientation.z = math.sin(0.5 * yaw)
            pose.pose.orientation.w = math.cos(0.5 * yaw)
            path.poses.append(pose)
            route_point = RoutePoint()
            route_point.pose = pose.pose
            route_point.gear = int(Gear.NEUTRAL)
            route_point.steering = 0.0
            route.points.append(route_point)
        subgoal_index = min(len(path.poses) - 1, max(1, len(path.poses) // 2))
        subgoal = path.poses[subgoal_index]
        if rolling_target_world is not None:
            target = np.asarray(rolling_target_world, dtype=float)
            if target.shape == (2,) and np.all(np.isfinite(target)):
                start = values[0]
                target_distance = float(np.linalg.norm(target - start))
                if target_distance > self.rolling_route_maximum_carrot_distance:
                    target = self.point_along_polyline(
                        values, self.rolling_route_maximum_carrot_distance
                    )
                    target_distance = float(np.linalg.norm(target - start))
                subgoal = PoseStamped()
                subgoal.header = path.header
                subgoal.pose.position.x = float(target[0])
                subgoal.pose.position.y = float(target[1])
                tangent_index = min(
                    len(values) - 2,
                    max(
                        0,
                        int(np.argmin(np.linalg.norm(values - target, axis=1))),
                    ),
                )
                tangent = values[tangent_index + 1] - values[tangent_index]
                yaw = math.atan2(tangent[1], tangent[0])
                subgoal.pose.orientation.z = math.sin(0.5 * yaw)
                subgoal.pose.orientation.w = math.cos(0.5 * yaw)
                evidence.update(
                    rolling_target_distance_m=target_distance,
                    rolling_target_latched=True,
                )
        command = LocalRouteCommand()
        command.header = path.header
        command.target = subgoal.pose
        command.requested_gear = int(gear)
        command.segment_index = 0
        command.segment_end = False
        command.navigation_mode = int(navigation_mode)
        self.generation += 1
        active_observation = (
            self.topology_route_cursor.last_observation
            if self.last_guidance_source == "EXPLORED_TOPOLOGY"
            else self.visibility_route_cursor.last_observation
            if self.last_guidance_source in (
                "FAR_KNOWN_VISIBILITY",
                "FAR_ATTEMPTABLE_VISIBILITY",
                "FAR_ATTEMPTABLE_NAVIGATION",
                "FAR_PARTIAL_ATTEMPTABLE",
            )
            else None
        )
        command.route_id = str(
            route_id
            if route_id is not None
            else active_observation.route_id
            if active_observation is not None
            else ""
        )
        command.route_source = str(
            route_source
            if route_source is not None
            else self.last_guidance_source
        )
        command.route_revision = int(
            route_revision
            if route_revision is not None
            else active_observation.revision
            if active_observation is not None
            else 0
        )
        command.authority_epoch = int(self.generation)
        command.rolling_target_latched = bool(
            rolling_target_world is not None
        )
        self.path_pub.publish(path)
        self.route_pub.publish(route)
        self.subgoal_pub.publish(subgoal)
        self.command_pub.publish(command)
        if (
            self.visibility_course_revalidation_pending
            and command.route_source.startswith("FAR_")
        ):
            self.visibility_course_revalidation_pending = False
            rospy.loginfo(
                "Published FAR route answer for local course revalidation "
                "route_id=%s revision=%d epoch=%d",
                command.route_id or "none",
                int(command.route_revision),
                int(command.authority_epoch),
            )
        evidence.update(generation=self.generation, requested_gear=int(gear))
        evidence.update(
            route_id=command.route_id,
            route_source=command.route_source,
            route_revision=int(command.route_revision),
            authority_epoch=int(command.authority_epoch),
            rolling_target_latched=bool(command.rolling_target_latched),
        )
        self.publish_status(state, detail, **evidence)

    def start_backtrack(
        self,
        odom_pose,
        goal_odom,
        stamp,
        before_site_index=None,
        *,
        certified_far_egress=False,
    ):
        # Preserve same-goal failed sides while retreating.  Otherwise the
        # next visit to the same obstacle hit repeats the identical choice.
        self.boundary.reset(clear_failures=False)
        self.last_boundary_decision = None
        nearest = self.trail.nearest_index(*odom_pose[:2])
        if nearest is None:
            self.far_emergency_egress_active = False
            self.recovery.fail_safe()
            return
        # Do not select the just-recorded current cell as a recovery site.
        before = max(0, nearest - 2)
        if before_site_index is not None:
            before = min(before, max(0, int(before_site_index) - 1))
        self.backtrack_start_index = nearest
        self.backtrack_cursor_index = nearest
        self.far_emergency_egress_active = bool(certified_far_egress)
        if self.far_emergency_egress_active:
            (
                self.backtrack_site_index,
                self.dead_end_escape_site_kind,
                self.dead_end_escape_target_distance_m,
            ) = select_dead_end_egress_site(
                self.trail,
                nearest,
                before_index=before,
                minimum_distance_m=self.dead_end_escape_minimum_distance,
                maximum_distance_m=self.far_static_egress_maximum,
            )
            if self.backtrack_site_index is None:
                self.far_emergency_egress_active = False
                self.dead_end_escape_completion_reason = (
                    "INSUFFICIENT_CERTIFIED_INGRESS"
                )
                self.recovery.fail_safe()
                return
            self.dead_end_escape_sequence += 1
            self.dead_end_escape_id = self.dead_end_escape_sequence
            self.dead_end_escape_route_revision = 1
            self.dead_end_escape_cross_track_error_m = 0.0
            self.dead_end_escape_last_route_map = []
            self.dead_end_escape_started_map_correction_generation = (
                self.map_correction_generation
            )
            self.dead_end_escape_completion_reason = "ACTIVE"
            self.dead_end_escape_diverged_since = None
            self.dead_end_escape_connector_unavailable_since = None
            self.dead_end_escape_live_reanchors = 0
            self.dead_end_escape_live_target_index = None
            self.failed_branch_exit_lock_branch_id = None
            self.failed_branch_exit_lock_origin_odom = None
            self.failed_branch_exit_lock_started_stamp = None
            self.failed_branch_exit_lock_progress_m = 0.0
        else:
            self.backtrack_site_index = self.trail.most_recent_recovery_site(
                before,
                from_index=nearest,
                minimum_path_distance_m=self.minimum_backtrack_before_resume,
            )
        self.backtrack_start_xy = np.asarray(odom_pose[:2], dtype=float)
        self.backtrack_started_stamp = float(stamp)
        self.backtrack_blocked_since = None
        self.resume_target_index = None
        self.resume_fallback_target_xy = None
        self.resume_started_stamp = None
        self.resume_travelled_m = 0.0
        self.resume_last_xy = None
        self.resume_blocked_since = None
        self.committed_topology_path = []
        self.topology_route_cursor.reset()
        self.topology_last_carrot_odom = None
        if self.backtrack_site_index is not None:
            self.last_guidance_source = (
                "FAR_DEAD_END_EGRESS"
                if self.far_emergency_egress_active
                else "BREADCRUMB_BACKTRACK"
            )
            lower, upper = sorted((self.backtrack_site_index, nearest))
            branch = self.topology.mark_failed_branch(
                [
                    (point.x, point.y)
                    for point in self.trail.points[lower : upper + 1]
                ],
                goal_xy=goal_odom,
                stamp=stamp,
                static=True,
            )
            if branch is not None:
                if self.far_emergency_egress_active:
                    self.dead_end_escape_branch_id = branch.branch_id
                rospy.loginfo(
                    "Remembered directed failed branch id=%d entry=(%.3f,%.3f) "
                    "terminal=(%.3f,%.3f) egress_distance=%.3fm "
                    "site_kind=%s escape_id=%d",
                    branch.branch_id,
                    branch.entry[0],
                    branch.entry[1],
                    branch.terminal[0],
                    branch.terminal[1],
                    self.dead_end_escape_target_distance_m,
                    self.dead_end_escape_site_kind,
                    self.dead_end_escape_id,
                )

    def breadcrumb_corridor_map(self, corridor, map_pose):
        """Coherently re-anchor a driven connector in the current SLAM map.

        Per-sample map poses remain diagnostic evidence, but mixing transforms
        captured before and after scan-matching corrections creates a kink at
        the live vehicle pose.  Reproject the short odom-frame connector with
        one current map<-odom transform; current map/LiDAR and hard veto then
        close the loop instead of replaying historical commands open-loop.
        """

        if len(corridor) < 2:
            return []
        output = list(
            self.odom_to_map([(point.x, point.y) for point in corridor])
        )
        output[0] = np.asarray(map_pose[:2], dtype=float)
        filtered = [output[0]]
        for point in output[1:]:
            if float(np.linalg.norm(point - filtered[-1])) > 1.0e-4:
                filtered.append(point)
        return filtered

    def validate_dead_end_egress_route(self, route):
        """Return the longest current-map-safe prefix of an egress connector."""

        with self.lock:
            accumulated = self.accumulated_grid
        if accumulated is None or len(route) < 2:
            return []
        values, resolution, origin, _ = accumulated
        return list(
            self.visibility.longest_margin_egress_prefix(
                route,
                values,
                resolution,
                origin,
                maximum_unknown_fraction=(
                    self.dead_end_escape_maximum_unknown_fraction
                ),
            )
        )

    def live_dead_end_egress_reanchor(self, odom_pose, map_pose):
        """Attach the measured pose to a nearby older ingress anchor.

        The driven trail decides which branch is being exited, but it is not
        replayed open-loop.  When wheel/SLAM correction leaves the vehicle
        beside the historical centreline, try several short live connectors
        to monotonically older samples and retain only one that passes the
        newest inflated-map check.  The local planner and hard veto still
        decide the exact signed Ackermann motion.
        """

        if (
            self.backtrack_cursor_index is None
            or self.backtrack_site_index is None
            or not self.trail.points
        ):
            return []
        lower = max(
            self.backtrack_site_index,
            self.trail.older_index_at_distance(
                self.backtrack_cursor_index,
                max(self.backtrack_chunk + 0.6, 1.2),
            ),
        )
        current_map = np.asarray(map_pose[:2], dtype=float)

        def connector(index):
            if not (
                self.backtrack_site_index
                <= int(index)
                < self.backtrack_cursor_index
            ):
                return [], math.inf
            target = self.trail.points[index]
            distance = math.hypot(
                target.x - odom_pose[0], target.y - odom_pose[1]
            )
            if distance <= self.dead_end_escape_live_target_capture:
                return [], distance
            candidate = list(
                self.odom_to_map([
                    (odom_pose[0], odom_pose[1]),
                    (target.x, target.y),
                ])
            )
            candidate[0] = current_map
            validated = self.validate_dead_end_egress_route(candidate)
            if len(validated) < 2:
                return [], distance
            if float(np.linalg.norm(validated[-1] - candidate[-1])) > 0.10:
                return [], distance
            return validated, distance

        # Keep steering toward one physical anchor until it is captured or
        # the newest occupancy map invalidates its connector.  Reissuing the
        # same target as a new route on every 5 Hz callback used to reset the
        # local controller dozens of times during one reverse arc.
        latched = self.dead_end_escape_live_target_index
        if latched is not None:
            validated, distance = connector(latched)
            if len(validated) >= 2:
                self.dead_end_escape_diverged_since = None
                return validated
            if distance <= self.dead_end_escape_live_target_capture:
                self.backtrack_cursor_index = min(
                    self.backtrack_cursor_index, int(latched)
                )
            self.dead_end_escape_live_target_index = None

        for index in range(self.backtrack_cursor_index - 1, lower - 1, -1):
            validated, distance = connector(index)
            if len(validated) < 2:
                continue
            self.dead_end_escape_live_target_index = int(index)
            self.dead_end_escape_live_reanchors += 1
            self.dead_end_escape_route_revision += 1
            self.dead_end_escape_diverged_since = None
            rospy.loginfo(
                "Latched live FAR dead-end egress anchor escape_id=%d "
                "target_index=%d connector=%.3fm revision=%d",
                self.dead_end_escape_id,
                index,
                distance,
                self.dead_end_escape_route_revision,
            )
            return validated
        return []

    def failed_branch_by_id(self, branch_id):
        return next(
            (
                branch
                for branch in self.topology.failed_branches
                if branch.branch_id == branch_id
            ),
            None,
        )

    def failed_branch_exit_lock_guidance(
        self, map_pose, odom_pose, goal_odom, plan, path, stamp
    ):
        """Keep one completed egress from immediately re-entering its branch."""

        branch = self.failed_branch_by_id(
            self.failed_branch_exit_lock_branch_id
        )
        if branch is None:
            self.failed_branch_exit_lock_branch_id = None
            self.failed_branch_exit_lock_origin_odom = None
            self.failed_branch_exit_lock_started_stamp = None
            self.failed_branch_exit_lock_progress_m = 0.0
            return None
        current = np.asarray(odom_pose[:2], dtype=float)
        origin = np.asarray(
            self.failed_branch_exit_lock_origin_odom, dtype=float
        )
        self.failed_branch_exit_lock_progress_m = float(
            np.linalg.norm(current - origin)
        )
        if (
            plan is not None
            and plan.status == "PASS"
            and len(path) >= 2
            and self.visibility_active_route_motion_authorized
        ):
            path_odom = self.map_to_odom(path)
            if not self.topology.polyline_enters_failed_branch(
                path_odom, goal_xy=goal_odom, stamp=stamp
            ):
                rospy.loginfo(
                    "Released failed-branch exit lock branch_id=%d after "
                    "branch-safe FAR route acquisition progress=%.3fm",
                    branch.branch_id,
                    self.failed_branch_exit_lock_progress_m,
                )
                self.failed_branch_exit_lock_branch_id = None
                self.failed_branch_exit_lock_origin_odom = None
                self.failed_branch_exit_lock_started_stamp = None
                self.failed_branch_exit_lock_progress_m = 0.0
                return None
        if (
            self.failed_branch_exit_lock_progress_m
            >= self.failed_branch_exit_lock_progress
        ):
            self.visibility_route_acquisition_reason = (
                "failed_branch_exit_lock_waiting_for_branch_safe_far_route"
            )
            self.last_guidance_source = "FAR_ACQUISITION_HOLD"
            return np.zeros(2, dtype=float)
        direction = np.asarray(branch.entry, dtype=float) - np.asarray(
            branch.terminal, dtype=float
        )
        norm = float(np.linalg.norm(direction))
        if norm <= 1.0e-6:
            self.last_guidance_source = "FAR_ACQUISITION_HOLD"
            return np.zeros(2, dtype=float)
        target_odom = current + direction / norm * self.route_horizon
        target_map = np.asarray(self.odom_to_map([target_odom])[0], dtype=float)
        dx, dy = target_map[0] - map_pose[0], target_map[1] - map_pose[1]
        cosine, sine = math.cos(map_pose[2]), math.sin(map_pose[2])
        self.visibility_route_acquisition_reason = (
            "failed_branch_exit_lock_outward_progress"
        )
        self.last_guidance_source = "FAILED_BRANCH_EXIT_LOCK"
        return np.asarray((cosine * dx + sine * dy, -sine * dx + cosine * dy))

    def complete_dead_end_egress(self, travelled, stamp, reason):
        """Release exclusive reverse authority only at an external anchor."""

        self.trail.truncate_after(self.backtrack_site_index)
        self.recovery.complete_certified_egress()
        self.far_emergency_egress_active = False
        self.backtrack_started_stamp = None
        self.backtrack_blocked_since = None
        self.visibility_static_replan_failures = 0
        self.visibility_static_blocked_since = None
        self.last_goal_improvement_stamp = float(stamp)
        self.dead_end_escape_completion_reason = str(reason)
        self.dead_end_escape_diverged_since = None
        self.dead_end_escape_connector_unavailable_since = None
        self.dead_end_escape_last_route_map = []
        self.dead_end_escape_live_target_index = None
        if self.dead_end_escape_branch_id is not None:
            site = self.trail.points[self.backtrack_site_index]
            self.failed_branch_exit_lock_branch_id = (
                self.dead_end_escape_branch_id
            )
            self.failed_branch_exit_lock_origin_odom = (site.x, site.y)
            self.failed_branch_exit_lock_started_stamp = float(stamp)
            self.failed_branch_exit_lock_progress_m = 0.0
        if not self.visibility_active_route_motion_authorized:
            self.invalidate_visibility_route(
                "dead_end_egress_anchor_reached", force=True
            )
        rospy.loginfo(
            "Completed FAR dead-end egress escape_id=%d branch_id=%s "
            "distance=%.3fm reason=%s map_reanchors=%d",
            self.dead_end_escape_id,
            str(self.dead_end_escape_branch_id),
            travelled,
            reason,
            max(
                0,
                self.map_correction_generation
                - self.dead_end_escape_started_map_correction_generation,
            ),
        )

    def backtrack_route(
        self,
        odom_pose,
        map_pose,
        features,
        *,
        far_route_ready=False,
        topology_exit_ready=False,
        stamp=0.0,
    ):
        if (
            self.backtrack_site_index is None
            or self.backtrack_start_xy is None
            or self.backtrack_cursor_index is None
        ):
            self.recovery.fail_safe()
            return None
        # Advance only toward older samples inside one metric replay window.
        # Four samples represented only about 0.8 m and a slow synchronous
        # callback could let the car leave that window before the next update.
        # A metric bound keeps self-crossing protection without manufacturing
        # a large cross-track error from scheduler latency.
        replay_window_lower = self.trail.older_index_at_distance(
            self.backtrack_cursor_index,
            max(self.backtrack_chunk + 0.5, 1.0),
        )
        nearest = self.trail.nearest_index_window(
            *odom_pose[:2],
            lower_index=max(
                self.backtrack_site_index,
                replay_window_lower,
            ),
            upper_index=self.backtrack_cursor_index,
        )
        if nearest is not None:
            self.backtrack_cursor_index = min(
                self.backtrack_cursor_index, nearest
            )
        cursor = self.backtrack_cursor_index
        travelled = (
            self.trail.path_distance(cursor, self.backtrack_start_index)
        )
        site = self.trail.points[self.backtrack_site_index]
        at_site = (
            cursor <= self.backtrack_site_index + 1
            and math.hypot(site.x - odom_pose[0], site.y - odom_pose[1]) <= 0.40
        )
        # A site certified when it was first traversed remains a valid
        # topological recovery candidate even if a single instantaneous scan
        # temporarily under-classifies the opening.  Exact motion is still
        # checked by the local planner's swept-footprint hard veto.
        robust_site = bool(
            site.turnaround
            or site.junction
            or features["turnaround"]
            or features["junction"]
        )
        if at_site and self.far_emergency_egress_active:
            # This site is the measured entry of the branch we just marked
            # failed, selected from a physically driven ingress suffix.  FAR
            # does not have to finish a synchronous graph solve in the exact
            # callback which captures that anchor.  Release reverse authority
            # here and let the failed-branch exit lock hold outward progress
            # until a branch-safe FAR route is available.
            reason = (
                "STABLE_FAR_ROUTE_REACQUIRED"
                if far_route_ready
                else "ALTERNATIVE_TOPOLOGY_AVAILABLE"
                if topology_exit_ready
                else "RECOVERY_ANCHOR_REACHED"
            )
            self.complete_dead_end_egress(travelled, stamp, reason)
            return None
        if (
            at_site
            and travelled >= self.minimum_backtrack_before_resume
            and robust_site
        ):
            self.recovery.begin_resume()
            self.resume_start_xy = np.asarray(odom_pose[:2], dtype=float)
            target_index = self.trail.older_index_at_distance(
                self.backtrack_site_index, self.resume_distance
            )
            if target_index < self.backtrack_site_index:
                self.resume_target_index = target_index
                self.resume_fallback_target_xy = None
            else:
                site_heading = self.trail.points[self.backtrack_site_index].yaw
                outbound = wrap_angle(site_heading + math.pi)
                self.resume_target_index = None
                self.resume_fallback_target_xy = np.asarray((
                    odom_pose[0] + self.resume_distance * math.cos(outbound),
                    odom_pose[1] + self.resume_distance * math.sin(outbound),
                ))
            self.resume_started_stamp = None
            self.resume_travelled_m = 0.0
            self.resume_last_xy = np.asarray(odom_pose[:2], dtype=float)
            self.last_guidance_source = "CERTIFIED_OUTBOUND_TRAIL"
            return None
        maximum_backtrack = (
            self.far_static_egress_maximum
            if self.far_emergency_egress_active
            else self.backtrack_maximum
        )
        if travelled >= maximum_backtrack:
            self.far_emergency_egress_active = False
            self.backtrack_started_stamp = None
            self.backtrack_blocked_since = None
            if self.recovery.state == MemoryNavigationState.FAR_DEAD_END_EGRESS:
                self.dead_end_escape_completion_reason = "EGRESS_DISTANCE_LIMIT"
            self.recovery.fail_safe()
            return None
        corridor, _, replay_direction = self.trail.reverse_replay_corridor(
            *odom_pose[:2],
            start_index=cursor,
            target_distance_m=self.backtrack_chunk,
        )
        if len(corridor) < 2:
            # Reaching the end of a short trail is not proof that the car can
            # turn there.  Fail closed instead of manufacturing a forward leg
            # from an uncertified recovery pose.
            self.recovery.fail_safe()
            return None
        gear = Gear.FORWARD if replay_direction > 0 else Gear.REVERSE
        route_map = self.breadcrumb_corridor_map(corridor, map_pose)
        expected = self.trail.points[cursor]
        self.dead_end_escape_cross_track_error_m = math.hypot(
            expected.x - odom_pose[0], expected.y - odom_pose[1]
        )
        live_reanchored = False
        if (
            self.far_emergency_egress_active
            and self.dead_end_escape_cross_track_error_m
            > self.dead_end_escape_maximum_cross_track
        ):
            reanchored = self.live_dead_end_egress_reanchor(
                odom_pose, map_pose
            )
            if len(reanchored) >= 2:
                route_map = reanchored
                live_reanchored = True
        if (
            self.far_emergency_egress_active
            and not live_reanchored
            and self.dead_end_escape_cross_track_error_m
            > self.dead_end_escape_maximum_cross_track
        ):
            if self.dead_end_escape_diverged_since is None:
                self.dead_end_escape_diverged_since = float(stamp)
                rospy.logwarn(
                    "FAR dead-end egress cross-track transient escape_id=%d "
                    "cross_track=%.3fm limit=%.3fm; confirming for %.2fs",
                    self.dead_end_escape_id,
                    self.dead_end_escape_cross_track_error_m,
                    self.dead_end_escape_maximum_cross_track,
                    self.dead_end_escape_divergence_confirmation,
                )
            elif (
                float(stamp) - self.dead_end_escape_diverged_since
                >= self.dead_end_escape_divergence_confirmation
            ):
                self.dead_end_escape_completion_reason = (
                    "EGRESS_LOCALIZATION_DIVERGED"
                )
                self.far_emergency_egress_active = False
                self.recovery.fail_safe()
                rospy.logwarn(
                    "FAR dead-end egress localization persistently diverged "
                    "escape_id=%d cross_track=%.3fm limit=%.3fm",
                    self.dead_end_escape_id,
                    self.dead_end_escape_cross_track_error_m,
                    self.dead_end_escape_maximum_cross_track,
                )
                return None
        else:
            self.dead_end_escape_diverged_since = None
        if self.far_emergency_egress_active:
            if not live_reanchored:
                route_map = self.validate_dead_end_egress_route(route_map)
            if len(route_map) < 2:
                route_map = self.live_dead_end_egress_reanchor(
                    odom_pose, map_pose
                )
            if len(route_map) < 2:
                if self.dead_end_escape_connector_unavailable_since is None:
                    self.dead_end_escape_connector_unavailable_since = float(
                        stamp
                    )
                unavailable_age = max(
                    0.0,
                    float(stamp)
                    - self.dead_end_escape_connector_unavailable_since,
                )
                if (
                    unavailable_age
                    < self.dead_end_escape_divergence_confirmation
                ):
                    self.dead_end_escape_completion_reason = (
                        "EGRESS_REANCHOR_REVALIDATION_HOLD"
                    )
                    rospy.logwarn_throttle(
                        1.0,
                        "FAR dead-end egress has no current-map-safe live "
                        "connector escape_id=%d; holding for map "
                        "revalidation %.2f/%.2fs",
                        self.dead_end_escape_id,
                        unavailable_age,
                        self.dead_end_escape_divergence_confirmation,
                    )
                    return None
                self.dead_end_escape_completion_reason = (
                    "EGRESS_REANCHOR_EXHAUSTED_CURRENT_MAP"
                )
                self.far_emergency_egress_active = False
                self.recovery.fail_safe()
                rospy.logwarn(
                    "FAR dead-end egress persistently exhausted current-map-"
                    "safe live connector reanchors escape_id=%d "
                    "cross_track=%.3fm confirmation=%.2fs",
                    self.dead_end_escape_id,
                    self.dead_end_escape_cross_track_error_m,
                    unavailable_age,
                )
                return None
            self.dead_end_escape_connector_unavailable_since = None
            self.dead_end_escape_diverged_since = None
            self.dead_end_escape_completion_reason = "ACTIVE"
            self.dead_end_escape_last_route_map = [
                tuple(float(value) for value in point) for point in route_map
            ]
        return (
            route_map,
            gear,
        )

    def resume_route(self, odom_pose, stamp):
        if self.resume_start_xy is None:
            self.resume_start_xy = np.asarray(odom_pose[:2], dtype=float)
        if self.resume_started_stamp is None:
            self.resume_started_stamp = float(stamp)
        current_xy = np.asarray(odom_pose[:2], dtype=float)
        if self.resume_last_xy is not None:
            self.resume_travelled_m += float(np.linalg.norm(current_xy - self.resume_last_xy))
        self.resume_last_xy = current_xy
        nearest = self.trail.nearest_index(*odom_pose[:2])
        if self.resume_target_index is not None:
            target = self.trail.points[self.resume_target_index]
            target_reached = (
                nearest is not None
                and nearest <= self.resume_target_index + 1
                and math.hypot(target.x - odom_pose[0], target.y - odom_pose[1]) <= 0.45
            )
        else:
            target_reached = bool(
                self.resume_fallback_target_xy is not None
                and np.linalg.norm(current_xy - self.resume_fallback_target_xy) <= 0.45
            )
        if target_reached:
            self.recovery.escaped()
            return None
        corridor, _ = self.trail.older_corridor(
            *odom_pose[:2], target_distance_m=max(1.5, self.resume_distance + 0.5)
        )
        if len(corridor) >= 2:
            return self.odom_to_map([(point.x, point.y) for point in corridor])
        # At the oldest breadcrumb the only certified outbound direction is
        # opposite the heading used when that point was first entered.
        heading = wrap_angle(odom_pose[2] + math.pi)
        points = np.column_stack((
            odom_pose[0] + np.cos(heading) * np.linspace(0.0, 1.5, self.route_points),
            odom_pose[1] + np.sin(heading) * np.linspace(0.0, 1.5, self.route_points),
        ))
        return self.odom_to_map(points)

    def update(self, _event):
        if rospy.is_shutdown():
            return
        with self.lock:
            odom, grid, goal, local_state, local_candidates = (
                self.odom, self.grid, self.goal, self.local_state, self.local_candidates
            )
        if odom is None or grid is None:
            self.publish_status("MAPPING_WAIT", "waiting for wheel/IMU odometry and local scan")
            return
        try:
            map_pose = self.lookup_pose()
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            self.publish_status("MAPPING_WAIT", "waiting for SLAM map->odom transform")
            return
        self.last_map_pose = map_pose
        if goal is None or self.recovery.state == MemoryNavigationState.IDLE:
            self.publish_status("IDLE", "online map active; use RViz 2D Nav Goal")
            return

        goal_endpoint = self.goal_endpoint_evidence(goal)
        if goal_endpoint["observed_occupied"]:
            if (
                self.visibility_plan is not None
                or self.visibility_route_cursor.active
            ):
                self.invalidate_visibility_route(
                    "goal_in_observed_obstacle", force=True
                )
            self.last_guidance_source = "INVALID_GOAL"
            self.publish_status(
                "INVALID_GOAL",
                "RViz goal lies in an observed occupied cell; choose a free "
                "vehicle-centre position",
                goal_endpoint=goal_endpoint,
                dense_replan_started=False,
                local_exploration_authorized=False,
            )
            return

        stamp = odom.header.stamp.to_sec() or rospy.Time.now().to_sec()
        odom_pose = self.odom_pose(odom)
        self.last_odom_pose = odom_pose
        self.observe_initial_local_exploration(
            odom_pose, odom.twist.twist.linear.x, stamp
        )
        local_route_revalidation_hold = bool(
            local_state is not None
            and (
                "waiting for a measured-pose far re-anchor"
                in str(local_state.detail).lower()
                or "forward-restoration scheduling budget exhausted"
                in str(local_state.detail).lower()
            )
        )
        local_maneuver_reported = bool(
            local_state is not None and local_state.maneuver_active
        )
        local_maneuver_active = bool(
            local_maneuver_reported
            # ``PlannerState.maneuver_active`` deliberately remains true
            # between the finite legs of one multi-point turn.  Once the
            # local planner enters course revalidation, however, the last
            # leg is already settled and FAR must be allowed to replace the
            # frozen suffix.  Treating this inter-leg barrier as an active
            # manoeuvre immediately re-latched the transaction released by
            # ``on_replan_request``: FAR then cached its valid answer while
            # the local planner waited forever for that same answer.
            and not local_route_revalidation_hold
            and not self.visibility_course_revalidation_pending
            and self.recovery.state
            not in (
                MemoryNavigationState.BACKTRACK_REVERSE,
                MemoryNavigationState.FAR_DEAD_END_EGRESS,
                MemoryNavigationState.RESUME_FORWARD,
            )
        )
        self.sync_visibility_maneuver_transaction(
            local_maneuver_active, map_pose, stamp
        )
        self.sync_route_turnaround_transaction(
            local_state, local_maneuver_reported, odom_pose, stamp
        )
        # Apply a loop-closure/scan-matching correction before converting the
        # mission goal or constructing a new route.  The odom memory remains
        # untouched and therefore continuous.  If the local planner is inside
        # a committed multi-gear turn, invalidation is deferred until that
        # route transaction is released.
        self.apply_pending_map_correction(stamp)

        try:
            goal_odom = self.map_to_odom(
                [(goal.pose.position.x, goal.pose.position.y)]
            )[0]
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            self.publish_status("MAPPING_WAIT", "waiting for map-to-odom goal transform")
            return
        features = self.local_features(grid)
        progress = self.progress(odom_pose, stamp)
        if local_maneuver_active:
            # A committed Ackermann leg is one atomic local transaction.  It
            # may have been launched from an accepted FAR route *or* from the
            # explored-topology fallback while FAR is still settling.  In
            # both cases keep the already-published route/status authoritative
            # until the leg settles; publishing FAR_MAPPING_WAIT here used to
            # stop the car halfway through the turn.
            # Do not write parking-style forward/reverse legs into the
            # ordinary dead-end ingress stack.  FAR planning itself must keep
            # running, however: a LOCAL_SAFE_EXPLORATION leg can begin after
            # the first acquisition confirmation and used to suppress the
            # second confirmation for the entire multi-leg turn.  The local
            # planner buffers this complete route transaction and applies it
            # only at a stopped leg boundary, preserving serial motion
            # authority while removing the planning blackout.
            self.last_goal_improvement_stamp = stamp
            background_plan, background_path = self.update_visibility_plan(
                map_pose, goal, stamp
            )
            background_far_ready = bool(
                self.visibility_active_route_motion_authorized
                and background_plan is not None
                and background_plan.status == "PASS"
                and len(background_path) >= 2
            )
            if background_far_ready:
                self.last_guidance_source = (
                    "FAR_KNOWN_VISIBILITY"
                    if self.visibility_active_route_accepted
                    and background_plan.mode == "KNOWN_VISIBILITY"
                    else "FAR_PARTIAL_ATTEMPTABLE"
                    if background_plan.mode == "PARTIAL_ATTEMPTABLE"
                    else "FAR_ATTEMPTABLE_VISIBILITY"
                    if self.visibility_active_route_accepted
                    else "FAR_ATTEMPTABLE_NAVIGATION"
                )
                body = self.visibility_corridor_body(map_pose)
                if body is not None:
                    observation = self.visibility_route_cursor.last_observation
                    self.publish_route(
                        self.body_to_map(body, map_pose),
                        Gear.FORWARD,
                        LocalRouteCommand.NAVIGATION_MEMORY_GOAL,
                        "GOAL_SEEK",
                        "FAR route prepared in parallel; local planner will "
                        "adopt it at the next stopped maneuver-leg boundary",
                        rolling_target_world=(
                            observation.carrot_xy
                            if observation is not None else None
                        ),
                        local_maneuver_hold=True,
                        maneuver_leg=int(local_state.maneuver_leg),
                        background_far_ready=True,
                        deferred_control_handoff=True,
                    )
                    return
            self.publish_status(
                "GOAL_SEEK",
                "committed local Ackermann leg owns motion; FAR keeps planning in parallel and hands off at a stopped leg boundary",
                local_maneuver_hold=True,
                maneuver_leg=int(local_state.maneuver_leg),
                background_far_ready=bool(
                    self.visibility_active_route_motion_authorized
                    and self.visibility_plan is not None
                    and self.visibility_plan.status == "PASS"
                ),
            )
            return
        goal_distance = math.hypot(
            goal.pose.position.x - map_pose[0], goal.pose.position.y - map_pose[1]
        )
        local_goal_hold = bool(
            local_state is not None
            and "latched goal hold" in str(local_state.detail).lower()
        )
        if local_goal_hold or (
            goal_distance <= self.goal_tolerance
            and abs(float(odom.twist.twist.linear.x)) <= self.goal_settled_speed
        ):
            self.boundary.reset()
            self.last_boundary_decision = None
            self.recovery.complete_goal()
            self.publish_status(
                "GOAL_REACHED",
                "position goal reached and vehicle settled",
                goal_distance_m=goal_distance,
            )
            return
        if self.best_goal_distance is None or goal_distance < self.best_goal_distance - 0.10:
            self.best_goal_distance = goal_distance
            self.last_goal_improvement_stamp = stamp
        goal_stalled = (
            self.last_goal_improvement_stamp is not None
            and stamp - self.last_goal_improvement_stamp >= self.goal_stall_confirmation
        )
        dense_replan_pending = self.dense_visibility_replan_is_pending()
        if dense_replan_pending:
            # Graph computation is not physical failure evidence.  Freeze the
            # mission-level recovery clocks so this bounded background task
            # cannot consume reverse, turnaround or restoration authority.
            self.last_goal_improvement_stamp = stamp
            goal_stalled = False
        if self.recovery.state in (
            MemoryNavigationState.GOAL_SEEK,
            MemoryNavigationState.SUSPECT_DEAD_END,
        ):
            if local_maneuver_active:
                # Goal-distance progress is not meaningful midway through a
                # multi-leg turn.  Pause the dead-end timer rather than letting
                # the memory authority pre-empt the committed manoeuvre.
                self.last_goal_improvement_stamp = stamp
                goal_stalled = False
            if local_route_revalidation_hold:
                # This is an intentional transaction barrier, not evidence
                # that the current FAR route is geometrically blocked.  The
                # local planner will release it on a newer synchronized route
                # transaction.  Counting it as a dead end used to make the
                # memory layer inject a breadcrumb reverse while FAR already
                # had an accepted forward route.
                self.last_goal_improvement_stamp = stamp
                goal_stalled = False
            # Breadcrumbs record ordinary mission motion.  Parking-style
            # local manoeuvres have their own atomic transaction and are not
            # eligible to become a directed failed branch.
            signed_speed = float(odom.twist.twist.linear.x)
            motion_direction = 1 if signed_speed > 0.02 else -1 if signed_speed < -0.02 else 0
            if (motion_direction or not self.trail.points) and not local_maneuver_reported:
                self.record_breadcrumb(
                    odom_pose,
                    stamp,
                    features,
                    motion_direction=motion_direction,
                    record_topology=(
                        motion_direction > 0 and not local_maneuver_reported
                    ),
                )
            dynamic_blocked = bool(local_state is not None and local_state.blocked_by_dynamic)
            local_static_blocked = bool(
                local_state is not None and local_state.blocked_by_static
            )
            if local_static_blocked:
                self.visibility_last_local_static_block_stamp = float(stamp)
            local_not_executable = bool(
                local_state is None or not local_state.executable
            )
            route_unavailable_after_recent_static_block = bool(
                self.last_guidance_source == "FAR_ACQUISITION_HOLD"
                and not self.visibility_active_route_motion_authorized
                and stamp - self.visibility_last_local_static_block_stamp
                <= self.visibility_no_route_static_evidence_timeout
            )
            static_blocked = bool(
                not dense_replan_pending
                and (
                    bool(
                        local_static_blocked
                        and not local_route_revalidation_hold
                    )
                    or bool(
                        goal_stalled
                        and goal_distance > self.dead_end_goal_exclusion
                        and not dynamic_blocked
                        and local_not_executable
                        and not local_route_revalidation_hold
                        and progress < 0.03
                    )
                    or bool(
                        route_unavailable_after_recent_static_block
                        and not local_route_revalidation_hold
                    )
                )
            )
            if route_unavailable_after_recent_static_block:
                local_not_executable = True
            request_certified_far_egress = False
            static_recovery_newly_armed = False
            if (
                static_blocked
                and not dynamic_blocked
                and not self.breadcrumb_motion_authority
            ):
                if self.visibility_static_blocked_since is None:
                    self.visibility_static_blocked_since = stamp
                elif (
                    not local_maneuver_active
                    # A FAR NO_ROUTE/acquisition hold is planning
                    # uncertainty, not a second physical obstruction.  Each
                    # retry counted toward certified reverse authority must
                    # be confirmed by a fresh local swept-footprint veto.
                    and local_static_blocked
                    and stamp - self.visibility_static_blocked_since
                    >= self.recovery.confirmation_s
                    and stamp - self.visibility_last_blocked_replan_stamp
                    >= self.visibility_replan_period
                ):
                    blocked_guidance_source = str(self.last_guidance_source)
                    failed_side = self.boundary.remember_current_failure()
                    self.boundary.reset(clear_failures=False)
                    self.last_boundary_decision = None
                    self.previous_goal_heading = None
                    self.last_heading_decision = None
                    # The local fallback has now produced real physical
                    # evidence that its direction cannot be executed.  Arm
                    # exactly one observed-prefix FAR handoff.  A FAR route
                    # which is itself blocked does not re-arm this bridge.
                    if blocked_guidance_source in (
                        "LOCAL_SAFE_EXPLORATION",
                        "RECENT_FAR_DIRECTION",
                    ):
                        static_recovery_newly_armed = bool(
                            not self.visibility_static_recovery_pending
                        )
                        self.visibility_static_recovery_pending = True
                    self.invalidate_visibility_route(
                        "confirmed_local_static_block"
                    )
                    self.visibility_last_blocked_replan_stamp = stamp
                    self.visibility_static_blocked_since = stamp
                    self.visibility_static_replan_failures += 1
                    nearest_trail = self.trail.nearest_index(*odom_pose[:2])
                    certified_trail_m = (
                        0.0
                        if nearest_trail is None
                        else self.trail.path_distance(0, nearest_trail)
                    )
                    request_certified_far_egress = bool(
                        self.far_static_egress_enabled
                        and not static_recovery_newly_armed
                        and self.visibility_static_replan_failures
                        >= self.far_static_replans_before_egress
                        and len(self.trail.points)
                        >= self.recovery.minimum_trail_points
                        and certified_trail_m
                        >= self.dead_end_escape_minimum_distance
                    )
                    if failed_side is not None:
                        rospy.loginfo(
                            "Visibility replan suppressed failed boundary side=%+d hit=(%.3f,%.3f)",
                            failed_side.side,
                            failed_side.x,
                            failed_side.y,
                        )
            else:
                self.visibility_static_blocked_since = None
                if progress >= 0.03:
                    self.visibility_static_replan_failures = 0
                    self.visibility_static_recovery_pending = False
                    # Meaningful physical progress invalidates any old local
                    # obstruction observation.  It must not be reused later
                    # merely because the evolving visibility graph briefly
                    # reports NO_ROUTE.
                    self.visibility_last_local_static_block_stamp = -math.inf
            previous_state = self.recovery.state
            state = self.recovery.update_goal_seek(
                stamp,
                progress_m=progress,
                static_blocked=static_blocked,
                dynamic_blocked=dynamic_blocked,
                rear_clear=float(features["rear"]) >= self.rear_clearance,
                trail_points=len(self.trail.points),
                maneuver_active=local_maneuver_active,
            )
            if request_certified_far_egress:
                self.recovery.force_certified_egress()
                self.start_backtrack(
                    odom_pose,
                    goal_odom,
                    stamp,
                    certified_far_egress=True,
                )
                self.publish_status(
                    self.recovery.state.value,
                    "repeated FAR replans remained statically blocked; "
                    "starting a signed, closed-loop exit along a bounded "
                    "certified ingress suffix",
                    failed_replans=self.visibility_static_replan_failures,
                    egress_limit_m=self.far_static_egress_maximum,
                )
                return
            if state == MemoryNavigationState.BACKTRACK_REVERSE and previous_state != state:
                failed_side = self.boundary.remember_current_failure()
                if failed_side is not None:
                    rospy.loginfo(
                        "Remembered failed boundary side=%+d hit=(%.3f,%.3f)",
                        failed_side.side,
                        failed_side.x,
                        failed_side.y,
                    )
                self.start_backtrack(odom_pose, goal_odom, stamp)
            if state in (MemoryNavigationState.GOAL_SEEK, MemoryNavigationState.SUSPECT_DEAD_END):
                goal_body = self.body_goal(map_pose, goal)
                if float(np.linalg.norm(goal_body)) < 0.05:
                    return
                body = self.forward_corridor_body(
                    goal, grid, map_pose, odom_pose, stamp
                )
                if body is None:
                    hard_route_hold = bool(
                        self.last_guidance_source in (
                            "FAR_ACQUISITION_HOLD",
                            "FAR_DENSE_REPLAN_PENDING",
                        )
                        and not self.visibility_active_route_motion_authorized
                    )
                    # Initial route settling is an intentional wait.  In
                    # either case the absence of a graph route is not physical
                    # static-block evidence.  Certified reverse egress is
                    # authorized only after the local planner has separately
                    # rejected every swept-footprint candidate on a route
                    # attempt.  This prevents ordinary SLAM/FAR graph churn
                    # on a clear road from commanding a breadcrumb reverse.
                    self.recovery.reset(active=True)
                    self.last_goal_improvement_stamp = stamp
                    self.visibility_static_blocked_since = None
                    self.publish_status(
                        "FAR_MAPPING_WAIT",
                        "holding while the online visibility route stabilizes",
                        route_acquisition_reason=(
                            self.visibility_route_acquisition_reason
                        ),
                        route_confirmations=(
                            self.visibility_route_acquisition.confirmations
                        ),
                        initial_local_exploration_distance_m=(
                            self.visibility_initial_exploration_distance_m
                        ),
                        initial_local_exploration_complete=(
                            self.visibility_initial_exploration_complete
                        ),
                        duplicate_map_updates_skipped=(
                            self.accumulated_map_duplicate_updates
                        ),
                        dense_replan_pending=bool(dense_replan_pending),
                        hard_route_hold=hard_route_hold,
                        static_evidence_created=False,
                    )
                    return
                decision = self.last_heading_decision
                boundary_decision = self.last_boundary_decision
                if boundary_decision.loop_detected:
                    if not self.breadcrumb_motion_authority:
                        # A Bug-style loop is evidence that the current local
                        # transaction failed, not permission to seize the
                        # gearbox.  Rebuild the FAR route on the latest SLAM
                        # map and let the Ackermann local planner decide any
                        # finite reverse/turnaround manoeuvre it requires.
                        self.invalidate_visibility_route(
                            "boundary_loop_%s" % boundary_decision.reason
                        )
                        self.previous_goal_heading = None
                        self.last_heading_decision = None
                        body = self.forward_corridor_body(
                            goal, grid, map_pose, odom_pose, stamp
                        )
                        if body is None:
                            self.publish_status(
                                "FAR_MAPPING_WAIT",
                                "boundary branch failed; holding while FAR "
                                "reacquires a stable route",
                                route_acquisition_reason=(
                                    self.visibility_route_acquisition_reason
                                ),
                            )
                            return
                        decision = self.last_heading_decision
                        boundary_decision = self.last_boundary_decision
                    elif (
                        float(features["rear"]) >= self.rear_clearance
                        and len(self.trail.points)
                        >= self.recovery.minimum_trail_points
                    ):
                        self.recovery.force_backtrack()
                        self.start_backtrack(odom_pose, goal_odom, stamp)
                        self.publish_status(
                            self.recovery.state.value,
                            "boundary loop closed without a valid leave corridor; replaying current-goal history",
                            boundary_loop_reason=boundary_decision.reason,
                            boundary_travelled_m=boundary_decision.travelled_m,
                        )
                        return
                    else:
                        self.recovery.fail_safe()
                        self.publish_status(
                            "SAFE_STOP",
                            "boundary loop detected but no certified reverse trail is available",
                            boundary_loop_reason=boundary_decision.reason,
                        )
                        return
                boundary_active = bool(boundary_decision.active)
                route_transaction_mode = (
                    LocalRouteCommand.NAVIGATION_CONNECTIVITY
                    if dense_replan_pending
                    and self.last_guidance_source == "LOCAL_SAFE_EXPLORATION"
                    else
                    LocalRouteCommand.NAVIGATION_MEMORY_GOAL
                    if self.visibility_active_route_motion_authorized
                    or self.last_guidance_source in (
                        "EXPLORED_TOPOLOGY",
                        "KNOWN_TERMINAL_DIRECT",
                        "LOCAL_SAFE_EXPLORATION",
                        "FAILED_BRANCH_EXIT_LOCK",
                    )
                    else LocalRouteCommand.NAVIGATION_CONNECTIVITY
                )
                self.publish_route(
                    self.body_to_map(body, map_pose),
                    Gear.FORWARD,
                    route_transaction_mode,
                    state.value,
                    (
                        "emergency obstacle-boundary following under visibility routing"
                        if boundary_active
                        else (
                            "dynamic visibility route handed to the DE-P local planner"
                            if self.last_guidance_source in (
                                "FAR_KNOWN_VISIBILITY",
                                "FAR_ATTEMPTABLE_VISIBILITY",
                                "FAR_ATTEMPTABLE_NAVIGATION",
                                "FAR_PARTIAL_ATTEMPTABLE",
                            )
                            else (
                                "rolling LiDAR-safe local exploration while FAR has no route"
                                if self.last_guidance_source
                                == "LOCAL_SAFE_EXPLORATION"
                                else "explored topology handed to the DE-P local planner"
                            )
                        )
                    ),
                    goal_bearing_rad=float(self.last_mission_goal_bearing),
                    guidance_bearing_rad=float(decision.goal_bearing),
                    local_heading_rad=float(decision.angle),
                    local_heading_sector=decision.sector,
                    local_heading_blocked=bool(decision.blocked),
                    local_heading_clearance_m=float(decision.clearance),
                    failed_branch_count=len(self.topology.failed_branches),
                    boundary_mode=boundary_decision.mode.value,
                    boundary_reason=boundary_decision.reason,
                    boundary_travelled_m=boundary_decision.travelled_m,
                    boundary_direct_clearance_m=(
                        boundary_decision.direct_clearance_m
                    ),
                    map_acquisition_forward_only=bool(
                        route_transaction_mode
                        == LocalRouteCommand.NAVIGATION_CONNECTIVITY
                    ),
                    breadcrumb_motion_authority=bool(
                        self.breadcrumb_motion_authority
                    ),
                    visibility_replan_reason=self.visibility_last_replan_reason,
                    rolling_target_world=(
                        self.visibility_route_cursor.last_observation.carrot_xy
                        if self.last_guidance_source in (
                            "FAR_KNOWN_VISIBILITY",
                            "FAR_ATTEMPTABLE_VISIBILITY",
                            "FAR_ATTEMPTABLE_NAVIGATION",
                            "FAR_PARTIAL_ATTEMPTABLE",
                        )
                        and self.visibility_route_cursor.last_observation
                        is not None
                        and not boundary_active
                        else self.odom_to_map([
                            self.topology_route_cursor.last_observation.carrot_xy
                        ])[0]
                        if self.last_guidance_source == "EXPLORED_TOPOLOGY"
                        and self.topology_route_cursor.last_observation
                        is not None
                        and not boundary_active
                        else None
                    ),
                )
                return

        if self.recovery.state in (
            MemoryNavigationState.BACKTRACK_REVERSE,
            MemoryNavigationState.FAR_DEAD_END_EGRESS,
        ):
            certified_dead_end_egress = bool(
                self.recovery.state
                == MemoryNavigationState.FAR_DEAD_END_EGRESS
            )
            if certified_dead_end_egress:
                # The exclusive short-horizon egress objective owns motion
                # until its external anchor.  Its connector gear is preferred,
                # but the local planner may insert one hard-safe opposite-gear
                # realignment leg; that does not transfer mission authority.
                # Running the synchronous FAR graph solve here previously
                # blocked this 5 Hz callback for several seconds while the
                # vehicle continued moving on an old command, then falsely
                # tripped localization divergence.  FAR is rebuilt immediately
                # after the anchor is reached.
                pass
            egress_bidirectional_safety_exhausted = bool(
                certified_dead_end_egress
                and local_state is not None
                and not local_state.maneuver_active
                and (
                    "all shortened forward/reverse primitives failed "
                    "continuous hard safety"
                    in str(local_state.detail).lower()
                )
            )
            backtrack_static_blocked = bool(
                local_state is not None
                and local_state.blocked_by_static
                and not local_state.blocked_by_dynamic
                and (
                    not certified_dead_end_egress
                    or egress_bidirectional_safety_exhausted
                )
            )
            backtrack_grace_elapsed = bool(
                self.backtrack_started_stamp is not None
                and stamp - self.backtrack_started_stamp >= 0.75
            )
            if backtrack_static_blocked and backtrack_grace_elapsed:
                if self.backtrack_blocked_since is None:
                    self.backtrack_blocked_since = stamp
                elif (
                    stamp - self.backtrack_blocked_since
                    >= self.far_static_egress_block_confirmation
                ):
                    if self.far_emergency_egress_active:
                        nearest = self.trail.nearest_index(*odom_pose[:2])
                        if nearest is not None:
                            self.trail.truncate_after(nearest)
                        self.far_emergency_egress_active = False
                        self.backtrack_started_stamp = None
                        self.backtrack_blocked_since = None
                        self.dead_end_escape_completion_reason = (
                            "EGRESS_BLOCKED_BY_CURRENT_HARD_SAFETY"
                        )
                        self.recovery.fail_safe()
                        self.publish_status(
                            "EGRESS_BLOCKED",
                            "FAR dead-end exit exhausted both the connector "
                            "gear and its opposite-gear realignment under "
                            "continuous hard safety",
                            escape_id=self.dead_end_escape_id,
                            failed_branch_id=self.dead_end_escape_branch_id,
                            bidirectional_hard_safety_exhausted=True,
                        )
                    else:
                        self.backtrack_started_stamp = None
                        self.backtrack_blocked_since = None
                        self.recovery.fail_safe()
                        self.publish_status(
                            "SAFE_STOP",
                            "breadcrumb reverse corridor failed continuous hard safety",
                        )
                    return
            else:
                self.backtrack_blocked_since = None
            far_route_ready = bool(
                certified_dead_end_egress
                and self.visibility_active_route_motion_authorized
                and self.visibility_plan is not None
                and self.visibility_plan.status == "PASS"
            )
            topology_exit = (
                self.topology.guidance_path(
                    odom_pose[:2], goal_odom, stamp=stamp
                )
                if certified_dead_end_egress
                else []
            )
            transaction = self.backtrack_route(
                odom_pose,
                map_pose,
                features,
                far_route_ready=far_route_ready,
                topology_exit_ready=len(topology_exit) >= 2,
                stamp=stamp,
            )
            if transaction is not None:
                route, replay_gear = transaction
                rolling_target = (
                    self.point_along_polyline(
                        route, self.dead_end_escape_lookahead
                    )
                    if certified_dead_end_egress
                    else None
                )
                self.publish_route(
                    route,
                    replay_gear,
                    (
                        LocalRouteCommand.NAVIGATION_FAR_DEAD_END_EGRESS
                        if certified_dead_end_egress
                        else LocalRouteCommand.NAVIGATION_MEMORY_BACKTRACK
                    ),
                    self.recovery.state.value,
                    (
                        "FAR-directed closed-loop exit toward the failed "
                        "branch entry"
                        if certified_dead_end_egress
                        else "replaying certified breadcrumbs toward recovery site"
                    ),
                    rolling_target_world=rolling_target,
                    route_id=(
                        "far_dead_end_egress:%d" % self.dead_end_escape_id
                        if certified_dead_end_egress
                        else None
                    ),
                    route_source=(
                        "FAR_DEAD_END_EGRESS"
                        if certified_dead_end_egress
                        else "BREADCRUMB_BACKTRACK"
                    ),
                    route_revision=(
                        self.dead_end_escape_route_revision
                        if certified_dead_end_egress
                        else 0
                    ),
                    escape_id=(
                        self.dead_end_escape_id
                        if certified_dead_end_egress
                        else 0
                    ),
                    failed_branch_id=(
                        self.dead_end_escape_branch_id
                        if certified_dead_end_egress
                        else None
                    ),
                    egress_site_kind=self.dead_end_escape_site_kind,
                    egress_target_distance_m=(
                        self.dead_end_escape_target_distance_m
                    ),
                    egress_cross_track_error_m=(
                        self.dead_end_escape_cross_track_error_m
                    ),
                    background_far_ready=far_route_ready,
                    background_far_status=(
                        self.visibility_plan.status
                        if self.visibility_plan is not None
                        else "NONE"
                    ),
                    map_reanchors=max(
                        0,
                        self.map_correction_generation
                        - self.dead_end_escape_started_map_correction_generation,
                    ),
                )
                return
            if (
                certified_dead_end_egress
                and self.recovery.state
                == MemoryNavigationState.FAR_DEAD_END_EGRESS
            ):
                # No connector is currently certified.  Publish an explicit
                # wait state so the local controller brakes instead of
                # continuing the previous reverse command while SLAM/LiDAR
                # gets the bounded revalidation interval above.
                self.publish_status(
                    "EGRESS_REANCHOR_WAIT",
                    "holding while the dead-end exit connector is "
                    "revalidated on the current occupancy map",
                    escape_id=self.dead_end_escape_id,
                    failed_branch_id=self.dead_end_escape_branch_id,
                    egress_cross_track_error_m=(
                        self.dead_end_escape_cross_track_error_m
                    ),
                    connector_unavailable_since=(
                        self.dead_end_escape_connector_unavailable_since
                    ),
                )
                return

        if self.recovery.state == MemoryNavigationState.RESUME_FORWARD:
            resume_exhausted = bool(
                self.resume_travelled_m >= self.resume_maximum_travel
                or (
                    self.resume_started_stamp is not None
                    and stamp - self.resume_started_stamp >= self.resume_timeout
                )
            )
            if resume_exhausted:
                failed_site = self.backtrack_site_index
                exhausted_travel = self.resume_travelled_m
                exhausted_elapsed = (
                    None
                    if self.resume_started_stamp is None
                    else stamp - self.resume_started_stamp
                )
                self.recovery.resume_failed()
                self.start_backtrack(
                    odom_pose,
                    goal_odom,
                    stamp,
                    before_site_index=failed_site,
                )
                self.publish_status(
                    "BACKTRACK_REVERSE",
                    "resume transaction made no certified outbound progress; continuing to an older site",
                    failed_site_index=failed_site,
                    resume_travelled_m=exhausted_travel,
                    resume_elapsed_s=exhausted_elapsed,
                )
                return
            resume_static_blocked = bool(
                local_state is not None
                and local_state.blocked_by_static
                and not local_state.blocked_by_dynamic
            )
            if resume_static_blocked:
                if self.resume_blocked_since is None:
                    self.resume_blocked_since = stamp
                elif stamp - self.resume_blocked_since >= self.resume_failure_confirmation:
                    failed_site = self.backtrack_site_index
                    self.recovery.resume_failed()
                    self.start_backtrack(
                        odom_pose,
                        goal_odom,
                        stamp,
                        before_site_index=failed_site,
                    )
                    self.publish_status(
                        "BACKTRACK_REVERSE",
                        "recovery site failed Ackermann safety; continuing to an older certified site",
                        failed_site_index=failed_site,
                    )
                    return
            else:
                self.resume_blocked_since = None
            route = self.resume_route(odom_pose, stamp)
            if route is not None:
                self.publish_route(
                    route,
                    Gear.FORWARD,
                    LocalRouteCommand.NAVIGATION_MEMORY_RESUME,
                    "RESUME_FORWARD",
                    "turning toward the certified outbound trail",
                    resume_target_index=self.resume_target_index,
                    resume_travelled_m=self.resume_travelled_m,
                )
                return

        if self.recovery.state == MemoryNavigationState.DEAD_END_ESCAPED:
            # Recovery is an internal navigation transaction, not a terminal
            # mission state.  Remove the abandoned dead-end suffix from the
            # driven-path memory and continue toward the same position goal.
            nearest = self.trail.nearest_index(*odom_pose[:2])
            if nearest is not None:
                self.trail.truncate_after(nearest)
            self.topology.reanchor(*odom_pose[:2], maximum_distance_m=1.5)
            self.record_breadcrumb(
                odom_pose,
                stamp,
                features,
                motion_direction=0,
            )
            self.backtrack_start_index = None
            self.backtrack_cursor_index = None
            self.backtrack_site_index = None
            self.backtrack_start_xy = None
            self.resume_start_xy = None
            self.resume_target_index = None
            self.resume_fallback_target_xy = None
            self.resume_started_stamp = None
            self.resume_travelled_m = 0.0
            self.resume_last_xy = None
            self.resume_blocked_since = None
            self.previous_goal_heading = None
            self.last_heading_decision = None
            self.last_boundary_decision = None
            self.boundary.reset()
            self.committed_topology_path = []
            self.topology_route_cursor.reset()
            self.topology_last_carrot_odom = None
            self.best_goal_distance = goal_distance
            self.last_goal_improvement_stamp = stamp
            self.recovery.continue_goal_seek()
            self.publish_status(
                "RECOVERED_GOAL_SEEK",
                "dead end escaped; continuing the current position goal",
                abandoned_suffix_pruned=True,
                failed_branch_retained=True,
            )
        elif self.recovery.state == MemoryNavigationState.SAFE_STOP:
            recoverable_egress_stop = bool(
                self.goal is not None
                and self.dead_end_escape_completion_reason in (
                    "EGRESS_REANCHOR_EXHAUSTED_CURRENT_MAP",
                    "EGRESS_LOCALIZATION_DIVERGED",
                    "EGRESS_DISTANCE_LIMIT",
                    "EGRESS_EXHAUSTED",
                )
            )
            recovery_candidate = (
                self.visibility_last_candidate_plan
                if self.visibility_last_candidate_plan is not None
                else self.visibility_plan
            )
            if (
                recoverable_egress_stop
                and self.far_recovery_prefix_authorized(
                    recovery_candidate,
                    map_pose,
                    goal,
                    stamp,
                    require_failed_branch_avoidance=True,
                )
                and self.bind_far_recovery_candidate(
                    recovery_candidate,
                    map_pose,
                    goal,
                    "branch_safe_far_route_recovered_egress_stop",
                )
            ):
                previous_reason = self.dead_end_escape_completion_reason
                self.recovery.reset(active=True)
                self.far_emergency_egress_active = False
                self.backtrack_started_stamp = None
                self.backtrack_blocked_since = None
                self.dead_end_escape_diverged_since = None
                self.dead_end_escape_live_target_index = None
                self.dead_end_escape_completion_reason = (
                    "RECOVERED_BY_BRANCH_SAFE_FAR_ROUTE"
                )
                self.visibility_initial_exploration_complete = True
                self.visibility_mapping_session_established = True
                self.visibility_initial_exploration_reason = (
                    "branch_safe_far_route_recovered_egress_stop"
                )
                self.publish_status(
                    "RECOVERED_GOAL_SEEK",
                    "a locally certified branch-safe FAR route replaced the "
                    "transient dead-end egress stop",
                    previous_egress_completion_reason=previous_reason,
                    escape_id=self.dead_end_escape_id,
                    failed_branch_id=self.dead_end_escape_branch_id,
                )
                return
            self.publish_status(
                "SAFE_STOP",
                (
                    "FAR dead-end egress stopped safely: %s"
                    % self.dead_end_escape_completion_reason
                    if self.dead_end_escape_completion_reason.startswith(
                        "EGRESS_"
                    )
                    else "breadcrumb recovery exhausted without a certified motion corridor"
                ),
                escape_id=self.dead_end_escape_id,
                failed_branch_id=self.dead_end_escape_branch_id,
                egress_completion_reason=(
                    self.dead_end_escape_completion_reason
                ),
            )


if __name__ == "__main__":
    rospy.init_node("dep_car_memory_navigation")
    NavigationMemoryNode()
    rospy.spin()
