#!/usr/bin/env python3
"""P6 local authority: deterministic gear/safety around learned candidates."""

import json
import math
import threading
from collections import Counter

import numpy as np
import rospy
from dep_car.core.gear import GearSupervisor
from dep_car.runtime.occupancy import RuntimeOccupancyGrid2D
from dep_car.runtime.route_guidance import (
    apply_corner_clearance_preference,
    apply_runtime_route_preference,
    corner_severity,
    corner_speed_limit,
    monotonic_route_reference_body,
    route_reference_body,
    route_turn_angle,
    segment_is_visible,
)
from dep_car.runtime.forward_preference import (
    ForwardPreferenceConfig,
    ForwardPreferenceState,
    ForwardPreferenceSupervisor,
    corridor_direction_body,
    navigation_authority_reference,
    route_requires_far_revalidation,
    terminal_capture_route_authorized,
)
from dep_car.core.planner import DeterministicPlanner
from dep_car.core.recovery import RecoveryManager, RecoveryState
from dep_car.core.types import Candidate as CoreCandidate
from dep_car.core.types import DynamicTrack, Gear, VehicleState
from dep_car.core.vehicle import (
    center_steering_from_wheel_angles,
    world_velocity_to_body_longitudinal,
)
from dep_car.runtime.p6_contract import (
    V42_EXECUTION_ARCHITECTURE_ID,
    V43_ARCHITECTURE_ID,
)
from dep_car.runtime.hybrid_execution import (
    HybridFirstActionLatch,
    align_trajectory_between_chassis_frames,
)
from dep_car.runtime.safety import (
    evaluate_hybrid_sequence_candidate_bank,
    evaluate_learned_candidate_bank,
)
from dep_car.runtime.arrival import ArrivalConfig, GoalArrivalController
from dep_car.runtime.maneuver import (
    CommittedManeuver,
    ManeuverConfig,
    ManeuverState,
    MeasuredPoseReplanGate,
    RouteRecoveryReplanGate,
)
from dep_car_msgs.msg import (
    AckermannCommand,
    AckermannRoute,
    Candidate,
    CandidateArray,
    DynamicTrackArray,
    LocalRouteCommand,
    PlannerState,
    PolicyCandidateArray,
    PolicyQuery,
    PolicyState,
)
from geometry_msgs.msg import Pose2D, PoseStamped
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from visualization_msgs.msg import Marker


def yaw_from_quaternion(quaternion):
    return math.atan2(
        2.0 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y),
        1.0 - 2.0 * (quaternion.y * quaternion.y + quaternion.z * quaternion.z),
    )


def wrap_angle(value):
    return math.atan2(math.sin(value), math.cos(value))


class LocalPlannerNode:
    def __init__(self):
        self.lock = threading.Lock()
        self.planner = DeterministicPlanner()
        self.recovery = RecoveryManager()
        self.gear_supervisor = GearSupervisor()
        self.hybrid_action_latch = HybridFirstActionLatch()
        self.grid = self.grid_stamp = self.joint_state = None
        self.odom = self.goal = self.mission_goal = self.route_command = self.route = None
        # Route geometry and its command are published on separate ROS topics.
        # Keep a bounded stamp-keyed buffer for each half: with a busy FAR
        # callback, route N+1 can arrive before command N and a single pending
        # slot phase-locks forever on adjacent generations.
        self.pending_routes = {}
        self.pending_route_commands = {}
        self.deferred_route_transaction = None
        self.route_transaction_buffer_limit = 16
        self.policy_raw = self.policy_inference_state = None
        self.policy_query_poses = {}
        self.control_pose = None
        self.global_planner_state = "IDLE"
        self.global_route_id = ""
        self.global_route_source = "NONE"
        self.global_route_revision = 0
        self.global_route_progress_m = 0.0
        self.global_route_carrot_m = 0.0
        self.route_authority_epoch = 0
        self.global_turnaround_transaction_id = 0
        self.route_reference_authority_epoch = 0
        self.route_reference_index = 0
        self.local_turnaround_transaction_sequence = 0
        self.local_turnaround_transaction_id = 0
        self.global_wait_states = frozenset((
            "RECEIVED",
            "PLANNING",
            "TIMEOUT",
            "NO_PATH",
            "INVALID_GOAL",
            "INVALID_START",
            "START_BLOCKED",
            "INVALID_STATUS",
            "MAPPING_WAIT",
            "FAR_MAPPING_WAIT",
            "SAFE_STOP",
            "MEMORY_HOLD",
        ))
        self.forward_capture_replan_requested = False
        # A course-capture revalidation is a route transaction barrier.  The
        # latch may only be released by a synchronized FAR transaction newer
        # than the request (or by an explicit global wait transition).  This
        # prevents an old route callback from releasing the stop, while also
        # handling FAR replans which complete too quickly to expose a
        # FAR_MAPPING_WAIT status to this node.
        self.forward_capture_replan_after_stamp = -math.inf
        self.tracks = []
        self.last_position = None
        self.query_generation = 0
        self.last_runtime_status = None
        self.mission_goal_tolerance = float(rospy.get_param("~mission_goal_tolerance", 0.22))
        self.mission_heading_tolerance = float(rospy.get_param("~mission_heading_tolerance", 0.35))
        self.require_mission_goal_heading = bool(
            rospy.get_param("~require_goal_heading", True)
        )
        self.arrival = GoalArrivalController(
            ArrivalConfig(
                position_tolerance_m=self.mission_goal_tolerance,
                heading_tolerance_rad=self.mission_heading_tolerance,
                settled_speed_mps=float(
                    rospy.get_param("~goal_settled_speed", 0.02)
                ),
                approach_radius_m=float(rospy.get_param("~approach_radius", 1.5)),
                comfortable_deceleration_mps2=float(
                    rospy.get_param("~approach_deceleration", 0.8)
                ),
                maximum_approach_speed_mps=float(
                    rospy.get_param("~approach_maximum_speed", 0.45)
                ),
            )
        )
        maneuver_leg_distance = float(
            rospy.get_param(
                "~maneuver_leg_distance",
                rospy.get_param("~reverse_distance", 0.85),
            )
        )
        self.maneuver = CommittedManeuver(
            ManeuverConfig(base_leg_distance_m=maneuver_leg_distance)
        )
        self.forward_preference = ForwardPreferenceSupervisor(
            ForwardPreferenceConfig(
                forward_reacquired_bearing_rad=float(
                    rospy.get_param(
                        "~forward_reacquired_bearing",
                        math.radians(75.0),
                    )
                ),
                forward_capture_maximum_bearing_rad=float(
                    rospy.get_param(
                        "~forward_capture_maximum_bearing",
                        math.radians(145.0),
                    )
                ),
                forward_capture_divergence_rad=float(
                    rospy.get_param(
                        "~forward_capture_divergence",
                        math.radians(18.0),
                    )
                ),
                behind_confirmation_cycles=int(
                    rospy.get_param("~turnaround_behind_confirmation_cycles", 8)
                ),
                forward_confirmation_cycles=int(
                    rospy.get_param("~forward_reacquired_confirmation_cycles", 8)
                ),
            )
        )
        self.maneuver_spatial_scales = (1.0, 0.75, 0.50, 0.35, 0.25)
        self.maneuver_minimum_yaw_progress = float(
            rospy.get_param("~maneuver_minimum_yaw_progress", 0.035)
        )
        self.turnaround_alignment_release_bearing = float(
            rospy.get_param(
                "~turnaround_alignment_release_bearing",
                math.radians(50.0),
            )
        )
        self.turnaround_alignment_minimum_spatial_scale = float(
            rospy.get_param(
                "~turnaround_alignment_minimum_spatial_scale", 0.75
            )
        )
        if not 0.0 < self.turnaround_alignment_release_bearing < 0.5 * math.pi:
            raise ValueError(
                "turnaround_alignment_release_bearing must be between 0 and pi/2"
            )
        if not 0.0 < self.turnaround_alignment_minimum_spatial_scale <= 1.0:
            raise ValueError(
                "turnaround_alignment_minimum_spatial_scale must be in (0,1]"
            )
        self.maneuver_retry_observation_hold = float(
            rospy.get_param("~maneuver_retry_observation_hold_s", 0.75)
        )
        if self.maneuver_retry_observation_hold < 0.0:
            raise ValueError("maneuver_retry_observation_hold_s must be nonnegative")
        self.maneuver_retry_not_before = 0.0
        self.turnaround_rearm_distance = float(
            rospy.get_param("~turnaround_rearm_forward_distance_m", 1.25)
        )
        if self.turnaround_rearm_distance <= 0.0:
            raise ValueError("turnaround rearm distance must be positive")
        self.turnaround_rearm_remaining = 0.0
        self.forward_restoration_budget_replans = 0
        self.forward_restoration_budget_replan_pending = False
        self.exact_replan_gate = MeasuredPoseReplanGate(
            float(rospy.get_param("~exact_replan_minimum_displacement", 0.25))
        )
        self.static_recovery_replan_gate = RouteRecoveryReplanGate(
            float(
                rospy.get_param(
                    "~static_recovery_replan_minimum_displacement", 0.25
                )
            )
        )
        self.maneuver_forward_speed = float(
            rospy.get_param("~maneuver_forward_speed", 0.45)
        )
        self.maneuver_reverse_speed = float(
            rospy.get_param("~maneuver_reverse_speed", 0.35)
        )
        self.forward_course_capture_speed = float(
            rospy.get_param("~forward_course_capture_speed", 0.35)
        )
        if self.forward_course_capture_speed <= 0.0:
            raise ValueError("forward_course_capture_speed must be positive")
        self.forward_course_revalidation_timeout = float(
            rospy.get_param("~forward_course_revalidation_timeout_s", 2.5)
        )
        if self.forward_course_revalidation_timeout <= 0.0:
            raise ValueError(
                "forward_course_revalidation_timeout_s must be positive"
            )
        self.memory_margin_egress_maximum_overlap = float(
            rospy.get_param("~memory_margin_egress_maximum_overlap", 0.06)
        )
        self.memory_margin_egress_minimum_improvement = float(
            rospy.get_param("~memory_margin_egress_minimum_improvement", 0.02)
        )
        self.memory_margin_egress_worsening_tolerance = float(
            rospy.get_param("~memory_margin_egress_worsening_tolerance", 0.01)
        )
        if min(
            self.memory_margin_egress_maximum_overlap,
            self.memory_margin_egress_minimum_improvement,
            self.memory_margin_egress_worsening_tolerance,
        ) < 0.0:
            raise ValueError("memory margin-egress thresholds cannot be negative")
        self.terminal_maneuver_radius = float(
            rospy.get_param("~terminal_maneuver_radius", 1.50)
        )
        self.terminal_maneuver_heading_trigger = float(
            rospy.get_param("~terminal_maneuver_heading_trigger", 0.55)
        )
        self.terminal_capture_radius = float(
            rospy.get_param("~terminal_capture_radius", 0.80)
        )
        self.terminal_capture_speed = float(
            rospy.get_param("~terminal_capture_speed", 0.22)
        )
        self.exact_route_speed = float(
            rospy.get_param("~exact_route_speed", 0.30)
        )
        self.route_reference_horizon = float(
            rospy.get_param("~route_reference_horizon", 2.50)
        )
        self.route_corridor_weight = float(
            rospy.get_param("~route_corridor_weight", 3.0)
        )
        self.route_clearance_target = float(
            rospy.get_param("~route_clearance_target", 0.15)
        )
        self.route_clearance_weight = float(
            rospy.get_param("~route_clearance_weight", 1.5)
        )
        self.corner_corridor_minimum_scale = float(
            rospy.get_param("~corner_corridor_minimum_scale", 0.25)
        )
        self.corner_soft_clearance_target = float(
            rospy.get_param("~corner_soft_clearance_target", 0.30)
        )
        self.corner_soft_clearance_weight = float(
            rospy.get_param("~corner_soft_clearance_weight", 3.0)
        )
        self.corner_soft_trigger = float(
            rospy.get_param("~corner_soft_trigger", 0.35)
        )
        self.corner_soft_full_strength = float(
            rospy.get_param("~corner_soft_full_strength", 1.20)
        )
        self.corner_straight_speed = float(
            rospy.get_param("~corner_straight_speed", 0.55)
        )
        self.corner_ninety_speed = float(
            rospy.get_param("~corner_ninety_speed", 0.26)
        )
        if self.route_reference_horizon <= 0.0:
            raise ValueError("route_reference_horizon must be positive")
        if min(
            self.route_corridor_weight,
            self.route_clearance_target,
            self.route_clearance_weight,
            self.corner_soft_clearance_target,
            self.corner_soft_clearance_weight,
            self.corner_straight_speed,
            self.corner_ninety_speed,
        ) < 0.0:
            raise ValueError("route guidance weights, clearances and speeds cannot be negative")
        if not 0.0 <= self.corner_corridor_minimum_scale <= 1.0:
            raise ValueError("corner_corridor_minimum_scale must be in [0,1]")
        if not 0.0 <= self.corner_soft_trigger < self.corner_soft_full_strength <= math.pi:
            raise ValueError("corner soft-clearance angle thresholds are invalid")
        self.policy_mode = str(rospy.get_param("~policy_mode", "disabled"))
        if self.policy_mode not in ("disabled", "shadow", "guarded", "active"):
            raise ValueError(
                "policy_mode must be disabled, shadow, guarded or active"
            )
        self.active_fallback_to_baseline = bool(
            rospy.get_param("~active_fallback_to_baseline", False)
        )
        self.policy_freshness = float(rospy.get_param("~policy_freshness", 0.35))
        self.policy_grid_skew = float(rospy.get_param("~policy_grid_skew", 0.15))
        self.policy_subgoal_tolerance = float(
            rospy.get_param("~policy_subgoal_tolerance", 0.50)
        )
        self.learned_route_authority = bool(
            rospy.get_param("~learned_route_authority", False)
        )
        self.odometry_twist_in_body_frame = bool(
            rospy.get_param("~odometry_twist_in_body_frame", False)
        )
        self.route_transaction_stamp_tolerance = float(
            rospy.get_param("~route_transaction_stamp_tolerance", 0.02)
        )
        if self.route_transaction_stamp_tolerance < 0.0:
            raise ValueError("route_transaction_stamp_tolerance cannot be negative")
        self.command_pub = rospy.Publisher(
            "/dep_car/cmd_ackermann", AckermannCommand, queue_size=1
        )
        self.candidates_pub = rospy.Publisher(
            "/dep_car/candidates", CandidateArray, queue_size=1
        )
        self.policy_candidates_pub = rospy.Publisher(
            "/dep_car/policy_candidates", CandidateArray, queue_size=1
        )
        # V4.3 DAgger collection observes the guarded policy state while a
        # deterministic, route-aware expert labels both transmission choices.
        # These publishers are disabled in every normal P6 launch and never
        # participate in runtime candidate selection or actuation.
        self.publish_dagger_teacher_banks = bool(
            rospy.get_param("~publish_dagger_teacher_banks", False)
        )
        self.dagger_teacher_forward_pub = rospy.Publisher(
            "/dep_car/dagger_teacher_forward", CandidateArray, queue_size=1
        )
        self.dagger_teacher_reverse_pub = rospy.Publisher(
            "/dep_car/dagger_teacher_reverse", CandidateArray, queue_size=1
        )
        self.selected_path_pub = rospy.Publisher(
            "/dep_car/selected_path", Path, queue_size=1
        )
        self.policy_selected_path_pub = rospy.Publisher(
            "/dep_car/policy_selected_path", Path, queue_size=1
        )
        self.state_pub = rospy.Publisher(
            "/dep_car/planner_state", PlannerState, queue_size=1
        )
        self.policy_query_pub = rospy.Publisher(
            "/dep_car/policy_query", PolicyQuery, queue_size=1
        )
        self.policy_state_pub = rospy.Publisher(
            "/dep_car/policy_state", PolicyState, queue_size=1, latch=True
        )
        self.status_marker_pub = rospy.Publisher(
            "/dep_car/local_planner_status_marker", Marker, queue_size=1, latch=True
        )
        self.replan_pub = rospy.Publisher(
            "/dep_car/replan_request", String, queue_size=1
        )
        rospy.Subscriber("/dep_car/local_costmap", OccupancyGrid, self.on_grid, queue_size=1)
        rospy.Subscriber(
            rospy.get_param("~odometry_topic", "/base_pose_ground_truth"),
            Odometry,
            self.on_odom,
            queue_size=1,
        )
        rospy.Subscriber(
            "/urban_model/joint_states", JointState, self.on_joint_state, queue_size=1
        )
        rospy.Subscriber(
            rospy.get_param("~goal_topic", "/dep_car/local_subgoal"),
            PoseStamped,
            self.on_goal,
            queue_size=1,
        )
        rospy.Subscriber(
            "/dep_car/local_route_command",
            LocalRouteCommand,
            self.on_route_command,
            queue_size=1,
        )
        rospy.Subscriber("/dep_car/global_route", AckermannRoute, self.on_route, queue_size=1)
        rospy.Subscriber(
            "/dep_car/global_planner_status",
            String,
            self.on_global_planner_status,
            queue_size=1,
        )
        rospy.Subscriber(
            "/move_base_simple/goal", PoseStamped, self.on_mission_goal, queue_size=1
        )
        rospy.Subscriber(
            "/dep_car/dynamic/tracks", DynamicTrackArray, self.on_tracks, queue_size=1
        )
        rospy.Subscriber(
            "/dep_car/policy_candidates_raw",
            PolicyCandidateArray,
            self.on_policy_raw,
            queue_size=1,
        )
        rospy.Subscriber(
            "/dep_car/policy_inference_state",
            PolicyState,
            self.on_policy_inference_state,
            queue_size=1,
        )
        rate = float(rospy.get_param("~control_rate", 10.0))
        if rate <= 0.0:
            raise ValueError("control_rate must be positive")
        self.timer = rospy.Timer(rospy.Duration(1.0 / rate), self.update)

    def on_grid(self, message):
        data = np.asarray(message.data, dtype=np.int16).reshape(
            (message.info.height, message.info.width)
        )
        grid = RuntimeOccupancyGrid2D(
            data,
            message.info.resolution,
            (message.info.origin.position.x, message.info.origin.position.y),
        )
        stamp = message.header.stamp.to_sec() or rospy.Time.now().to_sec()
        with self.lock:
            self.grid = grid
            self.grid_stamp = stamp

    def on_odom(self, message):
        with self.lock:
            self.odom = message

    def on_joint_state(self, message):
        with self.lock:
            self.joint_state = message

    def on_goal(self, message):
        with self.lock:
            # Once route transactions are active, LocalRouteCommand owns the
            # corresponding target.  Accepting this independently published
            # compatibility topic for one callback cycle can pair a new
            # subgoal with an old route.
            if self.route_command is None:
                self.goal = message

    @staticmethod
    def message_stamp(message):
        return float(message.header.stamp.to_sec())

    def buffer_route_transaction_half_locked(self, buffer, message):
        stamp = self.message_stamp(message)
        buffer[stamp] = message
        while len(buffer) > self.route_transaction_buffer_limit:
            del buffer[min(buffer)]

    def matching_route_transaction_locked(self):
        if not self.pending_routes or not self.pending_route_commands:
            return None
        matches = [
            (
                abs(route_stamp - command_stamp),
                max(route_stamp, command_stamp),
                route_stamp,
                command_stamp,
            )
            for route_stamp in self.pending_routes
            for command_stamp in self.pending_route_commands
            if (
                route_stamp <= 0.0
                or command_stamp <= 0.0
                or abs(route_stamp - command_stamp)
                <= self.route_transaction_stamp_tolerance
            )
        ]
        if not matches:
            rospy.logwarn_throttle(
                2.0,
                "Buffered unmatched FAR route transaction halves "
                "routes=%d commands=%d newest_route=%.6f newest_command=%.6f",
                len(self.pending_routes),
                len(self.pending_route_commands),
                max(self.pending_routes),
                max(self.pending_route_commands),
            )
            return None
        # Prefer an exact pair, and among equally close pairs consume the most
        # recent transaction.  Older unmatched halves remain available until
        # the bounded buffer prunes them.
        _, _, route_stamp, command_stamp = min(
            matches, key=lambda row: (row[0], -row[1])
        )
        route = self.pending_routes.pop(route_stamp)
        command = self.pending_route_commands.pop(command_stamp)
        cutoff = (
            min(route_stamp, command_stamp)
            - self.route_transaction_stamp_tolerance
        )
        self.pending_routes = {
            stamp: value
            for stamp, value in self.pending_routes.items()
            if stamp >= cutoff
        }
        self.pending_route_commands = {
            stamp: value
            for stamp, value in self.pending_route_commands.items()
            if stamp >= cutoff
        }
        return route, command

    def commit_route_transaction_locked(
        self, route=None, message=None, *, allow_defer=True
    ):
        """Atomically expose a route and command carrying the same stamp."""

        if route is None or message is None:
            matched = self.matching_route_transaction_locked()
            if matched is None:
                return False
            route, message = matched
        command_stamp = self.message_stamp(message)
        previous_source = (
            str(self.route_command.route_source)
            if self.route_command is not None else "NONE"
        )
        next_source = str(message.route_source or "NONE")
        previous_is_far = previous_source.startswith("FAR_")
        next_is_far = next_source.startswith("FAR_")
        far_authority_handoff = bool(next_is_far and not previous_is_far)
        previous_mode = (
            int(self.route_command.navigation_mode)
            if self.route_command is not None else None
        )
        next_mode = int(message.navigation_mode)
        memory_modes = (
            LocalRouteCommand.NAVIGATION_MEMORY_BACKTRACK,
            LocalRouteCommand.NAVIGATION_MEMORY_RESUME,
            LocalRouteCommand.NAVIGATION_FAR_DEAD_END_EGRESS,
        )
        if (
            allow_defer
            and self.maneuver.active
            and next_mode not in memory_modes
        ):
            # Compute and transport FAR in parallel with a committed leg, but
            # never replace its geometry mid-motion.  Only the newest complete
            # transaction is needed; it is committed on the first stopped
            # inter-leg control tick.
            existing = self.deferred_route_transaction
            existing_source = (
                str(existing[1].route_source or "NONE")
                if existing is not None else "NONE"
            )
            if not (
                existing_source.startswith("FAR_")
                and not next_source.startswith("FAR_")
            ):
                self.deferred_route_transaction = (route, message)
            rospy.loginfo_throttle(
                1.0,
                "Queued complete route transaction until committed maneuver "
                "leg boundary source=%s stamp=%.6f",
                next_source,
                command_stamp,
            )
            return False
        revalidation_fallback_expired = bool(
            self.forward_capture_replan_requested
            and not next_is_far
            and next_mode not in memory_modes
            and command_stamp > 0.0
            and command_stamp
            >= self.forward_capture_replan_after_stamp
            + self.forward_course_revalidation_timeout
        )
        continuing_forward_restoration = bool(
            self.maneuver.purpose == "forward_restoration"
            and self.maneuver.leg_count > 0
        )
        if (
            self.forward_capture_replan_requested
            and not next_is_far
            and next_mode not in memory_modes
            and not revalidation_fallback_expired
        ):
            # A local exploration transaction is not the answer to a
            # measured-pose FAR re-anchor.  Replacing the last FAR corridor
            # here leaves ForwardPreferenceSupervisor in ROUTE_REVALIDATION
            # while its visible green route points somewhere unrelated.  Keep
            # the previous transaction frozen until FAR answers, while still
            # allowing an explicit memory/egress authority to pre-empt it.
            rospy.loginfo_throttle(
                2.0,
                "Retained FAR route during course revalidation; rejected "
                "intermediate route_source=%s",
                next_source,
            )
            return False
        if revalidation_fallback_expired:
            # A speculative FAR replacement may never become motion-
            # authorized while the vehicle is deliberately stationary.  Do
            # not turn a safety handshake into an unbounded liveness stop:
            # fall back to the currently published topology/local corridor,
            # whose every primitive remains subject to the normal hard veto.
            if continuing_forward_restoration:
                # FAR failed to replace its corridor and the authority is now
                # switching to topology/local guidance.  This is not another
                # leg of the old FAR turnaround: retaining its turn sign and
                # leg counter made an unrelated rear topology edge spend the
                # remainder of the same eight-leg budget.  End that transaction
                # and let the fallback direction start one freshly confirmed
                # manoeuvre of its own.
                self.maneuver.reset()
                self.local_turnaround_transaction_id = 0
                self.forward_restoration_budget_replans = 0
                self.forward_restoration_budget_replan_pending = False
                self.forward_preference.reset(
                    preserve_forward_evidence=True
                )
                continuing_forward_restoration = False
            else:
                self.forward_preference.approve_revalidated_route()
            self.forward_capture_replan_requested = False
            self.forward_capture_replan_after_stamp = -math.inf
            self.replan_pub.publish(
                String(data="forward_course_revalidation_fallback")
            )
            rospy.logwarn(
                "FAR course revalidation timed out after %.2fs; accepting "
                "bounded fallback route_source=%s",
                self.forward_course_revalidation_timeout,
                next_source,
            )
        memory_transition = (
            previous_mode != next_mode
            and (
                next_mode in memory_modes
                or previous_mode in memory_modes
            )
        )
        if memory_transition:
            # Memory reverse/resume is an exclusive mission-level authority.
            # Entering or leaving it cancels every stale local manoeuvre and
            # deadlock timer from the previous route transaction.
            self.maneuver.reset()
            self.local_turnaround_transaction_id = 0
            self.maneuver_retry_not_before = 0.0
            self.turnaround_rearm_remaining = 0.0
            self.forward_restoration_budget_replans = 0
            self.forward_restoration_budget_replan_pending = False
            self.forward_preference.reset()
            self.hybrid_action_latch.reset()
            self.forward_capture_replan_requested = False
            self.forward_capture_replan_after_stamp = -math.inf
            self.exact_replan_gate.reset()
            self.static_recovery_replan_gate.reset()
            self.recovery.start_authority_transaction()
        stale_fallback_turnaround_replaced_by_far = bool(
            far_authority_handoff
            and continuing_forward_restoration
            and previous_source
            in ("EXPLORED_TOPOLOGY", "LOCAL_SAFE_EXPLORATION")
        )
        if stale_fallback_turnaround_replaced_by_far:
            # The car is stopped at a committed leg boundary (otherwise the
            # complete FAR transaction is deferred above).  A turnaround that
            # was created from drifting historical topology must not donate
            # its turn sign, gear phase or remaining leg budget to the newly
            # measured FAR corridor.  End it here and let FAR make a fresh
            # decision from the current pose.
            self.maneuver.reset()
            self.local_turnaround_transaction_id = 0
            self.maneuver_retry_not_before = 0.0
            self.turnaround_rearm_remaining = 0.0
            self.forward_restoration_budget_replans = 0
            self.forward_restoration_budget_replan_pending = False
            self.forward_preference.reset(preserve_forward_evidence=True)
            continuing_forward_restoration = False
            rospy.logwarn(
                "Cancelled stale fallback turnaround source=%s at atomic FAR "
                "handoff; FAR will re-evaluate gear and turn side from the "
                "measured pose",
                previous_source,
            )
        fresh_course_capture_reanchor = bool(
            self.forward_capture_replan_requested
            # A newer local-exploration command is not FAR's answer.  The
            # old timestamp-only check let LOCAL_SAFE_EXPLORATION clear the
            # barrier while the replacement visibility route was still
            # settling, which removed the rolling FAR carrot from control.
            and next_is_far
            and command_stamp > 0.0
            and command_stamp > self.forward_capture_replan_after_stamp + 1.0e-6
        )
        if fresh_course_capture_reanchor:
            # FAR is allowed to solve a measured-pose replan immediately.  In
            # that case its status can remain PASS throughout, so waiting for
            # a MAPPING_WAIT edge would leave ROUTE_REVALIDATION latched
            # forever even though a fresh certified route is already here.
            if continuing_forward_restoration:
                self.forward_preference.approve_continuation_route()
            else:
                self.forward_preference.approve_revalidated_route()
            self.forward_capture_replan_requested = False
            self.forward_capture_replan_after_stamp = -math.inf
            if self.forward_restoration_budget_replan_pending:
                if not self.maneuver.renew_leg_budget():
                    rospy.logerr(
                        "Fresh FAR route could not renew the exhausted "
                        "forward-restoration leg budget"
                    )
                else:
                    self.forward_restoration_budget_replans += 1
                    rospy.logwarn(
                        "Renewed forward-restoration leg budget once from "
                        "a fresh measured-pose FAR route"
                    )
                self.forward_restoration_budget_replan_pending = False
            rospy.loginfo(
                "Accepted fresh FAR route transaction after course-capture "
                "revalidation stamp=%.6f",
                command_stamp,
            )
        goal = PoseStamped()
        goal.header = message.header
        goal.pose = message.target
        self.route = route
        self.route_command = message
        self.goal = goal
        preserve_far_reference_cursor = bool(
            next_is_far
            and previous_is_far
            and str(message.route_id) == self.global_route_id
            and int(message.route_revision) == self.global_route_revision
        )
        self.global_route_id = str(message.route_id)
        self.global_route_source = next_source
        self.global_route_revision = int(message.route_revision)
        self.route_authority_epoch = int(message.authority_epoch)
        self.route_reference_authority_epoch = self.route_authority_epoch
        if not preserve_far_reference_cursor:
            self.route_reference_index = max(0, int(message.segment_index))
        if far_authority_handoff:
            # Route, target and authority metadata arrive in this one stamped
            # transaction.  Clear every exploratory direction latch before
            # the next 10 Hz planning tick so the accepted FAR carrot takes
            # effect immediately, rather than waiting for another FAR replan
            # or for an asynchronously delivered status message.
            if continuing_forward_restoration:
                # Route ownership may change while the car is stopped between
                # two legs.  Keep the already chosen turn side and transaction
                # identity; the new FAR corridor supplies geometry for the
                # next leg, not permission to declare the turn complete.
                if not fresh_course_capture_reanchor:
                    self.forward_preference.approve_continuation_route()
            elif not fresh_course_capture_reanchor:
                # ``approve_revalidated_route`` above carries the one-shot
                # permission for a confirmed rear FAR corridor to begin its
                # multi-leg turn.  Resetting it again merely because the
                # previous published source was exploratory recreates the
                # revalidation loop we have just completed.
                self.forward_preference.reset(
                    preserve_forward_evidence=True
                )
            self.recovery.start_authority_transaction()
            self.exact_replan_gate.reset()
            self.policy_raw = None
            self.policy_query_poses = {}
            self.control_pose = None
            self.last_position = None
            if self.maneuver.leg_count == 0:
                self.maneuver.reset()
                self.local_turnaround_transaction_id = 0
                self.maneuver_retry_not_before = 0.0
            rospy.loginfo(
                "Accepted atomic navigation authority handoff %s -> %s "
                "route_id=%s revision=%d epoch=%d",
                previous_source,
                next_source,
                self.global_route_id or "none",
                self.global_route_revision,
                self.route_authority_epoch,
            )
        return True

    def on_route_command(self, message):
        with self.lock:
            self.buffer_route_transaction_half_locked(
                self.pending_route_commands, message
            )
            self.commit_route_transaction_locked()

    def on_route(self, message):
        with self.lock:
            self.buffer_route_transaction_half_locked(
                self.pending_routes, message
            )
            self.commit_route_transaction_locked()

    def on_global_planner_status(self, message):
        try:
            document = json.loads(message.data)
            state = str(document.get("state", "IDLE"))
            visibility = document.get("visibility_graph", {})
            rolling = document.get(
                "active_rolling_route",
                visibility.get("rolling_route", {}),
            )
            turnaround = document.get("route_turnaround_transaction", {})
        except (TypeError, ValueError):
            document = {}
            rolling = {}
            turnaround = {}
            state = "INVALID_STATUS"
        with self.lock:
            previous_state = self.global_planner_state
            self.global_planner_state = state
            status_authority_epoch = int(
                document.get("authority_epoch", 0) or 0
            )
            if self.route_command is None:
                self.global_route_id = str(rolling.get("route_id", ""))
                self.global_route_source = str(
                    rolling.get(
                        "source", document.get("guidance_source", "NONE")
                    )
                )
                self.global_route_revision = int(
                    rolling.get("route_revision", 0)
                )
            status_route_id = str(rolling.get("route_id", ""))
            status_source = str(
                rolling.get("source", document.get("guidance_source", "NONE"))
            )
            status_revision = int(rolling.get("route_revision", 0))
            status_matches_command = bool(
                self.route_command is None
                or (
                    status_route_id == self.global_route_id
                    and status_source == self.global_route_source
                    and status_revision == self.global_route_revision
                    and status_authority_epoch == self.route_authority_epoch
                )
            )
            if status_matches_command:
                self.global_route_progress_m = float(
                    rolling.get("progress_m", 0.0)
                )
                self.global_route_carrot_m = float(
                    rolling.get("carrot_m", 0.0)
                )
            self.global_turnaround_transaction_id = int(
                turnaround.get("id", 0)
            )
            if (
                state in self.global_wait_states
                and previous_state not in self.global_wait_states
                and not self.maneuver.active
            ):
                # A measured-pose FAR re-anchor starts a new direction
                # transaction.  Do not carry the rejected course-capture
                # latch into the freshly certified route.  If this wait was
                # itself caused by our re-anchor request, however, preserve
                # the transaction barrier until a newer synchronized route
                # arrives; clearing it here makes a persistent rear route
                # bounce forever between WAIT and another replan request.
                if not self.forward_capture_replan_requested:
                    self.forward_preference.reset(
                        preserve_forward_evidence=True
                    )
                    self.forward_capture_replan_after_stamp = -math.inf

    def on_mission_goal(self, message):
        with self.lock:
            self.mission_goal = message
            self.goal = None
            self.route_command = None
            self.route = None
            self.pending_route_commands = {}
            self.pending_routes = {}
            self.deferred_route_transaction = None
            self.policy_raw = None
            self.last_position = None
            self.arrival.reset()
            self.maneuver.reset()
            self.local_turnaround_transaction_id = 0
            self.route_authority_epoch = 0
            self.route_reference_authority_epoch = 0
            self.route_reference_index = 0
            self.maneuver_retry_not_before = 0.0
            self.turnaround_rearm_remaining = 0.0
            self.forward_restoration_budget_replans = 0
            self.forward_restoration_budget_replan_pending = False
            self.forward_preference.reset()
            self.hybrid_action_latch.reset()
            self.forward_capture_replan_requested = False
            self.forward_capture_replan_after_stamp = -math.inf
            self.exact_replan_gate.reset()
            self.static_recovery_replan_gate.reset()
            self.recovery.set_mission_goal(
                (message.pose.position.x, message.pose.position.y)
            )
        rospy.loginfo(
            "Accepted mission goal as %s",
            "position+heading" if self.require_mission_goal_heading else "position-only",
        )

    def on_tracks(self, message):
        tracks = [
            DynamicTrack(
                track_id=item.track_id,
                x=item.x,
                y=item.y,
                vx=item.vx,
                vy=item.vy,
                radius=item.radius,
                confidence=item.confidence,
                covariance=np.asarray(item.position_covariance).reshape((2, 2)),
            )
            for item in message.tracks
        ]
        with self.lock:
            self.tracks = tracks

    def on_policy_raw(self, message):
        with self.lock:
            self.policy_raw = message

    def on_policy_inference_state(self, message):
        with self.lock:
            self.policy_inference_state = message

    def tracks_body(self, odom, tracks):
        heading = yaw_from_quaternion(odom.pose.pose.orientation)
        cosine, sine = math.cos(heading), math.sin(heading)
        origin_x, origin_y = odom.pose.pose.position.x, odom.pose.pose.position.y
        return tuple(
            DynamicTrack(
                track_id=item.track_id,
                x=cosine * (item.x - origin_x) + sine * (item.y - origin_y),
                y=-sine * (item.x - origin_x) + cosine * (item.y - origin_y),
                vx=cosine * item.vx + sine * item.vy,
                vy=-sine * item.vx + cosine * item.vy,
                radius=item.radius,
                covariance=item.covariance,
                confidence=item.confidence,
                stamp=item.stamp,
            )
            for item in tracks
        )

    def stop(self, source="local_planner"):
        if rospy.is_shutdown():
            return
        command = AckermannCommand()
        command.header.stamp = rospy.Time.now()
        command.gear = int(Gear.NEUTRAL)
        command.brake = True
        command.source = source
        self.command_pub.publish(command)
        if source == "waiting_for_inputs":
            state = "WAITING_FOR_ROUTE"
        elif "static" in source or "no_safe" in source:
            state = "STATIC_BLOCKED"
        elif "dynamic" in source:
            state = "DYNAMIC_YIELD"
        elif "gear_shift" in source:
            state = "SHIFTING"
        elif source == "mission_goal_reached":
            state = "GOAL_REACHED"
        elif "goal_active_braking" in source:
            state = "GOAL_BRAKING"
        elif "maneuver" in source:
            state = "MANEUVERING"
        elif "policy" in source:
            state = "POLICY_NOT_READY"
        else:
            state = "STOPPED"
        self.report_runtime_status(state, source)

    def report_runtime_status(self, state, detail):
        signature = (str(state), str(detail))
        if signature == self.last_runtime_status:
            return
        self.last_runtime_status = signature
        if state in ("STATIC_BLOCKED", "POLICY_NOT_READY"):
            rospy.logwarn("Local planner %s: %s", state, detail)
        else:
            rospy.loginfo("Local planner %s: %s", state, detail)
        marker = Marker()
        marker.header.stamp = rospy.Time.now()
        marker.header.frame_id = "chassis"
        marker.ns = "dep_car_local_planner_status"
        marker.id = 0
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        marker.pose.position.z = 0.65
        marker.pose.orientation.w = 1.0
        marker.scale.z = 0.16
        marker.color.a = 1.0
        colors = {
            "DRIVING_FORWARD": (0.1, 1.0, 0.1),
            "DRIVING_REVERSE": (0.1, 0.7, 1.0),
            "WAITING_FOR_ROUTE": (1.0, 0.75, 0.0),
            "SHIFTING": (1.0, 0.75, 0.0),
            "GOAL_REACHED": (0.1, 1.0, 0.1),
            "DYNAMIC_YIELD": (1.0, 0.4, 0.0),
        }
        marker.color.r, marker.color.g, marker.color.b = colors.get(
            state, (1.0, 0.1, 0.1)
        )
        marker.text = "%s\n%s" % (state, detail)
        self.status_marker_pub.publish(marker)

    def request_measured_pose_replan(self, reason):
        self.replan_pub.publish(String(data=str(reason)))

    @staticmethod
    def subgoal_body(odom, goal):
        yaw = yaw_from_quaternion(odom.pose.pose.orientation)
        dx = goal.pose.position.x - odom.pose.pose.position.x
        dy = goal.pose.position.y - odom.pose.pose.position.y
        return (
            math.cos(yaw) * dx + math.sin(yaw) * dy,
            -math.sin(yaw) * dx + math.cos(yaw) * dy,
        )

    @staticmethod
    def vehicle_state(odom, joint_state=None, twist_in_body_frame=False):
        yaw = yaw_from_quaternion(odom.pose.pose.orientation)
        velocity = odom.twist.twist.linear
        steering = 0.0
        if joint_state is not None:
            positions = dict(zip(joint_state.name, joint_state.position))
            names = ("front_left_steer_joint", "front_right_steer_joint")
            if all(name in positions for name in names):
                steering = center_steering_from_wheel_angles(
                    positions[names[0]], positions[names[1]], True
                )
        return VehicleState(
            speed=(
                float(velocity.x)
                if twist_in_body_frame
                else world_velocity_to_body_longitudinal(velocity.x, velocity.y, yaw)
            ),
            steering=steering,
            yaw_rate=odom.twist.twist.angular.z,
            stamp=odom.header.stamp.to_sec(),
        )

    @staticmethod
    def heading_error(odom, goal):
        return wrap_angle(
            yaw_from_quaternion(goal.pose.orientation)
            - yaw_from_quaternion(odom.pose.pose.orientation)
        )

    @staticmethod
    def reference_curvature(odom, route, requested_gear, start_index=0):
        if route is None or len(route.points) < 3:
            return 0.0
        position = np.asarray(
            [odom.pose.pose.position.x, odom.pose.pose.position.y], dtype=float
        )
        world = np.asarray(
            [[point.pose.position.x, point.pose.position.y] for point in route.points],
            dtype=float,
        )
        begin = min(max(0, int(start_index)), len(route.points) - 1)
        end = min(len(route.points), begin + 30)
        nearest = begin + int(
            np.argmin(np.linalg.norm(world[begin:end] - position, axis=1))
        )
        heading = yaw_from_quaternion(odom.pose.pose.orientation)
        active = []
        corridor_route = not any(int(point.gear) != 0 for point in route.points)
        for point in route.points[nearest:]:
            gear = int(point.gear)
            if not corridor_route and gear == 0 and not active:
                continue
            if not corridor_route and gear != int(requested_gear):
                if active:
                    break
                continue
            dx = point.pose.position.x - position[0]
            dy = point.pose.position.y - position[1]
            active.append(
                (
                    math.cos(heading) * dx + math.sin(heading) * dy,
                    -math.sin(heading) * dx + math.cos(heading) * dy,
                    wrap_angle(yaw_from_quaternion(point.pose.orientation) - heading),
                )
            )
            if len(active) >= 3:
                break
        if len(active) < 3:
            return 0.0
        distance = float(np.linalg.norm(np.asarray(active[2][:2]) - np.asarray(active[0][:2])))
        return 0.0 if distance < 1.0e-4 else wrap_angle(active[2][2] - active[0][2]) / distance

    def route_reference(self, odom, route, start_index=0, grid=None):
        if route is None or not route.points:
            return np.empty((0, 2), dtype=float)
        world = np.asarray(
            [[point.pose.position.x, point.pose.position.y] for point in route.points],
            dtype=float,
        )
        vehicle_pose = (
            odom.pose.pose.position.x,
            odom.pose.pose.position.y,
            yaw_from_quaternion(odom.pose.pose.orientation),
        )
        corridor_route = not any(
            int(point.gear) != int(Gear.NEUTRAL) for point in route.points
        )
        if corridor_route:
            reference, selected = monotonic_route_reference_body(
                world,
                vehicle_pose,
                max(int(start_index), self.route_reference_index),
                grid=grid,
                horizon_m=self.route_reference_horizon,
            )
            self.route_reference_index = max(
                self.route_reference_index, int(selected)
            )
            return reference
        return route_reference_body(
            world, vehicle_pose, start_index,
            horizon_m=self.route_reference_horizon,
        )

    def authority_direction_reference(
        self, odom, route_reference, mission_goal
    ):
        """Return the stable direction authority for a turnaround.

        A LOCAL_SAFE_EXPLORATION route is a short, disposable collision-safe
        tube.  Its endpoint must not become the direction authority of a
        multi-point turn: after the car passes that nearby point it would be
        behind again and manufacture another turnaround.  Until FAR acquires
        authority, use the distant position-only mission goal for direction
        while the unchanged local tube and hard veto continue to own safety.
        """

        mission_body = (
            None
            if mission_goal is None
            else self.subgoal_body(odom, mission_goal)
        )
        return navigation_authority_reference(
            route_reference,
            mission_body,
            self.global_route_source,
            horizon_m=self.route_reference_horizon,
        )

    def turnaround_probes(
        self, state, route_reference, grid, tracks, *, heading_error
    ):
        """Probe local Ackermann room without turning A* into a controller."""

        reference = np.asarray(route_reference, dtype=float)
        if reference.ndim != 2 or reference.shape[1] != 2 or len(reference) < 2:
            return {}, False
        bearing, _ = corridor_direction_body(
            reference, self.forward_preference.config.direction_lookahead_m
        )
        continuing_turn_sign = (
            self.maneuver.turn_sign
            if self.maneuver.purpose == "forward_restoration"
            and self.maneuver.leg_count > 0
            and self.maneuver.turn_sign != 0.0
            else 0.0
        )
        turn_hint = (
            continuing_turn_sign
            if continuing_turn_sign != 0.0
            else bearing
            if math.isfinite(bearing) and abs(bearing) > 1.0e-6
            else heading_error
            if math.isfinite(heading_error) and abs(heading_error) > 1.0e-6
            else 1.0
        )
        directional_reference = (
            float(reference[-1, 0]),
            float(reference[-1, 1]),
        )
        probes = {}
        for gear in (Gear.FORWARD, Gear.REVERSE):
            forward_exit_capture = bool(
                gear == Gear.FORWARD
                and self.maneuver.purpose == "forward_restoration"
                and self.maneuver.leg_count > 0
                and abs(bearing) < 0.5 * math.pi
            )
            reverse_alignment_preservation = bool(
                gear == Gear.REVERSE
                and self.maneuver.purpose == "forward_restoration"
                and self.maneuver.leg_count > 0
                and abs(bearing)
                <= self.turnaround_alignment_release_bearing
            )
            gear_turn_hint = (
                bearing
                if forward_exit_capture and abs(bearing) >= 0.05
                else None
                if forward_exit_capture
                else None
                if reverse_alignment_preservation
                else turn_hint
            )
            proposed = self.maneuver.proposed_subgoal(
                gear,
                directional_reference,
                bearing if math.isfinite(bearing) else heading_error,
                purpose="forward_restoration",
                turn_sign_hint=(
                    turn_hint if gear_turn_hint is None else gear_turn_hint
                ),
            )
            result = self.planner.plan(
                state,
                proposed,
                grid,
                tracks,
                requested_gear=gear,
                target_heading=bearing,
                spatial_scales=self.maneuver_spatial_scales,
                required_yaw_direction=gear_turn_hint,
                minimum_yaw_progress_rad=(
                    0.0
                    if gear_turn_hint is None
                    else self.maneuver_minimum_yaw_progress
                ),
                allow_static_margin_egress=(gear == Gear.REVERSE),
                maximum_margin_overlap_m=self.memory_margin_egress_maximum_overlap,
                minimum_margin_improvement_m=self.memory_margin_egress_minimum_improvement,
                margin_worsening_tolerance_m=self.memory_margin_egress_worsening_tolerance,
            )
            probes[gear] = (proposed, result)
        # A turnaround site must support both halves of a local multi-point
        # correction.  One safe reverse arc alone is an escape corridor, not
        # evidence that the car can already restore forward travel there.
        feasible = all(probes[gear][1].executable for gear in probes)
        return probes, feasible

    def stable_far_forward_exit_available(
        self, bearing, forward_probe, route_command
    ):
        """Allow normal rolling control once a refreshed FAR course is usable.

        A parking leg is a means to point the body into the route, not an
        obligation to consume a fixed number of 0.85 m strokes.  This gate is
        intentionally stricter than ordinary candidate feasibility: it needs
        FAR authority, a forward route hint, a small bearing error and at
        least a 75% geometric horizon.  The existing multi-cycle confirmation
        in ForwardPreferenceSupervisor remains in force.
        """

        result = None if forward_probe is None else forward_probe[1]
        return bool(
            self.maneuver.purpose == "forward_restoration"
            and self.maneuver.leg_count > 0
            and str(self.global_route_source).startswith("FAR_")
            and route_command is not None
            # Compare the wire value directly.  A transient neutral command
            # must simply fail this release gate, not raise from
            # ``Gear.require_drive`` inside the control callback.
            and int(route_command.requested_gear) == int(Gear.FORWARD)
            and math.isfinite(float(bearing))
            and abs(float(bearing))
            <= self.turnaround_alignment_release_bearing
            and result is not None
            and result.executable
            and result.spatial_scale is not None
            and float(result.spatial_scale)
            >= self.turnaround_alignment_minimum_spatial_scale
        )

    def turnaround_gear_order(self, probes):
        """Choose a geometry-driven first leg, then strictly alternate gears.

        A map ID or manoeuvre label must never decide the transmission.  Once
        a transaction has a completed leg, alternation owns the order.  At a
        new site, rank the two hard-safe probes by swept clearance and yaw
        progress; a reverse leg wins exact ties so a goal directly behind the
        vehicle cannot make it drive toward the separating wall first.
        """

        if self.maneuver.last_completed_gear in (Gear.FORWARD, Gear.REVERSE):
            return self.maneuver.recovery_gear_order(Gear.FORWARD)
        ranked = []
        for gear in (Gear.FORWARD, Gear.REVERSE):
            result = probes.get(gear, (None, None))[1]
            if result is None or not result.executable:
                continue
            selected = result.selected
            clearance = float(selected.static_clearance)
            yaw_progress = abs(float(selected.trajectory[-1, 3]))
            reverse_tie_break = 1 if gear == Gear.REVERSE else 0
            ranked.append((clearance, yaw_progress, reverse_tie_break, gear))
        ranked.sort(reverse=True)
        ordered = [row[-1] for row in ranked]
        ordered.extend(
            gear for gear in (Gear.REVERSE, Gear.FORWARD) if gear not in ordered
        )
        return tuple(ordered)

    @staticmethod
    def reference_steering(odom, route, requested_gear, start_index=0):
        """Return the next Hybrid-A* steering sample for local tie-breaking."""

        if route is None or not route.points:
            return None
        if not any(int(point.gear) != int(Gear.NEUTRAL) for point in route.points):
            return None
        position = np.asarray(
            [odom.pose.pose.position.x, odom.pose.pose.position.y], dtype=float
        )
        world = np.asarray(
            [[point.pose.position.x, point.pose.position.y] for point in route.points],
            dtype=float,
        )
        begin = min(max(0, int(start_index)), len(route.points) - 1)
        nearest = begin + int(
            np.argmin(np.linalg.norm(world[begin:] - position, axis=1))
        )
        for point in route.points[nearest:]:
            if int(point.gear) == int(requested_gear):
                return float(point.steering)
        return None

    def publish_policy_query(
        self,
        requested_gear,
        subgoal_body,
        *,
        heading_error=0.0,
        reference_curvature=0.0,
        recovery_mode=False,
        reference_path=None,
    ):
        if self.policy_mode == "disabled":
            return 0
        self.query_generation += 1
        message = PolicyQuery()
        message.header.stamp = rospy.Time.now()
        message.header.frame_id = "chassis"
        message.requested_gear = int(requested_gear)
        message.subgoal_body.x = float(subgoal_body[0])
        message.subgoal_body.y = float(subgoal_body[1])
        reference = np.asarray(reference_path, dtype=float)
        if not (
            reference.ndim == 2
            and reference.shape[1] == 2
            and len(reference) >= 2
            and np.all(np.isfinite(reference))
        ):
            # DEPCarNetV2 has a mandatory route-corridor contract.  During the
            # first route update or very close to the goal, the global route
            # slice can temporarily contain fewer than two points.  Preserve
            # fail-closed inference without creating an availability deadlock
            # by using the current-origin-to-subgoal segment as the minimal
            # local corridor.  This supplies direction only; hard veto still
            # decides whether any generated trajectory is executable.
            endpoint = np.asarray(subgoal_body, dtype=float)
            if not np.all(np.isfinite(endpoint)):
                endpoint = np.asarray((0.05, 0.0), dtype=float)
            if float(np.linalg.norm(endpoint)) < 0.05:
                endpoint = np.asarray((0.05, 0.0), dtype=float)
            reference = np.vstack((np.zeros(2, dtype=float), endpoint))
        if reference.ndim == 2 and reference.shape[1] == 2 and len(reference) >= 2:
            reference = reference[:80]
            for index, point in enumerate(reference):
                pose = Pose2D()
                pose.x = float(point[0])
                pose.y = float(point[1])
                if index + 1 < len(reference):
                    delta = reference[index + 1] - point
                else:
                    delta = point - reference[index - 1]
                pose.theta = float(math.atan2(delta[1], delta[0]))
                message.route_corridor_body.append(pose)
            message.route_corridor_valid = True
        else:  # Defensive; the normalized fallback above must make this unreachable.
            message.route_corridor_valid = False
        message.heading_error = float(heading_error)
        message.reference_curvature = float(reference_curvature)
        message.recovery_mode = bool(recovery_mode)
        message.generation = self.query_generation
        if self.control_pose is not None:
            self.policy_query_poses[self.query_generation] = tuple(
                float(value) for value in self.control_pose
            )
            while len(self.policy_query_poses) > 64:
                del self.policy_query_poses[min(self.policy_query_poses)]
        self.policy_query_pub.publish(message)
        return self.query_generation

    def align_policy_candidates_to_current_pose(self, candidates, generation, age):
        """Express a delayed learned bank in the current chassis frame.

        Inference runs asynchronously on the GPU.  Keeping a bank for longer
        than one control tick is safe only when its anchor pose moves with the
        car.  This rigid transform preserves the learned curve and kinematics;
        the transformed full sequence is then checked against the newest
        local occupancy grid and dynamic tracks.
        """

        anchor = self.policy_query_poses.get(int(generation))
        current = self.control_pose
        if anchor is None or current is None:
            raise ValueError("policy_pose_anchor_missing")
        for candidate in candidates:
            trajectory = align_trajectory_between_chassis_frames(
                candidate.trajectory,
                anchor,
                current,
            )
            candidate.trajectory = trajectory
            candidate.policy_anchor_age_s = float(max(0.0, age))
            # Rows 1..5 are the first learned action.  Select the control
            # sample appropriate for elapsed inference time without leaking
            # into the next learned gear action.
            candidate.command_lookahead_index = int(
                min(
                    5,
                    max(
                        1,
                        np.searchsorted(
                            trajectory[:, 0],
                            float(max(0.0, age)) + 0.20,
                            side="left",
                        ),
                    ),
                )
            )
        return candidates

    @staticmethod
    def policy_core_candidates(message):
        if len(message.candidates) != 15:
            raise ValueError("raw policy bank does not contain 15 candidates")
        hybrid = bool(message.hybrid_sequence)
        if hybrid and (
            message.architecture_id not in (
                V42_EXECUTION_ARCHITECTURE_ID, V43_ARCHITECTURE_ID
            )
            or int(message.actions_per_candidate) != 6
            or int(message.steps_per_action) != 5
            or len(message.gear_history) != 6
        ):
            raise ValueError("raw hybrid sequence contract changed")
        output = []
        for expected_id, item in enumerate(message.candidates):
            if int(item.candidate_id) != expected_id:
                raise ValueError("raw policy candidate ordering changed")
            arrays = [
                np.asarray(values, dtype=np.float64)
                for values in (item.time, item.x, item.y, item.yaw, item.speed, item.steering)
            ]
            expected_rows = 31 if hybrid else 11
            if any(values.shape != (expected_rows,) for values in arrays):
                raise ValueError(
                    "raw policy trajectory row count changed: %d" % expected_rows
                )
            trajectory = np.column_stack(arrays)
            gear = int(item.gear)
            core = CoreCandidate(
                candidate_id=expected_id,
                speed_anchor=float(item.speed_anchor),
                steering_anchor=float(item.steering_anchor),
                duration=float(item.duration),
                trajectory=trajectory,
                gear=(
                    Gear.require_drive(gear)
                    if gear in (-1, 1)
                    else Gear.FORWARD
                ),
                learned_score=float(item.learned_score),
            )
            if hybrid:
                core.action_gears = np.asarray(item.action_gears, dtype=np.int8)
                core.action_mask = np.asarray(item.action_mask, dtype=bool)
                core.action_durations = np.asarray(
                    item.action_durations, dtype=np.float64
                )
                core.shift_required = np.asarray(item.shift_required, dtype=bool)
                core.transition_duration = np.asarray(
                    item.transition_duration, dtype=np.float64
                )
                core.motion_gears = np.asarray(item.motion_gears, dtype=np.int8)
                core.hybrid_sequence = True
            output.append(core)
        return output

    def policy_result(
        self,
        grid,
        grid_stamp,
        tracks,
        requested_gear,
        subgoal,
        *,
        recovery_mode,
        reference_path=None,
    ):
        if self.policy_mode == "disabled":
            return None, "policy_disabled"
        with self.lock:
            raw = self.policy_raw
            inference = self.policy_inference_state
        if self.policy_mode in ("active", "guarded") and (
            inference is None
            or not inference.model_loaded
            or not inference.control_authorized
        ):
            return None, "policy_control_not_authorized"
        if raw is None:
            return None, "waiting_for_policy_bank"
        stamp = raw.header.stamp.to_sec()
        age = rospy.Time.now().to_sec() - stamp
        if age < -0.05 or age > self.policy_freshness:
            return None, "stale_policy_bank"
        grid_age = rospy.Time.now().to_sec() - float(grid_stamp)
        if grid_age < -0.05 or grid_age > self.policy_grid_skew:
            return None, "stale_current_costmap"
        # A newer grid is desirable: the pose-aligned trajectory is hard-
        # vetoed in the current chassis frame.  Reject only a grid that
        # predates the model anchor beyond the contract, rather than taking an
        # absolute difference that discards every asynchronous CUDA result.
        if stamp - float(grid_stamp) > self.policy_grid_skew:
            return None, "costmap_predates_policy_bank"
        hybrid = bool(raw.hybrid_sequence)
        if not hybrid and int(raw.requested_gear) != int(requested_gear):
            return None, "policy_gear_mismatch"
        if bool(raw.recovery_mode) != bool(recovery_mode):
            return None, "policy_context_mismatch"
        raw_subgoal = np.asarray([raw.subgoal_body.x, raw.subgoal_body.y], dtype=float)
        if float(np.linalg.norm(raw_subgoal - np.asarray(subgoal, dtype=float))) > self.policy_subgoal_tolerance:
            return None, "policy_subgoal_mismatch"
        try:
            candidates = self.policy_core_candidates(raw)
            if hybrid:
                candidates = self.align_policy_candidates_to_current_pose(
                    candidates, raw.generation, age
                )
            evaluator = (
                evaluate_hybrid_sequence_candidate_bank
                if hybrid
                else evaluate_learned_candidate_bank
            )
            result = evaluator(
                candidates, subgoal, grid, tracks, generation=raw.generation
            )
            if hybrid and not result.executable:
                vetoes = Counter(
                    candidate.veto_reason or "unknown"
                    for candidate in result.candidates
                    if not candidate.feasible
                )
                rospy.logwarn_throttle(
                    1.0,
                    "Hybrid current-frame hard veto rejected bank generation=%d "
                    "anchor_age=%.3f reasons=%s",
                    int(raw.generation),
                    float(age),
                    json.dumps(dict(sorted(vetoes.items())), sort_keys=True),
                )
            if not hybrid and not self.learned_route_authority:
                result = apply_runtime_route_preference(
                    result,
                    reference_path,
                    grid,
                    corridor_weight=self.route_corridor_weight,
                    desired_future_clearance_m=self.route_clearance_target,
                    clearance_weight=self.route_clearance_weight,
                    corner_corridor_minimum_scale=self.corner_corridor_minimum_scale,
                )
            if not hybrid:
                result = apply_corner_clearance_preference(
                    result,
                    reference_path,
                    grid,
                    soft_clearance_m=self.corner_soft_clearance_target,
                    weight=self.corner_soft_clearance_weight,
                    trigger_rad=self.corner_soft_trigger,
                    full_strength_rad=self.corner_soft_full_strength,
                    learned_score_base=self.learned_route_authority,
                )
        except Exception as exc:
            return None, "policy_bank_rejected:" + type(exc).__name__ + ":" + str(exc)
        prefix = (
            "policy_v43_hybrid"
            if hybrid and raw.architecture_id == V43_ARCHITECTURE_ID
            else "policy_v42_hybrid" if hybrid else "policy"
        )
        return result, prefix + ("_ready" if result.executable else "_zero_feasible")

    def plan_context(
        self,
        state,
        subgoal,
        grid,
        grid_stamp,
        tracks,
        requested_gear,
        *,
        heading_error=0.0,
        reference_curvature=0.0,
        reference_steering=None,
        recovery_mode=False,
        spatial_scales=(1.0,),
        force_baseline=False,
        reference_path=None,
        required_yaw_direction=None,
        minimum_yaw_progress_rad=0.0,
        allow_static_margin_egress=False,
    ):
        self.publish_policy_query(
            requested_gear,
            subgoal,
            heading_error=heading_error,
            reference_curvature=reference_curvature,
            recovery_mode=recovery_mode,
            reference_path=reference_path,
        )
        baseline = self.planner.plan(
            state,
            subgoal,
            grid,
            tracks,
            requested_gear=requested_gear,
            target_heading=heading_error,
            target_steering=reference_steering,
            spatial_scales=spatial_scales,
            required_yaw_direction=required_yaw_direction,
            minimum_yaw_progress_rad=minimum_yaw_progress_rad,
            allow_static_margin_egress=allow_static_margin_egress,
            maximum_margin_overlap_m=self.memory_margin_egress_maximum_overlap,
            minimum_margin_improvement_m=(
                self.memory_margin_egress_minimum_improvement
            ),
            margin_worsening_tolerance_m=(
                self.memory_margin_egress_worsening_tolerance
            ),
        )
        baseline = apply_runtime_route_preference(
            baseline,
            reference_path,
            grid,
            corridor_weight=self.route_corridor_weight,
            desired_future_clearance_m=self.route_clearance_target,
            clearance_weight=self.route_clearance_weight,
            corner_corridor_minimum_scale=self.corner_corridor_minimum_scale,
        )
        baseline = apply_corner_clearance_preference(
            baseline,
            reference_path,
            grid,
            soft_clearance_m=self.corner_soft_clearance_target,
            weight=self.corner_soft_clearance_weight,
            trigger_rad=self.corner_soft_trigger,
            full_strength_rad=self.corner_soft_full_strength,
        )
        if self.publish_dagger_teacher_banks:
            teacher_banks = {Gear.require_drive(requested_gear): baseline}
            opposite = (
                Gear.REVERSE
                if Gear.require_drive(requested_gear) == Gear.FORWARD
                else Gear.FORWARD
            )
            opposite_bank = self.planner.plan(
                state,
                subgoal,
                grid,
                tracks,
                requested_gear=opposite,
                target_heading=heading_error,
                target_steering=reference_steering,
                spatial_scales=spatial_scales,
                required_yaw_direction=required_yaw_direction,
                minimum_yaw_progress_rad=minimum_yaw_progress_rad,
                allow_static_margin_egress=allow_static_margin_egress,
                maximum_margin_overlap_m=self.memory_margin_egress_maximum_overlap,
                minimum_margin_improvement_m=(
                    self.memory_margin_egress_minimum_improvement
                ),
                margin_worsening_tolerance_m=(
                    self.memory_margin_egress_worsening_tolerance
                ),
            )
            opposite_bank = apply_runtime_route_preference(
                opposite_bank,
                reference_path,
                grid,
                corridor_weight=self.route_corridor_weight,
                desired_future_clearance_m=self.route_clearance_target,
                clearance_weight=self.route_clearance_weight,
                corner_corridor_minimum_scale=self.corner_corridor_minimum_scale,
            )
            opposite_bank = apply_corner_clearance_preference(
                opposite_bank,
                reference_path,
                grid,
                soft_clearance_m=self.corner_soft_clearance_target,
                weight=self.corner_soft_clearance_weight,
                trigger_rad=self.corner_soft_trigger,
                full_strength_rad=self.corner_soft_full_strength,
            )
            teacher_banks[opposite] = opposite_bank
            self.publish_candidates(
                teacher_banks[Gear.FORWARD],
                Gear.FORWARD,
                subgoal,
                recovery_mode=recovery_mode,
                publisher=self.dagger_teacher_forward_pub,
            )
            self.publish_candidates(
                teacher_banks[Gear.REVERSE],
                Gear.REVERSE,
                subgoal,
                recovery_mode=recovery_mode,
                publisher=self.dagger_teacher_reverse_pub,
            )
        policy, reason = self.policy_result(
            grid,
            grid_stamp,
            tracks,
            requested_gear,
            subgoal,
            recovery_mode=recovery_mode,
            reference_path=reference_path,
        )
        raw_policy_candidate_id = -1
        raw_policy_gear = Gear.NEUTRAL
        if policy is not None and policy.executable:
            raw_policy_candidate_id = int(policy.selected.candidate_id)
            raw_policy_gear = Gear.require_drive(policy.selected.gear)
            if self.policy_mode == "shadow":
                shadow_sequence = [
                    int(value)
                    for value in getattr(policy.selected, "action_gears", ())
                    if int(value) != 0
                ]
                rospy.loginfo_throttle(
                    1.0,
                    "V4.3 shadow recommendation candidate=%d first_gear=%s "
                    "sequence=%s feasible=%d requested_route_gear=%s "
                    "subgoal=(%.3f,%.3f); deterministic control remains "
                    "authoritative",
                    raw_policy_candidate_id,
                    raw_policy_gear.name,
                    shadow_sequence,
                    sum(candidate.feasible for candidate in policy.candidates),
                    Gear.require_drive(requested_gear).name,
                    float(subgoal[0]),
                    float(subgoal[1]),
                )
        if self.policy_mode == "guarded" and policy is not None:
            policy, latch_reason = self.hybrid_action_latch.select(
                policy, rospy.Time.now().to_sec()
            )
            reason += ":" + latch_reason
            if policy.executable:
                action_gears = [
                    int(value)
                    for value in getattr(policy.selected, "action_gears", ())
                    if int(value) != 0
                ]
                rospy.loginfo_throttle(
                    1.0,
                    "V4.2 guarded selection raw_candidate=%d raw_gear=%s "
                    "candidate=%d first_gear=%s "
                    "candidate_sequence=%s locked_sequence=%s action=%d latch=%s",
                    raw_policy_candidate_id,
                    raw_policy_gear.name,
                    policy.selected.candidate_id,
                    Gear.require_drive(policy.selected.gear).name,
                    action_gears,
                    list(self.hybrid_action_latch.locked_sequence),
                    int(self.hybrid_action_latch.action_index),
                    latch_reason,
                )
        if policy is not None:
            published_gear = (
                Gear.require_drive(policy.selected.gear)
                if self.policy_mode == "guarded" and policy.executable
                else requested_gear
            )
            self.publish_candidates(
                policy,
                published_gear,
                subgoal,
                recovery_mode=recovery_mode,
                publisher=self.policy_candidates_pub,
            )
        self.publish_policy_state(policy, reason)
        if force_baseline:
            return baseline, "deterministic_forced_safety_maneuver"
        if self.policy_mode in ("disabled", "shadow"):
            source = "deterministic_lattice" if self.policy_mode == "disabled" else "deterministic_shadow_control"
            return baseline, source
        if policy is not None and policy.executable:
            return policy, (
                "dep_car_net_v42_guarded"
                if self.policy_mode == "guarded"
                else "dep_car_net_v1_active"
            )
        if self.active_fallback_to_baseline:
            return baseline, "deterministic_active_fallback"
        return None, reason

    def publish_command(self, candidate, source, speed_limit=None):
        if rospy.is_shutdown():
            return
        lookahead = min(
            int(getattr(candidate, "command_lookahead_index", 2)),
            len(candidate.trajectory) - 1,
        )
        command = AckermannCommand()
        command.header.stamp = rospy.Time.now()
        speed = float(candidate.trajectory[lookahead, 4])
        if speed_limit is not None:
            speed = math.copysign(min(abs(speed), max(0.0, float(speed_limit))), speed)
        command.speed = speed
        command.steering_angle = float(candidate.trajectory[lookahead, 5])
        command.gear = int(candidate.gear)
        command.brake = False
        command.source = source
        self.command_pub.publish(command)
        self.report_runtime_status(
            "DRIVING_FORWARD" if int(candidate.gear) > 0 else "DRIVING_REVERSE",
            source,
        )

    def publish_active_brake(self, gear, source):
        """Command zero signed speed through PI before engaging neutral hold."""

        if rospy.is_shutdown():
            return
        gear = Gear.require_drive(gear)
        command = AckermannCommand()
        command.header.stamp = rospy.Time.now()
        command.speed = 0.0
        command.steering_angle = 0.0
        command.gear = int(gear)
        command.brake = False
        command.source = source
        self.command_pub.publish(command)
        self.report_runtime_status(
            "GOAL_BRAKING"
            if "goal" in source
            else ("MANEUVER_SETTLING" if "maneuver" in source else "SHIFTING"),
            source,
        )

    def publish_shift_hold(self, decision, state, source):
        """Actively reach zero before the GearSupervisor changes direction."""

        if decision.engaged in (Gear.FORWARD, Gear.REVERSE) and abs(state.speed) > 0.03:
            self.publish_active_brake(decision.engaged, source + "_active_braking")
        else:
            self.stop(source + "_" + decision.state.value.lower())

    def publish_state(
        self,
        result,
        detail="",
        *,
        executable_override=None,
        blocked_by_static_override=None,
    ):
        if rospy.is_shutdown():
            return
        message = PlannerState()
        message.header.stamp = rospy.Time.now()
        message.lifecycle_state = (
            self.maneuver.state.value
            if self.maneuver.active
            else self.recovery.state.value
        )
        message.executable = bool(
            result is not None and result.executable
            if executable_override is None
            else executable_override
        )
        message.blocked_by_static = bool(
            result is not None and result.blocked_by_static
            if blocked_by_static_override is None
            else blocked_by_static_override
        )
        message.blocked_by_dynamic = bool(result is not None and result.blocked_by_dynamic)
        message.planning_generation = result.generation if result is not None else 0
        message.retime_factor = (result.retime_factor or 0.0) if result is not None else 0.0
        message.maneuver_active = bool(
            self.maneuver.active
            or self.forward_restoration_budget_replan_pending
            or (
                self.maneuver.leg_count > 0
                and not self.maneuver.exhausted
                and self.recovery.state != RecoveryState.STATIC_DEADLOCK
            )
        )
        message.maneuver_purpose = str(self.maneuver.purpose)
        message.maneuver_leg = int(self.maneuver.leg_count)
        message.maneuver_gear = int(self.maneuver.gear)
        message.maneuver_travelled_m = float(self.maneuver.travelled_m)
        message.maneuver_target_m = float(self.maneuver.target_distance_m)
        message.detail = detail
        self.state_pub.publish(message)

    def publish_policy_state(self, result, reason):
        with self.lock:
            inference = self.policy_inference_state
        message = PolicyState()
        message.header.stamp = rospy.Time.now()
        message.mode = self.policy_mode
        if inference is not None:
            message.modality = inference.modality
            message.checkpoint_sha256 = inference.checkpoint_sha256
            message.architecture_id = inference.architecture_id
            message.hybrid_sequence = inference.hybrid_sequence
            message.model_loaded = inference.model_loaded
            message.sensor_ready = inference.sensor_ready
            message.inference_ok = inference.inference_ok
            message.inference_attempts = inference.inference_attempts
            message.synchronization_failures = inference.synchronization_failures
            message.inference_latency_ms = inference.inference_latency_ms
            message.sensor_skew_s = inference.sensor_skew_s
            inference_authorized = inference.control_authorized
        else:
            inference_authorized = False
        message.hard_safety_applied = result is not None
        message.executable = bool(result is not None and result.executable)
        message.control_authorized = bool(
            self.policy_mode in ("active", "guarded")
            and inference_authorized
        )
        message.generation = result.generation if result is not None else 0
        message.candidate_count = len(result.candidates) if result is not None else 0
        message.feasible_count = (
            sum(candidate.feasible for candidate in result.candidates)
            if result is not None
            else 0
        )
        message.selected_candidate_id = (
            result.selected.candidate_id if result is not None and result.selected else -1
        )
        if result is not None and result.selected is not None:
            selected = result.selected
            message.selected_first_gear = int(selected.gear)
            message.selected_action_gears = [
                int(value)
                for value in getattr(selected, "action_gears", ())
                if int(value) != 0
            ]
        else:
            message.selected_first_gear = 0
            message.selected_action_gears = []
        message.reason = str(reason)
        self.policy_state_pub.publish(message)

    def publish_candidates(
        self,
        result,
        requested_gear,
        subgoal_body,
        *,
        recovery_mode=False,
        publisher=None,
    ):
        if rospy.is_shutdown() or result is None:
            return
        publisher = self.candidates_pub if publisher is None else publisher
        array = CandidateArray()
        array.header.stamp = rospy.Time.now()
        array.header.frame_id = "chassis"
        array.selected_candidate_id = result.selected.candidate_id if result.selected else -1
        array.requested_gear = int(requested_gear)
        array.planning_generation = result.generation
        array.subgoal_body.x = float(subgoal_body[0])
        array.subgoal_body.y = float(subgoal_body[1])
        array.recovery_mode = bool(recovery_mode)
        for core in result.candidates:
            message = Candidate()
            message.candidate_id = core.candidate_id
            message.gear = int(core.gear)
            message.speed_anchor = core.speed_anchor
            message.steering_anchor = core.steering_anchor
            message.duration = core.duration
            message.retime_factor = core.retime_factor
            message.learned_score = core.learned_score
            message.guidance_cost = core.guidance_cost
            message.static_clearance = core.static_clearance
            message.dynamic_clearance = core.dynamic_clearance
            message.feasible = core.feasible
            message.veto_reason = core.veto_reason
            path = Path()
            path.header = array.header
            for row in core.trajectory:
                pose = PoseStamped()
                pose.header = array.header
                pose.pose.position.x = row[1]
                pose.pose.position.y = row[2]
                pose.pose.orientation.z = math.sin(0.5 * row[3])
                pose.pose.orientation.w = math.cos(0.5 * row[3])
                path.poses.append(pose)
            message.path = path
            array.candidates.append(message)
        publisher.publish(array)
        if result.selected is not None:
            selected_message = next(
                item
                for item in array.candidates
                if item.candidate_id == result.selected.candidate_id
            )
            selected_publisher = (
                self.policy_selected_path_pub
                if publisher is self.policy_candidates_pub
                else self.selected_path_pub
            )
            selected_publisher.publish(selected_message.path)

    def mission_error(self, odom, mission_goal):
        if mission_goal is None:
            return float("inf"), float("inf")
        distance = math.hypot(
            mission_goal.pose.position.x - odom.pose.pose.position.x,
            mission_goal.pose.position.y - odom.pose.pose.position.y,
        )
        heading = abs(
            wrap_angle(
                yaw_from_quaternion(mission_goal.pose.orientation)
                - yaw_from_quaternion(odom.pose.pose.orientation)
            )
        )
        return distance, heading

    @staticmethod
    def combined_speed_limit(*limits):
        values = [float(value) for value in limits if value is not None]
        return min(values) if values else None

    def braking_gear(self, state, route_command):
        if self.gear_supervisor.engaged != Gear.NEUTRAL:
            return self.gear_supervisor.engaged
        if route_command is not None and int(route_command.requested_gear) in (-1, 1):
            return Gear.require_drive(route_command.requested_gear)
        return Gear.REVERSE if state.speed < 0.0 else Gear.FORWARD

    def update(self, _event):
        if rospy.is_shutdown():
            return
        with self.lock:
            if (
                self.deferred_route_transaction is not None
                and not self.maneuver.active
            ):
                deferred_route, deferred_command = (
                    self.deferred_route_transaction
                )
                self.deferred_route_transaction = None
                self.commit_route_transaction_locked(
                    deferred_route,
                    deferred_command,
                    allow_defer=False,
                )
            grid, grid_stamp, odom, joint_state, goal, tracks, route_command, mission_goal, route, global_planner_state = (
                self.grid,
                self.grid_stamp,
                self.odom,
                self.joint_state,
                self.goal,
                tuple(self.tracks),
                self.route_command,
                self.mission_goal,
                self.route,
                self.global_planner_state,
            )
        if grid is None or odom is None:
            self.stop("waiting_for_inputs")
            return
        now = rospy.Time.now().to_sec()
        state = self.vehicle_state(
            odom, joint_state, self.odometry_twist_in_body_frame
        )
        self.control_pose = (
            float(odom.pose.pose.position.x),
            float(odom.pose.pose.position.y),
            float(yaw_from_quaternion(odom.pose.pose.orientation)),
        )
        mission_distance, mission_heading = self.mission_error(odom, mission_goal)
        arrival = (
            self.arrival.update(
                mission_distance,
                mission_heading,
                state.speed,
                heading_required=self.require_mission_goal_heading,
            )
            if mission_goal is not None
            else None
        )
        if arrival is not None and arrival.hold:
            self.gear_supervisor.update(Gear.NEUTRAL, state.speed, now)
            self.stop("mission_goal_reached")
            self.publish_state(None, "latched goal hold; waiting for a new mission")
            return
        if arrival is not None and arrival.active_braking:
            self.publish_active_brake(
                self.braking_gear(state, route_command), "goal_active_braking"
            )
            self.publish_state(None, "active zero-speed capture before neutral hold")
            return
        if (
            global_planner_state in self.global_wait_states
        ) and not self.maneuver.active:
            source = "global_" + global_planner_state.lower()
            if abs(state.speed) > 0.03:
                self.publish_active_brake(
                    self.braking_gear(state, route_command), source
                )
            else:
                self.stop(source)
            self.publish_state(None, "waiting for a valid global route")
            return
        if goal is None or route_command is None or route is None or not route.points:
            self.stop("waiting_for_inputs")
            return
        position = np.asarray(
            [odom.pose.pose.position.x, odom.pose.pose.position.y], dtype=float
        )
        progress = (
            0.0
            if self.last_position is None
            else float(np.linalg.norm(position - self.last_position))
        )
        self.last_position = position
        subgoal = self.subgoal_body(odom, goal)
        body_tracks = self.tracks_body(odom, tracks)
        heading_error = self.heading_error(odom, goal)
        arrival_speed_limit = arrival.speed_limit_mps if arrival is not None else None

        # Once a tight-space leg starts, keep its gear until the adaptive
        # target distance is reached or certified space is exhausted.  This
        # prevents a one-cycle reverse override from immediately flipping back
        # to forward at a 90-degree corner.
        if self.maneuver.active:
            active_bearing = None
            if (
                self.maneuver.purpose == "forward_restoration"
                and route is not None
                and route_command is not None
            ):
                active_reference = self.route_reference(
                    odom, route, route_command.segment_index, grid=grid
                )
                active_reference = self.authority_direction_reference(
                    odom, active_reference, mission_goal
                )
                active_bearing, _ = corridor_direction_body(
                    active_reference,
                    self.forward_preference.config.direction_lookahead_m,
                )
                # Do not terminate a committed leg on one transient route
                # bearing.  Completion is evaluated between stopped legs by
                # ForwardPreferenceSupervisor with confirmation hysteresis.
            if (
                mission_goal is not None
                and self.require_mission_goal_heading
                and mission_distance <= self.terminal_maneuver_radius
                and mission_heading <= self.mission_heading_tolerance
                and self.maneuver.purpose == "terminal_alignment"
                and self.maneuver.state == ManeuverState.DRIVE_LEG
            ):
                self.maneuver.settle("terminal_heading_aligned")
            if self.maneuver.state == ManeuverState.SETTLING:
                reason = self.maneuver.finish_reason
                maneuver_purpose = self.maneuver.purpose
                maneuver_travelled = float(self.maneuver.travelled_m)
                if self.maneuver.finish_if_stopped(state.speed, state.steering):
                    zero_progress_static_recovery = bool(
                        maneuver_purpose == "static_recovery"
                        and reason == "certified_space_exhausted"
                        and maneuver_travelled
                        < self.maneuver.config.minimum_useful_leg_m
                    )
                    if zero_progress_static_recovery:
                        route_key = (
                            self.global_route_source,
                            self.global_route_id,
                            self.global_route_revision,
                            self.route_authority_epoch,
                        )
                        self.static_recovery_replan_gate.block(
                            route_key, position
                        )
                        # This route transaction has proved that it cannot
                        # execute even one useful local recovery leg.  Release
                        # manoeuvre ownership now so memory/FAR can observe the
                        # persistent STATIC_BLOCKED evidence and replace the
                        # route, instead of silently spending all eight legs.
                        self.maneuver.reset()
                        self.local_turnaround_transaction_id = 0
                        self.maneuver_retry_not_before = 0.0
                        self.gear_supervisor.update(
                            Gear.NEUTRAL, state.speed, now
                        )
                        self.stop("static_recovery_far_replan_hold")
                        self.publish_state(
                            None,
                            "zero-progress local recovery released; waiting "
                            "for FAR route replacement",
                            executable_override=False,
                            blocked_by_static_override=True,
                        )
                        return
                    self.maneuver_retry_not_before = (
                        now + self.maneuver_retry_observation_hold
                        if reason == "certified_space_exhausted"
                        else 0.0
                    )
                    self.gear_supervisor.update(Gear.NEUTRAL, state.speed, now)
                    if self.maneuver.purpose == "forward_restoration":
                        # Rebuild the FAR suffix from the measured pose before
                        # selecting the next opposite-gear leg.  The memory
                        # layer keeps this as the same turnaround transaction.
                        self.forward_preference.request_route_revalidation()
                    self.stop("maneuver_leg_complete:" + reason)
                    self.publish_state(None, "committed maneuver leg completed: " + reason)
                else:
                    self.publish_active_brake(
                        self.maneuver.gear, "maneuver_active_braking:" + reason
                    )
                    self.publish_state(None, "settling committed maneuver leg: " + reason)
                return
            maneuver_gear = self.maneuver.gear
            # Complete the zero-speed gear transaction before evaluating a
            # trajectory in the requested direction.  Previously the first
            # active tick planned a REVERSE bank while the measured vehicle
            # was still braking in FORWARD.  Hard safety then rejected the
            # mismatched bank as ``certified_space_exhausted`` before reverse
            # was ever engaged.
            shift_decision = self.gear_supervisor.update(
                maneuver_gear, state.speed, now
            )
            if not shift_decision.drive_enabled:
                self.maneuver.hold_for_drive_authorization(position, now)
                self.publish_shift_hold(
                    shift_decision, state, "maneuver_gear_shift"
                )
                self.publish_state(
                    None,
                    "committed maneuver waiting for zero-speed gear "
                    "engagement; geometric leg is paused",
                    executable_override=True,
                )
                return
            self.maneuver.observe(position, now)
            if self.maneuver.purpose == "forward_restoration":
                self.forward_preference.observe_committed_motion(
                    progress, maneuver_gear
                )
            if self.maneuver.state == ManeuverState.SETTLING:
                reason = self.maneuver.finish_reason
                self.publish_active_brake(
                    maneuver_gear, "maneuver_active_braking:" + reason
                )
                self.publish_state(
                    None,
                    "settling committed maneuver leg: " + reason,
                    executable_override=True,
                )
                return
            maneuver_subgoal = self.maneuver.body_subgoal()
            maneuver_heading_error = (
                self.heading_error(odom, mission_goal)
                if self.maneuver.purpose == "terminal_alignment"
                and mission_goal is not None
                else active_bearing
                if self.maneuver.purpose == "forward_restoration"
                and active_bearing is not None
                else heading_error
            )
            maneuver_result, _ = self.plan_context(
                state,
                maneuver_subgoal,
                grid,
                grid_stamp,
                body_tracks,
                maneuver_gear,
                heading_error=maneuver_heading_error,
                recovery_mode=True,
                spatial_scales=self.maneuver_spatial_scales,
                force_baseline=True,
                required_yaw_direction=(
                    self.maneuver.turn_sign
                    if self.maneuver.purpose == "forward_restoration"
                    and abs(self.maneuver.lateral_target_m) > 1.0e-4
                    else None
                ),
                minimum_yaw_progress_rad=(
                    self.maneuver_minimum_yaw_progress
                    if self.maneuver.purpose == "forward_restoration"
                    and abs(self.maneuver.lateral_target_m) > 1.0e-4
                    else 0.0
                ),
                allow_static_margin_egress=(
                    (
                        self.maneuver.purpose == "forward_restoration"
                        and maneuver_gear == Gear.REVERSE
                    )
                    or self.maneuver.purpose
                    == "far_dead_end_egress_realign"
                ),
            )
            self.publish_candidates(
                maneuver_result,
                maneuver_gear,
                maneuver_subgoal,
                recovery_mode=True,
            )
            if not maneuver_result.executable:
                if maneuver_result.blocked_by_dynamic:
                    self.stop("dynamic_yield_during_maneuver")
                    self.publish_state(maneuver_result, "dynamic obstacle paused maneuver leg")
                else:
                    self.maneuver.settle("certified_space_exhausted")
                    self.publish_active_brake(
                        maneuver_gear,
                        "maneuver_active_braking:certified_space_exhausted",
                    )
                    self.publish_state(
                        maneuver_result,
                        "shortened primitives exhausted; settle before opposite leg",
                    )
                return
            self.recovery.update_blockage(now, progress, False, False)
            decision = self.gear_supervisor.update(
                maneuver_gear, state.speed, now
            )
            if decision.drive_enabled:
                maneuver_limit = (
                    self.maneuver_reverse_speed
                    if maneuver_gear == Gear.REVERSE
                    else self.maneuver_forward_speed
                )
                self.publish_command(
                    maneuver_result.selected,
                    "deterministic_committed_maneuver",
                    speed_limit=self.combined_speed_limit(
                        arrival_speed_limit, maneuver_limit
                    ),
                )
            else:
                self.publish_shift_hold(
                    decision, state, "maneuver_gear_shift"
                )
            self.publish_state(
                maneuver_result,
                "committed_%s_leg %.3f/%.3fm leg=%d"
                % (
                    maneuver_gear.name.lower(),
                    self.maneuver.travelled_m,
                    self.maneuver.target_distance_m,
                    self.maneuver.leg_count,
                ),
            )
            return

        requested_gear = Gear.require_drive(route_command.requested_gear)
        navigation_mode = int(route_command.navigation_mode)
        far_dead_end_egress = bool(
            navigation_mode
            == LocalRouteCommand.NAVIGATION_FAR_DEAD_END_EGRESS
        )
        breadcrumb_backtrack = bool(
            navigation_mode
            == LocalRouteCommand.NAVIGATION_MEMORY_BACKTRACK
        )
        memory_backtrack = (
            breadcrumb_backtrack or far_dead_end_egress
        )
        memory_resume = (
            navigation_mode == LocalRouteCommand.NAVIGATION_MEMORY_RESUME
        )
        exact_route = bool(
            route is not None
            and any(
                int(point.gear) != int(Gear.NEUTRAL) for point in route.points
            )
        )
        # NAVIGATION_CONNECTIVITY with a neutral route is the online mapping
        # probe contract.  It may collect forward observations but cannot
        # authorize a transmission reversal until FAR accepts a route.
        map_acquisition_probe = bool(
            navigation_mode == LocalRouteCommand.NAVIGATION_CONNECTIVITY
            and not exact_route
        )
        reference_path = self.route_reference(
            odom, route, route_command.segment_index, grid=grid
        )
        authority_reference = self.authority_direction_reference(
            odom, reference_path, mission_goal
        )
        reference_curvature = self.reference_curvature(
            odom, route, requested_gear, route_command.segment_index
        )
        turn_angle = route_turn_angle(reference_path)
        turn_soft_severity = corner_severity(
            reference_path,
            trigger_rad=self.corner_soft_trigger,
            full_strength_rad=self.corner_soft_full_strength,
        )
        turn_speed_limit = corner_speed_limit(
            turn_angle,
            straight_speed_mps=self.corner_straight_speed,
            ninety_degree_speed_mps=self.corner_ninety_speed,
        )
        if self.learned_route_authority and self.policy_mode in ("guarded", "active"):
            # V2 must demonstrate its own smooth corner trajectory.  The hard
            # safety layer remains active, but legacy manual corner shaping is
            # not allowed to manufacture a pass.
            turn_speed_limit = None
        mission_subgoal = (
            self.subgoal_body(odom, mission_goal)
            if mission_goal is not None
            else None
        )
        terminal_direct_visible = bool(
            mission_subgoal is not None
            and segment_is_visible(grid, (0.0, 0.0), mission_subgoal)
        )
        rospy.loginfo_throttle(
            2.0,
            "Local route guidance source=%s route_id=%s revision=%d command_index=%d "
            "reference_index=%d turn=%.3frad corner_soft=%.2f corner_speed=%s "
            "terminal_direct_visible=%s",
            self.global_route_source,
            self.global_route_id or "none",
            self.global_route_revision,
            route_command.segment_index,
            self.route_reference_index,
            turn_angle,
            turn_soft_severity,
            "none" if turn_speed_limit is None else "%.3f" % turn_speed_limit,
            terminal_direct_visible,
        )
        result = None
        command_source = ""
        active_subgoal = subgoal
        maneuver_started = False
        forward_course_capture_active = False
        forward_decision = None
        terminal_capture_active = (
            mission_goal is not None
            and not exact_route
            and terminal_capture_route_authorized(
                self.global_route_source
            )
            and mission_distance <= self.terminal_capture_radius
            and (
                not self.require_mission_goal_heading
                or mission_heading < self.terminal_maneuver_heading_trigger
            )
            and terminal_direct_visible
        )
        if terminal_capture_active:
            subgoal = self.subgoal_body(odom, mission_goal)
            active_subgoal = subgoal
            heading_error = (
                self.heading_error(odom, mission_goal)
                if self.require_mission_goal_heading
                else 0.0
            )
            if subgoal[0] > 0.08:
                requested_gear = Gear.FORWARD
            elif subgoal[0] < -0.08:
                requested_gear = Gear.REVERSE
            elif self.gear_supervisor.engaged in (Gear.FORWARD, Gear.REVERSE):
                requested_gear = self.gear_supervisor.engaged

        # A neutral topological corridor supplies connectivity only.  Do not
        # repeat its instantaneous behind/ahead dot product as a transmission
        # command for hundreds of points.  Instead, use local safe primitives
        # to turn around at the first viable site, or reverse only far enough
        # to reach such a site.
        if (
            self.policy_mode != "guarded"
            and
            not exact_route
            and not memory_backtrack
            and not map_acquisition_probe
            and not terminal_capture_active
            and (
                mission_goal is None
                or mission_distance > self.terminal_maneuver_radius
            )
        ):
            if state.speed > 0.03 and self.turnaround_rearm_remaining > 0.0:
                self.turnaround_rearm_remaining = max(
                    0.0, self.turnaround_rearm_remaining - progress
                )
            bearing, route_length = corridor_direction_body(
                authority_reference,
                self.forward_preference.config.direction_lookahead_m,
            )
            behind_or_recovering = (
                route_length > 0.15
                and abs(bearing)
                >= self.forward_preference.config.behind_bearing_rad
            ) or self.forward_preference.state != ForwardPreferenceState.FORWARD_CRUISE
            probes, turnaround_feasible = ({}, False)
            forward_capture_feasible = False
            # The aligned-exit gate is evaluated for both normal cruise and
            # recovery.  Normal cruise intentionally skips the expensive
            # bidirectional probes, so its absent forward probe must be an
            # explicit ``None`` rather than an unbound local.
            forward_probe = None
            continuing_forward_restoration = bool(
                self.maneuver.purpose == "forward_restoration"
                and self.maneuver.leg_count > 0
            )
            if behind_or_recovering:
                probes, turnaround_feasible = self.turnaround_probes(
                    state,
                    authority_reference,
                    grid,
                    body_tracks,
                    heading_error=(
                        math.atan2(
                            authority_reference[-1, 1],
                            authority_reference[-1, 0],
                        )
                        if len(authority_reference) >= 2
                        else heading_error
                    ),
                )
                if (
                    self.forward_preference.state
                    == ForwardPreferenceState.TURNAROUND_PENDING
                    and any(result.executable for _, result in probes.values())
                ):
                    # Both directions are required before starting a new
                    # turnaround site.  Once committed, the safe alternating
                    # next leg may itself create room for the following leg.
                    turnaround_feasible = True
                forward_probe = probes.get(Gear.FORWARD)
                forward_capture_feasible = bool(
                    forward_probe is not None
                    and forward_probe[1] is not None
                    and forward_probe[1].executable
                )
            forward_decision = self.forward_preference.update(
                authority_reference,
                turnaround_feasible=turnaround_feasible,
                forward_capture_feasible=forward_capture_feasible,
                forward_exit_verified=bool(
                    (
                        self.maneuver.purpose == "forward_restoration"
                        and self.maneuver.last_completed_gear == Gear.FORWARD
                        and self.maneuver.last_completed_reason
                        == "target_distance_reached"
                    )
                    or self.stable_far_forward_exit_available(
                        bearing, forward_probe, route_command
                    )
                ),
                progress_m=progress,
                route_requested_gear=route_command.requested_gear,
                turnaround_start_authorized=bool(
                    continuing_forward_restoration
                    or self.turnaround_rearm_remaining <= 0.0
                ),
            )
            if (
                forward_decision.state
                == ForwardPreferenceState.ROUTE_REVALIDATION
                and not route_requires_far_revalidation(
                    self.global_route_source
                )
            ):
                # The route source changed from FAR to an explicit fallback;
                # this is not a discontinuous FAR revision.  Give the durable
                # topology/local corridor one bounded turnaround transaction
                # of its own instead of opening a FAR-only request/hold loop.
                if continuing_forward_restoration:
                    self.forward_preference.approve_continuation_route()
                else:
                    self.forward_preference.approve_revalidated_route()
                forward_decision = self.forward_preference.update(
                    authority_reference,
                    turnaround_feasible=turnaround_feasible,
                    forward_capture_feasible=forward_capture_feasible,
                    forward_exit_verified=bool(
                        (
                            self.maneuver.purpose == "forward_restoration"
                            and self.maneuver.last_completed_gear == Gear.FORWARD
                            and self.maneuver.last_completed_reason
                            == "target_distance_reached"
                        )
                        or self.stable_far_forward_exit_available(
                            bearing, forward_probe, route_command
                        )
                    ),
                    progress_m=progress,
                    route_requested_gear=route_command.requested_gear,
                    turnaround_start_authorized=bool(
                        continuing_forward_restoration
                        or self.turnaround_rearm_remaining <= 0.0
                    ),
                )
                rospy.loginfo(
                    "Resolved rearward non-FAR route through bounded local "
                    "turnaround authority route_source=%s",
                    self.global_route_source,
                )
            if forward_decision.state == ForwardPreferenceState.REVERSE_ESCAPE_EXHAUSTED:
                self.stop("bounded_reverse_escape_exhausted")
                self.publish_state(
                    None,
                    "reverse escape reached %.2fm without finding a safe turnaround site"
                    % forward_decision.reverse_escape_m,
                )
                return
            requested_gear = forward_decision.requested_gear
            if (
                forward_decision.state
                == ForwardPreferenceState.ROUTE_REVALIDATION
            ):
                self.gear_supervisor.update(Gear.NEUTRAL, state.speed, now)
                if abs(state.speed) > 0.03:
                    self.publish_active_brake(
                        self.braking_gear(state, route_command),
                        "forward_course_capture_revalidation_braking",
                    )
                else:
                    self.stop("forward_course_capture_revalidation_hold")
                if not self.forward_capture_replan_requested:
                    # Arm the transaction barrier before publishing the
                    # request: subscriber callbacks can run on another rospy
                    # thread and FAR may answer within this control cycle.
                    current_route_stamp = self.message_stamp(route_command)
                    self.forward_capture_replan_after_stamp = max(
                        float(now), current_route_stamp
                    )
                    self.forward_capture_replan_requested = True
                    self.request_measured_pose_replan(
                        (
                            "abrupt_rear_route_change"
                            if forward_decision.reason
                            == "abrupt_rear_route_revalidation"
                            else "forward_course_capture_diverged"
                        )
                    )
                self.publish_state(
                    None,
                    (
                        "rolling route reversed direction; waiting for a "
                        "measured-pose FAR re-anchor"
                        if forward_decision.reason
                        == "abrupt_rear_route_revalidation"
                        else "forward course capture diverged; waiting for a "
                        "measured-pose FAR re-anchor"
                    ),
                )
                return
            if forward_decision.reason == "safe_forward_course_capture":
                proposed, capture_result = probes[Gear.FORWARD]
                result = capture_result
                active_subgoal = proposed
                command_source = "deterministic_forward_course_capture"
                forward_course_capture_active = True
            if forward_decision.state in (
                ForwardPreferenceState.TURNAROUND_CONFIRM,
                ForwardPreferenceState.TURNAROUND_VERIFY,
            ):
                self.gear_supervisor.update(Gear.NEUTRAL, state.speed, now)
                if abs(state.speed) > 0.03:
                    self.publish_active_brake(
                        self.braking_gear(state, route_command),
                        "turnaround_confirmation_active_braking",
                    )
                else:
                    self.stop("turnaround_confirmation_hold")
                self.publish_state(
                    None,
                    (
                        "confirming corridor behind before starting turnaround"
                        if forward_decision.state
                        == ForwardPreferenceState.TURNAROUND_CONFIRM
                        else "confirming stable forward corridor before releasing turnaround authority"
                    ),
                )
                return
            reverse_space_creation = bool(
                forward_decision.state == ForwardPreferenceState.REVERSE_ESCAPE
                and Gear.REVERSE in probes
                and probes[Gear.REVERSE][1].executable
            )
            if forward_decision.start_turnaround or reverse_space_creation:
                maneuver_order = (
                    (Gear.REVERSE,)
                    if reverse_space_creation
                    else self.turnaround_gear_order(probes)
                )
                for maneuver_gear in maneuver_order:
                    proposed, shortened = probes[maneuver_gear]
                    if not shortened.executable:
                        continue
                    starting_new_turnaround = self.maneuver.leg_count == 0
                    if not self.maneuver.begin(
                        maneuver_gear,
                        position,
                        now,
                        authority_reference[-1],
                        forward_decision.corridor_bearing_rad,
                        purpose="forward_restoration",
                        turn_sign_hint=forward_decision.corridor_bearing_rad,
                    ):
                        if (
                            self.forward_restoration_budget_replans < 1
                            and not self.forward_restoration_budget_replan_pending
                        ):
                            self.forward_restoration_budget_replan_pending = True
                            self.forward_preference.request_route_revalidation()
                            self.forward_capture_replan_after_stamp = max(
                                float(now), self.message_stamp(route_command)
                            )
                            self.forward_capture_replan_requested = True
                            self.request_measured_pose_replan(
                                "forward_restoration_budget_exhausted"
                            )
                            self.stop("forward_restoration_budget_revalidation")
                            self.publish_state(
                                shortened,
                                "forward-restoration scheduling budget exhausted; requesting one measured-pose FAR continuation",
                                executable_override=False,
                                blocked_by_static_override=False,
                            )
                        else:
                            self.stop("forward_restoration_budget_exhausted")
                            self.publish_state(
                                shortened,
                                "turnaround_budget_exhausted: measured-pose continuation also exhausted",
                                executable_override=False,
                                blocked_by_static_override=False,
                            )
                        return
                    if starting_new_turnaround:
                        self.local_turnaround_transaction_sequence += 1
                        self.local_turnaround_transaction_id = (
                            self.local_turnaround_transaction_sequence
                        )
                    requested_gear = maneuver_gear
                    result = shortened
                    active_subgoal = proposed
                    command_source = "deterministic_forward_restoration"
                    maneuver_started = True
                    rospy.loginfo(
                        "Starting forward-restoration %s%s leg target=%.3fm "
                        "corridor_bearing=%.3frad turn_sign=%+.0f lateral_target=%.3fm "
                        "reverse_escape=%.3fm leg=%d route_id=%s "
                        "route_source=%s route_revision=%d progress=%.3fm "
                        "carrot=%.3fm turnaround_id=%d",
                        maneuver_gear.name,
                        " space-creation" if reverse_space_creation else "",
                        self.maneuver.target_distance_m,
                        forward_decision.corridor_bearing_rad,
                        self.maneuver.turn_sign,
                        self.maneuver.lateral_target_m,
                        forward_decision.reverse_escape_m,
                        self.maneuver.leg_count,
                        self.global_route_id or "none",
                        self.global_route_source,
                        self.global_route_revision,
                        self.global_route_progress_m,
                        self.global_route_carrot_m,
                        self.global_turnaround_transaction_id
                        or self.local_turnaround_transaction_id,
                    )
                    break
            rospy.loginfo_throttle(
                2.0,
                "Forward preference state=%s gear=%s bearing=%.3f "
                "reverse_escape=%.3fm reason=%s",
                forward_decision.state.value,
                forward_decision.requested_gear.name,
                forward_decision.corridor_bearing_rad,
                forward_decision.reverse_escape_m,
                forward_decision.reason,
            )

        # A topological corridor can reach the correct goal position while
        # leaving the car facing the wrong way.  That is not a static-blockage
        # event, so trigger an explicit multi-leg Ackermann alignment before
        # the corridor's forward/reverse hint starts oscillating.
        terminal_alignment_required = (
            mission_goal is not None
            and not memory_backtrack
            and self.require_mission_goal_heading
            and not exact_route
            and mission_distance <= self.terminal_maneuver_radius
            and mission_heading >= self.terminal_maneuver_heading_trigger
            and terminal_direct_visible
        )
        if terminal_alignment_required:
            mission_heading_error = self.heading_error(odom, mission_goal)
            alignment_reference = (max(0.10, mission_distance), 0.0)
            for maneuver_gear in self.maneuver.recovery_gear_order(requested_gear):
                proposed = self.maneuver.proposed_subgoal(
                    maneuver_gear,
                    alignment_reference,
                    mission_heading_error,
                    purpose="terminal_alignment",
                )
                shortened, _ = self.plan_context(
                    state,
                    proposed,
                    grid,
                    grid_stamp,
                    body_tracks,
                    maneuver_gear,
                    heading_error=mission_heading_error,
                    recovery_mode=True,
                    spatial_scales=self.maneuver_spatial_scales,
                    force_baseline=True,
                )
                if not shortened.executable:
                    continue
                if not self.maneuver.begin(
                    maneuver_gear,
                    position,
                    now,
                    alignment_reference,
                    mission_heading_error,
                    purpose="terminal_alignment",
                ):
                    self.stop("terminal_maneuver_leg_limit_reached")
                    self.publish_state(
                        shortened, "maximum terminal alignment legs reached"
                    )
                    return
                requested_gear = maneuver_gear
                result = shortened
                active_subgoal = proposed
                command_source = "deterministic_terminal_maneuver"
                maneuver_started = True
                rospy.loginfo(
                    "Starting terminal %s maneuver leg target=%.3fm "
                    "distance=%.3fm heading_error=%.3frad",
                    maneuver_gear.name,
                    self.maneuver.target_distance_m,
                    mission_distance,
                    mission_heading_error,
                )
                break

        if result is None:
            result, command_source = self.plan_context(
                state,
                subgoal,
                grid,
                grid_stamp,
                body_tracks,
                requested_gear,
                heading_error=heading_error,
                reference_curvature=reference_curvature,
                reference_steering=self.reference_steering(
                    odom, route, requested_gear, route_command.segment_index
                ),
                reference_path=reference_path,
                recovery_mode=memory_backtrack or memory_resume,
                allow_static_margin_egress=memory_backtrack,
            )
            if terminal_capture_active:
                command_source = "deterministic_terminal_capture"
        if result is None:
            self.stop("policy_not_ready")
            self.publish_state(None, command_source)
            return
        if (
            self.policy_mode == "guarded"
            and result.blocked_by_static
            and not result.blocked_by_dynamic
        ):
            # V4.2 has already evaluated both gears and all six learned
            # actions under the full-sequence hard veto.  Do not replace its
            # decision with the legacy deterministic turnaround state
            # machine.  Publish real blockage evidence so FAR/memory can
            # replace the route; the model will receive that recovery route
            # on the next query and still owns gear selection.
            self.stop("hybrid_policy_static_blocked")
            self.publish_state(
                result,
                "V4.2 has no full-sequence hard-safe candidate; waiting for "
                "FAR/memory route replacement",
                executable_override=False,
                blocked_by_static_override=True,
            )
            return
        # A full one-second bank being blocked does not mean the car has no
        # room.  Probe progressively shorter certified primitives in the
        # preferred direction and then the opposite direction.  The selected
        # direction becomes a committed leg rather than a one-cycle override.
        if result.blocked_by_static and not result.blocked_by_dynamic:
            route_key = (
                self.global_route_source,
                self.global_route_id,
                self.global_route_revision,
                self.route_authority_epoch,
            )
            if (
                not memory_backtrack
                and not memory_resume
                and self.static_recovery_replan_gate.held(route_key, position)
            ):
                # Keep publishing real local static evidence with manoeuvre
                # ownership released.  Memory navigation can now invalidate
                # the failed FAR suffix and choose a new route/egress; retrying
                # the identical green-route micro-leg cannot add information.
                self.publish_active_brake(
                    self.braking_gear(state, route_command),
                    "static_recovery_far_replan_hold",
                )
                self.publish_state(
                    result,
                    "same FAR route has a zero-progress recovery failure; "
                    "waiting for route replacement",
                    executable_override=False,
                    blocked_by_static_override=True,
                )
                return
            if (
                not memory_backtrack
                and not memory_resume
                and now < self.maneuver_retry_not_before
            ):
                # The previous short leg stopped on this sensor snapshot.
                # Wait for a bounded fresh-observation window before retrying
                # the same gear and geometry; otherwise the 10 Hz loop emits
                # misleading STATIC_BLOCKED/start/stop flashes while SLAM and
                # LiDAR still describe effectively the same scene.
                self.publish_active_brake(
                    self.braking_gear(state, route_command),
                    "maneuver_retry_observation_hold",
                )
                self.publish_state(
                    result,
                    "waiting for a fresh safety observation before the next maneuver leg",
                )
                return
            # Hybrid A* has already supplied a collision-free kinematic tail.
            # Near a segment boundary a full one-second lattice rollout can
            # extend beyond that tail and report a false static dead end.  In
            # this case first retain the route's gear and target but shorten
            # only the control horizon; do not invent an unrelated recovery
            # arc or let the global path replace the local safety check.
            if exact_route:
                shortened, _ = self.plan_context(
                    state,
                    subgoal,
                    grid,
                    grid_stamp,
                    body_tracks,
                    requested_gear,
                    heading_error=heading_error,
                    reference_curvature=reference_curvature,
                    reference_steering=self.reference_steering(
                        odom, route, requested_gear, route_command.segment_index
                    ),
                    reference_path=reference_path,
                    recovery_mode=True,
                    spatial_scales=self.maneuver_spatial_scales,
                    force_baseline=True,
                )
                if shortened.executable:
                    result = shortened
                    command_source = "deterministic_exact_route_micro"
                elif self.exact_replan_gate.authorize(position):
                    self.request_measured_pose_replan(
                        "exact_route_static_safety_exhausted"
                    )
                    self.publish_active_brake(
                        self.braking_gear(state, route_command),
                        "exact_route_replan_active_braking",
                    )
                    self.publish_state(
                        shortened,
                        "exact route has no safe micro primitive; requested replan",
                    )
                    return
                else:
                    # The measured pose has not changed enough for Hybrid A*
                    # to produce materially new geometry.  Let the local
                    # Ackermann recovery below create that room instead of
                    # entering a plan/brake/replan loop at 1 Hz.
                    result = shortened
                    command_source = "deterministic_exact_route_local_recovery"
                    rospy.logwarn_throttle(
                        2.0,
                        "Exact route remained locally blocked at the same "
                        "measured pose; delegating space creation to local recovery",
                    )
            # Ordinary breadcrumb reverse and recovery resume are already
            # certified mission-level gear transactions, so a local fallback
            # must not flip them.  FAR dead-end egress is different: it is a
            # *closed-loop objective* (reach the failed-branch entry), not an
            # open-loop reverse command.  If its current reverse micro-
            # primitive is blocked, a short hard-safe forward steering leg is
            # allowed to create room before the connector resumes.  Suppressing
            # the opposite gear here previously made one blocked reverse ray
            # look like proof that the whole egress was impossible.
            gear_order = (
                ()
                if (
                    breadcrumb_backtrack
                    or memory_resume
                    or map_acquisition_probe
                )
                else self.maneuver.recovery_gear_order(requested_gear)
            )
            for maneuver_gear in gear_order if not result.executable else ():
                maneuver_purpose = (
                    "far_dead_end_egress_realign"
                    if far_dead_end_egress
                    else "static_recovery"
                )
                proposed = self.maneuver.proposed_subgoal(
                    maneuver_gear,
                    subgoal,
                    heading_error,
                    purpose=maneuver_purpose,
                )
                shortened, _ = self.plan_context(
                    state,
                    proposed,
                    grid,
                    grid_stamp,
                    body_tracks,
                    maneuver_gear,
                    heading_error=heading_error,
                    recovery_mode=True,
                    spatial_scales=self.maneuver_spatial_scales,
                    force_baseline=True,
                    # This exception never bypasses hard safety.  It only
                    # permits a primitive which starts inside conservative
                    # inflation to monotonically reduce that overlap.
                    allow_static_margin_egress=far_dead_end_egress,
                )
                if not shortened.executable:
                    continue
                if not self.maneuver.begin(
                    maneuver_gear,
                    position,
                    now,
                    subgoal,
                    heading_error,
                    purpose=maneuver_purpose,
                ):
                    self.stop("maneuver_leg_limit_reached")
                    self.publish_state(shortened, "maximum committed maneuver legs reached")
                    return
                rospy.loginfo(
                    "Starting committed %s maneuver leg target=%.3fm "
                    "subgoal=(%.3f,%.3f) purpose=%s requested_route_gear=%s",
                    maneuver_gear.name,
                    self.maneuver.target_distance_m,
                    proposed[0],
                    proposed[1],
                    maneuver_purpose,
                    requested_gear.name,
                )
                requested_gear = maneuver_gear
                result = shortened
                active_subgoal = proposed
                command_source = (
                    "deterministic_far_egress_realign"
                    if far_dead_end_egress
                    else "deterministic_committed_maneuver"
                )
                maneuver_started = True
                break
        lifecycle = self.recovery.update_blockage(
            now,
            progress,
            result.blocked_by_static,
            result.blocked_by_dynamic,
        )
        if lifecycle == RecoveryState.STATIC_DEADLOCK:
            self.stop("maneuver_no_safe_micro_primitive")
            self.publish_state(
                result,
                "all shortened forward/reverse primitives failed continuous hard safety",
            )
            return
        self.publish_candidates(
            result,
            requested_gear,
            active_subgoal,
            recovery_mode=maneuver_started,
        )
        if result.executable:
            if not maneuver_started and self.maneuver.leg_count:
                completed_forward_restoration = bool(
                    self.maneuver.purpose == "forward_restoration"
                )
                verified_forward_restoration_exit = bool(
                    completed_forward_restoration
                    and forward_decision is not None
                    and forward_decision.reason == "forward_corridor_reacquired"
                )
                if (
                    completed_forward_restoration
                    and not verified_forward_restoration_exit
                ):
                    # A safe ordinary path after one reverse leg is not proof
                    # that the body has completed the turn.  Keep the atomic
                    # transaction alive so the next planning tick schedules
                    # its forward/second reverse leg instead of arming the
                    # new-turn suppression gate and waiting forever.
                    rospy.loginfo_throttle(
                        1.0,
                        "Retaining forward-restoration transaction between "
                        "legs completed=%d decision=%s",
                        self.maneuver.leg_count,
                        (
                            "none"
                            if forward_decision is None
                            else forward_decision.reason
                        ),
                    )
                else:
                    rospy.loginfo(
                        "Verified full-horizon local path after %d maneuver legs",
                        self.maneuver.leg_count,
                    )
                    self.maneuver.reset()
                    self.local_turnaround_transaction_id = 0
                    self.maneuver_retry_not_before = 0.0
                    self.forward_restoration_budget_replans = 0
                    self.forward_restoration_budget_replan_pending = False
                if verified_forward_restoration_exit:
                    # Suppress source-flap reinterpretation of the same
                    # mission as a brand-new turnaround until the car has
                    # made meaningful forward progress (or the bounded rearm
                    # progress.  Explicit memory/dead-end authority and a new
                    # RViz goal reset this latch independently.
                    self.turnaround_rearm_remaining = (
                        self.turnaround_rearm_distance
                    )
            if self.policy_mode == "guarded":
                requested_gear = Gear.require_drive(result.selected.gear)
            decision = self.gear_supervisor.update(requested_gear, state.speed, now)
            if decision.drive_enabled:
                if self.policy_mode == "guarded":
                    self.hybrid_action_latch.observe_drive_authorized(
                        now, requested_gear
                    )
                maneuver_limit = None
                if maneuver_started:
                    maneuver_limit = (
                        self.maneuver_reverse_speed
                        if requested_gear == Gear.REVERSE
                        else self.maneuver_forward_speed
                    )
                self.publish_command(
                    result.selected,
                    command_source,
                    speed_limit=self.combined_speed_limit(
                        arrival_speed_limit,
                        maneuver_limit,
                        self.terminal_capture_speed
                        if terminal_capture_active
                        else None,
                        self.exact_route_speed if exact_route else None,
                        turn_speed_limit,
                        self.forward_course_capture_speed
                        if forward_course_capture_active
                        else None,
                    ),
                )
            else:
                self.publish_shift_hold(decision, state, "route_gear_shift")
        else:
            self.stop("dynamic_yield" if result.blocked_by_dynamic else "static_blocked")
        self.publish_state(result, command_source)


if __name__ == "__main__":
    rospy.init_node("dep_car_local_planner")
    LocalPlannerNode()
    rospy.spin()
