#!/usr/bin/env python3
"""P6 local authority: deterministic gear/safety around learned candidates."""

import json
import math
import threading

import numpy as np
import rospy
from dep_car.core.gear import GearSupervisor
from dep_car.runtime.occupancy import RuntimeOccupancyGrid2D
from dep_car.runtime.route_guidance import (
    apply_corner_clearance_preference,
    apply_runtime_route_preference,
    corner_severity,
    corner_speed_limit,
    route_reference_body,
    route_turn_angle,
    segment_is_visible,
)
from dep_car.runtime.forward_preference import (
    ForwardPreferenceConfig,
    ForwardPreferenceState,
    ForwardPreferenceSupervisor,
    corridor_direction_body,
)
from dep_car.core.planner import DeterministicPlanner
from dep_car.core.recovery import RecoveryManager, RecoveryState
from dep_car.core.types import Candidate as CoreCandidate
from dep_car.core.types import DynamicTrack, Gear, VehicleState
from dep_car.core.vehicle import (
    center_steering_from_wheel_angles,
    world_velocity_to_body_longitudinal,
)
from dep_car.runtime.safety import evaluate_learned_candidate_bank
from dep_car.runtime.arrival import ArrivalConfig, GoalArrivalController
from dep_car.runtime.maneuver import (
    CommittedManeuver,
    ManeuverConfig,
    ManeuverState,
    MeasuredPoseReplanGate,
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
        self.grid = self.grid_stamp = self.joint_state = None
        self.odom = self.goal = self.mission_goal = self.route_command = self.route = None
        self.policy_raw = self.policy_inference_state = None
        self.global_planner_state = "IDLE"
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
                )
            )
        )
        self.maneuver_spatial_scales = (1.0, 0.75, 0.50, 0.35, 0.25)
        self.maneuver_minimum_yaw_progress = float(
            rospy.get_param("~maneuver_minimum_yaw_progress", 0.035)
        )
        self.exact_replan_gate = MeasuredPoseReplanGate(
            float(rospy.get_param("~exact_replan_minimum_displacement", 0.25))
        )
        self.maneuver_forward_speed = float(
            rospy.get_param("~maneuver_forward_speed", 0.45)
        )
        self.maneuver_reverse_speed = float(
            rospy.get_param("~maneuver_reverse_speed", 0.35)
        )
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
        if self.policy_mode not in ("disabled", "shadow", "active"):
            raise ValueError("policy_mode must be disabled, shadow or active")
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
        self.command_pub = rospy.Publisher(
            "/dep_car/cmd_ackermann", AckermannCommand, queue_size=1
        )
        self.candidates_pub = rospy.Publisher(
            "/dep_car/candidates", CandidateArray, queue_size=1
        )
        self.policy_candidates_pub = rospy.Publisher(
            "/dep_car/policy_candidates", CandidateArray, queue_size=1
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
        rospy.Subscriber("/base_pose_ground_truth", Odometry, self.on_odom, queue_size=1)
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
            self.goal = message

    def on_route_command(self, message):
        goal = PoseStamped()
        goal.header = message.header
        goal.pose = message.target
        with self.lock:
            self.route_command = message
            self.goal = goal

    def on_route(self, message):
        with self.lock:
            self.route = message

    def on_global_planner_status(self, message):
        try:
            state = str(json.loads(message.data).get("state", "IDLE"))
        except (TypeError, ValueError):
            state = "INVALID_STATUS"
        with self.lock:
            self.global_planner_state = state

    def on_mission_goal(self, message):
        with self.lock:
            self.mission_goal = message
            self.goal = None
            self.route_command = None
            self.route = None
            self.policy_raw = None
            self.last_position = None
            self.arrival.reset()
            self.maneuver.reset()
            self.forward_preference.reset()
            self.exact_replan_gate.reset()
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
    def vehicle_state(odom, joint_state=None):
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
            speed=world_velocity_to_body_longitudinal(velocity.x, velocity.y, yaw),
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

    def route_reference(self, odom, route, start_index=0):
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
        return route_reference_body(
            world,
            vehicle_pose,
            start_index,
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
            proposed = self.maneuver.proposed_subgoal(
                gear,
                directional_reference,
                bearing if math.isfinite(bearing) else heading_error,
                purpose="forward_restoration",
                turn_sign_hint=turn_hint,
            )
            result = self.planner.plan(
                state,
                proposed,
                grid,
                tracks,
                requested_gear=gear,
                target_heading=bearing,
                spatial_scales=self.maneuver_spatial_scales,
                required_yaw_direction=turn_hint,
                minimum_yaw_progress_rad=self.maneuver_minimum_yaw_progress,
            )
            probes[gear] = (proposed, result)
        # A turnaround site must support both halves of a local multi-point
        # correction.  One safe reverse arc alone is an escape corridor, not
        # evidence that the car can already restore forward travel there.
        feasible = all(probes[gear][1].executable for gear in probes)
        return probes, feasible

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
        self.policy_query_pub.publish(message)
        return self.query_generation

    @staticmethod
    def policy_core_candidates(message):
        if len(message.candidates) != 15:
            raise ValueError("raw policy bank does not contain 15 candidates")
        output = []
        for expected_id, item in enumerate(message.candidates):
            if int(item.candidate_id) != expected_id:
                raise ValueError("raw policy candidate ordering changed")
            arrays = [
                np.asarray(values, dtype=np.float64)
                for values in (item.time, item.x, item.y, item.yaw, item.speed, item.steering)
            ]
            if any(values.shape != (11,) for values in arrays):
                raise ValueError("raw policy trajectory must contain eleven rows")
            trajectory = np.column_stack(arrays)
            output.append(
                CoreCandidate(
                    candidate_id=expected_id,
                    speed_anchor=float(item.speed_anchor),
                    steering_anchor=float(item.steering_anchor),
                    duration=float(item.duration),
                    trajectory=trajectory,
                    gear=Gear.require_drive(item.gear),
                    learned_score=float(item.learned_score),
                )
            )
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
        if self.policy_mode == "active" and (
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
        if abs(float(grid_stamp) - stamp) > self.policy_grid_skew:
            return None, "policy_costmap_skew"
        if int(raw.requested_gear) != int(requested_gear):
            return None, "policy_gear_mismatch"
        if bool(raw.recovery_mode) != bool(recovery_mode):
            return None, "policy_context_mismatch"
        raw_subgoal = np.asarray([raw.subgoal_body.x, raw.subgoal_body.y], dtype=float)
        if float(np.linalg.norm(raw_subgoal - np.asarray(subgoal, dtype=float))) > self.policy_subgoal_tolerance:
            return None, "policy_subgoal_mismatch"
        try:
            candidates = self.policy_core_candidates(raw)
            result = evaluate_learned_candidate_bank(
                candidates,
                subgoal,
                grid,
                tracks,
                generation=raw.generation,
            )
            if not self.learned_route_authority:
                result = apply_runtime_route_preference(
                    result,
                    reference_path,
                    grid,
                    corridor_weight=self.route_corridor_weight,
                    desired_future_clearance_m=self.route_clearance_target,
                    clearance_weight=self.route_clearance_weight,
                    corner_corridor_minimum_scale=self.corner_corridor_minimum_scale,
                )
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
        return result, "policy_ready" if result.executable else "policy_zero_feasible"

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
        policy, reason = self.policy_result(
            grid,
            grid_stamp,
            tracks,
            requested_gear,
            subgoal,
            recovery_mode=recovery_mode,
            reference_path=reference_path,
        )
        if policy is not None:
            self.publish_candidates(
                policy,
                requested_gear,
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
            return policy, "dep_car_net_v1_active"
        if self.active_fallback_to_baseline:
            return baseline, "deterministic_active_fallback"
        return None, reason

    def publish_command(self, candidate, source, speed_limit=None):
        if rospy.is_shutdown():
            return
        lookahead = min(2, len(candidate.trajectory) - 1)
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

    def publish_state(self, result, detail=""):
        if rospy.is_shutdown():
            return
        message = PlannerState()
        message.header.stamp = rospy.Time.now()
        message.lifecycle_state = (
            self.maneuver.state.value
            if self.maneuver.active
            else self.recovery.state.value
        )
        message.executable = bool(result is not None and result.executable)
        message.blocked_by_static = bool(result is not None and result.blocked_by_static)
        message.blocked_by_dynamic = bool(result is not None and result.blocked_by_dynamic)
        message.planning_generation = result.generation if result is not None else 0
        message.retime_factor = (result.retime_factor or 0.0) if result is not None else 0.0
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
            message.model_loaded = inference.model_loaded
            message.sensor_ready = inference.sensor_ready
            message.inference_ok = inference.inference_ok
            message.inference_latency_ms = inference.inference_latency_ms
            message.sensor_skew_s = inference.sensor_skew_s
            inference_authorized = inference.control_authorized
        else:
            inference_authorized = False
        message.hard_safety_applied = result is not None
        message.executable = bool(result is not None and result.executable)
        message.control_authorized = bool(
            self.policy_mode == "active" and inference_authorized
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
        state = self.vehicle_state(odom, joint_state)
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
        if global_planner_state in (
            "RECEIVED",
            "PLANNING",
            "TIMEOUT",
            "NO_PATH",
            "INVALID_GOAL",
            "INVALID_START",
            "START_BLOCKED",
            "INVALID_STATUS",
        ):
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
            self.maneuver.observe(position, now)
            active_bearing = None
            if (
                self.maneuver.purpose == "forward_restoration"
                and route is not None
                and route_command is not None
            ):
                active_reference = self.route_reference(
                    odom, route, route_command.segment_index
                )
                active_bearing, _ = corridor_direction_body(
                    active_reference,
                    self.forward_preference.config.direction_lookahead_m,
                )
                if self.forward_preference.forward_corridor_reacquired(
                    active_reference,
                    route_requested_gear=route_command.requested_gear,
                ):
                    self.maneuver.settle("forward_corridor_reacquired")
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
                if self.maneuver.finish_if_stopped(state.speed, state.steering):
                    self.gear_supervisor.update(Gear.NEUTRAL, state.speed, now)
                    self.stop("maneuver_leg_complete:" + reason)
                    self.publish_state(None, "committed maneuver leg completed: " + reason)
                else:
                    self.publish_active_brake(
                        self.maneuver.gear, "maneuver_active_braking:" + reason
                    )
                    self.publish_state(None, "settling committed maneuver leg: " + reason)
                return
            maneuver_gear = self.maneuver.gear
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
                    else None
                ),
                minimum_yaw_progress_rad=(
                    self.maneuver_minimum_yaw_progress
                    if self.maneuver.purpose == "forward_restoration"
                    else 0.0
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
        exact_route = bool(
            route is not None
            and any(
                int(point.gear) != int(Gear.NEUTRAL) for point in route.points
            )
        )
        reference_path = self.route_reference(
            odom, route, route_command.segment_index
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
        if self.learned_route_authority and self.policy_mode == "active":
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
            "Local route guidance index=%d turn=%.3frad corner_soft=%.2f corner_speed=%s "
            "terminal_direct_visible=%s",
            route_command.segment_index,
            turn_angle,
            turn_soft_severity,
            "none" if turn_speed_limit is None else "%.3f" % turn_speed_limit,
            terminal_direct_visible,
        )
        result = None
        command_source = ""
        active_subgoal = subgoal
        maneuver_started = False
        terminal_capture_active = (
            mission_goal is not None
            and not exact_route
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
            not exact_route
            and not terminal_capture_active
            and (
                mission_goal is None
                or mission_distance > self.terminal_maneuver_radius
            )
        ):
            bearing, route_length = corridor_direction_body(
                reference_path,
                self.forward_preference.config.direction_lookahead_m,
            )
            behind_or_recovering = (
                route_length > 0.15
                and abs(bearing)
                >= self.forward_preference.config.behind_bearing_rad
            ) or self.forward_preference.state != ForwardPreferenceState.FORWARD_CRUISE
            probes, turnaround_feasible = ({}, False)
            if behind_or_recovering:
                probes, turnaround_feasible = self.turnaround_probes(
                    state,
                    reference_path,
                    grid,
                    body_tracks,
                    heading_error=heading_error,
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
            forward_decision = self.forward_preference.update(
                reference_path,
                turnaround_feasible=turnaround_feasible,
                progress_m=progress,
                route_requested_gear=route_command.requested_gear,
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
            if forward_decision.start_turnaround:
                for maneuver_gear in self.maneuver.recovery_gear_order(Gear.FORWARD):
                    proposed, shortened = probes[maneuver_gear]
                    if not shortened.executable:
                        continue
                    if not self.maneuver.begin(
                        maneuver_gear,
                        position,
                        now,
                        reference_path[-1],
                        forward_decision.corridor_bearing_rad,
                        purpose="forward_restoration",
                        turn_sign_hint=forward_decision.corridor_bearing_rad,
                    ):
                        self.stop("forward_restoration_leg_limit_reached")
                        self.publish_state(
                            shortened,
                            "maximum forward-restoration legs reached",
                        )
                        return
                    requested_gear = maneuver_gear
                    result = shortened
                    active_subgoal = proposed
                    command_source = "deterministic_forward_restoration"
                    maneuver_started = True
                    rospy.loginfo(
                        "Starting forward-restoration %s leg target=%.3fm "
                        "corridor_bearing=%.3frad turn_sign=%+.0f "
                        "reverse_escape=%.3fm leg=%d",
                        maneuver_gear.name,
                        self.maneuver.target_distance_m,
                        forward_decision.corridor_bearing_rad,
                        self.maneuver.turn_sign,
                        forward_decision.reverse_escape_m,
                        self.maneuver.leg_count,
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
            )
            if terminal_capture_active:
                command_source = "deterministic_terminal_capture"
        if result is None:
            self.stop("policy_not_ready")
            self.publish_state(None, command_source)
            return
        # A full one-second bank being blocked does not mean the car has no
        # room.  Probe progressively shorter certified primitives in the
        # preferred direction and then the opposite direction.  The selected
        # direction becomes a committed leg rather than a one-cycle override.
        if result.blocked_by_static and not result.blocked_by_dynamic:
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
            gear_order = self.maneuver.recovery_gear_order(requested_gear)
            for maneuver_gear in gear_order if not result.executable else ():
                proposed = self.maneuver.proposed_subgoal(
                    maneuver_gear, subgoal, heading_error
                )
                shortened = self.planner.plan(
                    state,
                    proposed,
                    grid,
                    body_tracks,
                    requested_gear=maneuver_gear,
                    target_heading=heading_error,
                    spatial_scales=self.maneuver_spatial_scales,
                )
                if not shortened.executable:
                    continue
                if not self.maneuver.begin(
                    maneuver_gear,
                    position,
                    now,
                    subgoal,
                    heading_error,
                ):
                    self.stop("maneuver_leg_limit_reached")
                    self.publish_state(shortened, "maximum committed maneuver legs reached")
                    return
                rospy.loginfo(
                    "Starting committed %s maneuver leg target=%.3fm subgoal=(%.3f,%.3f)",
                    maneuver_gear.name,
                    self.maneuver.target_distance_m,
                    proposed[0],
                    proposed[1],
                )
                requested_gear = maneuver_gear
                result = shortened
                active_subgoal = proposed
                command_source = "deterministic_committed_maneuver"
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
                rospy.loginfo(
                    "Normal full-horizon local path reacquired after %d maneuver legs",
                    self.maneuver.leg_count,
                )
                self.maneuver.reset()
            decision = self.gear_supervisor.update(
                requested_gear, state.speed, now
            )
            if decision.drive_enabled:
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
