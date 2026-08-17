#!/usr/bin/env python3
"""ROS wrapper for learned offsets plus deterministic runtime safety."""

import math
import threading

import numpy as np
import rospy
from dep_car.core.gear import GearSupervisor
from dep_car.core.occupancy import OccupancyGrid2D
from dep_car.core.planner import DeterministicPlanner
from dep_car.core.recovery import RecoveryManager, RecoveryState
from dep_car.core.types import DynamicTrack, Gear, VehicleState
from dep_car.core.vehicle import world_velocity_to_body_longitudinal
from dep_car_msgs.msg import AckermannCommand, Candidate, CandidateArray, DynamicTrackArray, LocalRouteCommand, PlannerState
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from sensor_msgs.msg import Image


def yaw_from_quaternion(quaternion):
    return math.atan2(
        2.0 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y),
        1.0 - 2.0 * (quaternion.y * quaternion.y + quaternion.z * quaternion.z),
    )


class LocalPlannerNode:
    def __init__(self):
        self.lock = threading.Lock()
        self.planner = DeterministicPlanner()
        self.recovery = RecoveryManager()
        self.gear_supervisor = GearSupervisor()
        self.grid = None
        self.odom = None
        self.goal = None
        self.mission_goal = None
        self.route_command = None
        self.range_image = self.validity_mask = None
        self.tracks = []
        self.last_position = None
        self.reverse_origin = None
        self.reverse_distance = rospy.get_param("~reverse_distance", 0.6)
        self.mission_goal_tolerance = rospy.get_param("~mission_goal_tolerance", 0.22)
        self.mission_heading_tolerance = rospy.get_param("~mission_heading_tolerance", 0.35)
        self.command_pub = rospy.Publisher("/dep_car/cmd_ackermann", AckermannCommand, queue_size=1)
        self.candidates_pub = rospy.Publisher("/dep_car/candidates", CandidateArray, queue_size=1)
        self.state_pub = rospy.Publisher("/dep_car/planner_state", PlannerState, queue_size=1)
        rospy.Subscriber("/dep_car/local_costmap", OccupancyGrid, self.on_grid, queue_size=1)
        rospy.Subscriber("/base_pose_ground_truth", Odometry, self.on_odom, queue_size=1)
        rospy.Subscriber(rospy.get_param("~goal_topic", "/dep_car/local_subgoal"), PoseStamped, self.on_goal, queue_size=1)
        rospy.Subscriber("/dep_car/local_route_command", LocalRouteCommand, self.on_route_command, queue_size=1)
        rospy.Subscriber("/move_base_simple/goal", PoseStamped, self.on_mission_goal, queue_size=1)
        rospy.Subscriber("/dep_car/dynamic/tracks", DynamicTrackArray, self.on_tracks, queue_size=1)
        rospy.Subscriber("/dep_car/lidar/range_image", Image, self.on_range_image, queue_size=1)
        rospy.Subscriber("/dep_car/lidar/validity_mask", Image, self.on_validity_mask, queue_size=1)
        self.learned_model = self._load_model()
        rate = rospy.get_param("~control_rate", 10.0)
        self.timer = rospy.Timer(rospy.Duration(1.0 / rate), self.update)

    def on_grid(self, message):
        data = np.asarray(message.data, dtype=np.int16).reshape((message.info.height, message.info.width))
        with self.lock:
            self.grid = OccupancyGrid2D(
                data,
                message.info.resolution,
                (message.info.origin.position.x, message.info.origin.position.y),
            )

    def on_odom(self, message):
        with self.lock:
            self.odom = message

    def on_goal(self, message):
        with self.lock:
            self.goal = message

    def on_route_command(self, message):
        goal = PoseStamped(); goal.header = message.header; goal.pose = message.target
        with self.lock:
            self.route_command = message
            self.goal = goal

    def on_mission_goal(self, message):
        with self.lock:
            self.mission_goal = message
            # Invalidate the previous mission's latched subgoal/gear command.
            # Translation resumes only after Hybrid A* publishes a fresh pair.
            self.goal = None
            self.route_command = None
            self.recovery.set_mission_goal((message.pose.position.x, message.pose.position.y))

    def on_tracks(self, message):
        with self.lock:
            self.tracks = [
                DynamicTrack(
                    track_id=item.track_id, x=item.x, y=item.y, vx=item.vx, vy=item.vy,
                    radius=item.radius, confidence=item.confidence,
                    covariance=np.asarray(item.position_covariance).reshape((2, 2)),
                ) for item in message.tracks
            ]

    def tracks_body(self, odom, tracks):
        heading = yaw_from_quaternion(odom.pose.pose.orientation); cosine, sine = math.cos(heading), math.sin(heading)
        origin_x, origin_y = odom.pose.pose.position.x, odom.pose.pose.position.y
        return tuple(DynamicTrack(
            track_id=item.track_id,
            x=cosine * (item.x - origin_x) + sine * (item.y - origin_y),
            y=-sine * (item.x - origin_x) + cosine * (item.y - origin_y),
            vx=cosine * item.vx + sine * item.vy,
            vy=-sine * item.vx + cosine * item.vy,
            radius=item.radius, covariance=item.covariance, confidence=item.confidence, stamp=item.stamp,
        ) for item in tracks)

    @staticmethod
    def decode_image(message):
        return np.frombuffer(message.data, dtype=np.float32).reshape(message.height, message.width).copy()

    def on_range_image(self, message): self.range_image = self.decode_image(message)
    def on_validity_mask(self, message): self.validity_mask = self.decode_image(message)

    def _load_model(self):
        checkpoint = rospy.get_param("~checkpoint", "")
        contract = rospy.get_param("~checkpoint_contract", "")
        if not checkpoint:
            rospy.logwarn("No trained DE-P-Car checkpoint configured; using deterministic lattice baseline")
            return None
        try:
            import torch
            from dep_car.model.checkpoint import verify_checkpoint
            from dep_car.model.lidar_dep import LidarDEPCarV1
            verify_checkpoint(checkpoint, contract)
            model = LidarDEPCarV1()
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            model.load_state_dict(payload["model_state_dict"]); model.eval()
            rospy.loginfo("Loaded production-qualified DE-P-Car checkpoint: %s", checkpoint)
            return model
        except Exception as exc:
            rospy.logerr("DE-P-Car checkpoint rejected; deterministic fallback active: %s", exc)
            return None

    def learned_offsets(self, state, subgoal, requested_gear):
        if requested_gear != Gear.FORWARD or self.learned_model is None or self.range_image is None or self.validity_mask is None:
            return None
        import torch
        distance = math.hypot(*subgoal); bearing = math.atan2(subgoal[1], subgoal[0])
        lidar = torch.from_numpy(np.stack((self.range_image, self.validity_mask))[None]).float()
        vehicle = torch.tensor([[state.speed, state.acceleration, state.steering, state.yaw_rate,
                                 distance, math.sin(bearing), math.cos(bearing), 0.0]], dtype=torch.float32)
        with torch.no_grad(): outputs = self.learned_model(lidar, vehicle)
        return tuple(output[0].cpu().numpy() for output in outputs)

    def stop(self, source="local_planner"):
        if rospy.is_shutdown():
            return
        command = AckermannCommand()
        command.header.stamp = rospy.Time.now()
        command.gear = int(Gear.NEUTRAL)
        command.brake = True
        command.source = source
        self.command_pub.publish(command)

    def subgoal_body(self, odom, goal):
        yaw = yaw_from_quaternion(odom.pose.pose.orientation)
        dx = goal.pose.position.x - odom.pose.pose.position.x
        dy = goal.pose.position.y - odom.pose.pose.position.y
        return (math.cos(yaw) * dx + math.sin(yaw) * dy, -math.sin(yaw) * dx + math.cos(yaw) * dy)

    def vehicle_state(self, odom):
        yaw = yaw_from_quaternion(odom.pose.pose.orientation)
        velocity = odom.twist.twist.linear
        return VehicleState(
            speed=world_velocity_to_body_longitudinal(velocity.x, velocity.y, yaw),
            yaw_rate=odom.twist.twist.angular.z,
            stamp=odom.header.stamp.to_sec(),
        )

    def publish_command(self, candidate, source="learned_candidate"):
        if rospy.is_shutdown():
            return
        lookahead = min(2, len(candidate.trajectory) - 1)
        command = AckermannCommand()
        command.header.stamp = rospy.Time.now()
        command.speed = float(candidate.trajectory[lookahead, 4])
        command.steering_angle = float(candidate.trajectory[lookahead, 5])
        command.gear = int(candidate.gear)
        command.brake = False
        command.source = source
        self.command_pub.publish(command)

    def publish_state(self, result, detail=""):
        if rospy.is_shutdown():
            return
        message = PlannerState()
        message.header.stamp = rospy.Time.now()
        message.lifecycle_state = self.recovery.state.value
        message.executable = result.executable
        message.blocked_by_static = result.blocked_by_static
        message.blocked_by_dynamic = result.blocked_by_dynamic
        message.planning_generation = result.generation
        message.retime_factor = result.retime_factor or 0.0
        message.detail = detail
        self.state_pub.publish(message)

    def publish_candidates(self, result, requested_gear, subgoal_body, recovery_mode=False):
        if rospy.is_shutdown():
            return
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
        self.candidates_pub.publish(array)

    def update(self, _event):
        if rospy.is_shutdown():
            return
        with self.lock:
            grid, odom, goal, tracks, route_command, mission_goal = (
                self.grid, self.odom, self.goal, tuple(self.tracks), self.route_command, self.mission_goal
            )
        if grid is None or odom is None or goal is None or route_command is None:
            self.stop("waiting_for_inputs")
            return
        position = np.asarray([odom.pose.pose.position.x, odom.pose.pose.position.y])
        progress = 0.0 if self.last_position is None else float(np.linalg.norm(position - self.last_position))
        self.last_position = position
        state = self.vehicle_state(odom)
        if mission_goal is not None:
            mission_distance = math.hypot(
                mission_goal.pose.position.x - odom.pose.pose.position.x,
                mission_goal.pose.position.y - odom.pose.pose.position.y,
            )
            mission_heading_error = abs(math.atan2(
                math.sin(yaw_from_quaternion(mission_goal.pose.orientation) - yaw_from_quaternion(odom.pose.pose.orientation)),
                math.cos(yaw_from_quaternion(mission_goal.pose.orientation) - yaw_from_quaternion(odom.pose.pose.orientation)),
            ))
            if mission_distance <= self.mission_goal_tolerance and mission_heading_error <= self.mission_heading_tolerance:
                self.gear_supervisor.update(Gear.NEUTRAL, state.speed, rospy.Time.now().to_sec())
                self.stop("mission_goal_reached")
                return
        subgoal = self.subgoal_body(odom, goal)
        requested_gear = Gear(int(route_command.requested_gear))
        result = self.planner.plan(
            state,
            subgoal,
            grid,
            self.tracks_body(odom, tracks),
            requested_gear=requested_gear,
            learned_offsets=self.learned_offsets(state, subgoal, requested_gear),
        )
        lifecycle = self.recovery.update_blockage(
            rospy.Time.now().to_sec(), progress, result.blocked_by_static, result.blocked_by_dynamic
        )
        if lifecycle == RecoveryState.STATIC_DEADLOCK:
            recovery_result = self.planner.plan(
                state, (-self.reverse_distance, 0.0), grid,
                self.tracks_body(odom, tracks), requested_gear=Gear.REVERSE,
            )
            self.publish_candidates(
                recovery_result, Gear.REVERSE, (-self.reverse_distance, 0.0), recovery_mode=True,
            )
            if recovery_result.executable:
                self.recovery.begin_recovery(
                    rospy.Time.now().to_sec(), tuple(recovery_result.selected.trajectory[-1, 1:3])
                )
                self.reverse_origin = position.copy()
            else:
                self.stop("recovery_no_safe_reverse")
                self.publish_state(recovery_result, "no certified reverse candidate bank")
                return
        if self.recovery.state == RecoveryState.REVERSE_RECOVERY:
            travelled = float(np.linalg.norm(position - self.reverse_origin))
            if travelled >= self.reverse_distance:
                self.stop("recovery_reverse_complete")
                self.recovery.finish_reverse()
            else:
                remaining = max(0.15, self.reverse_distance - travelled)
                recovery_result = self.planner.plan(
                    state, (-remaining, 0.0), grid,
                    self.tracks_body(odom, tracks), requested_gear=Gear.REVERSE,
                )
                self.publish_candidates(
                    recovery_result, Gear.REVERSE, (-remaining, 0.0), recovery_mode=True,
                )
                if not recovery_result.executable:
                    self.stop("recovery_no_safe_reverse")
                    self.publish_state(recovery_result, "reverse bank lost hard-safety certification")
                    return
                decision = self.gear_supervisor.update(Gear.REVERSE, state.speed, rospy.Time.now().to_sec())
                if decision.drive_enabled:
                    self.publish_command(recovery_result.selected, "bounded_reverse_recovery")
                else:
                    self.stop("recovery_gear_shift_" + decision.state.value.lower())
                self.publish_state(recovery_result, "replanned reverse bank owns translation")
                return
            self.publish_candidates(result, requested_gear, subgoal)
            self.publish_state(result, "recovery reverse complete")
            return
        self.publish_candidates(result, requested_gear, subgoal)
        if self.recovery.state == RecoveryState.FORWARD_ESCAPE and result.executable:
            decision = self.gear_supervisor.update(Gear.FORWARD, state.speed, rospy.Time.now().to_sec())
            if decision.drive_enabled:
                self.recovery.finish_escape()
                self.recovery.mission_plan_reacquired()
            else:
                self.stop("escape_gear_shift_" + decision.state.value.lower())
                self.publish_state(result, "stop-before-forward-shift authority")
                return
        if result.executable:
            decision = self.gear_supervisor.update(requested_gear, state.speed, rospy.Time.now().to_sec())
            if decision.drive_enabled:
                source = "learned_candidate" if self.learned_model is not None else "deterministic_lattice"
                self.publish_command(result.selected, source)
            else:
                self.stop("route_gear_shift_" + decision.state.value.lower())
        else:
            self.stop("dynamic_yield" if result.blocked_by_dynamic else "static_blocked")
        self.publish_state(result)


if __name__ == "__main__":
    rospy.init_node("dep_car_local_planner")
    LocalPlannerNode()
    rospy.spin()
