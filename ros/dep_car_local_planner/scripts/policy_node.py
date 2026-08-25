#!/usr/bin/env python3
"""GPU DE-P-Car inference node for P6 shadow/active simulation.

The process intentionally publishes raw learned trajectories only.  The
system-Python local planner remains the sole static/dynamic hard-safety and
actuation authority.
"""

import math
import threading

import numpy as np
import rospy
from dep_car.core.vehicle import (
    center_steering_from_wheel_angles,
    world_velocity_to_body_longitudinal,
)
from dep_car.runtime.p6_policy import P6PolicyRuntime, PolicyArtifactError
from dep_car.runtime.hybrid_sequence import JointGearHistoryTracker
from dep_car.runtime.online_sync import (
    StampedHistory,
    interpolated,
    nearest,
    newest_synchronized_anchor,
)
from dep_car.runtime.preprocessing import (
    build_joint_policy_state,
    build_policy_state,
    current_gear_from_speed,
    normalize_depth_metric,
    normalize_lidar_bev,
)
from dep_car_msgs.msg import (
    PolicyCandidate,
    PolicyCandidateArray,
    PolicyQuery,
    PolicyState,
)
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, Imu, JointState


def yaw_from_quaternion(quaternion):
    return math.atan2(
        2.0 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y),
        1.0 - 2.0 * (quaternion.y * quaternion.y + quaternion.z * quaternion.z),
    )


def message_stamp(message):
    stamp = message.header.stamp.to_sec()
    return stamp if stamp > 0.0 else rospy.Time.now().to_sec()


class PolicyNode:
    def __init__(self):
        self.lock = threading.Lock()
        history_length = int(rospy.get_param("~synchronization_history", 32))
        self.histories = {
            name: StampedHistory(history_length)
            for name in ("depth", "bev", "odom", "imu", "joints")
        }
        self.query = None
        self.last_generation = -1
        self.inference_attempts = 0
        self.synchronization_failures = 0
        self.maximum_sensor_age = float(rospy.get_param("~maximum_sensor_age", 0.35))
        self.depth_tolerance = float(rospy.get_param("~depth_sync_tolerance", 0.05))
        self.odom_tolerance = float(rospy.get_param("~odom_sync_tolerance", 0.02))
        self.imu_tolerance = float(rospy.get_param("~imu_sync_tolerance", 0.02))
        self.joint_tolerance = float(rospy.get_param("~joint_sync_tolerance", 0.05))
        self.sim_positive_right = bool(rospy.get_param("~simulator_positive_right", True))
        self.odometry_twist_in_body_frame = bool(
            rospy.get_param("~odometry_twist_in_body_frame", False)
        )
        self.mode = str(rospy.get_param("~mode", "shadow"))
        self.modality = str(rospy.get_param("~modality", "fusion"))
        self.gear_history = JointGearHistoryTracker()
        self.raw_pub = rospy.Publisher(
            "/dep_car/policy_candidates_raw",
            PolicyCandidateArray,
            queue_size=1,
        )
        self.state_pub = rospy.Publisher(
            "/dep_car/policy_inference_state", PolicyState, queue_size=1, latch=True
        )
        self.runtime = None
        self.load_error = ""
        try:
            self.runtime = P6PolicyRuntime(
                rospy.get_param("~checkpoint"),
                rospy.get_param("~checkpoint_contract"),
                modality=self.modality,
                device=rospy.get_param("~device", "cuda"),
                mode=self.mode,
                p6_authority=rospy.get_param("~p6_authority", ""),
                fusion_sensor_mode=rospy.get_param("~fusion_sensor_mode", "normal"),
            )
            rospy.loginfo(
                "Loaded P6 %s %s policy %s",
                self.mode,
                self.modality,
                self.runtime.checkpoint_sha256,
            )
        except Exception as exc:
            self.load_error = type(exc).__name__ + ": " + str(exc)
            rospy.logerr("P6 policy unavailable: %s", self.load_error)
            self.publish_state(reason="model_load_failed:" + self.load_error)
        rospy.Subscriber(
            "/camera/depth/image_raw", Image, self.on_depth, queue_size=1, buff_size=2 ** 22
        )
        rospy.Subscriber(
            "/dep_car/lidar/bev", Image, self.on_bev, queue_size=1, buff_size=2 ** 22
        )
        rospy.Subscriber(
            rospy.get_param("~odometry_topic", "/base_pose_ground_truth"),
            Odometry,
            self.on_odom,
            queue_size=1,
        )
        rospy.Subscriber("/imu/data", Imu, self.on_imu, queue_size=1)
        rospy.Subscriber("/urban_model/joint_states", JointState, self.on_joints, queue_size=1)
        rospy.Subscriber("/dep_car/policy_query", PolicyQuery, self.on_query, queue_size=1)
        rate = float(rospy.get_param("~inference_rate", 10.0))
        if rate <= 0.0:
            raise ValueError("inference_rate must be positive")
        self.timer = rospy.Timer(rospy.Duration(1.0 / rate), self.update)

    def on_depth(self, message):
        try:
            if message.encoding != "32FC1":
                raise ValueError("depth encoding must be 32FC1")
            expected = int(message.height) * int(message.width)
            values = np.frombuffer(message.data, dtype=np.float32)
            if values.size != expected:
                raise ValueError("depth byte count does not match image dimensions")
            depth = values.reshape(message.height, message.width).copy()
            self.store("depth", message_stamp(message), normalize_depth_metric(depth))
        except Exception as exc:
            rospy.logwarn_throttle(2.0, "Rejected policy depth frame: %s", exc)

    def on_bev(self, message):
        try:
            if message.encoding != "32FC6" or message.height != 160 or message.width != 160:
                raise ValueError("LiDAR BEV image must be 32FC6 160x160")
            expected = int(message.height) * int(message.width) * 6
            values = np.frombuffer(message.data, dtype=np.float32)
            if values.size != expected:
                raise ValueError("BEV byte count does not match image dimensions")
            chw = values.reshape(message.height, message.width, 6).transpose(2, 0, 1).copy()
            self.store("bev", message_stamp(message), normalize_lidar_bev(chw))
        except Exception as exc:
            rospy.logwarn_throttle(2.0, "Rejected policy LiDAR BEV: %s", exc)

    def on_odom(self, message):
        heading = yaw_from_quaternion(message.pose.pose.orientation)
        velocity = message.twist.twist.linear
        speed = (
            float(velocity.x)
            if self.odometry_twist_in_body_frame
            else world_velocity_to_body_longitudinal(velocity.x, velocity.y, heading)
        )
        self.store("odom", message_stamp(message), np.asarray([speed], dtype=np.float64))

    def on_imu(self, message):
        self.store(
            "imu",
            message_stamp(message),
            np.asarray(
                [message.linear_acceleration.x, message.angular_velocity.z],
                dtype=np.float64,
            ),
        )

    def on_joints(self, message):
        try:
            steering = self.actual_steering(message)
            self.store(
                "joints",
                message_stamp(message),
                np.asarray([steering], dtype=np.float64),
            )
        except Exception as exc:
            rospy.logwarn_throttle(2.0, "Rejected policy joint state: %s", exc)

    def on_query(self, message):
        with self.lock:
            self.query = (message_stamp(message), message)

    def store(self, name, stamp, value):
        with self.lock:
            self.histories[name].append(stamp, value)

    def publish_state(
        self,
        *,
        reason,
        sensor_ready=False,
        inference_ok=False,
        generation=0,
        latency_ms=0.0,
        sensor_skew=0.0,
        candidate_count=0,
    ):
        state = PolicyState()
        state.header.stamp = rospy.Time.now()
        state.mode = self.mode
        state.modality = self.modality
        state.checkpoint_sha256 = self.runtime.checkpoint_sha256 if self.runtime else ""
        state.architecture_id = self.runtime.architecture_id if self.runtime else ""
        state.hybrid_sequence = bool(
            self.runtime is not None and self.runtime.hybrid_sequence
        )
        state.model_loaded = self.runtime is not None
        state.sensor_ready = bool(sensor_ready)
        state.inference_ok = bool(inference_ok)
        state.hard_safety_applied = False
        state.executable = False
        state.control_authorized = bool(
            self.runtime is not None and self.runtime.control_authorized
        )
        state.generation = int(generation)
        state.inference_attempts = int(self.inference_attempts)
        state.synchronization_failures = int(self.synchronization_failures)
        state.inference_latency_ms = float(latency_ms)
        state.sensor_skew_s = float(sensor_skew)
        state.candidate_count = int(candidate_count)
        state.feasible_count = 0
        state.selected_candidate_id = -1
        state.selected_first_gear = 0
        state.selected_action_gears = []
        state.reason = str(reason)
        self.state_pub.publish(state)

    def synchronized_snapshot(self):
        with self.lock:
            histories = {
                name: history.snapshot() for name, history in self.histories.items()
            }
            query_entry = self.query
        required_modalities = self.runtime.modality_mask
        anchor_name = "bev" if required_modalities[1] > 0.5 else "depth"
        missing = [
            name
            for name in (anchor_name, "odom", "imu", "joints")
            if not histories[name]
        ]
        if required_modalities[0] > 0.5 and not histories["depth"]:
            missing.append("depth")
        if query_entry is None:
            missing.append("query")
        if missing:
            raise ValueError("missing:" + "+".join(dict.fromkeys(missing)))
        interpolated_sources = {
            "odom": (histories["odom"], self.odom_tolerance),
            "imu": (histories["imu"], self.imu_tolerance),
            "joints": (histories["joints"], self.joint_tolerance),
        }
        nearest_sources = {}
        if required_modalities[0] > 0.5 and anchor_name != "depth":
            nearest_sources["depth"] = (
                histories["depth"], self.depth_tolerance
            )
        synchronized = newest_synchronized_anchor(
            histories[anchor_name], interpolated_sources, nearest_sources
        )
        if synchronized is None:
            latest_stamp = histories[anchor_name][-1][0]
            unmatched = [
                name
                for name, (entries, tolerance) in interpolated_sources.items()
                if interpolated(entries, latest_stamp, tolerance) is None
            ]
            unmatched.extend(
                name
                for name, (entries, tolerance) in nearest_sources.items()
                if nearest(entries, latest_stamp, tolerance) is None
            )
            evidence = []
            for name in unmatched:
                entries, tolerance = interpolated_sources.get(
                    name, nearest_sources.get(name)
                )
                previous = [stamp for stamp, _ in entries if stamp <= latest_stamp]
                following = [stamp for stamp, _ in entries if stamp >= latest_stamp]
                before = (
                    "none" if not previous else "%.4f" % (latest_stamp - previous[-1])
                )
                after = (
                    "none" if not following else "%.4f" % (following[0] - latest_stamp)
                )
                evidence.append(
                    "%s(before=%s,after=%s,tol=%.4f)"
                    % (name, before, after, tolerance)
                )
            raise ValueError(
                "unsynchronized:" + "+".join(evidence or ["unknown"])
            )
        anchor_stamp, anchor_value, matches = synchronized
        now = rospy.Time.now().to_sec()
        sensor_age = now - anchor_stamp
        query_age = now - query_entry[0]
        if sensor_age < -0.05 or sensor_age > self.maximum_sensor_age:
            raise ValueError("stale_sensor_age=%.6f" % sensor_age)
        if query_age < -0.05 or query_age > self.maximum_sensor_age:
            raise ValueError("stale_query_age=%.6f" % query_age)
        values = {
            name: matched[0] for name, matched in matches.items()
        }
        values[anchor_name] = anchor_value
        values["query"] = query_entry[1]
        skew = max([0.0] + [matched[1] for matched in matches.values()])
        return values, anchor_stamp, skew

    def actual_steering(self, message):
        positions = dict(zip(message.name, message.position))
        names = ("front_left_steer_joint", "front_right_steer_joint")
        if not all(name in positions for name in names):
            raise ValueError("front steering joints are missing")
        return center_steering_from_wheel_angles(
            positions[names[0]], positions[names[1]], self.sim_positive_right
        )

    def update(self, _event):
        if self.runtime is None:
            self.publish_state(reason="model_load_failed:" + self.load_error)
            return
        try:
            self.inference_attempts += 1
            values, anchor_stamp, skew = self.synchronized_snapshot()
            query = values["query"]
            if int(query.generation) == self.last_generation:
                return
            speed = float(values["odom"][0])
            acceleration, yaw_rate = (float(value) for value in values["imu"])
            steering = float(values["joints"][0])
            current_gear, gear_history = self.gear_history.observe(
                anchor_stamp,
                speed,
                recovery_mode=bool(query.recovery_mode),
            )
            if self.runtime.hybrid_sequence:
                state = build_joint_policy_state(
                    signed_speed=speed,
                    longitudinal_acceleration=acceleration,
                    steering=steering,
                    yaw_rate=yaw_rate,
                    subgoal_body=(query.subgoal_body.x, query.subgoal_body.y),
                    heading_error=query.heading_error,
                    reference_curvature=query.reference_curvature,
                )
            else:
                state = build_policy_state(
                    signed_speed=speed,
                    longitudinal_acceleration=acceleration,
                    steering=steering,
                    yaw_rate=yaw_rate,
                    subgoal_body=(query.subgoal_body.x, query.subgoal_body.y),
                    heading_error=query.heading_error,
                    reference_curvature=query.reference_curvature,
                    requested_gear=query.requested_gear,
                    current_gear=current_gear_from_speed(speed),
                )
            depth = values.get("depth", np.zeros((2, 96, 160), dtype=np.float32))
            bev = values.get("bev", np.zeros((6, 160, 160), dtype=np.float32))
            route_pose = np.zeros((80, 3), dtype=np.float32)
            route_mask = np.zeros(80, dtype=bool)
            if bool(query.route_corridor_valid):
                count = min(80, len(query.route_corridor_body))
                if count < 2:
                    raise ValueError("route_corridor_valid requires at least two poses")
                for index, pose in enumerate(query.route_corridor_body[:count]):
                    route_pose[index] = (pose.x, pose.y, pose.theta)
                route_mask[:count] = True
            inference = self.runtime.infer(
                depth,
                bev,
                state,
                int(query.requested_gear),
                route_pose=route_pose,
                route_mask=route_mask,
                current_gear=current_gear,
                gear_history=gear_history,
            )
            if self.runtime.hybrid_sequence:
                trajectories = inference.trajectories
                controls = inference.controls
                scores = inference.scores
            else:
                trajectories, controls, scores = inference
            output = PolicyCandidateArray()
            output.header.stamp = rospy.Time.from_sec(anchor_stamp)
            output.header.frame_id = "chassis"
            output.architecture_id = self.runtime.architecture_id
            output.hybrid_sequence = bool(self.runtime.hybrid_sequence)
            output.actions_per_candidate = 6 if self.runtime.hybrid_sequence else 1
            output.steps_per_action = 5 if self.runtime.hybrid_sequence else 10
            output.requested_gear = int(query.requested_gear)
            output.current_gear = int(current_gear)
            output.gear_history = gear_history.astype(np.float64).tolist()
            output.generation = int(query.generation)
            output.subgoal_body = query.subgoal_body
            output.recovery_mode = bool(query.recovery_mode)
            for candidate_id in range(15):
                trajectory = trajectories[candidate_id]
                candidate = PolicyCandidate()
                candidate.candidate_id = candidate_id
                if self.runtime.hybrid_sequence:
                    first = controls[candidate_id, 0]
                    candidate.gear = int(inference.action_gears[candidate_id, 0])
                    candidate.speed_anchor = float(first[2])
                    candidate.steering_anchor = float(first[1])
                    candidate.duration = float(first[3])
                    candidate.action_gears = inference.action_gears[candidate_id].tolist()
                    candidate.action_mask = inference.action_mask[candidate_id].tolist()
                    candidate.action_durations = controls[candidate_id, :, 3].tolist()
                    candidate.shift_required = inference.shift_required[candidate_id].tolist()
                    candidate.transition_duration = inference.transition_duration[candidate_id].tolist()
                    candidate.motion_gears = inference.motion_gears[candidate_id].tolist()
                else:
                    candidate.gear = int(query.requested_gear)
                    candidate.speed_anchor = float(controls[candidate_id, 2])
                    candidate.steering_anchor = float(controls[candidate_id, 1])
                    candidate.duration = float(controls[candidate_id, 3])
                candidate.learned_score = float(scores[candidate_id])
                candidate.time = trajectory[:, 0].tolist()
                candidate.x = trajectory[:, 1].tolist()
                candidate.y = trajectory[:, 2].tolist()
                candidate.yaw = trajectory[:, 3].tolist()
                candidate.speed = trajectory[:, 4].tolist()
                candidate.steering = trajectory[:, 5].tolist()
                output.candidates.append(candidate)
            self.raw_pub.publish(output)
            self.last_generation = int(query.generation)
            self.publish_state(
                reason="inference_ok",
                sensor_ready=True,
                inference_ok=True,
                generation=query.generation,
                latency_ms=self.runtime.last_latency_ms,
                sensor_skew=skew,
                candidate_count=15,
            )
        except Exception as exc:
            generation = values["query"].generation if "values" in locals() and values.get("query") else 0
            reason = type(exc).__name__ + ":" + str(exc)
            if "unsynchronized:" in reason:
                self.synchronization_failures += 1
            waiting_for_query = (
                reason.startswith("ValueError:missing:") and "query" in reason
            ) or reason.startswith("ValueError:stale_query_age=")
            if waiting_for_query:
                # No query before a goal (or after reaching it) is a normal idle
                # state.  Keep publishing machine-readable state without filling
                # an interactive RViz terminal with false alarms.
                rospy.logdebug_throttle(2.0, "P6 policy waiting: %s", reason)
            else:
                rospy.logwarn_throttle(2.0, "P6 policy inference unavailable: %s", reason)
            self.publish_state(reason=reason, generation=generation)


if __name__ == "__main__":
    rospy.init_node("dep_car_policy")
    PolicyNode()
    rospy.spin()
