#!/usr/bin/env python3
"""Capture strictly synchronized StaticAckermannSampleV2 snapshots."""

import json
import math
import threading
import uuid
from pathlib import Path

import message_filters
import numpy as np
import rospy
import tf2_ros
from dep_car.core.types import Candidate as CoreCandidate
from dep_car.core.types import Gear
from dep_car.core.vehicle import center_steering_from_wheel_angles, world_velocity_to_body_longitudinal
from dep_car.perception.bev import LidarBEVConfig, build_lidar_bev, lidar_bev_preprocessing_contract
from dep_car.perception.pointcloud import filter_lidar_obstacles
from dep_car.training.dataset import save_multimodal_sample
from dep_car_msgs.msg import AckermannRoute, CandidateArray, LocalRouteCommand
from nav_msgs.msg import Odometry
from sensor_msgs import point_cloud2
from sensor_msgs.msg import Image, Imu, JointState, PointCloud2
from std_srvs.srv import Trigger, TriggerResponse


def yaw_from_quaternion(quaternion):
    return math.atan2(
        2.0 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y),
        1.0 - 2.0 * (quaternion.y * quaternion.y + quaternion.z * quaternion.z),
    )


def transform_matrix(transform):
    q = transform.rotation
    x, y, z, w = q.x, q.y, q.z, q.w
    rotation = np.asarray([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float32)
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = [transform.translation.x, transform.translation.y, transform.translation.z]
    return matrix


def decode_depth(message, minimum=0.2, maximum=10.0):
    if message.encoding != "32FC1":
        raise ValueError("depth input must use 32FC1 metric encoding")
    depth = np.frombuffer(message.data, dtype=np.float32).reshape(message.height, message.width).copy()
    validity = np.isfinite(depth) & (depth >= minimum) & (depth <= maximum)
    depth[~validity] = maximum
    return depth, validity.astype(np.uint8)


def decode_cloud(message):
    available = {field.name for field in message.fields}
    selected = [name for name in ("x", "y", "z", "intensity", "ring") if name in available]
    if selected[:3] != ["x", "y", "z"]:
        raise ValueError("point cloud does not contain x/y/z")
    values = np.asarray(list(point_cloud2.read_points(message, field_names=selected, skip_nans=True)), dtype=np.float32)
    if values.size == 0:
        return np.empty((0, 5), dtype=np.float32)
    values = values.reshape((-1, len(selected)))
    output = np.zeros((len(values), 5), dtype=np.float32)
    output[:, :3] = values[:, :3]
    for source, target in (("intensity", 3), ("ring", 4)):
        if source in selected:
            output[:, target] = values[:, selected.index(source)]
    return output


def candidates_from_message(message):
    candidates = []
    for item in message.candidates:
        count = len(item.path.poses)
        trajectory = np.zeros((count, 6), dtype=np.float64)
        trajectory[:, 0] = np.linspace(0.0, item.duration, count)
        for index, pose in enumerate(item.path.poses):
            trajectory[index, 1] = pose.pose.position.x
            trajectory[index, 2] = pose.pose.position.y
            trajectory[index, 3] = yaw_from_quaternion(pose.pose.orientation)
            trajectory[index, 4] = item.speed_anchor
            trajectory[index, 5] = item.steering_anchor
        candidates.append(CoreCandidate(
            item.candidate_id,
            item.speed_anchor,
            item.steering_anchor,
            item.duration,
            trajectory,
            gear=Gear(int(item.gear)),
            retime_factor=item.retime_factor,
            learned_score=item.learned_score,
            guidance_cost=item.guidance_cost,
            static_clearance=item.static_clearance,
            dynamic_clearance=item.dynamic_clearance,
            feasible=item.feasible,
            veto_reason=item.veto_reason,
        ))
    return candidates


def imu_vector(message):
    return np.asarray([
        message.orientation.x, message.orientation.y, message.orientation.z, message.orientation.w,
        message.angular_velocity.x, message.angular_velocity.y, message.angular_velocity.z,
        message.linear_acceleration.x, message.linear_acceleration.y, message.linear_acceleration.z,
    ], dtype=np.float32)


def transform_contract(matrix, stamp, source_frame, target_frame):
    return {
        "matrix": matrix.tolist(), "measurement_stamp": stamp,
        "source_frame": source_frame.lstrip("/"), "target_frame": target_frame.lstrip("/"),
    }


class MultimodalCollector:
    def __init__(self):
        self.output = Path(rospy.get_param("~output", "data/static_multimodal_v2"))
        self.map_uuid = rospy.get_param("~map_uuid")
        self.map_hash = rospy.get_param("~map_hash", "")
        self.simulator_seed = rospy.get_param("~simulator_seed", -1)
        self.body_frame = rospy.get_param("~body_frame", "chassis")
        self.maximum_depth_lidar_skew = rospy.get_param("~maximum_depth_lidar_skew", 0.05)
        self.maximum_odom_skew = rospy.get_param("~maximum_odom_skew", 0.02)
        self.maximum_imu_skew = rospy.get_param("~maximum_imu_skew", 0.02)
        self.maximum_joint_skew = rospy.get_param("~maximum_joint_skew", 0.05)
        self.maximum_candidate_skew = rospy.get_param("~maximum_candidate_skew", 0.15)
        self.sim_positive_right = rospy.get_param("~simulator_positive_right", True)
        self.lock = threading.Lock()
        self.route = self.route_command = self.candidates = self.snapshot = None
        self.previous_speed = self.previous_speed_stamp = None
        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        rospy.Subscriber("/dep_car/global_route", AckermannRoute, self.on_route, queue_size=1)
        rospy.Subscriber("/dep_car/local_route_command", LocalRouteCommand, self.on_route_command, queue_size=1)
        rospy.Subscriber("/dep_car/candidates", CandidateArray, self.on_candidates, queue_size=1)
        subscribers = (
            message_filters.Subscriber("/camera/depth/image_raw", Image),
            message_filters.Subscriber("/velodyne_points", PointCloud2),
            message_filters.Subscriber("/base_pose_ground_truth", Odometry),
            message_filters.Subscriber("/imu/data", Imu),
            message_filters.Subscriber("/urban_model/joint_states", JointState),
        )
        synchronizer = message_filters.ApproximateTimeSynchronizer(
            subscribers, queue_size=50, slop=self.maximum_depth_lidar_skew, allow_headerless=False
        )
        synchronizer.registerCallback(self.on_synchronized)
        self.synchronizer = synchronizer
        rospy.Service("~capture", Trigger, self.capture)

    def on_route(self, message):
        with self.lock: self.route = message

    def on_route_command(self, message):
        with self.lock: self.route_command = message

    def on_candidates(self, message):
        with self.lock: self.candidates = message

    @staticmethod
    def _stamp(message):
        return message.header.stamp.to_sec()

    def _actual_steering(self, joint_state):
        positions = dict(zip(joint_state.name, joint_state.position))
        names = ("front_left_steer_joint", "front_right_steer_joint")
        if not all(name in positions for name in names):
            raise ValueError("front steering joints are missing")
        return center_steering_from_wheel_angles(positions[names[0]], positions[names[1]], self.sim_positive_right)

    @staticmethod
    def _body_xy(world_x, world_y, odom):
        heading = yaw_from_quaternion(odom.pose.pose.orientation)
        dx = world_x - odom.pose.pose.position.x
        dy = world_y - odom.pose.pose.position.y
        return math.cos(heading) * dx + math.sin(heading) * dy, -math.sin(heading) * dx + math.cos(heading) * dy

    @staticmethod
    def _signed_body_speed(odom):
        heading = yaw_from_quaternion(odom.pose.pose.orientation)
        velocity = odom.twist.twist.linear
        return world_velocity_to_body_longitudinal(velocity.x, velocity.y, heading)

    def _local_route(self, route, odom, maximum_points=80):
        rows, gears = [], []
        heading = yaw_from_quaternion(odom.pose.pose.orientation)
        for point in route.points[:maximum_points]:
            x, y = self._body_xy(point.pose.position.x, point.pose.position.y, odom)
            route_yaw = yaw_from_quaternion(point.pose.orientation)
            relative_yaw = math.atan2(math.sin(route_yaw - heading), math.cos(route_yaw - heading))
            rows.append((x, y, relative_yaw)); gears.append(point.gear)
        return np.asarray(rows, dtype=np.float32), np.asarray(gears, dtype=np.int8)

    @staticmethod
    def _reference_curvature(path):
        if len(path) < 3:
            return 0.0
        delta = path[2, :2] - path[0, :2]
        distance = float(np.linalg.norm(delta))
        return 0.0 if distance < 1e-4 else float(path[2, 2] - path[0, 2]) / distance

    def on_synchronized(self, depth_message, cloud_message, odom, imu, joint_state):
        lidar_stamp = self._stamp(cloud_message)
        stamps = {
            "lidar": lidar_stamp,
            "depth": self._stamp(depth_message),
            "odom": self._stamp(odom),
            "imu": self._stamp(imu),
            "joint_state": self._stamp(joint_state),
        }
        if abs(stamps["depth"] - lidar_stamp) > self.maximum_depth_lidar_skew:
            return
        if abs(stamps["odom"] - lidar_stamp) > self.maximum_odom_skew:
            return
        if abs(stamps["imu"] - lidar_stamp) > self.maximum_imu_skew:
            return
        if abs(stamps["joint_state"] - lidar_stamp) > self.maximum_joint_skew:
            return
        with self.lock:
            route, route_command, candidate_message = self.route, self.route_command, self.candidates
        if route is None or route_command is None or candidate_message is None:
            return
        stamps["candidates"] = self._stamp(candidate_message)
        if abs(stamps["candidates"] - lidar_stamp) > self.maximum_candidate_skew:
            return
        try:
            transform = self.tf_buffer.lookup_transform(
                self.body_frame, cloud_message.header.frame_id, cloud_message.header.stamp, rospy.Duration(0.05)
            )
            matrix = transform_matrix(transform.transform)
            camera_transform = self.tf_buffer.lookup_transform(
                self.body_frame, depth_message.header.frame_id, cloud_message.header.stamp, rospy.Duration(0.05)
            )
            map_transform = self.tf_buffer.lookup_transform(
                "map", self.body_frame, cloud_message.header.stamp, rospy.Duration(0.05)
            )
            camera_matrix = transform_matrix(camera_transform.transform)
            map_matrix = transform_matrix(map_transform.transform)
            lidar_points = decode_cloud(cloud_message)
            if len(lidar_points):
                homogeneous = np.column_stack((lidar_points[:, :3], np.ones(len(lidar_points), dtype=np.float32)))
                lidar_points[:, :3] = (matrix @ homogeneous.T).T[:, :3]
            depth, depth_validity = decode_depth(depth_message)
            environment_points = filter_lidar_obstacles(lidar_points)
            lidar_bev = build_lidar_bev(environment_points, LidarBEVConfig())
            local_path, local_gears = self._local_route(route, odom)
            if not len(local_path):
                return
            gx, gy = self._body_xy(route_command.target.position.x, route_command.target.position.y, odom)
            vehicle_yaw = yaw_from_quaternion(odom.pose.pose.orientation)
            route_yaw = yaw_from_quaternion(route_command.target.orientation)
            heading_error = math.atan2(math.sin(route_yaw - vehicle_yaw), math.cos(route_yaw - vehicle_yaw))
            speed = self._signed_body_speed(odom)
            self.previous_speed, self.previous_speed_stamp = speed, lidar_stamp
            steering = self._actual_steering(joint_state)
            state = np.asarray([
                speed, imu.linear_acceleration.x, steering, imu.angular_velocity.z,
                gx, gy, math.sin(heading_error), math.cos(heading_error), self._reference_curvature(local_path),
            ], dtype=np.float32)
            current_gear = Gear.NEUTRAL if abs(speed) < 0.03 else (Gear.FORWARD if speed > 0.0 else Gear.REVERSE)
            snapshot = {
                "depth_metric": depth,
                "depth_validity": depth_validity,
                "lidar_points": lidar_points,
                "lidar_bev": lidar_bev,
                "imu_measurement": imu_vector(imu),
                "vehicle_state": state,
                "current_gear": current_gear,
                "requested_gear": Gear(int(route_command.requested_gear)),
                "local_path": local_path,
                "local_path_gears": local_gears,
                "subgoal_body": (gx, gy),
                "candidates": candidates_from_message(candidate_message),
                "timestamps": stamps,
                "transforms": {
                    "lidar_to_chassis": transform_contract(matrix, lidar_stamp, cloud_message.header.frame_id, self.body_frame),
                    "camera_to_chassis": transform_contract(camera_matrix, lidar_stamp, depth_message.header.frame_id, self.body_frame),
                    "chassis_to_map": transform_contract(map_matrix, lidar_stamp, self.body_frame, "map"),
                },
                "raw_authority": {
                    "kind": "embedded_diagnostic",
                    "messages": {
                        name: {"topic": topic, "message_index": -1, "timestamp": stamps[name]}
                        for name, topic in (
                            ("lidar", "/velodyne_points"), ("depth", "/camera/depth/image_raw"),
                            ("imu", "/imu/data"), ("odom", "/base_pose_ground_truth"),
                            ("joint_state", "/urban_model/joint_states"),
                            ("candidates", "/dep_car/candidates"),
                        )
                    },
                },
                "preprocessing": {"lidar_bev": lidar_bev_preprocessing_contract(LidarBEVConfig())},
                "interpolation": {
                    name: {"method": "approximate_time_nearest", "target_stamp": lidar_stamp, "source_stamps": [stamps[name]]}
                    for name in ("odom", "joint_state", "imu")
                },
                "lidar_timing": {
                    "model": "gazebo_ray_instantaneous_snapshot", "scan_period_s": 0.10,
                    "scan_start_stamp": lidar_stamp, "scan_end_stamp": lidar_stamp,
                    "per_point_time_field_present": True, "per_point_time_available": False,
                    "per_point_time_observed_range_s": [0.0, 0.0], "deskew_applied": False,
                    "deskew_reason": "Gazebo cloud time field is present but every observed value is zero",
                },
                "maneuver_mode": "GEAR_SHIFT" if current_gear != Gear(int(route_command.requested_gear)) else ("REVERSE_EXIT" if int(route_command.requested_gear) == -1 else "NORMAL"),
            }
        except Exception as exc:
            rospy.logwarn_throttle(2.0, "Rejected synchronized multimodal snapshot: %s", exc)
            return
        with self.lock: self.snapshot = snapshot

    def capture(self, _request):
        with self.lock:
            snapshot = self.snapshot
        if snapshot is None:
            return TriggerResponse(False, "waiting for a valid synchronized multimodal snapshot")
        sample_id = str(uuid.uuid4())
        path = self.output / self.map_uuid / (sample_id + ".npz")
        metadata = {
            "sample_id": sample_id,
            "map_occupancy_sha256": self.map_hash,
            "simulator_seed": self.simulator_seed,
            "bev_contract": "LidarBEVPreprocessingV1",
            "lidar_self_filter": "physical_chassis_box_plus_0.03m",
            "lidar_points_include_self_returns": True,
            "formal_training_authority": False,
        }
        manifest = save_multimodal_sample(path, map_uuid=self.map_uuid, metadata=metadata, **snapshot)
        return TriggerResponse(True, "saved %s split=%s" % (path, manifest["split"]))


if __name__ == "__main__":
    rospy.init_node("dep_car_multimodal_collector")
    MultimodalCollector()
    rospy.spin()
