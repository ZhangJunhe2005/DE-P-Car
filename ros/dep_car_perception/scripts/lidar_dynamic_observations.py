#!/usr/bin/env python3
"""Map-difference LiDAR frontend; publishes detections, never simulator GT."""

import math
import threading

import cv2
import numpy as np
import rospy
import tf2_ros
from dep_car.perception.pointcloud import filter_lidar_obstacles
from geometry_msgs.msg import Pose, PoseArray
from nav_msgs.msg import OccupancyGrid
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2


def yaw_from_quaternion(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class LidarDynamicObservations:
    def __init__(self):
        self.lock = threading.Lock(); self.map = None
        self.resolution = rospy.get_param("~cluster_resolution", 0.15)
        self.extent = rospy.get_param("~cluster_extent", 12.0)
        self.static_margin = rospy.get_param("~static_map_margin", 0.25)
        self.minimum_cells = rospy.get_param("~minimum_cluster_cells", 3)
        self.body_frame = rospy.get_param("~body_frame", "chassis")
        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.publisher = rospy.Publisher("/dep_car/dynamic/observations", PoseArray, queue_size=1)
        rospy.Subscriber("/map", OccupancyGrid, self.on_map, queue_size=1)
        rospy.Subscriber("/velodyne_points", PointCloud2, self.on_cloud, queue_size=1, buff_size=2 ** 24)

    def on_map(self, message):
        grid = np.asarray(message.data, dtype=np.int16).reshape(message.info.height, message.info.width)
        radius = max(1, int(math.ceil(self.static_margin / message.info.resolution)))
        occupied = ((grid >= 50) | (grid < 0)).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
        with self.lock:
            self.map = (cv2.dilate(occupied, kernel), message.info.resolution, message.info.origin.position.x, message.info.origin.position.y)

    def on_cloud(self, message):
        with self.lock: map_contract = self.map
        if map_contract is None: return
        points = np.asarray(list(point_cloud2.read_points(message, field_names=("x", "y", "z"), skip_nans=True)), dtype=np.float32)
        if points.size == 0: return
        try:
            transform = self.tf_buffer.lookup_transform(
                self.body_frame, message.header.frame_id, message.header.stamp, rospy.Duration(0.05)
            ).transform
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as exc:
            rospy.logwarn_throttle(2.0, "Skipping dynamic lidar frame without %s transform: %s", self.body_frame, exc)
            return
        q = transform.rotation; x, y, z, w = q.x, q.y, q.z, q.w
        rotation = np.asarray([
            [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
            [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
            [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
        ], dtype=np.float32)
        points = points @ rotation.T + np.asarray(
            [transform.translation.x, transform.translation.y, transform.translation.z], dtype=np.float32
        )
        points = filter_lidar_obstacles(points)
        if not len(points): return
        try:
            world_transform = self.tf_buffer.lookup_transform(
                "map", self.body_frame, message.header.stamp, rospy.Duration(0.05)
            ).transform
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as exc:
            rospy.logwarn_throttle(2.0, "Skipping dynamic projection without map TF: %s", exc)
            return
        grid, map_resolution, origin_x, origin_y = map_contract
        angle = yaw_from_quaternion(world_transform.rotation)
        cosine, sine = math.cos(angle), math.sin(angle)
        world_origin_x = world_transform.translation.x
        world_origin_y = world_transform.translation.y
        local_x = points[:, 0]; local_y = points[:, 1]
        world_x = world_origin_x + cosine * local_x - sine * local_y
        world_y = world_origin_y + sine * local_x + cosine * local_y
        map_x = np.floor((world_x - origin_x) / map_resolution).astype(int); map_y = np.floor((world_y - origin_y) / map_resolution).astype(int)
        inside = (map_x >= 0) & (map_y >= 0) & (map_x < grid.shape[1]) & (map_y < grid.shape[0])
        dynamic = inside & (grid[np.clip(map_y, 0, grid.shape[0]-1), np.clip(map_x, 0, grid.shape[1]-1)] == 0)
        local = np.column_stack((local_x[dynamic], local_y[dynamic]))
        local = local[np.all(np.abs(local) < self.extent, axis=1)]
        size = int(math.ceil(2 * self.extent / self.resolution)); raster = np.zeros((size, size), dtype=np.uint8)
        cells = np.floor((local + self.extent) / self.resolution).astype(int)
        if len(cells): raster[cells[:, 1], cells[:, 0]] = 1
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(raster, connectivity=8)
        output = PoseArray(); output.header = message.header; output.header.frame_id = "map"
        for component in range(1, count):
            if stats[component, cv2.CC_STAT_AREA] < self.minimum_cells: continue
            local_center_x = centroids[component, 0] * self.resolution - self.extent
            local_center_y = centroids[component, 1] * self.resolution - self.extent
            pose = Pose(); pose.position.x = world_origin_x + cosine * local_center_x - sine * local_center_y
            pose.position.y = world_origin_y + sine * local_center_x + cosine * local_center_y; pose.orientation.w = 1.0
            output.poses.append(pose)
        self.publisher.publish(output)


if __name__ == "__main__": rospy.init_node("dep_car_lidar_dynamic_observations"); LidarDynamicObservations(); rospy.spin()
