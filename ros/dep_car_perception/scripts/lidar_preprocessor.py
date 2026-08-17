#!/usr/bin/env python3
"""Publish both learned range-image input and non-learned local occupancy."""

import cv2
import numpy as np
import rospy
import tf2_ros
from dep_car.core.vehicle import DYNAMIC_EGO_RADIUS_M
from dep_car.perception.pointcloud import SelfFilterConfig, lidar_environment_mask
from dep_car.perception.range_image import build_range_image
from nav_msgs.msg import OccupancyGrid
from sensor_msgs import point_cloud2
from sensor_msgs.msg import Image, PointCloud2


class LidarPreprocessor:
    def __init__(self):
        self.bins = rospy.get_param("~azimuth_bins", 440)
        self.channels = rospy.get_param("~channels", 16)
        self.min_range = rospy.get_param("~minimum_range", 0.9)
        self.max_range = rospy.get_param("~maximum_range", 40.0)
        self.map_range = rospy.get_param("~local_map_range", 10.0)
        self.resolution = rospy.get_param("~local_map_resolution", 0.10)
        self.unknown_ego_clearance = rospy.get_param(
            "~unknown_ego_clearance_radius",
            DYNAMIC_EGO_RADIUS_M + 0.5 * np.sqrt(2.0) * self.resolution,
        )
        self.min_height = rospy.get_param("~obstacle_min_height", -0.35)
        self.max_height = rospy.get_param("~obstacle_max_height", 1.30)
        # Occupancy is a measurement authority.  Planning clearance belongs to
        # FootprintConfig and must not also be baked into this grid.
        self.inflation = rospy.get_param("~inflation_radius", 0.0)
        self.body_frame = rospy.get_param("~body_frame", "chassis")
        self.self_filter = SelfFilterConfig(
            length=rospy.get_param("~self_filter_length", SelfFilterConfig.length),
            width=rospy.get_param("~self_filter_width", SelfFilterConfig.width),
            padding=rospy.get_param("~self_filter_padding", SelfFilterConfig.padding),
        )
        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.range_pub = rospy.Publisher("/dep_car/lidar/range_image", Image, queue_size=1)
        self.mask_pub = rospy.Publisher("/dep_car/lidar/validity_mask", Image, queue_size=1)
        self.grid_pub = rospy.Publisher("/dep_car/local_costmap", OccupancyGrid, queue_size=1)
        rospy.Subscriber("/velodyne_points", PointCloud2, self.callback, queue_size=1, buff_size=2 ** 24)

    @staticmethod
    def image_message(array, header):
        message = Image()
        message.header = header
        message.height, message.width = array.shape
        message.encoding = "32FC1"
        message.is_bigendian = False
        message.step = message.width * 4
        message.data = np.asarray(array, dtype=np.float32).tobytes()
        return message

    @staticmethod
    def transform_matrix(transform):
        translation = transform.transform.translation
        quaternion = transform.transform.rotation
        x, y, z, w = quaternion.x, quaternion.y, quaternion.z, quaternion.w
        rotation = np.asarray([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ], dtype=np.float32)
        matrix = np.eye(4, dtype=np.float32)
        matrix[:3, :3] = rotation
        matrix[:3, 3] = (translation.x, translation.y, translation.z)
        return matrix

    def points_in_body(self, points, cloud):
        transform = self.tf_buffer.lookup_transform(
            self.body_frame, cloud.header.frame_id, cloud.header.stamp, rospy.Duration(0.05)
        )
        homogeneous = np.column_stack((points, np.ones(len(points), dtype=np.float32)))
        return (self.transform_matrix(transform) @ homogeneous.T).T[:, :3]

    def occupancy_message(self, points, header):
        size = int(np.ceil(2.0 * self.map_range / self.resolution))
        grid = np.full((size, size), -1, dtype=np.int8)
        center = size // 2
        # Every valid non-self return proves free space before its endpoint,
        # including ground/ceiling returns.  Only endpoints in the chassis
        # obstacle-height band become occupied.
        ray_cells = np.floor(points[:, :2] / self.resolution).astype(int) + center
        ray_cells = ray_cells[np.all((ray_cells >= 0) & (ray_cells < size), axis=1)]
        for cell in ray_cells:
            cv2.line(grid, (center, center), (int(cell[0]), int(cell[1])), 0, 1)
        obstacles = points[(points[:, 2] >= self.min_height) & (points[:, 2] <= self.max_height)]
        cells = np.floor(obstacles[:, :2] / self.resolution).astype(int) + center
        cells = cells[np.all((cells >= 0) & (cells < size), axis=1)]
        if len(cells):
            grid[cells[:, 1], cells[:, 0]] = 100
        radius = int(np.ceil(self.inflation / self.resolution))
        if radius > 0:
            occupied = (grid == 100).astype(np.uint8)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
            grid[cv2.dilate(occupied, kernel) > 0] = 100
        # The lidar cannot observe through the vehicle body.  Explicitly clear
        # only the physical ego box (not the planning safety margin), matching
        # standard rolling-costmap footprint clearing.
        coordinates = (np.arange(size, dtype=np.float32) + 0.5 - center) * self.resolution
        xx, yy = np.meshgrid(coordinates, coordinates)
        unknown_under_ego = (xx * xx + yy * yy <= self.unknown_ego_clearance ** 2) & (grid < 0)
        grid[unknown_under_ego] = 0
        ego_x = np.abs(coordinates) <= (0.5 * self.self_filter.length + self.self_filter.padding)
        ego_y = np.abs(coordinates) <= (0.5 * self.self_filter.width + self.self_filter.padding)
        grid[np.ix_(ego_y, ego_x)] = 0
        message = OccupancyGrid()
        message.header = header
        message.header.frame_id = self.body_frame
        message.info.resolution = self.resolution
        message.info.width = size
        message.info.height = size
        message.info.origin.position.x = -self.map_range
        message.info.origin.position.y = -self.map_range
        message.info.origin.orientation.w = 1.0
        message.data = grid.flatten().tolist()
        return message

    def callback(self, cloud):
        points = np.asarray(list(point_cloud2.read_points(cloud, field_names=("x", "y", "z"), skip_nans=True)), dtype=np.float32)
        if points.size == 0:
            points = np.empty((0, 3), dtype=np.float32)
        try:
            body_points = self.points_in_body(points, cloud)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as exc:
            rospy.logwarn_throttle(2.0, "Skipping lidar frame without %s transform: %s", self.body_frame, exc)
            return
        environment = lidar_environment_mask(body_points, self.self_filter)
        sensor_points = points[environment]
        body_points = body_points[environment]
        normalized, mask = build_range_image(sensor_points, self.bins, self.channels, self.min_range, self.max_range)
        self.range_pub.publish(self.image_message(normalized, cloud.header))
        self.mask_pub.publish(self.image_message(mask, cloud.header))
        self.grid_pub.publish(self.occupancy_message(body_points, cloud.header))


if __name__ == "__main__":
    rospy.init_node("dep_car_lidar")
    LidarPreprocessor()
    rospy.spin()
