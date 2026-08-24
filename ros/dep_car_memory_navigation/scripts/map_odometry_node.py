#!/usr/bin/env python3
"""Express filtered odometry pose in the current SLAM map frame."""

import json
import math

import rospy
import tf2_ros
from dep_car.runtime.wheel_odometry import (
    PlanarTransformRevisionTracker,
    compose_planar_pose,
)
from nav_msgs.msg import Odometry
from std_msgs.msg import String


def yaw_from_quaternion(quaternion):
    return math.atan2(
        2.0 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y),
        1.0 - 2.0 * (quaternion.y * quaternion.y + quaternion.z * quaternion.z),
    )


class MapOdometryNode:
    def __init__(self):
        self.map_frame = str(rospy.get_param("~map_frame", "map")).lstrip("/")
        self.odom_frame = str(rospy.get_param("~odom_frame", "odom")).lstrip("/")
        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(30.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.publisher = rospy.Publisher(
            rospy.get_param("~output_topic", "/dep_car/map_odometry"),
            Odometry,
            queue_size=10,
        )
        self.correction_publisher = rospy.Publisher(
            "/dep_car/map_odom_correction", String, queue_size=10
        )
        self.maximum_transform_skew = float(
            rospy.get_param("~maximum_transform_skew_s", 0.15)
        )
        if self.maximum_transform_skew <= 0.0:
            raise ValueError("maximum_transform_skew_s must be positive")
        self.transform_revisions = PlanarTransformRevisionTracker()
        rospy.Subscriber(
            rospy.get_param("~input_topic", "/odometry/filtered"),
            Odometry,
            self.on_odometry,
            queue_size=10,
            tcp_nodelay=True,
        )

    def on_odometry(self, message):
        try:
            # Use one coherent, most-recent rigid SLAM correction for all EKF
            # messages until SLAM publishes the next correction.  Mixing an
            # interpolated historical transform on one callback with a latest
            # fallback on the next made map-frame odometry alternate between
            # two poses at the 100 Hz EKF rate.
            transform_stamped = self.tf_buffer.lookup_transform(
                self.map_frame,
                self.odom_frame,
                rospy.Time(0),
                rospy.Duration(0.05),
            )
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as exc:
            rospy.logwarn_throttle(2.0, "Map odometry is waiting for SLAM TF: %s", exc)
            return
        skew = (
            0.0
            if message.header.stamp == rospy.Time()
            or transform_stamped.header.stamp == rospy.Time()
            else abs(
                (message.header.stamp - transform_stamped.header.stamp).to_sec()
            )
        )
        if skew > self.maximum_transform_skew:
            rospy.logwarn_throttle(
                2.0,
                "Map odometry rejected stale SLAM TF skew=%.3fs contract=%.3fs",
                skew,
                self.maximum_transform_skew,
            )
            return
        transform = transform_stamped.transform
        parent = (
            transform.translation.x,
            transform.translation.y,
            yaw_from_quaternion(transform.rotation),
        )
        transform_stamp = (
            transform_stamped.header.stamp.secs,
            transform_stamped.header.stamp.nsecs,
        )
        revision = self.transform_revisions.observe(transform_stamp, parent)
        # /dep_car/map_odometry remains a high-rate current-pose stream, but a
        # correction event exists only when SLAM actually publishes a newly
        # stamped map->odom transform.  Re-publishing the same latest TF on
        # every 100 Hz EKF callback caused false route-revalidation churn.
        if revision is not None:
            correction = {
                "schema": "DEPCarMapOdomCorrectionV1",
                "stamp": message.header.stamp.to_sec(),
                "translation_delta_m": revision.translation_delta,
                "yaw_delta_rad": revision.yaw_delta,
                "map_to_odom": list(parent),
                "transform_stamp": transform_stamped.header.stamp.to_sec(),
                "transform_skew_s": skew,
                "within_skew_contract": True,
                "time_aligned": skew <= 0.02,
            }
            self.correction_publisher.publish(
                String(data=json.dumps(correction, sort_keys=True))
            )
            if (
                revision.translation_delta >= 0.12
                or abs(revision.yaw_delta) >= 0.06
            ):
                rospy.logwarn_throttle(
                    1.0,
                    "SLAM map->odom correction: translation=%.3fm yaw=%.3frad; "
                    "odom history remains internally rigid and committed map routes "
                    "will be revalidated",
                    revision.translation_delta,
                    revision.yaw_delta,
                )
        child = (
            message.pose.pose.position.x,
            message.pose.pose.position.y,
            yaw_from_quaternion(message.pose.pose.orientation),
        )
        x, y, yaw = compose_planar_pose(parent, child)
        output = Odometry()
        output.header.stamp = message.header.stamp
        output.header.frame_id = self.map_frame
        output.child_frame_id = message.child_frame_id or "dummy"
        output.pose = message.pose
        output.pose.pose.position.x = x
        output.pose.pose.position.y = y
        output.pose.pose.orientation.x = 0.0
        output.pose.pose.orientation.y = 0.0
        output.pose.pose.orientation.z = math.sin(0.5 * yaw)
        output.pose.pose.orientation.w = math.cos(0.5 * yaw)
        # Twist is already expressed in child_frame_id by robot_localization.
        output.twist = message.twist
        self.publisher.publish(output)


if __name__ == "__main__":
    rospy.init_node("dep_car_map_odometry")
    MapOdometryNode()
    rospy.spin()
