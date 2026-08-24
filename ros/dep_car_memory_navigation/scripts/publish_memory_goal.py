#!/usr/bin/env python3
"""Publish a reproducible position-only mission goal."""

import argparse
import math

import rospy
import tf2_ros
from geometry_msgs.msg import PoseStamped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--x", type=float, required=True)
    parser.add_argument("--y", type=float, required=True)
    parser.add_argument("--yaw", type=float, default=0.0)
    parser.add_argument(
        "--frame",
        default="map",
        help="coordinate frame of x/y/yaw; output is always transformed to map",
    )
    parser.add_argument("--topic", default="/move_base_simple/goal")
    args = parser.parse_args(rospy.myargv()[1:])
    rospy.init_node("dep_car_publish_memory_goal", anonymous=True)
    x, y, yaw = args.x, args.y, args.yaw
    source_frame = str(args.frame).lstrip("/") or "map"
    if source_frame != "map":
        buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
        listener = tf2_ros.TransformListener(buffer)
        try:
            transform = buffer.lookup_transform(
                "map", source_frame, rospy.Time(0), rospy.Duration(10.0)
            ).transform
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as exc:
            raise RuntimeError("cannot transform fixed goal into online map: %s" % exc)
        quaternion = transform.rotation
        transform_yaw = math.atan2(
            2.0 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y),
            1.0 - 2.0 * (quaternion.y * quaternion.y + quaternion.z * quaternion.z),
        )
        cosine, sine = math.cos(transform_yaw), math.sin(transform_yaw)
        x, y = (
            transform.translation.x + cosine * args.x - sine * args.y,
            transform.translation.y + sine * args.x + cosine * args.y,
        )
        yaw = math.atan2(
            math.sin(transform_yaw + args.yaw), math.cos(transform_yaw + args.yaw)
        )
    publisher = rospy.Publisher(args.topic, PoseStamped, queue_size=1, latch=True)
    deadline = rospy.Time.now() + rospy.Duration(5.0)
    while publisher.get_num_connections() == 0 and rospy.Time.now() < deadline:
        rospy.sleep(0.05)
    message = PoseStamped()
    message.header.stamp = rospy.Time.now()
    message.header.frame_id = "map"
    message.pose.position.x = x
    message.pose.position.y = y
    message.pose.orientation.z = math.sin(0.5 * yaw)
    message.pose.orientation.w = math.cos(0.5 * yaw)
    publisher.publish(message)
    rospy.sleep(0.25)
    print(
        "published memory-navigation goal source=%s (%.3f, %.3f) map=(%.3f, %.3f)"
        % (source_frame, args.x, args.y, x, y)
    )


if __name__ == "__main__":
    main()
