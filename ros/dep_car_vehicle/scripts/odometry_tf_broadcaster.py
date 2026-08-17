#!/usr/bin/env python3
"""Broadcast the simulated map-to-vehicle transform from Gazebo odometry."""

import rospy
import tf2_ros
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry


class OdometryTFBroadcaster:
    def __init__(self):
        self.parent_override = rospy.get_param("~parent_frame", "")
        self.child_override = rospy.get_param("~child_frame", "")
        self.broadcaster = tf2_ros.TransformBroadcaster()
        topic = rospy.get_param("~odometry_topic", "/base_pose_ground_truth")
        rospy.Subscriber(topic, Odometry, self.on_odometry, queue_size=1, tcp_nodelay=True)

    @staticmethod
    def normalize(frame):
        return frame.lstrip("/")

    def on_odometry(self, message):
        if rospy.is_shutdown():
            return
        parent = self.normalize(self.parent_override or message.header.frame_id)
        child = self.normalize(self.child_override or message.child_frame_id)
        if not parent or not child or parent == child:
            rospy.logwarn_throttle(5.0, "Cannot publish odometry TF with invalid frames %r -> %r", parent, child)
            return
        transform = TransformStamped()
        transform.header.stamp = message.header.stamp or rospy.Time.now()
        transform.header.frame_id = parent
        transform.child_frame_id = child
        transform.transform.translation.x = message.pose.pose.position.x
        transform.transform.translation.y = message.pose.pose.position.y
        transform.transform.translation.z = message.pose.pose.position.z
        transform.transform.rotation = message.pose.pose.orientation
        try:
            self.broadcaster.sendTransform(transform)
        except rospy.ROSException:
            if not rospy.is_shutdown():
                raise


if __name__ == "__main__":
    rospy.init_node("dep_car_odometry_tf")
    OdometryTFBroadcaster()
    rospy.spin()
