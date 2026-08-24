#!/usr/bin/env python3
"""Publish wheel-only Ackermann odometry without simulator pose truth."""

import math

import rospy
from dep_car.core.vehicle import (
    WHEELBASE_M,
    WHEEL_RADIUS_M,
    center_steering_from_wheel_angles,
)
from dep_car.runtime.wheel_odometry import AckermannWheelOdometry
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState


class AckermannWheelOdometryNode:
    def __init__(self):
        self.left_wheel = rospy.get_param("~left_wheel_joint", "rear_left_wheel_joint")
        self.right_wheel = rospy.get_param("~right_wheel_joint", "rear_right_wheel_joint")
        self.left_steer = rospy.get_param("~left_steer_joint", "front_left_steer_joint")
        self.right_steer = rospy.get_param("~right_steer_joint", "front_right_steer_joint")
        self.sim_positive_right = bool(rospy.get_param("~simulator_positive_right", True))
        self.estimator = AckermannWheelOdometry(
            wheel_radius=float(rospy.get_param("~wheel_radius", WHEEL_RADIUS_M)),
            wheelbase=float(rospy.get_param("~wheelbase", WHEELBASE_M)),
            direction=float(rospy.get_param("~wheel_direction", 1.0)),
            speed_deadband_mps=float(
                rospy.get_param("~speed_deadband_mps", 0.015)
            ),
        )
        self.estimator.reset(
            x=float(rospy.get_param("~initial_x", 0.0)),
            y=float(rospy.get_param("~initial_y", 0.0)),
            yaw=float(rospy.get_param("~initial_yaw", 0.0)),
        )
        self.frame_id = str(rospy.get_param("~frame_id", "odom")).lstrip("/")
        self.child_frame_id = str(rospy.get_param("~child_frame_id", "dummy")).lstrip("/")
        self.publisher = rospy.Publisher(
            rospy.get_param("~output_topic", "/dep_car/wheel_odom"),
            Odometry,
            queue_size=20,
        )
        rospy.Subscriber(
            rospy.get_param("~joint_state_topic", "/urban_model/joint_states"),
            JointState,
            self.on_joint_state,
            queue_size=20,
            tcp_nodelay=True,
        )

    @staticmethod
    def named_values(message, field):
        values = getattr(message, field)
        if len(values) != len(message.name):
            return {}
        return dict(zip(message.name, values))

    def on_joint_state(self, message):
        velocities = self.named_values(message, "velocity")
        positions = self.named_values(message, "position")
        required_velocity = (self.left_wheel, self.right_wheel)
        required_position = (self.left_steer, self.right_steer)
        if not all(name in velocities for name in required_velocity) or not all(
            name in positions for name in required_position
        ):
            rospy.logwarn_throttle(
                2.0,
                "Wheel odometry is waiting for rear-wheel velocities and front steering positions",
            )
            return
        stamp = message.header.stamp
        if stamp.is_zero():
            stamp = rospy.Time.now()
        steering = center_steering_from_wheel_angles(
            positions[self.left_steer],
            positions[self.right_steer],
            self.sim_positive_right,
        )
        state = self.estimator.update(
            stamp.to_sec(),
            velocities[self.left_wheel],
            velocities[self.right_wheel],
            steering,
        )
        output = Odometry()
        output.header.stamp = stamp
        output.header.frame_id = self.frame_id
        output.child_frame_id = self.child_frame_id
        output.pose.pose.position.x = state.x
        output.pose.pose.position.y = state.y
        output.pose.pose.orientation.z = math.sin(0.5 * state.yaw)
        output.pose.pose.orientation.w = math.cos(0.5 * state.yaw)
        output.twist.twist.linear.x = state.speed
        output.twist.twist.angular.z = state.yaw_rate
        # Wheel integration constrains planar pose/twist only.  Large unused
        # variances prevent a downstream EKF from interpreting zero z/roll/
        # pitch as measurements.
        output.pose.covariance = [0.0] * 36
        output.twist.covariance = [0.0] * 36
        for index, variance in ((0, 0.03), (7, 0.03), (35, 0.06)):
            output.pose.covariance[index] = variance
        for index, variance in ((0, 0.02), (35, 0.04)):
            output.twist.covariance[index] = variance
        for index in (14, 21, 28):
            output.pose.covariance[index] = 1.0e6
            output.twist.covariance[index] = 1.0e6
        self.publisher.publish(output)


if __name__ == "__main__":
    rospy.init_node("dep_car_wheel_odometry")
    AckermannWheelOdometryNode()
    rospy.spin()
