#!/usr/bin/env python3
"""Translate safe speed/steering commands to Urban Car ros_control topics."""

import math
import threading

import rospy
from dep_car.core.vehicle import EFFORT_SCALE, FRONT_TRACK_M, STEERING_OPERATING_LIMIT_RAD, WHEELBASE_M, world_velocity_to_body_longitudinal
from dep_car.core.types import Gear
from dep_car_msgs.msg import AckermannCommand
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


class UrbanCarAdapter:
    def __init__(self):
        self.wheelbase = rospy.get_param("~wheelbase", WHEELBASE_M)
        self.front_track = rospy.get_param("~front_track", FRONT_TRACK_M)
        self.steering_limit = rospy.get_param("~steering_limit", STEERING_OPERATING_LIMIT_RAD)
        self.effort_limit = rospy.get_param("~effort_limit", 120.0 * EFFORT_SCALE)
        self.kp = rospy.get_param("~speed_kp", 65.0 * EFFORT_SCALE)
        self.ki = rospy.get_param("~speed_ki", 8.0 * EFFORT_SCALE)
        self.active_stop_deadband = float(
            rospy.get_param("~active_stop_deadband", 0.02)
        )
        self.active_stop_minimum_effort_fraction = float(
            rospy.get_param("~active_stop_minimum_effort_fraction", 0.10)
        )
        self.brake_deadband = float(rospy.get_param("~brake_deadband", 0.005))
        self.brake_minimum_effort_fraction = float(
            rospy.get_param("~brake_minimum_effort_fraction", 0.10)
        )
        self.timeout = rospy.get_param("~command_timeout", 0.35)
        self.sim_positive_right = rospy.get_param("~simulator_positive_right", True)
        self.lock = threading.Lock()
        self.command = AckermannCommand(brake=True)
        self.last_command = rospy.Time(0)
        self.speed = 0.0
        self.integral = 0.0
        self.last_update = rospy.Time.now()
        self.steering_pub = rospy.Publisher("/urban_model/steer_controller/command", JointTrajectory, queue_size=1)
        self.left_pub = rospy.Publisher("/urban_model/left_motor_controller/command", Float64, queue_size=1)
        self.right_pub = rospy.Publisher("/urban_model/right_motor_controller/command", Float64, queue_size=1)
        rospy.Subscriber(rospy.get_param("~command_topic", "/dep_car/cmd_ackermann"), AckermannCommand, self.on_command, queue_size=1)
        rospy.Subscriber(rospy.get_param("~odometry_topic", "/base_pose_ground_truth"), Odometry, self.on_odometry, queue_size=1)
        self.timer = rospy.Timer(rospy.Duration(0.02), self.update)

    def on_command(self, message):
        with self.lock:
            self.command = message
            self.last_command = rospy.Time.now()

    def on_odometry(self, message):
        # gazebo_ros_p3d publishes world-axis velocity components even though
        # this Odometry message names the vehicle child frame.  Project onto
        # the vehicle heading before closing the longitudinal PI loop.
        q = message.pose.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        velocity = message.twist.twist.linear
        self.speed = world_velocity_to_body_longitudinal(velocity.x, velocity.y, yaw)

    def ackermann_angles(self, rep103_center_angle):
        center = max(-self.steering_limit, min(self.steering_limit, rep103_center_angle))
        simulator_center = -center if self.sim_positive_right else center
        if abs(simulator_center) < 1e-6:
            return 0.0, 0.0
        radius = self.wheelbase / math.tan(simulator_center)
        # The center limit is already derived from the inner wheel's URDF hard
        # stop; this guard only prevents a singular/negative inner radius.
        minimum = 0.5 * self.front_track + 1e-6
        radius = max(radius, minimum) if radius > 0 else min(radius, -minimum)
        left = math.atan(self.wheelbase / (radius + 0.5 * self.front_track))
        right = math.atan(self.wheelbase / (radius - 0.5 * self.front_track))
        return left, right

    def update(self, _event):
        now = rospy.Time.now()
        with self.lock:
            command = self.command
            stale = (now - self.last_command).to_sec() > self.timeout
        valid_gear = command.gear in (int(Gear.REVERSE), int(Gear.FORWARD))
        valid_sign = (command.gear == int(Gear.FORWARD) and command.speed >= 0.0) or (
            command.gear == int(Gear.REVERSE) and command.speed <= 0.0
        )
        brake = stale or command.brake or not valid_gear or not valid_sign
        active_stop = (
            not brake and valid_gear and abs(float(command.speed)) <= 1.0e-6
        )
        if not stale and not command.brake and (not valid_gear or not valid_sign):
            rospy.logerr_throttle(2.0, "Rejected Ackermann command with inconsistent speed/gear contract")
        target_speed = 0.0 if brake else command.speed
        dt = max(1e-3, min(0.1, (now - self.last_update).to_sec()))
        self.last_update = now
        error = target_speed - self.speed
        if brake:
            self.integral = 0.0
            # ``brake`` is a zero-speed hold, not neutral coasting.  Keep a
            # bounded opposing wheel effort until the chassis is genuinely
            # stationary; otherwise the goal latch can drift back outside its
            # position tolerance after the planner has declared success.
            if abs(self.speed) <= self.brake_deadband:
                effort = 0.0
            else:
                magnitude = max(
                    self.kp * abs(self.speed),
                    self.effort_limit * self.brake_minimum_effort_fraction,
                )
                effort = -math.copysign(
                    min(self.effort_limit, magnitude), self.speed
                )
        elif active_stop:
            # A zero-speed command in a drive gear is an active stop, not a
            # neutral/coast command.  Discard the previous drive integral so
            # it cannot keep propelling the car through a shift or goal.  A
            # small bounded effort floor avoids an excessively long tail near
            # zero speed; it is disabled inside the deadband.
            self.integral = 0.0
            if abs(self.speed) <= self.active_stop_deadband:
                effort = 0.0
            else:
                magnitude = max(
                    self.kp * abs(self.speed),
                    self.effort_limit * self.active_stop_minimum_effort_fraction,
                )
                effort = -math.copysign(min(self.effort_limit, magnitude), self.speed)
        else:
            # Do not let integral accumulated while accelerating oppose a
            # newly requested deceleration (or vice versa).  This is
            # especially important for the terminal speed envelope, whose
            # target decreases continuously near the goal.
            if error * self.integral < 0.0:
                self.integral = 0.0
            self.integral = max(-2.0, min(2.0, self.integral + error * dt))
            effort = max(-self.effort_limit, min(self.effort_limit, self.kp * error + self.ki * self.integral))
        left, right = self.ackermann_angles(0.0 if brake else command.steering_angle)
        trajectory = JointTrajectory()
        trajectory.header.stamp = now
        trajectory.joint_names = ["front_left_steer_joint", "front_right_steer_joint"]
        point = JointTrajectoryPoint()
        point.positions = [left, right]
        point.time_from_start = rospy.Duration(0.05)
        trajectory.points = [point]
        self.steering_pub.publish(trajectory)
        self.left_pub.publish(Float64(data=effort))
        self.right_pub.publish(Float64(data=effort))


if __name__ == "__main__":
    rospy.init_node("urban_car_adapter")
    UrbanCarAdapter()
    rospy.spin()
