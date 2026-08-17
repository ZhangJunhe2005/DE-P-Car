#!/usr/bin/env python3
"""Measure steady-state Urban Car curvature from Gazebo odometry."""

import argparse
import json
import math
import threading
from pathlib import Path

import numpy as np
import rospy
from dep_car.core.vehicle import center_steering_from_wheel_angles, world_velocity_to_body_longitudinal
from dep_car_msgs.msg import AckermannCommand
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState


class TurningCalibration:
    def __init__(self):
        self.lock = threading.Lock()
        self.poses, self.speeds, self.steering = [], [], []
        self.stamps = []
        self.recording = False
        self.publisher = rospy.Publisher("/dep_car/cmd_ackermann", AckermannCommand, queue_size=1)
        rospy.Subscriber("/base_pose_ground_truth", Odometry, self.on_odom, queue_size=100)
        rospy.Subscriber("/urban_model/joint_states", JointState, self.on_joints, queue_size=100)

    def on_odom(self, message):
        if not self.recording:
            return
        with self.lock:
            self.poses.append((message.pose.pose.position.x, message.pose.pose.position.y))
            self.stamps.append(message.header.stamp.to_sec())
            q = message.pose.pose.orientation
            yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
            velocity = message.twist.twist.linear
            self.speeds.append(world_velocity_to_body_longitudinal(velocity.x, velocity.y, yaw))

    def on_joints(self, message):
        if not self.recording:
            return
        positions = dict(zip(message.name, message.position))
        names = ("front_left_steer_joint", "front_right_steer_joint")
        if all(name in positions for name in names):
            self.steering.append(center_steering_from_wheel_angles(positions[names[0]], positions[names[1]]))

    def command(self, speed, steering, brake=False):
        message = AckermannCommand()
        message.header.stamp = rospy.Time.now()
        message.speed = speed
        message.steering_angle = steering
        message.acceleration = 0.0
        message.gear = AckermannCommand.FORWARD if speed >= 0.0 else AckermannCommand.REVERSE
        message.brake = brake
        message.source = "p0_turning_calibration"
        self.publisher.publish(message)

    def run(self, speed, steering, settle_s, sample_s):
        rate = rospy.Rate(30)
        deadline = rospy.Time.now().to_sec() + settle_s
        while not rospy.is_shutdown() and rospy.Time.now().to_sec() < deadline:
            self.command(speed, steering)
            rate.sleep()
        self.recording = True
        deadline = rospy.Time.now().to_sec() + sample_s
        while not rospy.is_shutdown() and rospy.Time.now().to_sec() < deadline:
            self.command(speed, steering)
            rate.sleep()
        self.recording = False
        for _ in range(8):
            self.command(0.0, steering, brake=True)
            rate.sleep()

        points = np.asarray(self.poses, dtype=np.float64)
        if len(points) < 20:
            raise RuntimeError("not enough odometry samples")
        # Algebraic least-squares circle: x^2+y^2 = 2*cx*x+2*cy*y+c.
        design = np.column_stack((2.0 * points[:, 0], 2.0 * points[:, 1], np.ones(len(points))))
        cx, cy, constant = np.linalg.lstsq(design, np.sum(points * points, axis=1), rcond=None)[0]
        radius = math.sqrt(max(0.0, constant + cx * cx + cy * cy))
        residual = np.linalg.norm(points - (cx, cy), axis=1) - radius
        travelled = float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))
        elapsed = max(self.stamps[-1] - self.stamps[0], 1e-9)
        sign = 1.0 if steering >= 0.0 else -1.0
        return {
            "command_speed_mps": speed,
            "command_steering_rad": steering,
            "measured_speed_mean_mps": float(np.mean(self.speeds)),
            "position_derived_speed_mps": travelled / elapsed,
            "measured_steering_mean_rad": float(np.mean(self.steering)),
            "measured_turn_radius_m": radius,
            "measured_curvature_per_m": sign / radius,
            "circle_fit_rms_m": float(np.sqrt(np.mean(residual * residual))),
            "odometry_samples": len(points),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--speed", type=float, required=True)
    parser.add_argument("--steering", type=float, required=True)
    parser.add_argument("--settle", type=float, default=2.0)
    parser.add_argument("--sample", type=float, default=5.0)
    parser.add_argument("--output", type=Path)
    args, _ = parser.parse_known_args()
    rospy.init_node("dep_car_turning_calibration")
    calibration = TurningCalibration()
    rospy.sleep(1.0)
    payload = calibration.run(args.speed, args.steering, args.settle, args.sample)
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
