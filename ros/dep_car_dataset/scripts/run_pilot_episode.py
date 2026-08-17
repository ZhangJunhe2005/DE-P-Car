#!/usr/bin/env python3
"""Run one P3 mission and emit a machine-readable episode outcome."""

import argparse
import json
import math
import threading
import time
from collections import Counter
from pathlib import Path

import rospy
import tf2_ros
from dep_car.core.vehicle import world_velocity_to_body_longitudinal
from dep_car_msgs.msg import (
    AckermannCommand,
    AckermannRoute,
    CandidateArray,
    LocalRouteCommand,
    PlannerState,
)
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid, Odometry
from sensor_msgs.msg import Image, Imu, PointCloud2


def yaw_from_quaternion(quaternion):
    return math.atan2(
        2.0 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y),
        1.0 - 2.0 * (quaternion.y * quaternion.y + quaternion.z * quaternion.z),
    )


def wrap_angle(value):
    return math.atan2(math.sin(value), math.cos(value))


class PilotEpisode:
    def __init__(self, args):
        self.args = args
        self.lock = threading.Lock()
        self.odom = None
        self.candidate_messages = 0
        self.zero_feasible_messages = 0
        self.feasible_counts = []
        self.requested_gears = Counter()
        self.planner_lifecycle = Counter()
        self.command_sources = Counter()
        self.illegal_shift_count = 0
        self.last_drive_gear = 0
        self.goal_reached_command = False
        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        rospy.Subscriber("/base_pose_ground_truth", Odometry, self.on_odom, queue_size=1)
        rospy.Subscriber("/dep_car/candidates", CandidateArray, self.on_candidates, queue_size=20)
        rospy.Subscriber("/dep_car/planner_state", PlannerState, self.on_planner_state, queue_size=20)
        rospy.Subscriber("/dep_car/cmd_ackermann", AckermannCommand, self.on_command, queue_size=20)
        self.goal_pub = rospy.Publisher("/move_base_simple/goal", PoseStamped, queue_size=1, latch=True)

    def on_odom(self, message):
        with self.lock:
            self.odom = message

    def signed_speed(self):
        if self.odom is None:
            return 0.0
        yaw = yaw_from_quaternion(self.odom.pose.pose.orientation)
        velocity = self.odom.twist.twist.linear
        return world_velocity_to_body_longitudinal(velocity.x, velocity.y, yaw)

    def on_candidates(self, message):
        feasible = sum(bool(candidate.feasible) for candidate in message.candidates)
        with self.lock:
            self.candidate_messages += 1
            self.feasible_counts.append(feasible)
            self.zero_feasible_messages += int(feasible == 0)
            self.requested_gears[str(int(message.requested_gear))] += 1

    def on_planner_state(self, message):
        with self.lock:
            self.planner_lifecycle[message.lifecycle_state] += 1

    def on_command(self, message):
        with self.lock:
            self.command_sources[message.source] += 1
            if message.source == "mission_goal_reached":
                self.goal_reached_command = True
            gear = int(message.gear)
            if not message.brake and gear in (-1, 1):
                if self.last_drive_gear in (-1, 1) and gear != self.last_drive_gear and abs(self.signed_speed()) > 0.03:
                    self.illegal_shift_count += 1
                self.last_drive_gear = gear

    @staticmethod
    def remaining_timeout(deadline, description):
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            raise rospy.ROSException("timed out waiting for " + description)
        return remaining

    def wait_for_inputs(self):
        deadline = time.monotonic() + self.args.startup_timeout
        requirements = (
            ("/map", OccupancyGrid), ("/velodyne_points", PointCloud2),
            ("/camera/depth/image_raw", Image), ("/imu/data", Imu),
            ("/base_pose_ground_truth", Odometry),
        )
        for topic, message_type in requirements:
            rospy.wait_for_message(
                topic, message_type,
                timeout=self.remaining_timeout(deadline, topic),
            )
        last_tf_error = None
        while not rospy.is_shutdown():
            try:
                self.tf_buffer.lookup_transform(
                    "map", "chassis", rospy.Time(0), rospy.Duration(0.0)
                )
                return
            except (
                tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException,
            ) as exc:
                last_tf_error = exc
            remaining = self.remaining_timeout(deadline, "map -> chassis TF")
            time.sleep(min(0.05, remaining))
        raise rospy.ROSInterruptException(
            "shutdown while waiting for map -> chassis TF: %s" % last_tf_error
        )

    def wait_for_planner_outputs(self):
        """Do not charge slow Hybrid A* startup against the episode window."""

        deadline = time.monotonic() + self.args.startup_timeout
        requirements = (
            ("/dep_car/global_route", AckermannRoute),
            ("/dep_car/local_route_command", LocalRouteCommand),
            ("/dep_car/candidates", CandidateArray),
        )
        for topic, message_type in requirements:
            rospy.wait_for_message(
                topic, message_type,
                timeout=self.remaining_timeout(deadline, topic),
            )

    def publish_goal(self):
        goal = PoseStamped()
        goal.header.stamp = rospy.Time.now()
        goal.header.frame_id = "map"
        goal.pose.position.x, goal.pose.position.y = self.args.goal_x, self.args.goal_y
        goal.pose.orientation.z = math.sin(0.5 * self.args.goal_yaw)
        goal.pose.orientation.w = math.cos(0.5 * self.args.goal_yaw)
        deadline = time.monotonic() + 3.0
        while not rospy.is_shutdown() and self.goal_pub.get_num_connections() == 0 and time.monotonic() < deadline:
            time.sleep(0.05)
        for _ in range(3):
            goal.header.stamp = rospy.Time.now()
            self.goal_pub.publish(goal)
            time.sleep(0.10)

    def goal_error(self):
        with self.lock:
            odom = self.odom
        if odom is None:
            return float("inf"), float("inf")
        distance = math.hypot(odom.pose.pose.position.x - self.args.goal_x, odom.pose.pose.position.y - self.args.goal_y)
        heading = abs(wrap_angle(yaw_from_quaternion(odom.pose.pose.orientation) - self.args.goal_yaw))
        return distance, heading

    def run(self):
        self.wait_for_inputs()
        self.publish_goal()
        self.wait_for_planner_outputs()
        started_wall = time.monotonic()
        started_sim = rospy.Time.now().to_sec()
        status = "TIMEOUT"
        rate = rospy.Rate(20)
        while not rospy.is_shutdown() and time.monotonic() - started_wall < self.args.episode_timeout:
            distance, heading = self.goal_error()
            with self.lock:
                reached_command = self.goal_reached_command
            if reached_command or (distance <= self.args.goal_tolerance and heading <= self.args.heading_tolerance):
                status = "SUCCESS"
                break
            rate.sleep()
        distance, heading = self.goal_error()
        with self.lock:
            if self.candidate_messages == 0:
                status = "NO_CANDIDATES"
            payload = {
                "schema": "DEPCarP3PilotEpisodeResultV1",
                "task_id": self.args.task_id,
                "maneuver_mode": self.args.maneuver_mode,
                "status": status,
                "wall_duration_s": time.monotonic() - started_wall,
                "sim_duration_s": max(0.0, rospy.Time.now().to_sec() - started_sim),
                "final_goal_distance_m": distance,
                "final_heading_error_rad": heading,
                "candidate_messages": self.candidate_messages,
                "zero_feasible_messages": self.zero_feasible_messages,
                "feasible_candidate_min": min(self.feasible_counts) if self.feasible_counts else 0,
                "feasible_candidate_median": float(sorted(self.feasible_counts)[len(self.feasible_counts) // 2]) if self.feasible_counts else 0.0,
                "requested_gear_counts": dict(self.requested_gears),
                "planner_lifecycle_counts": dict(self.planner_lifecycle),
                "command_source_counts": dict(self.command_sources),
                "illegal_shift_count": self.illegal_shift_count,
            }
        return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--maneuver-mode", required=True)
    parser.add_argument("--goal-x", type=float, required=True)
    parser.add_argument("--goal-y", type=float, required=True)
    parser.add_argument("--goal-yaw", type=float, required=True)
    parser.add_argument("--startup-timeout", type=float, default=90.0)
    parser.add_argument("--episode-timeout", type=float, default=18.0)
    parser.add_argument("--goal-tolerance", type=float, default=0.22)
    parser.add_argument("--heading-tolerance", type=float, default=0.35)
    parser.add_argument("--output", type=Path, required=True)
    args, _ = parser.parse_known_args()
    rospy.init_node("dep_car_pilot_episode", anonymous=True)
    try:
        payload = PilotEpisode(args).run()
    except Exception as exc:
        payload = {
            "schema": "DEPCarP3PilotEpisodeResultV1", "task_id": args.task_id,
            "maneuver_mode": args.maneuver_mode, "status": "INFRASTRUCTURE_ERROR",
            "error": type(exc).__name__ + ": " + str(exc),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(payload, indent=2, sort_keys=True))
    raise SystemExit(0 if payload["status"] != "INFRASTRUCTURE_ERROR" else 2)


if __name__ == "__main__":
    main()
