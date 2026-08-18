#!/usr/bin/env python3
"""Run one fixed P6 static mission and emit closed-loop evidence."""

import argparse
import json
import math
import statistics
import threading
import time
from collections import Counter
from pathlib import Path

import numpy as np
import rospy
from dep_car.core.occupancy import FootprintConfig
from dep_car.runtime.occupancy import RuntimeOccupancyGrid2D
from dep_car.core.vehicle import world_velocity_to_body_longitudinal
from dep_car_msgs.msg import (
    AckermannCommand,
    CandidateArray,
    PlannerState,
    PolicyCandidateArray,
    PolicyState,
)
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid, Odometry
from std_msgs.msg import String


def yaw_from_quaternion(quaternion):
    return math.atan2(
        2.0 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y),
        1.0 - 2.0 * (quaternion.y * quaternion.y + quaternion.z * quaternion.z),
    )


def wrap_angle(value):
    return math.atan2(math.sin(value), math.cos(value))


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    index = int(math.ceil(fraction * len(ordered))) - 1
    return ordered[max(0, min(len(ordered) - 1, index))]


class StaticEpisode:
    def __init__(self, args):
        self.args = args
        self.lock = threading.Lock()
        self.odom = None
        self.grid = None
        self.footprint = FootprintConfig()
        self.minimum_clearance = float("inf")
        self.collision_samples = 0
        self.distance = 0.0
        self.last_position = None
        self.command_sources = Counter()
        self.requested_gears = Counter()
        self.illegal_shifts = 0
        self.last_drive_gear = 0
        self.reverse_drive_commands = 0
        self.direction_changes = 0
        self.raw_policy_messages = 0
        self.active_stop_commands = 0
        self.last_command_source = ""
        self.safe_policy_messages = 0
        self.zero_feasible_messages = 0
        self.hard_safety_samples = 0
        self.policy_reasons = Counter()
        self.inference_reasons = Counter()
        self.global_planner_states = Counter()
        self.local_planner_details = Counter()
        self.latencies = []
        self.sensor_skews = []
        self.model_loaded = False
        self.inference_ok_samples = 0
        self.control_authorized_samples = 0
        self.checkpoint_sha256 = ""
        self.modality = ""
        self.latest_control_candidate = None
        self.policy_selection_comparisons = 0
        self.policy_selection_disagreements = 0
        self.goal_pub = rospy.Publisher(
            "/move_base_simple/goal", PoseStamped, queue_size=1, latch=True
        )
        rospy.Subscriber("/map", OccupancyGrid, self.on_map, queue_size=1)
        rospy.Subscriber("/base_pose_ground_truth", Odometry, self.on_odom, queue_size=1)
        rospy.Subscriber("/dep_car/cmd_ackermann", AckermannCommand, self.on_command, queue_size=50)
        rospy.Subscriber("/dep_car/candidates", CandidateArray, self.on_control_candidates, queue_size=20)
        rospy.Subscriber("/dep_car/policy_candidates", CandidateArray, self.on_policy_candidates, queue_size=20)
        rospy.Subscriber(
            "/dep_car/policy_candidates_raw",
            PolicyCandidateArray,
            self.on_raw_policy,
            queue_size=20,
        )
        rospy.Subscriber(
            "/dep_car/policy_state", PolicyState, self.on_safety_state, queue_size=50
        )
        rospy.Subscriber(
            "/dep_car/global_planner_status", String, self.on_global_planner_status,
            queue_size=20,
        )
        rospy.Subscriber(
            "/dep_car/planner_state", PlannerState, self.on_local_planner_state,
            queue_size=50,
        )
        rospy.Subscriber(
            "/dep_car/policy_inference_state",
            PolicyState,
            self.on_inference_state,
            queue_size=50,
        )

    def on_map(self, message):
        data = np.asarray(message.data, dtype=np.int16).reshape(
            (message.info.height, message.info.width)
        )
        grid = RuntimeOccupancyGrid2D(
            data,
            message.info.resolution,
            (message.info.origin.position.x, message.info.origin.position.y),
        )
        with self.lock:
            self.grid = grid

    def on_odom(self, message):
        point = np.asarray(
            [message.pose.pose.position.x, message.pose.pose.position.y], dtype=float
        )
        yaw = yaw_from_quaternion(message.pose.pose.orientation)
        with self.lock:
            if self.last_position is not None:
                self.distance += float(np.linalg.norm(point - self.last_position))
            self.last_position = point
            self.odom = message
            grid = self.grid
        if grid is not None:
            pose = np.asarray([[0.0, point[0], point[1], yaw, 0.0, 0.0]])
            safe, clearance = grid.swept_footprint_clearance(pose, self.footprint)
            with self.lock:
                self.minimum_clearance = min(self.minimum_clearance, clearance)
                self.collision_samples += int(not safe)

    def signed_speed(self):
        with self.lock:
            odom = self.odom
        if odom is None:
            return 0.0
        yaw = yaw_from_quaternion(odom.pose.pose.orientation)
        velocity = odom.twist.twist.linear
        return world_velocity_to_body_longitudinal(velocity.x, velocity.y, yaw)

    def on_command(self, message):
        gear = int(message.gear)
        speed = self.signed_speed()
        with self.lock:
            self.command_sources[message.source] += 1
            self.last_command_source = str(message.source)
            active_stop = (
                not message.brake
                and gear in (-1, 1)
                and abs(float(message.speed)) <= 1.0e-6
            )
            self.active_stop_commands += int(active_stop)
            if not message.brake and gear in (-1, 1) and not active_stop:
                if self.last_drive_gear in (-1, 1) and gear != self.last_drive_gear:
                    self.direction_changes += 1
                    if abs(speed) > 0.03:
                        self.illegal_shifts += 1
                self.last_drive_gear = gear
                self.reverse_drive_commands += int(gear == -1)

    def on_control_candidates(self, message):
        with self.lock:
            self.latest_control_candidate = int(message.selected_candidate_id)
            self.requested_gears[str(int(message.requested_gear))] += 1

    def on_policy_candidates(self, message):
        feasible = sum(bool(candidate.feasible) for candidate in message.candidates)
        with self.lock:
            self.safe_policy_messages += 1
            self.zero_feasible_messages += int(feasible == 0)
            if self.latest_control_candidate is not None and message.selected_candidate_id >= 0:
                self.policy_selection_comparisons += 1
                self.policy_selection_disagreements += int(
                    int(message.selected_candidate_id) != self.latest_control_candidate
                )

    def on_raw_policy(self, _message):
        with self.lock:
            self.raw_policy_messages += 1

    def on_inference_state(self, message):
        with self.lock:
            self.model_loaded = self.model_loaded or bool(message.model_loaded)
            self.inference_ok_samples += int(message.inference_ok)
            self.control_authorized_samples += int(message.control_authorized)
            self.inference_reasons[message.reason] += 1
            if message.inference_ok:
                self.latencies.append(float(message.inference_latency_ms))
                self.sensor_skews.append(float(message.sensor_skew_s))
            if message.checkpoint_sha256:
                self.checkpoint_sha256 = message.checkpoint_sha256
            if message.modality:
                self.modality = message.modality

    def on_safety_state(self, message):
        with self.lock:
            self.hard_safety_samples += int(message.hard_safety_applied)
            self.policy_reasons[message.reason] += 1

    def on_global_planner_status(self, message):
        try:
            state = str(json.loads(message.data).get("state", "UNKNOWN"))
        except (TypeError, ValueError, json.JSONDecodeError):
            state = "MALFORMED"
        with self.lock:
            self.global_planner_states[state] += 1

    def on_local_planner_state(self, message):
        with self.lock:
            self.local_planner_details[message.detail or "EMPTY"] += 1

    @staticmethod
    def remaining(deadline, description):
        value = deadline - time.monotonic()
        if value <= 0.0:
            raise rospy.ROSException("timed out waiting for " + description)
        return value

    def wait_for_readiness(self):
        deadline = time.monotonic() + self.args.startup_timeout
        for topic, message_type in (
            ("/map", OccupancyGrid),
            ("/base_pose_ground_truth", Odometry),
            ("/dep_car/policy_inference_state", PolicyState),
        ):
            rospy.wait_for_message(topic, message_type, timeout=self.remaining(deadline, topic))
        while not rospy.is_shutdown():
            with self.lock:
                ready = self.model_loaded
            if ready:
                return
            time.sleep(min(0.05, self.remaining(deadline, "P6 policy readiness")))

    def wait_for_policy_flow(self):
        deadline = time.monotonic() + self.args.startup_timeout
        while not rospy.is_shutdown():
            with self.lock:
                ready = (
                    self.inference_ok_samples > 0
                    and self.raw_policy_messages > 0
                    and self.safe_policy_messages > 0
                    and self.hard_safety_samples > 0
                )
            if ready:
                return
            time.sleep(min(0.05, self.remaining(deadline, "P6 policy/safety flow")))

    def publish_goal(self):
        goal = PoseStamped()
        goal.header.frame_id = "map"
        goal.pose.position.x = self.args.goal_x
        goal.pose.position.y = self.args.goal_y
        goal.pose.orientation.z = math.sin(0.5 * self.args.goal_yaw)
        goal.pose.orientation.w = math.cos(0.5 * self.args.goal_yaw)
        deadline = time.monotonic() + 3.0
        while (
            not rospy.is_shutdown()
            and self.goal_pub.get_num_connections() == 0
            and time.monotonic() < deadline
        ):
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
        distance = math.hypot(
            odom.pose.pose.position.x - self.args.goal_x,
            odom.pose.pose.position.y - self.args.goal_y,
        )
        heading = abs(
            wrap_angle(yaw_from_quaternion(odom.pose.pose.orientation) - self.args.goal_yaw)
        )
        return distance, heading

    def run(self):
        self.wait_for_readiness()
        self.publish_goal()
        self.wait_for_policy_flow()
        started_wall = time.monotonic()
        started_sim = rospy.Time.now().to_sec()
        # Scenario limits are expressed in simulated seconds so a slower
        # Gazebo real-time factor cannot turn an otherwise identical replay
        # into a host-dependent timeout.  Keep a separate generous wall-clock
        # watchdog so a paused or crashed simulator still terminates.
        wall_watchdog_s = max(
            3.0 * self.args.episode_timeout,
            self.args.episode_timeout + 60.0,
        )
        status = "TIMEOUT"
        timeout_basis = "simulation"
        settled_since = None
        achieved_goal_hold_s = 0.0
        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            wall_elapsed = time.monotonic() - started_wall
            sim_elapsed = max(0.0, rospy.Time.now().to_sec() - started_sim)
            if sim_elapsed >= self.args.episode_timeout:
                timeout_basis = "simulation"
                break
            if wall_elapsed >= wall_watchdog_s:
                timeout_basis = "wall_watchdog"
                break
            distance, heading = self.goal_error()
            with self.lock:
                collided = self.collision_samples > 0
            if collided:
                status = "COLLISION"
                break
            in_goal = (
                distance <= self.args.goal_tolerance
                and heading <= self.args.heading_tolerance
            )
            with self.lock:
                latched_hold = self.last_command_source == "mission_goal_reached"
            settled = abs(self.signed_speed()) <= self.args.settled_speed
            if in_goal and settled and latched_hold:
                if settled_since is None:
                    settled_since = time.monotonic()
                achieved_goal_hold_s = time.monotonic() - settled_since
                if achieved_goal_hold_s >= self.args.goal_hold_time:
                    status = "SUCCESS"
                    break
            else:
                settled_since = None
                achieved_goal_hold_s = 0.0
            rate.sleep()
        distance, heading = self.goal_error()
        final_signed_speed = self.signed_speed()
        with self.lock:
            if not self.model_loaded or self.raw_policy_messages == 0:
                status = "POLICY_UNAVAILABLE"
            if self.args.policy_mode == "shadow" and any(
                source == "dep_car_net_v1_active" for source in self.command_sources
            ):
                status = "SHADOW_AUTHORITY_VIOLATION"
            if self.args.policy_mode == "active" and self.control_authorized_samples == 0:
                status = "ACTIVE_AUTHORITY_MISSING"
            minimum_clearance = (
                self.minimum_clearance if math.isfinite(self.minimum_clearance) else None
            )
            payload = {
                "schema": "DEPCarP6StaticEpisodeV1",
                "status": status,
                "scenario_id": self.args.scenario_id,
                "maneuver_mode": self.args.maneuver_mode,
                "cohort": self.args.cohort,
                "policy_mode": self.args.policy_mode,
                "requested_modality": self.args.modality,
                "observed_modality": self.modality,
                "checkpoint_sha256": self.checkpoint_sha256,
                "scenario_manifest_sha256": self.args.scenario_manifest_sha256,
                "runtime_implementation_sha256": self.args.runtime_implementation_sha256,
                "map_uuid": self.args.map_uuid,
                "map_seed": self.args.map_seed,
                "map_occupancy_sha256": self.args.map_occupancy_sha256,
                "gazebo_seed": self.args.gazebo_seed,
                "start": [self.args.start_x, self.args.start_y, self.args.start_yaw],
                "goal": [self.args.goal_x, self.args.goal_y, self.args.goal_yaw],
                "wall_duration_s": time.monotonic() - started_wall,
                "sim_duration_s": max(0.0, rospy.Time.now().to_sec() - started_sim),
                "timeout_basis": timeout_basis if status == "TIMEOUT" else None,
                "wall_watchdog_s": wall_watchdog_s,
                "final_goal_distance_m": distance,
                "final_heading_error_rad": heading,
                "final_signed_speed_mps": final_signed_speed,
                "goal_hold_s": achieved_goal_hold_s,
                "distance_travelled_m": self.distance,
                "minimum_static_clearance_m": minimum_clearance,
                "collision_samples": self.collision_samples,
                "illegal_shift_count": self.illegal_shifts,
                "direction_change_count": self.direction_changes,
                "reverse_drive_commands": self.reverse_drive_commands,
                "active_stop_commands": self.active_stop_commands,
                "command_source_counts": dict(self.command_sources),
                "requested_gear_counts": dict(self.requested_gears),
                "raw_policy_messages": self.raw_policy_messages,
                "safe_policy_messages": self.safe_policy_messages,
                "zero_feasible_messages": self.zero_feasible_messages,
                "hard_safety_samples": self.hard_safety_samples,
                "inference_ok_samples": self.inference_ok_samples,
                "control_authorized_samples": self.control_authorized_samples,
                "policy_reason_counts": dict(self.policy_reasons),
                "inference_reason_counts": dict(self.inference_reasons),
                "global_planner_state_counts": dict(self.global_planner_states),
                "local_planner_detail_counts": dict(self.local_planner_details),
                "inference_latency_ms": {
                    "samples": len(self.latencies),
                    "mean": statistics.fmean(self.latencies) if self.latencies else None,
                    "p95": percentile(self.latencies, 0.95),
                    "maximum": max(self.latencies) if self.latencies else None,
                },
                "sensor_skew_s": {
                    "samples": len(self.sensor_skews),
                    "p95": percentile(self.sensor_skews, 0.95),
                    "maximum": max(self.sensor_skews) if self.sensor_skews else None,
                },
                "shadow_selection": {
                    "comparisons": self.policy_selection_comparisons,
                    "disagreements": self.policy_selection_disagreements,
                    "disagreement_rate": (
                        self.policy_selection_disagreements
                        / self.policy_selection_comparisons
                        if self.policy_selection_comparisons
                        else None
                    ),
                },
            }
        return payload


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario-id", required=True)
    parser.add_argument("--maneuver-mode", required=True)
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--policy-mode", choices=("shadow", "active"), required=True)
    parser.add_argument("--modality", choices=("depth_only", "lidar_only", "fusion"), required=True)
    parser.add_argument("--scenario-manifest-sha256", required=True)
    parser.add_argument("--runtime-implementation-sha256", required=True)
    parser.add_argument("--map-uuid", required=True)
    parser.add_argument("--map-seed", type=int, required=True)
    parser.add_argument("--map-occupancy-sha256", required=True)
    parser.add_argument("--gazebo-seed", type=int, required=True)
    parser.add_argument("--start-x", type=float, required=True)
    parser.add_argument("--start-y", type=float, required=True)
    parser.add_argument("--start-yaw", type=float, required=True)
    parser.add_argument("--goal-x", type=float, required=True)
    parser.add_argument("--goal-y", type=float, required=True)
    parser.add_argument("--goal-yaw", type=float, required=True)
    parser.add_argument("--startup-timeout", type=float, default=120.0)
    parser.add_argument("--episode-timeout", type=float, default=60.0)
    parser.add_argument("--goal-tolerance", type=float, default=0.22)
    parser.add_argument("--heading-tolerance", type=float, default=0.35)
    parser.add_argument("--settled-speed", type=float, default=0.04)
    parser.add_argument("--goal-hold-time", type=float, default=0.50)
    parser.add_argument("--output", type=Path, required=True)
    args, _ = parser.parse_known_args()
    rospy.init_node("dep_car_p6_static_episode", anonymous=True)
    try:
        payload = StaticEpisode(args).run()
    except Exception as exc:
        payload = {
            "schema": "DEPCarP6StaticEpisodeV1",
            "status": "INFRASTRUCTURE_ERROR",
            "scenario_id": args.scenario_id,
            "policy_mode": args.policy_mode,
            "requested_modality": args.modality,
            "runtime_implementation_sha256": args.runtime_implementation_sha256,
            "error": type(exc).__name__ + ": " + str(exc),
        }
    write_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    raise SystemExit(0 if payload.get("status") == "SUCCESS" else 2)


if __name__ == "__main__":
    main()
