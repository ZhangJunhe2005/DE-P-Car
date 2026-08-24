#!/usr/bin/env python3
"""Replay position goals on one frozen map without adding map-specific policy."""

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import threading
import time

import rospy
from dep_car_msgs.msg import PlannerState, PolicyState
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import String


class GoalReplay:
    def __init__(self, goals, timeout_s):
        self.goals = tuple(tuple(float(value) for value in goal) for goal in goals)
        self.timeout_s = float(timeout_s)
        self.condition = threading.Condition()
        self.global_state = ""
        self.previous_global_state = ""
        self.global_counts = Counter()
        self.global_transition_counts = Counter()
        self.guidance_source_counts = Counter()
        self.guidance_source_transitions = Counter()
        self.boundary_mode_counts = Counter()
        self.boundary_loop_events = 0
        self.maximum_boundary_travel_m = 0.0
        self.maximum_failed_boundary_sides = 0
        self.previous_guidance_source = ""
        self.maximum_breadcrumb_count = 0
        self.maximum_topology_nodes = 0
        self.maximum_topology_edges = 0
        self.maximum_failed_branches = 0
        self.visibility_mode_counts = Counter()
        self.maximum_visibility_nodes = 0
        self.maximum_visibility_edges = 0
        self.maximum_visibility_planning_time_ms = 0.0
        self.route_progress_by_id = {}
        self.route_progress_regressions = 0
        self.route_ids_seen = set()
        self.turnaround_transaction_ids = set()
        self.goal_turnaround_transaction_ids = set()
        self.topology_rear_suppressed_samples = 0
        self.accumulated_map_seen = False
        self.map_odom_correction_samples = 0
        self.time_aligned_map_odom_samples = 0
        self.maximum_map_odom_translation_correction = 0.0
        self.maximum_map_odom_yaw_correction = 0.0
        self.maximum_map_odom_transform_skew = 0.0
        self.recovery_cycles = 0
        self.recoveries_after_escape = 0
        self.goal_recovery_cycles = 0
        self.goal_recovered_once = False
        self.dead_end_escape_ids = set()
        self.dead_end_escape_active_samples = 0
        self.dead_end_escape_completion_reasons = Counter()
        self.maximum_dead_end_escape_target_distance_m = 0.0
        self.maximum_dead_end_escape_cross_track_error_m = 0.0
        self.maximum_dead_end_escape_map_reanchors = 0
        self.previous_dead_end_escape_completion_reason = "NONE"
        self.maximum_maneuver_leg = 0
        self.static_blocks = 0
        self.policy_attempts = 0
        self.policy_sync_failures = 0
        self.map_position = None
        self.previous_map_position = None
        self.active_goal = None
        self.goal_start_distance = None
        self.goal_minimum_distance = None
        self.forward_distance = 0.0
        self.reverse_distance = 0.0
        self.stationary_distance = 0.0
        self.stationary_anchor = None
        self.maximum_stationary_excursion = 0.0
        self.stationary_map_jump_events = 0
        self.goal_metrics = []
        self.trace = []
        self.latest_status_document = {}
        self.last_trace_stamp = None
        self.publisher = rospy.Publisher(
            "/move_base_simple/goal", PoseStamped, queue_size=1, latch=True
        )
        rospy.Subscriber(
            "/dep_car/global_planner_status", String, self.on_global_status, queue_size=10
        )
        rospy.Subscriber(
            "/dep_car/planner_state", PlannerState, self.on_planner_state, queue_size=20
        )
        rospy.Subscriber(
            "/dep_car/policy_inference_state", PolicyState, self.on_policy_state, queue_size=10
        )
        rospy.Subscriber(
            "/dep_car/map_odometry", Odometry, self.on_map_odometry, queue_size=20
        )
        rospy.Subscriber(
            "/dep_car/map_odom_correction",
            String,
            self.on_map_odom_correction,
            queue_size=20,
        )

    def on_global_status(self, message):
        try:
            document = json.loads(message.data)
            if not isinstance(document, dict):
                raise ValueError("status document must be an object")
            state = str(document.get("state", "INVALID_STATUS"))
        except (TypeError, ValueError):
            document = {}
            state = "INVALID_STATUS"
        with self.condition:
            self.global_state = state
            self.global_counts[state] += 1
            if state != self.previous_global_state:
                self.global_transition_counts[state] += 1
                if state in ("BACKTRACK_REVERSE", "FAR_DEAD_END_EGRESS"):
                    self.recovery_cycles += 1
                    self.goal_recovery_cycles += 1
                    if self.goal_recovered_once:
                        self.recoveries_after_escape += 1
                elif state == "RECOVERED_GOAL_SEEK":
                    self.goal_recovered_once = True
                self.previous_global_state = state
            self.guidance_source_counts[str(document.get("guidance_source", "UNKNOWN"))] += 1
            guidance_source = str(document.get("guidance_source", "UNKNOWN"))
            if guidance_source != self.previous_guidance_source:
                self.guidance_source_transitions[guidance_source] += 1
                self.previous_guidance_source = guidance_source
            self.latest_status_document = document
            boundary = document.get("boundary_follow", {})
            self.boundary_mode_counts[str(boundary.get("mode", "UNKNOWN"))] += 1
            self.maximum_boundary_travel_m = max(
                self.maximum_boundary_travel_m,
                float(boundary.get("travelled_m", 0.0) or 0.0),
            )
            self.maximum_failed_boundary_sides = max(
                self.maximum_failed_boundary_sides,
                int(boundary.get("failed_side_count", 0) or 0),
            )
            self.boundary_loop_events += int("boundary_loop_reason" in document)
            self.maximum_breadcrumb_count = max(
                self.maximum_breadcrumb_count,
                int(document.get("breadcrumb_count", 0)),
            )
            topology = document.get("topology", {})
            self.maximum_topology_nodes = max(
                self.maximum_topology_nodes, int(topology.get("nodes", 0))
            )
            self.maximum_topology_edges = max(
                self.maximum_topology_edges, int(topology.get("edges", 0))
            )
            self.maximum_failed_branches = max(
                self.maximum_failed_branches,
                int(topology.get("failed_branches", 0)),
            )
            visibility = document.get("visibility_graph", {})
            self.visibility_mode_counts[
                str(visibility.get("mode", "UNAVAILABLE"))
            ] += 1
            self.maximum_visibility_nodes = max(
                self.maximum_visibility_nodes,
                int(visibility.get("nodes", 0) or 0),
            )
            self.maximum_visibility_edges = max(
                self.maximum_visibility_edges,
                int(visibility.get("edges", 0) or 0),
            )
            self.maximum_visibility_planning_time_ms = max(
                self.maximum_visibility_planning_time_ms,
                float(visibility.get("planning_time_ms", 0.0) or 0.0),
            )
            rolling = document.get("active_rolling_route", {})
            route_id = str(rolling.get("route_id", ""))
            if route_id:
                progress = float(rolling.get("progress_m", 0.0) or 0.0)
                previous_progress = self.route_progress_by_id.get(route_id)
                if (
                    previous_progress is not None
                    and progress + 1.0e-6 < previous_progress
                ):
                    self.route_progress_regressions += 1
                self.route_progress_by_id[route_id] = max(
                    progress,
                    -math.inf
                    if previous_progress is None
                    else previous_progress,
                )
                self.route_ids_seen.add(route_id)
            turnaround = document.get("route_turnaround_transaction", {})
            turnaround_id = int(turnaround.get("id", 0) or 0)
            if turnaround_id > 0:
                self.turnaround_transaction_ids.add(turnaround_id)
                self.goal_turnaround_transaction_ids.add(turnaround_id)
            self.topology_rear_suppressed_samples += int(
                visibility.get("route_acquisition_reason", "")
                == "consumed_topology_rear_route_rolling_local_exploration"
            )
            self.accumulated_map_seen |= bool(
                document.get("accumulated_map_available", False)
            )
            escape = document.get("dead_end_escape", {})
            escape_id = int(escape.get("escape_id", 0) or 0)
            if escape_id > 0:
                self.dead_end_escape_ids.add(escape_id)
            self.dead_end_escape_active_samples += int(
                bool(escape.get("active", False))
            )
            self.maximum_dead_end_escape_target_distance_m = max(
                self.maximum_dead_end_escape_target_distance_m,
                float(escape.get("target_distance_m", 0.0) or 0.0),
            )
            self.maximum_dead_end_escape_cross_track_error_m = max(
                self.maximum_dead_end_escape_cross_track_error_m,
                float(escape.get("cross_track_error_m", 0.0) or 0.0),
            )
            self.maximum_dead_end_escape_map_reanchors = max(
                self.maximum_dead_end_escape_map_reanchors,
                int(escape.get("map_reanchors", 0) or 0),
            )
            completion_reason = str(
                escape.get("completion_reason", "NONE") or "NONE"
            )
            if (
                completion_reason not in ("NONE", "ACTIVE")
                and completion_reason
                != self.previous_dead_end_escape_completion_reason
            ):
                self.dead_end_escape_completion_reasons[completion_reason] += 1
            self.previous_dead_end_escape_completion_reason = completion_reason
            self.condition.notify_all()

    def on_map_odom_correction(self, message):
        try:
            document = json.loads(message.data)
            translation = abs(float(document.get("translation_delta_m", 0.0)))
            yaw = abs(float(document.get("yaw_delta_rad", 0.0)))
            skew = abs(float(document.get("transform_skew_s", 0.0)))
            time_aligned = bool(document.get("time_aligned", False))
        except (TypeError, ValueError):
            return
        with self.condition:
            self.map_odom_correction_samples += 1
            self.time_aligned_map_odom_samples += int(time_aligned)
            self.maximum_map_odom_translation_correction = max(
                self.maximum_map_odom_translation_correction, translation
            )
            self.maximum_map_odom_yaw_correction = max(
                self.maximum_map_odom_yaw_correction, yaw
            )
            self.maximum_map_odom_transform_skew = max(
                self.maximum_map_odom_transform_skew, skew
            )

    def on_planner_state(self, message):
        with self.condition:
            self.maximum_maneuver_leg = max(
                self.maximum_maneuver_leg, int(message.maneuver_leg)
            )
            self.static_blocks += int(message.blocked_by_static)

    def on_policy_state(self, message):
        with self.condition:
            self.policy_attempts = max(
                self.policy_attempts, int(message.inference_attempts)
            )
            self.policy_sync_failures = max(
                self.policy_sync_failures, int(message.synchronization_failures)
            )

    def on_map_odometry(self, message):
        position = (
            float(message.pose.pose.position.x),
            float(message.pose.pose.position.y),
        )
        with self.condition:
            if self.previous_map_position is not None:
                step = math.hypot(
                    position[0] - self.previous_map_position[0],
                    position[1] - self.previous_map_position[1],
                )
                speed = float(message.twist.twist.linear.x)
                if speed > 0.02:
                    self.forward_distance += step
                    self.stationary_anchor = None
                elif speed < -0.02:
                    self.reverse_distance += step
                    self.stationary_anchor = None
                else:
                    self.stationary_distance += step
                    if self.stationary_anchor is None:
                        self.stationary_anchor = self.previous_map_position
                    excursion = math.hypot(
                        position[0] - self.stationary_anchor[0],
                        position[1] - self.stationary_anchor[1],
                    )
                    self.maximum_stationary_excursion = max(
                        self.maximum_stationary_excursion, excursion
                    )
                    self.stationary_map_jump_events += int(step >= 0.08)
            self.previous_map_position = position
            self.map_position = position
            if self.active_goal is not None:
                distance = math.hypot(
                    position[0] - self.active_goal[0],
                    position[1] - self.active_goal[1],
                )
                if self.goal_start_distance is None:
                    self.goal_start_distance = distance
                self.goal_minimum_distance = (
                    distance
                    if self.goal_minimum_distance is None
                    else min(self.goal_minimum_distance, distance)
                )
                stamp = message.header.stamp.to_sec() or rospy.Time.now().to_sec()
                if (
                    self.last_trace_stamp is None
                    or stamp - self.last_trace_stamp >= 0.50
                ):
                    status = self.latest_status_document
                    topology = status.get("topology", {})
                    boundary = status.get("boundary_follow", {})
                    visibility = status.get("visibility_graph", {})
                    escape = status.get("dead_end_escape", {})
                    self.trace.append({
                        "stamp": float(stamp),
                        "x": position[0],
                        "y": position[1],
                        "goal_distance_m": float(distance),
                        "state": str(status.get("state", self.global_state)),
                        "guidance_source": str(
                            status.get("guidance_source", "UNKNOWN")
                        ),
                        "topology_nodes": int(topology.get("nodes", 0)),
                        "topology_edges": int(topology.get("edges", 0)),
                        "failed_branches": int(
                            topology.get("failed_branches", 0)
                        ),
                        "boundary_mode": str(
                            boundary.get("mode", "UNKNOWN")
                        ),
                        "boundary_reason": str(
                            boundary.get("reason", "")
                        ),
                        "boundary_travelled_m": float(
                            boundary.get("travelled_m", 0.0) or 0.0
                        ),
                        "visibility_mode": str(
                            visibility.get("mode", "UNAVAILABLE")
                        ),
                        "visibility_path_points": int(
                            visibility.get("path_points", 0) or 0
                        ),
                        "visibility_nodes": int(
                            visibility.get("nodes", 0) or 0
                        ),
                        "visibility_edges": int(
                            visibility.get("edges", 0) or 0
                        ),
                        "rolling_route_id": str(
                            status.get("active_rolling_route", {}).get(
                                "route_id", ""
                            )
                        ),
                        "rolling_route_progress_m": float(
                            status.get("active_rolling_route", {}).get(
                                "progress_m", 0.0
                            ) or 0.0
                        ),
                        "rolling_route_carrot_m": float(
                            status.get("active_rolling_route", {}).get(
                                "carrot_m", 0.0
                            ) or 0.0
                        ),
                        "turnaround_transaction_id": int(
                            status.get(
                                "route_turnaround_transaction", {}
                            ).get("id", 0) or 0
                        ),
                        "dead_end_escape_id": int(
                            escape.get("escape_id", 0) or 0
                        ),
                        "dead_end_escape_active": bool(
                            escape.get("active", False)
                        ),
                        "dead_end_escape_target_distance_m": float(
                            escape.get("target_distance_m", 0.0) or 0.0
                        ),
                        "dead_end_escape_cross_track_error_m": float(
                            escape.get("cross_track_error_m", 0.0) or 0.0
                        ),
                        "dead_end_escape_completion_reason": str(
                            escape.get("completion_reason", "NONE") or "NONE"
                        ),
                    })
                    self.last_trace_stamp = stamp

    def publish(self, goal):
        message = PoseStamped()
        message.header.stamp = rospy.Time.now()
        message.header.frame_id = "map"
        message.pose.position.x = goal[0]
        message.pose.position.y = goal[1]
        message.pose.orientation.z = math.sin(0.5 * goal[2])
        message.pose.orientation.w = math.cos(0.5 * goal[2])
        with self.condition:
            self.global_state = ""
            self.previous_global_state = ""
            self.active_goal = goal
            self.goal_start_distance = None
            self.goal_minimum_distance = None
            self.goal_recovery_cycles = 0
            self.goal_recovered_once = False
            self.goal_turnaround_transaction_ids = set()
            self.last_trace_stamp = None
        self.publisher.publish(message)

    def finish_goal_metric(self, index, reached):
        final_distance = (
            None
            if self.map_position is None or self.active_goal is None
            else math.hypot(
                self.map_position[0] - self.active_goal[0],
                self.map_position[1] - self.active_goal[1],
            )
        )
        self.goal_metrics.append({
            "goal_index": int(index),
            "reached": bool(reached),
            "start_distance_m": self.goal_start_distance,
            "minimum_distance_m": self.goal_minimum_distance,
            "final_distance_m": final_distance,
            "recovery_cycles": int(self.goal_recovery_cycles),
            "turnaround_transactions": int(
                len(self.goal_turnaround_transaction_ids)
            ),
        })
        self.active_goal = None

    def wait_for_state(self, expected, simulation_start, wall_deadline):
        with self.condition:
            while not rospy.is_shutdown():
                if self.global_state == expected:
                    return True
                remaining = wall_deadline - time.monotonic()
                simulation_elapsed = rospy.Time.now().to_sec() - simulation_start
                if remaining <= 0.0 or simulation_elapsed >= self.timeout_s:
                    return False
                self.condition.wait(timeout=min(0.25, remaining))
        return False

    def run(self):
        wall_deadline = time.monotonic() + 10.0
        while self.publisher.get_num_connections() == 0 and not rospy.is_shutdown():
            if time.monotonic() >= wall_deadline:
                raise RuntimeError("goal publisher has no subscribers")
            rospy.sleep(0.05)
        completed = 0
        for index, goal in enumerate(self.goals):
            self.publish(goal)
            rospy.loginfo(
                "Regression replay goal %d/%d map=(%.3f,%.3f); heading is position-only",
                index + 1,
                len(self.goals),
                goal[0],
                goal[1],
            )
            simulation_start = rospy.Time.now().to_sec()
            wall_deadline = time.monotonic() + 3.0 * self.timeout_s
            if not self.wait_for_state(
                "GOAL_REACHED", simulation_start, wall_deadline
            ):
                self.finish_goal_metric(index, False)
                return self.report("FAIL", completed, "goal_timeout", index)
            self.finish_goal_metric(index, True)
            completed += 1
            rospy.sleep(0.5)
        return self.report("PASS", completed, "all_goals_reached", None)

    def report(self, status, completed, reason, failed_index):
        attempts = int(self.policy_attempts)
        failures = int(self.policy_sync_failures)
        output = {
            "schema": "DEPCarMemoryGoalReplayV3",
            "status": status,
            "reason": reason,
            "goals": len(self.goals),
            "completed_goals": int(completed),
            "failed_goal_index": failed_index,
            "global_state_counts": dict(sorted(self.global_counts.items())),
            "global_transition_counts": dict(sorted(self.global_transition_counts.items())),
            "guidance_source_counts": dict(sorted(self.guidance_source_counts.items())),
            "guidance_source_transitions": dict(
                sorted(self.guidance_source_transitions.items())
            ),
            "boundary_mode_counts": dict(sorted(self.boundary_mode_counts.items())),
            "boundary_loop_events": int(self.boundary_loop_events),
            "maximum_boundary_travel_m": float(self.maximum_boundary_travel_m),
            "maximum_failed_boundary_sides": int(
                self.maximum_failed_boundary_sides
            ),
            "maximum_breadcrumb_count": int(self.maximum_breadcrumb_count),
            "maximum_topology_nodes": int(self.maximum_topology_nodes),
            "maximum_topology_edges": int(self.maximum_topology_edges),
            "maximum_failed_branches": int(self.maximum_failed_branches),
            "visibility_mode_counts": dict(
                sorted(self.visibility_mode_counts.items())
            ),
            "maximum_visibility_nodes": int(self.maximum_visibility_nodes),
            "maximum_visibility_edges": int(self.maximum_visibility_edges),
            "maximum_visibility_planning_time_ms": float(
                self.maximum_visibility_planning_time_ms
            ),
            "rolling_route_ids_seen": int(len(self.route_ids_seen)),
            "rolling_route_progress_regressions": int(
                self.route_progress_regressions
            ),
            "turnaround_transactions": int(
                len(self.turnaround_transaction_ids)
            ),
            "topology_rear_suppressed_samples": int(
                self.topology_rear_suppressed_samples
            ),
            "accumulated_map_seen": bool(self.accumulated_map_seen),
            "map_odom_correction_samples": int(
                self.map_odom_correction_samples
            ),
            "maximum_map_odom_translation_correction_m": float(
                self.maximum_map_odom_translation_correction
            ),
            "maximum_map_odom_yaw_correction_rad": float(
                self.maximum_map_odom_yaw_correction
            ),
            "maximum_map_odom_transform_skew_s": float(
                self.maximum_map_odom_transform_skew
            ),
            "time_aligned_map_odom_rate": (
                None
                if self.map_odom_correction_samples == 0
                else float(self.time_aligned_map_odom_samples)
                / self.map_odom_correction_samples
            ),
            "recovery_cycles": int(self.recovery_cycles),
            "recoveries_after_escape": int(self.recoveries_after_escape),
            "far_dead_end_egress_transactions": int(
                len(self.dead_end_escape_ids)
            ),
            "far_dead_end_egress_active_samples": int(
                self.dead_end_escape_active_samples
            ),
            "far_dead_end_egress_completion_reasons": dict(
                sorted(self.dead_end_escape_completion_reasons.items())
            ),
            "maximum_far_dead_end_egress_target_distance_m": float(
                self.maximum_dead_end_escape_target_distance_m
            ),
            "maximum_far_dead_end_egress_cross_track_error_m": float(
                self.maximum_dead_end_escape_cross_track_error_m
            ),
            "maximum_far_dead_end_egress_map_reanchors": int(
                self.maximum_dead_end_escape_map_reanchors
            ),
            "maximum_maneuver_leg": int(self.maximum_maneuver_leg),
            "static_block_samples": int(self.static_blocks),
            "forward_distance_m": float(self.forward_distance),
            "reverse_distance_m": float(self.reverse_distance),
            # The old field summed 100 Hz jitter and therefore grew even when
            # the pose remained inside one small stationary cloud.  Report
            # maximum excursion as drift and retain the cumulative diagnostic
            # under an explicit name.
            "stationary_pose_drift_m": float(
                self.maximum_stationary_excursion
            ),
            "cumulative_stationary_map_motion_m": float(
                self.stationary_distance
            ),
            "stationary_map_jump_events": int(
                self.stationary_map_jump_events
            ),
            "goal_metrics": list(self.goal_metrics),
            "trace": list(self.trace),
            "policy_inference_attempts": attempts,
            "policy_synchronization_failures": failures,
            "policy_sync_success_rate": (
                None if attempts == 0 else float(attempts - failures) / attempts
            ),
        }
        print(json.dumps(output, indent=2, sort_keys=True), flush=True)
        return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--goal", nargs=3, type=float, action="append", required=True)
    parser.add_argument("--goal-timeout", type=float, default=90.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(rospy.myargv()[1:])
    rospy.init_node("dep_car_memory_goal_replay", anonymous=True)
    report = GoalReplay(args.goal, args.goal_timeout).run()
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
