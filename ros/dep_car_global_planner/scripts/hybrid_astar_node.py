#!/usr/bin/env python3
"""Plan a car-feasible global path and publish a bounded local subgoal."""

import json
import math
import threading
import time

import numpy as np
import rospy
from dep_car.runtime.occupancy import RuntimeOccupancyGrid2D
from dep_car.core.types import Gear
from dep_car.global_planner.hybrid_astar import HybridAStar
from dep_car.global_planner.topological_astar import TopologicalAStar, corridor_gear_hint
from dep_car.runtime.route_guidance import (
    monotonic_route_index,
    visible_corridor_subgoal,
)
from dep_car_msgs.msg import AckermannRoute, LocalRouteCommand, RoutePoint
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from std_msgs.msg import String
from visualization_msgs.msg import Marker


def yaw(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class HybridAStarNode:
    def __init__(self):
        self.lock = threading.Lock()
        self.plan_lock = threading.Lock()
        self.grid = self.odom = self.goal = None
        self.cached_poses = None
        self.cached_plan_mode = None
        self.cached_goal = None
        self.cached_status = "IDLE"
        self.terminal_exact_goal = None
        self.route_index = 0
        self.goal_generation = 0
        self.plan_revision = 0
        self.hybrid_validator = HybridAStar()
        self.require_goal_heading = bool(
            rospy.get_param("~require_goal_heading", True)
        )
        self.route_mode = str(rospy.get_param("~route_mode", "topology_corridor"))
        if self.route_mode not in ("topology_corridor", "hybrid_exact"):
            raise ValueError("route_mode must be topology_corridor or hybrid_exact")
        self.planner = (
            TopologicalAStar()
            if self.route_mode == "topology_corridor"
            else self.hybrid_validator
        )
        self.lookahead = float(rospy.get_param("~local_subgoal_distance", 1.50))
        self.exact_route_lookahead = float(
            rospy.get_param("~exact_route_lookahead", 0.50)
        )
        self.terminal_hybrid_radius = float(
            rospy.get_param("~terminal_hybrid_radius", 2.0)
        )
        self.terminal_hybrid_heading_trigger = float(
            rospy.get_param("~terminal_hybrid_heading_trigger", 0.55)
        )
        self.terminal_hybrid_capture_radius = float(
            rospy.get_param("~terminal_hybrid_capture_radius", 0.80)
        )
        self.corridor_replan_deviation = float(
            rospy.get_param("~corridor_replan_deviation", 1.50)
        )
        self.exact_replan_deviation = float(
            rospy.get_param("~exact_replan_deviation", 0.75)
        )
        self.planning_timeout = float(rospy.get_param("~planning_timeout_s", 5.0))
        self.terminal_planning_timeout = float(
            rospy.get_param("~terminal_planning_timeout_s", 15.0)
        )
        self.duplicate_goal_window = float(
            rospy.get_param("~duplicate_goal_window_s", 0.5)
        )
        if self.planning_timeout <= 0.0:
            raise ValueError("planning_timeout_s must be positive")
        if self.lookahead <= 0.0:
            raise ValueError("local_subgoal_distance must be positive")
        if self.exact_route_lookahead <= 0.0:
            raise ValueError("exact_route_lookahead must be positive")
        if self.terminal_planning_timeout <= 0.0:
            raise ValueError("terminal_planning_timeout_s must be positive")
        if (
            self.terminal_hybrid_capture_radius <= 0.0
            or self.terminal_hybrid_capture_radius > self.terminal_hybrid_radius
        ):
            raise ValueError(
                "terminal_hybrid_capture_radius must be in (0, terminal_hybrid_radius]"
            )
        if self.duplicate_goal_window < 0.0:
            raise ValueError("duplicate_goal_window_s must be non-negative")
        if min(
            self.corridor_replan_deviation,
            self.exact_replan_deviation,
        ) <= 0.0:
            raise ValueError("route replan deviations must be positive")
        self.last_goal_signature = None
        self.last_goal_received_wall = float("-inf")
        self.path_pub = rospy.Publisher("/dep_car/global_path", Path, queue_size=1, latch=True)
        self.route_pub = rospy.Publisher("/dep_car/global_route", AckermannRoute, queue_size=1, latch=True)
        self.subgoal_pub = rospy.Publisher("/dep_car/local_subgoal", PoseStamped, queue_size=1, latch=True)
        self.route_command_pub = rospy.Publisher("/dep_car/local_route_command", LocalRouteCommand, queue_size=1, latch=True)
        self.status_pub = rospy.Publisher(
            "/dep_car/global_planner_status", String, queue_size=1, latch=True
        )
        self.marker_pub = rospy.Publisher(
            "/dep_car/global_planner_status_marker", Marker, queue_size=1, latch=True
        )
        rospy.Subscriber("/map", OccupancyGrid, self.on_grid, queue_size=1)
        rospy.Subscriber("/base_pose_ground_truth", Odometry, self.on_odom, queue_size=1)
        rospy.Subscriber("/move_base_simple/goal", PoseStamped, self.on_goal, queue_size=1)
        rospy.Subscriber(
            "/dep_car/replan_request", String, self.on_replan_request, queue_size=1
        )
        self.timer = rospy.Timer(rospy.Duration(1.0), self.update)

    def on_grid(self, message):
        data = np.asarray(message.data, dtype=np.int16).reshape(message.info.height, message.info.width)
        with self.lock:
            self.grid = RuntimeOccupancyGrid2D(data, message.info.resolution, (message.info.origin.position.x, message.info.origin.position.y))
            self.cached_poses = None
            self.cached_plan_mode = None
            self.cached_status = "IDLE"
            self.terminal_exact_goal = None
            self.plan_revision += 1

    def on_odom(self, message):
        with self.lock: self.odom = message

    def on_goal(self, message):
        signature = self.goal_signature(
            message.pose.position.x,
            message.pose.position.y,
            yaw(message.pose.orientation),
        )
        received_wall = time.monotonic()
        with self.lock:
            if (
                signature == self.last_goal_signature
                and received_wall - self.last_goal_received_wall
                <= self.duplicate_goal_window
            ):
                return
            self.last_goal_signature = signature
            self.last_goal_received_wall = received_wall
            self.goal = message
            self.cached_poses = None
            self.cached_plan_mode = None
            self.cached_goal = None
            self.cached_status = "RECEIVED"
            self.terminal_exact_goal = None
            self.route_index = 0
            self.goal_generation += 1
            self.plan_revision += 1
            generation = self.goal_generation
        self.clear_route_visualization()
        self.publish_status("RECEIVED", "waiting for global planning", message, generation)

    def goal_signature(self, x, y, heading):
        """Ignore RViz's mandatory arrow angle for position-only missions."""

        return (
            round(float(x), 4),
            round(float(y), 4),
            round(float(heading), 4) if self.require_goal_heading else 0.0,
        )

    def position_only_target(self, grid, start, x, y):
        """Choose any collision-free terminal body orientation at a goal point."""

        approach = math.atan2(float(y) - start[1], float(x) - start[0])
        angles = [approach, start[2], approach + math.pi, start[2] + math.pi]
        angles.extend(np.linspace(-math.pi, math.pi, 24, endpoint=False).tolist())
        best = None
        seen = set()
        for angle in angles:
            angle = math.atan2(math.sin(angle), math.cos(angle))
            key = round(angle, 6)
            if key in seen:
                continue
            seen.add(key)
            target = (float(x), float(y), angle)
            valid, reason, clearance = self.hybrid_validator.validate_goal_pose(
                grid, target
            )
            if valid and (best is None or clearance > best[3]):
                best = (target, valid, "GOAL_POSITION_VALID", clearance)
        if best is not None:
            return best
        target = (float(x), float(y), approach)
        valid, reason, clearance = self.hybrid_validator.validate_goal_pose(
            grid, target
        )
        return target, valid, reason, clearance

    def on_replan_request(self, message):
        with self.lock:
            goal = self.goal
            if goal is None or self.cached_status in ("RECEIVED", "PLANNING"):
                return
            self.cached_poses = None
            self.cached_plan_mode = None
            self.cached_status = "PLANNING"
            self.route_index = 0
            self.plan_revision += 1
            generation = self.goal_generation
        self.clear_route_visualization()
        self.publish_status(
            "PLANNING",
            "local safety requested measured-pose replan: " + str(message.data),
            goal,
            generation,
        )

    def clear_route_visualization(self):
        path = Path()
        path.header.stamp = rospy.Time.now()
        path.header.frame_id = "map"
        route = AckermannRoute()
        route.header = path.header
        self.path_pub.publish(path)
        self.route_pub.publish(route)

    def publish_status(
        self,
        state,
        detail,
        goal,
        generation,
        *,
        elapsed=0.0,
        expansions=0,
        diagnostics=None,
    ):
        payload = {
            "state": str(state),
            "detail": str(detail),
            "goal_generation": int(generation),
            "planning_time_s": float(elapsed),
            "expansions": int(expansions),
        }
        if goal is not None:
            payload["goal"] = [
                float(goal.pose.position.x),
                float(goal.pose.position.y),
                float(yaw(goal.pose.orientation)),
            ]
        if diagnostics:
            payload["diagnostics"] = diagnostics
        self.status_pub.publish(String(data=json.dumps(payload, sort_keys=True)))
        if goal is None:
            return
        marker = Marker()
        marker.header.stamp = rospy.Time.now()
        marker.header.frame_id = "map"
        marker.ns = "dep_car_global_planner_status"
        marker.id = 0
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        marker.pose.position.x = goal.pose.position.x
        marker.pose.position.y = goal.pose.position.y
        marker.pose.position.z = 0.45
        marker.pose.orientation = goal.pose.orientation
        marker.scale.z = 0.22
        marker.color.a = 1.0
        colors = {
            "READY": (0.1, 1.0, 0.1),
            "READY_CORRIDOR": (0.1, 1.0, 0.1),
            "READY_TERMINAL": (0.1, 1.0, 0.1),
            "PLANNING": (1.0, 0.75, 0.0),
            "RECEIVED": (1.0, 0.75, 0.0),
            "CANCELED": (0.65, 0.65, 0.65),
        }
        marker.color.r, marker.color.g, marker.color.b = colors.get(
            str(state), (1.0, 0.1, 0.1)
        )
        marker.text = "%s\n%s" % (state, detail)
        self.marker_pub.publish(marker)

    def update(self, _event):
        if not self.plan_lock.acquire(False):
            return
        try:
            self._update()
        finally:
            self.plan_lock.release()

    def _update(self):
        with self.lock:
            grid, odom, goal = self.grid, self.odom, self.goal
            generation, revision = self.goal_generation, self.plan_revision
            cached_goal, cached_status = self.cached_goal, self.cached_status
            cached_plan_mode = self.cached_plan_mode
            terminal_exact_goal = self.terminal_exact_goal
        if grid is None or odom is None or goal is None: return
        start = (odom.pose.pose.position.x, odom.pose.pose.position.y, yaw(odom.pose.pose.orientation))
        raw_target = (
            goal.pose.position.x,
            goal.pose.position.y,
            yaw(goal.pose.orientation),
        )
        target = raw_target
        target_signature = self.goal_signature(*raw_target)
        terminal_distance = float(
            np.hypot(start[0] - target[0], start[1] - target[1])
        )
        terminal_heading_error = abs(
            math.atan2(
                math.sin(target[2] - start[2]),
                math.cos(target[2] - start[2]),
            )
        )
        terminal_exact_required = (
            self.route_mode == "topology_corridor"
            and self.require_goal_heading
            and terminal_distance <= self.terminal_hybrid_radius
            and (
                terminal_heading_error >= self.terminal_hybrid_heading_trigger
                or terminal_distance <= self.terminal_hybrid_capture_radius
                or terminal_exact_goal == target_signature
            )
        )
        start_valid, start_reason, start_clearance, safe_primitives = (
            self.hybrid_validator.validate_start_pose(grid, start)
        )
        start_center_clearance = grid.point_clearance(start[:2])
        start_diagnostics = {
            "start": [float(value) for value in start],
            "start_center_clearance_m": float(start_center_clearance),
            "start_footprint_clearance_m": float(start_clearance),
            "safe_ackermann_primitives": int(safe_primitives),
            "distance_sampling": "BilinearCellCentreDistanceV1",
        }
        if not start_valid:
            state = (
                "START_BLOCKED"
                if start_reason == "START_NO_SAFE_PRIMITIVE"
                else "INVALID_START"
            )
            if cached_goal != target_signature or cached_status != state:
                detail = (
                    "%s center_clearance=%.3fm footprint_clearance=%.3fm "
                    "safe_primitives=%d"
                    % (
                        start_reason,
                        start_center_clearance,
                        start_clearance,
                        safe_primitives,
                    )
                )
                rospy.logwarn(
                    "%s generation=%d start=(%.3f,%.3f,%.3f): %s",
                    state, generation, start[0], start[1], start[2], detail,
                )
                self.publish_status(
                    state, detail, goal, generation, diagnostics=start_diagnostics
                )
                with self.lock:
                    if self.plan_revision == revision:
                        self.cached_goal = target_signature
                        self.cached_poses = None
                        self.cached_status = state
            return
        if self.require_goal_heading:
            valid, validity_reason, goal_clearance = (
                self.hybrid_validator.validate_goal_pose(grid, target)
            )
        else:
            target, valid, validity_reason, goal_clearance = (
                self.position_only_target(
                    grid, start, raw_target[0], raw_target[1]
                )
            )
        if not valid:
            if cached_goal != target_signature or cached_status != "INVALID_GOAL":
                detail = "%s clearance=%.3fm" % (validity_reason, goal_clearance)
                rospy.logwarn(
                    "Rejected goal generation=%d target=(%.3f,%.3f,%.3f): %s",
                    generation,
                    target[0], target[1], target[2], detail,
                )
                self.publish_status(
                    "INVALID_GOAL", detail, goal, generation,
                    diagnostics=start_diagnostics,
                )
                with self.lock:
                    if self.plan_revision == revision:
                        self.cached_goal = target_signature
                        self.cached_poses = None
                        self.cached_status = "INVALID_GOAL"
            return
        with self.lock:
            poses = self.cached_poses if self.cached_goal == target_signature else None
            cached_status = self.cached_status
            plan_mode = cached_plan_mode
        if (
            poses is not None
            and plan_mode == "topology_corridor"
            and terminal_exact_required
        ):
            with self.lock:
                if self.plan_revision == revision:
                    self.cached_poses = None
                    self.cached_plan_mode = None
                    self.cached_status = "PLANNING"
                    self.route_index = 0
                    self.terminal_exact_goal = target_signature
            self.clear_route_visualization()
            self.publish_status(
                "PLANNING",
                "switching from topological corridor to precise kinematic tail",
                goal,
                generation,
                diagnostics=start_diagnostics,
            )
            return
        if poses is None:
            if cached_goal == target_signature and cached_status in ("NO_PATH", "TIMEOUT"):
                return
            plan_mode = self.route_mode
            active_planner = self.planner
            active_timeout = self.planning_timeout
            if terminal_exact_required:
                plan_mode = "hybrid_exact_tail"
                active_planner = self.hybrid_validator
                active_timeout = self.terminal_planning_timeout
                with self.lock:
                    if self.plan_revision == revision:
                        self.terminal_exact_goal = target_signature
            started = time.monotonic()
            rospy.loginfo(
                "Planning goal generation=%d target=(%.3f,%.3f,%.3f) "
                "route_mode=%s timeout=%.1fs",
                generation, target[0], target[1], target[2], plan_mode,
                active_timeout,
            )
            self.publish_status(
                "PLANNING",
                "%s route_mode=%s timeout=%.1fs"
                % (validity_reason, plan_mode, active_timeout),
                goal,
                generation,
                diagnostics=start_diagnostics,
            )

            def cancel_requested():
                with self.lock:
                    stale = self.plan_revision != revision
                return stale or time.monotonic() - started >= active_timeout
            poses = active_planner.plan(
                grid, start, target, cancel_requested=cancel_requested
            )
            elapsed = time.monotonic() - started
            with self.lock:
                stale = self.plan_revision != revision
            if stale:
                rospy.loginfo(
                    "Canceled stale goal generation=%d after %.3fs and %d expansions",
                    generation, elapsed, active_planner.last_expansions,
                )
                # on_goal has already published RECEIVED for the newer goal;
                # do not overwrite its latched status with this stale result.
                return
            status = "TIMEOUT" if elapsed >= active_timeout else "NO_PATH"
            if poses:
                status = (
                    "READY_TERMINAL"
                    if plan_mode == "hybrid_exact_tail"
                    else (
                        "READY_CORRIDOR"
                        if plan_mode == "topology_corridor"
                        else "READY"
                    )
                )
            with self.lock:
                self.cached_poses = poses
                self.cached_plan_mode = plan_mode
                self.cached_goal = target_signature
                self.cached_status = status
                self.route_index = 0
            if not poses:
                detail = "%s (%s, route_mode=%s)" % (
                    status, active_planner.last_status, plan_mode
                )
                rospy.logwarn(
                    "Global plan failed generation=%d target=(%.3f,%.3f,%.3f) "
                    "status=%s elapsed=%.3fs expansions=%d",
                    generation, target[0], target[1], target[2], detail,
                    elapsed, active_planner.last_expansions,
                )
                self.publish_status(
                    status, detail, goal, generation,
                    elapsed=elapsed, expansions=active_planner.last_expansions,
                    diagnostics=start_diagnostics,
                )
                return
            rospy.loginfo(
                "Global plan ready generation=%d elapsed=%.3fs expansions=%d poses=%d",
                generation, elapsed, active_planner.last_expansions, len(poses),
            )
            self.publish_status(
                status,
                (
                    "terminal kinematic route available"
                    if status == "READY_TERMINAL"
                    else (
                        "topological corridor available"
                        if status == "READY_CORRIDOR"
                        else "route available"
                    )
                ),
                goal, generation,
                elapsed=elapsed, expansions=active_planner.last_expansions,
                diagnostics=start_diagnostics,
            )
        if not poses:
            return
        exact_route = plan_mode in ("hybrid_exact", "hybrid_exact_tail")
        with self.lock:
            begin = min(self.route_index, len(poses) - 1)
        position = np.asarray(start[:2])
        if exact_route:
            end = min(len(poses), begin + 20)
            distances = [
                float(np.linalg.norm(np.asarray(pose[:2]) - position))
                for pose in poses[begin:end]
            ]
            nearest_index = begin + int(np.argmin(distances))
            route_distance = distances[nearest_index - begin]
        else:
            nearest_index, route_distance = monotonic_route_index(
                poses, begin, position, grid=grid, maximum_search=20
            )
        replan_deviation = (
            self.exact_replan_deviation
            if exact_route
            else self.corridor_replan_deviation
        )
        if route_distance > replan_deviation:
            rospy.logwarn(
                "Route deviation generation=%d distance=%.3fm threshold=%.3fm "
                "route_index=%d; replanning from measured pose",
                generation,
                route_distance,
                replan_deviation,
                begin,
            )
            with self.lock:
                self.cached_poses = None
                self.cached_status = "PLANNING"
                self.route_index = 0
                if exact_route:
                    self.terminal_exact_goal = target_signature
            self.clear_route_visualization()
            self.publish_status(
                "PLANNING",
                "route deviation %.3fm exceeded %.3fm; replanning from measured pose"
                % (route_distance, replan_deviation),
                goal,
                generation,
                diagnostics=start_diagnostics,
            )
            return
        with self.lock: self.route_index = max(self.route_index, nearest_index)
        remaining = poses[self.route_index:]
        path = Path(); path.header.stamp = rospy.Time.now(); path.header.frame_id = "map"
        route = AckermannRoute(); route.header = path.header
        drive_poses = [pose for pose in remaining[1:]]
        if not drive_poses:
            drive_poses = [remaining[-1]] if remaining else []
        if not drive_poses:
            rospy.logwarn_throttle(3.0, "Global planner returned no corridor segment")
            return
        for pose in poses:
            stamped = PoseStamped(); stamped.header = path.header
            stamped.pose.position.x, stamped.pose.position.y = pose[0], pose[1]
            stamped.pose.orientation.z = math.sin(0.5 * pose[2]); stamped.pose.orientation.w = math.cos(0.5 * pose[2])
            path.poses.append(stamped)
            point = RoutePoint(); point.pose = stamped.pose
            point.gear = int(pose.gear) if exact_route else int(Gear.NEUTRAL)
            point.steering = pose.steering if exact_route else 0.0
            route.points.append(point)
        requested_gear = drive_poses[0].gear
        active_lookahead = (
            min(self.lookahead, self.exact_route_lookahead)
            if exact_route
            else self.lookahead
        )
        travelled = 0.0
        selected = drive_poses[0]
        segment_end = False
        selection_reason = "lookahead"
        hint_pose = drive_poses[0]
        selected_clearance = float("inf")
        if exact_route:
            previous = np.asarray(remaining[0][:2])
            selected_index = self.route_index + 1
            for offset, pose in enumerate(drive_poses, start=1):
                if pose.gear != requested_gear:
                    selection_reason = "gear_boundary"
                    break
                current = np.asarray(pose[:2])
                travelled += float(np.linalg.norm(current - previous))
                previous = current
                selected = pose
                selected_index = self.route_index + offset
                if travelled <= min(0.75, active_lookahead):
                    hint_pose = pose
                if travelled >= active_lookahead:
                    selection_reason = "lookahead"
                    break
            segment_end = selected_index >= len(poses) - 1
        else:
            selected_index, travelled, selected_clearance, selection_reason = (
                visible_corridor_subgoal(
                    poses,
                    self.route_index,
                    position,
                    grid,
                    active_lookahead,
                )
            )
            selected = poses[selected_index]
            segment_end = selected_index >= len(poses) - 1
            hint_travel = 0.0
            previous = np.asarray(poses[self.route_index][:2])
            for pose in poses[self.route_index + 1 : selected_index + 1]:
                current = np.asarray(pose[:2])
                hint_travel += float(np.linalg.norm(current - previous))
                previous = current
                hint_pose = pose
                if hint_travel >= min(0.60, active_lookahead):
                    break
        if not exact_route:
            requested_gear = corridor_gear_hint(start, hint_pose[:2])
            rospy.loginfo_throttle(
                2.0,
                "Corridor tracking generation=%d route_index=%d subgoal_index=%d "
                "arc=%.3fm direct_clearance=%.3fm reason=%s gear=%s",
                generation,
                self.route_index,
                selected_index,
                travelled,
                selected_clearance,
                selection_reason,
                requested_gear.name,
            )
        self.path_pub.publish(path)
        self.route_pub.publish(route)
        subgoal = PoseStamped(); subgoal.header = path.header
        subgoal.pose.position.x, subgoal.pose.position.y = selected[0], selected[1]
        selected_yaw = selected[2]
        if not exact_route and not segment_end and requested_gear == Gear.REVERSE:
            selected_yaw = math.atan2(math.sin(selected_yaw + math.pi), math.cos(selected_yaw + math.pi))
        subgoal.pose.orientation.z = math.sin(0.5 * selected_yaw); subgoal.pose.orientation.w = math.cos(0.5 * selected_yaw)
        self.subgoal_pub.publish(subgoal)
        command = LocalRouteCommand(); command.header = path.header; command.target = subgoal.pose
        command.requested_gear = int(requested_gear); command.segment_index = int(self.route_index); command.segment_end = segment_end
        self.route_command_pub.publish(command)


if __name__ == "__main__":
    rospy.init_node("dep_car_hybrid_astar")
    HybridAStarNode()
    rospy.spin()
