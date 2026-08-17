#!/usr/bin/env python3
"""Plan a car-feasible global path and publish a bounded local subgoal."""

import math
import threading

import numpy as np
import rospy
from dep_car.core.occupancy import OccupancyGrid2D
from dep_car.core.types import Gear
from dep_car.global_planner.hybrid_astar import HybridAStar
from dep_car_msgs.msg import AckermannRoute, LocalRouteCommand, RoutePoint
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid, Odometry, Path


def yaw(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class HybridAStarNode:
    def __init__(self):
        self.lock = threading.Lock()
        self.plan_lock = threading.Lock()
        self.grid = self.odom = self.goal = None
        self.cached_poses = None
        self.cached_goal = None
        self.route_index = 0
        self.planner = HybridAStar()
        self.lookahead = rospy.get_param("~local_subgoal_distance", 4.0)
        self.path_pub = rospy.Publisher("/dep_car/global_path", Path, queue_size=1, latch=True)
        self.route_pub = rospy.Publisher("/dep_car/global_route", AckermannRoute, queue_size=1, latch=True)
        self.subgoal_pub = rospy.Publisher("/dep_car/local_subgoal", PoseStamped, queue_size=1, latch=True)
        self.route_command_pub = rospy.Publisher("/dep_car/local_route_command", LocalRouteCommand, queue_size=1, latch=True)
        rospy.Subscriber("/map", OccupancyGrid, self.on_grid, queue_size=1)
        rospy.Subscriber("/base_pose_ground_truth", Odometry, self.on_odom, queue_size=1)
        rospy.Subscriber("/move_base_simple/goal", PoseStamped, self.on_goal, queue_size=1)
        self.timer = rospy.Timer(rospy.Duration(1.0), self.update)

    def on_grid(self, message):
        data = np.asarray(message.data, dtype=np.int16).reshape(message.info.height, message.info.width)
        with self.lock:
            self.grid = OccupancyGrid2D(data, message.info.resolution, (message.info.origin.position.x, message.info.origin.position.y))
            self.cached_poses = None

    def on_odom(self, message):
        with self.lock: self.odom = message

    def on_goal(self, message):
        with self.lock:
            self.goal = message
            self.cached_poses = None

    def update(self, _event):
        if not self.plan_lock.acquire(False):
            return
        try:
            self._update()
        finally:
            self.plan_lock.release()

    def _update(self):
        with self.lock: grid, odom, goal = self.grid, self.odom, self.goal
        if grid is None or odom is None or goal is None: return
        start = (odom.pose.pose.position.x, odom.pose.pose.position.y, yaw(odom.pose.pose.orientation))
        target = (goal.pose.position.x, goal.pose.position.y, yaw(goal.pose.orientation))
        target_signature = tuple(round(value, 4) for value in target)
        with self.lock:
            poses = self.cached_poses if self.cached_goal == target_signature else None
        if poses is None:
            poses = self.planner.plan(grid, start, target)
            with self.lock:
                self.cached_poses = poses
                self.cached_goal = target_signature
                self.route_index = 0
        if not poses:
            rospy.logwarn_throttle(3.0, "Hybrid A* could not find a path")
            return
        with self.lock:
            begin = min(self.route_index, len(poses) - 1)
        end = min(len(poses), begin + 20)
        position = np.asarray(start[:2])
        distances = [float(np.linalg.norm(np.asarray(pose[:2]) - position)) for pose in poses[begin:end]]
        nearest_index = begin + int(np.argmin(distances))
        if distances[nearest_index - begin] > 0.75:
            with self.lock: self.cached_poses = None
            return
        with self.lock: self.route_index = max(self.route_index, nearest_index)
        remaining = poses[self.route_index:]
        path = Path(); path.header.stamp = rospy.Time.now(); path.header.frame_id = "map"
        route = AckermannRoute(); route.header = path.header
        drive_poses = [pose for pose in remaining if pose.gear != Gear.NEUTRAL]
        if not drive_poses:
            rospy.logwarn_throttle(3.0, "Hybrid A* returned no drive segment")
            return
        for pose in poses:
            stamped = PoseStamped(); stamped.header = path.header
            stamped.pose.position.x, stamped.pose.position.y = pose[0], pose[1]
            stamped.pose.orientation.z = math.sin(0.5 * pose[2]); stamped.pose.orientation.w = math.cos(0.5 * pose[2])
            path.poses.append(stamped)
            point = RoutePoint(); point.pose = stamped.pose; point.gear = int(pose.gear); point.steering = pose.steering
            route.points.append(point)
        requested_gear = drive_poses[0].gear
        travelled = 0.0; selected = drive_poses[0]; segment_end = True
        previous = np.asarray(remaining[0][:2])
        for pose in drive_poses:
            if pose.gear != requested_gear:
                break
            current = np.asarray(pose[:2]); travelled += float(np.linalg.norm(current - previous)); previous = current
            selected = pose
            if travelled >= self.lookahead:
                segment_end = False
                break
        self.path_pub.publish(path)
        self.route_pub.publish(route)
        subgoal = PoseStamped(); subgoal.header = path.header
        subgoal.pose.position.x, subgoal.pose.position.y = selected[0], selected[1]
        subgoal.pose.orientation.z = math.sin(0.5 * selected[2]); subgoal.pose.orientation.w = math.cos(0.5 * selected[2])
        self.subgoal_pub.publish(subgoal)
        command = LocalRouteCommand(); command.header = path.header; command.target = subgoal.pose
        command.requested_gear = int(requested_gear); command.segment_index = 0; command.segment_end = segment_end
        self.route_command_pub.publish(command)


if __name__ == "__main__":
    rospy.init_node("dep_car_hybrid_astar")
    HybridAStarNode()
    rospy.spin()
