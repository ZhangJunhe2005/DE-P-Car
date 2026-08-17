#!/usr/bin/env python3
"""Capture a synchronized planner sample on explicit service requests."""

import math
import uuid
from pathlib import Path

import numpy as np
import rospy
from dep_car.core.types import Candidate
from dep_car.core.vehicle import world_velocity_to_body_longitudinal
from dep_car.training.dataset import save_sample
from dep_car_msgs.msg import CandidateArray
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from std_srvs.srv import Trigger, TriggerResponse


class Collector:
    def __init__(self):
        self.output = Path(rospy.get_param("~output", "data/static_raw"))
        self.map_uuid = rospy.get_param("~map_uuid")
        self.range_image = self.mask = self.candidates = self.odom = self.subgoal = None
        rospy.Subscriber("/dep_car/lidar/range_image", Image, self.on_range, queue_size=1)
        rospy.Subscriber("/dep_car/lidar/validity_mask", Image, self.on_mask, queue_size=1)
        rospy.Subscriber("/dep_car/candidates", CandidateArray, self.on_candidates, queue_size=1)
        rospy.Subscriber("/base_pose_ground_truth", Odometry, self.on_odom, queue_size=1)
        rospy.Subscriber("/dep_car/local_subgoal", PoseStamped, self.on_subgoal, queue_size=1)
        rospy.Service("~capture", Trigger, self.capture)

    @staticmethod
    def decode(message): return np.frombuffer(message.data, dtype=np.float32).reshape(message.height, message.width).copy()
    def on_range(self, message): self.range_image = self.decode(message)
    def on_mask(self, message): self.mask = self.decode(message)
    def on_candidates(self, message): self.candidates = message
    def on_odom(self, message): self.odom = message
    def on_subgoal(self, message): self.subgoal = message
    def capture(self, _request):
        if any(item is None for item in (self.range_image, self.mask, self.candidates, self.odom, self.subgoal)):
            return TriggerResponse(False, "waiting for range, mask, candidates, odometry and subgoal")
        position = self.odom.pose.pose.position
        orientation = self.odom.pose.pose.orientation
        yaw = math.atan2(2.0 * (orientation.w * orientation.z + orientation.x * orientation.y), 1.0 - 2.0 * (orientation.y ** 2 + orientation.z ** 2))
        dx = self.subgoal.pose.position.x - position.x; dy = self.subgoal.pose.position.y - position.y
        subgoal = (math.cos(yaw) * dx + math.sin(yaw) * dy, -math.sin(yaw) * dx + math.cos(yaw) * dy)
        core_candidates = []
        for message in self.candidates.candidates:
            count = len(message.path.poses)
            trajectory = np.zeros((count, 6), dtype=np.float64)
            trajectory[:, 0] = np.linspace(0.0, message.duration, count)
            for index, pose in enumerate(message.path.poses):
                q = pose.pose.orientation
                trajectory[index, 1] = pose.pose.position.x; trajectory[index, 2] = pose.pose.position.y
                trajectory[index, 3] = math.atan2(2.0 * q.w * q.z, 1.0 - 2.0 * q.z * q.z)
                trajectory[index, 4] = message.speed_anchor; trajectory[index, 5] = message.steering_anchor
            core_candidates.append(Candidate(
                message.candidate_id, message.speed_anchor, message.steering_anchor, message.duration, trajectory,
                retime_factor=message.retime_factor, learned_score=message.learned_score,
                guidance_cost=message.guidance_cost, static_clearance=message.static_clearance,
                dynamic_clearance=message.dynamic_clearance, feasible=message.feasible, veto_reason=message.veto_reason,
            ))
        velocity = self.odom.twist.twist.linear
        speed = world_velocity_to_body_longitudinal(velocity.x, velocity.y, yaw)
        yaw_rate = self.odom.twist.twist.angular.z
        bearing = math.atan2(subgoal[1], subgoal[0])
        vehicle_state = [speed, 0.0, 0.0, yaw_rate, math.hypot(*subgoal), math.sin(bearing), math.cos(bearing), 0.0]
        sample_id = str(uuid.uuid4())
        path = self.output / self.map_uuid / f"{sample_id}.npz"
        manifest = save_sample(path, map_uuid=self.map_uuid, range_image=self.range_image, validity_mask=self.mask,
                               vehicle_state=vehicle_state, subgoal_body=subgoal, candidates=core_candidates,
                               metadata={"sample_id": sample_id, "planning_generation": self.candidates.planning_generation})
        return TriggerResponse(True, f"saved {path} split={manifest['split']}")


if __name__ == "__main__": rospy.init_node("dep_car_static_collector"); Collector(); rospy.spin()
