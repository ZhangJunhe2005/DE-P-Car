#!/usr/bin/env python3
"""Evaluation-only consumer of Gazebo ground truth; never publishes planner inputs."""

import json
import math
from pathlib import Path

import rospy
from dep_car_msgs.msg import PlannerState
from gazebo_msgs.msg import ModelStates
from nav_msgs.msg import Odometry


class Evaluator:
    def __init__(self):
        self.output = Path(rospy.get_param("~output", "data/dynamic_eval/result.json"))
        self.minimum_actor_distance = float("inf")
        self.distance = 0.0; self.last = None; self.states = {}; self.start = rospy.Time.now()
        rospy.Subscriber("/gazebo/model_states", ModelStates, self.on_models, queue_size=1)
        rospy.Subscriber("/base_pose_ground_truth", Odometry, self.on_odom, queue_size=1)
        rospy.Subscriber("/dep_car/planner_state", PlannerState, self.on_state, queue_size=1)
        rospy.on_shutdown(self.write)

    def on_models(self, message):
        if "urban_model" not in message.name: return
        ego = message.pose[message.name.index("urban_model")].position
        for name, pose in zip(message.name, message.pose):
            if name.startswith(("person", "dep_car_actor")):
                self.minimum_actor_distance = min(self.minimum_actor_distance, math.hypot(ego.x - pose.position.x, ego.y - pose.position.y))

    def on_odom(self, message):
        point = (message.pose.pose.position.x, message.pose.pose.position.y)
        if self.last is not None: self.distance += math.hypot(point[0] - self.last[0], point[1] - self.last[1])
        self.last = point

    def on_state(self, message): self.states[message.lifecycle_state] = self.states.get(message.lifecycle_state, 0) + 1
    def write(self):
        self.output.parent.mkdir(parents=True, exist_ok=True)
        payload = {"schema": "DEPCarEvaluationV1", "duration_s": (rospy.Time.now() - self.start).to_sec(), "distance_m": self.distance, "minimum_actor_distance_m": self.minimum_actor_distance, "planner_state_samples": self.states, "ground_truth_consumers": ["dep_car_evaluation"]}
        self.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__": rospy.init_node("dep_car_evaluation"); Evaluator(); rospy.spin()

