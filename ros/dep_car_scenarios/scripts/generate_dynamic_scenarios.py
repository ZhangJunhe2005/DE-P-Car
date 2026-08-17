#!/usr/bin/env python3
"""Generate deterministic Arena scenario JSON; GT remains evaluation-only."""

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def agent(agent_id, waypoints, speed):
    return {
        "name": f"dep_car_actor_{agent_id}", "id": agent_id, "pos": waypoints[0], "type": "adult",
        "yaml_file": str(PROJECT_ROOT / "third_party/arena-rosnav-3D/simulator_setup/dynamic_obstacles/person_two_legged.model.yaml"),
        "number_of_peds": 1, "vmax": speed, "start_up_mode": "default", "wait_time": 0.0,
        "trigger_zone_radius": 0.0, "chatting_probability": 0.0, "tell_story_probability": 0.0,
        "group_talking_probability": 0.0, "talking_and_walking_probability": 0.0,
        "requesting_service_probability": 0.0, "requesting_guide_probability": 0.0,
        "requesting_follower_probability": 0.0, "max_talking_distance": 5.0, "max_servicing_radius": 5.0,
        "talking_base_time": 10.0, "tell_story_base_time": 0.0, "group_talking_base_time": 10.0,
        "talking_and_walking_base_time": 6.0, "receiving_service_base_time": 20.0,
        "requesting_service_base_time": 30.0, "force_factor_desired": 1.0,
        "force_factor_obstacle": 1.0, "force_factor_social": 5.0, "force_factor_robot": 0.0,
        "waypoints": waypoints, "waypoint_mode": 0,
    }


SCENARIOS = {
    "crossing": [agent(0, [[0.0, -4.0], [0.0, 4.0]], 1.0)],
    "head_on": [agent(0, [[6.0, 0.0], [-6.0, 0.0]], 0.8)],
    "multi_agent": [
        agent(0, [[-2.0, -5.0], [-2.0, 5.0]], 0.8), agent(1, [[2.0, 5.0], [2.0, -5.0]], 1.0),
        agent(2, [[-5.0, 2.0], [5.0, 2.0]], 0.7), agent(3, [[5.0, -2.0], [-5.0, -2.0]], 0.9),
    ],
}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data/dynamic_eval/scenarios"); parser.add_argument("--map", default="")
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    for name, agents in SCENARIOS.items():
        document = {"pedsim_agents": agents, "static_obstacles": [], "robot_position": [-6.0, 0.0], "robot_goal": [6.0, 0.0], "map_path": args.map, "format": "arena-tools"}
        (args.output / f"{name}.json").write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "scenarios": sorted(SCENARIOS)}))


if __name__ == "__main__": main()

