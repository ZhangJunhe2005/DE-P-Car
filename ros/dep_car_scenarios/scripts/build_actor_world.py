#!/usr/bin/env python3
"""Offline Arena scenario-to-Gazebo actor injector for reproducible tests."""

import argparse
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path


def add_actor(world, agent, actor_mesh_uri, collision_plugin_xml=None):
    waypoints = agent["waypoints"]
    actor = ET.SubElement(world, "actor", name=agent["name"])
    ET.SubElement(actor, "pose").text = f"{waypoints[0][0]} {waypoints[0][1]} 0 0 0 0"
    skin = ET.SubElement(actor, "skin"); ET.SubElement(skin, "filename").text = actor_mesh_uri + "/SKIN_man_green_shirt.dae"; ET.SubElement(skin, "scale").text = "1"
    animation = ET.SubElement(actor, "animation", name="walking"); ET.SubElement(animation, "filename").text = actor_mesh_uri + "/ANIMATION_walking.dae"; ET.SubElement(animation, "interpolate_x").text = "true"
    if collision_plugin_xml:
        actor.append(ET.fromstring(collision_plugin_xml))
    script = ET.SubElement(actor, "script"); ET.SubElement(script, "loop").text = "true"; ET.SubElement(script, "auto_start").text = "true"
    trajectory = ET.SubElement(script, "trajectory", id=str(agent["id"]), type="walking")
    time_s = 0.0
    loop = waypoints + [waypoints[0]]
    for start, end in zip(loop, loop[1:]):
        yaw = math.atan2(end[1] - start[1], end[0] - start[0]); waypoint = ET.SubElement(trajectory, "waypoint")
        ET.SubElement(waypoint, "time").text = f"{time_s:.6f}"; ET.SubElement(waypoint, "pose").text = f"{start[0]} {start[1]} 0 0 0 {yaw}"
        time_s += math.hypot(end[0] - start[0], end[1] - start[1]) / agent["vmax"]
    final = ET.SubElement(trajectory, "waypoint"); ET.SubElement(final, "time").text = f"{time_s:.6f}"
    ET.SubElement(final, "pose").text = f"{loop[-1][0]} {loop[-1][1]} 0 0 0 0"
    return actor


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("base_world", type=Path); parser.add_argument("scenario", type=Path); parser.add_argument("output", type=Path)
    parser.add_argument("--actor-mesh-uri", default="model://actor/meshes")
    parser.add_argument("--collision-plugin-sdf", type=Path, default=Path(__file__).resolve().parents[3] / "third_party/arena-rosnav-3D/simulator_setup/obstacles/utils/collision-actor-plugin")
    args = parser.parse_args(); tree = ET.parse(str(args.base_world)); world = tree.getroot().find("world"); scenario = json.loads(args.scenario.read_text(encoding="utf-8"))
    collision_xml = args.collision_plugin_sdf.read_text(encoding="utf-8") if args.collision_plugin_sdf.is_file() else None
    for agent in scenario["pedsim_agents"]: add_actor(world, agent, args.actor_mesh_uri, collision_xml)
    args.output.parent.mkdir(parents=True, exist_ok=True); tree.write(str(args.output), encoding="unicode", xml_declaration=True)


if __name__ == "__main__": main()
