#!/usr/bin/env python3
"""Render the pinned upstream Urban xacro and scale the resulting URDF."""

import argparse
import sys

import xacro

from dep_car.core.urdf_scale import configure_ros_depth_camera, configure_sim_lidar, scale_urdf_xml
from dep_car.core.vehicle import URBAN_CAR_LINEAR_SCALE


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--scale", type=float, default=URBAN_CAR_LINEAR_SCALE)
    parser.add_argument("mappings", nargs="*")
    args = parser.parse_args()

    mappings = {}
    for mapping in args.mappings:
        if ":=" not in mapping:
            parser.error(f"invalid xacro mapping: {mapping!r}")
        key, value = mapping.split(":=", 1)
        mappings[key] = value

    rendered = xacro.process_file(args.source, mappings=mappings).toxml()
    configured = configure_ros_depth_camera(rendered)
    configured = configure_sim_lidar(configured)
    sys.stdout.write(scale_urdf_xml(configured, args.scale))


if __name__ == "__main__":
    main()
