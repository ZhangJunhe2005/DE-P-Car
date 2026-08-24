#!/usr/bin/env python3
"""Read-only diagnostics for the online-memory local safety boundary."""

import json
import math

import numpy as np
import rospy
from nav_msgs.msg import OccupancyGrid, Odometry
from sensor_msgs.msg import JointState

from dep_car.core.lattice import AckermannLattice
from dep_car.core.occupancy import FootprintConfig, densify_trajectory_se2
from dep_car.core.types import Gear, VehicleState
from dep_car.core.vehicle import center_steering_from_wheel_angles
from dep_car.runtime.occupancy import RuntimeOccupancyGrid2D


def clearance_profile(grid, trajectory, footprint):
    dense = densify_trajectory_se2(trajectory)
    headings = np.column_stack((np.cos(dense[:, 3]), np.sin(dense[:, 3])))
    centers = (
        dense[:, None, 1:3]
        + headings[:, None, :] * footprint.longitudinal_offsets[None, :, None]
    )
    sampled = grid.sample_distance_field(centers)
    return np.min(sampled, axis=1) - footprint.circle_radius - grid.fixed_grid_allowance()


def main():
    rospy.init_node("dep_car_memory_runtime_diagnostic", anonymous=True)
    costmap = rospy.wait_for_message("/dep_car/local_costmap", OccupancyGrid, timeout=8.0)
    odom = rospy.wait_for_message("/odometry/filtered", Odometry, timeout=8.0)
    joints = rospy.wait_for_message("/urban_model/joint_states", JointState, timeout=8.0)
    data = np.asarray(costmap.data, dtype=np.int16).reshape(
        (costmap.info.height, costmap.info.width)
    )
    grid = RuntimeOccupancyGrid2D(
        data,
        costmap.info.resolution,
        (costmap.info.origin.position.x, costmap.info.origin.position.y),
    )
    positions = dict(zip(joints.name, joints.position))
    steering = center_steering_from_wheel_angles(
        positions.get("front_left_steer_joint", 0.0),
        positions.get("front_right_steer_joint", 0.0),
        True,
    )
    state = VehicleState(
        speed=float(odom.twist.twist.linear.x),
        steering=float(steering),
        yaw_rate=float(odom.twist.twist.angular.z),
    )
    strict_footprint = FootprintConfig()
    physical_footprint = FootprintConfig(safety_margin=0.0)
    lattice = AckermannLattice()
    output = {
        "schema": "DEPCarMemoryRuntimeDiagnosticV1",
        "odometry": {
            "x": float(odom.pose.pose.position.x),
            "y": float(odom.pose.pose.position.y),
            "speed": state.speed,
            "steering": state.steering,
        },
        "grid": {
            "resolution": float(costmap.info.resolution),
            "free": int(np.count_nonzero(data == 0)),
            "occupied": int(np.count_nonzero(data >= 50)),
            "unknown": int(np.count_nonzero(data < 0)),
        },
        "gears": {},
    }
    for gear in (Gear.FORWARD, Gear.REVERSE):
        rows = []
        for scale in (1.0, 0.75, 0.50, 0.35, 0.25):
            for candidate in lattice.generate(
                state, gear=gear, duration_scale=scale
            ):
                strict = clearance_profile(grid, candidate.trajectory, strict_footprint)
                physical = clearance_profile(grid, candidate.trajectory, physical_footprint)
                rows.append(
                    {
                        "scale": scale,
                        "candidate_id": candidate.candidate_id,
                        "strict_start": float(strict[0]),
                        "strict_min": float(np.min(strict)),
                        "strict_end": float(strict[-1]),
                        "physical_min": float(np.min(physical)),
                        "travel": float(
                            math.hypot(
                                candidate.trajectory[-1, 1],
                                candidate.trajectory[-1, 2],
                            )
                        ),
                    }
                )
        output["gears"][gear.name] = {
            "best_strict_end": max(rows, key=lambda row: row["strict_end"]),
            "best_physical_min": max(rows, key=lambda row: row["physical_min"]),
            "physically_safe_improving": sum(
                row["physical_min"] > 0.0
                and row["strict_end"] > row["strict_start"] + 0.02
                for row in rows
            ),
        }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
