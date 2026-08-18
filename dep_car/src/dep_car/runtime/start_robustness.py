"""P6 start-pose perturbation audit shared by freezing and runtime selection."""

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from dep_car.core.occupancy import OccupancyGrid2D
from dep_car.global_planner.hybrid_astar import HybridAStar


START_ROBUSTNESS_SCHEMA = "DEPCarP6StartRobustnessV1"


@dataclass(frozen=True)
class StartRobustnessConfig:
    position_delta_m: float = 0.02
    yaw_delta_rad: float = 0.035


def audit_start_robustness(
    grid: OccupancyGrid2D,
    start: Tuple[float, float, float],
    config: StartRobustnessConfig = StartRobustnessConfig(),
) -> Dict[str, object]:
    """Test a frozen start against 27 position/yaw perturbations.

    Gazebo spawn and P3D odometry can differ by millimetres.  Requiring every
    cross-product perturbation to retain a valid footprint and at least one
    short Ackermann primitive prevents a scenario from being valid only on one
    side of an occupancy-cell boundary.
    """

    planner = HybridAStar()
    deltas_xy = (-config.position_delta_m, 0.0, config.position_delta_m)
    deltas_yaw = (-config.yaw_delta_rad, 0.0, config.yaw_delta_rad)
    failures = []
    minimum_center_clearance = float("inf")
    minimum_footprint_clearance = float("inf")
    minimum_safe_primitives = 10
    count = 0
    for dx in deltas_xy:
        for dy in deltas_xy:
            for dyaw in deltas_yaw:
                perturbed = (
                    float(start[0]) + dx,
                    float(start[1]) + dy,
                    float(start[2]) + dyaw,
                )
                valid, reason, footprint_clearance, safe_primitives = (
                    planner.validate_start_pose(grid, perturbed)
                )
                center_clearance = grid.point_clearance(perturbed[:2])
                minimum_center_clearance = min(
                    minimum_center_clearance, float(center_clearance)
                )
                minimum_footprint_clearance = min(
                    minimum_footprint_clearance, float(footprint_clearance)
                )
                minimum_safe_primitives = min(
                    minimum_safe_primitives, int(safe_primitives)
                )
                count += 1
                if not valid:
                    failures.append({
                        "delta": [float(dx), float(dy), float(dyaw)],
                        "reason": str(reason),
                        "center_clearance_m": float(center_clearance),
                        "footprint_clearance_m": float(footprint_clearance),
                        "safe_ackermann_primitives": int(safe_primitives),
                    })
    return {
        "schema": START_ROBUSTNESS_SCHEMA,
        "status": "PASS" if not failures else "FAIL",
        "position_delta_m": float(config.position_delta_m),
        "yaw_delta_rad": float(config.yaw_delta_rad),
        "perturbations": int(count),
        "minimum_center_clearance_m": float(minimum_center_clearance),
        "minimum_footprint_clearance_m": float(minimum_footprint_clearance),
        "minimum_safe_ackermann_primitives": int(minimum_safe_primitives),
        "failure_count": len(failures),
        "failures": failures,
    }
