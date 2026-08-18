"""Compact deterministic Hybrid A* suitable for training-label guidance."""

import heapq
from dataclasses import dataclass
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple

import numpy as np

from dep_car.core.occupancy import FootprintConfig, OccupancyGrid2D
from dep_car.core.vehicle import PLANNER_ROLLOUT_WHEELBASE_M, URBAN_CAR_LINEAR_SCALE
from dep_car.core.types import Gear


class HybridPathPose(NamedTuple):
    x: float
    y: float
    yaw: float
    gear: Gear
    steering: float


@dataclass(frozen=True)
class HybridAStarConfig:
    wheelbase: float = PLANNER_ROLLOUT_WHEELBASE_M
    step_length: float = 0.25
    steering_values: Tuple[float, ...] = (-0.52, -0.26, 0.0, 0.26, 0.52)
    heading_bins: int = 48
    reverse_penalty: float = 2.0
    gear_switch_penalty: float = 0.75
    steering_penalty: float = 0.15
    steering_change_penalty: float = 0.10
    goal_tolerance: float = 0.55 * URBAN_CAR_LINEAR_SCALE
    goal_heading_tolerance: float = 0.35
    collision_sample_step: float = 0.04
    heading_heuristic_weight: float = 1.0
    maximum_expansions: int = 60000


class HybridAStar:
    def __init__(self, config: HybridAStarConfig = HybridAStarConfig()):
        self.config = config
        self.last_expansions = 0
        self.last_status = "IDLE"

    @staticmethod
    def _validate_static_pose(
        grid: OccupancyGrid2D,
        pose_values: Tuple[float, float, float],
        label: str,
        footprint: FootprintConfig = FootprintConfig(),
    ) -> Tuple[bool, str, float]:
        values = np.asarray(pose_values, dtype=np.float64)
        prefix = str(label).upper()
        if values.shape != (3,) or not np.all(np.isfinite(values)):
            return False, prefix + "_NON_FINITE", 0.0
        cell = grid.world_to_cell(values[None, :2])[0]
        if (
            cell[0] < 0
            or cell[1] < 0
            or cell[0] >= grid.data.shape[1]
            or cell[1] >= grid.data.shape[0]
        ):
            return False, prefix + "_OUTSIDE_MAP", 0.0
        if bool(grid.is_occupied(values[None, :2])[0]):
            return False, prefix + "_CELL_OCCUPIED", 0.0
        pose = np.asarray([[0.0, values[0], values[1], values[2], 0.0, 0.0]])
        safe, clearance = grid.swept_footprint_clearance(pose, footprint)
        if not safe:
            return False, prefix + "_FOOTPRINT_COLLISION", float(clearance)
        return True, prefix + "_VALID", float(clearance)

    @staticmethod
    def validate_goal_pose(
        grid: OccupancyGrid2D,
        goal: Tuple[float, float, float],
        footprint: FootprintConfig = FootprintConfig(),
    ) -> Tuple[bool, str, float]:
        """Check only whether the static endpoint can contain the vehicle.

        This deliberately does not certify a route or replace the local
        planner.  It avoids spending a full search on an endpoint outside the
        map, inside an occupied cell or too tight for the frozen footprint.
        """

        return HybridAStar._validate_static_pose(grid, goal, "GOAL", footprint)

    def validate_start_pose(
        self,
        grid: OccupancyGrid2D,
        start: Tuple[float, float, float],
        footprint: FootprintConfig = FootprintConfig(),
    ) -> Tuple[bool, str, float, int]:
        """Validate the measured start and count immediately safe controls.

        A valid stationary footprint with no complete 0.25 m Ackermann
        primitive is reported separately as ``START_BLOCKED`` by the ROS
        wrapper.  This makes spawn/odometry faults distinguishable from a
        legitimate but locally immobile pose.
        """

        valid, reason, clearance = self._validate_static_pose(
            grid, start, "START", footprint
        )
        if not valid:
            return False, reason, clearance, 0
        safe_primitives = 0
        for direction in (1.0, -1.0):
            for steering in self.config.steering_values:
                trajectory = self._primitive_trajectory(start, steering, direction)
                safe, _ = grid.swept_footprint_clearance(trajectory, footprint)
                safe_primitives += int(safe)
        if safe_primitives == 0:
            return False, "START_NO_SAFE_PRIMITIVE", clearance, 0
        return True, "START_VALID", clearance, safe_primitives

    def _key(self, pose, grid, gear=Gear.NEUTRAL):
        cell = grid.world_to_cell(np.asarray([[pose[0], pose[1]]]))[0]
        heading = int(np.floor((pose[2] + np.pi) / (2 * np.pi) * self.config.heading_bins)) % self.config.heading_bins
        return int(cell[0]), int(cell[1]), heading, int(gear)

    def _primitive(self, pose, steering, direction):
        distance = direction * self.config.step_length
        yaw_delta = distance * np.tan(steering) / self.config.wheelbase
        mid_yaw = pose[2] + 0.5 * yaw_delta
        return (
            pose[0] + distance * np.cos(mid_yaw),
            pose[1] + distance * np.sin(mid_yaw),
            float(np.arctan2(np.sin(pose[2] + yaw_delta), np.cos(pose[2] + yaw_delta))),
        )

    def _primitive_trajectory(self, pose, steering, direction):
        count = max(2, int(np.ceil(self.config.step_length / self.config.collision_sample_step)) + 1)
        distances = np.linspace(0.0, direction * self.config.step_length, count)
        trajectory = np.zeros((count, 6), dtype=np.float64)
        for index, distance in enumerate(distances):
            yaw_delta = distance * np.tan(steering) / self.config.wheelbase
            mid_yaw = pose[2] + 0.5 * yaw_delta
            trajectory[index] = (
                index,
                pose[0] + distance * np.cos(mid_yaw),
                pose[1] + distance * np.sin(mid_yaw),
                pose[2] + yaw_delta,
                direction,
                steering,
            )
        return trajectory

    @staticmethod
    def _heading_error(first, second):
        return abs(float(np.arctan2(np.sin(first - second), np.cos(first - second))))

    def plan(
        self,
        grid: OccupancyGrid2D,
        start: Tuple[float, float, float],
        goal: Tuple[float, float, float],
        footprint: FootprintConfig = FootprintConfig(),
        cancel_requested: Optional[Callable[[], bool]] = None,
    ) -> Optional[List[Tuple[float, float, float]]]:
        self.last_expansions = 0
        self.last_status = "PLANNING"
        start_key = self._key(start, grid, Gear.NEUTRAL)
        queue = [(float(np.hypot(start[0] - goal[0], start[1] - goal[1])), 0.0, start_key, start)]
        costs: Dict[Tuple[int, int, int], float] = {start_key: 0.0}
        parents = {start_key: None}
        poses = {start_key: HybridPathPose(*start, Gear.NEUTRAL, 0.0)}
        tie = 0

        for expansion in range(self.config.maximum_expansions):
            self.last_expansions = expansion
            if cancel_requested is not None and cancel_requested():
                self.last_status = "CANCELED"
                return None
            if not queue:
                self.last_status = "NO_PATH"
                return None
            _, current_cost, current_key, current = heapq.heappop(queue)
            if current_cost > costs.get(current_key, float("inf")):
                continue
            if (
                np.hypot(current[0] - goal[0], current[1] - goal[1]) <= self.config.goal_tolerance
                and self._heading_error(current[2], goal[2]) <= self.config.goal_heading_tolerance
            ):
                path = []
                key = current_key
                while key is not None:
                    path.append(poses[key])
                    key = parents[key]
                self.last_expansions = expansion + 1
                self.last_status = "SUCCESS"
                return list(reversed(path))
            for direction in (1.0, -1.0):
                gear = Gear.FORWARD if direction > 0 else Gear.REVERSE
                for steering in self.config.steering_values:
                    nxt = self._primitive(current, steering, direction)
                    trajectory = self._primitive_trajectory(current, steering, direction)
                    safe, _ = grid.swept_footprint_clearance(trajectory, footprint)
                    if not safe:
                        continue
                    key = self._key(nxt, grid, gear)
                    step_cost = self.config.step_length * (1.0 if direction > 0 else self.config.reverse_penalty)
                    step_cost += self.config.steering_penalty * abs(steering)
                    previous_gear = Gear(int(current_key[3]))
                    if previous_gear != Gear.NEUTRAL and previous_gear != gear:
                        step_cost += self.config.gear_switch_penalty
                    step_cost += self.config.steering_change_penalty * abs(steering - poses[current_key].steering)
                    new_cost = current_cost + step_cost
                    if new_cost >= costs.get(key, float("inf")):
                        continue
                    costs[key] = new_cost
                    parents[key] = current_key
                    poses[key] = HybridPathPose(*nxt, gear, float(steering))
                    heuristic = float(np.hypot(nxt[0] - goal[0], nxt[1] - goal[1]))
                    heuristic += self.config.heading_heuristic_weight * self._heading_error(nxt[2], goal[2])
                    tie += 1
                    heapq.heappush(queue, (new_cost + heuristic + tie * 1e-12, new_cost, key, nxt))
        self.last_expansions = self.config.maximum_expansions
        self.last_status = "EXPANSION_LIMIT"
        return None
