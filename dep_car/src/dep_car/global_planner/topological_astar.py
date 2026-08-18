"""Fast 2-D topological corridor planning for online P6 navigation.

This planner deliberately does not command steering or certify Ackermann
motion.  It answers the global question "which connected free-space corridor
reaches the goal without entering a dead end?"  The local planner remains the
authority for drive direction, non-holonomic motion and hard footprint vetoes.
"""

import heapq
from dataclasses import dataclass
from typing import Callable, List, NamedTuple, Optional, Tuple

import numpy as np

from dep_car.core.occupancy import OccupancyGrid2D
from dep_car.core.types import Gear
from dep_car.core.vehicle import DYNAMIC_EGO_RADIUS_M


class CorridorPose(NamedTuple):
    x: float
    y: float
    yaw: float
    gear: Gear = Gear.NEUTRAL
    steering: float = 0.0


@dataclass(frozen=True)
class TopologicalAStarConfig:
    # The corridor uses a circular width envelope.  Full vehicle length and
    # orientation are intentionally left to the continuously replanned local
    # hard-safety layer.
    minimum_center_clearance_m: float = DYNAMIC_EGO_RADIUS_M
    # Keep the corridor near the medial axis when space permits.  This is a
    # soft A* cost, not a stricter traversability gate, so narrow passages
    # remain reachable instead of being declared blocked.
    preferred_center_clearance_m: float = 0.55
    clearance_penalty_weight: float = 2.0
    allow_diagonal: bool = True
    simplify_line_of_sight: bool = False
    maximum_expansions: int = 250000


class TopologicalAStar:
    """Inflated-grid A* producing sparse, non-commanding corridor waypoints."""

    def __init__(self, config: TopologicalAStarConfig = TopologicalAStarConfig()):
        self.config = config
        self.last_expansions = 0
        self.last_status = "IDLE"

    def _traversable(self, grid: OccupancyGrid2D) -> np.ndarray:
        threshold = (
            self.config.minimum_center_clearance_m
            + grid.fixed_grid_allowance()
        )
        return np.asarray(grid._distance_field > threshold, dtype=bool)

    @staticmethod
    def _cell(grid: OccupancyGrid2D, pose) -> Tuple[int, int]:
        value = grid.world_to_cell(np.asarray([[pose[0], pose[1]]]))[0]
        return int(value[0]), int(value[1])

    @staticmethod
    def _world(grid: OccupancyGrid2D, cell) -> Tuple[float, float]:
        return (
            float(grid.origin[0] + (cell[0] + 0.5) * grid.resolution),
            float(grid.origin[1] + (cell[1] + 0.5) * grid.resolution),
        )

    @staticmethod
    def _inside(mask, cell) -> bool:
        return 0 <= cell[0] < mask.shape[1] and 0 <= cell[1] < mask.shape[0]

    @staticmethod
    def _line_cells(first, second):
        """Integer supercover adequate for conservative corridor smoothing."""

        x0, y0 = first
        x1, y1 = second
        count = max(abs(x1 - x0), abs(y1 - y0)) + 1
        if count <= 1:
            return [(x0, y0)]
        xs = np.rint(np.linspace(x0, x1, count)).astype(np.int64)
        ys = np.rint(np.linspace(y0, y1, count)).astype(np.int64)
        return list(dict.fromkeys(zip(xs.tolist(), ys.tolist())))

    def _visible(self, mask, first, second) -> bool:
        cells = self._line_cells(first, second)
        return all(self._inside(mask, cell) and mask[cell[1], cell[0]] for cell in cells)

    def _simplify(self, mask, cells):
        if not self.config.simplify_line_of_sight or len(cells) < 3:
            return cells
        result = [cells[0]]
        anchor = 0
        while anchor < len(cells) - 1:
            furthest = anchor + 1
            for candidate in range(anchor + 2, len(cells)):
                if not self._visible(mask, cells[anchor], cells[candidate]):
                    break
                furthest = candidate
            result.append(cells[furthest])
            anchor = furthest
        return result

    def plan(
        self,
        grid: OccupancyGrid2D,
        start: Tuple[float, float, float],
        goal: Tuple[float, float, float],
        cancel_requested: Optional[Callable[[], bool]] = None,
    ) -> Optional[List[CorridorPose]]:
        self.last_status = "PLANNING"
        self.last_expansions = 0
        mask = self._traversable(grid)
        start_cell, goal_cell = self._cell(grid, start), self._cell(grid, goal)
        if not self._inside(mask, start_cell) or not self._inside(mask, goal_cell):
            self.last_status = "ENDPOINT_BLOCKED"
            return None
        # Full endpoint footprints were validated by the wrapper.  Preserve
        # them even when the width-only inflation falls exactly on a threshold.
        mask[start_cell[1], start_cell[0]] = True
        mask[goal_cell[1], goal_cell[0]] = True

        cardinal = ((1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0))
        diagonal = (
            (1, 1, np.sqrt(2.0)), (1, -1, np.sqrt(2.0)),
            (-1, 1, np.sqrt(2.0)), (-1, -1, np.sqrt(2.0)),
        ) if self.config.allow_diagonal else ()
        queue = [(0.0, 0.0, start_cell)]
        costs = {start_cell: 0.0}
        parents = {start_cell: None}
        tie = 0
        for expansion in range(self.config.maximum_expansions):
            self.last_expansions = expansion + 1
            if cancel_requested is not None and cancel_requested():
                self.last_status = "CANCELED"
                return None
            if not queue:
                self.last_status = "NO_PATH"
                return None
            _, current_cost, current = heapq.heappop(queue)
            if current_cost > costs.get(current, float("inf")):
                continue
            if current == goal_cell:
                cells = []
                node = current
                while node is not None:
                    cells.append(node)
                    node = parents[node]
                cells.reverse()
                cells = self._simplify(mask, cells)
                points = [self._world(grid, cell) for cell in cells]
                points[0] = (float(start[0]), float(start[1]))
                points[-1] = (float(goal[0]), float(goal[1]))
                poses = []
                for index, point in enumerate(points):
                    if index == 0:
                        heading = float(start[2])
                    elif index == len(points) - 1:
                        heading = float(goal[2])
                    else:
                        following = points[index + 1]
                        heading = float(np.arctan2(following[1] - point[1], following[0] - point[0]))
                    poses.append(CorridorPose(point[0], point[1], heading))
                self.last_status = "SUCCESS"
                return poses

            for dx, dy, step_cost in cardinal + diagonal:
                nxt = (current[0] + dx, current[1] + dy)
                if not self._inside(mask, nxt) or not mask[nxt[1], nxt[0]]:
                    continue
                # Do not squeeze diagonally through two touching occupied cells.
                if dx and dy and (
                    not mask[current[1], current[0] + dx]
                    or not mask[current[1] + dy, current[0]]
                ):
                    continue
                clearance = float(grid._distance_field[nxt[1], nxt[0]])
                clearance_deficit = max(
                    0.0,
                    self.config.preferred_center_clearance_m - clearance,
                )
                clearance_penalty = (
                    self.config.clearance_penalty_weight
                    * clearance_deficit
                    / max(self.config.preferred_center_clearance_m, grid.resolution)
                )
                new_cost = current_cost + float(step_cost) * (1.0 + clearance_penalty)
                if new_cost >= costs.get(nxt, float("inf")):
                    continue
                costs[nxt] = new_cost
                parents[nxt] = current
                heuristic = float(np.hypot(nxt[0] - goal_cell[0], nxt[1] - goal_cell[1]))
                tie += 1
                heapq.heappush(queue, (new_cost + heuristic + tie * 1.0e-12, new_cost, nxt))
        self.last_status = "EXPANSION_LIMIT"
        self.last_expansions = self.config.maximum_expansions
        return None


def corridor_gear_hint(
    vehicle_pose: Tuple[float, float, float],
    nearby_waypoint: Tuple[float, float],
) -> Gear:
    """Choose a preferred direction from local corridor geometry only.

    This is a hint to the local two-direction authority, never a hard route
    gear.  Looking at a nearby waypoint avoids a far waypoint on a curved
    corridor incorrectly appearing behind the vehicle.
    """

    dx = float(nearby_waypoint[0]) - float(vehicle_pose[0])
    dy = float(nearby_waypoint[1]) - float(vehicle_pose[1])
    longitudinal = np.cos(vehicle_pose[2]) * dx + np.sin(vehicle_pose[2]) * dy
    return Gear.FORWARD if longitudinal >= 0.0 else Gear.REVERSE
