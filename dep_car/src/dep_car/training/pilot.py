"""P3 Pilot task sampling, maneuver labels and reproducible manifests."""

import hashlib
import json
import math
from dataclasses import asdict, dataclass

import numpy as np

from dep_car.core.occupancy import FootprintConfig, OccupancyGrid2D
from dep_car.core.types import Gear


PILOT_MANEUVER_MODES = (
    "NORMAL",
    "SHARP_TURN",
    "NARROW_CORRIDOR",
    "U_TURN",
    "DEAD_END_ESCAPE",
    "REVERSE_EXIT",
    "THREE_POINT_TURN",
)


def wrap_angle(value):
    return math.atan2(math.sin(float(value)), math.cos(float(value)))


def compressed_gears(path):
    output = []
    for pose in path:
        gear = Gear(int(pose.gear))
        if gear == Gear.NEUTRAL:
            continue
        if not output or output[-1] != gear:
            output.append(gear)
    return output


def canonical_sha256(payload):
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def path_sha256(path):
    rows = [
        [round(float(pose.x), 6), round(float(pose.y), 6), round(float(pose.yaw), 6), int(pose.gear), round(float(pose.steering), 6)]
        for pose in path
    ]
    return canonical_sha256(rows)


def probe_clearance(grid, pose, direction=1.0, maximum=2.0, step=0.10):
    distances = np.arange(step, maximum + 0.5 * step, step)
    points = np.column_stack((
        pose[0] + direction * distances * math.cos(pose[2]),
        pose[1] + direction * distances * math.sin(pose[2]),
    ))
    hits = np.flatnonzero(grid.is_occupied(points))
    return maximum if not len(hits) else float(distances[hits[0]])


def classify_maneuver(grid, start, goal, path, narrow_clearance_m=0.52):
    """Classify one planned route from geometry and its signed gear sequence."""

    gears = compressed_gears(path)
    drive_path = [pose for pose in path if pose.gear != Gear.NEUTRAL]
    if not drive_path:
        raise ValueError("route has no drive pose")
    clearances = [grid.point_clearance((pose.x, pose.y)) for pose in drive_path]
    heading_change = abs(wrap_angle(goal[2] - start[2]))
    forward_clearance = probe_clearance(grid, start, direction=1.0)
    reverse_clearance = probe_clearance(grid, start, direction=-1.0)
    gear_switches = max(0, len(gears) - 1)
    initial_reverse = gears[0] == Gear.REVERSE

    if gear_switches >= 2:
        mode = "THREE_POINT_TURN"
    elif initial_reverse and forward_clearance <= 0.70 and reverse_clearance >= 1.0:
        mode = "DEAD_END_ESCAPE"
    elif heading_change >= 2.35:
        mode = "U_TURN"
    elif min(clearances) <= narrow_clearance_m:
        mode = "NARROW_CORRIDOR"
    elif initial_reverse:
        mode = "REVERSE_EXIT"
    elif max(abs(float(pose.steering)) for pose in drive_path) >= 0.40 or heading_change >= 0.80:
        mode = "SHARP_TURN"
    else:
        mode = "NORMAL"
    return mode, {
        "gear_runs": [int(gear) for gear in gears],
        "gear_switches": gear_switches,
        "initial_reverse": initial_reverse,
        "minimum_center_clearance_m": float(min(clearances)),
        "forward_probe_clearance_m": forward_clearance,
        "reverse_probe_clearance_m": reverse_clearance,
        "start_goal_heading_change_rad": heading_change,
        "maximum_route_steering_rad": max(abs(float(pose.steering)) for pose in drive_path),
        "route_pose_count": len(path),
        "route_sha256": path_sha256(path),
    }


@dataclass(frozen=True)
class PilotTask:
    task_id: str
    map_name: str
    map_uuid: str
    map_split: str
    map_occupancy_sha256: str
    map_seed: int
    world: str
    map_yaml: str
    start: tuple
    goal: tuple
    maneuver_mode: str
    task_seed: int
    route_evidence: dict

    def as_dict(self):
        payload = asdict(self)
        payload["start"] = list(self.start)
        payload["goal"] = list(self.goal)
        return payload


class PilotTaskSampler:
    """Generate mode-targeted start/goal proposals on one static map."""

    def __init__(self, grid: OccupancyGrid2D, rng):
        self.grid = grid
        self.rng = rng
        free_y, free_x = np.where(grid.data == 0)
        self.free_points = np.column_stack((
            grid.origin[0] + (free_x + 0.5) * grid.resolution,
            grid.origin[1] + (free_y + 0.5) * grid.resolution,
        ))
        self.free_clearance = grid._distance_field[free_y, free_x]
        self.minimum_pose_clearance = FootprintConfig().circle_radius + 0.06

    def _pose(self, minimum_clearance=None, maximum_clearance=float("inf")):
        minimum = self.minimum_pose_clearance if minimum_clearance is None else minimum_clearance
        eligible = np.flatnonzero((self.free_clearance >= minimum) & (self.free_clearance <= maximum_clearance))
        if not len(eligible):
            raise RuntimeError("map has no free pose in requested clearance band")
        point = self.free_points[int(self.rng.choice(eligible))]
        return float(point[0]), float(point[1]), float(self.rng.uniform(-math.pi, math.pi))

    def _point_pose(self, point, yaw, clearance=None):
        required = self.minimum_pose_clearance if clearance is None else clearance
        if self.grid.point_clearance(point) < required:
            return None
        return float(point[0]), float(point[1]), wrap_angle(yaw)

    def _random_pair(self, start_clearance=0.75, goal_clearance=0.75, minimum_distance=2.5):
        for _ in range(80):
            start = self._pose(start_clearance)
            goal = self._pose(goal_clearance)
            if np.linalg.norm(np.asarray(goal[:2]) - np.asarray(start[:2])) < minimum_distance:
                continue
            bearing = math.atan2(goal[1] - start[1], goal[0] - start[0])
            return (start[0], start[1], bearing), (goal[0], goal[1], bearing)
        raise RuntimeError("could not sample a separated free-space pair")

    def _corridor_pose(self, maximum_clearance=0.62):
        start = self._pose(self.minimum_pose_clearance, maximum_clearance)
        angles = np.linspace(-math.pi, math.pi, 24, endpoint=False)
        scored = []
        for yaw in angles:
            pose = (start[0], start[1], float(yaw))
            forward = probe_clearance(self.grid, pose, 1.0)
            reverse = probe_clearance(self.grid, pose, -1.0)
            scored.append((forward + reverse, min(forward, reverse), pose))
        _, usable, pose = max(scored, key=lambda row: (row[0], row[1]))
        if usable < 0.9:
            raise RuntimeError("narrow point has no usable corridor axis")
        return pose, usable

    def propose(self, mode):
        if mode == "NORMAL":
            start, goal = self._random_pair()
            jitter = float(self.rng.uniform(-0.30, 0.30))
            return (start[0], start[1], wrap_angle(start[2] + jitter)), goal

        if mode == "SHARP_TURN":
            start = self._pose(0.75)
            turn = float(self.rng.choice((-1.0, 1.0))) * float(self.rng.uniform(0.9, 1.5))
            distance = float(self.rng.uniform(2.0, 4.5))
            bearing = start[2] + turn
            goal = self._point_pose((start[0] + distance * math.cos(bearing), start[1] + distance * math.sin(bearing)), bearing, 0.60)
            if goal is None:
                raise RuntimeError("sharp-turn endpoint is occupied")
            return start, goal

        if mode == "U_TURN":
            start = self._pose(1.05)
            distance = float(self.rng.uniform(1.0, 2.0))
            goal = self._point_pose((start[0] + distance * math.cos(start[2]), start[1] + distance * math.sin(start[2])), start[2] + math.pi, 0.85)
            if goal is None:
                raise RuntimeError("U-turn endpoint is occupied")
            return start, goal

        if mode == "REVERSE_EXIT":
            start = self._pose(0.65)
            distance = float(self.rng.uniform(1.2, 2.4))
            goal = self._point_pose((start[0] - distance * math.cos(start[2]), start[1] - distance * math.sin(start[2])), start[2], 0.55)
            if goal is None:
                raise RuntimeError("reverse endpoint is occupied")
            return start, goal

        if mode == "NARROW_CORRIDOR":
            start, usable = self._corridor_pose(0.52)
            distance = min(1.8, usable - 0.15)
            goal = self._point_pose((start[0] + distance * math.cos(start[2]), start[1] + distance * math.sin(start[2])), start[2])
            if goal is None:
                raise RuntimeError("narrow corridor endpoint is occupied")
            return start, goal

        if mode in ("DEAD_END_ESCAPE", "THREE_POINT_TURN"):
            for _ in range(120):
                maximum = 0.70 if mode == "DEAD_END_ESCAPE" else 0.62
                start = self._pose(self.minimum_pose_clearance, maximum)
                if mode == "DEAD_END_ESCAPE":
                    if probe_clearance(self.grid, start, 1.0) > 0.70 or probe_clearance(self.grid, start, -1.0) < 1.0:
                        continue
                    distance, goal_yaw = 1.5, start[2]
                    goal_xy = (start[0] - distance * math.cos(start[2]), start[1] - distance * math.sin(start[2]))
                else:
                    start, usable = self._corridor_pose(maximum)
                    distance, goal_yaw = min(1.2, usable - 0.15), start[2] + math.pi
                    goal_xy = (start[0] + distance * math.cos(start[2]), start[1] + distance * math.sin(start[2]))
                goal = self._point_pose(goal_xy, goal_yaw)
                if goal is not None:
                    return start, goal
            raise RuntimeError("could not construct " + mode)

        raise ValueError("unsupported Pilot maneuver mode: " + str(mode))


def make_pilot_manifest(tasks, *, seed, map_selection, quotas, generator_contract):
    task_rows = [task.as_dict() if isinstance(task, PilotTask) else task for task in tasks]
    payload = {
        "schema": "DEPCarPilotTaskManifestV1",
        "seed": int(seed),
        "map_selection": map_selection,
        "maneuver_quotas": quotas,
        "generator_contract": generator_contract,
        "tasks": task_rows,
    }
    payload["task_manifest_sha256"] = canonical_sha256(payload)
    return payload


def candidate_expressiveness(trajectories, feasible, local_path, guidance_cost, static_clearance):
    """Compute Oracle-of-15 route alignment without using a learned score."""

    trajectories = np.asarray(trajectories, dtype=np.float64)
    feasible = np.asarray(feasible, dtype=bool)
    local_path = np.asarray(local_path, dtype=np.float64)
    guidance_cost = np.asarray(guidance_cost, dtype=np.float64)
    static_clearance = np.asarray(static_clearance, dtype=np.float64)
    if trajectories.ndim != 3 or trajectories.shape[0] != len(feasible) or trajectories.shape[2] < 4:
        raise ValueError("candidate trajectories have an invalid shape")
    if local_path.ndim != 2 or local_path.shape[1] != 3 or not len(local_path):
        raise ValueError("local path is empty or malformed")
    route_steps = np.linalg.norm(np.diff(local_path[:, :2], axis=0), axis=1)
    route_progress = np.concatenate(([0.0], np.cumsum(route_steps)))
    route_costs, progresses = [], []
    for candidate in trajectories:
        endpoint = candidate[-1, 1:3]
        distances = np.linalg.norm(local_path[:, :2] - endpoint, axis=1)
        index = int(np.argmin(distances))
        heading_error = abs(wrap_angle(candidate[-1, 3] - local_path[index, 2]))
        route_costs.append(float(distances[index] + 0.35 * heading_error))
        progresses.append(float(route_progress[index]))
    safe_indices = np.flatnonzero(feasible)
    if not len(safe_indices):
        return {
            "feasible_count": 0, "zero_feasible": True,
            "oracle_route_error_m": float("inf"), "oracle_route_progress_m": 0.0,
            "oracle_guidance_cost": float("inf"), "oracle_static_clearance_m": 0.0,
        }
    best_route = min(safe_indices, key=lambda index: route_costs[index])
    return {
        "feasible_count": int(len(safe_indices)), "zero_feasible": False,
        "oracle_route_error_m": route_costs[best_route],
        "oracle_route_progress_m": progresses[best_route],
        "oracle_guidance_cost": float(np.min(guidance_cost[safe_indices])),
        "oracle_static_clearance_m": float(np.max(static_clearance[safe_indices])),
    }
