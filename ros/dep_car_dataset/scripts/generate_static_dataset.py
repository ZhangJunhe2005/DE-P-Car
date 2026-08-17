#!/usr/bin/env python3
"""Generate a small offline static pilot dataset from arena-tools maps.

The 2-D ray renderer is deliberately marked synthetic; production samples
should be captured from the Urban Car VLP-16 using collect_static_sample.py.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "dep_car/src"))
from dep_car.core.occupancy import OccupancyGrid2D
from dep_car.core.planner import DeterministicPlanner
from dep_car.core.types import VehicleState
from dep_car.global_planner.hybrid_astar import HybridAStar, HybridAStarConfig
from dep_car.training.dataset import save_sample


def load_map(folder):
    metadata = yaml.safe_load((folder / "map.yaml").read_text(encoding="utf-8"))
    image = np.asarray(Image.open(str(folder / metadata["image"])).convert("L"))
    data = np.flipud(np.where(image < 128, 100, 0).astype(np.int16))
    return OccupancyGrid2D(data, metadata["resolution"], tuple(metadata["origin"][:2]))


def local_grid(global_grid, pose, extent=10.0):
    resolution = global_grid.resolution; size = int(round(2 * extent / resolution))
    indices_y, indices_x = np.indices((size, size)); local_x = (indices_x + 0.5) * resolution - extent; local_y = (indices_y + 0.5) * resolution - extent
    cosine, sine = math.cos(pose[2]), math.sin(pose[2])
    world = np.column_stack((pose[0] + cosine * local_x.ravel() - sine * local_y.ravel(), pose[1] + sine * local_x.ravel() + cosine * local_y.ravel()))
    data = np.where(global_grid.is_occupied(world), 100, 0).reshape(size, size).astype(np.int16)
    return OccupancyGrid2D(data, resolution, (-extent, -extent))


def synthetic_range(local, bins=440, channels=16, max_sensor_range=40.0, render_range=10.0):
    angles = np.linspace(-math.pi, math.pi, bins, endpoint=False); distances = np.arange(0.9, render_range, local.resolution)
    ranges = np.full(bins, max_sensor_range, dtype=np.float32); valid = np.zeros(bins, dtype=np.float32)
    for index, angle in enumerate(angles):
        points = np.column_stack((distances * math.cos(angle), distances * math.sin(angle)))
        hits = np.flatnonzero(local.is_occupied(points))
        if len(hits): ranges[index] = distances[hits[0]]; valid[index] = 1.0
    return np.repeat((ranges / max_sensor_range)[None], channels, axis=0), np.repeat(valid[None], channels, axis=0)


def sample_pose(grid, rng):
    free_y, free_x = np.where(grid.data == 0)
    for _ in range(200):
        index = rng.integers(len(free_x)); x = grid.origin[0] + (free_x[index] + 0.5) * grid.resolution; y = grid.origin[1] + (free_y[index] + 0.5) * grid.resolution
        if grid.point_clearance((x, y)) > 0.8: return (x, y, rng.uniform(-math.pi, math.pi))
    raise RuntimeError("map has no pose with Urban Car clearance")


def path_subgoal(path, start, lookahead=4.0):
    travelled = 0.0; previous = np.asarray(path[0][:2]); selected = path[-1]
    for pose in path[1:]:
        current = np.asarray(pose[:2]); travelled += float(np.linalg.norm(current - previous)); previous = current; selected = pose
        if travelled >= lookahead: break
    dx, dy = selected[0] - start[0], selected[1] - start[1]; cosine, sine = math.cos(start[2]), math.sin(start[2])
    return (cosine * dx + sine * dy, -sine * dx + cosine * dy)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--maps", type=Path, default=ROOT / "data/arena_maps"); parser.add_argument("--output", type=Path, default=ROOT / "data/static_raw")
    parser.add_argument("--samples-per-map", type=int, default=4); parser.add_argument("--seed", type=int, default=49001)
    args = parser.parse_args(); rng = np.random.default_rng(args.seed); planner = DeterministicPlanner(); count = 0; failures = 0
    for folder in sorted(path for path in args.maps.iterdir() if (path / "manifest.json").is_file()):
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8")); grid = load_map(folder)
        hybrid = HybridAStar(HybridAStarConfig(maximum_expansions=12000))
        made = 0
        for attempt in range(args.samples_per_map * 20):
            if made >= args.samples_per_map: break
            start, goal = sample_pose(grid, rng), sample_pose(grid, rng)
            path = hybrid.plan(grid, start, goal)
            if not path: failures += 1; continue
            subgoal = path_subgoal(path, start); local = local_grid(grid, start); result = planner.plan(VehicleState(), subgoal, local)
            if not result.executable: failures += 1; continue
            range_image, mask = synthetic_range(local); sample_id = f"{manifest['map_uuid']}-{made:05d}"
            save_sample(args.output / manifest["map_uuid"] / f"{sample_id}.npz", map_uuid=manifest["map_uuid"], range_image=range_image,
                        validity_mask=mask, vehicle_state=[0.0, 0.0, 0.0, 0.0, math.hypot(*subgoal), math.sin(math.atan2(subgoal[1], subgoal[0])), math.cos(math.atan2(subgoal[1], subgoal[0])), 0.0],
                        subgoal_body=subgoal, candidates=result.candidates,
                        metadata={"sample_id": sample_id, "sensor_authority": "synthetic_2d_pilot_not_vlp16", "map_occupancy_sha256": manifest["occupancy_sha256"]})
            made += 1; count += 1
    print(json.dumps({"status": "PASS", "schema": "StaticAckermannSampleV1", "samples": count, "failed_attempts": failures, "authority": "synthetic_2d_pilot_not_vlp16"}, indent=2))


if __name__ == "__main__": main()
