#!/usr/bin/env python3
"""Headless deterministic wrapper around the pinned arena-tools generator."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import sys
import uuid
from pathlib import Path

import numpy as np
import yaml
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ARENA_TOOLS = PROJECT_ROOT / "third_party" / "arena-tools"
sys.path.insert(0, str(ARENA_TOOLS))
from MapGenerator import MapGenerator  # noqa: E402


def occupied_runs(row):
    start = None
    for index, occupied in enumerate(np.append(row.astype(bool), False)):
        if occupied and start is None:
            start = index
        elif not occupied and start is not None:
            yield start, index
            start = None


def model_sdf(name, occupancy, resolution, origin, wall_height=1.8):
    links = []
    height, width = occupancy.shape
    link_id = 0
    for row_index, row in enumerate(occupancy):
        for start, stop in occupied_runs(row):
            run_length = (stop - start) * resolution
            x = origin[0] + (start + (stop - start) / 2.0) * resolution
            y = origin[1] + (height - row_index - 0.5) * resolution
            geometry = f"<geometry><box><size>{run_length:.6f} {resolution:.6f} {wall_height:.6f}</size></box></geometry>"
            links.append(
                f"<link name='wall_{link_id}'><pose>{x:.6f} {y:.6f} {wall_height/2:.6f} 0 0 0</pose>"
                f"<collision name='collision'>{geometry}</collision>"
                f"<visual name='visual'>{geometry}<material><ambient>0.55 0.55 0.58 1</ambient></material></visual></link>"
            )
            link_id += 1
    return "<?xml version='1.0'?><sdf version='1.6'><model name='%s'><static>true</static>%s</model></sdf>" % (name, "".join(links))


def world_sdf(name):
    return f"""<?xml version='1.0'?>
<sdf version='1.6'><world name='default'>
  <include><uri>model://sun</uri></include>
  <include><uri>model://ground_plane</uri></include>
  <include><uri>model://{name}</uri></include>
  <physics name='default_physics' type='ode'><max_step_size>0.002</max_step_size><real_time_update_rate>500</real_time_update_rate></physics>
</world></sdf>
"""


def generate_one(output_root, map_type, seed, index, width, height, resolution, corridor_radius, iterations, obstacles, obstacle_radius, resume=False):
    np.random.seed(seed)
    generator = MapGenerator.__new__(MapGenerator)
    if map_type == "indoor":
        occupancy = generator.create_indoor_map(height, width, corridor_radius, iterations)
    else:
        occupancy = generator.create_outdoor_map(height, width, obstacles, obstacle_radius)
    identity = f"arena-tools:{map_type}:{seed}:{width}:{height}:{resolution}:{corridor_radius}:{iterations}:{obstacles}:{obstacle_radius}"
    map_uuid = str(uuid.uuid5(uuid.NAMESPACE_URL, identity))
    name = f"dep_car_map_{index:04d}_{map_uuid[:8]}"
    folder = output_root / name
    if folder.exists() and resume:
        manifest_path = folder / "manifest.json"
        required_files = ("manifest.json", "map.png", "map.yaml", "map.world", "model.sdf", "model.config")
        missing = [filename for filename in required_files if not (folder / filename).is_file()]
        if missing:
            raise RuntimeError("refusing incomplete existing map folder: " + str(folder))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = {
            "schema": "DEPCarArenaStaticMapV1", "name": name, "map_uuid": map_uuid,
            "seed": seed, "type": map_type,
            "arena_tools_commit": "664b950d88c91b34e3cdce62512c8864b574b9d4",
        }
        if any(manifest.get(key) != value for key, value in expected.items()):
            raise RuntimeError("existing map provenance differs: " + str(folder))
        saved_image = np.asarray(Image.open(str(folder / "map.png")).convert("L"), dtype=np.uint8)
        saved_digest = hashlib.sha256(saved_image.tobytes()).hexdigest()
        if saved_image.shape != occupancy.shape or manifest.get("occupancy_sha256") != saved_digest:
            raise RuntimeError("existing map occupancy hash differs: " + str(folder))
        manifest["reused"] = True
        return manifest
    folder.mkdir(parents=True, exist_ok=False)
    image = ((1 - occupancy) * 255).astype(np.uint8)
    Image.fromarray(image, mode="L").save(str(folder / "map.png"))
    origin = [-width * resolution / 2.0, -height * resolution / 2.0, 0.0]
    map_yaml = {"image": "map.png", "resolution": resolution, "origin": origin, "negate": 0, "occupied_thresh": 0.65, "free_thresh": 0.196}
    (folder / "map.yaml").write_text(yaml.safe_dump(map_yaml, sort_keys=False), encoding="utf-8")
    sdf = model_sdf(name, occupancy, resolution, origin)
    (folder / "model.sdf").write_text(sdf, encoding="utf-8")
    (folder / "model.config").write_text(
        f"<?xml version='1.0'?><model><name>{name}</name><version>1.0</version><sdf version='1.6'>model.sdf</sdf></model>", encoding="utf-8"
    )
    (folder / "map.world").write_text(world_sdf(name), encoding="utf-8")
    digest = hashlib.sha256(image.tobytes()).hexdigest()
    manifest = {"schema": "DEPCarArenaStaticMapV1", "name": name, "map_uuid": map_uuid, "seed": seed, "type": map_type, "occupancy_sha256": digest, "arena_tools_commit": "664b950d88c91b34e3cdce62512c8864b574b9d4"}
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data" / "arena_maps")
    parser.add_argument("--type", choices=("indoor", "outdoor"), default="indoor")
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=49100)
    parser.add_argument("--width", type=int, default=201); parser.add_argument("--height", type=int, default=201)
    parser.add_argument("--resolution", type=float, default=0.10)
    parser.add_argument("--corridor-radius", type=int, default=8); parser.add_argument("--iterations", type=int, default=45)
    parser.add_argument("--obstacles", type=int, default=90); parser.add_argument("--obstacle-radius", type=int, default=3)
    parser.add_argument("--resume", action="store_true", help="reuse deterministic maps whose provenance already matches")
    parser.add_argument("--workers", type=int, default=1, help="parallel deterministic map workers")
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    jobs = [
        (
            args.output, args.type, args.seed + index, index, args.width,
            args.height, args.resolution, args.corridor_radius, args.iterations,
            args.obstacles, args.obstacle_radius, args.resume,
        )
        for index in range(args.count)
    ]
    if args.workers == 1:
        manifests = []
        for index, job in enumerate(jobs, 1):
            manifests.append(generate_one(*job))
            print("prepared map %d/%d" % (index, len(jobs)), flush=True)
    else:
        indexed = {}
        with ProcessPoolExecutor(max_workers=min(args.workers, max(1, args.count))) as pool:
            futures = {pool.submit(generate_one, *job): index for index, job in enumerate(jobs)}
            completed = 0
            for future in as_completed(futures):
                indexed[futures[future]] = future.result()
                completed += 1
                print("prepared map %d/%d" % (completed, len(jobs)), flush=True)
        manifests = [indexed[index] for index in range(len(jobs))]
    print(json.dumps({
        "status": "PASS", "generated": len(manifests),
        "parallel_workers": min(args.workers, max(1, args.count)), "maps": manifests,
    }, indent=2))


if __name__ == "__main__": main()
