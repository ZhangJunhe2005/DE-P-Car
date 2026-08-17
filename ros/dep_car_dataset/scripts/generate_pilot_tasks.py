#!/usr/bin/env python3
"""Generate the deterministic, maneuver-stratified P3 Gazebo task manifest."""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml
from PIL import Image


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "dep_car/src"))
from dep_car.core.occupancy import OccupancyGrid2D
from dep_car.global_planner.hybrid_astar import HybridAStar, HybridAStarConfig
from dep_car.training.dataset import map_split
from dep_car.training.pilot import PilotTask, PilotTaskSampler, canonical_sha256, classify_maneuver, make_pilot_manifest


def load_grid(folder):
    metadata = yaml.safe_load((folder / "map.yaml").read_text(encoding="utf-8"))
    image = np.asarray(Image.open(str(folder / metadata["image"])).convert("L"))
    data = np.flipud(np.where(image < 128, 100, 0).astype(np.int16))
    return OccupancyGrid2D(data, metadata["resolution"], tuple(metadata["origin"][:2]))


def load_maps(root):
    output = []
    for folder in sorted(path for path in root.iterdir() if (path / "manifest.json").is_file()):
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        output.append((folder, manifest, map_split(manifest["map_uuid"])))
    return output


def select_maps(maps, required, total, rng):
    groups = defaultdict(list)
    for item in maps:
        groups[item[2]].append(item)
    selected = []
    for split, minimum in required.items():
        values = list(groups[split])
        rng.shuffle(values)
        if len(values) < int(minimum):
            raise RuntimeError("not enough %s maps: need %d, found %d" % (split, minimum, len(values)))
        selected.extend(values[:int(minimum)])
    selected_ids = {item[1]["map_uuid"] for item in selected}
    remaining = [item for item in maps if item[1]["map_uuid"] not in selected_ids]
    rng.shuffle(remaining)
    if len(selected) + len(remaining) < total:
        raise RuntimeError("not enough generated maps for requested selection")
    selected.extend(remaining[:total - len(selected)])
    return selected


def relative(path):
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "dep_car/config/p3_pilot.yaml")
    parser.add_argument("--maps", type=Path, default=ROOT / "data/p3_pilot/maps")
    parser.add_argument("--output", type=Path, default=ROOT / "data/p3_pilot/task_manifest.json")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--allow-partial", action="store_true", help="write a manifest with explicit quota deficits")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    map_config, task_config = config["maps"], config["tasks"]
    quotas = {str(name): int(value) for name, value in task_config["maneuver_quotas"].items()}
    if sum(quotas.values()) != int(task_config["total"]):
        raise ValueError("maneuver quotas must sum to tasks.total")
    maps = load_maps(args.maps)
    rng = np.random.default_rng(int(map_config["task_seed"]))
    selected = select_maps(maps, map_config["required_split_maps"], int(map_config["selected_count"]), rng)
    selection_summary = {
        "available": len(maps), "selected": len(selected),
        "split_counts": dict(Counter(item[2] for item in selected)),
        "map_uuids": [item[1]["map_uuid"] for item in selected],
    }
    if args.validate_only:
        print(json.dumps({"status": "PASS", "map_selection": selection_summary, "maneuver_quotas": quotas}, indent=2, sort_keys=True))
        return

    grids = {manifest["map_uuid"]: load_grid(folder) for folder, manifest, _ in selected}
    planners = {
        manifest["map_uuid"]: HybridAStar(HybridAStarConfig(maximum_expansions=int(task_config["hybrid_astar_maximum_expansions"])))
        for _, manifest, _ in selected
    }
    counts = Counter()
    failures = Counter()
    tasks = []
    priority = {
        "THREE_POINT_TURN": 0, "DEAD_END_ESCAPE": 1, "U_TURN": 2,
        "NARROW_CORRIDOR": 3, "REVERSE_EXIT": 4, "SHARP_TURN": 5, "NORMAL": 6,
    }
    requested_modes = [mode for mode, count in quotas.items() for _ in range(count)]
    requested_modes.sort(key=lambda mode: (priority[mode], float(rng.random())))
    per_map_limit = int(task_config["maximum_per_map"])
    attempts_per_map = int(task_config["proposal_attempts_per_map"])

    for task_number, target_mode in enumerate(requested_modes):
        candidates = [item for item in selected if counts[item[1]["map_uuid"]] < per_map_limit]
        candidates.sort(key=lambda item: (counts[item[1]["map_uuid"]], float(rng.random())))
        accepted = None
        task_seed = int(rng.integers(0, 2**31 - 1))
        for folder, map_manifest, split in candidates:
            local_rng = np.random.default_rng(task_seed ^ int(map_manifest["map_uuid"].replace("-", "")[:8], 16))
            sampler = PilotTaskSampler(grids[map_manifest["map_uuid"]], local_rng)
            for _ in range(attempts_per_map):
                try:
                    start, goal = sampler.propose(target_mode)
                    path = planners[map_manifest["map_uuid"]].plan(grids[map_manifest["map_uuid"]], start, goal)
                    if not path:
                        failures[target_mode + ":no_path"] += 1
                        continue
                    observed_mode, evidence = classify_maneuver(grids[map_manifest["map_uuid"]], start, goal, path)
                    if observed_mode != target_mode:
                        failures[target_mode + ":classified_as_" + observed_mode] += 1
                        continue
                    identity = "%s:%s:%d:%d" % (map_manifest["map_uuid"], target_mode, task_seed, task_number)
                    accepted = PilotTask(
                        task_id="p3_" + canonical_sha256(identity)[:16],
                        map_name=map_manifest["name"], map_uuid=map_manifest["map_uuid"], map_split=split,
                        map_occupancy_sha256=map_manifest["occupancy_sha256"], map_seed=int(map_manifest["seed"]),
                        world=relative(folder / "map.world"), map_yaml=relative(folder / "map.yaml"),
                        start=tuple(float(value) for value in start), goal=tuple(float(value) for value in goal),
                        maneuver_mode=target_mode, task_seed=task_seed, route_evidence=evidence,
                    )
                    break
                except (RuntimeError, ValueError) as exc:
                    failures[target_mode + ":" + type(exc).__name__] += 1
            if accepted is not None:
                counts[map_manifest["map_uuid"]] += 1
                tasks.append(accepted)
                print("prepared %d/%d %s %s" % (len(tasks), len(requested_modes), target_mode, accepted.task_id), flush=True)
                break
        if accepted is None:
            failures[target_mode + ":quota_deficit"] += 1
            if not args.allow_partial:
                raise RuntimeError("could not fill %s quota; rerun with more maps or --allow-partial" % target_mode)

    achieved = Counter(task.maneuver_mode for task in tasks)
    deficits = {mode: quotas[mode] - achieved.get(mode, 0) for mode in quotas if achieved.get(mode, 0) < quotas[mode]}
    generator_contract = {
        "config": config,
        "config_sha256": canonical_sha256(config),
        "planner": "gear-aware HybridAStar with calibrated Ackermann rollout",
        "classification": "P3PilotManeuverClassifierV1",
        "partial": bool(deficits),
        "deficits": deficits,
        "proposal_failures": dict(failures),
    }
    manifest = make_pilot_manifest(
        tasks, seed=map_config["task_seed"], map_selection=selection_summary,
        quotas=quotas, generator_contract=generator_contract,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps({
        "status": "PASS" if not deficits else "PARTIAL", "tasks": len(tasks),
        "mode_counts": dict(achieved), "map_counts": dict(Counter(task.map_uuid for task in tasks)),
        "task_manifest_sha256": manifest["task_manifest_sha256"], "output": str(args.output),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
