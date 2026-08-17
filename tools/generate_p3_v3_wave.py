#!/usr/bin/env python3
"""Generate a deterministic P3 V3 reinforcement-wave task manifest.

Every accepted route is preflighted with the exact current 3x5 Ackermann
lattice and production signed-SDF continuous swept-footprint evaluator.  Test
maps are rejected before their YAML/PNG is opened.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dep_car" / "src"))
sys.path.insert(0, str(ROOT / "tools"))

import audit_p3_footprint_upgrade as geometry
from dep_car.core.types import Gear
from dep_car.global_planner.hybrid_astar import HybridAStar, HybridAStarConfig
from dep_car.training.dataset import map_split
from dep_car.training.pilot import (
    PilotTask,
    PilotTaskSampler,
    canonical_sha256,
    classify_maneuver,
    make_pilot_manifest,
)


CONFIG_SCHEMA = "DEPCarP3V3IncrementalConfigV1"
PREFLIGHT_SCHEMA = "P3V3CanonicalLatticeContinuousSweepPreflightV1"
ALLOWED_SPLITS = ("train", "validation")
_WORKER_CACHE = {}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path):
    path = Path(path).resolve()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def load_config(path):
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != CONFIG_SCHEMA:
        raise ValueError("P3 V3 incremental config schema mismatch")
    wave = payload.get("wave", {})
    if not str(wave.get("id", "")).strip():
        raise ValueError("wave.id is required")
    if set(wave.get("selected_split_maps", {})) != set(ALLOWED_SPLITS):
        raise ValueError("selected_split_maps must contain train and validation only")
    quotas = wave.get("task_quotas", {})
    if set(quotas) != set(ALLOWED_SPLITS):
        raise ValueError("task_quotas must contain train and validation only")
    allowed_modes = set(geometry.PILOT_MANEUVER_MODES)
    for split, rows in quotas.items():
        if not isinstance(rows, dict) or not rows:
            raise ValueError("task quota is empty for " + split)
        if set(rows).difference(allowed_modes):
            raise ValueError("task quota contains an unknown maneuver")
        if any(isinstance(value, bool) or int(value) < 0 for value in rows.values()):
            raise ValueError("task quotas must be nonnegative integers")
    preflight = wave.get("preflight", {})
    if preflight.get("schema") != PREFLIGHT_SCHEMA:
        raise ValueError("preflight schema mismatch")
    if int(preflight.get("route_checkpoints", 0)) < 1:
        raise ValueError("route_checkpoints must be positive")
    if int(preflight.get("minimum_feasible_candidates_per_probe", 0)) < 2:
        raise ValueError("V3 task preflight requires at least two feasible candidates")
    return payload


def load_map_inventory(maps_root):
    maps_root = Path(maps_root).resolve()
    output = []
    for folder in sorted(path for path in maps_root.iterdir() if path.is_dir()):
        manifest_path = folder / "manifest.json"
        if not manifest_path.is_file():
            continue
        # The manifest is safe to open for split selection.  YAML/PNG geometry
        # is opened later only for selected train/validation UUIDs.
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        map_uuid = str(manifest.get("map_uuid", ""))
        split = map_split(map_uuid)
        if split not in ALLOWED_SPLITS:
            continue
        output.append({
            "folder": str(folder.resolve()),
            "manifest": manifest,
            "split": split,
        })
    if not output:
        raise RuntimeError("no train/validation maps are available")
    return output


def select_maps(inventory, required, seed):
    rng = np.random.default_rng(int(seed))
    groups = defaultdict(list)
    for item in inventory:
        groups[item["split"]].append(item)
    selected = []
    for split in ALLOWED_SPLITS:
        values = sorted(groups[split], key=lambda row: row["manifest"]["map_uuid"])
        order = rng.permutation(len(values)) if values else []
        values = [values[int(index)] for index in order]
        count = int(required[split])
        if len(values) < count:
            raise RuntimeError(
                "not enough %s maps: need %d, found %d" % (split, count, len(values))
            )
        selected.extend(values[:count])
    return selected


def _pose_transform(pose):
    cosine, sine = math.cos(float(pose.yaw)), math.sin(float(pose.yaw))
    return np.asarray(
        (
            (cosine, -sine, 0.0, float(pose.x)),
            (sine, cosine, 0.0, float(pose.y)),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
        dtype=np.float32,
    )


def _worker_map(record, maximum_expansions):
    key = (record["folder"], int(maximum_expansions))
    if key not in _WORKER_CACHE:
        folder = Path(record["folder"])
        map_uuid = str(record["manifest"]["map_uuid"])
        specs, _contract = geometry.load_map_specs(
            folder.parent, {map_uuid}, expected_aggregate=None
        )
        spec = specs[map_uuid]
        grid = geometry.grid_from_spec(spec)
        sdf = geometry.map_signed_sdf_from_spec(spec)
        planner = HybridAStar(HybridAStarConfig(
            maximum_expansions=int(maximum_expansions)
        ))
        _WORKER_CACHE[key] = (spec, grid, sdf, planner)
    return _WORKER_CACHE[key]


def _route_checkpoints(path, count):
    drive = [pose for pose in path if int(pose.gear) in (-1, 1)]
    if not drive:
        raise RuntimeError("route contains no drive pose")
    indices = np.linspace(0, len(drive) - 1, min(int(count), len(drive)))
    return [drive[int(index)] for index in sorted(set(np.rint(indices).astype(int)))]


def preflight_route(path, spec, sdf, preflight):
    rows = []
    minimum_required = int(preflight["minimum_feasible_candidates_per_probe"])
    for checkpoint_index, pose in enumerate(
        _route_checkpoints(path, int(preflight["route_checkpoints"]))
    ):
        gear = Gear.require_drive(int(pose.gear))
        speeds = (
            preflight["forward_speed_probes_mps"]
            if gear == Gear.FORWARD
            else preflight["reverse_speed_probes_mps"]
        )
        for speed in speeds:
            state = np.zeros(9, dtype=np.float64)
            state[0] = float(speed)
            state[2] = float(pose.steering)
            trajectories = geometry.regenerate_candidate_bank(
                state, int(gear), int(gear)
            )
            feasible, clearance = geometry.p5_exact_candidate_bank(
                trajectories,
                _pose_transform(pose),
                sdf,
                spec.resolution_m,
                spec.origin_xy,
            )
            count = int(np.sum(feasible))
            rows.append({
                "checkpoint": checkpoint_index,
                "gear": int(gear),
                "speed_mps": float(speed),
                "feasible_candidates": count,
                "best_clearance_m": float(np.max(clearance)),
            })
            if bool(preflight.get("require_every_probe", True)) and count < minimum_required:
                return None
    if not rows or min(row["feasible_candidates"] for row in rows) < minimum_required:
        return None
    evidence = {
        "schema": PREFLIGHT_SCHEMA,
        "qualification_evaluator": (
            "regenerated_AckermannLattice_plus_production_"
            "signed_SDF_bilinear_continuous_swept_footprint"
        ),
        "minimum_required_feasible_candidates": minimum_required,
        "route_checkpoints": len({row["checkpoint"] for row in rows}),
        "probe_count": len(rows),
        "minimum_feasible_candidates": min(row["feasible_candidates"] for row in rows),
        "minimum_best_clearance_m": min(row["best_clearance_m"] for row in rows),
        "probes": rows,
    }
    evidence["sha256"] = canonical_sha256(evidence)
    return evidence


def generate_map_mode(arguments):
    record, target_mode, seed, wave = arguments
    torch.set_num_threads(1)
    try:
        import cv2
        cv2.setNumThreads(1)
    except Exception:
        pass
    spec, grid, sdf, planner = _worker_map(
        record, wave["hybrid_astar_maximum_expansions"]
    )
    map_uuid = str(record["manifest"]["map_uuid"])
    rng = np.random.default_rng(int(seed))
    sampler = PilotTaskSampler(grid, rng)
    accepted = []
    failures = Counter()
    for attempt in range(int(wave["proposal_attempts_per_map_mode"])):
        if len(accepted) >= int(wave["proposals_per_map_mode"]):
            break
        try:
            start, goal = sampler.propose(target_mode)
            route = planner.plan(grid, start, goal)
            if not route:
                failures["no_path"] += 1
                continue
            observed, route_evidence = classify_maneuver(grid, start, goal, route)
            if observed != target_mode:
                failures["classified_as_" + observed] += 1
                continue
            preflight = preflight_route(route, spec, sdf, wave["preflight"])
            if preflight is None:
                failures["continuous_preflight"] += 1
                continue
            identity = canonical_sha256({
                "wave": wave["id"],
                "map_uuid": map_uuid,
                "mode": target_mode,
                "seed": int(seed),
                "attempt": attempt,
                "start": [round(float(value), 8) for value in start],
                "goal": [round(float(value), 8) for value in goal],
            })
            accepted.append({
                "proposal_id": identity,
                "map_name": record["manifest"]["name"],
                "map_uuid": map_uuid,
                "map_split": record["split"],
                "map_occupancy_sha256": record["manifest"]["occupancy_sha256"],
                "map_seed": int(record["manifest"]["seed"]),
                "world": relative(Path(record["folder"]) / "map.world"),
                "map_yaml": relative(Path(record["folder"]) / "map.yaml"),
                "start": list(map(float, start)),
                "goal": list(map(float, goal)),
                "maneuver_mode": target_mode,
                "task_seed": int(seed),
                "route_evidence": {
                    **route_evidence,
                    "p3_v3_preflight": preflight,
                },
            })
        except (RuntimeError, ValueError) as exc:
            failures[type(exc).__name__] += 1
    return {
        "map_uuid": map_uuid,
        "split": record["split"],
        "mode": target_mode,
        "accepted": accepted,
        "failures": dict(failures),
    }


def select_proposals(results, quotas, maximum_per_map):
    pools = defaultdict(list)
    failures = Counter()
    for result in results:
        pools[(result["split"], result["mode"])].extend(result["accepted"])
        for name, count in result["failures"].items():
            failures[result["split"] + ":" + result["mode"] + ":" + name] += count
    for values in pools.values():
        values.sort(key=lambda row: row["proposal_id"])
    map_counts = Counter()
    selected = []
    deficits = {}
    for split in ALLOWED_SPLITS:
        for mode, requested in sorted(quotas[split].items()):
            pool = list(pools[(split, mode)])
            for _ in range(int(requested)):
                eligible = [
                    row for row in pool
                    if map_counts[row["map_uuid"]] < int(maximum_per_map)
                ]
                if not eligible:
                    deficits[split + ":" + mode] = int(requested) - sum(
                        row["map_split"] == split and row["maneuver_mode"] == mode
                        for row in selected
                    )
                    break
                chosen = min(
                    eligible,
                    key=lambda row: (
                        map_counts[row["map_uuid"]], row["map_uuid"], row["proposal_id"]
                    ),
                )
                selected.append(chosen)
                map_counts[chosen["map_uuid"]] += 1
                pool.remove(chosen)
    return selected, deficits, failures, map_counts


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path,
        default=ROOT / "dep_car/config/p3_v3_incremental.yaml",
    )
    parser.add_argument(
        "--maps", type=Path, default=ROOT / "data/p3_pilot/maps"
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "data/p3_v3/waves/wave01/task_manifest.json",
    )
    parser.add_argument("--workers", type=int)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    wave = config["wave"]
    workers = int(wave["workers"] if args.workers is None else args.workers)
    if workers < 1 or workers > (os.cpu_count() or 1):
        raise ValueError("workers must fit the visible CPU-thread count")
    inventory = load_map_inventory(args.maps)
    selected_maps = select_maps(
        inventory, wave["selected_split_maps"], wave["task_seed"]
    )
    total_quotas = {
        split: sum(int(value) for value in wave["task_quotas"][split].values())
        for split in ALLOWED_SPLITS
    }
    capacity = {
        split: sum(row["split"] == split for row in selected_maps)
        * int(wave["maximum_tasks_per_map"])
        for split in ALLOWED_SPLITS
    }
    if any(total_quotas[split] > capacity[split] for split in ALLOWED_SPLITS):
        raise ValueError("task quotas exceed per-split selected-map capacity")
    selection_summary = {
        "available_development_maps": len(inventory),
        "selected": len(selected_maps),
        "split_counts": dict(sorted(Counter(row["split"] for row in selected_maps).items())),
        "map_uuids": [row["manifest"]["map_uuid"] for row in selected_maps],
        "test_maps_opened": False,
    }
    if args.validate_only:
        print(json.dumps({
            "status": "VALIDATION_PASS",
            "workers": workers,
            "map_selection": selection_summary,
            "task_quotas": wave["task_quotas"],
            "task_counts_by_split": total_quotas,
        }, indent=2, sort_keys=True))
        return 0

    modes_by_split = {
        split: sorted(mode for mode, count in wave["task_quotas"][split].items() if int(count))
        for split in ALLOWED_SPLITS
    }
    jobs = []
    for record in selected_maps:
        for mode in modes_by_split[record["split"]]:
            seed_material = "%s:%s:%s:%d" % (
                wave["id"], record["manifest"]["map_uuid"], mode, wave["task_seed"]
            )
            seed = int(hashlib.sha256(seed_material.encode("utf-8")).hexdigest()[:8], 16)
            jobs.append((record, mode, seed, wave))
    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(generate_map_mode, jobs, chunksize=1))
    selected, deficits, failures, map_counts = select_proposals(
        results, wave["task_quotas"], wave["maximum_tasks_per_map"]
    )
    tasks = []
    for number, row in enumerate(sorted(
        selected,
        key=lambda item: (item["map_split"], item["maneuver_mode"], item["proposal_id"]),
    )):
        task_id = "p3v3_" + canonical_sha256({
            "wave": wave["id"], "proposal": row["proposal_id"], "number": number
        })[:16]
        tasks.append(PilotTask(
            task_id=task_id,
            map_name=row["map_name"],
            map_uuid=row["map_uuid"],
            map_split=row["map_split"],
            map_occupancy_sha256=row["map_occupancy_sha256"],
            map_seed=row["map_seed"],
            world=row["world"],
            map_yaml=row["map_yaml"],
            start=tuple(row["start"]),
            goal=tuple(row["goal"]),
            maneuver_mode=row["maneuver_mode"],
            task_seed=row["task_seed"],
            route_evidence=row["route_evidence"],
        ))
    if deficits and not args.allow_partial:
        raise RuntimeError(
            "V3 task quotas could not be filled after continuous preflight: "
            + json.dumps(deficits, sort_keys=True)
        )
    aggregate_quotas = Counter()
    for rows in wave["task_quotas"].values():
        aggregate_quotas.update({name: int(value) for name, value in rows.items()})
    generator_contract = {
        "schema": "DEPCarP3V3WaveGeneratorContractV1",
        "config": config,
        "config_sha256": canonical_sha256(config),
        "tool": relative(__file__),
        "tool_sha256": file_sha256(__file__),
        "geometry_audit_tool": relative(geometry.__file__),
        "geometry_audit_tool_sha256": file_sha256(geometry.__file__),
        "planner": "gear-aware HybridAStar with calibrated Ackermann rollout",
        "classification": "P3PilotManeuverClassifierV1",
        "qualification_preflight": wave["preflight"],
        "split_task_quotas": wave["task_quotas"],
        "test_maps_opened": False,
        "partial": bool(deficits),
        "deficits": deficits,
        "proposal_failures": dict(sorted(failures.items())),
    }
    manifest = make_pilot_manifest(
        tasks,
        seed=wave["task_seed"],
        map_selection=selection_summary,
        quotas=dict(sorted(aggregate_quotas.items())),
        generator_contract=generator_contract,
    )
    atomic_json(args.output, manifest)
    payload = {
        "status": "PASS" if not deficits else "PARTIAL",
        "tasks": len(tasks),
        "tasks_by_split": dict(sorted(Counter(task.map_split for task in tasks).items())),
        "tasks_by_mode": dict(sorted(Counter(task.maneuver_mode for task in tasks).items())),
        "maps_used": dict(sorted((key, int(value)) for key, value in map_counts.items())),
        "parallel_workers": workers,
        "test_maps_opened": False,
        "task_manifest_sha256": manifest["task_manifest_sha256"],
        "output": str(args.output.resolve()),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not deficits else 2


if __name__ == "__main__":
    raise SystemExit(main())
