#!/usr/bin/env python3
"""Read-only counterfactual evaluation of P3 V3 Ackermann lattices.

This diagnostic authenticates the sealed development bundle, regenerates
several fixed 3x5 lattices from each indexed physical state, and evaluates all
of them with the exact P5 swept-footprint objective.  It does not modify the
bundle, its index, the formal audit, or any training authority.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dep_car" / "src"))
sys.path.insert(0, str(ROOT / "tools"))

import audit_p3_footprint_upgrade as geometry
import audit_p3_v3_bundle as bundle_audit
from dep_car.core.lattice import AckermannLattice, LatticeConfig


SCHEMA = "DEPCarP3V3LatticeCounterfactualV1"
DEFAULT_OUTPUT = ROOT / "reports/p3_v3_lattice_counterfactual.json"
BASELINE = "baseline_1p00s"

# All alternatives retain the P4 3 speed x 5 steering tensor shape.  The
# balanced lattice keeps nominal cruise and high-speed reach through the
# learnable speed residual, while adding one useful low-speed anchor per gear.
CONFIGS = {
    BASELINE: LatticeConfig(),
    "baseline_0p75s": LatticeConfig(horizon=0.75),
    "balanced_creep_1p00s": LatticeConfig(
        speed_anchors=(0.15, 0.60, 1.20),
        reverse_speed_anchors=(0.10, 0.25, 0.50),
    ),
    "balanced_creep_0p75s": LatticeConfig(
        speed_anchors=(0.15, 0.60, 1.20),
        reverse_speed_anchors=(0.10, 0.25, 0.50),
        horizon=0.75,
    ),
    "low_creep_1p00s": LatticeConfig(
        speed_anchors=(0.10, 0.35, 0.80),
        reverse_speed_anchors=(0.05, 0.15, 0.30),
    ),
    "low_creep_0p75s": LatticeConfig(
        speed_anchors=(0.10, 0.35, 0.80),
        reverse_speed_anchors=(0.05, 0.15, 0.30),
        horizon=0.75,
    ),
}

_WORKER_LATTICES = {}


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def initialize_worker(specs, configs):
    geometry.initialize_worker(specs)
    global _WORKER_LATTICES
    _WORKER_LATTICES = {
        name: AckermannLattice(config) for name, config in configs.items()
    }


def regenerate(config_name, vehicle_state, current_gear, requested_gear):
    state, requested = geometry.canonical_vehicle_state(
        vehicle_state, current_gear, requested_gear
    )
    candidates = _WORKER_LATTICES[config_name].generate(state, gear=requested)
    if len(candidates) != geometry.EXPECTED_CANDIDATES:
        raise RuntimeError("counterfactual lattice did not produce 15 candidates")
    trajectories = np.stack(
        [np.asarray(candidate.trajectory, dtype=np.float64) for candidate in candidates]
    )
    if trajectories.ndim != 3 or trajectories.shape[:1] != (15,):
        raise RuntimeError("counterfactual lattice trajectory shape is invalid")
    return trajectories


def inspect_entry(arguments):
    entry, sample_root = arguments
    # The frozen inspector authenticates all sample, map, transform, and
    # metadata contracts and provides the exact baseline result.
    baseline = geometry.inspect_indexed_sample(entry, sample_root)
    if "failure" in baseline:
        return baseline

    path = geometry._safe_indexed_sample_path(sample_root, entry)
    sample_bytes = path.read_bytes()
    with np.load(io.BytesIO(sample_bytes), allow_pickle=False) as data:
        manifest = json.loads(str(data["manifest_json"]))
        map_uuid = str(manifest["map_uuid"])
        transform = np.asarray(
            manifest["transforms"]["chassis_to_map"]["matrix"], dtype=np.float64
        )
        vehicle_state = np.asarray(data["vehicle_state"], dtype=np.float64)
        current_gear = int(data["current_gear"])
        requested_gear = int(data["requested_gear"])

    baseline_trajectories = regenerate(
        BASELINE, vehicle_state, current_gear, requested_gear
    )
    initial_feasible, _ = geometry.p5_exact_candidate_bank(
        baseline_trajectories[:, :1, :],
        transform,
        geometry._WORKER_MAP_DISTANCE_FIELDS[map_uuid],
        geometry._WORKER_SPECS[map_uuid].resolution_m,
        geometry._WORKER_SPECS[map_uuid].origin_xy,
    )
    # Every candidate has the same t=0 pose.  Retain an explicit check so a
    # collection/geometry problem cannot be mistaken for lattice coverage.
    if not bool(np.all(initial_feasible == initial_feasible[0])):
        raise RuntimeError("candidate t=0 footprint is not invariant")

    results = {}
    for name in _WORKER_LATTICES:
        if name == BASELINE:
            feasible_count = int(baseline["new_feasible_count"])
        else:
            trajectories = regenerate(
                name, vehicle_state, current_gear, requested_gear
            )
            feasible, _ = geometry.p5_exact_candidate_bank(
                trajectories,
                transform,
                geometry._WORKER_MAP_DISTANCE_FIELDS[map_uuid],
                geometry._WORKER_SPECS[map_uuid].resolution_m,
                geometry._WORKER_SPECS[map_uuid].origin_xy,
            )
            feasible_count = int(np.count_nonzero(feasible))
        results[name] = {
            "feasible_candidates": feasible_count,
            "zero_feasible": feasible_count == 0,
        }
    return {
        "path": baseline["path"],
        "split": baseline["split"],
        "maneuver_mode": baseline["mode"],
        "candidate_context": str(entry.get("candidate_context", "UNKNOWN")),
        "requested_gear": int(baseline["requested_gear"]),
        "map_uuid": baseline["map_uuid"],
        "task_id": str(entry.get("task_id", "")),
        "initial_pose_feasible": bool(initial_feasible[0]),
        "results": results,
    }


def summarize(rows, config_name):
    counts = [row["results"][config_name]["feasible_candidates"] for row in rows]
    zeros = sum(row["results"][config_name]["zero_feasible"] for row in rows)
    baseline_zeros = [row["results"][BASELINE]["zero_feasible"] for row in rows]
    candidate_zeros = [row["results"][config_name]["zero_feasible"] for row in rows]
    rescued = sum(old and not new for old, new in zip(baseline_zeros, candidate_zeros))
    lost = sum(not old and new for old, new in zip(baseline_zeros, candidate_zeros))
    initial_infeasible = [not row["initial_pose_feasible"] for row in rows]
    return {
        "samples": len(rows),
        "zero_feasible_samples": int(zeros),
        "zero_feasible_rate": zeros / len(rows) if rows else None,
        "feasible_candidates_mean": float(np.mean(counts)) if counts else None,
        "feasible_candidates_median": float(np.median(counts)) if counts else None,
        "baseline_zero_samples_rescued": int(rescued),
        "baseline_feasible_samples_lost": int(lost),
        "initial_pose_infeasible_samples": int(sum(initial_infeasible)),
        "zero_due_to_initial_pose": int(sum(
            initial and zero for initial, zero in zip(initial_infeasible, candidate_zeros)
        )),
        "zero_after_safe_initial_pose": int(sum(
            (not initial) and zero
            for initial, zero in zip(initial_infeasible, candidate_zeros)
        )),
    }


def grouped(rows, config_name, field):
    buckets = defaultdict(list)
    for row in rows:
        buckets[str(row[field])].append(row)
    output = []
    for value, values in buckets.items():
        item = {field: value}
        item.update(summarize(values, config_name))
        output.append(item)
    return sorted(output, key=lambda row: (-row["zero_feasible_rate"], row[field]))


def config_contract(config):
    payload = asdict(config)
    for key in ("speed_anchors", "reverse_speed_anchors", "steering_anchors"):
        payload[key] = list(payload[key])
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path,
        default=ROOT / "dep_car/config/p3_v3_incremental.yaml",
    )
    parser.add_argument(
        "--authority", type=Path,
        default=ROOT / "data/p3_v3/bundle_v1/bundle_authority.json",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--maximum-samples", type=int, default=0)
    args = parser.parse_args(argv)
    if args.workers < 1 or args.workers > (os.cpu_count() or 1):
        raise ValueError("workers must fit the visible CPU-thread count")
    if args.maximum_samples < 0:
        raise ValueError("maximum-samples cannot be negative")

    authority, authority_file_hash, authority_errors = (
        bundle_audit.verify_bundle_authority(args.authority, args.config)
    )
    if authority_errors:
        raise RuntimeError(
            "bundle authority verification failed: " + ",".join(authority_errors)
        )
    index_path = Path(authority["index"]).resolve()
    sample_root = Path(authority["sample_root"]).resolve()
    maps_root = Path(authority["maps_root"]).resolve()
    index = geometry.load_index_authority(
        index_path,
        sample_root,
        maps_root,
        expected_index_sha256=authority["index_sha256"],
        expected_content_aggregate=authority["content_aggregate_sha256"],
        expected_split_counts=authority["counts_by_split"],
    )
    selected = index.entries[:args.maximum_samples] if args.maximum_samples else index.entries
    specs, map_contract = geometry.load_map_specs(
        maps_root,
        {str(entry["map_uuid"]) for entry in index.entries},
        expected_aggregate=authority["map_contract_aggregate_sha256"],
    )
    selected_specs = {
        map_uuid: specs[map_uuid]
        for map_uuid in {str(entry["map_uuid"]) for entry in selected}
    }

    rows, failures = [], {}
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=initialize_worker,
        initargs=(selected_specs, CONFIGS),
    ) as executor:
        tasks = ((entry, str(sample_root)) for entry in selected)
        for number, row in enumerate(
            executor.map(inspect_entry, tasks, chunksize=8), 1
        ):
            if "failure" in row:
                failures[row["path"]] = row["failure"]
            else:
                rows.append(row)
            if number % 256 == 0 or number == len(selected):
                print(
                    "evaluated %d/%d samples with %d workers"
                    % (number, len(selected), args.workers),
                    file=sys.stderr,
                    flush=True,
                )
    if failures or len(rows) != len(selected):
        raise RuntimeError(
            "counterfactual did not cover the sealed selection: "
            + json.dumps(failures, sort_keys=True)[:500]
        )

    alternatives = {}
    for name, config in CONFIGS.items():
        alternatives[name] = {
            "lattice": config_contract(config),
            "overall": summarize(rows, name),
            "by_mode": grouped(rows, name, "maneuver_mode"),
            "by_context": grouped(rows, name, "candidate_context"),
            "by_gear": grouped(rows, name, "requested_gear"),
        }
    report = {
        "schema": SCHEMA,
        "status": "DIAGNOSTIC_PASS",
        "qualification_authority": False,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "bundle_authority": str(args.authority.resolve()),
        "bundle_authority_file_sha256": authority_file_hash,
        "bundle_authority_sha256": authority["bundle_authority_sha256"],
        "index_sha256": authority["index_sha256"],
        "map_contract_aggregate_sha256": map_contract["aggregate_sha256"],
        "parallel_workers": args.workers,
        "limited": bool(args.maximum_samples),
        "samples": len(rows),
        "baseline": BASELINE,
        "alternatives": alternatives,
        "interpretation": (
            "Counterfactual evidence only. A selected lattice must be synchronized "
            "with the P4 differentiable rollout and re-qualified before use."
        ),
        "test_npz_opened": False,
    }
    atomic_json(args.output, report)
    print(json.dumps({
        "status": report["status"],
        "samples": report["samples"],
        "alternatives": {
            name: value["overall"] for name, value in alternatives.items()
        },
        "report": str(args.output.resolve()),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
