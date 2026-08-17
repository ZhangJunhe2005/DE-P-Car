#!/usr/bin/env python3
"""Decompose P3 V3 zero-feasible frames without modifying sealed authority.

The formal audit intentionally emits only frozen qualification summaries.  This
diagnostic reuses its exact regenerated lattice and signed-SDF evaluator, then
attributes every result to source, split, maneuver, candidate context, gear,
map, and episode.  Its output is diagnostic evidence, never training authority.
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
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dep_car" / "src"))
sys.path.insert(0, str(ROOT / "tools"))

import audit_p3_footprint_upgrade as geometry
import audit_p3_v3_bundle as bundle_audit


SCHEMA = "DEPCarP3V3FeasibilityDiagnosisV1"
DEFAULT_OUTPUT = ROOT / "reports/p3_v3_feasibility_diagnosis.json"
DEFAULT_ZERO_INVENTORY = ROOT / "reports/p3_v3_zero_feasible_inventory.json"
GEAR_NAMES = {-1: "REVERSE", 1: "FORWARD"}


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_zero_removals_needed(samples, zero, limit):
    """Smallest zero-only removal count that makes the rate strictly < limit."""

    samples, zero, limit = int(samples), int(zero), float(limit)
    if samples < 0 or zero < 0 or zero > samples or not 0.0 < limit < 1.0:
        raise ValueError("invalid strict-rate inputs")
    if zero / max(1, samples) < limit:
        return 0
    boundary = (zero - limit * samples) / (1.0 - limit)
    return min(zero, int(math.floor(boundary + 1.0e-12)) + 1)


def strict_feasible_additions_needed(samples, zero, limit):
    """Smallest all-feasible addition count that makes the rate strictly < limit."""

    samples, zero, limit = int(samples), int(zero), float(limit)
    if samples < 0 or zero < 0 or zero > samples or not 0.0 < limit < 1.0:
        raise ValueError("invalid strict-rate inputs")
    if zero / max(1, samples) < limit:
        return 0
    boundary = zero / limit - samples
    return max(0, int(math.floor(boundary + 1.0e-12)) + 1)


def load_task_sources(authority):
    task_sources = {}
    for source in authority.get("sources", ()):
        name = str(source.get("name", "")).strip()
        source_authority = Path(str(source.get("authority", "")))
        payload = json.loads(source_authority.read_text(encoding="utf-8"))
        task_ids = set(payload.get("tasks", {}))
        manifest_path = source.get("task_manifest")
        if manifest_path:
            manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
            task_ids.update(str(row["task_id"]) for row in manifest.get("tasks", ()))
        for task_id in task_ids:
            previous = task_sources.setdefault(str(task_id), name)
            if previous != name:
                raise RuntimeError("task belongs to multiple bundle sources: " + task_id)
    return task_sources


def inspect_entry(arguments):
    entry, sample_root, source_name = arguments
    row = geometry.inspect_indexed_sample(entry, sample_root)
    if "failure" in row:
        return row
    return {
        "path": row["path"],
        "source": source_name,
        "split": row["split"],
        "maneuver_mode": row["mode"],
        "candidate_context": str(entry.get("candidate_context", "UNKNOWN")),
        "requested_gear": GEAR_NAMES.get(int(row["requested_gear"]), "INVALID"),
        "map_uuid": row["map_uuid"],
        "task_id": str(entry.get("task_id", "")),
        "sample_id": str(entry.get("sample_id", "")),
        "content_sha256": str(entry.get("content_sha256", "")),
        "feasible_candidates": int(row["new_feasible_count"]),
        "zero_feasible": bool(row["new_zero"]),
        "best_clearance_m": float(max(row["new_clearance"])),
    }


def summary(rows):
    counts = [int(row["feasible_candidates"]) for row in rows]
    zero = sum(bool(row["zero_feasible"]) for row in rows)
    return {
        "samples": len(rows),
        "zero_feasible_samples": int(zero),
        "zero_feasible_rate": zero / len(rows) if rows else None,
        "feasible_candidates_median": float(np.median(counts)) if counts else None,
        "feasible_candidates_mean": float(np.mean(counts)) if counts else None,
    }


def grouped(rows, fields):
    buckets = defaultdict(list)
    for row in rows:
        buckets[tuple(row[field] for field in fields)].append(row)
    output = []
    for key, values in buckets.items():
        item = {field: value for field, value in zip(fields, key)}
        item.update(summary(values))
        output.append(item)
    return sorted(
        output,
        key=lambda row: (
            -float(row["zero_feasible_rate"] or 0.0),
            -int(row["samples"]),
            tuple(str(row[field]) for field in fields),
        ),
    )


def repair_math(rows, overall_limit, per_mode_limit):
    overall = summary(rows)
    modes = grouped(rows, ("maneuver_mode",))
    return {
        "interpretation": (
            "Arithmetic bounds only. Removal is not automatically authorized; "
            "real blocked/recovery frames must not be silently discarded."
        ),
        "overall": {
            **overall,
            "strict_limit": float(overall_limit),
            "minimum_zero_frames_to_exclude": strict_zero_removals_needed(
                overall["samples"], overall["zero_feasible_samples"], overall_limit
            ),
            "minimum_all_feasible_frames_to_add": strict_feasible_additions_needed(
                overall["samples"], overall["zero_feasible_samples"], overall_limit
            ),
        },
        "by_mode": [
            {
                **row,
                "strict_limit": float(per_mode_limit),
                "minimum_zero_frames_to_exclude": strict_zero_removals_needed(
                    row["samples"], row["zero_feasible_samples"], per_mode_limit
                ),
                "minimum_all_feasible_frames_to_add": strict_feasible_additions_needed(
                    row["samples"], row["zero_feasible_samples"], per_mode_limit
                ),
            }
            for row in modes
        ],
    }


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
    parser.add_argument("--zero-inventory", type=Path, default=DEFAULT_ZERO_INVENTORY)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--maximum-samples", type=int, default=0)
    args = parser.parse_args(argv)
    if args.workers < 1 or args.workers > (os.cpu_count() or 1):
        raise ValueError("workers must fit the visible CPU-thread count")
    if args.maximum_samples < 0:
        raise ValueError("maximum-samples cannot be negative")

    config = bundle_audit.load_config(args.config)
    authority, authority_file_hash, authority_errors = (
        bundle_audit.verify_bundle_authority(args.authority, args.config)
    )
    if authority_errors:
        raise RuntimeError("bundle authority verification failed: " + ",".join(authority_errors))
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
    selected = (
        index.entries[:args.maximum_samples] if args.maximum_samples else index.entries
    )
    task_sources = load_task_sources(authority)
    unknown_tasks = sorted({
        str(entry.get("task_id", "")) for entry in selected
        if str(entry.get("task_id", "")) not in task_sources
    })
    if unknown_tasks:
        raise RuntimeError("indexed tasks have no source authority: " + ",".join(unknown_tasks[:5]))
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
        initializer=geometry.initialize_worker,
        initargs=(selected_specs,),
    ) as executor:
        tasks = (
            (entry, str(sample_root), task_sources[str(entry.get("task_id", ""))])
            for entry in selected
        )
        for number, row in enumerate(executor.map(inspect_entry, tasks, chunksize=8), 1):
            if "failure" in row:
                failures[row["path"]] = row["failure"]
            else:
                rows.append(row)
            if number % 512 == 0 or number == len(selected):
                print(
                    "diagnosed %d/%d samples with %d workers"
                    % (number, len(selected), args.workers),
                    file=sys.stderr,
                    flush=True,
                )
    if failures or len(rows) != len(selected):
        raise RuntimeError(
            "diagnosis did not cover the sealed selection: "
            + json.dumps(failures, sort_keys=True)[:500]
        )

    zero_rows = [row for row in rows if row["zero_feasible"]]
    source_counts = Counter(row["source"] for row in rows)
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
        "tool": str(Path(__file__).resolve()),
        "tool_sha256": file_sha256(__file__),
        "parallel_workers": args.workers,
        "limited": bool(args.maximum_samples),
        "samples": len(rows),
        "zero_feasible_samples": len(zero_rows),
        "overall": summary(rows),
        "source_sample_counts": dict(sorted(source_counts.items())),
        "by_source": grouped(rows, ("source",)),
        "by_split": grouped(rows, ("split",)),
        "by_mode": grouped(rows, ("maneuver_mode",)),
        "by_context": grouped(rows, ("candidate_context",)),
        "by_gear": grouped(rows, ("requested_gear",)),
        "by_source_mode": grouped(rows, ("source", "maneuver_mode")),
        "by_context_mode": grouped(rows, ("candidate_context", "maneuver_mode")),
        "by_context_gear": grouped(rows, ("candidate_context", "requested_gear")),
        "top_tasks": grouped(rows, ("source", "task_id", "maneuver_mode"))[:50],
        "top_maps": grouped(rows, ("map_uuid", "split"))[:50],
        "repair_arithmetic": repair_math(
            rows,
            config["gates"]["maximum_zero_feasible_rate"],
            config["gates"]["maximum_per_mode_zero_feasible_rate"],
        ),
        "zero_inventory": str(args.zero_inventory.resolve()),
        "test_npz_opened": False,
    }
    zero_inventory = {
        "schema": "DEPCarP3V3ZeroFeasibleInventoryV1",
        "status": "DIAGNOSTIC_ONLY",
        "qualification_authority": False,
        "bundle_authority_sha256": authority["bundle_authority_sha256"],
        "index_sha256": authority["index_sha256"],
        "samples": len(zero_rows),
        "entries": zero_rows,
        "test_npz_opened": False,
    }
    atomic_json(args.zero_inventory, zero_inventory)
    atomic_json(args.output, report)
    print(json.dumps({
        "status": report["status"],
        "samples": report["samples"],
        "zero_feasible_samples": report["zero_feasible_samples"],
        "zero_feasible_rate": report["overall"]["zero_feasible_rate"],
        "by_source": report["by_source"],
        "by_context": report["by_context"],
        "by_mode": report["by_mode"],
        "report": str(args.output.resolve()),
        "zero_inventory": str(args.zero_inventory.resolve()),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
