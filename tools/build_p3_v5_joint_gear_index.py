#!/usr/bin/env python3
"""Build a sealed six-macro-action V3 view over the immutable P3 V4 bundle."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "DEPCarJointGearSequenceIndexV1"
HISTORY_SCHEMA = "DEPCarJointGearHistoryV1"
SEQUENCE_ACTIONS = 6


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value):
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def resolve(path):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def gear_runs(values):
    runs = []
    for value in values:
        value = int(value)
        if value not in (-1, 1):
            continue
        if not runs or runs[-1] != value:
            runs.append(value)
    return runs


def read_row(arguments):
    entry, sample_root = arguments
    path = (sample_root / entry["path"]).resolve()
    if sample_root not in path.parents or not path.is_file():
        raise RuntimeError("indexed V3 source is unavailable: %s" % path)
    if sha256_file(path) != entry["content_sha256"]:
        raise RuntimeError("indexed V3 source hash changed: %s" % path)
    with np.load(path, allow_pickle=False) as data:
        manifest = json.loads(str(data["manifest_json"]))
        timestamps = manifest.get("timestamps", {})
        stamp = float(timestamps.get("candidates", timestamps.get("odom", math.nan)))
        current_gear = int(data["current_gear"])
        requested_gear = int(data["requested_gear"])
        state = np.asarray(data["vehicle_state"], dtype=np.float64)
        local_gears = np.asarray(data["local_path_gears"], dtype=np.int8)
        context = str(manifest.get("metadata", {}).get("candidate_context", "UNKNOWN"))
    if (
        not math.isfinite(stamp)
        or current_gear not in (-1, 0, 1)
        or requested_gear not in (-1, 1)
        or state.shape != (9,)
        or not np.all(np.isfinite(state))
        or local_gears.ndim != 1
        or not np.all(np.isin(local_gears, (-1, 0, 1)))
    ):
        raise RuntimeError("V3 sequence source fields are invalid: %s" % path)
    sequence = gear_runs(local_gears.tolist())[:SEQUENCE_ACTIONS]
    if not sequence:
        sequence = [requested_gear]
    return {
        "sample_id": entry["sample_id"],
        "task_id": entry["task_id"],
        "split": entry["split"],
        "map_uuid": entry["map_uuid"],
        "source_content_sha256": entry["content_sha256"],
        "stamp": stamp,
        "speed_mps": float(state[0]),
        "current_gear": current_gear,
        "observed_requested_gear": requested_gear,
        "candidate_context": context,
        "reference_gear_runs": sequence,
    }


def attach_history(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["task_id"]].append(row)
    output = []
    for task_id in sorted(groups):
        episode = sorted(groups[task_id], key=lambda row: (row["stamp"], row["sample_id"]))
        previous = 0
        shift_age = 0.0
        recent = []
        forward_progress = 0.0
        reverse_progress = 0.0
        last_stamp = None
        if len({row["split"] for row in episode}) != 1:
            raise RuntimeError("one episode crosses dataset splits: %s" % task_id)
        future_observed = [row["observed_requested_gear"] for row in episode]
        for episode_index, row in enumerate(episode):
            stamp = float(row["stamp"])
            dt = 0.0 if last_stamp is None else min(0.5, max(0.0, stamp - last_stamp))
            speed = float(row["speed_mps"])
            forward_progress += max(0.0, speed) * dt
            reverse_progress += max(0.0, -speed) * dt
            gear = int(row["observed_requested_gear"])
            if previous in (-1, 1) and gear == previous:
                shift_age += dt
            else:
                shift_age = 0.0
            if not recent or recent[-1][0] != gear:
                recent.append((gear, stamp))
            recent = [item for item in recent if stamp - item[1] <= 6.0]
            switches = max(0, len(recent) - 1)
            # A single saved local path usually contains only the active gear.
            # Append future *observed* episode gears before run compression so
            # R-F-R-F manoeuvres remain learnable without inventing labels or
            # crossing a task/map boundary.
            sequence = gear_runs(
                list(row["reference_gear_runs"])
                + future_observed[episode_index:]
            )[:SEQUENCE_ACTIONS]
            mask = [True] * len(sequence)
            sequence += [0] * (SEQUENCE_ACTIONS - len(sequence))
            mask += [False] * (SEQUENCE_ACTIONS - len(mask))
            bound = dict(row)
            bound["history"] = [
                float(previous if previous in (-1, 1) else gear),
                min(1.0, shift_age / 5.0),
                min(2.0, forward_progress / 2.0),
                min(2.0, reverse_progress / 2.0),
                float(min(4, switches)),
                1.0 if row["candidate_context"] == "RECOVERY" else 0.0,
            ]
            bound["sequence_gears"] = sequence
            bound["sequence_mask"] = mask
            output.append(bound)
            previous = gear
            last_stamp = stamp
    return sorted(output, key=lambda row: row["sample_id"])


def build(index_path, output_path, workers, maximum_samples=None, dry_run=False):
    index_path = resolve(index_path)
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    if (
        payload.get("schema") != "P3TrainingIndexV2"
        or set(payload.get("splits", ())) != {"train", "validation"}
        or any(entry.get("split") == "test" for entry in payload.get("entries", ()))
    ):
        raise RuntimeError("V3 source must be the sealed development P3 index")
    entries = list(payload["entries"])
    if maximum_samples is not None:
        count = min(len(entries), int(maximum_samples))
        indices = np.linspace(0, len(entries) - 1, count, dtype=np.int64)
        entries = [entries[int(index)] for index in indices]
    sample_root = resolve(payload["sample_root"])
    started = time.monotonic()
    rows = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for index, row in enumerate(
            pool.map(read_row, ((entry, sample_root) for entry in entries)), start=1
        ):
            rows.append(row)
            if index % 500 == 0 or index == len(entries):
                elapsed = max(1.0e-6, time.monotonic() - started)
                print(
                    "V3 index %d/%d samples %.1f samples/s"
                    % (index, len(entries), index / elapsed),
                    flush=True,
                    file=sys.stderr,
                )
    rows = attach_history(rows)
    sequence_counts = Counter(
        "-".join("F" if value > 0 else "R" for value in row["sequence_gears"] if value)
        for row in rows
    )
    multileg = sum(
        1
        for row in rows
        if sum(bool(value) for value in row["sequence_mask"]) >= 4
    )
    content = {
        "schema": SCHEMA,
        "history_schema": HISTORY_SCHEMA,
        "source_index": str(index_path),
        "source_index_sha256": sha256_file(index_path),
        "source_content_aggregate_sha256": payload["content_aggregate_sha256"],
        "sample_root": str(sample_root),
        "sequence_actions": SEQUENCE_ACTIONS,
        "sequence_authority": (
            "local_path_kinematic_gear_runs_weak_sequence_supervision;"
            "candidate_choice_is_counterfactual_joint_cost"
        ),
        "test_split_opened": False,
        "bounded": maximum_samples is not None,
        "samples": len(rows),
        "counts_by_split": dict(sorted(Counter(row["split"] for row in rows).items())),
        "multi_action_samples_ge4": multileg,
        "sequence_counts": dict(sorted(sequence_counts.items())),
        "rows": rows,
    }
    content["content_sha256"] = canonical_sha256(content)
    status = "DRY_RUN_PASS" if dry_run else "PASS"
    report = {
        "status": status,
        "schema": SCHEMA,
        "output": str(resolve(output_path)),
        "parallel_workers": workers,
        "samples": len(rows),
        "counts_by_split": content["counts_by_split"],
        "multi_action_samples_ge4": multileg,
        "sequence_counts": content["sequence_counts"],
        "content_sha256": content["content_sha256"],
        "test_split_opened": False,
        "bounded": maximum_samples is not None,
    }
    if not dry_run:
        output_path = resolve(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_suffix(output_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(content, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, output_path)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", default="data/p3_v4/bundle_v1/training_index.json")
    parser.add_argument("--output", default="data/p3_v5/joint_gear_sequence_index.json")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--maximum-samples", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.workers < 1:
        raise SystemExit("workers must be positive")
    if args.maximum_samples is not None and args.maximum_samples < 1:
        raise SystemExit("maximum-samples must be positive")
    report = build(
        args.index,
        args.output,
        args.workers,
        maximum_samples=args.maximum_samples,
        dry_run=args.dry_run,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
