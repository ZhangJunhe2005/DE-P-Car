#!/usr/bin/env python3
"""Freeze a map-diverse P6 DAgger subset without opening sealed test maps."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import random


ROOT = Path(__file__).resolve().parents[1]


def canonical_sha256(value):
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def resolve(value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def select_group(rows, quota, seed):
    by_map = defaultdict(list)
    for row in rows:
        by_map[row["map_uuid"]].append(row)
    rng = random.Random(seed)
    maps = sorted(by_map)
    rng.shuffle(maps)
    for values in by_map.values():
        rng.shuffle(values)
    output = []
    while len(output) < quota and maps:
        remaining = []
        for map_uuid in maps:
            if by_map[map_uuid] and len(output) < quota:
                output.append(by_map[map_uuid].pop())
            if by_map[map_uuid]:
                remaining.append(map_uuid)
        maps = remaining
    if len(output) != quota:
        raise RuntimeError("insufficient tasks for requested V4.3 quota")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="data/p3_v4/waves/corner01/task_manifest.json")
    parser.add_argument("--output", default="data/p3_v7_v43/task_manifest.json")
    parser.add_argument("--seed", type=int, default=86430)
    args = parser.parse_args()
    source_path, output_path = resolve(args.source), resolve(args.output)
    source = json.loads(source_path.read_text(encoding="utf-8"))
    content = dict(source); claimed = content.pop("task_manifest_sha256", None)
    if claimed != canonical_sha256(content):
        raise RuntimeError("source task manifest internal SHA-256 differs")
    quotas = {
        "train": {
            "SHARP_TURN": 32, "THREE_POINT_TURN": 16,
            "DEAD_END_ESCAPE": 8, "NARROW_CORRIDOR": 8,
        },
        "validation": {
            "SHARP_TURN": 8, "THREE_POINT_TURN": 4,
            "DEAD_END_ESCAPE": 2, "NARROW_CORRIDOR": 2,
        },
    }
    selected = []
    for split, modes in quotas.items():
        for mode, quota in modes.items():
            rows = [
                row for row in source["tasks"]
                if row["map_split"] == split and row["maneuver_mode"] == mode
            ]
            selected.extend(select_group(
                rows, quota,
                int(hashlib.sha256(
                    (str(args.seed) + split + mode).encode("utf-8")
                ).hexdigest()[:8], 16),
            ))
    selected.sort(key=lambda row: (row["map_split"], row["map_uuid"], row["task_id"]))
    payload = {
        "schema": "DEPCarV43DaggerTaskManifestV1",
        "seed": int(args.seed),
        "source_manifest": str(source_path),
        "source_manifest_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "source_task_manifest_sha256": claimed,
        "selection_contract": {
            "strategy": "round_robin_map_diverse_stratified_maneuver",
            "quotas": quotas,
            "runtime_policy_specialized_per_map": False,
            "test_maps_opened": False,
        },
        "tasks": selected,
        "counts_by_split": dict(sorted(Counter(row["map_split"] for row in selected).items())),
        "counts_by_mode": dict(sorted(Counter(row["maneuver_mode"] for row in selected).items())),
        "map_uuids": sorted({row["map_uuid"] for row in selected}),
        "test_maps_opened": False,
    }
    payload["task_manifest_sha256"] = canonical_sha256(payload)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output_path)
    print(json.dumps({
        "schema": payload["schema"], "status": "PASS", "output": str(output_path),
        "tasks": len(selected), "counts_by_split": payload["counts_by_split"],
        "counts_by_mode": payload["counts_by_mode"],
        "maps": len(payload["map_uuids"]),
        "task_manifest_sha256": payload["task_manifest_sha256"],
        "test_maps_opened": False,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
