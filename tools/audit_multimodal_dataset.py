#!/usr/bin/env python3
"""Audit V2 samples, timestamp skews, map splits and gear coverage."""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dep_car/src"))
from dep_car.training.dataset import MULTIMODAL_SCHEMA_VERSION, audit_multimodal_sample


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    failures, splits, gears, skews, point_counts = {}, defaultdict(set), Counter(), defaultdict(list), []
    raw_authorities, maneuvers, preprocessing_hashes = Counter(), Counter(), set()
    files = sorted(args.root.glob("*/*.npz"))
    for path in files:
        errors = audit_multimodal_sample(path)
        if errors:
            failures[str(path.relative_to(args.root))] = errors
            continue
        with np.load(str(path), allow_pickle=False) as data:
            manifest = json.loads(str(data["manifest_json"]))
            splits[manifest["split"]].add(manifest["map_uuid"])
            gears[str(int(data["requested_gear"]))] += 1
            point_counts.append(len(data["lidar_points"]))
            raw_authorities[manifest["raw_authority"]["kind"]] += 1
            maneuvers[manifest["maneuver_mode"]] += 1
            preprocessing_hashes.add(manifest["preprocessing"]["lidar_bev"]["sha256"])
            for name, value in manifest["skew_s"].items():
                skews[name].append(value)
    overlap = (splits["train"] & splits["validation"]) | (splits["train"] & splits["test"]) | (splits["validation"] & splits["test"])
    payload = {
        "schema": "StaticAckermannDatasetAuditV2",
        "sample_schema": MULTIMODAL_SCHEMA_VERSION,
        "status": "PASS" if files and not failures and not overlap else "FAIL",
        "samples": len(files),
        "valid_samples": len(files) - len(failures),
        "failures": failures,
        "map_counts": {name: len(values) for name, values in splits.items()},
        "map_split_overlap": sorted(overlap),
        "requested_gear_counts": dict(gears),
        "maneuver_mode_counts": dict(maneuvers),
        "raw_authority_counts": dict(raw_authorities),
        "preprocessing_hashes": sorted(preprocessing_hashes),
        "lidar_point_count": {
            "minimum": min(point_counts) if point_counts else 0,
            "median": float(np.median(point_counts)) if point_counts else 0,
            "maximum": max(point_counts) if point_counts else 0,
        },
        "skew_s_p99": {name: float(np.percentile(values, 99)) for name, values in skews.items() if values},
    }
    report = args.report or args.root / "audit_v2.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    raise SystemExit(0 if payload["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
