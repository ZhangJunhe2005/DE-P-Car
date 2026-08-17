#!/usr/bin/env python3
"""Audit P3 coverage, provenance and Oracle-of-15 candidate expressiveness."""

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dep_car/src"))
from dep_car.training.dataset import audit_multimodal_sample
from dep_car.training.pilot import PILOT_MANEUVER_MODES, candidate_expressiveness, canonical_sha256


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values, percentile_value, default=None):
    finite = [value for value in values if np.isfinite(value)]
    return default if not finite else float(np.percentile(finite, percentile_value))


def mode_summary(rows):
    feasible = [row["feasible_count"] for row in rows]
    return {
        "samples": len(rows),
        "zero_feasible_rate": float(np.mean([row["zero_feasible"] for row in rows])) if rows else 1.0,
        "feasible_candidates_median": float(np.median(feasible)) if feasible else 0.0,
        "oracle_route_error_p50_m": percentile([row["oracle_route_error_m"] for row in rows], 50),
        "oracle_route_error_p90_m": percentile([row["oracle_route_error_m"] for row in rows], 90),
        "oracle_route_progress_p50_m": percentile([row["oracle_route_progress_m"] for row in rows], 50, 0.0),
        "oracle_static_clearance_p10_m": percentile([row["oracle_static_clearance_m"] for row in rows], 10, 0.0),
    }


def verify_task_manifest(manifest):
    expected = manifest.get("task_manifest_sha256", "")
    content = dict(manifest)
    content.pop("task_manifest_sha256", None)
    return expected and expected == canonical_sha256(content)


def inspect_sample(path, task_by_id, task_manifest_sha256):
    violations = audit_multimodal_sample(path)
    if violations:
        return {"path": str(path), "failure": violations}
    with np.load(str(path), allow_pickle=False) as data:
        manifest = json.loads(str(data["manifest_json"]))
        metadata = manifest.get("metadata", {})
        task_id = metadata.get("pilot_task_id", "")
        task = task_by_id.get(task_id)
        if task is None:
            return {"path": str(path), "failure": ["unknown_pilot_task_id"]}
        if metadata.get("pilot_task_manifest_sha256") != task_manifest_sha256:
            return {"path": str(path), "failure": ["pilot_task_manifest_sha256"]}
        if manifest.get("maneuver_mode") != task["maneuver_mode"]:
            return {"path": str(path), "failure": ["maneuver_mode_task_mismatch"]}
        if manifest.get("map_uuid") != task["map_uuid"] or manifest.get("split") != task["map_split"]:
            return {"path": str(path), "failure": ["map_task_mismatch"]}
        metric = candidate_expressiveness(
            data["trajectories"], data["feasible"], data["local_path"],
            data["guidance_cost"], data["static_clearance"],
        )
        metric["mode"] = manifest["maneuver_mode"]
        raw = manifest["raw_authority"]
        return {
            "path": str(path), "metric": metric, "task_id": task_id,
            "mode": manifest["maneuver_mode"], "map_uuid": manifest["map_uuid"],
            "split": manifest["split"], "requested_gear": str(int(data["requested_gear"])),
            "preprocessing_hash": manifest["preprocessing"]["lidar_bev"]["sha256"],
            "bag_path": raw["bag_path"], "bag_sha256": raw["bag_sha256"],
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/p3_pilot/task_manifest.json")
    parser.add_argument("--config", type=Path, default=ROOT / "dep_car/config/p3_pilot.yaml")
    parser.add_argument("--collection-state", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--skip-bag-hash", action="store_true", help="diagnostic only; final P3 audit must verify bags")
    parser.add_argument("--workers", type=int, default=8, help="parallel sample and bag audit workers")
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    gates = config["gates"]
    task_manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    errors = []
    if not verify_task_manifest(task_manifest):
        errors.append("task_manifest_sha256")
    if task_manifest.get("schema") != "DEPCarPilotTaskManifestV1":
        errors.append("task_manifest_schema")
    if task_manifest.get("generator_contract", {}).get("partial"):
        errors.append("partial_task_manifest")
    task_ids = [task.get("task_id") for task in task_manifest.get("tasks", [])]
    if None in task_ids or len(task_ids) != len(set(task_ids)):
        errors.append("task_manifest_task_ids")
    task_by_id = {task["task_id"]: task for task in task_manifest["tasks"]}
    files = sorted(args.dataset.glob("*/*.npz"))
    modes = Counter()
    maps_by_split = defaultdict(set)
    map_declared_splits = defaultdict(set)
    samples_by_map = Counter()
    samples_by_task = Counter()
    requested_gears = Counter()
    preprocessing_hashes = set()
    metric_rows = []
    metrics_by_mode = defaultdict(list)
    sample_failures = {}
    bag_claims = defaultdict(set)

    with ThreadPoolExecutor(max_workers=args.workers, thread_name_prefix="p3-audit") as executor:
        results = executor.map(
            lambda path: inspect_sample(path, task_by_id, task_manifest.get("task_manifest_sha256")),
            files,
        )
        for row in results:
            if "failure" in row:
                sample_failures[row["path"]] = row["failure"]
                continue
            metric = row["metric"]
            metric_rows.append(metric)
            metrics_by_mode[row["mode"]].append(metric)
            modes[row["mode"]] += 1
            maps_by_split[row["split"]].add(row["map_uuid"])
            map_declared_splits[row["map_uuid"]].add(row["split"])
            samples_by_map[row["map_uuid"]] += 1
            samples_by_task[row["task_id"]] += 1
            requested_gears[row["requested_gear"]] += 1
            preprocessing_hashes.add(row["preprocessing_hash"])
            bag_claims[Path(row["bag_path"])].add(row["bag_sha256"])

    if sample_failures:
        errors.append("sample_contract_or_task_binding")
    overlap = sorted(map_uuid for map_uuid, splits in map_declared_splits.items() if len(splits) > 1)
    if overlap:
        errors.append("map_split_overlap")
    sample_count = len(metric_rows)
    unique_maps = len(samples_by_map)
    if sample_count < int(gates["minimum_samples"]):
        errors.append("minimum_samples")
    if sample_count > int(gates["maximum_samples"]):
        errors.append("maximum_samples")
    if unique_maps < int(gates["minimum_maps"]):
        errors.append("minimum_maps")
    if len(maps_by_split["validation"]) < int(gates["minimum_validation_maps"]):
        errors.append("minimum_validation_maps")
    if len(maps_by_split["test"]) < int(gates["minimum_test_maps"]):
        errors.append("minimum_test_maps")
    for mode in PILOT_MANEUVER_MODES:
        required = 1 if mode == "NORMAL" else int(gates["minimum_samples_per_non_normal_mode"])
        if modes[mode] < required:
            errors.append("mode_coverage_" + mode)
    overall = mode_summary(metric_rows)
    if overall["zero_feasible_rate"] > float(gates["maximum_zero_feasible_rate"]):
        errors.append("zero_feasible_rate")
    if overall["feasible_candidates_median"] < float(gates["minimum_median_feasible_candidates"]):
        errors.append("median_feasible_candidates")
    if (
        overall["oracle_route_error_p90_m"] is None
        or overall["oracle_route_error_p90_m"] > float(gates["maximum_oracle_route_error_p90_m"])
    ):
        errors.append("oracle_route_error_p90")
    summaries = {mode: mode_summary(metrics_by_mode[mode]) for mode in PILOT_MANEUVER_MODES}
    for mode, summary in summaries.items():
        if summary["samples"] and summary["zero_feasible_rate"] > float(gates["maximum_per_mode_zero_feasible_rate"]):
            errors.append("mode_zero_feasible_rate_" + mode)
    reverse_fraction = requested_gears["-1"] / max(1, sum(requested_gears.values()))
    if reverse_fraction < float(gates["minimum_reverse_sample_fraction"]):
        errors.append("reverse_sample_fraction")
    if gates.get("require_single_preprocessing_hash", True) and len(preprocessing_hashes) != 1:
        errors.append("preprocessing_hash_count")

    bag_audit = {}

    def inspect_bag(item):
        bag_path, claims = item
        row = {"claimed_sha256": sorted(claims), "exists": bag_path.is_file()}
        if bag_path.is_file() and not args.skip_bag_hash:
            row["actual_sha256"] = file_sha256(bag_path)
        return bag_path, claims, row

    with ThreadPoolExecutor(max_workers=args.workers, thread_name_prefix="p3-bag-hash") as executor:
        bag_results = executor.map(inspect_bag, bag_claims.items())
        for bag_path, claims, row in bag_results:
            if len(claims) != 1:
                errors.append("conflicting_bag_hash_claim")
            if not row["exists"]:
                errors.append("missing_authoritative_bag")
            elif not args.skip_bag_hash and row["actual_sha256"] not in claims:
                errors.append("authoritative_bag_hash")
            bag_audit[str(bag_path)] = row

    state_summary = {}
    collection_state_path = args.collection_state or args.dataset.parent / "collection_state.json"
    if collection_state_path.is_file():
        collection = json.loads(collection_state_path.read_text(encoding="utf-8"))
        if collection.get("task_manifest_sha256") != task_manifest.get("task_manifest_sha256"):
            errors.append("collection_task_manifest_sha256")
        if collection.get("config_sha256") != canonical_sha256(config):
            errors.append("collection_config_sha256")
        state_counts = Counter(row.get("status", "UNKNOWN") for row in collection.get("tasks", {}).values())
        completed = [row for row in collection.get("tasks", {}).values() if row.get("status") == "COMPLETE"]
        completion_rate = state_counts["COMPLETE"] / max(1, len(task_manifest["tasks"]))
        illegal = sum(int(row.get("illegal_shift_count", 0)) for row in completed)
        state_summary = {"counts": dict(state_counts), "completion_rate": completion_rate, "illegal_shift_count": illegal}
        if completion_rate < float(gates["minimum_episode_completion_rate"]):
            errors.append("episode_completion_rate")
        if illegal > int(gates["maximum_illegal_shift_count"]):
            errors.append("illegal_shift_count")
        for task_id, count in samples_by_task.items():
            state_row = collection.get("tasks", {}).get(task_id, {})
            if state_row.get("status") != "COMPLETE":
                errors.append("sample_from_incomplete_task")
            if int(state_row.get("samples", -1)) != count:
                errors.append("collection_sample_count_mismatch")
    else:
        errors.append("collection_state_missing")

    payload = {
        "schema": "DEPCarP3PilotDatasetAuditV1",
        "status": "PASS" if not errors else "FAIL",
        "errors": sorted(set(errors)),
        "samples": sample_count,
        "sample_files": len(files),
        "sample_failures": sample_failures,
        "maps": unique_maps,
        "map_counts_by_split": {split: len(values) for split, values in maps_by_split.items()},
        "map_split_overlap": overlap,
        "samples_by_mode": dict(modes),
        "samples_by_map": dict(samples_by_map),
        "samples_by_task": dict(samples_by_task),
        "requested_gear_counts": dict(requested_gears),
        "reverse_sample_fraction": reverse_fraction,
        "preprocessing_hashes": sorted(preprocessing_hashes),
        "candidate_expressiveness": {"overall": overall, "by_mode": summaries},
        "authoritative_bags": bag_audit,
        "collection_state": state_summary,
        "task_manifest_sha256": task_manifest.get("task_manifest_sha256"),
        "parallel_audit_workers": args.workers,
    }
    report = args.report or args.dataset.parent / "p3_pilot_audit.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    raise SystemExit(0 if payload["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
