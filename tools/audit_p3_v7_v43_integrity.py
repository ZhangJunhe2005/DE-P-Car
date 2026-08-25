#!/usr/bin/env python3
"""Independently re-audit the complete V4.3 DAgger authority chain."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dep_car/src"))
from dep_car.training.dataset import audit_multimodal_sample
from dep_car.training.p4_dataset import (
    _load_map_catalog,
    indexed_map_contract_aggregate,
    training_index_content_aggregate,
)


def resolve(value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


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


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def audit_sample(job):
    sample_root, entry, manifest_hash = job
    path = Path(sample_root) / entry["path"]
    errors = list(audit_multimodal_sample(path))
    actual_hash = sha256_file(path)
    if actual_hash != entry["content_sha256"]:
        errors.append("content_sha256")
    stat = path.stat()
    if stat.st_size != int(entry["size_bytes"]):
        errors.append("size_bytes")
    try:
        with np.load(path, allow_pickle=False) as data:
            sample_manifest = json.loads(str(data["manifest_json"]))
            for prefix in ("forward", "reverse"):
                for suffix in (
                    "trajectories", "feasible", "static_clearance", "guidance_cost"
                ):
                    name = "dagger_%s_%s" % (prefix, suffix)
                    if name not in data:
                        errors.append(name + "_shape")
                        continue
                    value = data[name]
                    shape_ok = (
                        value.ndim == 3 and value.shape[0] == 15
                        and value.shape[1] >= 2 and value.shape[2] == 6
                    ) if suffix == "trajectories" else value.shape == (15,)
                    if not shape_ok:
                        errors.append(name + "_shape")
                    elif value.dtype.kind in "fc" and not np.all(np.isfinite(value)):
                        errors.append(name + "_finite")
    except Exception as exc:
        return {"path": entry["path"], "errors": errors + [type(exc).__name__ + ":" + str(exc)[:120]]}
    metadata = sample_manifest.get("metadata", {})
    if sample_manifest.get("map_uuid") != entry["map_uuid"]:
        errors.append("map_uuid")
    if sample_manifest.get("split") != entry["split"]:
        errors.append("split")
    if metadata.get("sample_id") != entry["sample_id"]:
        errors.append("sample_id")
    if metadata.get("pilot_task_id") != entry["task_id"]:
        errors.append("task_id")
    if metadata.get("pilot_task_manifest_sha256") != manifest_hash:
        errors.append("task_manifest_sha256")
    if metadata.get("dagger_schema") != "DEPCarV43ClosedLoopObservationV1":
        errors.append("dagger_schema")
    if metadata.get("dagger_reobserved_state") is not True:
        errors.append("dagger_reobserved_state")
    if metadata.get("dagger_ground_truth_used_for_offline_map_label_only") is not True:
        errors.append("offline_ground_truth_boundary")
    raw = sample_manifest.get("raw_authority", {})
    expected_bag = entry["task_id"] + ".bag"
    if Path(str(raw.get("bag_path", ""))).name != expected_bag:
        errors.append("source_bag")
    timestamp = float(sample_manifest.get("timestamps", {}).get("lidar", float("nan")))
    if not math.isfinite(timestamp):
        errors.append("lidar_timestamp")
    return {
        "path": entry["path"], "task_id": entry["task_id"],
        "timestamp": timestamp, "errors": sorted(set(errors)),
    }


def percentile(values, fraction):
    if not values:
        return 0.0
    ordered = sorted(values)
    return float(ordered[min(len(ordered) - 1, int(fraction * (len(ordered) - 1)))])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--authority", default="data/p3_v7_v43/index/closed_loop_data_authority.json"
    )
    parser.add_argument(
        "--collection-config", default="dep_car/config/p5_closed_loop_v43_collection.yaml"
    )
    parser.add_argument(
        "--output", default="reports/p3_v7_v43_independent_integrity_audit.json"
    )
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.workers < 1:
        raise SystemExit("workers must be positive")

    authority_path, output = resolve(args.authority), resolve(args.output)
    authority = json.loads(authority_path.read_text(encoding="utf-8"))
    authority_content = dict(authority); claimed_authority = authority_content.pop("authority_sha256", None)
    manifest_path = resolve(authority["task_manifest"])
    state_path = resolve(authority["collection_state"])
    training_path = resolve(authority["training_index"])
    sequence_path = resolve(authority["sequence_index"])
    sample_root, maps_root = resolve(authority["sample_root"]), resolve(authority["maps_root"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_content = dict(manifest); claimed_manifest = manifest_content.pop("task_manifest_sha256", None)
    state = json.loads(state_path.read_text(encoding="utf-8"))
    training = json.loads(training_path.read_text(encoding="utf-8"))
    sequence = json.loads(sequence_path.read_text(encoding="utf-8"))
    sequence_content = dict(sequence); claimed_sequence = sequence_content.pop("content_sha256", None)
    config = yaml.safe_load(resolve(args.collection_config).read_text(encoding="utf-8"))

    tasks = {row["task_id"]: row for row in manifest["tasks"]}
    task_state = state.get("tasks", {})
    entries = training["entries"]
    entry_by_id = {row["sample_id"]: row for row in entries}
    rows = sequence["rows"]
    row_by_id = {row["sample_id"]: row for row in rows}
    indexed_paths = {str(row["path"]) for row in entries}
    actual_paths = {
        str(path.relative_to(sample_root)) for path in sample_root.rglob("*.npz")
    }
    bags_root = state_path.parent / "bags"
    bag_ids = {path.stem for path in bags_root.rglob("*.bag")}

    jobs = [(str(sample_root), row, claimed_manifest) for row in entries]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        sample_results = list(executor.map(audit_sample, jobs, chunksize=8))
    sample_errors = {
        row["path"]: row["errors"] for row in sample_results if row["errors"]
    }
    timestamps = defaultdict(list)
    for row in sample_results:
        if math.isfinite(row.get("timestamp", float("nan"))):
            timestamps[row["task_id"]].append(row["timestamp"])
    episode_spans = {
        task_id: max(values) - min(values) if values else 0.0
        for task_id, values in timestamps.items()
    }

    task_counts = Counter(row["task_id"] for row in entries)
    mode_counts = defaultdict(list)
    for task_id, count in task_counts.items():
        mode_counts[tasks[task_id]["maneuver_mode"]].append(count)
    recovered = [
        task_id for task_id, row in task_state.items()
        if row.get("recovered_extraction_without_gazebo") is True
    ]
    recovered_ratios = {}
    recovery_evidence_ok = True
    for task_id in recovered:
        mode = tasks[task_id]["maneuver_mode"]
        baseline = [
            task_counts[other] for other in task_counts
            if tasks[other]["maneuver_mode"] == mode and other not in recovered
        ]
        recovered_ratios[task_id] = task_counts[task_id] / max(1.0, statistics.median(baseline))
        evidence = Path(task_state[task_id].get("recovery_evidence", ""))
        recovery_evidence_ok = recovery_evidence_ok and evidence.is_file()
        if evidence.is_file():
            payload = json.loads(evidence.read_text(encoding="utf-8"))
            recovery_evidence_ok = recovery_evidence_ok and (
                payload.get("status") == "RECOVERED_DAGGER_HARD_VETO_OBSERVATION"
                and payload.get("gazebo_rerun") is False
            )

    task_splits = defaultdict(set)
    for task in manifest["tasks"]:
        task_splits[task["map_uuid"]].add(task["map_split"])
    entry_splits = defaultdict(set)
    for entry in entries:
        entry_splits[entry["map_uuid"]].add(entry["split"])

    patterns = Counter()
    reverse = reverse_forward = multi = stop = truncated = disagreements = 0
    sequence_errors = []
    expansions, attempt_statuses, target_sources = [], Counter(), Counter()
    for row in rows:
        source = entry_by_id.get(row["sample_id"])
        if source is None:
            sequence_errors.append(row["sample_id"] + ":missing_training_entry")
            continue
        if any(row[key] != source[key] for key in ("task_id", "map_uuid", "split")):
            sequence_errors.append(row["sample_id"] + ":identity")
        if row["source_content_sha256"] != source["content_sha256"]:
            sequence_errors.append(row["sample_id"] + ":content")
        gears, mask = row["sequence_gears"], row["sequence_mask"]
        active = [int(gear) for gear, keep in zip(gears, mask) if keep]
        contiguous = mask == sorted(mask, reverse=True)
        if (
            len(gears) != 6 or len(mask) != 6 or not contiguous
            or any(value not in (-1, 1) for value in active)
            or any(first == second for first, second in zip(active, active[1:]))
            or any(int(gear) != 0 for gear, keep in zip(gears, mask) if not keep)
            or np.asarray(row["action_plan_endpoints_body"]).shape != (6, 3)
            or not np.all(np.isfinite(row["action_plan_endpoints_body"]))
        ):
            sequence_errors.append(row["sample_id"] + ":sequence_contract")
        pattern = "-".join("F" if gear > 0 else "R" for gear in active) or "STOP"
        patterns[pattern] += 1
        reverse += int(-1 in active); multi += int(len(active) >= 2)
        reverse_forward += int(any(a < 0 and b > 0 for a, b in zip(active, active[1:])))
        stop += int(not active); truncated += int(row["teacher_plan_truncated"])
        disagreements += int(bool(active) and active[0] != row["diagnostic_one_step_teacher_gear"])
        expansions.append(int(row["teacher_plan_expansions"]))
        target_sources[row["teacher_target_source"]] += 1
        attempt_statuses.update(item["status"] for item in row["teacher_plan_attempts"])
        if row["teacher_plan_status"] != ("SUCCESS" if active else "STOP_NO_EXACT_SIGNED_PLAN"):
            sequence_errors.append(row["sample_id"] + ":teacher_status")

    selected_maps = {entry["map_uuid"] for entry in entries}
    maps = _load_map_catalog(maps_root, selected_maps)
    map_aggregate = indexed_map_contract_aggregate(maps)["aggregate_sha256"]
    implementation_errors = []
    for name, relative in authority["implementation"].items():
        if name.endswith("_sha256"):
            continue
        expected = authority["implementation"].get(name + "_sha256")
        path = resolve(relative)
        if not path.is_file() or sha256_file(path) != expected:
            implementation_errors.append(name)

    file_hashes = {
        "authority": sha256_file(authority_path),
        "manifest": sha256_file(manifest_path),
        "collection_state": sha256_file(state_path),
        "training_index": sha256_file(training_path),
        "sequence_index": sha256_file(sequence_path),
    }
    gates = {
        "authority_internal_sha256": claimed_authority == canonical_sha256(authority_content),
        "authority_references_current_files": (
            file_hashes["manifest"] == authority["task_manifest_sha256"]
            and file_hashes["collection_state"] == authority["collection_state_sha256"]
            and file_hashes["training_index"] == authority["training_index_sha256"]
            and file_hashes["sequence_index"] == authority["sequence_index_sha256"]
        ),
        "manifest_internal_sha256": claimed_manifest == canonical_sha256(manifest_content),
        "collection_config_sha256": state.get("config_sha256") == canonical_sha256(config),
        "task_set_exactly_matches_manifest": set(task_state) == set(tasks),
        "all_episodes_complete": all(row.get("status") == "COMPLETE" for row in task_state.values()),
        "one_current_bag_per_task": bag_ids == set(tasks),
        "recovery_evidence_valid": recovery_evidence_ok,
        # Full-timeout failure episodes are expected to contribute more frames
        # than missions that reach the goal early.  The timestamp gate below
        # is authoritative; this looser ratio catches only gross imbalance.
        "recovered_episode_weight_bounded": all(0.50 <= value <= 2.50 for value in recovered_ratios.values()),
        "episode_timestamp_window_bounded": all(
            value <= float(config["collection"]["episode_timeout_s"]) + 0.25
            for value in episode_spans.values()
        ),
        "sample_root_exactly_matches_index": actual_paths == indexed_paths,
        "sample_ids_unique": len(entry_by_id) == len(entries),
        "sample_content_hashes_and_contracts": not sample_errors,
        "content_aggregate_sha256": (
            training_index_content_aggregate(entries)
            == training["content_aggregate_sha256"]
            == authority["content_aggregate_sha256"]
        ),
        "sequence_internal_sha256": claimed_sequence == canonical_sha256(sequence_content),
        "sequence_rows_exactly_match_samples": (
            len(row_by_id) == len(rows) == len(entries)
            and set(row_by_id) == set(entry_by_id)
        ),
        "sequence_contracts_valid": not sequence_errors,
        "sequence_statistics_match_authority": (
            dict(sorted(patterns.items())) == authority["sequence_patterns"]
            and reverse == authority["reverse_sequence_samples"]
            and reverse_forward == authority["reverse_then_forward_samples"]
            and multi == authority["multi_action_samples"]
            and stop == authority["stop_samples"]
            and truncated == authority["truncated_samples"]
            and disagreements == authority["diagnostic_one_step_teacher_disagreements"]
        ),
        "exact_teacher_plan_success": stop == 0 and attempt_statuses.get("SUCCESS", 0) == len(rows),
        "train_validation_maps_disjoint": (
            all(len(value) == 1 for value in task_splits.values())
            and all(len(value) == 1 for value in entry_splits.values())
        ),
        "map_contract_aggregate_sha256": map_aggregate == authority["map_contract_aggregate_sha256"],
        "implementation_hashes_current": not implementation_errors,
        "sealed_test_split": (
            manifest.get("test_maps_opened") is False
            and sequence.get("test_split_opened") is False
            and authority.get("test_split_opened") is False
            and "test" not in training.get("splits", [])
        ),
    }
    report = {
        "schema": "DEPCarV43IndependentIntegrityAuditV1",
        "status": "PASS" if all(gates.values()) else "FAIL",
        "errors": sorted(key for key, value in gates.items() if not value),
        "gates": gates,
        "authority": str(authority_path),
        "authority_file_sha256": file_hashes["authority"],
        "authority_content_sha256": claimed_authority,
        "file_hashes": file_hashes,
        "samples": len(entries), "episodes": len(tasks),
        "sample_error_count": len(sample_errors),
        "sample_errors": dict(list(sample_errors.items())[:100]),
        "sequence_errors": sequence_errors[:100],
        "implementation_errors": implementation_errors,
        "per_task_samples": {
            "minimum": min(task_counts.values()),
            "median": statistics.median(task_counts.values()),
            "maximum": max(task_counts.values()),
        },
        "recovered_tasks": recovered,
        "recovered_sample_ratios_to_same_mode_median": recovered_ratios,
        "maximum_episode_timestamp_span_s": max(episode_spans.values()),
        "sequence_patterns": dict(sorted(patterns.items())),
        "teacher": {
            "final_success_rate": (len(rows) - stop) / max(1, len(rows)),
            "attempt_statuses": dict(sorted(attempt_statuses.items())),
            "target_sources": dict(sorted(target_sources.items())),
            "expansions_median": statistics.median(expansions),
            "expansions_p95": percentile(expansions, 0.95),
            "expansions_maximum": max(expansions),
        },
        "test_split_opened": False,
    }
    atomic_json(output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["status"] == "PASS" else 2)


if __name__ == "__main__":
    main()
