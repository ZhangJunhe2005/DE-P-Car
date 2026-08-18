#!/usr/bin/env python3
"""Sign repeated P3 failures caused by generator-admitted invalid goal poses."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dep_car/src"))
sys.path.insert(0, str(ROOT / "tools"))

import audit_p3_footprint_upgrade as geometry
from dep_car.global_planner.hybrid_astar import HybridAStar
from dep_car.training.pilot import canonical_sha256


SCHEMA = "DEPCarP3CollectionExclusionAuthorityV1"
EXCLUDED_STATUS = "EXCLUDED_INVALID_GOAL"


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def verify_manifest(path):
    raw = Path(path).read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    content = dict(payload)
    claimed = content.pop("task_manifest_sha256", "")
    if claimed != canonical_sha256(content):
        raise RuntimeError("task manifest internal SHA-256 mismatch")
    return payload, hashlib.sha256(raw).hexdigest()


def latest_runtime_log(state_path, task_id, attempt):
    path = (
        state_path.parent / "logs" / task_id
        / ("attempt_%02d" % int(attempt)) / "roslaunch.log"
    )
    if not path.is_file():
        raise FileNotFoundError(path)
    content = path.read_text(encoding="utf-8", errors="replace")
    return path, "GOAL_FOOTPRINT_COLLISION" in content


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--maps", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest, manifest_file_sha256 = verify_manifest(args.manifest)
    state_raw = args.state.read_bytes()
    state = json.loads(state_raw.decode("utf-8"))
    if state.get("task_manifest_sha256") != manifest["task_manifest_sha256"]:
        raise RuntimeError("collection state/manifest mismatch")
    tasks = {row["task_id"]: row for row in manifest["tasks"]}
    failed_ids = sorted(
        task_id for task_id, row in state.get("tasks", {}).items()
        if row.get("status") == "FAILED"
    )
    if not failed_ids:
        raise RuntimeError("collection has no failed tasks to qualify")

    map_uuids = {tasks[task_id]["map_uuid"] for task_id in failed_ids}
    if any(tasks[task_id].get("map_split") not in ("train", "validation") for task_id in failed_ids):
        raise RuntimeError("test tasks cannot be opened by exclusion qualification")
    specs, _contract = geometry.load_map_specs(
        args.maps, map_uuids, expected_aggregate=None
    )
    planner = HybridAStar()
    entries = []
    for task_id in failed_ids:
        task = tasks[task_id]
        row = state["tasks"][task_id]
        if int(row.get("attempt", 0)) < 2:
            raise RuntimeError("invalid goal exclusion requires two failed attempts")
        grid = geometry.grid_from_spec(specs[task["map_uuid"]])
        start_valid, start_reason, start_clearance, safe_primitives = (
            planner.validate_start_pose(grid, tuple(task["start"]))
        )
        goal_valid, goal_reason, goal_clearance = planner.validate_goal_pose(
            grid, tuple(task["goal"])
        )
        runtime_log, runtime_observed = latest_runtime_log(
            args.state, task_id, row["attempt"]
        )
        if (
            not start_valid
            or goal_valid
            or goal_reason != "GOAL_FOOTPRINT_COLLISION"
            or not runtime_observed
        ):
            raise RuntimeError(
                "task is not a qualified invalid-goal exclusion: " + task_id
            )
        entries.append({
            "task_id": task_id,
            "map_uuid": task["map_uuid"],
            "map_split": task["map_split"],
            "maneuver_mode": task["maneuver_mode"],
            "attempts": int(row["attempt"]),
            "start": list(map(float, task["start"])),
            "goal": list(map(float, task["goal"])),
            "start_reason": start_reason,
            "start_footprint_clearance_m": float(start_clearance),
            "start_safe_ackermann_primitives": int(safe_primitives),
            "reason": goal_reason,
            "goal_footprint_clearance_m": float(goal_clearance),
            "runtime_goal_rejection_observed": True,
            "runtime_log": str(runtime_log.resolve()),
            "runtime_log_sha256": file_sha256(runtime_log),
            "previous_collection_error": row.get("error", ""),
            "test_map_opened": False,
        })

    payload = {
        "schema": SCHEMA,
        "status": "PASS",
        "qualification_authority": True,
        "reason": "generator_missing_runtime_endpoint_footprint_gate",
        "evaluator": "HybridAStar.validate_goal_pose",
        "task_manifest": str(args.manifest.resolve()),
        "task_manifest_file_sha256": manifest_file_sha256,
        "task_manifest_sha256": manifest["task_manifest_sha256"],
        "collection_state_before_sha256": hashlib.sha256(state_raw).hexdigest(),
        "excluded_tasks": len(entries),
        "entries": entries,
        "replacement_tasks_required": len(entries),
        "source_samples_modified": False,
        "test_map_opened": False,
        "test_npz_opened": False,
    }
    payload["exclusion_authority_sha256"] = canonical_sha256(payload)
    output = args.state.parent / "collection_exclusion_authority.json"
    summary = {
        "status": "DRY_RUN_PASS" if args.dry_run else "PASS",
        "excluded_tasks": len(entries),
        "task_ids": [row["task_id"] for row in entries],
        "replacement_tasks_required": len(entries),
        "authority": str(output.resolve()),
        "exclusion_authority_sha256": payload["exclusion_authority_sha256"],
        "test_map_opened": False,
    }
    if args.dry_run:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    archive = args.state.with_name(
        "collection_state.before_invalid_goal_exclusions.%d.json" % int(time.time())
    )
    shutil.copy2(args.state, archive)
    atomic_json(output, payload)
    for entry in entries:
        row = state["tasks"][entry["task_id"]]
        row["original_status"] = row.get("status")
        row["original_error"] = row.get("error", "")
        row["status"] = EXCLUDED_STATUS
        row["excluded_at_unix"] = time.time()
        row["exclusion_reason"] = entry["reason"]
        row["exclusion_authority"] = str(output.resolve())
        row["exclusion_authority_sha256"] = payload[
            "exclusion_authority_sha256"
        ]
        row.pop("error", None)
        row.pop("failed_at_unix", None)
    state["collection_exclusion"] = {
        "schema": SCHEMA,
        "status": "PASS",
        "authority": str(output.resolve()),
        "authority_file_sha256": file_sha256(output),
        "exclusion_authority_sha256": payload["exclusion_authority_sha256"],
        "excluded_tasks": len(entries),
        "replacement_tasks_required": len(entries),
        "state_archive": str(archive.resolve()),
    }
    state["updated_at_unix"] = time.time()
    atomic_json(args.state, state)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
