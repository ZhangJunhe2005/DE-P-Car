#!/usr/bin/env python3
"""Recover P3 tasks whose finalized rosbag raced offline extraction."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dep_car/src"))
from dep_car.training.dataset import audit_multimodal_sample
from dep_car.training.pilot import canonical_sha256


RECOVERABLE_ERROR = "RuntimeError: offline extraction failed with code 1"


def atomic_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def resolve_project_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def extraction_command(task, collection, manifest, work_root, episode, bag):
    return [
        "rosrun", "dep_car_dataset", "extract_multimodal_bag.py", str(bag),
        "--output", str(work_root / "samples"),
        "--map-uuid", task["map_uuid"],
        "--map-hash", task["map_occupancy_sha256"],
        "--simulator-seed", str(task["map_seed"]),
        "--stride", str(collection["extraction_stride"]),
        "--task-id", task["task_id"],
        "--task-manifest-sha256", manifest["task_manifest_sha256"],
        "--maneuver-mode", task["maneuver_mode"],
        "--episode-result", str(episode),
    ]


def eligible_tasks(manifest, state, work_root):
    selected = []
    for task in manifest["tasks"]:
        task_id = task["task_id"]
        row = state.get("tasks", {}).get(task_id, {})
        bag = work_root / "bags" / task["map_split"] / (task_id + ".bag")
        episode = work_root / "episodes" / (task_id + ".json")
        if row.get("status") != "FAILED" or row.get("error") != RECOVERABLE_ERROR:
            continue
        if not bag.is_file() or not episode.is_file():
            continue
        try:
            episode_status = json.loads(episode.read_text(encoding="utf-8")).get("status")
        except (OSError, ValueError):
            continue
        if episode_status not in ("SUCCESS", "TIMEOUT"):
            continue
        selected.append((task, bag, episode))
    return selected


def recover_one(task, bag, episode, collection, manifest, work_root):
    task_id = task["task_id"]
    log_root = work_root / "logs" / task_id / "extraction_recovery"
    log_root.mkdir(parents=True, exist_ok=True)
    log_path = log_root / ("attempt_%02d.log" % (len(tuple(log_root.glob("attempt_*.log"))) + 1))
    environment = dict(os.environ)
    for variable in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        environment[variable] = "1"
    command = extraction_command(task, collection, manifest, work_root, episode, bag)
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(
            command, env=environment, stdout=log, stderr=subprocess.STDOUT,
            text=True, timeout=900.0,
        )
    if result.returncode != 0:
        raise RuntimeError("offline extraction returned code %d" % result.returncode)
    samples = sorted(
        (work_root / "samples" / task["map_uuid"]).glob(task_id + "-*.npz")
    )
    if len(samples) < int(collection["minimum_samples_per_episode"]):
        raise RuntimeError("only %d samples were recovered" % len(samples))
    failures = {str(path): audit_multimodal_sample(path) for path in samples}
    failures = {path: errors for path, errors in failures.items() if errors}
    if failures:
        raise RuntimeError(
            "recovered sample audit failed: "
            + json.dumps(failures, sort_keys=True)[:500]
        )
    outcome = json.loads(episode.read_text(encoding="utf-8"))
    return {
        "samples": len(samples),
        "bag_size_bytes": bag.stat().st_size,
        "episode_status": outcome["status"],
        "illegal_shift_count": int(outcome.get("illegal_shift_count", 0)),
        "candidate_messages": int(outcome.get("candidate_messages", 0)),
        "zero_feasible_messages": int(outcome.get("zero_feasible_messages", 0)),
        "recovery_log": str(log_path),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--maximum-tasks", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or args.maximum_tasks < 0:
        raise ValueError("worker/task limits are invalid")
    if args.workers > (os.cpu_count() or 1):
        raise ValueError("workers exceed visible CPU threads")

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    collection = config["collection"]
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    state_path = args.work_root / "collection_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("task_manifest_sha256") != manifest.get("task_manifest_sha256"):
        raise RuntimeError("collection state belongs to another manifest")
    if state.get("config_sha256") != canonical_sha256(config):
        raise RuntimeError("collection state belongs to another config")

    selected = eligible_tasks(manifest, state, args.work_root)
    if args.maximum_tasks:
        selected = selected[:args.maximum_tasks]
    preview = {
        "schema": "DEPCarP3ExtractionRecoveryV1",
        "status": "DRY_RUN_PASS" if args.dry_run else "RUNNING",
        "recoverable_tasks": len(selected),
        "parallel_workers": min(args.workers, max(1, len(selected))),
        "preserves_rosbags": True,
        "reruns_gazebo": False,
    }
    if args.dry_run:
        print(json.dumps(preview, indent=2, sort_keys=True))
        return 0
    if not selected:
        preview["status"] = "PASS"
        print(json.dumps(preview, indent=2, sort_keys=True))
        return 0
    free_gb = shutil.disk_usage(args.work_root).free / (1024 ** 3)
    if free_gb < float(collection["minimum_free_disk_gb"]):
        raise RuntimeError("free disk is below the P3 safety floor")

    lock_path = args.work_root / ".collection.lock"
    with lock_path.open("w", encoding="utf-8") as lock_stream:
        try:
            fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("another P3 process owns " + str(lock_path))
        counts = Counter()
        errors = []
        state_lock = threading.Lock()
        with ThreadPoolExecutor(max_workers=args.workers, thread_name_prefix="p3-recover") as pool:
            futures = {
                pool.submit(
                    recover_one, task, bag, episode, collection, manifest,
                    args.work_root,
                ): task["task_id"]
                for task, bag, episode in selected
            }
            for future in as_completed(futures):
                task_id = futures[future]
                try:
                    result = future.result()
                    with state_lock:
                        row = state["tasks"][task_id]
                        row.update({
                            "status": "COMPLETE",
                            "completed_at_unix": time.time(),
                            "extraction_recovered": True,
                            "extraction_recovered_at_unix": time.time(),
                            **result,
                        })
                        row.pop("error", None)
                        row.pop("failed_at_unix", None)
                        state["updated_at_unix"] = time.time()
                        atomic_json(state_path, state)
                    counts["COMPLETE"] += 1
                    print("P3 extraction recovery task=%s status=COMPLETE" % task_id, flush=True)
                except Exception as exc:
                    message = type(exc).__name__ + ": " + str(exc)
                    with state_lock:
                        row = state["tasks"][task_id]
                        row["extraction_recovery_error"] = message
                        row["updated_at_unix"] = time.time()
                        state["updated_at_unix"] = time.time()
                        atomic_json(state_path, state)
                    counts["FAILED"] += 1
                    errors.append(task_id + ":" + message)
                    print("P3 extraction recovery task=%s status=FAILED" % task_id, flush=True)

    totals = {
        name: sum(row.get("status") == name for row in state["tasks"].values())
        for name in ("COMPLETE", "FAILED", "INTERRUPTED", "RUNNING")
    }
    payload = {
        "schema": "DEPCarP3ExtractionRecoveryV1",
        "status": "PASS" if not errors else "PARTIAL",
        "parallel_workers": min(args.workers, len(selected)),
        "this_run": dict(counts),
        "state_totals": totals,
        "errors": errors,
        "reran_gazebo": False,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
