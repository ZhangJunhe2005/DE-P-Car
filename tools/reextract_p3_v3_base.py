#!/usr/bin/env python3
"""Non-destructively re-extract the P3 development bags with context V2.

The legacy sample tree is never modified.  Only train/validation tasks are
selected from the frozen base manifest, and every result is written to a new
P3 V3 root with resumable, content-addressed state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dep_car" / "src"))

from dep_car.training.dataset import audit_multimodal_sample
from dep_car.training.pilot import canonical_sha256


STATE_SCHEMA = "DEPCarP3V3BaseReextractStateV1"
ALLOWED_CONTEXTS = ("MISSION", "RECOVERY")
ALLOWED_SPLITS = ("train", "validation")


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def verify_manifest(manifest):
    claimed = manifest.get("task_manifest_sha256", "")
    content = dict(manifest)
    content.pop("task_manifest_sha256", None)
    if claimed != canonical_sha256(content):
        raise ValueError("base task manifest SHA-256 mismatch")


def read_context(path):
    with np.load(str(path), allow_pickle=False) as data:
        manifest = json.loads(str(data["manifest_json"].item()))
        context = str(manifest.get("metadata", {}).get("candidate_context", ""))
        source_bag_sha256 = str(
            manifest.get("metadata", {}).get("source_bag_sha256", "")
        )
        requested_gear = int(np.asarray(data["requested_gear"]).item())
    if context not in ALLOWED_CONTEXTS:
        raise RuntimeError("re-extracted sample has no authoritative candidate_context")
    if requested_gear not in (-1, 1):
        raise RuntimeError("re-extracted sample requested_gear is not a drive gear")
    if len(source_bag_sha256) != 64:
        raise RuntimeError("re-extracted sample has no source bag SHA-256")
    return context, requested_gear, source_bag_sha256


def task_result_is_reusable(row, output_root):
    if row.get("status") != "COMPLETE" or not isinstance(row.get("samples"), list):
        return False
    for sample in row["samples"]:
        relative = Path(str(sample.get("path", "")))
        path = (Path(output_root) / relative).resolve()
        if (
            Path(output_root).resolve() not in path.parents
            or not path.is_file()
            or path.stat().st_size != int(sample.get("size_bytes", -1))
            or file_sha256(path) != sample.get("sha256")
        ):
            return False
    return True


def extraction_command(task, state_row, manifest, output, episode):
    return [
        "/usr/bin/python3",
        str(ROOT / "ros/dep_car_dataset/scripts/extract_multimodal_bag.py"),
        str(Path(state_row["bag"]).resolve()),
        "--output", str(output),
        "--map-uuid", str(task["map_uuid"]),
        "--map-hash", str(task["map_occupancy_sha256"]),
        "--simulator-seed", str(task["map_seed"]),
        "--stride", "1",
        "--task-id", str(task["task_id"]),
        "--task-manifest-sha256", str(manifest["task_manifest_sha256"]),
        "--maneuver-mode", str(task["maneuver_mode"]),
        "--episode-result", str(episode),
    ]


def reextract_one(task, source_row, manifest, output_root, logs_root):
    task_id = str(task["task_id"])
    bag = Path(source_row["bag"]).resolve()
    episode = ROOT / "data/p3_pilot/run/episodes" / (task_id + ".json")
    if not bag.is_file() or not episode.is_file():
        raise FileNotFoundError("authoritative bag/episode is unavailable for " + task_id)
    temporary_parent = output_root / ".task_staging"
    temporary_parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=task_id + ".", dir=str(temporary_parent)))
    log_path = logs_root / (task_id + ".log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment[name] = "1"
    try:
        command = extraction_command(task, source_row, manifest, staging, episode)
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run(
                command,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
        generated = sorted((staging / task["map_uuid"]).glob(task_id + "-*.npz"))
        if result.returncode != 0 or not generated:
            raise RuntimeError("extractor failed with code %d" % result.returncode)
        staged_rows = []
        context_counts = Counter()
        gear_counts = Counter()
        bag_hashes = set()
        for path in generated:
            failures = audit_multimodal_sample(path)
            if failures:
                raise RuntimeError("sample audit failed: " + ",".join(failures))
            context, requested_gear, source_bag_sha256 = read_context(path)
            context_counts[context] += 1
            gear_counts[str(requested_gear)] += 1
            bag_hashes.add(source_bag_sha256)
            destination = output_root / "samples" / task["map_uuid"] / path.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                raise RuntimeError("refusing to overwrite existing V3 sample " + str(destination))
            staged_rows.append((path, destination, {
                "path": str(destination.relative_to(output_root)),
                "size_bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }))
        if len(bag_hashes) != 1:
            raise RuntimeError("re-extracted samples disagree on source bag SHA-256")
        moved = []
        try:
            for path, destination, _row in staged_rows:
                path.replace(destination)
                moved.append((path, destination))
        except Exception:
            for path, destination in reversed(moved):
                if destination.exists():
                    destination.replace(path)
            raise
        rows = [row for _path, _destination, row in staged_rows]
        return {
            "status": "COMPLETE",
            "completed_at_unix": time.time(),
            "map_uuid": task["map_uuid"],
            "map_split": task["map_split"],
            "maneuver_mode": task["maneuver_mode"],
            "source_bag": str(bag),
            "source_bag_sha256": next(iter(bag_hashes)),
            "source_episode": str(episode.resolve()),
            "source_episode_sha256": file_sha256(episode),
            "candidate_context_contract": "CandidateContextV2",
            "context_counts": dict(sorted(context_counts.items())),
            "requested_gear_counts": dict(sorted(gear_counts.items())),
            "samples": rows,
        }
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", type=Path,
        default=ROOT / "data/p3_pilot/task_manifest.json",
    )
    parser.add_argument(
        "--collection-state", type=Path,
        default=ROOT / "data/p3_pilot/run/collection_state.json",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "data/p3_v3/base_reextracted",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--maximum-tasks", type=int, default=0)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.workers < 1 or args.workers > (os.cpu_count() or 1):
        raise ValueError("workers must fit the visible CPU-thread count")
    if args.maximum_tasks < 0:
        raise ValueError("maximum-tasks cannot be negative")
    manifest_bytes = args.manifest.read_bytes()
    collection_bytes = args.collection_state.read_bytes()
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    source_state = json.loads(collection_bytes.decode("utf-8"))
    verify_manifest(manifest)
    if source_state.get("task_manifest_sha256") != manifest["task_manifest_sha256"]:
        raise ValueError("base collection state is not bound to the task manifest")
    task_by_id = {str(row["task_id"]): row for row in manifest.get("tasks", ())}
    selected = []
    for task_id, task in sorted(task_by_id.items()):
        if task.get("map_split") not in ALLOWED_SPLITS:
            continue
        source_row = source_state.get("tasks", {}).get(task_id, {})
        if source_row.get("status") != "COMPLETE":
            raise RuntimeError("base development task is not COMPLETE: " + task_id)
        selected.append((task, source_row))
    if args.maximum_tasks:
        selected = selected[:args.maximum_tasks]
    identity = {
        "schema": STATE_SCHEMA,
        "source_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "source_task_manifest_sha256": manifest["task_manifest_sha256"],
        "source_collection_state_sha256": hashlib.sha256(collection_bytes).hexdigest(),
        "extractor_sha256": file_sha256(
            ROOT / "ros/dep_car_dataset/scripts/extract_multimodal_bag.py"
        ),
        "splits": list(ALLOWED_SPLITS),
        "test_bags_opened": False,
    }
    state_path = args.output / "reextract_state.json"
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        for key, expected in identity.items():
            if state.get(key) != expected:
                raise RuntimeError("existing reextract state has different " + key)
    else:
        state = {**identity, "started_at_unix": time.time(), "tasks": {}}
    pending = []
    for task, source_row in selected:
        previous = state["tasks"].get(task["task_id"], {})
        if task_result_is_reusable(previous, args.output):
            continue
        if previous.get("status") == "FAILED" and not args.retry_failed:
            continue
        pending.append((task, source_row))
    preview = {
        "status": "DRY_RUN_PASS" if args.dry_run else "RUNNING",
        "selected_development_tasks": len(selected),
        "pending_tasks": len(pending),
        "parallel_workers": min(args.workers, max(1, len(pending))),
        "splits": list(ALLOWED_SPLITS),
        "test_bags_opened": False,
        "first_task_ids": [task["task_id"] for task, _row in pending[:3]],
    }
    if args.dry_run:
        print(json.dumps(preview, indent=2, sort_keys=True))
        return 0
    args.output.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    atomic_json(state_path, state)
    errors = {}
    completed = 0
    with ThreadPoolExecutor(
        max_workers=min(args.workers, max(1, len(pending))),
        thread_name_prefix="p3v3-reextract",
    ) as executor:
        futures = {
            executor.submit(
                reextract_one,
                task,
                source_row,
                manifest,
                args.output,
                args.output / "logs",
            ): task["task_id"]
            for task, source_row in pending
        }
        try:
            for future in as_completed(futures):
                task_id = futures[future]
                try:
                    row = future.result()
                    completed += 1
                except Exception as exc:
                    row = {
                        "status": "FAILED",
                        "failed_at_unix": time.time(),
                        "error": type(exc).__name__ + ": " + str(exc),
                    }
                    errors[task_id] = row["error"]
                with lock:
                    state["tasks"][task_id] = row
                    state["updated_at_unix"] = time.time()
                    atomic_json(state_path, state)
                print(
                    "P3 V3 base reextract task=%s status=%s" % (task_id, row["status"]),
                    flush=True,
                )
        except KeyboardInterrupt:
            for future in futures:
                future.cancel()
            raise
    reusable = sum(
        task_result_is_reusable(state["tasks"].get(task["task_id"], {}), args.output)
        for task, _source_row in selected
    )
    partial = bool(args.maximum_tasks) or reusable != len(selected)
    state["status"] = "PARTIAL" if partial or errors else "PASS"
    state["updated_at_unix"] = time.time()
    state["selected_tasks"] = len(selected)
    state["complete_tasks"] = reusable
    state["errors"] = errors
    atomic_json(state_path, state)
    run_status = (
        "SMOKE_PASS"
        if args.maximum_tasks and not errors and reusable == len(selected)
        else state["status"]
    )
    print(json.dumps({
        "status": run_status,
        "full_dataset_status": state["status"],
        "selected_tasks": len(selected),
        "complete_tasks": reusable,
        "completed_this_run": completed,
        "errors": errors,
        "parallel_workers": args.workers,
        "test_bags_opened": False,
        "state": str(state_path.resolve()),
    }, indent=2, sort_keys=True))
    return 0 if run_status in ("PASS", "SMOKE_PASS") else 2


if __name__ == "__main__":
    raise SystemExit(main())
