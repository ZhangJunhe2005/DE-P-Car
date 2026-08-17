#!/usr/bin/env python3
"""Recover P3 samples from authoritative bags without rerunning Gazebo."""

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "dep_car/src"))
from dep_car.training.dataset import audit_multimodal_sample


RECOVERY_REJECTION = "candidate bank gear must equal the supervisor-requested gear"


def atomic_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def current_extraction_log(work_root, task_id, state_row):
    attempt = int(state_row.get("attempt", 0))
    return work_root / "logs" / task_id / ("attempt_%02d" % attempt) / "extraction.log"


def extraction_command(task, state_row, config, manifest, work_root):
    return [
        "/usr/bin/python3", str(ROOT / "ros/dep_car_dataset/scripts/extract_multimodal_bag.py"),
        state_row["bag"], "--output", str(work_root / "samples"),
        "--map-uuid", task["map_uuid"], "--map-hash", task["map_occupancy_sha256"],
        "--simulator-seed", str(task["map_seed"]),
        "--stride", str(config["collection"]["extraction_stride"]),
        "--task-id", task["task_id"],
        "--task-manifest-sha256", manifest["task_manifest_sha256"],
        "--maneuver-mode", task["maneuver_mode"],
        "--episode-result", str(work_root / "episodes" / (task["task_id"] + ".json")),
    ]


def reextract_one(task, state_row, config, manifest, work_root, run_stamp):
    task_id = task["task_id"]
    sample_root = work_root / "samples" / task["map_uuid"]
    previous = sorted(sample_root.glob(task_id + "-*.npz"))
    archive = work_root / "previous_attempts" / task_id / ("reextract_" + run_stamp)
    archive.mkdir(parents=True, exist_ok=False)
    for path in previous:
        path.replace(archive / path.name)
    log_root = work_root / "logs" / task_id / ("reextract_" + run_stamp)
    log_root.mkdir(parents=True, exist_ok=False)
    environment = dict(os.environ)
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment[variable] = "1"
    try:
        with (log_root / "extraction.log").open("w", encoding="utf-8") as log:
            result = subprocess.run(
                extraction_command(task, state_row, config, manifest, work_root),
                env=environment, stdout=log, stderr=subprocess.STDOUT, text=True,
            )
        samples = sorted(sample_root.glob(task_id + "-*.npz"))
        if result.returncode != 0 or not samples:
            raise RuntimeError("extractor failed with code %d" % result.returncode)
        failures = {str(path): audit_multimodal_sample(path) for path in samples}
        failures = {path: errors for path, errors in failures.items() if errors}
        if failures:
            raise RuntimeError("sample audit failed: " + json.dumps(failures, sort_keys=True)[:500])
        return task_id, len(samples), str(archive)
    except Exception:
        failed = archive / "failed_reextract"
        failed.mkdir(exist_ok=True)
        for path in sample_root.glob(task_id + "-*.npz"):
            path.replace(failed / path.name)
        for path in archive.glob(task_id + "-*.npz"):
            path.replace(sample_root / path.name)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "dep_car/config/p3_pilot.yaml")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/p3_pilot/task_manifest.json")
    parser.add_argument("--work-root", type=Path, default=ROOT / "data/p3_pilot/run")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--all", action="store_true", help="reextract all tasks, not only recovery-context rejects")
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    state_path = args.work_root / "collection_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    tasks_by_id = {task["task_id"]: task for task in manifest["tasks"]}
    selected = []
    for task_id, row in state["tasks"].items():
        if row.get("status") != "COMPLETE":
            continue
        log_path = current_extraction_log(args.work_root, task_id, row)
        recovery_rejected = log_path.is_file() and RECOVERY_REJECTION in log_path.read_text(
            encoding="utf-8", errors="replace"
        )
        if args.all or recovery_rejected:
            selected.append((tasks_by_id[task_id], dict(row)))
    run_stamp = time.strftime("%Y%m%dT%H%M%S", time.localtime())
    completed = {}
    errors = {}
    with ThreadPoolExecutor(max_workers=min(args.workers, max(1, len(selected)))) as executor:
        futures = {
            executor.submit(reextract_one, task, row, config, manifest, args.work_root, run_stamp): task["task_id"]
            for task, row in selected
        }
        for future in as_completed(futures):
            task_id = futures[future]
            try:
                _, count, archive = future.result()
                completed[task_id] = {"samples": count, "archive": archive}
                print("P3 reextract task=%s samples=%d" % (task_id, count), flush=True)
            except Exception as exc:
                errors[task_id] = type(exc).__name__ + ": " + str(exc)
    for task_id, result in completed.items():
        state["tasks"][task_id]["samples"] = result["samples"]
        state["tasks"][task_id]["reextracted_at_unix"] = time.time()
        state["tasks"][task_id]["candidate_context_contract"] = "CandidateContextV2"
    state["updated_at_unix"] = time.time()
    atomic_json(state_path, state)
    payload = {
        "status": "PASS" if not errors else "FAIL", "selected": len(selected),
        "completed": len(completed), "errors": errors, "parallel_workers": args.workers,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    raise SystemExit(0 if not errors else 1)


if __name__ == "__main__":
    main()
