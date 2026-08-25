#!/usr/bin/env python3
"""Recover valid V4.3 hard-veto observations without rerunning Gazebo.

The original episode runner incorrectly required the guarded executable bank.
When every proposal was vetoed, rosbag still recorded the raw V4.2 policy,
both DAgger teacher banks, sensors and routes.  This tool validates that
authority, re-extracts a bounded episode window, audits every sample, and only
then changes the task from FAILED to COMPLETE.
"""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import rosbag
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dep_car/src"))
from dep_car.training.dataset import audit_multimodal_sample


REQUIRED_TOPICS = (
    "/camera/depth/image_raw",
    "/velodyne_points",
    "/base_pose_ground_truth",
    "/imu/data",
    "/urban_model/joint_states",
    "/dep_car/global_route",
    "/dep_car/local_route_command",
    "/dep_car/lidar/bev",
    "/dep_car/policy_query",
    "/dep_car/policy_candidates_raw",
    "/dep_car/dagger_teacher_forward",
    "/dep_car/dagger_teacher_reverse",
    "/dep_car/cmd_ackermann",
    "/tf",
    "/tf_static",
)


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def topic_counts(bag_path):
    with rosbag.Bag(str(bag_path), "r") as bag:
        information = bag.get_type_and_topic_info().topics
    return {topic: int(information[topic].message_count) if topic in information else 0
            for topic in REQUIRED_TOPICS + ("/dep_car/candidates",)}


def recover_task(task, config, work_root, temporary_root):
    task_id = task["task_id"]
    bag = work_root / "bags" / task["map_split"] / (task_id + ".bag")
    if not bag.is_file():
        raise RuntimeError(task_id + ": finalized bag is missing")
    counts = topic_counts(bag)
    missing = [topic for topic in REQUIRED_TOPICS if counts[topic] <= 0]
    if missing:
        raise RuntimeError(task_id + ": missing recorded authority: " + ",".join(missing))

    recovered_episode = work_root / "episodes" / (task_id + ".recovered.json")
    episode_payload = {
        "schema": "DEPCarV43RecoveredEpisodeEvidenceV1",
        "task_id": task_id,
        "maneuver_mode": task["maneuver_mode"],
        "status": "RECOVERED_DAGGER_HARD_VETO_OBSERVATION",
        "recovery_reason": "legacy runner required guarded executable candidates",
        "guarded_candidate_messages": counts["/dep_car/candidates"],
        "policy_raw_messages": counts["/dep_car/policy_candidates_raw"],
        "teacher_forward_messages": counts["/dep_car/dagger_teacher_forward"],
        "teacher_reverse_messages": counts["/dep_car/dagger_teacher_reverse"],
        "illegal_shift_count": None,
        "illegal_shift_evidence": (
            "unavailable because the legacy runner exited during readiness; "
            "executed commands are diagnostic inputs, never teacher authority"
        ),
        "recorded_topic_counts": counts,
        "bag": str(bag),
        "bag_sha256_authority": "embedded_in_each_extracted_sample",
        "gazebo_rerun": False,
    }
    atomic_json(recovered_episode, episode_payload)

    extraction = config["collection"]
    stride = int(extraction["extraction_stride"])
    # Use recorded timestamps, not a nominal sensor frequency.  The simulated
    # VLP-16 update rate and extraction stride may change independently.
    maximum_duration_s = float(extraction["episode_timeout_s"])
    task_output = temporary_root / (task_id + "_" + str(time.time_ns()))
    command = [
        sys.executable,
        str(ROOT / "ros/dep_car_dataset/scripts/extract_multimodal_bag.py"),
        str(bag), "--output", str(task_output),
        "--map-uuid", task["map_uuid"],
        "--map-hash", task["map_occupancy_sha256"],
        "--simulator-seed", str(task["map_seed"]),
        "--stride", str(stride),
        "--maximum-duration-s", str(maximum_duration_s),
        "--task-id", task_id,
        "--task-manifest-sha256", os.environ["DEP_CAR_P3_TASK_MANIFEST_SHA256"],
        "--maneuver-mode", task["maneuver_mode"],
        "--episode-result", str(recovered_episode),
        "--dagger-v43",
    ]
    environment = dict(os.environ)
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment[variable] = "1"
    result = subprocess.run(
        command, env=environment, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(task_id + ": extraction failed:\n" + result.stdout[-2000:])
    samples = sorted((task_output / task["map_uuid"]).glob(task_id + "-*.npz"))
    minimum = int(extraction["minimum_samples_per_episode"])
    if len(samples) < minimum:
        raise RuntimeError("%s: only %d recovered samples" % (task_id, len(samples)))
    failures = {str(path): audit_multimodal_sample(path) for path in samples}
    failures = {path: errors for path, errors in failures.items() if errors}
    if failures:
        raise RuntimeError(task_id + ": sample audit failed: " + json.dumps(failures)[:1000])
    return {
        "task": task,
        "bag": bag,
        "episode": recovered_episode,
        "episode_payload": episode_payload,
        "samples": samples,
        "extractor_output": result.stdout,
        "maximum_duration_s": maximum_duration_s,
    }


def install_recovery(result, work_root, state):
    task = result["task"]
    task_id = task["task_id"]
    destination = work_root / "samples" / task["map_uuid"]
    destination.mkdir(parents=True, exist_ok=True)
    old = sorted(destination.glob(task_id + "-*.npz"))
    if old:
        archive = work_root / "recovery_archive" / task_id / str(time.time_ns())
        archive.mkdir(parents=True, exist_ok=True)
        for path in old:
            path.replace(archive / path.name)
    for path in result["samples"]:
        path.replace(destination / path.name)
    episode = result["episode_payload"]
    row = state["tasks"][task_id]
    row.pop("error", None)
    row.pop("failed_at_unix", None)
    row.update({
        "status": "COMPLETE",
        "completed_at_unix": time.time(),
        "episode_status": episode["status"],
        "samples": len(result["samples"]),
        "bag": str(result["bag"]),
        "bag_size_bytes": result["bag"].stat().st_size,
        "candidate_messages": episode["guarded_candidate_messages"],
        "guarded_candidate_messages": episode["guarded_candidate_messages"],
        "zero_feasible_messages": 0,
        "policy_raw_messages": episode["policy_raw_messages"],
        "teacher_forward_messages": episode["teacher_forward_messages"],
        "teacher_reverse_messages": episode["teacher_reverse_messages"],
        "post_settle_pose_reset": True,
        "recovered_extraction_without_gazebo": True,
        "recovery_evidence": str(result["episode"]),
        "recovery_extractor_sha256": sha256_file(
            ROOT / "ros/dep_car_dataset/scripts/extract_multimodal_bag.py"
        ),
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--rerun-recovered", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    state_path = args.work_root / "collection_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    tasks = {task["task_id"]: task for task in manifest["tasks"]}
    failed = [
        tasks[task_id] for task_id, row in state["tasks"].items()
        if task_id in tasks and (
            row.get("status") == "FAILED"
            or (
                args.rerun_recovered
                and row.get("recovered_extraction_without_gazebo") is True
            )
        )
    ]
    print(json.dumps({
        "schema": "DEPCarV43ExtractionRecoveryPlanV1",
        "status": "DRY_RUN_PASS" if args.dry_run else "RUNNING",
        "failed_tasks": [task["task_id"] for task in failed],
        "gazebo_rerun": False,
        "requested_workers": args.workers,
        "rerun_recovered": args.rerun_recovered,
        "effective_workers": min(args.workers, 2, max(1, len(failed))),
    }, indent=2, sort_keys=True), flush=True)
    if args.dry_run:
        return
    if not failed:
        print(json.dumps({"status": "PASS", "recovered": 0}, indent=2))
        return
    os.environ["DEP_CAR_P3_TASK_MANIFEST_SHA256"] = manifest["task_manifest_sha256"]
    lock_path = args.work_root / ".collection.lock"
    with lock_path.open("w", encoding="utf-8") as lock_stream:
        fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        temporary_root = args.work_root / "recovery_tmp"
        temporary_root.mkdir(parents=True, exist_ok=True)
        results, errors = [], []
        effective = min(args.workers, 2, len(failed))
        with ThreadPoolExecutor(max_workers=effective) as executor:
            futures = {
                executor.submit(recover_task, task, config, args.work_root, temporary_root): task
                for task in failed
            }
            for future in as_completed(futures):
                task_id = futures[future]["task_id"]
                try:
                    result = future.result()
                    results.append(result)
                    print("V4.3 recovery task=%s samples=%d status=PASS" % (
                        task_id, len(result["samples"])
                    ), flush=True)
                except Exception as exc:
                    errors.append(task_id + ": " + type(exc).__name__ + ": " + str(exc))
                    print("V4.3 recovery task=%s status=FAIL" % task_id, flush=True)
        if not errors:
            for result in results:
                install_recovery(result, args.work_root, state)
            state["updated_at_unix"] = time.time()
            atomic_json(state_path, state)
        totals = Counter(row.get("status") for row in state["tasks"].values())
        payload = {
            "schema": "DEPCarV43ExtractionRecoveryV1",
            "status": "PASS" if not errors else "FAIL",
            "gazebo_rerun": False,
            "recovered": len(results) if not errors else 0,
            "errors": errors,
            "state_totals": dict(sorted(totals.items())),
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        raise SystemExit(0 if not errors else 2)


if __name__ == "__main__":
    main()
