#!/usr/bin/env python3
"""Parallel, resumable P3 Gazebo/rosbag collection orchestrator."""

import argparse
import fcntl
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "dep_car/src"))
from dep_car.training.dataset import audit_multimodal_sample
from dep_car.training.pilot import canonical_sha256


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def resolve_project_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as connection:
        connection.settimeout(0.2)
        return connection.connect_ex(("127.0.0.1", int(port))) == 0


def stop_process(process, timeout, log):
    if process is None or process.poll() is not None:
        return
    for requested_signal in (signal.SIGINT, signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(process.pid, requested_signal)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=timeout if requested_signal != signal.SIGKILL else 5.0)
            return
        except subprocess.TimeoutExpired:
            log.write("process %d did not stop after %s\n" % (process.pid, requested_signal.name))
            log.flush()


def wait_for_finalized_bag(bag_path, timeout, log):
    """Wait until rosbag has renamed its compressed .active file.

    With BZ2 the rosrun wrapper can finish before the recorder's final chunk
    compression and rename are visible to the extraction process.  LZ4 was
    fast enough to hide this race; BZ2 makes it reproducible under parallel
    collection.
    """

    bag_path = Path(bag_path)
    active_path = Path(str(bag_path) + ".active")
    deadline = time.monotonic() + float(timeout)
    previous_size = None
    stable_observations = 0
    while time.monotonic() < deadline:
        if bag_path.is_file() and not active_path.exists():
            size = bag_path.stat().st_size
            if size > 0 and size == previous_size:
                stable_observations += 1
                if stable_observations >= 2:
                    return True
            else:
                stable_observations = 0
            previous_size = size
        else:
            previous_size = None
            stable_observations = 0
        time.sleep(0.5)
    log.write(
        "rosbag did not finalize within %.1fs: bag=%s active=%s\n"
        % (float(timeout), bag_path.is_file(), active_path.exists())
    )
    log.flush()
    return False


class ProcessRegistry:
    """Track child process groups so Ctrl+C can stop all parallel workers."""

    def __init__(self):
        self.lock = threading.Lock()
        self.processes = set()

    def add(self, process):
        with self.lock:
            self.processes.add(process)

    def discard(self, process):
        with self.lock:
            self.processes.discard(process)

    def interrupt_all(self):
        with self.lock:
            processes = tuple(self.processes)
        for process in processes:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGINT)
                except ProcessLookupError:
                    pass


def run_logged(command, env, log_path, registry, timeout=None):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command, env=env, stdout=log, stderr=subprocess.STDOUT,
            text=True, start_new_session=True,
        )
        registry.add(process)
        try:
            return process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            stop_process(process, 5.0, log)
            raise
        finally:
            registry.discard(process)


def wait_for_model(env, timeout, log_path, stop_event):
    deadline = time.monotonic() + timeout
    command = ["rosservice", "call", "/gazebo/get_model_state", "model_name: 'urban_model'"]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        while time.monotonic() < deadline and not stop_event.is_set():
            try:
                result = subprocess.run(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=4.0)
                log.write(result.stdout)
                log.flush()
                if result.returncode == 0 and "success: True" in result.stdout:
                    return True
            except subprocess.TimeoutExpired:
                pass
            time.sleep(1.0)
    return False


def verify_manifest(manifest):
    expected = manifest.get("task_manifest_sha256", "")
    payload = dict(manifest)
    payload.pop("task_manifest_sha256", None)
    actual = canonical_sha256(payload)
    if expected != actual:
        raise ValueError("task manifest SHA256 does not match its content")


def initial_state(manifest, config):
    return {
        "schema": "DEPCarP3CollectionStateV1",
        "task_manifest_sha256": manifest["task_manifest_sha256"],
        "config_sha256": canonical_sha256(config),
        "started_at_unix": time.time(),
        "updated_at_unix": time.time(),
        "tasks": {},
    }


def task_commands(task, collection, work_root, env, ros_master_port):
    world = resolve_project_path(task["world"])
    map_yaml = resolve_project_path(task["map_yaml"])
    maps_root = world.parent.parent
    launch = [
        "roslaunch", "-p", str(ros_master_port), "dep_car_bringup", "urban_sim.launch",
        "world:=" + str(world), "map_yaml:=" + str(map_yaml), "gazebo_model_path:=" + str(maps_root),
        "gui:=false", "enable_rviz:=false", "enable_stack:=true",
        "x:=" + str(task["start"][0]), "y:=" + str(task["start"][1]), "yaw:=" + str(task["start"][2]),
    ]
    bag = work_root / "bags" / task["map_split"] / (task["task_id"] + ".bag")
    episode = work_root / "episodes" / (task["task_id"] + ".json")
    reset_pose = [
        "rosrun", "dep_car_dataset", "reset_pilot_pose.py",
        "--x", str(task["start"][0]), "--y", str(task["start"][1]), "--yaw", str(task["start"][2]),
    ]
    # BZ2 reduced the representative V4 corner bag by about 26% while keeping
    # the raw ROS messages and their authority contract intact.  Keep this a
    # command-level default rather than a required config key so an existing
    # resumable collection state retains the same config hash.
    bag_compression = str(collection.get("bag_compression", "bz2"))
    if bag_compression not in ("bz2", "lz4", "none"):
        raise ValueError("unsupported rosbag compression: " + bag_compression)
    recorder = [
        "rosrun", "dep_car_dataset", "record_multimodal_episode.sh",
        str(bag), bag_compression,
    ]
    runner = [
        "rosrun", "dep_car_dataset", "run_pilot_episode.py",
        "--task-id", task["task_id"], "--maneuver-mode", task["maneuver_mode"],
        "--goal-x", str(task["goal"][0]), "--goal-y", str(task["goal"][1]), "--goal-yaw", str(task["goal"][2]),
        "--startup-timeout", str(collection["startup_timeout_s"]),
        "--episode-timeout", str(collection["episode_timeout_s"]), "--output", str(episode),
    ]
    extraction = [
        "rosrun", "dep_car_dataset", "extract_multimodal_bag.py", str(bag),
        "--output", str(work_root / "samples"), "--map-uuid", task["map_uuid"],
        "--map-hash", task["map_occupancy_sha256"], "--simulator-seed", str(task["map_seed"]),
        "--stride", str(collection["extraction_stride"]), "--task-id", task["task_id"],
        "--task-manifest-sha256", env["DEP_CAR_P3_TASK_MANIFEST_SHA256"],
        "--maneuver-mode", task["maneuver_mode"], "--episode-result", str(episode),
    ]
    return launch, reset_pose, recorder, runner, extraction, bag, episode


def archive_previous_artifacts(task, work_root, bag_path, episode_path, attempt):
    archive = work_root / "previous_attempts" / task["task_id"] / ("attempt_%02d" % max(1, attempt - 1))
    candidates = [bag_path, Path(str(bag_path) + ".active"), episode_path]
    sample_root = work_root / "samples" / task["map_uuid"]
    candidates.extend(sample_root.glob(task["task_id"] + "-*.npz"))
    existing = [path for path in candidates if path.exists()]
    if not existing:
        return
    archive.mkdir(parents=True, exist_ok=True)
    for path in existing:
        destination = archive / path.name
        if destination.exists():
            destination = archive / (str(time.time_ns()) + "_" + path.name)
        path.replace(destination)


def worker_environment(manifest, collection, worker_index):
    environment = dict(os.environ)
    ros_port = int(collection["ros_master_port"]) + worker_index
    gazebo_port = int(collection["gazebo_master_port"]) + worker_index
    environment["ROS_MASTER_URI"] = "http://127.0.0.1:%d" % ros_port
    environment["GAZEBO_MASTER_URI"] = "http://127.0.0.1:%d" % gazebo_port
    environment["DEP_CAR_P3_TASK_MANIFEST_SHA256"] = manifest["task_manifest_sha256"]
    # Eight independent pipelines provide the parallelism.  Keep numerical
    # libraries inside each extractor single-threaded to prevent 8x oversubscription.
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment[variable] = "1"
    environment["PYTHONUNBUFFERED"] = "1"
    return environment, ros_port, gazebo_port


def shard_tasks_by_map(tasks, workers):
    """Balance work while ensuring one map is never simulated concurrently."""

    buckets = [[] for _ in range(workers)]
    map_worker = {}
    for task in tasks:
        map_uuid = task["map_uuid"]
        if map_uuid not in map_worker:
            map_worker[map_uuid] = min(range(workers), key=lambda index: (len(buckets[index]), index))
        buckets[map_worker[map_uuid]].append(task)
    return buckets


def save_state(state_path, state, state_lock):
    with state_lock:
        state["updated_at_unix"] = time.time()
        atomic_json(state_path, state)


def task_is_pending(
    task, state, retry_failed=False, rerun_complete=False,
    rerun_all_complete=False, zero_rate_above=None,
):
    """Apply the same resume/retry policy to dry-runs and real collection."""

    previous = state.get("tasks", {}).get(task["task_id"], {})
    status = previous.get("status")
    if str(status).startswith("EXCLUDED_"):
        return False
    if status == "COMPLETE":
        candidate_messages = int(previous.get("candidate_messages", 0))
        zero_feasible_messages = int(previous.get("zero_feasible_messages", 0))
        zero_rate = zero_feasible_messages / max(1, candidate_messages)
        return rerun_complete or rerun_all_complete or (
            zero_rate_above is not None and zero_rate > zero_rate_above
        )
    if status == "FAILED":
        return retry_failed
    return True


def collect_one_task(
    task, worker_index, collection, work_root, environment, ros_port, gazebo_port,
    state, state_path, state_lock, stop_event, registry,
):
    if stop_event.is_set():
        return "INTERRUPTED"
    free_gb = shutil.disk_usage(work_root).free / (1024 ** 3)
    if free_gb < float(collection["minimum_free_disk_gb"]):
        stop_event.set()
        raise RuntimeError("free disk %.1f GiB is below P3 safety floor" % free_gb)
    if port_in_use(ros_port) or port_in_use(gazebo_port):
        raise RuntimeError("worker %d ROS/Gazebo port is already in use" % worker_index)

    with state_lock:
        previous = dict(state["tasks"].get(task["task_id"], {}))
    attempt = int(previous.get("attempt", 0)) + 1
    task_log_root = work_root / "logs" / task["task_id"] / ("attempt_%02d" % attempt)
    ros_log_root = task_log_root / "ros"
    ros_log_root.mkdir(parents=True, exist_ok=True)
    task_environment = dict(environment)
    task_environment["ROS_LOG_DIR"] = str(ros_log_root)
    launch_cmd, reset_cmd, recorder_cmd, runner_cmd, extraction_cmd, bag_path, episode_path = task_commands(
        task, collection, work_root, task_environment, ros_port,
    )
    archive_previous_artifacts(task, work_root, bag_path, episode_path, attempt)
    bag_path.parent.mkdir(parents=True, exist_ok=True)
    launch_log = (task_log_root / "roslaunch.log").open("w", encoding="utf-8")
    recorder_log = (task_log_root / "rosbag.log").open("w", encoding="utf-8")
    launch = recorder = None
    with state_lock:
        state["tasks"][task["task_id"]] = {
            "status": "RUNNING", "attempt": attempt, "worker_index": worker_index,
            "ros_master_port": ros_port, "gazebo_master_port": gazebo_port,
            "started_at_unix": time.time(), "maneuver_mode": task["maneuver_mode"],
            "map_uuid": task["map_uuid"],
        }
        state["updated_at_unix"] = time.time()
        atomic_json(state_path, state)
    status = "FAILED"
    try:
        launch = subprocess.Popen(
            launch_cmd, env=task_environment, stdout=launch_log, stderr=subprocess.STDOUT,
            text=True, start_new_session=True,
        )
        registry.add(launch)
        if not wait_for_model(
            task_environment, float(collection["startup_timeout_s"]),
            task_log_root / "readiness.log", stop_event,
        ):
            raise RuntimeError("Gazebo Urban Car did not become ready")
        reset_timeout = float(collection["startup_timeout_s"]) + 15.0
        reset_code = run_logged(
            reset_cmd, task_environment, task_log_root / "pose_reset.log", registry,
            timeout=reset_timeout,
        )
        if reset_code != 0:
            raise RuntimeError("post-settle pose reset failed with code %d" % reset_code)
        recorder = subprocess.Popen(
            recorder_cmd, env=task_environment, stdout=recorder_log, stderr=subprocess.STDOUT,
            text=True, start_new_session=True,
        )
        registry.add(recorder)
        if stop_event.wait(1.0):
            raise RuntimeError("collection interrupted")
        # The episode runner has two bounded readiness phases: sensor/TF and
        # post-goal planner outputs.  Keep this outer watchdog strictly larger
        # than both phases plus the episode window.
        runner_timeout = (
            2.0 * float(collection["startup_timeout_s"])
            + float(collection["episode_timeout_s"])
            + 30.0
        )
        runner_code = run_logged(
            runner_cmd, task_environment, task_log_root / "episode.log", registry,
            timeout=runner_timeout,
        )
        bag_finalize_timeout = float(collection.get("bag_finalize_timeout_s", 180.0))
        stop_process(recorder, bag_finalize_timeout, recorder_log)
        registry.discard(recorder)
        recorder = None
        if not wait_for_finalized_bag(bag_path, bag_finalize_timeout, recorder_log):
            raise RuntimeError("rosbag did not finish BZ2 finalization")
        stop_process(launch, float(collection["shutdown_timeout_s"]), launch_log)
        registry.discard(launch)
        launch = None
        if runner_code != 0 or not episode_path.is_file():
            raise RuntimeError("episode runner failed with code %d" % runner_code)
        episode = json.loads(episode_path.read_text(encoding="utf-8"))
        if episode.get("status") in ("INFRASTRUCTURE_ERROR", "NO_CANDIDATES"):
            raise RuntimeError("episode status is " + str(episode.get("status")))
        if int(episode.get("illegal_shift_count", 0)) != 0:
            raise RuntimeError("episode contains an illegal direction shift")
        extraction_code = run_logged(
            extraction_cmd, task_environment, task_log_root / "extraction.log", registry,
            timeout=600.0,
        )
        if extraction_code != 0:
            raise RuntimeError("offline extraction failed with code %d" % extraction_code)
        samples = sorted((work_root / "samples" / task["map_uuid"]).glob(task["task_id"] + "-*.npz"))
        if len(samples) < int(collection["minimum_samples_per_episode"]):
            raise RuntimeError("only %d samples were extracted" % len(samples))
        failures = {str(path): audit_multimodal_sample(path) for path in samples}
        failures = {path: errors for path, errors in failures.items() if errors}
        if failures:
            raise RuntimeError("sample audit failed: " + json.dumps(failures, sort_keys=True)[:500])
        with state_lock:
            state["tasks"][task["task_id"]].update({
                "status": "COMPLETE", "completed_at_unix": time.time(), "episode_status": episode["status"],
                "samples": len(samples), "bag": str(bag_path), "bag_size_bytes": bag_path.stat().st_size,
                "illegal_shift_count": int(episode.get("illegal_shift_count", 0)),
                "candidate_messages": int(episode.get("candidate_messages", 0)),
                "zero_feasible_messages": int(episode.get("zero_feasible_messages", 0)),
                "post_settle_pose_reset": True,
            })
        status = "COMPLETE"
    except Exception as exc:
        status = "INTERRUPTED" if stop_event.is_set() else "FAILED"
        with state_lock:
            state["tasks"][task["task_id"]].update({
                "status": status, "failed_at_unix": time.time(),
                "error": type(exc).__name__ + ": " + str(exc),
            })
    finally:
        stop_process(
            recorder,
            float(collection.get("bag_finalize_timeout_s", 180.0)),
            recorder_log,
        )
        stop_process(launch, float(collection["shutdown_timeout_s"]), launch_log)
        registry.discard(recorder)
        registry.discard(launch)
        recorder_log.close()
        launch_log.close()
        save_state(state_path, state, state_lock)
    print("P3 worker=%d task=%s status=%s" % (worker_index, task["task_id"], status), flush=True)
    return status


def worker_loop(
    worker_index, tasks, manifest, collection, work_root, state, state_path,
    state_lock, stop_event, registry, fail_fast, startup_stagger_s,
):
    environment, ros_port, gazebo_port = worker_environment(manifest, collection, worker_index)
    if stop_event.wait(worker_index * startup_stagger_s):
        return Counter()
    counts = Counter()
    for task in tasks:
        if stop_event.is_set():
            break
        status = collect_one_task(
            task, worker_index, collection, work_root, environment, ros_port, gazebo_port,
            state, state_path, state_lock, stop_event, registry,
        )
        counts[status] += 1
        if fail_fast and status != "COMPLETE":
            stop_event.set()
            break
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "dep_car/config/p3_pilot.yaml")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/p3_pilot/task_manifest.json")
    parser.add_argument("--work-root", type=Path, default=ROOT / "data/p3_pilot/run")
    parser.add_argument("--maximum-tasks", type=int, default=0)
    parser.add_argument("--task-id")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument(
        "--rerun-complete", action="store_true",
        help="rerun a COMPLETE task selected by --task-id (the old artifacts are archived)",
    )
    parser.add_argument(
        "--rerun-all-complete", action="store_true",
        help="explicitly rerun every COMPLETE task to migrate a dataset-wide runtime contract",
    )
    parser.add_argument(
        "--retry-zero-feasible-rate-above", type=float, default=None,
        help="rerun COMPLETE tasks whose recorded zero-feasible fraction exceeds this value",
    )
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--workers", type=int, default=8, help="parallel Gazebo/rosbag pipelines")
    parser.add_argument("--startup-stagger", type=float, default=1.0, help="seconds between worker launches")
    args = parser.parse_args()
    if args.maximum_tasks < 0:
        raise ValueError("maximum tasks cannot be negative")
    if args.workers < 1:
        raise ValueError("workers must be positive")
    if args.startup_stagger < 0.0:
        raise ValueError("startup stagger cannot be negative")
    if args.rerun_complete and not args.task_id:
        raise ValueError("--rerun-complete requires --task-id")
    if args.rerun_complete and args.rerun_all_complete:
        raise ValueError("choose either --rerun-complete or --rerun-all-complete")
    if (
        args.retry_zero_feasible_rate_above is not None
        and not 0.0 <= args.retry_zero_feasible_rate_above <= 1.0
    ):
        raise ValueError("--retry-zero-feasible-rate-above must be in [0, 1]")
    available_cpus = os.cpu_count() or 1
    if args.workers > available_cpus:
        raise ValueError("workers=%d exceeds the %d visible CPU threads" % (args.workers, available_cpus))
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    collection = config["collection"]
    if not collection.get("keep_bags", True):
        raise ValueError("P2 raw-authority contract requires keep_bags=true")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    verify_manifest(manifest)
    task_ids = [task["task_id"] for task in manifest["tasks"]]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("task manifest contains duplicate task IDs")
    tasks = [task for task in manifest["tasks"] if not args.task_id or task["task_id"] == args.task_id]
    if args.task_id and not tasks:
        raise ValueError("task id is not present in manifest: " + args.task_id)
    for executable in ("roslaunch", "rosrun", "rosservice"):
        if shutil.which(executable) is None:
            raise RuntimeError(executable + " is not available; source ROS and catkin setup first")
    for task in tasks:
        for key in ("world", "map_yaml"):
            if not resolve_project_path(task[key]).is_file():
                raise FileNotFoundError(resolve_project_path(task[key]))
    if args.dry_run:
        preview_state_path = args.work_root / "collection_state.json"
        preview_state = (
            json.loads(preview_state_path.read_text(encoding="utf-8"))
            if preview_state_path.is_file() else {"tasks": {}}
        )
        selected = [
            task for task in tasks
            if task_is_pending(
                task, preview_state, args.retry_failed, args.rerun_complete,
                args.rerun_all_complete, args.retry_zero_feasible_rate_above,
            )
        ]
        selected = selected[:args.maximum_tasks] if args.maximum_tasks else selected
        active_workers = min(args.workers, max(1, len(selected)))
        preview = []
        for index, task in enumerate(selected[:3]):
            worker_index = index % active_workers
            environment, ros_port, gazebo_port = worker_environment(manifest, collection, worker_index)
            preview.append({
                "worker_index": worker_index, "ros_master_port": ros_port,
                "gazebo_master_port": gazebo_port,
                "launch": task_commands(task, collection, args.work_root, environment, ros_port)[0],
            })
        print(json.dumps({
            "status": "DRY_RUN_PASS", "tasks_selected": len(selected),
            "parallel_workers": active_workers,
            "task_manifest_sha256": manifest["task_manifest_sha256"], "launch_preview": preview,
        }, indent=2, sort_keys=True))
        return

    args.work_root.mkdir(parents=True, exist_ok=True)
    lock_path = args.work_root / ".collection.lock"
    lock_stream = lock_path.open("w", encoding="utf-8")
    try:
        fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError("another P3 collection process owns " + str(lock_path))
    state_path = args.work_root / "collection_state.json"
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("task_manifest_sha256") != manifest["task_manifest_sha256"]:
            raise RuntimeError("existing run state belongs to a different task manifest")
        if state.get("config_sha256") != canonical_sha256(config):
            raise RuntimeError("existing run state belongs to a different P3 configuration")
    else:
        state = initial_state(manifest, config)

    pending = []
    for task in tasks:
        if task_is_pending(
            task, state, args.retry_failed, args.rerun_complete,
            args.rerun_all_complete, args.retry_zero_feasible_rate_above,
        ):
            pending.append(task)
    if args.maximum_tasks:
        pending = pending[:args.maximum_tasks]
    state_lock = threading.Lock()
    with state_lock:
        state["parallel_workers"] = args.workers
        state["visible_cpu_threads"] = available_cpus
        state["updated_at_unix"] = time.time()
        atomic_json(state_path, state)
    if not pending:
        totals = {
            name: sum(item.get("status") == name for item in state["tasks"].values())
            for name in (
                "COMPLETE", "FAILED", "INTERRUPTED", "RUNNING",
                "EXCLUDED_INVALID_GOAL",
            )
        }
        status = (
            "PASS_WITH_EXCLUSIONS"
            if totals["EXCLUDED_INVALID_GOAL"] and not any(
                totals[name] for name in ("FAILED", "INTERRUPTED", "RUNNING")
            )
            else "PASS"
            if not any(totals[name] for name in ("FAILED", "INTERRUPTED", "RUNNING"))
            else "PARTIAL"
        )
        print(json.dumps({
            "status": status, "parallel_workers": 0, "this_run": {},
            "state_totals": totals, "worker_errors": [],
        }, indent=2, sort_keys=True))
        raise SystemExit(0 if status == "PASS" else 2)
    active_workers = min(args.workers, max(1, len(pending)))
    ports = []
    for worker_index in range(active_workers):
        _, ros_port, gazebo_port = worker_environment(manifest, collection, worker_index)
        ports.extend((ros_port, gazebo_port))
    if len(ports) != len(set(ports)) or any(port <= 0 or port > 65535 for port in ports):
        raise ValueError("parallel worker ROS/Gazebo port ranges overlap or are invalid")
    occupied = [port for port in ports if port_in_use(port)]
    if occupied:
        raise RuntimeError("parallel worker ports already in use: " + ",".join(map(str, occupied)))
    stop_event = threading.Event()
    registry = ProcessRegistry()
    run_counts = Counter()
    worker_errors = []
    buckets = shard_tasks_by_map(pending, active_workers)
    executor = ThreadPoolExecutor(max_workers=active_workers, thread_name_prefix="p3-worker")
    futures = [
        executor.submit(
            worker_loop, worker_index, bucket, manifest, collection, args.work_root,
            state, state_path, state_lock, stop_event, registry, args.fail_fast, args.startup_stagger,
        )
        for worker_index, bucket in enumerate(buckets) if bucket
    ]
    try:
        for future in as_completed(futures):
            try:
                run_counts.update(future.result())
            except Exception as exc:
                worker_errors.append(type(exc).__name__ + ": " + str(exc))
                stop_event.set()
    except KeyboardInterrupt:
        stop_event.set()
        registry.interrupt_all()
        for future in futures:
            future.cancel()
        executor.shutdown(wait=True)
        raise
    finally:
        stop_event.set()
        executor.shutdown(wait=True)
    totals = {
        name: sum(item.get("status") == name for item in state["tasks"].values())
        for name in (
            "COMPLETE", "FAILED", "INTERRUPTED", "RUNNING",
            "EXCLUDED_INVALID_GOAL",
        )
    }
    status = (
        "PASS_WITH_EXCLUSIONS"
        if not worker_errors
        and totals["EXCLUDED_INVALID_GOAL"]
        and not any(totals[name] for name in ("FAILED", "INTERRUPTED", "RUNNING"))
        else "PASS"
        if not worker_errors
        and not any(totals[name] for name in ("FAILED", "INTERRUPTED", "RUNNING"))
        else "PARTIAL"
    )
    print(json.dumps({
        "status": status, "parallel_workers": active_workers,
        "this_run": dict(run_counts), "state_totals": totals, "worker_errors": worker_errors,
    }, indent=2, sort_keys=True))
    raise SystemExit(0 if status == "PASS" else 2)


if __name__ == "__main__":
    main()
