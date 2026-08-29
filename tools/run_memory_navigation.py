#!/usr/bin/env python3
"""Host entry for online-SLAM visibility/memory Gazebo validation."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import socket
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

from PIL import Image
import yaml


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def resolve(value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as connection:
        connection.settimeout(0.2)
        return connection.connect_ex(("127.0.0.1", int(port))) == 0


def wait_for_ports_free(ports, timeout_s):
    deadline = time.monotonic() + float(timeout_s)
    while time.monotonic() < deadline:
        if not any(port_in_use(port) for port in ports):
            return True
        time.sleep(0.25)
    return not any(port_in_use(port) for port in ports)


def stop_process(process, timeout):
    if process is None or process.poll() is not None:
        return
    for requested in (signal.SIGINT, signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(process.pid, requested)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=timeout if requested != signal.SIGKILL else 5.0)
            return
        except subprocess.TimeoutExpired:
            continue


def request_orderly_shutdown(_signum, _frame):
    """Convert terminal loss/termination into the normal cleanup path."""

    raise KeyboardInterrupt


def wait_for_model(env, timeout, process):
    deadline = time.monotonic() + float(timeout)
    command = [
        "rosservice", "call", "/gazebo/get_model_state", "model_name: 'urban_model'"
    ]
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("memory roslaunch exited with code %d" % process.returncode)
        try:
            output = subprocess.run(
                command,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=4.0,
            )
            if output.returncode == 0 and "success: True" in output.stdout:
                return
        except subprocess.TimeoutExpired:
            pass
        time.sleep(1.0)
    raise RuntimeError("Urban Car did not become ready")


def require_runtime_packages(dry_run=False):
    missing = []
    for package in ("robot_localization", "slam_toolbox"):
        result = subprocess.run(
            ["rospack", "find", package],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if result.returncode:
            missing.append(package)
    if missing and not dry_run:
        raise RuntimeError(
            "missing ROS packages: %s; install ros-noetic-robot-localization "
            "ros-noetic-slam-toolbox" % ",".join(missing)
        )
    return missing


def load_scenario_manifest(config):
    path = resolve(config["scenario_manifest"])
    if not path.is_file():
        raise ValueError("scenario manifest does not exist: " + str(path))
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("unable to read scenario manifest: %s" % exc) from exc
    if manifest.get("schema") not in {
        "DEPCarP6StaticScenarioManifestV1",
        "DEPCarP6ReproductionScenarioManifestV1",
    }:
        raise ValueError("unsupported P6 scenario manifest schema")
    scenarios = manifest.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("scenario manifest contains no scenarios")
    identifiers = [row.get("scenario_id") for row in scenarios]
    if any(not isinstance(value, str) or not value for value in identifiers):
        raise ValueError("scenario manifest contains an invalid scenario_id")
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("scenario manifest contains duplicate scenario_id values")
    expected = manifest.get("scenario_manifest_sha256")
    if expected is not None:
        content = dict(manifest)
        content.pop("scenario_manifest_sha256", None)
        if canonical_sha256(content) != expected:
            raise ValueError("scenario manifest identity mismatch: " + str(path))
    return manifest, path


def scenario_artifact_preflight(scenario, require_robust_start=True):
    """Verify a frozen map and its audited start before Gazebo is launched."""

    scenario_id = str(scenario.get("scenario_id", "<unknown>"))
    errors = []
    required = (
        "world", "world_sha256", "map_yaml", "map_yaml_sha256",
        "map_name", "map_uuid", "map_seed", "map_occupancy_sha256",
        "start", "goal", "gazebo_seed",
    )
    errors.extend("missing field: " + key for key in required if key not in scenario)
    world = resolve(scenario.get("world", ""))
    map_yaml = resolve(scenario.get("map_yaml", ""))
    map_folder = world.parent
    if not world.is_file():
        errors.append("world does not exist: " + str(world))
    elif sha256_file(world) != scenario.get("world_sha256"):
        errors.append("world SHA-256 mismatch: " + str(world))
    if not map_yaml.is_file():
        errors.append("map YAML does not exist: " + str(map_yaml))
    elif sha256_file(map_yaml) != scenario.get("map_yaml_sha256"):
        errors.append("map YAML SHA-256 mismatch: " + str(map_yaml))
    if world.parent != map_yaml.parent:
        errors.append("world and map YAML are not in the same frozen map folder")
    if map_folder.name != scenario.get("map_name"):
        errors.append("map folder name does not match map_name")

    map_manifest_path = map_folder / "manifest.json"
    model_path = map_folder / "model.sdf"
    if not model_path.is_file():
        errors.append("frozen map has no model.sdf: " + str(model_path))
    if not map_manifest_path.is_file():
        errors.append("frozen map has no manifest.json: " + str(map_manifest_path))
    else:
        try:
            map_manifest = json.loads(map_manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            errors.append("unable to read frozen map manifest: %s" % exc)
        else:
            identity = {
                "name": scenario.get("map_name"),
                "map_uuid": scenario.get("map_uuid"),
                "seed": scenario.get("map_seed"),
                "occupancy_sha256": scenario.get("map_occupancy_sha256"),
            }
            for key, expected in identity.items():
                observed = map_manifest.get(key)
                if key == "seed":
                    try:
                        observed, expected = int(observed), int(expected)
                    except (TypeError, ValueError):
                        pass
                if observed != expected:
                    errors.append("map manifest %s identity mismatch" % key)

    if map_yaml.is_file():
        try:
            map_metadata = yaml.safe_load(map_yaml.read_text(encoding="utf-8"))
            image = (map_yaml.parent / str(map_metadata["image"])).resolve()
            if image.parent != map_yaml.parent or not image.is_file():
                errors.append("map image is missing or escapes its frozen folder")
            else:
                with Image.open(str(image)) as map_image:
                    occupancy_bytes = map_image.convert("L").tobytes()
                occupancy_sha256 = hashlib.sha256(occupancy_bytes).hexdigest()
                if occupancy_sha256 != scenario.get("map_occupancy_sha256"):
                    errors.append("decoded map occupancy SHA-256 mismatch")
        except (OSError, KeyError, TypeError, ValueError) as exc:
            errors.append("unable to validate map image: %s" % exc)

    for key in ("start", "goal"):
        pose = scenario.get(key)
        try:
            valid_pose = (
                isinstance(pose, (list, tuple))
                and len(pose) == 3
                and all(math.isfinite(float(value)) for value in pose)
            )
        except (TypeError, ValueError):
            valid_pose = False
        if not valid_pose:
            errors.append("%s must contain finite x, y and yaw" % key)

    robustness = scenario.get("start_robustness", {})
    try:
        robust = bool(
            robustness.get("schema") == "DEPCarP6StartRobustnessV1"
            and robustness.get("status") == "PASS"
            and int(robustness.get("failure_count", -1)) == 0
            and int(robustness.get("perturbations", 0)) > 0
            and int(robustness.get("minimum_safe_ackermann_primitives", 0)) > 0
        )
    except (AttributeError, TypeError, ValueError):
        robust = False
    if require_robust_start and not robust:
        errors.append("start robustness evidence is not PASS")
    return {
        "scenario_id": scenario_id,
        "status": "PASS" if not errors else "FAIL",
        "map_name": scenario.get("map_name"),
        "map_seed": scenario.get("map_seed"),
        "map_uuid": scenario.get("map_uuid"),
        "world": str(world),
        "map_yaml": str(map_yaml),
        "start_robustness_passed": robust,
        "errors": errors,
    }


def load_scenario(config, scenario_id):
    manifest, _ = load_scenario_manifest(config)
    scenario = next(
        (row for row in manifest["scenarios"] if row["scenario_id"] == scenario_id),
        None,
    )
    if scenario is None:
        raise ValueError("scenario is not present in frozen manifest: " + scenario_id)
    preflight = scenario_artifact_preflight(scenario, require_robust_start=True)
    if preflight["status"] != "PASS":
        raise ValueError(
            "scenario preflight failed for %s: %s"
            % (scenario_id, "; ".join(preflight["errors"]))
        )
    return manifest, scenario


def command(config, scenario, args, ros_port):
    policy_mode = args.policy_mode
    launch = [
        "roslaunch", "-p", str(ros_port), "dep_car_bringup", "p6_memory_static.launch",
        "world:=" + str(resolve(scenario["world"])),
        "gazebo_model_path:=" + str(resolve(scenario["world"]).parent.parent),
        "x:=" + str(scenario["start"][0]),
        "y:=" + str(scenario["start"][1]),
        "yaw:=" + str(scenario["start"][2]),
        "gazebo_seed:=" + str(scenario["gazebo_seed"]),
        "checkpoint:=" + str(resolve(config["checkpoint"])),
        "checkpoint_contract:=" + str(resolve(config["checkpoint_contract"])),
        "policy_mode:=" + policy_mode,
        "gui:=" + ("false" if args.headless else "true"),
        "enable_rviz:=" + ("false" if args.headless else "true"),
        "paused:=true",
    ]
    if config.get("p6_authority"):
        launch.append(
            "p6_authority:=" + str(resolve(config["p6_authority"]))
        )
    reset = [
        "rosrun", "dep_car_dataset", "reset_pilot_pose.py",
        "--x", str(scenario["start"][0]),
        "--y", str(scenario["start"][1]),
        "--yaw", str(scenario["start"][2]),
    ]
    fixed_probe = config.get("fixed_recovery_probe", {})
    use_fixed_probe = (
        args.stage == "fixed"
        and args.goal_x is None
        and scenario.get("scenario_id") == config.get("fixed_default_scenario")
        and isinstance(fixed_probe.get("goal"), list)
    )
    goal = list(fixed_probe["goal"] if use_fixed_probe else scenario["goal"])
    if args.goal_x is not None:
        goal[0], goal[1] = args.goal_x, args.goal_y
        if args.goal_yaw is not None:
            goal[2] = args.goal_yaw
    publish = [
        "rosrun", "dep_car_memory_navigation", "publish_memory_goal.py",
        "--x", str(goal[0]), "--y", str(goal[1]), "--yaw", str(goal[2]),
        "--frame", str(fixed_probe.get("frame", "odom") if use_fixed_probe else "odom"),
    ]
    return launch, reset, publish, goal


def automated_goal_command(goals, timeout_s, report_path=None, *, frame="map"):
    command = [
        "rosrun",
        "dep_car_memory_navigation",
        "replay_memory_goals.py",
        "--goal-timeout",
        str(float(timeout_s)),
        "--frame",
        str(frame),
    ]
    if report_path is not None:
        command.extend(["--output", str(Path(report_path).resolve())])
    for goal in goals:
        command.extend(["--goal", str(goal[0]), str(goal[1]), str(goal[2])])
    return command


def episode_automated_goal_command(effective_goal, timeout_s, report_path=None):
    """Build an episode replay from the effective, possibly CLI-overridden goal."""
    return automated_goal_command(
        [list(effective_goal)], timeout_s, report_path, frame="odom"
    )


def _scenario_collision_boxes(scenario):
    """Read axis-aligned Gazebo wall boxes for host-only replay preflight.

    These boxes are test-fixture evidence only.  They are never passed to the
    online SLAM, FAR or local-planner runtime.
    """

    world = resolve(scenario["world"])
    model = world.parent / "model.sdf"
    if not model.is_file():
        raise ValueError("frozen scenario has no model.sdf beside its world")
    boxes = []
    for link in ET.parse(str(model)).getroot().findall(".//link"):
        pose_text = link.findtext("pose")
        size_text = link.findtext("collision/geometry/box/size")
        if pose_text is None or size_text is None:
            continue
        pose = [float(value) for value in pose_text.split()]
        size = [float(value) for value in size_text.split()]
        if len(pose) < 2 or len(size) < 2:
            continue
        boxes.append({
            "name": str(link.get("name", "unnamed_collision")),
            "center": (pose[0], pose[1]),
            "size": (size[0], size[1]),
        })
    if not boxes:
        raise ValueError("frozen scenario model contains no wall collision boxes")
    return boxes


def regression_goal_preflight(sequence, scenario, minimum_clearance_m=0.70):
    """Reject stale online-map coordinates and goals inside frozen walls."""

    frame = str(sequence.get("coordinate_frame", "map")).lstrip("/") or "map"
    goals = sequence.get("goals", [])
    errors = []
    evidence = []
    if frame != "odom":
        errors.append(
            "fixed replay goals must use stable odom coordinates; an online "
            "SLAM map frame is not reproducible across restarts"
        )
    boxes = _scenario_collision_boxes(scenario)
    x_min = min(box["center"][0] - 0.5 * box["size"][0] for box in boxes)
    x_max = max(box["center"][0] + 0.5 * box["size"][0] for box in boxes)
    y_min = min(box["center"][1] - 0.5 * box["size"][1] for box in boxes)
    y_max = max(box["center"][1] + 0.5 * box["size"][1] for box in boxes)
    for index, goal in enumerate(goals):
        if not isinstance(goal, (list, tuple)) or len(goal) != 3:
            errors.append("goal %d must contain x, y and yaw" % index)
            continue
        x, y, yaw = (float(value) for value in goal)
        nearest = None
        for box in boxes:
            center_x, center_y = box["center"]
            size_x, size_y = box["size"]
            dx = max(abs(x - center_x) - 0.5 * size_x, 0.0)
            dy = max(abs(y - center_y) - 0.5 * size_y, 0.0)
            clearance = (dx * dx + dy * dy) ** 0.5
            if nearest is None or clearance < nearest[0]:
                nearest = (clearance, box["name"])
        clearance, obstacle = nearest
        inside_bounds = x_min <= x <= x_max and y_min <= y <= y_max
        valid = bool(inside_bounds and clearance >= float(minimum_clearance_m))
        evidence.append({
            "goal_index": int(index),
            "goal": [x, y, yaw],
            "minimum_static_clearance_m": float(clearance),
            "nearest_collision": obstacle,
            "inside_frozen_world_bounds": bool(inside_bounds),
            "valid": valid,
        })
        if not valid:
            errors.append(
                "goal %d has %.3fm static clearance (required %.3fm; nearest %s)"
                % (index, clearance, float(minimum_clearance_m), obstacle)
            )
    return {
        "schema": "DEPCarRegressionGoalPreflightV1",
        "status": "PASS" if not errors else "FAIL",
        "reason": "VALID_REPLAY_GOALS" if not errors else "INVALID_REPLAY_GOAL",
        "coordinate_frame": frame,
        "minimum_clearance_m": float(minimum_clearance_m),
        "goals": evidence,
        "errors": errors,
        "runtime_ground_truth_input": False,
    }


def archive_previous_report(path):
    """Keep an interrupted run from masquerading as fresh acceptance evidence."""

    path = Path(path)
    if not path.is_file():
        return None
    history = path.parent / "history"
    history.mkdir(parents=True, exist_ok=True)
    stamp = int(path.stat().st_mtime_ns)
    archived = history / ("%s.%d%s" % (path.stem, stamp, path.suffix))
    path.replace(archived)
    return archived


def matrix_commands(config, manifest, args):
    rows = [
        row
        for row in manifest["scenarios"]
        if row.get("cohort") == args.cohort
        and row.get("start_robustness", {}).get("status") == "PASS"
    ]
    # A bounded matrix should exercise different generated maps before taking
    # a second scenario from the same map.  The shuffle is seeded so failures
    # remain exactly replayable while changing --selection-seed produces a new
    # robustness sample without changing runtime policy.
    random.Random(int(args.selection_seed)).shuffle(rows)
    distinct_maps, repeated_maps, seen_maps = [], [], set()
    for row in rows:
        map_key = (row.get("map_seed"), row.get("map_uuid"))
        target = distinct_maps if map_key not in seen_maps else repeated_maps
        target.append(row)
        seen_maps.add(map_key)
    rows = distinct_maps + repeated_maps
    if args.maximum_scenarios > 0:
        rows = rows[: args.maximum_scenarios]
    commands = []
    for row in rows:
        item = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--stage",
            "episode",
            "--config",
            str(args.config.resolve()),
            "--scenario",
            row["scenario_id"],
            "--policy-mode",
            args.policy_mode,
            "--goal-timeout",
            str(args.goal_timeout),
            "--headless",
        ]
        scenario_manifest = getattr(args, "scenario_manifest", None)
        if scenario_manifest is not None:
            item.extend([
                "--scenario-manifest",
                str(resolve(scenario_manifest)),
            ])
        commands.append((row["scenario_id"], item, row))
    return commands


def implementation_audit(config):
    missing = require_runtime_packages(dry_run=True)
    launch_args = [
        "world:=/tmp/dep_car_memory_audit.world",
        "gazebo_model_path:=/tmp",
        "checkpoint:=" + str(resolve(config["checkpoint"])),
        "checkpoint_contract:=" + str(resolve(config["checkpoint_contract"])),
    ]
    if config.get("p6_authority"):
        launch_args.append(
            "p6_authority:=" + str(resolve(config["p6_authority"]))
        )
    nodes_result = subprocess.run(
        ["roslaunch", "--nodes", "dep_car_bringup", "p6_memory_static.launch"]
        + launch_args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    nodes = sorted(
        line.strip() for line in nodes_result.stdout.splitlines() if line.startswith("/")
    )
    required_nodes = {
        "/dep_car_wheel_odometry",
        "/dep_car_ekf",
        "/dep_car_map_odometry",
        "/slam_toolbox",
        "/dep_car_memory_navigation",
        "/dep_car_local_planner",
        "/dep_car_lidar",
    }
    forbidden_nodes = {"/map_server", "/dep_car_hybrid_astar", "/dep_car_odometry_tf"}
    memory_files = sorted(
        path for path in (ROOT / "ros/dep_car_memory_navigation").rglob("*") if path.is_file()
    )
    truth_mentions = [
        str(path.relative_to(ROOT))
        for path in memory_files
        if "/base_pose_ground_truth" in path.read_text(encoding="utf-8", errors="ignore")
    ]
    lidar_source = (ROOT / "ros/dep_car_perception/scripts/lidar_preprocessor.py").read_text(
        encoding="utf-8"
    )
    local_launch = (ROOT / "ros/dep_car_local_planner/launch/local_planner.launch").read_text(
        encoding="utf-8"
    )
    memory_launch = (ROOT / "ros/dep_car_bringup/launch/p6_memory_static.launch").read_text(
        encoding="utf-8"
    )
    estimator_launch = (
        ROOT / "ros/dep_car_memory_navigation/launch/memory_navigation.launch"
    ).read_text(encoding="utf-8")
    message = (ROOT / "ros/dep_car_msgs/msg/LocalRouteCommand.msg").read_text(
        encoding="utf-8"
    )
    planner_message = (ROOT / "ros/dep_car_msgs/msg/PlannerState.msg").read_text(
        encoding="utf-8"
    )
    policy_message = (ROOT / "ros/dep_car_msgs/msg/PolicyState.msg").read_text(
        encoding="utf-8"
    )
    local_source = (
        ROOT / "ros/dep_car_local_planner/scripts/local_planner_node.py"
    ).read_text(encoding="utf-8")
    memory_source = (
        ROOT / "ros/dep_car_memory_navigation/scripts/navigation_memory_node.py"
    ).read_text(encoding="utf-8")
    replay_source = (
        ROOT / "ros/dep_car_memory_navigation/scripts/replay_memory_goals.py"
    ).read_text(encoding="utf-8")
    runtime_source = Path(__file__).read_text(encoding="utf-8")
    memory_runtime_source = (
        ROOT / "dep_car/src/dep_car/runtime/navigation_memory.py"
    ).read_text(encoding="utf-8")
    visibility_runtime_source = (
        ROOT / "dep_car/src/dep_car/runtime/far_visibility.py"
    ).read_text(encoding="utf-8")
    map_odometry_source = (
        ROOT / "ros/dep_car_memory_navigation/scripts/map_odometry_node.py"
    ).read_text(encoding="utf-8")
    memory_config_source = (
        ROOT / "ros/dep_car_memory_navigation/config/navigation_memory.yaml"
    ).read_text(encoding="utf-8")
    rviz_source = (
        ROOT / "ros/dep_car_bringup/rviz/dep_car.rviz"
    ).read_text(encoding="utf-8")
    ekf = yaml.safe_load(
        (ROOT / "ros/dep_car_memory_navigation/config/ekf.yaml").read_text(
            encoding="utf-8"
        )
    )
    regression = config.get("regression_sequences", {})
    regression_values = json.dumps(regression, sort_keys=True)
    upstream_directory = ROOT / "third_party/far_planner"
    upstream_revision = subprocess.run(
        ["git", "-C", str(upstream_directory), "rev-parse", "HEAD"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    ) if upstream_directory.is_dir() else None
    upstream_status = subprocess.run(
        ["git", "-C", str(upstream_directory), "status", "--porcelain"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    ) if upstream_directory.is_dir() else None
    locked_far_revision = "2799b6964c141cacd1c32a14b19bc7abffbe0e52"
    gates = {
        "M0_dual_backend": (
            nodes_result.returncode == 0
            and required_nodes.issubset(nodes)
            and not forbidden_nodes.intersection(nodes)
        ),
        "M1_no_runtime_ground_truth": (
            not truth_mentions
            and "odometry_topic" in local_launch
            and "/dep_car/map_odometry" in memory_launch
            and "/odometry/filtered" in estimator_launch
            and (ROOT / "ros/dep_car_vehicle/scripts/ackermann_wheel_odometry.py").is_file()
        ),
        "M2_online_slam_and_scan": (
            'Publisher("/dep_car/scan"' in lidar_source
            and (ROOT / "ros/dep_car_memory_navigation/config/slam_toolbox.yaml").is_file()
            and "/map_server" not in nodes
        ),
        "M3_breadcrumb_topology_memory": (
            (ROOT / "dep_car/src/dep_car/runtime/navigation_memory.py").is_file()
            and "BreadcrumbTrail" in (
                ROOT / "dep_car/src/dep_car/runtime/navigation_memory.py"
            ).read_text(encoding="utf-8")
        ),
        "M4_dead_end_backtrack_authority": (
            "NAVIGATION_MEMORY_BACKTRACK=2" in message
            and "NAVIGATION_MEMORY_RESUME=3" in message
            and "BACKTRACK_REVERSE" in (
                ROOT / "ros/dep_car_memory_navigation/scripts/navigation_memory_node.py"
            ).read_text(encoding="utf-8")
            and "allow_static_margin_egress=memory_backtrack" in (
                ROOT / "ros/dep_car_local_planner/scripts/local_planner_node.py"
            ).read_text(encoding="utf-8")
            and "evaluate_static_margin_egress" in (
                ROOT / "dep_car/src/dep_car/core/safety.py"
            ).read_text(encoding="utf-8")
        ),
        "M4_1_replay_is_not_runtime_specialization": (
            bool(regression)
            and "matrix" in runtime_source
            and "runtime_policy_has_map_specific_branches" in runtime_source
            and not any(value in memory_source for value in ("6.536", "-4.264"))
            and not any(value in local_source for value in ("6.536", "-4.264"))
            and "p6_c344e792d09ac2d6" in regression_values
        ),
        "M4_2_policy_sync_contract": (
            int(ekf.get("frequency", 0)) >= 50
            and "policy_odometry_topic" in memory_launch
            and "/odometry/filtered" in memory_launch
            and "synchronization_failures" in policy_message
        ),
        "M4_3_transaction_authority": (
            "TURNAROUND_VERIFY" in local_source
            and "turnaround_gear_order" in local_source
            and "maneuver_active" in planner_message
            and "forward_corridor_reacquired\")" not in local_source
        ),
        "M4_4_robust_memory_handoff": (
            "minimum_backtrack_before_resume" in memory_source
            and "maneuver_active=local_maneuver_active" in memory_source
            and "resume_failed" in memory_source
            and "continue_goal_seek" in memory_source
            and "truncate_after" in memory_source
        ),
        "M4_5_multi_seed_and_runtime_evidence": (
            "selection_seed" in runtime_source
            and "selected_scenarios" in runtime_source
            and "policy_sync_success_rate" in replay_source
            and "goal_metrics" in replay_source
            and "forward_distance_m" in replay_source
        ),
        "M5_1_accumulated_map_local_fusion": (
            "accumulated_map_topic: /map" in memory_config_source
            and "on_accumulated_map" in memory_source
            and "unknown_is_occupied=False" in memory_source
            and "persistent_profile" in memory_source
            and "/map_server" not in nodes
        ),
        "M5_2_sparse_topology_and_failed_branch_memory": (
            "class TopologicalMemory" in memory_runtime_source
            and "mark_failed_branch" in memory_runtime_source
            and "failed_branch_retained=True" in memory_source
            and "failed_branches" in memory_source
        ),
        "M5_3_goal_conditioned_branch_guidance_without_grid_astar": (
            "guidance_path" in memory_runtime_source
            and "EXPLORED_TOPOLOGY" in memory_source
            and '"uses_full_map_search": False' in memory_source
            and "/dep_car_hybrid_astar" not in nodes
        ),
        "M5_4_progress_bounded_recovery_and_goal_preemption": (
            "older_index_at_distance" in memory_runtime_source
            and "resume_target_index" in memory_source
            and "resume_maximum_travel" in memory_source
            and "preempted_state" in memory_source
        ),
        "M5_5_rviz_explainability_and_v2_route_handoff": (
            "/dep_car/navigation_memory_markers" in memory_source
            and "Navigation Memory" in rviz_source
            and "guidance_source" in memory_source
            and "/dep_car/global_route" in memory_source
        ),
        "M5_6_multi_map_runtime_metrics": (
            "DEPCarMemoryGoalReplayV3" in replay_source
            and "recoveries_after_escape" in replay_source
            and "maximum_failed_branches" in replay_source
            and "wait_for_ports_free" in runtime_source
        ),
        "M5_7_stateful_boundary_following_without_dense_astar": (
            "class BoundaryFollowSupervisor" in memory_runtime_source
            and "progressing_direct_corridor_confirmed" in memory_runtime_source
            and "returned_to_boundary_hit_region" in memory_runtime_source
            and "class BoundarySideFailure" in memory_runtime_source
            and "remember_current_failure" in memory_source
            and "BOUNDARY_FOLLOW_LEFT" in memory_source
            and "boundary_loop_reason" in memory_source
            and "boundary_leave_progress_m" in memory_config_source
            and "boundary_failure_radius_m" in memory_config_source
            and '"uses_full_map_search": False' in memory_source
            and "/dep_car_hybrid_astar" not in nodes
        ),
        "M6_1_dynamic_polygon_visibility_routing": (
            "class DynamicVisibilityPlanner" in visibility_runtime_source
            and "findContours" in visibility_runtime_source
            and "KNOWN_VISIBILITY" in visibility_runtime_source
            and "ATTEMPTABLE_VISIBILITY" in visibility_runtime_source
            and "FAR_KNOWN_VISIBILITY" in memory_source
            and "/dep_car/visibility_path" in memory_source
            and '"uses_dynamic_visibility_graph": True' in memory_source
            and "astar" not in visibility_runtime_source.lower()
            and "/dep_car_hybrid_astar" not in nodes
        ),
        "M6_2_v2_route_tube_handoff": (
            "visibility_corridor_body" in memory_source
            and "forward_corridor_body" in memory_source
            and "/dep_car/global_route" in memory_source
            and "FAR Visibility Route" in rviz_source
            and "FAR Visibility Graph" in rviz_source
        ),
        "M6_3_coherent_bounded_slam_odometry_diagnostics": (
            "maximum_transform_skew_s" in map_odometry_source
            and "rejected stale SLAM TF skew" in map_odometry_source
            and "Mixing an" in map_odometry_source
            and "/dep_car/map_odom_correction" in map_odometry_source
            and "DEPCarMapOdomCorrectionV1" in map_odometry_source
            and "PlanarTransformRevisionTracker" in map_odometry_source
            and "if revision is not None:" in map_odometry_source
            and ekf.get("odom0_config") == [
                True, True, False,
                False, False, False,
                True, False, False,
                False, False, False,
                False, False, False,
            ]
            and ekf.get("imu0_config") == [
                False, False, False,
                False, False, True,
                False, False, False,
                False, False, True,
                False, False, False,
            ]
            and ekf.get("imu0_relative") is False
            and "map_odom_correction" in replay_source
        ),
        "M6_4_locked_upstream_provenance": (
            (ROOT / "scripts/fetch_far_planner_upstream.sh").is_file()
            and locked_far_revision
            in (ROOT / "scripts/fetch_far_planner_upstream.sh").read_text(
                encoding="utf-8"
            )
            and "far_planner:" in (ROOT / "third_party.lock.yaml").read_text(
                encoding="utf-8"
            )
            and upstream_revision is not None
            and upstream_revision.returncode == 0
            and upstream_revision.stdout.strip() == locked_far_revision
            and upstream_status is not None
            and upstream_status.returncode == 0
            and not upstream_status.stdout.strip()
        ),
        "M6_5_visibility_route_owns_recovery_authority": (
            "breadcrumb_motion_authority: false" in memory_config_source
            and "backtrack_enabled=self.breadcrumb_motion_authority" in memory_source
            and '"topology_anchor_and_closed_loop_far_dead_end_egress"' in memory_source
            and '"confirmed_local_static_block"' in memory_source
            and '"slam_map_odom_correction"' in memory_source
            and "last_significant_map_correction" in memory_source
            and "map_correction_replan_minimum_period_s" in memory_config_source
            and 'marker.header.frame_id = "odom"' in memory_source
            and "marker.frame_locked = True" in memory_source
        ),
        "M6_6_online_route_acquisition_and_bounded_static_egress": (
            "class VisibilityRouteAcquisitionGate" in visibility_runtime_source
            and "visibility_initial_exploration_minimum_m"
            in memory_config_source
            and "visibility_initial_exploration_maximum_duration_s"
            in memory_config_source
            and "far_bootstrap_motion_authorized" in memory_source
            and "visibility_start_heading_weight_m" in memory_config_source
            and "far_static_replans_before_egress" in memory_config_source
            and "force_certified_egress" in memory_source
            and "certified_far_egress=True" in memory_source
            and '"FAR_MAPPING_WAIT"' in memory_source
        ),
        "M6_7_single_navigation_authority_and_maze_detour_gate": (
            "path_unknown_fraction" in visibility_runtime_source
            and "stable_observed_high_detour_route" in visibility_runtime_source
            and "visibility_terminal_direct_handoff_radius_m" in memory_config_source
            and "KNOWN_TERMINAL_DIRECT" in memory_source
            and "FAR_ATTEMPTABLE_NAVIGATION" in memory_source
            and "LOCAL_SAFE_EXPLORATION" in memory_source
            and "EXPLORED_TOPOLOGY_ROUTE" in memory_source
            and "explored_topology_corridor_body" in memory_source
            and "allow_direct_goal=False" in memory_source
            and "breadcrumb_backtrack or far_dead_end_egress" in local_source
            and "breadcrumb_backtrack\n                    or memory_resume"
            in local_source
            and "far_dead_end_egress_realign" in local_source
            and "start_authority_transaction" in local_source
        ),
        "M6_8_rolling_route_renewal_and_continuity": (
            "visibility_route_renewal_period_s" in memory_config_source
            and "visibility_route_renewal_distance_m" in memory_config_source
            and "visibility_route_continuity_minimum_m" in memory_config_source
            and "visibility_route_dropout_grace_s" in memory_config_source
            and "visibility_route_lease_prefix_m" in memory_config_source
            and "visibility_no_route_static_evidence_timeout_s"
            in memory_config_source
            and "duplicate_map_updates_skipped" in memory_source
            and "zlib.crc32" in memory_source
            and "visibility_route_replacement_maximum_direction_change_rad"
            in memory_config_source
            and "retaining_safe_active_route_" in memory_source
            and "retaining_leased_active_route_safe_local_prefix"
            in memory_source
            and "transient_route_lease_authorized"
            in visibility_runtime_source
            and "measured_pose_revalidation_retaining_active_route"
            in memory_source
            and "route_lease_refresh_due" in memory_source
            and "recent_far_authority_dropout_local_continuation"
            in memory_source
            and "explored_topology_motion_authority: false"
            in memory_config_source
            and "transient_route_lease_refresh" in memory_source
            and "replacement_direction_discontinuous" in memory_source
            and "rolling_route_renewal" in memory_source
            and "visibility_active_route_accepted" in memory_source
            and "visibility_active_route_motion_authorized" in memory_source
            and "not candidate_motion_authorized" in memory_source
            and "preserve_acquisition=preserve_acquisition" in memory_source
            and "A local recovery manoeuvre may start while FAR has no"
            in memory_source
            and "route_unavailable_after_recent_static_block" in memory_source
            and "hard_route_hold" in memory_source
            and "ends the acquisition transaction completely"
            in visibility_runtime_source
        ),
        "M6_9_attemptable_navigation_liveness": (
            "motion_authorized" in visibility_runtime_source
            and "stable_attemptable_navigation_high_detour"
            in visibility_runtime_source
            and "FAR_ATTEMPTABLE_NAVIGATION" in memory_source
            and "visibility_active_route_motion_authorized" in memory_source
            and "Do not stop after a fixed pulse" in memory_source
        ),
        "M6_10_no_route_local_liveness_and_replan_efficiency": (
            "LOCAL_SAFE_EXPLORATION" in memory_source
            and "no_route_rolling_local_exploration" in memory_source
            and "unconfirmed_route_rolling_local_exploration" in memory_source
            and "accumulated_map_observation_revision" in memory_source
            and "FAR stable candidate bootstrap promotion" in memory_source
            and "same cached grid cannot turn one speculative" in memory_source
            and "currently observed LiDAR-free swept footprints"
            in memory_config_source
            and "preserve_acquisition=preserve_acquisition" in memory_source
            and "duplicate_map_updates_skipped" in memory_source
            and "planning_ms=%.1f" in memory_source
        ),
        "M6_11_monotonic_route_carrot_and_single_turnaround_transaction": (
            "class MonotonicRouteProgress" in memory_runtime_source
            and "preview_handoff" in memory_runtime_source
            and "projection_rollback_tolerance_m" in memory_runtime_source
            and "rolling_route_minimum_lookahead_m" in memory_config_source
            and "rolling_route_maximum_lookahead_m" in memory_config_source
            and "active_rolling_route" in memory_source
            and "candidate_tangent_discontinuous" in memory_runtime_source
            and "TOPOLOGY_REAR_SUPPRESSED" in memory_source
            and "topology_last_turnaround_route_id" in memory_source
            and "sync_route_turnaround_transaction" in memory_source
            and "route_turnaround_transaction" in memory_source
            and "local_turnaround_transaction_id" in local_source
            and "Buffered unmatched FAR route transaction halves"
            in local_source
            and "Queued complete route transaction until committed maneuver"
            in local_source
            and "far_rolling_carrot" in memory_source
        ),
        "M6_12_local_first_handoff_and_verified_turnaround_exit": (
            "visibility_initial_exploration_minimum_m" in memory_config_source
            and "visibility_initial_exploration_maximum_duration_s"
            in memory_config_source
            and "observe_initial_local_exploration" in memory_source
            and "far_bootstrap_motion_authorized" in memory_source
            and "same cached grid cannot turn one speculative" in memory_source
            and "sufficiently_observed = bool(" in visibility_runtime_source
            and "and sufficiently_observed" in visibility_runtime_source
            and "self.confirmations >= required_confirmations"
            in visibility_runtime_source
            and "forward_exit_verified" in local_source
            and "last_completed_reason" in local_source
            and "forward_exit_capture" in local_source
            and "force=True" in memory_source
        ),
        "M6_13_capture_gated_distance_bounded_rolling_subgoal": (
            "carrot_capture_radius_m" in memory_runtime_source
            and "maximum_carrot_advance_m" in memory_runtime_source
            and "maximum_carrot_distance_m" in memory_runtime_source
            and "waiting_for_vehicle_capture" in memory_runtime_source
            and "candidate_carrot_m" in memory_runtime_source
            and "rolling_target_world" in memory_source
            and "rolling_target_latched=True" in memory_source
            and "rolling_route_carrot_capture_radius_m"
            in memory_config_source
            and "rolling_route_maximum_carrot_advance_m"
            in memory_config_source
            and "rolling_route_maximum_carrot_distance_m"
            in memory_config_source
        ),
        "M6_14_atomic_far_handoff_and_serial_local_turnaround": (
            "string route_source" in message
            and "uint64 authority_epoch" in message
            and "bool rolling_target_latched" in message
            and "Accepted atomic navigation authority handoff"
            in local_source
            and "status_matches_command" in local_source
            and "navigation_authority_reference" in local_source
            and "FAR keeps planning in parallel"
            in memory_source
            and "request_route_revalidation" in local_source
            and "background_plan, background_path = self.update_visibility_plan("
            in memory_source.split("if local_maneuver_active:", 1)[1].split(
                "goal_distance =", 1
            )[0]
            and "deferred_route_transaction" in local_source
            and "Queued complete route transaction until committed maneuver"
            in local_source
            and "stale_fallback_turnaround_replaced_by_far" in local_source
            and '"EXPLORED_TOPOLOGY", "LOCAL_SAFE_EXPLORATION"'
            in local_source
            and "stable_far_forward_exit_available" in local_source
            and "turnaround_alignment_minimum_spatial_scale" in local_source
            and "forward_probe = None" in local_source
            and (
                "local_state, local_maneuver_reported, odom_pose, stamp"
                in memory_source
            )
        ),
        "M6_15_directional_far_dead_end_egress": (
            "NAVIGATION_FAR_DEAD_END_EGRESS=4" in message
            and "FAR_DEAD_END_EGRESS" in memory_runtime_source
            and "directed_failed_branches" in visibility_runtime_source
            and "_transition_enters_failed_branch" in visibility_runtime_source
            and "longest_margin_egress_prefix" in visibility_runtime_source
            and "validate_dead_end_egress_route" in memory_source
            and "breadcrumb_corridor_map" in memory_source
            and "signed, closed-loop exit" in memory_source
            and "dead_end_escape_lookahead_m" in memory_config_source
            and "EGRESS_BLOCKED_BY_CURRENT_HARD_SAFETY" in memory_source
            and "egress_bidirectional_safety_exhausted" in memory_source
            and "far_dead_end_egress_realign" in local_source
            and "topology_anchor_and_closed_loop_far_dead_end_egress"
            in memory_source
            and "far_dead_end_egress_route" in memory_source
            and "HARD_VETO>FAR_DEAD_END_EGRESS>LOCAL_TURNAROUND"
            in memory_source
            and "far_dead_end_egress_transactions" in replay_source
            and "maximum_far_dead_end_egress_cross_track_error_m"
            in replay_source
        ),
        "runtime_dependencies": not missing,
    }
    tracked = [
        ROOT / "dep_car/src/dep_car/runtime/wheel_odometry.py",
        ROOT / "dep_car/src/dep_car/runtime/navigation_memory.py",
        ROOT / "dep_car/src/dep_car/runtime/far_visibility.py",
        ROOT / "dep_car/src/dep_car/core/occupancy.py",
        ROOT / "dep_car/src/dep_car/runtime/occupancy.py",
        ROOT / "dep_car/src/dep_car/core/safety.py",
        ROOT / "dep_car/src/dep_car/core/planner.py",
        ROOT / "dep_car/src/dep_car/core/recovery.py",
        ROOT / "ros/dep_car_memory_navigation/scripts/navigation_memory_node.py",
        ROOT / "ros/dep_car_memory_navigation/scripts/map_odometry_node.py",
        ROOT / "ros/dep_car_memory_navigation/scripts/replay_memory_goals.py",
        ROOT / "ros/dep_car_memory_navigation/config/ekf.yaml",
        ROOT / "ros/dep_car_memory_navigation/config/slam_toolbox.yaml",
        ROOT / "ros/dep_car_memory_navigation/config/navigation_memory.yaml",
        ROOT / "ros/dep_car_memory_navigation/launch/memory_navigation.launch",
        ROOT / "ros/dep_car_memory_navigation/package.xml",
        ROOT / "ros/dep_car_bringup/launch/p6_memory_static.launch",
        ROOT / "ros/dep_car_bringup/rviz/dep_car.rviz",
        ROOT / "ros/dep_car_local_planner/scripts/local_planner_node.py",
        ROOT / "ros/dep_car_local_planner/launch/local_planner.launch",
        ROOT / "ros/dep_car_msgs/msg/PlannerState.msg",
        ROOT / "ros/dep_car_msgs/msg/PolicyState.msg",
        ROOT / "ros/dep_car_msgs/msg/LocalRouteCommand.msg",
        ROOT / "dep_car/config/p6_memory_navigation.yaml",
        ROOT / "scripts/fetch_far_planner_upstream.sh",
        ROOT / "scripts/run_far_navigation.sh",
        ROOT / "third_party.lock.yaml",
    ]
    report = {
        "schema": "DEPCarM0M6ImplementationAuditV1",
        "status": (
            "PASS"
            if all(gates.values())
            else "BLOCKED_RUNTIME_DEPENDENCIES"
            if all(value for key, value in gates.items() if key != "runtime_dependencies")
            else "FAIL"
        ),
        "gates": gates,
        "missing_packages": missing,
        "memory_launch_nodes": nodes,
        "forbidden_nodes_present": sorted(forbidden_nodes.intersection(nodes)),
        "far_upstream": {
            "directory": str(upstream_directory),
            "locked_revision": locked_far_revision,
            "actual_revision": (
                None
                if upstream_revision is None or upstream_revision.returncode
                else upstream_revision.stdout.strip()
            ),
            "clean": bool(
                upstream_status is not None
                and upstream_status.returncode == 0
                and not upstream_status.stdout.strip()
            ),
        },
        "ground_truth_mentions_in_memory_authority": truth_mentions,
        "artifacts": {
            str(path.relative_to(ROOT)): sha256_file(path) for path in tracked
        },
        "scope": "M0-M6 dynamic visibility/memory implementation and static authority audit; Gazebo acceptance is separate",
    }
    output = ROOT / "reports/m0_m6_implementation_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report["report"] = str(output)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("audit", "list", "interactive", "fixed", "replay", "episode", "matrix"),
        required=True,
    )
    parser.add_argument(
        "--config", type=Path,
        default=ROOT / "dep_car/config/p6_memory_navigation.yaml",
    )
    parser.add_argument("--scenario")
    parser.add_argument(
        "--scenario-manifest",
        type=Path,
        help=(
            "override the config's frozen scenario manifest; relative paths "
            "are resolved from the DE-P-Car project root"
        ),
    )
    parser.add_argument("--sequence", default="logged_t_junction_turnaround")
    parser.add_argument("--cohort", choices=("development", "holdout"), default="development")
    parser.add_argument("--maximum-scenarios", type=int, default=0)
    parser.add_argument(
        "--selection-seed",
        type=int,
        default=20260820,
        help="reproducible multi-map matrix sampler; change it to draw another map order",
    )
    parser.add_argument("--goal-timeout", type=float, default=90.0)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--allow-holdout", action="store_true")
    parser.add_argument(
        "--policy-mode",
        choices=("disabled", "shadow", "guarded", "active"),
        default="shadow",
    )
    parser.add_argument("--goal-x", type=float)
    parser.add_argument("--goal-y", type=float)
    parser.add_argument("--goal-yaw", type=float)
    parser.add_argument(
        "--headless",
        action="store_true",
        help="disable gzclient and RViz for bounded CI/smoke validation",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if (args.goal_x is None) != (args.goal_y is None):
        raise ValueError("--goal-x and --goal-y must be provided together")
    config = yaml.safe_load(args.config.resolve().read_text(encoding="utf-8"))
    if args.scenario_manifest is not None:
        config["scenario_manifest"] = str(resolve(args.scenario_manifest))
    if args.stage == "audit":
        return implementation_audit(config)
    manifest, manifest_path = load_scenario_manifest(config)
    if args.stage == "list":
        launchable, excluded = [], []
        for row in manifest["scenarios"]:
            preflight = scenario_artifact_preflight(
                row, require_robust_start=True
            )
            item = {
                key: row.get(key)
                for key in (
                    "scenario_id", "cohort", "maneuver_mode", "map_name",
                    "map_seed", "map_uuid", "start", "goal",
                )
            }
            item["preflight"] = preflight["status"]
            if preflight["status"] == "PASS":
                launchable.append(item)
            else:
                item["errors"] = preflight["errors"]
                excluded.append(item)
        print(json.dumps({
            "status": "PASS",
            "scenario_manifest": str(manifest_path),
            "scenario_manifest_file_sha256": sha256_file(manifest_path),
            "launchable_scenarios": len(launchable),
            "excluded_scenarios": len(excluded),
            "scenarios": launchable,
            "excluded": excluded,
        }, indent=2, sort_keys=True))
        return 0
    if args.stage == "matrix":
        if args.cohort == "holdout" and not args.allow_holdout:
            raise RuntimeError(
                "holdout seeds are sealed; pass --allow-holdout only for final acceptance"
            )
        commands = matrix_commands(config, manifest, args)
        if args.dry_run:
            print(json.dumps({
                "schema": "DEPCarMemoryMultiSeedMatrixV2",
                "status": "DRY_RUN_PASS",
                "cohort": args.cohort,
                "scenario_count": len(commands),
                "selection_seed": int(args.selection_seed),
                "selected_scenarios": [
                    {
                        "scenario_id": scenario_id,
                        "map_seed": row.get("map_seed"),
                        "map_uuid": row.get("map_uuid"),
                        "maneuver_mode": row.get("maneuver_mode"),
                    }
                    for scenario_id, _, row in commands
                ],
                "commands": [command_item for _, command_item, _ in commands],
                "runtime_policy_has_map_specific_branches": False,
            }, indent=2, sort_keys=True))
            return 0
        results = []
        ros_port = int(config["runtime"]["ros_master_port"])
        gazebo_port = int(config["runtime"]["gazebo_master_port"])
        shutdown_timeout = float(config["runtime"]["shutdown_timeout_s"])
        for scenario_id, command_item, row in commands:
            if not wait_for_ports_free((ros_port, gazebo_port), shutdown_timeout + 5.0):
                results.append({
                    "scenario_id": scenario_id,
                    "map_seed": row.get("map_seed"),
                    "map_uuid": row.get("map_uuid"),
                    "maneuver_mode": row.get("maneuver_mode"),
                    "returncode": 3,
                    "status": "INFRASTRUCTURE_INVALID",
                    "reason": "ROS/Gazebo ports did not become free",
                })
                if args.fail_fast:
                    break
                continue
            print("+ " + " ".join(command_item), flush=True)
            completed = subprocess.run(command_item, cwd=str(ROOT))
            episode_path = (
                ROOT / "reports/memory_navigation" / ("episode_%s.json" % scenario_id)
            )
            episode = {}
            if episode_path.is_file():
                try:
                    episode = json.loads(episode_path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    episode = {}
            teardown_ready = wait_for_ports_free(
                (ros_port, gazebo_port), shutdown_timeout + 5.0
            )
            results.append({
                "scenario_id": scenario_id,
                "map_seed": row.get("map_seed"),
                "map_uuid": row.get("map_uuid"),
                "maneuver_mode": row.get("maneuver_mode"),
                "returncode": int(completed.returncode),
                "status": (
                    "PASS"
                    if completed.returncode == 0 and teardown_ready
                    else "INFRASTRUCTURE_INVALID"
                    if not teardown_ready
                    else "FAIL"
                ),
                "runtime_evidence": {
                    key: episode.get(key)
                    for key in (
                        "reason",
                        "recovery_cycles",
                        "recoveries_after_escape",
                        "far_dead_end_egress_transactions",
                        "far_dead_end_egress_active_samples",
                        "far_dead_end_egress_completion_reasons",
                        "maximum_far_dead_end_egress_target_distance_m",
                        "maximum_far_dead_end_egress_cross_track_error_m",
                        "maximum_far_dead_end_egress_map_reanchors",
                        "maximum_topology_nodes",
                        "maximum_topology_edges",
                        "maximum_failed_branches",
                        "maximum_visibility_nodes",
                        "maximum_visibility_edges",
                        "maximum_visibility_planning_time_ms",
                        "visibility_mode_counts",
                        "map_odom_correction_samples",
                        "maximum_map_odom_translation_correction_m",
                        "maximum_map_odom_yaw_correction_rad",
                        "maximum_map_odom_transform_skew_s",
                        "time_aligned_map_odom_rate",
                        "maximum_failed_boundary_sides",
                        "boundary_loop_events",
                        "accumulated_map_seen",
                        "forward_distance_m",
                        "reverse_distance_m",
                        "stationary_pose_drift_m",
                        "cumulative_stationary_map_motion_m",
                        "stationary_map_jump_events",
                        "policy_sync_success_rate",
                    )
                    if key in episode
                },
                "teardown_ready": bool(teardown_ready),
            })
            if completed.returncode and args.fail_fast:
                break
        passed = sum(row["status"] == "PASS" for row in results)
        report = {
            "schema": "DEPCarMemoryMultiSeedMatrixV2",
            "status": "PASS" if passed == len(commands) else "PARTIAL",
            "cohort": args.cohort,
            "selection_seed": int(args.selection_seed),
            "planned_scenarios": len(commands),
            "completed_scenarios": len(results),
            "passed_scenarios": passed,
            "results": results,
            "runtime_policy_has_map_specific_branches": False,
        }
        output = (
            ROOT
            / "reports/memory_navigation"
            / ("matrix_%s_seed_%d.json" % (args.cohort, int(args.selection_seed)))
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        report["report"] = str(output)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["status"] == "PASS" else 2
    if args.stage == "replay":
        sequence = config.get("regression_sequences", {}).get(args.sequence)
        if sequence is None:
            raise ValueError("unknown regression sequence: " + args.sequence)
        scenario_id = args.scenario or sequence["scenario_id"]
    else:
        sequence = None
        scenario_id = args.scenario or config[
            "interactive_default_scenario" if args.stage == "interactive" else "fixed_default_scenario"
        ]
    _, scenario = load_scenario(config, scenario_id)
    if scenario.get("cohort") == "holdout" and not args.allow_holdout:
        raise RuntimeError(
            "holdout scenario is sealed; pass --allow-holdout only for final acceptance"
        )
    missing = require_runtime_packages(args.dry_run)
    ros_port = int(config["runtime"]["ros_master_port"])
    gazebo_port = int(config["runtime"]["gazebo_master_port"])
    launch, reset, publish, goal = command(config, scenario, args, ros_port)
    automatic_report = (
        ROOT
        / "reports/memory_navigation"
        / (
            ("replay_%s.json" % args.sequence)
            if args.stage == "replay"
            else ("episode_%s.json" % scenario_id)
        )
    )
    if args.stage == "replay":
        replay_frame = str(sequence.get("coordinate_frame", "map"))
        replay_preflight = regression_goal_preflight(
            sequence,
            scenario,
            minimum_clearance_m=float(
                sequence.get("minimum_static_goal_clearance_m", 0.70)
            ),
        )
        automated = automated_goal_command(
            sequence["goals"],
            sequence.get("goal_timeout_s", args.goal_timeout),
            automatic_report,
            frame=replay_frame,
        )
    elif args.stage == "episode":
        replay_frame = "odom"
        replay_preflight = None
        automated = episode_automated_goal_command(
            goal, args.goal_timeout, automatic_report
        )
    else:
        replay_frame = None
        replay_preflight = None
        automated = None
    if args.dry_run:
        dry_run_goal_valid = bool(
            replay_preflight is None
            or replay_preflight["status"] == "PASS"
        )
        print(json.dumps({
            "status": (
                "DRY_RUN_INVALID_REPLAY_GOAL"
                if not dry_run_goal_valid
                else "DRY_RUN_PASS"
                if not missing
                else "DRY_RUN_DEPENDENCIES_MISSING"
            ),
            "missing_packages": missing,
            "navigation_backend": "online_far_visibility_memory",
            "map_server_started": False,
            "hybrid_astar_started": False,
            "scenario": scenario,
            "goal": goal,
            "fixed_recovery_probe": (
                config.get("fixed_recovery_probe")
                if args.stage == "fixed" and args.goal_x is None
                else None
            ),
            "launch": launch,
            "reset": reset,
            "publish_goal": None if args.stage == "interactive" else publish,
            "automated_goal_replay": automated,
            "automated_goal_frame": replay_frame,
            "replay_goal_preflight": replay_preflight,
            "automatic_report": str(automatic_report),
            "regression_is_runtime_policy_input": False,
        }, indent=2, sort_keys=True))
        return 0 if dry_run_goal_valid else 2
    if replay_preflight is not None and replay_preflight["status"] != "PASS":
        automatic_report.parent.mkdir(parents=True, exist_ok=True)
        automatic_report.write_text(
            json.dumps(replay_preflight, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(replay_preflight, indent=2, sort_keys=True))
        return 2
    if port_in_use(ros_port) or port_in_use(gazebo_port):
        raise RuntimeError("configured memory-navigation ROS/Gazebo port is in use")
    env = dict(os.environ)
    env["ROS_MASTER_URI"] = "http://127.0.0.1:%d" % ros_port
    env["GAZEBO_MASTER_URI"] = "http://127.0.0.1:%d" % gazebo_port
    policy_python = os.environ.get("DEP_CAR_POLICY_PYTHON") or str(
        config["runtime"].get("policy_python", "")
    )
    if not policy_python or not Path(policy_python).is_file():
        raise RuntimeError(
            "PyTorch policy interpreter is unavailable; create the yopo "
            "environment or export DEP_CAR_POLICY_PYTHON"
        )
    env["DEP_CAR_POLICY_PYTHON"] = str(Path(policy_python).resolve())
    env["PYTHONUNBUFFERED"] = "1"
    log_root = ROOT / "logs/memory_navigation" / scenario_id
    log_root.mkdir(parents=True, exist_ok=True)
    env["ROS_LOG_DIR"] = str(log_root / "ros")
    process = None
    try:
        print("+ " + " ".join(launch), flush=True)
        process = subprocess.Popen(launch, env=env, start_new_session=True)
        wait_for_model(env, config["runtime"]["startup_timeout_s"], process)
        subprocess.run(reset, env=env, check=True)
        subprocess.run(["rosservice", "call", "/gazebo/unpause_physics"], env=env, check=True)
        if args.stage == "fixed":
            time.sleep(float(config["runtime"]["fixed_goal_delay_s"]))
            subprocess.run(publish, env=env, check=True)
            print(
                "Fixed memory scenario %s is running; goal=(%.3f, %.3f), "
                "expected=%s. Ctrl+C exits."
                % (
                    scenario_id,
                    goal[0],
                    goal[1],
                    config.get("fixed_recovery_probe", {}).get(
                        "expected_terminal_state", "scenario-dependent"
                    ),
                ),
                flush=True,
            )
        elif args.stage in ("replay", "episode"):
            archived_report = archive_previous_report(automatic_report)
            if archived_report is not None:
                print(
                    "Archived previous report to %s" % archived_report,
                    flush=True,
                )
            print("+ " + " ".join(automated), flush=True)
            goals_count = len(sequence["goals"]) if sequence is not None else 1
            timeout_s = (
                float(sequence.get("goal_timeout_s", args.goal_timeout))
                if sequence is not None
                else float(args.goal_timeout)
            )
            completed = subprocess.run(
                automated,
                env=env,
                timeout=3.0 * timeout_s * goals_count + 60.0,
            )
            return int(completed.returncode)
        else:
            print(
                "Interactive memory navigation is ready; use RViz 2D Nav Goal. Ctrl+C exits.",
                flush=True,
            )
        return process.wait()
    except KeyboardInterrupt:
        return 130
    finally:
        if process is not None and process.poll() is None:
            for command_item in (
                ["rosservice", "call", "/gazebo/pause_physics"],
                ["rosservice", "call", "/gazebo/delete_model", "model_name: 'urban_model'"],
            ):
                subprocess.run(
                    command_item, env=env, stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL, timeout=5.0,
                )
        stop_process(process, float(config["runtime"]["shutdown_timeout_s"]))


if __name__ == "__main__":
    # roslaunch owns a separate process group.  Without these handlers an SSH
    # or terminal SIGHUP could kill this wrapper before its ``finally`` block,
    # orphaning Gazebo and keeping the configured ports occupied.
    signal.signal(signal.SIGHUP, request_orderly_shutdown)
    signal.signal(signal.SIGTERM, request_orderly_shutdown)
    raise SystemExit(main())
