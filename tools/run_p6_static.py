#!/usr/bin/env python3
"""Host-side P6 static prepare, interactive, episode and audit entry point."""

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dep_car/src"))
from dep_car.runtime.p6_contract import build_p6_runtime_contract


PREPARE = ROOT / "tools/prepare_p6_static_scenarios.py"
AUDIT = ROOT / "tools/audit_p6_shadow.py"


def resolve(value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as connection:
        connection.settimeout(0.2)
        return connection.connect_ex(("127.0.0.1", int(port))) == 0


def run_checked(command, env=None, output=None, timeout=None):
    print("+ " + " ".join(str(item) for item in command), flush=True)
    return subprocess.run(
        [str(item) for item in command],
        env=env,
        stdout=output,
        stderr=subprocess.STDOUT if output is not None else None,
        timeout=timeout,
    ).returncode


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


def prepare_gazebo_shutdown(env, timeout=5.0):
    """Unload controllers and rendering sensors while ROS is still alive."""

    commands = (
        ("rosnode", "kill", "/urban_car_adapter"),
        ("rosnode", "kill", "/urban_model/urban_controllers"),
        ("rosservice", "call", "/gazebo/pause_physics"),
        (
            "rosservice",
            "call",
            "/gazebo/delete_model",
            "model_name: 'urban_model'",
        ),
    )
    for command in commands:
        try:
            subprocess.run(
                command,
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=timeout,
            )
        except (OSError, subprocess.TimeoutExpired):
            # Startup failures may leave only part of the Gazebo API alive.
            # The process-group fallback below must still be allowed to run.
            pass
    time.sleep(0.25)


def wait_for_model(env, timeout, launch_process=None):
    deadline = time.monotonic() + timeout
    command = [
        "rosservice", "call", "/gazebo/get_model_state", "model_name: 'urban_model'"
    ]
    while time.monotonic() < deadline:
        if launch_process is not None and launch_process.poll() is not None:
            raise RuntimeError(
                "Gazebo launch exited during startup with code %d"
                % launch_process.returncode
            )
        try:
            result = subprocess.run(
                command,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=4.0,
            )
            if result.returncode == 0 and "success: True" in result.stdout:
                return
        except subprocess.TimeoutExpired:
            pass
        time.sleep(1.0)
    raise RuntimeError("Gazebo Urban Car did not become ready")


def wait_for_ports_released(ports, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not any(port_in_use(port) for port in ports):
            return
        time.sleep(0.1)
    raise RuntimeError(
        "P6 startup retry ports were not released: "
        + ",".join(str(port) for port in ports)
    )


def load_manifest(path):
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema") != "DEPCarP6StaticScenarioManifestV1":
        raise ValueError("unknown P6 scenario manifest schema")
    return document


def select_scenarios(manifest, args, config):
    scenarios = list(manifest["scenarios"])
    if args.gate_suite:
        if args.stage not in ("shadow", "active"):
            raise ValueError("--gate-suite applies only to shadow or active")
        if args.scenario or args.cohort or args.maximum_scenarios:
            raise ValueError("--gate-suite cannot be combined with scenario/cohort limits")
        gates = config["shadow_gates" if args.stage == "shadow" else "active_gates"]
        pool = [
            row for row in scenarios
            if row["cohort"] == gates["cohort"]
            and row.get("start_robustness", {}).get("status") == "PASS"
        ]
        required_modes = list(gates.get("required_maneuvers", ()))
        selected = []
        selected_ids = set()
        for mode in required_modes:
            match = next(
                (row for row in pool if row["maneuver_mode"] == mode), None
            )
            if match is None:
                raise ValueError("gate suite has no %s scenario" % mode)
            selected.append(match)
            selected_ids.add(match["scenario_id"])
        for row in pool:
            if len(selected) >= int(gates["minimum_episodes"]):
                break
            if row["scenario_id"] not in selected_ids:
                selected.append(row)
                selected_ids.add(row["scenario_id"])
        if len(selected) < int(gates["minimum_episodes"]):
            raise ValueError("gate suite cannot meet minimum episode count")
        scenarios = selected
    if args.scenario:
        scenarios = [row for row in scenarios if row["scenario_id"] == args.scenario]
        if not scenarios:
            raise ValueError("scenario id is not present in the manifest: " + args.scenario)
        if scenarios[0].get("start_robustness", {}).get("status") != "PASS":
            raise ValueError(
                "scenario start is not perturbation-robust; inspect "
                "reports/p6_start_robustness_audit.json: " + args.scenario
            )
    elif args.cohort:
        scenarios = [row for row in scenarios if row["cohort"] == args.cohort]
    if args.stage == "interactive" and not args.scenario:
        robust = [
            row for row in scenarios
            if row.get("start_robustness", {}).get("status") == "PASS"
        ]
        normal = [row for row in robust if row["maneuver_mode"] == "NORMAL"]
        candidates = normal or robust
        if not candidates:
            raise ValueError(
                "interactive selection has no perturbation-robust start; run "
                "scripts/run_p6_static.sh --stage prepare --verify after re-audit"
            )
        # Prefer a normal, open start with the largest measured worst-case
        # footprint clearance.  Manifest ordering no longer makes a tight
        # three-point-turn scene the accidental interactive default.
        scenarios = sorted(
            candidates,
            key=lambda row: (
                -float(row["start_robustness"]["minimum_footprint_clearance_m"]),
                row["scenario_id"],
            ),
        )
    if args.maximum_scenarios:
        scenarios = scenarios[: args.maximum_scenarios]
    if not scenarios:
        raise ValueError("no P6 scenarios selected")
    if args.stage == "interactive" and len(scenarios) != 1:
        raise ValueError("interactive mode requires --scenario or --maximum-scenarios 1")
    return scenarios


def environment(config, run_root):
    ros_port = int(config["runtime"]["ros_master_port"])
    gazebo_port = int(config["runtime"]["gazebo_master_port"])
    if port_in_use(ros_port) or port_in_use(gazebo_port):
        raise RuntimeError("configured P6 ROS/Gazebo port is already in use")
    env = dict(os.environ)
    env["ROS_MASTER_URI"] = "http://127.0.0.1:%d" % ros_port
    env["GAZEBO_MASTER_URI"] = "http://127.0.0.1:%d" % gazebo_port
    env["ROS_LOG_DIR"] = str(run_root / "ros")
    env["DEP_CAR_POLICY_PYTHON"] = str(config["runtime"]["policy_python"])
    env["PYTHONUNBUFFERED"] = "1"
    return env, ros_port


def commands(config, manifest, scenario, args, output, ros_port):
    artifact = manifest["checkpoints"][args.modality]
    checkpoint = resolve(artifact["checkpoint"])
    contract = resolve(artifact["contract"])
    authority = (
        resolve(args.p6_authority)
        if args.p6_authority
        else ROOT / "reports" / ("p6_shadow_acceptance_%s.json" % args.modality)
    )
    launch = [
        "roslaunch", "-p", str(ros_port), "dep_car_bringup", "p6_static.launch",
        "world:=" + str(resolve(scenario["world"])),
        "map_yaml:=" + str(resolve(scenario["map_yaml"])),
        "gazebo_model_path:=" + str(resolve(scenario["world"]).parent.parent),
        "x:=" + str(scenario["start"][0]),
        "y:=" + str(scenario["start"][1]),
        "yaw:=" + str(scenario["start"][2]),
        "gazebo_seed:=" + str(scenario["gazebo_seed"]),
        "paused:=true",
        "checkpoint:=" + str(checkpoint),
        "checkpoint_contract:=" + str(contract),
        "policy_mode:=" + ("shadow" if args.stage == "interactive" else args.stage),
        "policy_modality:=" + args.modality,
        "fusion_sensor_mode:=" + args.fusion_sensor_mode,
        "p6_authority:=" + (str(authority) if args.stage == "active" else ""),
        "gui:=" + ("true" if args.gui or args.stage == "interactive" else "false"),
        "enable_rviz:=" + ("true" if args.rviz or args.stage == "interactive" else "false"),
        "active_fallback_to_baseline:=" + ("true" if args.active_fallback else "false"),
    ]
    reset = [
        "rosrun", "dep_car_dataset", "reset_pilot_pose.py",
        "--x", str(scenario["start"][0]),
        "--y", str(scenario["start"][1]),
        "--yaw", str(scenario["start"][2]),
    ]
    episode = [
        "rosrun", "dep_car_evaluation", "run_p6_static_episode.py",
        "--scenario-id", scenario["scenario_id"],
        "--maneuver-mode", scenario["maneuver_mode"],
        "--cohort", scenario["cohort"],
        "--policy-mode", args.stage,
        "--modality", args.modality,
        "--scenario-manifest-sha256", manifest["scenario_manifest_sha256"],
        "--runtime-implementation-sha256",
        build_p6_runtime_contract()["aggregate_sha256"],
        "--map-uuid", scenario["map_uuid"],
        "--map-seed", str(scenario["map_seed"]),
        "--map-occupancy-sha256", scenario["map_occupancy_sha256"],
        "--gazebo-seed", str(scenario["gazebo_seed"]),
        "--start-x", str(scenario["start"][0]),
        "--start-y", str(scenario["start"][1]),
        "--start-yaw", str(scenario["start"][2]),
        "--goal-x", str(scenario["goal"][0]),
        "--goal-y", str(scenario["goal"][1]),
        "--goal-yaw", str(scenario["goal"][2]),
        "--startup-timeout", str(config["runtime"]["startup_timeout_s"]),
        "--episode-timeout", str(config["runtime"]["episode_timeout_s"]),
        "--goal-tolerance", str(config["runtime"]["goal_tolerance_m"]),
        "--heading-tolerance", str(config["runtime"]["heading_tolerance_rad"]),
        "--output", str(output),
    ]
    return launch, reset, episode


def run_scenario(config, manifest, scenario, args, root):
    output = root / "run" / args.stage / args.modality / (scenario["scenario_id"] + ".json")
    if output.is_file() and not args.rerun:
        previous = json.loads(output.read_text(encoding="utf-8"))
        checkpoint = manifest["checkpoints"][args.modality]
        reusable = (
            previous.get("status") == "SUCCESS"
            and previous.get("scenario_manifest_sha256")
            == manifest["scenario_manifest_sha256"]
            and previous.get("checkpoint_sha256")
            == checkpoint["checkpoint_sha256"]
            and previous.get("observed_modality") == args.modality
            and previous.get("policy_mode") == args.stage
            and previous.get("runtime_implementation_sha256")
            == build_p6_runtime_contract()["aggregate_sha256"]
        )
        if reusable:
            print("skip completed " + scenario["scenario_id"], flush=True)
            return 0
        print(
            "stale report will be rerun " + scenario["scenario_id"], flush=True
        )
    run_root = root / "run" / args.stage / args.modality / "logs" / scenario["scenario_id"]
    run_root.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        ros_port = int(config["runtime"]["ros_master_port"])
        launch, reset, episode = commands(
            config, manifest, scenario, args, output, ros_port
        )
        print(json.dumps({
            "status": "DRY_RUN_PASS", "scenario": scenario,
            "launch": launch, "reset": reset, "episode": episode,
        }, indent=2, sort_keys=True))
        return 0
    env, ros_port = environment(config, run_root)
    launch, reset, episode = commands(config, manifest, scenario, args, output, ros_port)
    launch_log_path = run_root / "roslaunch.log"
    launch_log = None if args.stage == "interactive" else launch_log_path.open("w", encoding="utf-8")
    process = None
    try:
        startup_attempts = int(config["runtime"].get("startup_attempts", 2))
        if startup_attempts < 1:
            raise ValueError("runtime.startup_attempts must be positive")
        for attempt in range(1, startup_attempts + 1):
            print(
                "+ " + " ".join(launch) + " [startup %d/%d]"
                % (attempt, startup_attempts),
                flush=True,
            )
            process = subprocess.Popen(
                launch,
                env=env,
                stdout=launch_log,
                stderr=subprocess.STDOUT if launch_log is not None else None,
                text=True,
                start_new_session=True,
            )
            try:
                wait_for_model(
                    env,
                    float(config["runtime"]["startup_timeout_s"]),
                    launch_process=process,
                )
                reset_code = run_checked(
                    reset,
                    env=env,
                    timeout=float(config["runtime"]["startup_timeout_s"]) + 15.0,
                )
                if process.poll() is not None:
                    raise RuntimeError(
                        "Gazebo launch exited during pose reset with code %d"
                        % process.returncode
                    )
                if reset_code:
                    raise RuntimeError("post-settle pose reset failed")
                break
            except (RuntimeError, subprocess.TimeoutExpired) as exc:
                stop_process(
                    process, float(config["runtime"]["shutdown_timeout_s"])
                )
                process = None
                if attempt >= startup_attempts:
                    raise
                print(
                    "P6 Gazebo startup failed (%s); retrying after cleanup"
                    % exc,
                    file=sys.stderr,
                    flush=True,
                )
                wait_for_ports_released(
                    (
                        int(config["runtime"]["ros_master_port"]),
                        int(config["runtime"]["gazebo_master_port"]),
                    ),
                    float(config["runtime"]["shutdown_timeout_s"]),
                )
        if args.stage == "interactive":
            print(
                "P6 interactive shadow is ready; use RViz 2D Nav Goal. Ctrl+C exits.",
                flush=True,
            )
            return process.wait()
        episode_log_path = run_root / "episode.log"
        with episode_log_path.open("w", encoding="utf-8") as episode_log:
            code = run_checked(
                episode,
                env=env,
                output=episode_log,
                timeout=(
                    2.0 * float(config["runtime"]["startup_timeout_s"])
                    + float(config["runtime"]["episode_timeout_s"])
                    + 30.0
                ),
            )
        if not output.is_file():
            raise RuntimeError("P6 episode produced no report")
        report = json.loads(output.read_text(encoding="utf-8"))
        print(json.dumps({
            "scenario_id": scenario["scenario_id"],
            "status": report.get("status"),
            "report": str(output),
            "runner_exit_code": code,
        }, indent=2, sort_keys=True))
        return 0 if report.get("status") == "SUCCESS" else 2
    finally:
        if process is not None and process.poll() is None:
            prepare_gazebo_shutdown(env)
        stop_process(process, float(config["runtime"]["shutdown_timeout_s"]))
        if launch_log is not None:
            launch_log.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("prepare", "validate", "start_audit", "interactive", "shadow", "active", "audit"),
        required=True,
    )
    parser.add_argument("--config", type=Path, default=ROOT / "dep_car/config/p6_static.yaml")
    parser.add_argument("--root", type=Path, default=ROOT / "data/p6_static")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--scenario")
    parser.add_argument("--cohort", choices=("development", "holdout"))
    parser.add_argument("--maximum-scenarios", type=int, default=0)
    parser.add_argument("--modality", choices=("depth_only", "lidar_only", "fusion"), default="fusion")
    parser.add_argument("--fusion-sensor-mode", choices=("normal", "drop_depth", "drop_lidar"), default="normal")
    parser.add_argument("--p6-authority")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--rviz", action="store_true")
    parser.add_argument("--active-fallback", action="store_true")
    parser.add_argument(
        "--gate-suite",
        action="store_true",
        help="run the smallest deterministic scenario set that satisfies the stage gate",
    )
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.maximum_scenarios < 0 or args.workers < 1:
        raise ValueError("workers must be positive and maximum-scenarios non-negative")
    config_path, root = args.config.resolve(), args.root.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    python = str(config["runtime"]["policy_python"])
    if args.stage in ("prepare", "validate", "start_audit"):
        command = [python, str(PREPARE), "--config", str(config_path), "--root", str(root), "--workers", str(args.workers)]
        if args.stage == "validate":
            command.append("--verify")
        if args.stage == "start_audit":
            command.append("--reaudit-starts")
        if args.dry_run:
            command.append("--dry-run")
        raise SystemExit(run_checked(command))
    manifest_path = root / "scenario_manifest.json"
    verify = [python, str(PREPARE), "--config", str(config_path), "--root", str(root), "--verify"]
    if run_checked(verify):
        raise SystemExit(2)
    if args.stage == "audit":
        command = [
            python, str(AUDIT), "--config", str(config_path),
            "--manifest", str(manifest_path),
            "--reports", str(root / "run/shadow"),
            "--modality", args.modality,
        ]
        raise SystemExit(run_checked(command))
    manifest = load_manifest(manifest_path)
    scenarios = select_scenarios(manifest, args, config)
    failures = 0
    try:
        for scenario in scenarios:
            failures += int(run_scenario(config, manifest, scenario, args, root) != 0)
    except KeyboardInterrupt:
        print("P6 run interrupted", file=sys.stderr)
        raise SystemExit(130)
    print(json.dumps({
        "status": "PASS" if failures == 0 else "PARTIAL",
        "stage": args.stage,
        "modality": args.modality,
        "scenarios": len(scenarios),
        "failures": failures,
    }, indent=2, sort_keys=True))
    raise SystemExit(0 if failures == 0 else 2)


if __name__ == "__main__":
    main()
