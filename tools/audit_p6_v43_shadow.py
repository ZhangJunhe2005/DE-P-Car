#!/usr/bin/env python3
"""Audit the V4.3 checkpoint-to-ROS P6 shadow boundary."""

import json
from pathlib import Path
import subprocess
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dep_car/src"))
sys.path.insert(0, str(ROOT / "tools"))

from dep_car.runtime.p6_contract import (
    V43_ARCHITECTURE_ID,
    sha256_file,
    verify_v43_shadow_authority,
)
from run_memory_navigation import load_scenario, regression_goal_preflight


CONFIG = ROOT / "dep_car/config/p6_memory_navigation_v43_shadow.yaml"


def resolve(path):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def main():
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    checkpoint = resolve(config["checkpoint"])
    contract = resolve(config["checkpoint_contract"])
    authority = resolve(config["p6_authority"])
    errors = []
    try:
        authority_document = verify_v43_shadow_authority(
            authority,
            checkpoint_path=checkpoint,
            checkpoint_contract_path=contract,
        )
    except Exception as exc:
        authority_document = {}
        errors.append("authority:" + type(exc).__name__ + ":" + str(exc))

    planner = (
        ROOT / "ros/dep_car_local_planner/scripts/local_planner_node.py"
    ).read_text(encoding="utf-8")
    policy = (ROOT / "dep_car/src/dep_car/runtime/p6_policy.py").read_text(
        encoding="utf-8"
    )
    memory = (
        ROOT
        / "ros/dep_car_memory_navigation/scripts/navigation_memory_node.py"
    ).read_text(encoding="utf-8")
    far_visibility = (
        ROOT / "dep_car/src/dep_car/runtime/far_visibility.py"
    ).read_text(encoding="utf-8")
    runner = (ROOT / "tools/run_memory_navigation.py").read_text(
        encoding="utf-8"
    )
    replay = (
        ROOT
        / "ros/dep_car_memory_navigation/scripts/replay_memory_goals.py"
    ).read_text(encoding="utf-8")
    required = {
        "planner": (
            "V43_ARCHITECTURE_ID",
            "evaluate_hybrid_sequence_candidate_bank",
            '"policy_v43_hybrid"',
            'self.policy_mode in ("disabled", "shadow")',
        ),
        "policy": (
            "DEPCarNetV43",
            "verify_v43_shadow_authority",
            'self.mode != "shadow"',
            "V4.3 is authorized for P6 shadow only",
        ),
        "memory": (
            "locally_certified_route_motion",
            "revalidation_prefix_clear",
            "bootstrap_motion_authorized",
            "live_dead_end_egress_reanchor",
            "FAILED_BRANCH_EXIT_LOCK",
            "EGRESS_REANCHOR_EXHAUSTED_CURRENT_MAP",
        ),
        "far_visibility": (
            "stable_attemptable_navigation_high_detour",
            "def locally_certified_route_motion",
            "observed_prefix_clear",
        ),
        "runner": (
            "regression_goal_preflight",
            "coordinate_frame",
            "INVALID_REPLAY_GOAL",
        ),
        "replay": (
            "goal_in_map",
            "INVALID_REPLAY_GOAL",
            "goal_preflight_inflation_radius_m",
        ),
    }
    for owner, tokens in required.items():
        source = {
            "planner": planner,
            "policy": policy,
            "memory": memory,
            "far_visibility": far_visibility,
            "runner": runner,
            "replay": replay,
        }[owner]
        errors.extend(
            owner + ":" + token for token in tokens if token not in source
        )
    wrapper = (ROOT / "scripts/run_p6_v43_shadow.sh").read_text(
        encoding="utf-8"
    )
    if "--policy-mode shadow" not in wrapper or "--policy-mode guarded" in wrapper:
        errors.append("shadow_entry_mode")

    regression_preflights = {}
    try:
        _, scenario = load_scenario(
            config, config["interactive_default_scenario"]
        )
        for name, sequence in sorted(config["regression_sequences"].items()):
            preflight = regression_goal_preflight(
                sequence,
                scenario,
                minimum_clearance_m=float(
                    sequence.get("minimum_static_goal_clearance_m", 0.70)
                ),
            )
            regression_preflights[name] = preflight
            if preflight.get("status") != "PASS":
                errors.append("regression_goal_preflight:" + name)
    except Exception as exc:
        regression_preflights = {
            "audit_error": type(exc).__name__ + ":" + str(exc)
        }
        errors.append("regression_goal_preflight")

    required_fields = {
        "ros/dep_car_msgs/msg/PolicyCandidate.msg": (
            "int8[] action_gears",
            "bool[] action_mask",
            "float64[] action_durations",
            "int8[] motion_gears",
        ),
        "ros/dep_car_msgs/msg/PolicyCandidateArray.msg": (
            "string architecture_id",
            "bool hybrid_sequence",
            "float64[] gear_history",
        ),
        "ros/dep_car_msgs/msg/PolicyState.msg": (
            "string architecture_id",
            "int8[] selected_action_gears",
        ),
    }
    for relative, fields in required_fields.items():
        source = (ROOT / relative).read_text(encoding="utf-8")
        errors.extend(
            relative + ":" + field for field in fields if field not in source
        )
    generated = (
        ROOT
        / "catkin_ws/devel/lib/python3/dist-packages/dep_car_msgs/msg/"
        "_PolicyCandidate.py"
    )
    if not generated.is_file() or "action_gears" not in generated.read_text(
        encoding="utf-8"
    ):
        errors.append("generated_ros_messages")

    launch = subprocess.run(
        [
            "roslaunch", "--nodes", "dep_car_bringup", "p6_memory_static.launch",
            "world:=/tmp/dep_car_v43_shadow_audit.world",
            "gazebo_model_path:=/tmp",
            "checkpoint:=" + str(checkpoint),
            "checkpoint_contract:=" + str(contract),
            "p6_authority:=" + str(authority),
            "policy_mode:=shadow",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    nodes = sorted(
        line.strip() for line in launch.stdout.splitlines() if line.startswith("/")
    )
    required_nodes = {
        "/dep_car_policy",
        "/dep_car_local_planner",
        "/dep_car_memory_navigation",
        "/gazebo",
        "/slam_toolbox",
    }
    if launch.returncode or not required_nodes.issubset(nodes):
        errors.append("roslaunch_nodes")

    report = {
        "schema": "DEPCarP6V43ShadowImplementationAuditV1",
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "architecture_id": V43_ARCHITECTURE_ID,
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_contract_sha256": sha256_file(contract),
        "authority_sha256": sha256_file(authority) if authority.is_file() else "",
        "authority_status": authority_document.get("status"),
        "model_selects_first_gear": authority_document.get(
            "model_selects_first_gear"
        ),
        "model_selects_complete_gear_sequence": authority_document.get(
            "model_selects_complete_gear_sequence"
        ),
        "mandatory_full_sequence_hard_veto": authority_document.get(
            "mandatory_full_sequence_hard_veto"
        ),
        "deterministic_shadow_control": authority_document.get(
            "deterministic_shadow_control"
        ),
        "model_control_authorized": False,
        "legacy_turnaround_state_machine_enabled": False,
        "active_control_authorized": False,
        "physical_vehicle_authorized": False,
        "production_qualified": False,
        "regression_goal_preflight": regression_preflights,
        "ros_nodes": nodes,
        "entry_points": {
            "interactive": "scripts/run_p6_v43_shadow.sh --stage interactive",
            "replay": (
                "scripts/run_p6_v43_shadow.sh --stage replay "
                "--sequence logged_t_junction_turnaround"
            ),
        },
    }
    output = ROOT / "reports/p6_v43_shadow_implementation_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report["report"] = str(output)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
