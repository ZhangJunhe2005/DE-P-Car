#!/usr/bin/env python3
"""Audit the V4.2 simulation-only guarded learned-control boundary."""

import json
from pathlib import Path
import subprocess
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dep_car/src"))

from dep_car.runtime.p6_contract import (
    V42_EXECUTION_ARCHITECTURE_ID,
    sha256_file,
    verify_v42_guarded_simulation_authority,
)


CONFIG = ROOT / "dep_car/config/p6_memory_navigation_v42_guarded.yaml"


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
        authority_document = verify_v42_guarded_simulation_authority(
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
    required_planner_contract = (
        'self.policy_mode == "guarded"',
        "evaluate_hybrid_sequence_candidate_bank",
        "align_policy_candidates_to_current_pose",
        "policy_anchor_age_s",
        "self.hybrid_action_latch.select",
        "locked_sequence",
        "Gear.require_drive(result.selected.gear)",
        'self.stop("hybrid_policy_static_blocked")',
        '"dep_car_net_v42_guarded"',
    )
    errors.extend(
        "planner_contract:" + token
        for token in required_planner_contract
        if token not in planner
    )
    wrapper = (ROOT / "scripts/run_p6_v42_guarded.sh").read_text(
        encoding="utf-8"
    )
    if "--policy-mode guarded" not in wrapper or "--policy-mode active" in wrapper:
        errors.append("guarded_entry_mode")
    local_launch = (
        ROOT / "ros/dep_car_local_planner/launch/local_planner.launch"
    ).read_text(encoding="utf-8")
    if (
        "0.65 if arg('policy_mode') == 'guarded' else 0.35"
        not in local_launch
    ):
        errors.append("guarded_pose_compensated_freshness")

    launch = subprocess.run(
        [
            "roslaunch",
            "--nodes",
            "dep_car_bringup",
            "p6_memory_static.launch",
            "world:=/tmp/dep_car_v42_guarded_audit.world",
            "gazebo_model_path:=/tmp",
            "checkpoint:=" + str(checkpoint),
            "checkpoint_contract:=" + str(contract),
            "p6_authority:=" + str(authority),
            "policy_mode:=guarded",
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
        "schema": "DEPCarP6V42GuardedImplementationAuditV1",
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "architecture_id": V42_EXECUTION_ARCHITECTURE_ID,
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
        "runtime_executes_learned_sequence_prefix": authority_document.get(
            "runtime_executes_learned_sequence_prefix"
        ),
        "mandatory_full_sequence_hard_veto": authority_document.get(
            "mandatory_full_sequence_hard_veto"
        ),
        "legacy_turnaround_state_machine_enabled": authority_document.get(
            "legacy_turnaround_state_machine_enabled"
        ),
        "deterministic_motion_fallback": authority_document.get(
            "deterministic_motion_fallback"
        ),
        "gazebo_simulation_only": True,
        "physical_vehicle_authorized": False,
        "production_qualified": False,
        "ros_nodes": nodes,
        "entry_points": {
            "interactive": "scripts/run_p6_v42_guarded.sh --stage interactive",
            "replay": (
                "scripts/run_p6_v42_guarded.sh --stage replay "
                "--sequence logged_t_junction_turnaround"
            ),
        },
    }
    output = ROOT / "reports/p6_v42_guarded_implementation_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report["report"] = str(output)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
