#!/usr/bin/env python3
"""Static authority and ROS-interface audit for V4.2 P6 shadow mode."""

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
    verify_v42_execution_authority,
)


CONFIG = ROOT / "dep_car/config/p6_memory_navigation_v42_shadow.yaml"


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
        authority_document = verify_v42_execution_authority(
            authority,
            checkpoint_path=checkpoint,
            checkpoint_contract_path=contract,
        )
    except Exception as exc:
        authority_document = {}
        errors.append("authority:" + type(exc).__name__ + ":" + str(exc))

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
            "%s:%s" % (relative, field)
            for field in fields
            if field not in source
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
            "roslaunch",
            "--nodes",
            "dep_car_bringup",
            "p6_memory_static.launch",
            "world:=/tmp/dep_car_v42_audit.world",
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
        "/slam_toolbox",
    }
    if launch.returncode or not required_nodes.issubset(nodes):
        errors.append("roslaunch_nodes")

    tracked = (
        ROOT / "dep_car/src/dep_car/runtime/p6_contract.py",
        ROOT / "dep_car/src/dep_car/runtime/hybrid_sequence.py",
        ROOT / "dep_car/src/dep_car/runtime/p6_policy.py",
        ROOT / "dep_car/src/dep_car/runtime/preprocessing.py",
        ROOT / "dep_car/src/dep_car/runtime/safety.py",
        ROOT / "ros/dep_car_msgs/msg/PolicyCandidate.msg",
        ROOT / "ros/dep_car_msgs/msg/PolicyCandidateArray.msg",
        ROOT / "ros/dep_car_msgs/msg/PolicyState.msg",
        ROOT / "ros/dep_car_local_planner/scripts/policy_node.py",
        ROOT / "ros/dep_car_local_planner/scripts/local_planner_node.py",
        ROOT / "ros/dep_car_memory_navigation/scripts/replay_memory_goals.py",
        ROOT / "scripts/run_p6_v42_shadow.sh",
        CONFIG,
    )
    report = {
        "schema": "DEPCarP6V42ShadowImplementationAuditV1",
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "architecture_id": V42_EXECUTION_ARCHITECTURE_ID,
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_contract_sha256": sha256_file(contract),
        "execution_authority_sha256": sha256_file(authority),
        "authority_status": authority_document.get("status"),
        "mandatory_hard_veto": authority_document.get("mandatory_hard_veto"),
        "active_control_authorized": False,
        "production_qualified": False,
        "ros_nodes": nodes,
        "entry_points": {
            "interactive": "scripts/run_p6_v42_shadow.sh --stage interactive",
            "replay": "scripts/run_p6_v42_shadow.sh --stage replay",
        },
        "artifacts": {
            str(path.relative_to(ROOT)): sha256_file(path) for path in tracked
        },
    }
    output = ROOT / "reports/p6_v42_shadow_implementation_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report["report"] = str(output)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
