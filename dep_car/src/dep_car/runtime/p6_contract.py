"""Identity and simulation-authority checks for the P6 ROS runtime."""

import hashlib
import json
import re
from pathlib import Path


P6_RUNTIME_SCHEMA = "DEPCarP6RuntimeImplementationV1"
P6_SHADOW_ACCEPTANCE_SCHEMA = "DEPCarP6ShadowAcceptanceV1"
P6_RUNTIME_FILES = (
    "dep_car/src/dep_car/core/gear.py",
    "dep_car/src/dep_car/core/planner.py",
    "dep_car/src/dep_car/core/recovery.py",
    "dep_car/src/dep_car/core/safety.py",
    "dep_car/src/dep_car/global_planner/hybrid_astar.py",
    "dep_car/src/dep_car/global_planner/topological_astar.py",
    "dep_car/src/dep_car/runtime/p6_contract.py",
    "dep_car/src/dep_car/runtime/online_sync.py",
    "dep_car/src/dep_car/runtime/arrival.py",
    "dep_car/src/dep_car/runtime/maneuver.py",
    "dep_car/src/dep_car/runtime/occupancy.py",
    "dep_car/src/dep_car/runtime/p6_policy.py",
    "dep_car/src/dep_car/runtime/preprocessing.py",
    "dep_car/src/dep_car/runtime/route_guidance.py",
    "dep_car/src/dep_car/runtime/safety.py",
    "dep_car/src/dep_car/runtime/start_robustness.py",
    "ros/dep_car_msgs/msg/PolicyCandidate.msg",
    "ros/dep_car_msgs/msg/PolicyCandidateArray.msg",
    "ros/dep_car_msgs/msg/PolicyQuery.msg",
    "ros/dep_car_msgs/msg/PolicyState.msg",
    "ros/dep_car_perception/scripts/lidar_preprocessor.py",
    "ros/dep_car_global_planner/scripts/hybrid_astar_node.py",
    "ros/dep_car_local_planner/scripts/policy_node.py",
    "ros/dep_car_local_planner/scripts/local_planner_node.py",
    "ros/dep_car_local_planner/launch/local_planner.launch",
    "ros/dep_car_vehicle/scripts/urban_car_adapter.py",
    "ros/dep_car_vehicle/config/urban_adapter.yaml",
    "ros/dep_car_vehicle/scripts/render_scaled_urban_urdf.py",
    "ros/dep_car_vehicle/config/urban_adapter.yaml",
    "ros/dep_car_bringup/config/urban_controllers.yaml",
    "ros/dep_car_bringup/launch/urban_sim.launch",
    "ros/dep_car_bringup/launch/p6_static.launch",
    "ros/dep_car_evaluation/scripts/run_p6_static_episode.py",
    "tools/audit_p6_shadow.py",
    "tools/prepare_p6_static_scenarios.py",
    "tools/run_p6_static.py",
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def project_root():
    return Path(__file__).resolve().parents[4]


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value):
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_p6_runtime_contract(root=None):
    root = project_root() if root is None else Path(root).resolve()
    files = {}
    for relative in P6_RUNTIME_FILES:
        path = (root / relative).resolve()
        if root not in path.parents or not path.is_file():
            raise ValueError("P6 runtime authority is missing: " + relative)
        files[relative] = sha256_file(path)
    return {
        "schema": P6_RUNTIME_SCHEMA,
        "files": files,
        "aggregate_sha256": canonical_sha256(files),
    }


def verify_p6_shadow_acceptance(
    authority_path,
    *,
    checkpoint_sha256,
    checkpoint_contract_sha256,
    modality,
    root=None,
):
    path = Path(authority_path).resolve()
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("unable to read P6 shadow authority: %s" % exc) from exc
    if document.get("schema") != P6_SHADOW_ACCEPTANCE_SCHEMA:
        raise ValueError("unknown P6 shadow authority schema")
    if document.get("status") != "PASS":
        raise ValueError("P6 shadow authority has not passed")
    expected = {
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_contract_sha256": checkpoint_contract_sha256,
        "modality": modality,
    }
    mismatches = [key for key, value in expected.items() if document.get(key) != value]
    if mismatches:
        raise ValueError("P6 shadow authority mismatch: " + ",".join(mismatches))
    runtime = build_p6_runtime_contract(root)
    if document.get("runtime_implementation") != runtime:
        raise ValueError("P6 runtime changed after shadow acceptance")
    for key in (
        "scenario_manifest_sha256",
        "report_aggregate_sha256",
        "checkpoint_sha256",
        "checkpoint_contract_sha256",
    ):
        if not isinstance(document.get(key), str) or _SHA256.fullmatch(document[key]) is None:
            raise ValueError("P6 shadow authority contains an invalid " + key)
    return document


__all__ = [
    "P6_RUNTIME_FILES",
    "P6_RUNTIME_SCHEMA",
    "P6_SHADOW_ACCEPTANCE_SCHEMA",
    "build_p6_runtime_contract",
    "canonical_sha256",
    "sha256_file",
    "verify_p6_shadow_acceptance",
]
