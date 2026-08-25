"""Identity and simulation-authority checks for the P6 ROS runtime."""

import hashlib
import json
import re
from pathlib import Path


P6_RUNTIME_SCHEMA = "DEPCarP6RuntimeImplementationV1"
P6_SHADOW_ACCEPTANCE_SCHEMA = "DEPCarP6ShadowAcceptanceV1"
V42_EXECUTION_AUTHORITY_SCHEMA = "DEPCarV42ExecutionAuthorityV1"
V42_GUARDED_AUTHORITY_SCHEMA = "DEPCarP6V42GuardedSimulationAuthorityV1"
V42_EXECUTION_ARCHITECTURE_ID = (
    "dep_car_multimodal_v42_calibrated_hybrid_sequence_execution_15x6"
)
V43_ARCHITECTURE_ID = (
    "dep_car_multimodal_v43_guarded_contextual_residual_closed_loop_hybrid_"
    "sequence_ackermann_15x6"
)
V43_SHADOW_AUTHORITY_SCHEMA = "DEPCarP6V43ShadowAuthorityV1"
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
    "dep_car/src/dep_car/runtime/hybrid_sequence.py",
    "dep_car/src/dep_car/runtime/hybrid_execution.py",
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
    "tools/audit_p5_route_v2_score.py",
    "tools/prepare_p6_static_scenarios.py",
    "tools/run_p6_static.py",
)
V42_GUARDED_RUNTIME_FILES = (
    "dep_car/src/dep_car/model/dep_car_net_v42.py",
    "dep_car/src/dep_car/runtime/hybrid_execution.py",
    "dep_car/src/dep_car/runtime/hybrid_sequence.py",
    "dep_car/src/dep_car/runtime/far_visibility.py",
    "dep_car/src/dep_car/runtime/p6_contract.py",
    "dep_car/src/dep_car/runtime/p6_policy.py",
    "dep_car/src/dep_car/runtime/preprocessing.py",
    "dep_car/src/dep_car/runtime/safety.py",
    "ros/dep_car_msgs/msg/PolicyCandidate.msg",
    "ros/dep_car_msgs/msg/PolicyCandidateArray.msg",
    "ros/dep_car_msgs/msg/PolicyState.msg",
    "ros/dep_car_local_planner/scripts/policy_node.py",
    "ros/dep_car_local_planner/scripts/local_planner_node.py",
    "ros/dep_car_local_planner/launch/local_planner.launch",
    "ros/dep_car_memory_navigation/config/navigation_memory.yaml",
    "ros/dep_car_memory_navigation/scripts/navigation_memory_node.py",
    "ros/dep_car_bringup/launch/urban_sim.launch",
    "ros/dep_car_bringup/launch/p6_memory_static.launch",
    "tools/run_memory_navigation.py",
    "scripts/run_p6_v42_guarded.sh",
    "dep_car/config/p6_memory_navigation_v42_guarded.yaml",
)
V43_SHADOW_RUNTIME_FILES = (
    "dep_car/src/dep_car/model/dep_car_net.py",
    "dep_car/src/dep_car/model/dep_car_net_v2.py",
    "dep_car/src/dep_car/model/dep_car_net_v3.py",
    "dep_car/src/dep_car/model/dep_car_net_v4.py",
    "dep_car/src/dep_car/model/dep_car_net_v43.py",
    "dep_car/src/dep_car/model/hybrid_sequence_rollout.py",
    "dep_car/src/dep_car/runtime/hybrid_execution.py",
    "dep_car/src/dep_car/runtime/hybrid_sequence.py",
    "dep_car/src/dep_car/runtime/far_visibility.py",
    "dep_car/src/dep_car/runtime/p6_contract.py",
    "dep_car/src/dep_car/runtime/p6_policy.py",
    "dep_car/src/dep_car/runtime/preprocessing.py",
    "dep_car/src/dep_car/runtime/safety.py",
    "ros/dep_car_msgs/msg/PolicyCandidate.msg",
    "ros/dep_car_msgs/msg/PolicyCandidateArray.msg",
    "ros/dep_car_msgs/msg/PolicyQuery.msg",
    "ros/dep_car_msgs/msg/PolicyState.msg",
    "ros/dep_car_local_planner/scripts/policy_node.py",
    "ros/dep_car_local_planner/scripts/local_planner_node.py",
    "ros/dep_car_local_planner/launch/local_planner.launch",
    "ros/dep_car_memory_navigation/config/navigation_memory.yaml",
    "ros/dep_car_memory_navigation/scripts/navigation_memory_node.py",
    "ros/dep_car_memory_navigation/scripts/replay_memory_goals.py",
    "ros/dep_car_bringup/launch/p6_memory_static.launch",
    "tools/run_memory_navigation.py",
    "scripts/run_memory_navigation.sh",
    "scripts/run_p6_v43_shadow.sh",
    "dep_car/config/p6_memory_navigation_v43_shadow.yaml",
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


def resolve_project_artifact(reference, root=None):
    """Resolve signed artifact paths after moving or cloning the repository.

    Early training reports recorded an absolute development-workstation path.
    The file hashes remain useful and immutable, so a new checkout relocates
    only recognised repository-relative tails instead of rewriting evidence.
    Arbitrary absolute paths are never redirected.
    """

    root = project_root() if root is None else Path(root).resolve()
    reference = Path(str(reference))
    if not reference.is_absolute():
        return (root / reference).resolve()
    anchors = (
        "models", "reports", "data", "dep_car", "ros", "scripts", "tools"
    )
    parts = reference.parts
    for anchor in anchors:
        if anchor not in parts:
            continue
        candidate = (root / Path(*parts[parts.index(anchor):])).resolve()
        if candidate == root or root in candidate.parents:
            return candidate
    return reference.resolve()


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


def build_v42_guarded_runtime_contract(root=None):
    """Hash the simulation-only learned-control boundary as one unit."""

    root = project_root() if root is None else Path(root).resolve()
    files = {}
    for relative in V42_GUARDED_RUNTIME_FILES:
        path = (root / relative).resolve()
        if root not in path.parents or not path.is_file():
            raise ValueError("V4.2 guarded runtime is missing: " + relative)
        files[relative] = sha256_file(path)
    return {
        "schema": "DEPCarP6V42GuardedRuntimeV1",
        "files": files,
        "aggregate_sha256": canonical_sha256(files),
    }


def build_v43_shadow_runtime_contract(root=None):
    """Hash every model/ROS file allowed to handle a V4.3 shadow bank."""

    root = project_root() if root is None else Path(root).resolve()
    files = {}
    for relative in V43_SHADOW_RUNTIME_FILES:
        path = (root / relative).resolve()
        if root not in path.parents or not path.is_file():
            raise ValueError("V4.3 shadow runtime is missing: " + relative)
        files[relative] = sha256_file(path)
    return {
        "schema": "DEPCarP6V43ShadowRuntimeV1",
        "files": files,
        "aggregate_sha256": canonical_sha256(files),
    }


def verify_v43_training_acceptance(
    checkpoint_path, checkpoint_contract_path, acceptance_path, root=None
):
    """Verify the formal P5 V4.3 PASS without granting control authority."""

    root = project_root() if root is None else Path(root).resolve()
    checkpoint_path = Path(checkpoint_path).resolve()
    checkpoint_contract_path = Path(checkpoint_contract_path).resolve()
    acceptance_path = Path(acceptance_path).resolve()
    try:
        contract = json.loads(
            checkpoint_contract_path.read_text(encoding="utf-8")
        )
        acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("unable to read V4.3 training evidence: %s" % exc) from exc

    checkpoint_sha = sha256_file(checkpoint_path)
    contract_sha = sha256_file(checkpoint_contract_path)
    fixed_contract = {
        "schema": "DEPCarV43ArtifactContractV5",
        "architecture_id": V43_ARCHITECTURE_ID,
        "objective_id": (
            "dep_car_objective_v19_guarded_contextual_exact_closed_loop_selector"
        ),
        "training_stage": "dagger_guarded_closed_loop_sequence_selector",
        "artifact_role": "best",
        "status": "TRAINED_UNQUALIFIED",
        "qualification_status": "UNQUALIFIED",
        "run_completed": True,
        "partial_epoch": False,
        "active_control_authorized": False,
        "production_qualified": False,
        "continuous_sequence_authority": (
            "REOBSERVED_STATE_EXACT_SIGNED_HYBRID_ASTAR_PLAN"
        ),
        "high_level_gear_state_machine": False,
        "checkpoint_sha256": checkpoint_sha,
    }
    errors = [
        "contract." + key
        for key, value in fixed_contract.items()
        if contract.get(key) != value
    ]
    if int(contract.get("completed_epochs", 0)) != 24:
        errors.append("contract.completed_epochs")
    ownership = contract.get("stage_ownership", {})
    if (
        ownership.get("high_level_gear_state_machine") is not False
        or ownership.get("candidate_geometry_trainable") is not False
        or ownership.get("gear_sequence_trainable") is not False
        or ownership.get("closed_loop_residual_score_trainable") is not True
    ):
        errors.append("contract.stage_ownership")

    fixed_acceptance = {
        "schema": "DEPCarV43AcceptanceV1",
        "status": "PASS",
        "gate_passed": True,
        "errors": [],
        "scope": "P6_SHADOW_ONLY",
        "formal_population": True,
        "population_samples": 1997,
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_contract_sha256": contract_sha,
        "continuous_sequence_authority": (
            "REOBSERVED_STATE_EXACT_SIGNED_HYBRID_ASTAR_PLAN"
        ),
        "high_level_gear_state_machine": False,
        "closed_loop_gazebo_qualification_pending": True,
        "active_control_authorized": False,
        "production_qualified": False,
        "test_split_accessed": False,
    }
    errors.extend(
        "acceptance." + key
        for key, value in fixed_acceptance.items()
        if acceptance.get(key) != value
    )
    if (
        resolve_project_artifact(acceptance.get("checkpoint", ""), root)
        != checkpoint_path
        or resolve_project_artifact(
            acceptance.get("checkpoint_contract", ""), root
        )
        != checkpoint_contract_path
    ):
        errors.append("acceptance.paths")
    if (
        not isinstance(acceptance.get("checks"), dict)
        or not acceptance["checks"]
        or any(value != "PASS" for value in acceptance["checks"].values())
    ):
        errors.append("acceptance.checks")
    if (
        acceptance.get("data_authority_gate", {}).get("passed") is not True
        or acceptance.get("data_authority_gate", {}).get("test_split_sealed")
        is not True
        or acceptance.get("source_gate", {}).get("passed") is not True
        or acceptance.get("source_gate", {}).get("test_split_accessed")
        is not False
    ):
        errors.append("acceptance.lineage")
    implementation = contract.get("implementation_sha256", {})
    implementation_paths = {
        "model": "dep_car/src/dep_car/model/dep_car_net_v43.py",
        "candidate_model": "dep_car/src/dep_car/model/dep_car_net_v4.py",
        "rollout": "dep_car/src/dep_car/model/hybrid_sequence_rollout.py",
    }
    for name, relative in implementation_paths.items():
        if implementation.get(name) != sha256_file(root / relative):
            errors.append("contract.implementation." + name)
    if errors:
        raise ValueError(
            "V4.3 training acceptance mismatch: "
            + ",".join(sorted(set(errors)))
        )
    return acceptance


def verify_v43_shadow_authority(
    authority_path,
    *,
    checkpoint_path,
    checkpoint_contract_path,
    root=None,
):
    """Verify V4.3 ROS shadow authority; never authorize learned actuation."""

    root = project_root() if root is None else Path(root).resolve()
    authority_path = Path(authority_path).resolve()
    checkpoint_path = Path(checkpoint_path).resolve()
    checkpoint_contract_path = Path(checkpoint_contract_path).resolve()
    try:
        document = json.loads(authority_path.read_text(encoding="utf-8"))
        acceptance_path = resolve_project_artifact(
            document["acceptance_report"], root
        )
    except (OSError, KeyError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("unable to read V4.3 shadow authority: %s" % exc) from exc
    acceptance = verify_v43_training_acceptance(
        checkpoint_path, checkpoint_contract_path, acceptance_path, root=root
    )
    fixed = {
        "schema": V43_SHADOW_AUTHORITY_SCHEMA,
        "status": "P6_SHADOW_AUTHORIZED",
        "architecture_id": V43_ARCHITECTURE_ID,
        "mandatory_full_sequence_hard_veto": True,
        "model_selects_first_gear": True,
        "model_selects_complete_gear_sequence": True,
        "runtime_executes_learned_sequence_prefix": False,
        "legacy_turnaround_state_machine_enabled": False,
        "deterministic_shadow_control": True,
        "model_control_authorized": False,
        "active_control_authorized": False,
        "physical_vehicle_authorized": False,
        "production_qualified": False,
        "test_split_accessed": False,
    }
    errors = [key for key, value in fixed.items() if document.get(key) != value]
    if (
        resolve_project_artifact(document.get("checkpoint", ""), root)
        != checkpoint_path
        or document.get("checkpoint_sha256") != sha256_file(checkpoint_path)
        or resolve_project_artifact(
            document.get("checkpoint_contract", ""), root
        )
        != checkpoint_contract_path
        or document.get("checkpoint_contract_sha256")
        != sha256_file(checkpoint_contract_path)
    ):
        errors.append("checkpoint_identity")
    if (
        document.get("acceptance_report_sha256") != sha256_file(acceptance_path)
        or acceptance.get("status") != "PASS"
    ):
        errors.append("acceptance_report")
    runtime = build_v43_shadow_runtime_contract(root)
    if document.get("runtime_implementation") != runtime:
        errors.append("runtime_implementation")
    if errors:
        raise ValueError(
            "V4.3 shadow authority mismatch: "
            + ",".join(sorted(set(errors)))
        )
    return document


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


def verify_v42_execution_authority(
    authority_path,
    *,
    checkpoint_path,
    checkpoint_contract_path,
    root=None,
):
    """Verify the parameter-free V4.2 P6-shadow execution authority.

    V4.2 deliberately reuses the V4.1 state dict.  Its separate authority
    binds that checkpoint to the calibrated score adapter and to a full
    validation PASS.  It never grants active-control or production authority.
    """

    root = project_root() if root is None else Path(root).resolve()
    authority_path = Path(authority_path).resolve()
    checkpoint_path = Path(checkpoint_path).resolve()
    checkpoint_contract_path = Path(checkpoint_contract_path).resolve()
    try:
        document = json.loads(authority_path.read_text(encoding="utf-8"))
        contract = json.loads(
            checkpoint_contract_path.read_text(encoding="utf-8")
        )
        report_path = Path(document["acceptance_report"]).resolve()
        adapter_path = Path(document["adapter"]).resolve()
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, KeyError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("unable to read V4.2 execution authority: %s" % exc) from exc

    expected_adapter = (
        root / "dep_car/src/dep_car/model/dep_car_net_v42.py"
    ).resolve()
    errors = []
    fixed = {
        "schema": V42_EXECUTION_AUTHORITY_SCHEMA,
        "status": "P6_SHADOW_AUTHORIZED",
        "architecture_id": V42_EXECUTION_ARCHITECTURE_ID,
        "mandatory_hard_veto": True,
        "p6_shadow_authorized": True,
        "active_control_authorized": False,
        "production_qualified": False,
        "test_split_accessed": False,
        "safety_risk_weight": 0.0,
        "viability_risk_weight": 8.0,
    }
    errors.extend(
        key for key, value in fixed.items() if document.get(key) != value
    )
    if (
        Path(document.get("source_checkpoint", "")).resolve()
        != checkpoint_path
        or document.get("source_checkpoint_sha256")
        != sha256_file(checkpoint_path)
    ):
        errors.append("source_checkpoint")
    if (
        adapter_path != expected_adapter
        or document.get("adapter_sha256") != sha256_file(adapter_path)
    ):
        errors.append("adapter")
    if document.get("acceptance_report_sha256") != sha256_file(report_path):
        errors.append("acceptance_report_sha256")
    report_fixed = {
        "schema": "DEPCarV42ExecutionAcceptanceV1",
        "status": "PASS",
        "gate_passed": True,
        "architecture_id": V42_EXECUTION_ARCHITECTURE_ID,
        "mandatory_hard_veto": True,
        "p6_shadow_authorized": True,
        "active_control_authorized": False,
        "production_qualified": False,
        "test_split_accessed": False,
        "training_required": False,
        "safety_risk_weight": 0.0,
        "viability_risk_weight": 8.0,
        "source_checkpoint_sha256": sha256_file(checkpoint_path),
    }
    errors.extend(
        "report." + key
        for key, value in report_fixed.items()
        if report.get(key) != value
    )
    if (
        report.get("errors") != []
        or not isinstance(report.get("checks"), dict)
        or not report["checks"]
        or any(value != "PASS" for value in report["checks"].values())
    ):
        errors.append("report.checks")
    contract_fixed = {
        "schema": "DEPCarV41ScoreArtifactContractV1",
        "architecture_id": (
            "dep_car_multimodal_v4_hybrid_sequence_route_ackermann_15x6"
        ),
        "artifact_role": "best",
        "status": "TRAINED_UNQUALIFIED",
        "qualification_status": "UNQUALIFIED",
        "production_qualified": False,
        "run_completed": True,
        "partial_epoch": False,
        "unified_hybrid_sequence": True,
        "high_level_gear_state_machine": False,
    }
    errors.extend(
        "contract." + key
        for key, value in contract_fixed.items()
        if contract.get(key) != value
    )
    if contract.get("checkpoint_sha256") != sha256_file(checkpoint_path):
        errors.append("contract.checkpoint_sha256")
    if errors:
        raise ValueError(
            "V4.2 execution authority mismatch: "
            + ",".join(sorted(set(errors)))
        )
    return document


def verify_v42_guarded_simulation_authority(
    authority_path,
    *,
    checkpoint_path,
    checkpoint_contract_path,
    root=None,
):
    """Verify experimental V4.2 model control for Gazebo development only."""

    root = project_root() if root is None else Path(root).resolve()
    authority_path = Path(authority_path).resolve()
    checkpoint_path = Path(checkpoint_path).resolve()
    checkpoint_contract_path = Path(checkpoint_contract_path).resolve()
    try:
        document = json.loads(authority_path.read_text(encoding="utf-8"))
        base_path = Path(document["shadow_execution_authority"]).resolve()
    except (OSError, KeyError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            "unable to read V4.2 guarded simulation authority: %s" % exc
        ) from exc
    base = verify_v42_execution_authority(
        base_path,
        checkpoint_path=checkpoint_path,
        checkpoint_contract_path=checkpoint_contract_path,
        root=root,
    )
    fixed = {
        "schema": V42_GUARDED_AUTHORITY_SCHEMA,
        "status": "P6_GUARDED_SIMULATION_AUTHORIZED",
        "architecture_id": V42_EXECUTION_ARCHITECTURE_ID,
        "mandatory_full_sequence_hard_veto": True,
        "model_selects_first_gear": True,
        "model_selects_complete_gear_sequence": True,
        "runtime_executes_learned_sequence_prefix": True,
        "legacy_turnaround_state_machine_enabled": False,
        "deterministic_motion_fallback": False,
        "gazebo_simulation_only": True,
        "active_control_authorized": True,
        "physical_vehicle_authorized": False,
        "production_qualified": False,
        "test_split_accessed": False,
    }
    errors = [key for key, value in fixed.items() if document.get(key) != value]
    if (
        document.get("source_checkpoint_sha256")
        != sha256_file(checkpoint_path)
        or document.get("checkpoint_contract_sha256")
        != sha256_file(checkpoint_contract_path)
    ):
        errors.append("checkpoint_identity")
    if (
        document.get("shadow_execution_authority_sha256")
        != sha256_file(base_path)
        or base.get("status") != "P6_SHADOW_AUTHORIZED"
    ):
        errors.append("shadow_execution_authority")
    runtime = build_v42_guarded_runtime_contract(root)
    if document.get("runtime_implementation") != runtime:
        errors.append("runtime_implementation")
    if errors:
        raise ValueError(
            "V4.2 guarded simulation authority mismatch: "
            + ",".join(sorted(set(errors)))
        )
    return document


__all__ = [
    "P6_RUNTIME_FILES",
    "P6_RUNTIME_SCHEMA",
    "P6_SHADOW_ACCEPTANCE_SCHEMA",
    "V42_EXECUTION_ARCHITECTURE_ID",
    "V42_EXECUTION_AUTHORITY_SCHEMA",
    "V42_GUARDED_AUTHORITY_SCHEMA",
    "V42_GUARDED_RUNTIME_FILES",
    "V43_ARCHITECTURE_ID",
    "V43_SHADOW_AUTHORITY_SCHEMA",
    "V43_SHADOW_RUNTIME_FILES",
    "build_p6_runtime_contract",
    "build_v42_guarded_runtime_contract",
    "build_v43_shadow_runtime_contract",
    "canonical_sha256",
    "sha256_file",
    "verify_p6_shadow_acceptance",
    "verify_v42_execution_authority",
    "verify_v42_guarded_simulation_authority",
    "verify_v43_shadow_authority",
    "verify_v43_training_acceptance",
]
