import json
from pathlib import Path

import pytest

from dep_car.runtime.p6_contract import (
    V43_ARCHITECTURE_ID,
    V43_SHADOW_AUTHORITY_SCHEMA,
    build_v43_shadow_runtime_contract,
    project_root,
    resolve_project_artifact,
    sha256_file,
    verify_v43_shadow_authority,
    verify_v43_training_acceptance,
)


def test_signed_project_artifact_paths_relocate_without_rewriting_evidence(
    tmp_path,
):
    relocated = tmp_path / "clone"
    expected = relocated / "models/dep_car/p5_closed_loop_v43/model.pth"
    assert resolve_project_artifact(
        "/home/another-user/DE-P-Car/models/dep_car/"
        "p5_closed_loop_v43/model.pth",
        relocated,
    ) == expected.resolve()
    external = Path("/opt/external/model.pth")
    assert resolve_project_artifact(external, relocated) == external


def write_v43_evidence(tmp_path):
    root = project_root()
    checkpoint = tmp_path / "v43.best.pth"
    checkpoint.write_bytes(b"v43-shadow-unit-checkpoint")
    contract = checkpoint.with_suffix(".contract.json")
    contract.write_text(
        json.dumps(
            {
                "schema": "DEPCarV43ArtifactContractV5",
                "architecture_id": V43_ARCHITECTURE_ID,
                "objective_id": (
                    "dep_car_objective_v19_guarded_contextual_exact_"
                    "closed_loop_selector"
                ),
                "training_stage": "dagger_guarded_closed_loop_sequence_selector",
                "artifact_role": "best",
                "status": "TRAINED_UNQUALIFIED",
                "qualification_status": "UNQUALIFIED",
                "run_completed": True,
                "partial_epoch": False,
                "completed_epochs": 24,
                "active_control_authorized": False,
                "production_qualified": False,
                "continuous_sequence_authority": (
                    "REOBSERVED_STATE_EXACT_SIGNED_HYBRID_ASTAR_PLAN"
                ),
                "high_level_gear_state_machine": False,
                "checkpoint_sha256": sha256_file(checkpoint),
                "stage_ownership": {
                    "high_level_gear_state_machine": False,
                    "candidate_geometry_trainable": False,
                    "gear_sequence_trainable": False,
                    "closed_loop_residual_score_trainable": True,
                },
                "implementation_sha256": {
                    "model": sha256_file(
                        root / "dep_car/src/dep_car/model/dep_car_net_v43.py"
                    ),
                    "candidate_model": sha256_file(
                        root / "dep_car/src/dep_car/model/dep_car_net_v4.py"
                    ),
                    "rollout": sha256_file(
                        root
                        / "dep_car/src/dep_car/model/hybrid_sequence_rollout.py"
                    ),
                },
            }
        ),
        encoding="utf-8",
    )
    acceptance = checkpoint.with_suffix(".acceptance.json")
    acceptance.write_text(
        json.dumps(
            {
                "schema": "DEPCarV43AcceptanceV1",
                "status": "PASS",
                "gate_passed": True,
                "errors": [],
                "scope": "P6_SHADOW_ONLY",
                "formal_population": True,
                "population_samples": 1997,
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": sha256_file(checkpoint),
                "checkpoint_contract": str(contract.resolve()),
                "checkpoint_contract_sha256": sha256_file(contract),
                "continuous_sequence_authority": (
                    "REOBSERVED_STATE_EXACT_SIGNED_HYBRID_ASTAR_PLAN"
                ),
                "high_level_gear_state_machine": False,
                "closed_loop_gazebo_qualification_pending": True,
                "active_control_authorized": False,
                "production_qualified": False,
                "test_split_accessed": False,
                "checks": {"exact_sequence": "PASS", "hard_veto": "PASS"},
                "data_authority_gate": {
                    "passed": True,
                    "test_split_sealed": True,
                },
                "source_gate": {"passed": True, "test_split_accessed": False},
            }
        ),
        encoding="utf-8",
    )
    return checkpoint, contract, acceptance


def test_v43_acceptance_and_shadow_authority_never_grant_control(tmp_path):
    checkpoint, contract, acceptance = write_v43_evidence(tmp_path)
    verified = verify_v43_training_acceptance(
        checkpoint, contract, acceptance
    )
    assert verified["scope"] == "P6_SHADOW_ONLY"

    authority = tmp_path / "shadow.authority.json"
    authority.write_text(
        json.dumps(
            {
                "schema": V43_SHADOW_AUTHORITY_SCHEMA,
                "status": "P6_SHADOW_AUTHORIZED",
                "architecture_id": V43_ARCHITECTURE_ID,
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": sha256_file(checkpoint),
                "checkpoint_contract": str(contract.resolve()),
                "checkpoint_contract_sha256": sha256_file(contract),
                "acceptance_report": str(acceptance.resolve()),
                "acceptance_report_sha256": sha256_file(acceptance),
                "runtime_implementation": build_v43_shadow_runtime_contract(),
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
        ),
        encoding="utf-8",
    )
    document = verify_v43_shadow_authority(
        authority,
        checkpoint_path=checkpoint,
        checkpoint_contract_path=contract,
    )
    assert document["deterministic_shadow_control"] is True
    assert document["model_control_authorized"] is False

    runtime_files = document["runtime_implementation"]["files"]
    assert "dep_car/src/dep_car/runtime/far_visibility.py" in runtime_files
    assert (
        "ros/dep_car_memory_navigation/scripts/navigation_memory_node.py"
        in runtime_files
    )

    tampered = json.loads(authority.read_text(encoding="utf-8"))
    tampered["model_control_authorized"] = True
    authority.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="model_control_authorized"):
        verify_v43_shadow_authority(
            authority,
            checkpoint_path=checkpoint,
            checkpoint_contract_path=contract,
        )
