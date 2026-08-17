import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import accept_p5_candidate as acceptance
import train_dep_car as trainer
from dep_car.model.implementation_contract import build_p4_implementation_contract


@pytest.fixture(autouse=True)
def lightweight_state_signature(monkeypatch):
    monkeypatch.setattr(
        trainer, "_validate_model_state_dict_structure", lambda _state: None
    )
    monkeypatch.setattr(
        trainer,
        "verify_checkpoint",
        lambda _checkpoint, contract, **_kwargs: json.loads(
            Path(contract).read_text(encoding="utf-8")
        ),
    )


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def authority_for_test():
    authority = copy.deepcopy(trainer._training_config_authority())
    qualification = authority["raw"]["qualification"]
    qualification["corrected_footprint_p3_status"] = "PASS"
    qualification["p5_formal_training_allowed"] = True
    qualification["blocked_gates"] = []
    thresholds = qualification["candidate_acceptance"]
    thresholds.update({
        "minimum_completed_epochs": 2,
        "minimum_global_steps": 2,
        "required_maneuvers": ["NORMAL"],
        "required_requested_gears": ["FORWARD", "REVERSE"],
        "required_candidate_contexts": ["UNKNOWN"],
        "minimum_validation_frames_per_maneuver": 1,
        "minimum_validation_frames_per_requested_gear": 1,
        "minimum_validation_frames_per_candidate_context": 1,
    })
    return authority


def full_contract(**updates):
    contract = json.loads(
        trainer.DEFAULT_CANDIDATE_INITIALIZATION.with_suffix(
            ".contract.json"
        ).read_text(encoding="utf-8")
    )
    contract["implementation_contract"] = build_p4_implementation_contract(ROOT)
    contract["training_contract"]["objective_id"] = trainer.DEPCarObjectiveV1.objective_id
    contract["training_contract"]["objective_revision"] = (
        trainer.DEPCarObjectiveV1.objective_revision
    )
    contract.update(updates)
    return contract


def candidate_artifact(
    tmp_path, authority, *, smoke=False, partial=False, bad_group=False
):
    checkpoint = tmp_path / "candidate.pth"
    implementation = build_p4_implementation_contract(ROOT)
    index_path = authority["authority_paths"]["index"]
    configured_dataset = authority["dataset"]
    payload = {
        "schema": trainer.CHECKPOINT_SCHEMA,
        "checkpoint_version": trainer.CHECKPOINT_VERSION,
        "architecture_id": trainer.P4_ARCHITECTURE_ID,
        "model_state_dict": {"sentinel": torch.zeros(1)},
        "production_qualified": False,
        "status": "TRAINED_UNQUALIFIED",
        "qualification_status": "UNQUALIFIED",
        "training_stage": "candidate_capacity",
        "modality": "fusion",
        "smoke_lineage": smoke,
        "partial_epoch": partial,
        "completed_epochs": 2,
        "global_step": 2,
        "training_config_sha256": authority["file_sha256"],
        "loss_config_sha256": authority["loss_config_sha256"],
        "trainer_sha256": sha256(trainer.TRAINER_PATH),
        "training_index_sha256": sha256(index_path),
        "training_index_content_sha256": configured_dataset[
            "content_aggregate_sha256"
        ],
        "map_contract_aggregate_sha256": configured_dataset[
            "map_contract_aggregate_sha256"
        ],
        "implementation_aggregate_sha256": implementation["aggregate_sha256"],
    }
    torch.save(payload, checkpoint)
    metrics_path = tmp_path / "candidate.metrics.json"
    def metric_row(frames):
        return {
            "frames": frames,
            "zero_feasible_rate": 0.01,
            "mean_feasible_candidates": 10.0,
            "kinematic_violation_rate": 0.01,
        }
    candidate_metrics = {
        "overall": {
            "frames": 10,
            "zero_feasible_rate": 0.01,
            "mean_feasible_candidates": 10.0,
            "kinematic_violation_rate": 0.01,
        },
        "by_maneuver": {"NORMAL": metric_row(10)},
        "by_requested_gear": {
            "FORWARD": metric_row(5),
            "REVERSE": metric_row(5),
        },
        "by_candidate_context": {"UNKNOWN": metric_row(10)},
    }
    if bad_group:
        candidate_metrics["by_maneuver"]["NORMAL"]["zero_feasible_rate"] = 0.9
    metrics_payload = {"validation": {
        "total": 1.0,
        "geometry_valid_fraction": 0.95,
        "candidate_metrics": candidate_metrics,
    }}
    payload["metrics"] = metrics_payload
    torch.save(payload, checkpoint)
    metrics_path.write_text(json.dumps({
        "schema": "DEPCarP5TrainingMetricsV1",
        "architecture_id": trainer.P4_ARCHITECTURE_ID,
        "training_stage": "candidate_capacity",
        "modality": "fusion",
        "qualification_status": "UNQUALIFIED",
        "production_qualified": False,
        "completed_epochs": 2,
        "global_step": 2,
        "partial_epoch": partial,
        "metrics": metrics_payload,
    }), encoding="utf-8")
    contract = full_contract(**{
        "checkpoint_sha256": sha256(checkpoint),
        "production_qualified": False,
        "status": "TRAINED_UNQUALIFIED",
        "qualification_status": "UNQUALIFIED",
        "training_stage": "candidate_capacity",
        "modality": "fusion",
        "training_run": {
            "smoke_limited": smoke,
            "smoke_lineage": smoke,
            "partial_epoch": partial,
            "training_config_sha256": authority["file_sha256"],
            "loss_config_sha256": authority["loss_config_sha256"],
            "trainer_sha256": sha256(trainer.TRAINER_PATH),
            "implementation_aggregate_sha256": implementation[
                "aggregate_sha256"
            ],
        },
        "p3_footprint_gate": {"passed": True},
        "index_content_gate": {"passed": True},
        "dataset_authority_gate": {"passed": True},
        "validation_coverage_gate": {"passed": True},
        "training_yaml_qualification_gate": {"passed": True},
        "dataset_provenance": {
            **full_contract()["dataset_provenance"],
            "index_sha256": sha256(index_path),
            "content_aggregate_sha256": configured_dataset[
                "content_aggregate_sha256"
            ],
            "map_contract_aggregate_sha256": configured_dataset[
                "map_contract_aggregate_sha256"
            ],
        },
        "artifacts": {
            "metrics": metrics_path.name,
            "metrics_sha256": sha256(metrics_path),
        },
    })
    checkpoint.with_suffix(".contract.json").write_text(
        json.dumps(contract), encoding="utf-8"
    )
    return checkpoint


def test_acceptance_tool_generates_machine_verifiable_pass(tmp_path):
    authority = authority_for_test()
    checkpoint = candidate_artifact(tmp_path, authority)
    result = acceptance.evaluate_candidate(checkpoint, authority=authority)
    assert result["status"] == "PASS"
    assert result["gate_passed"] is True
    written = json.loads(
        checkpoint.with_suffix(".candidate_acceptance.json").read_text()
    )
    assert written["checkpoint_sha256"] == sha256(checkpoint)
    assert written["acceptance_tool_sha256"] == sha256(
        ROOT / "tools/accept_p5_candidate.py"
    )


@pytest.mark.parametrize("smoke,partial", ((True, False), (False, True)))
def test_acceptance_tool_rejects_smoke_or_partial_candidate(
    tmp_path, smoke, partial
):
    authority = authority_for_test()
    checkpoint = candidate_artifact(
        tmp_path, authority, smoke=smoke, partial=partial
    )
    result = acceptance.evaluate_candidate(checkpoint, authority=authority)
    assert result["status"] == "FAIL"
    assert result["gate_passed"] is False
    assert "smoke_lineage" in result["errors"] or "partial_epoch" in result["errors"]


def test_acceptance_rejects_catastrophic_required_group_even_if_overall_passes(tmp_path):
    authority = authority_for_test()
    checkpoint = candidate_artifact(tmp_path, authority, bad_group=True)
    result = acceptance.evaluate_candidate(checkpoint, authority=authority)
    assert result["gate_passed"] is False
    assert any("maneuver_NORMAL_maximum_validation_zero_feasible_rate" == error
               for error in result["errors"])


def test_acceptance_rejects_unknown_context_without_counting_it_as_recovery(
    tmp_path,
):
    authority = authority_for_test()
    authority["raw"]["qualification"]["candidate_acceptance"][
        "required_candidate_contexts"
    ] = ["MISSION", "RECOVERY"]
    checkpoint = candidate_artifact(tmp_path, authority)
    result = acceptance.evaluate_candidate(checkpoint, authority=authority)
    assert result["status"] == "FAIL"
    assert "candidate_context_unexpected_UNKNOWN" in result["errors"]
    assert "candidate_context_coverage_RECOVERY" in result["errors"]


def test_acceptance_cli_exposes_no_threshold_override():
    with pytest.raises(SystemExit):
        acceptance.main(["candidate.pth", "--maximum-zero-feasible-rate", "1.0"])
