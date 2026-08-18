import copy
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import verify_p4
from dep_car.model.dep_car_net import DEPCarNetV1
from dep_car.model.implementation_contract import build_p4_implementation_contract
from dep_car.training.losses import DEPCarObjectiveV1
from dep_car.training.stages import parameter_partitions


def _training_authority(samples=2):
    return {
        "training_index_sha256": "1" * 64,
        "training_index_content_aggregate_sha256": "2" * 64,
        "expected_map_contract_aggregate_sha256": "3" * 64,
        "indexed_samples": samples,
    }


def _p3_reaudit_payload(status="PASS", samples=2):
    errors = [] if status == "PASS" else ["overall_zero_feasible_rate_lt_0_10"]
    implementation = build_p4_implementation_contract(ROOT)
    return {
        "schema": "DEPCarP3DevelopmentReauditV3",
        "status": status,
        "errors": errors,
        "qualification_eligible": True,
        "sample_files_audited": samples,
        "sample_files_discovered": samples,
        "sample_failures": {},
        "audit_implementation": {
            "p4_implementation_schema": implementation["schema"],
            "p4_implementation_aggregate_sha256": implementation[
                "aggregate_sha256"
            ],
            "p4_implementation_files": implementation["files"],
        },
        "scope": {
            "test_split_used_for_tuning": False,
            "test_npz_opened": False,
            "test_map_yaml_or_png_opened": False,
        },
        "training_authority": {
            "index_sha256": "1" * 64,
            "content_aggregate_sha256": "2" * 64,
            "map_contract_aggregate_sha256": "3" * 64,
            "test_split_used": False,
        },
        "statistics": {
            "overall": {
                "new": {
                    "zero_feasible_rate": 0.05,
                    "feasible_candidates_median": 12.0,
                }
            },
            "by_mode": {
                "NORMAL": {
                    "samples": samples,
                    "new": {
                        "zero_feasible_rate": 0.01,
                        "feasible_candidates_median": 15.0,
                    },
                },
                "THREE_POINT_TURN": {
                    "samples": 1,
                    "new": {
                        "zero_feasible_rate": 0.1,
                        "feasible_candidates_median": 8.0,
                    },
                },
            },
        },
    }


def test_terminal_history_uses_terminal_window_and_last_not_minimum():
    summary = verify_p4.terminal_history([10.0, 1.0, 8.0, 6.0], window=2)

    assert summary["minimum_observed"] == 1.0
    assert summary["terminal_window_mean"] == 7.0
    assert summary["terminal_last"] == 6.0
    assert summary["terminal_window_to_initial_ratio"] == 0.7
    assert summary["terminal_last_to_initial_ratio"] == 0.6


def test_candidate_smoke_fixed_feature_groups_account_for_candidate_partition():
    torch.manual_seed(7)
    model = DEPCarNetV1()
    groups, evidence = verify_p4.candidate_smoke_parameter_groups(model)
    partitions = parameter_partitions(model)

    trainable_ids = {
        id(parameter) for group in groups for parameter in group["params"]
    }
    candidate_ids = {id(parameter) for parameter in partitions["candidate"]}
    score_ids = {id(parameter) for parameter in partitions["score"]}
    fixed_ids = {
        id(parameter)
        for parameter in partitions["candidate"]
        if not parameter.requires_grad
    }

    assert trainable_ids | fixed_ids == candidate_ids
    assert not trainable_ids.intersection(fixed_ids)
    assert not trainable_ids.intersection(score_ids)
    assert groups[0]["lr"] == verify_p4.CANDIDATE_SMOKE_READOUT_LR
    assert evidence["candidate_ownership_accounted"] is True
    assert evidence["fixed_feature_parameters_frozen"] is True
    assert evidence["readout_parameters_trainable"] is True
    assert evidence["score_parameters_excluded"] is True


def test_checkpoint_payload_hashes_and_loads_the_same_single_byte_snapshot(
    tmp_path, monkeypatch
):
    checkpoint = tmp_path / "init.pth"
    torch.save({"model_state_dict": {"weight": torch.ones(2)}}, checkpoint)
    expected = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    original_read_bytes = Path.read_bytes
    reads = []

    def counted_read_bytes(path):
        reads.append(Path(path))
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", counted_read_bytes)
    payload, evidence = verify_p4.load_verified_checkpoint_payload(
        checkpoint, {"checkpoint_sha256": expected}
    )

    assert reads == [checkpoint]
    assert torch.equal(payload["model_state_dict"]["weight"], torch.ones(2))
    assert evidence["checkpoint_sha256"] == expected
    assert evidence["path_reads_after_contract_verification"] == 1
    assert evidence["deserialization_source"] == "BytesIO_of_the_hashed_byte_snapshot"
    assert evidence["torch_load_weights_only"] is True


def test_checkpoint_payload_rejects_bytes_that_differ_from_contract(tmp_path):
    checkpoint = tmp_path / "init.pth"
    torch.save({"model_state_dict": {}}, checkpoint)

    with pytest.raises(RuntimeError, match="differ from verified contract"):
        verify_p4.load_verified_checkpoint_payload(
            checkpoint, {"checkpoint_sha256": "0" * 64}
        )


def test_training_authority_records_frozen_map_value_and_dataset_actual(tmp_path):
    index_path = tmp_path / "training_index.json"
    training_path = tmp_path / "training.yaml"
    index = {
        "content_aggregate_sha256": "2" * 64,
        "samples": 2,
        "splits": ["train", "validation"],
        "entries": [{"path": "a"}, {"path": "b"}],
    }
    index_path.write_text(json.dumps(index), encoding="utf-8")
    training_path.write_text(
        "dataset:\n"
        "  content_aggregate_sha256: " + "2" * 64 + "\n"
        "  map_contract_aggregate_sha256: " + "3" * 64 + "\n",
        encoding="utf-8",
    )

    configured = verify_p4.training_authority_evidence(
        index_path, index, training_path
    )
    bound = verify_p4.bind_dataset_map_authority(
        configured,
        {
            "schema": "IndexedMapContractAggregateV1",
            "map_count": 27,
            "aggregate_sha256": "3" * 64,
        },
    )

    assert bound["expected_map_contract_aggregate_sha256"] == "3" * 64
    assert bound["actual_map_contract_aggregate_sha256"] == "3" * 64
    assert bound["actual_equals_expected_map_contract_aggregate"] is True
    assert bound["actual_map_count"] == 27
    assert "training.yaml::dataset.map_contract" in bound[
        "expected_map_contract_aggregate_source"
    ]


def test_p4_smoke_resolves_curated_paths_from_training_authority():
    paths = verify_p4.configured_dataset_paths(
        ROOT / "dep_car/config/training.yaml"
    )

    assert paths == {
        "root": (ROOT / "data/p3_v3/bundle_v2_curated/samples").resolve(),
        "maps": (ROOT / "data/p3_pilot/maps").resolve(),
        "index": (
            ROOT / "data/p3_v3/bundle_v2_curated/training_index.json"
        ).resolve(),
    }


def test_p4_smoke_rejects_dataset_path_outside_project(tmp_path):
    training = tmp_path / "training.yaml"
    training.write_text(
        "dataset:\n"
        "  root: /var/tmp/outside\n"
        "  maps: data/maps\n"
        "  index: data/index.json\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="escapes project root"):
        verify_p4.configured_dataset_paths(training, project_root=tmp_path)


def test_dataset_actual_map_aggregate_must_equal_frozen_configuration():
    with pytest.raises(RuntimeError, match="actual map aggregate differs"):
        verify_p4.bind_dataset_map_authority(
            _training_authority(),
            {"aggregate_sha256": "4" * 64},
        )


def test_p3_fail_is_reported_as_p5_blocker_not_p4_smoke_failure(tmp_path):
    report_path = tmp_path / "p3.json"
    report_path.write_text(
        json.dumps(_p3_reaudit_payload(status="FAIL")), encoding="utf-8"
    )

    evidence = verify_p4.summarize_p3_development_reaudit(
        report_path, _training_authority()
    )

    assert evidence["status"] == "FAIL"
    assert evidence["sample_files_audited"] == 2
    assert evidence["overall_new_metrics"]["zero_feasible_rate"] == 0.05
    assert sorted(evidence["by_mode_new_metrics"]) == [
        "NORMAL", "THREE_POINT_TURN"
    ]
    assert evidence["p5_gate_eligible"] is False
    assert evidence["p4_implementation_status_is_independent"] is True
    assert evidence["test_not_accessed"] is True


def test_p3_pass_is_p5_eligible_only_for_complete_matching_sealed_audit(tmp_path):
    report_path = tmp_path / "p3.json"
    report_path.write_text(
        json.dumps(_p3_reaudit_payload(status="PASS")), encoding="utf-8"
    )

    evidence = verify_p4.summarize_p3_development_reaudit(
        report_path, _training_authority()
    )

    assert evidence["p5_gate_eligible"] is True
    assert evidence["p5_gate_errors"] == []
    assert all(evidence["training_authority_matches"].values())


def test_p3_test_access_or_incomplete_inventory_blocks_p5_eligibility(tmp_path):
    payload = _p3_reaudit_payload(status="PASS")
    payload["scope"]["test_npz_opened"] = True
    payload["sample_files_audited"] = 1
    report_path = tmp_path / "p3.json"
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    evidence = verify_p4.summarize_p3_development_reaudit(
        report_path, _training_authority()
    )

    assert evidence["p5_gate_eligible"] is False
    assert "p3_reaudit_did_not_cover_the_frozen_index" in evidence["p5_gate_errors"]
    assert "p3_reaudit_accessed_or_used_sealed_test_data" in evidence[
        "p5_gate_errors"
    ]


def test_training_runtime_and_initialization_loss_contract_match():
    contract_path = (
        ROOT / "models/dep_car/dep_car_net_v1_depth_v483_init.contract.json"
    )
    contract = json.loads(contract_path.read_text(encoding="utf-8"))

    evidence = verify_p4.verify_loss_contract(
        contract,
        ROOT / "dep_car/config/training.yaml",
        DEPCarObjectiveV1(),
    )

    assert evidence["status"] == "PASS"
    assert len(set(evidence["loss_config_hashes"].values())) == 1
    assert evidence["objective_id"] == DEPCarObjectiveV1.objective_id
    assert evidence["objective_revision"] == DEPCarObjectiveV1.objective_revision


def test_loss_contract_rejects_mutated_initialization_hash():
    contract_path = (
        ROOT / "models/dep_car/dep_car_net_v1_depth_v483_init.contract.json"
    )
    contract = copy.deepcopy(json.loads(contract_path.read_text(encoding="utf-8")))
    contract["training_contract"]["loss_config_sha256"] = "0" * 64

    with pytest.raises(RuntimeError, match="loss config differs"):
        verify_p4.verify_loss_contract(
            contract,
            ROOT / "dep_car/config/training.yaml",
            DEPCarObjectiveV1(),
        )


def test_direct_oracle_gate_requires_step0_seed_and_real_convergence():
    candidate = {
        "initial": 10.0,
        "terminal_last": 5.0,
    }
    valid = {
        "initial": 10.0,
        "achieved_floor": 4.0,
        "relative_improvement": 0.6,
        "optimization_relative_improvement": 0.5,
        "starting_point": "network_step0_raw_residuals",
        "starting_raw_sha256": "initial",
        "exact_step0_restart_present": True,
        "exact_step0_restart_loss": 10.0,
        "finite": True,
        "exact_production_objective_and_rollout": True,
    }

    evidence = verify_p4.direct_oracle_gate(
        valid, candidate, "initial", "terminal"
    )

    assert evidence["errors"] == []
    assert evidence["starts_at_network_step0"]
    assert evidence["exact_restart_matches_step0"]
    assert evidence["independent_from_network_terminal"]
    assert evidence["significantly_converged"]
    assert evidence["no_worse_than_network_terminal"]


def test_direct_oracle_reconstructs_loss_per_independent_sample():
    objective = DEPCarObjectiveV1()
    trajectories = torch.zeros((2, 15, 11, 6), dtype=torch.float32)
    trajectories[:, :, :, 0] = torch.linspace(0.0, 1.0, 11)
    output = SimpleNamespace(
        trajectories=trajectories,
        residuals=torch.zeros((2, 15, 4), dtype=torch.float32),
    )
    candidate_cost = torch.arange(30, dtype=torch.float32).reshape(2, 15) / 30.0
    safety = torch.full((2, 15), 0.1, dtype=torch.float32)
    kinematic = torch.full((2, 15), 0.2, dtype=torch.float32)
    batch = {
        "geometry_valid": torch.ones(2, dtype=torch.bool),
        "requested_gear": torch.tensor([1, -1]),
    }
    top_k = torch.topk(
        candidate_cost,
        objective.config.capacity_top_k,
        dim=1,
        largest=False,
    ).values.mean(dim=1)
    diversity = verify_p4.candidate_diversity_loss(
        trajectories,
        batch["requested_gear"],
        objective.config.minimum_normalized_terminal_separation,
        objective.config.forward_diversity_scales,
        objective.config.reverse_diversity_scales,
    )
    expected = (
        top_k
        + objective.config.weights.safety * safety.mean(dim=1)
        + objective.config.weights.diversity * diversity
        + objective.config.weights.kinematic_all * kinematic.mean(dim=1)
    )
    result = {
        "candidate_cost": candidate_cost,
        "safety_per_candidate": safety,
        "kinematic_per_candidate": kinematic,
        "candidate": expected.mean(),
    }

    actual = verify_p4.candidate_loss_per_sample(
        objective, output, result, batch
    )

    assert torch.allclose(actual, expected)


def test_terminal_seeded_or_stalled_direct_oracle_cannot_pass():
    candidate = {
        "initial": 10.0,
        "terminal_last": 5.0,
    }
    terminal_seeded_and_stalled = {
        "initial": 5.0,
        "achieved_floor": 5.0,
        "relative_improvement": 0.0,
        "optimization_relative_improvement": 0.0,
        "starting_point": "network_terminal_raw_residuals",
        "starting_raw_sha256": "terminal",
        "exact_step0_restart_present": False,
        "exact_step0_restart_loss": 5.0,
        "finite": True,
        "exact_production_objective_and_rollout": True,
    }

    evidence = verify_p4.direct_oracle_gate(
        terminal_seeded_and_stalled, candidate, "initial", "terminal"
    )

    assert "direct_residual_oracle_did_not_start_at_network_step0" in evidence["errors"]
    assert "direct_residual_oracle_is_not_independent_from_network_terminal" in evidence["errors"]
    assert "direct_residual_oracle_did_not_significantly_converge" in evidence["errors"]
