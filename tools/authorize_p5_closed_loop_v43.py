#!/usr/bin/env python3
"""Bind the completed V4.3 DAgger authority to an immutable training contract."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
DATA_AUTHORITY_SCHEMA = "DEPCarV43ClosedLoopDataAuthorityV2"
SEQUENCE_AUTHORITY = "REOBSERVED_STATE_EXACT_SIGNED_HYBRID_ASTAR_PLAN"


def resolve(value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authority", default="data/p3_v7_v43/index/closed_loop_data_authority.json")
    parser.add_argument(
        "--integrity-audit",
        default="reports/p3_v7_v43_independent_integrity_audit.json",
    )
    parser.add_argument(
        "--capacity-diagnostic",
        default="reports/p5_closed_loop_v43_capacity_diagnostic.json",
    )
    parser.add_argument("--output", default="dep_car/config/p5_closed_loop_v43.yaml")
    args = parser.parse_args()
    authority_path, output = resolve(args.authority), resolve(args.output)
    integrity_path = resolve(args.integrity_audit)
    capacity_path = resolve(args.capacity_diagnostic)
    authority = json.loads(authority_path.read_text(encoding="utf-8"))
    claimed = authority.get("authority_sha256"); content = dict(authority); content.pop("authority_sha256", None)
    if (
        authority.get("schema") != DATA_AUTHORITY_SCHEMA
        or authority.get("status") != "PASS" or authority.get("errors") != []
        or claimed != canonical_sha256(content)
        or authority.get("continuous_sequence_authority") != SEQUENCE_AUTHORITY
        or authority.get("runtime_hybrid_astar_dependency") is not False
        or authority.get("test_split_opened") is not False
    ):
        raise RuntimeError("V4.3 closed-loop data authority is not signable")
    integrity = json.loads(integrity_path.read_text(encoding="utf-8"))
    if (
        integrity.get("schema") != "DEPCarV43IndependentIntegrityAuditV1"
        or integrity.get("status") != "PASS"
        or integrity.get("errors") != []
        or integrity.get("authority_file_sha256") != sha256_file(authority_path)
        or integrity.get("test_split_opened") is not False
    ):
        raise RuntimeError("V4.3 independent integrity audit is not signable")
    implementation = {
        "model": "dep_car/src/dep_car/model/dep_car_net_v43.py",
        "candidate_model": "dep_car/src/dep_car/model/dep_car_net_v4.py",
        "rollout": "dep_car/src/dep_car/model/hybrid_sequence_rollout.py",
        "loss": "dep_car/src/dep_car/training/losses_v43.py",
        "dataset": "dep_car/src/dep_car/training/v43_dataset.py",
        "score_dataset": "dep_car/src/dep_car/training/score_dataset.py",
    }
    for name, path in tuple(implementation.items()):
        implementation[name + "_sha256"] = sha256_file(resolve(path))
    source_checkpoint = resolve("models/dep_car/p5_hybrid_sequence_v41/fusion_hierarchical_score.best.pth")
    source_contract = source_checkpoint.with_suffix(".contract.json")
    guarded = resolve("models/dep_car/p5_hybrid_sequence_v42/fusion_guarded_simulation.authority.json")
    guarded_payload = json.loads(guarded.read_text(encoding="utf-8"))
    capacity = json.loads(capacity_path.read_text(encoding="utf-8"))
    if (
        guarded_payload.get("status") != "P6_GUARDED_SIMULATION_AUTHORIZED"
        or guarded_payload.get("source_checkpoint_sha256") != sha256_file(source_checkpoint)
    ):
        raise RuntimeError("V4.3 source is not the diagnosed V4.2 guarded policy")
    if (
        capacity.get("schema") != "DEPCarV43CapacityDiagnosticV1"
        or capacity.get("status") != "PASS"
        or capacity.get("checkpoint_sha256") != sha256_file(source_checkpoint)
        or capacity.get("data_authority_gate", {}).get("authority_sha256")
        != sha256_file(authority_path)
        or capacity.get("test_split_accessed") is not False
    ):
        raise RuntimeError("V4.3 source capacity diagnostic is not signable")
    config = {
        "schema": "DEPCarV43ClosedLoopTrainingContractV4",
        "architecture_id": "dep_car_multimodal_v43_guarded_contextual_residual_closed_loop_hybrid_sequence_ackermann_15x6",
        "objective_id": "dep_car_objective_v19_guarded_contextual_exact_closed_loop_selector",
        "scope": "fusion_only_frozen_unified_candidates_guarded_reobserved_dagger_selector",
        "test_split_sealed": True,
        "implementation": implementation,
        "dataset": {
            "authority": str(authority_path.relative_to(ROOT)),
            "authority_sha256": sha256_file(authority_path),
            "continuous_sequence_authority": SEQUENCE_AUTHORITY,
            "runtime_ground_truth_input": False,
            "test_split_sealed": True,
            "integrity_audit": str(integrity_path.relative_to(ROOT)),
            "integrity_audit_sha256": sha256_file(integrity_path),
        },
        "source": {
            "checkpoint": str(source_checkpoint.relative_to(ROOT)),
            "checkpoint_sha256": sha256_file(source_checkpoint),
            "checkpoint_contract": str(source_contract.relative_to(ROOT)),
            "checkpoint_contract_sha256": sha256_file(source_contract),
            "guarded_authority": str(guarded.relative_to(ROOT)),
            "guarded_authority_sha256": sha256_file(guarded),
            "role": "V42_guarded_failure_policy_initialization",
            "capacity_diagnostic": str(capacity_path.relative_to(ROOT)),
            "capacity_diagnostic_sha256": sha256_file(capacity_path),
            "candidate_decoder_role": "FROZEN_ACCEPTED_UNIFIED_SEQUENCE_CAPACITY",
        },
        "artifacts": {
            "output": "models/dep_car/p5_closed_loop_v43/fusion_closed_loop_sequence.pth",
            "pilot": "models/dep_car/p5_closed_loop_v43/pilot_contextual_exact/fusion_closed_loop_sequence.pth",
        },
        "model": {
            "candidates": 15, "actions": 6, "hidden_dim": 128,
            "primitive_dim": 64, "stage_dim": 32,
            "template_logit_bias": 1.25, "steps_per_action": 5,
            "local_distance_extent_m": 8.0,
            "residual_score_span": 3.0,
        },
        "training": {
            "epochs": 24, "pilot_epochs": 8,
            "batch_size": 128,
            "learning_rate": 0.0010,
            "weight_decay": 0.00001,
            "gradient_clip": 5.0, "sensor_dropout_probability": 0.10,
            "mixed_precision": True, "workers": 8, "prefetch_factor": 4,
            "progress_interval_steps": 25, "torch_threads": 8, "seed": 86431,
        },
        "base_loss": {
            "route_tube_radius_m": 0.35, "route_outer_tube_radius_m": 0.55,
            "minimum_sequence_progress_m": 0.20,
            "maximum_terminal_heading_error_rad": 0.80,
            "clearance_margin_m": 0.06, "sequence_actions": 6,
            "capacity_top_k": 3, "safety_weight": 4.0, "route_weight": 2.0,
            "kinematic_weight": 2.0, "comfort_weight": 0.02,
            "sequence_gear_weight": 1.5, "first_action_geometry_weight": 1.0,
            "diversity_weight": 0.15, "safety_head_weight": 0.75,
            "viability_head_weight": 0.75, "score_weight": 1.0,
            "conditional_shift_weight": 0.08,
            "conditional_reverse_distance_weight": 0.05,
            "score_temperature": 0.25,
        },
        "closed_loop_loss": {
            "ranking_weight": 4.0,
            "selected_sequence_weight": 2.0,
            "reverse_forward_weight": 8.0,
            "action_plan_geometry_weight": 0.5,
            "safety_head_weight": 1.0,
            "eligible_margin_weight": 8.0,
            "egress_ranking_weight": 4.0,
            "egress_margin_weight": 4.0,
            "eligible_margin": 0.40,
            "score_temperature": 0.20,
            "egress_clearance_tolerance_m": 0.03,
            "egress_terminal_gain_m": 0.05,
        },
        "qualification": {
            "maximum_initial_pose_hard_unsafe_rate": 0.13,
            "maximum_zero_hard_feasible_rate_given_safe_initial_pose": 0.21,
            "minimum_unsafe_initial_egress_candidate_rate": 0.99,
            "maximum_best_of_15_gear_error_rate": 0.08,
            "minimum_multiaction_prefix_coverage": 0.90,
            "minimum_selected_hard_feasible_rate_when_available": 0.96,
            "minimum_selected_egress_rate_when_available": 0.85,
            "minimum_selected_navigation_eligible_rate_when_available": 0.90,
            "minimum_selected_exact_sequence_rate_when_available": 0.85,
            "maximum_plan_gear_prefix_error_rate": 0.20,
            "minimum_reverse_then_forward_coverage": 0.75,
            "minimum_mandatory_guard_stop_rate_when_no_candidate": 1.0,
            "active_control_authorized_by_training": False,
            "production_qualified_by_training": False,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    os.replace(temporary, output)
    print(json.dumps({
        "schema": "DEPCarV43TrainingAuthorizationV1", "status": "PASS",
        "output": str(output), "output_sha256": sha256_file(output),
        "data_authority": str(authority_path), "data_authority_sha256": sha256_file(authority_path),
        "integrity_audit": str(integrity_path), "integrity_audit_sha256": sha256_file(integrity_path),
        "capacity_diagnostic": str(capacity_path), "capacity_diagnostic_sha256": sha256_file(capacity_path),
        "source_checkpoint_sha256": sha256_file(source_checkpoint),
        "formal_training_started": False, "test_split_opened": False,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
