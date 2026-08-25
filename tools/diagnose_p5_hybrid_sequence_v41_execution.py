#!/usr/bin/env python3
"""Joint Pareto audit of V4.1 raw and mandatory-hard-veto execution policy."""

import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dep_car/src"))
sys.path.insert(0, str(ROOT / "tools"))

from dep_car.training.losses_v41 import DEPCarObjectiveV41
from dep_car.training.p4_dataset import p3_training_collate, p3_training_worker_init
from dep_car.training.v4_dataset import P3HybridSequenceDatasetV4
import train_dep_car_hybrid_sequence_v4 as v4
import train_dep_car_hybrid_sequence_v41 as trainer


def policy_metrics(scores, hard, viable, costs, *, apply_hard_veto):
    any_hard = hard.any(axis=1)
    any_viable = viable.any(axis=1)
    preferred = np.where(any_viable[:, None], viable, hard)
    effective = scores.copy()
    if apply_hard_veto:
        effective = np.where(hard, effective, np.inf)
        # Frames without any hard-safe sequence are a deterministic STOP.  A
        # candidate index is retained only to keep the array shape; it is never
        # counted as a hard or hierarchy success.
        effective[~any_hard] = scores[~any_hard]
    selected = effective.argmin(axis=1)
    row = np.arange(len(scores))
    selected_hard = hard[row, selected]
    selected_viable = viable[row, selected]
    selected_preferred = preferred[row, selected] & any_hard
    hierarchy_denominator = max(1, int(any_hard.sum()))
    preferred_oracle = np.where(preferred, costs, np.inf).min(axis=1)
    selected_cost = costs[row, selected]
    hierarchy_regret_rows = selected_preferred
    hierarchy_regret = np.maximum(
        0.0, selected_cost[hierarchy_regret_rows]
        - preferred_oracle[hierarchy_regret_rows],
    )
    hard_oracle = np.where(hard, costs, np.inf).min(axis=1)
    legacy_rows = any_hard
    legacy_regret = np.maximum(
        0.0, selected_cost[legacy_rows] - hard_oracle[legacy_rows]
    )
    return {
        "selected_hard_feasible_rate": float(selected_hard.mean()),
        "selected_viable_rate": float(selected_viable.mean()),
        "hierarchy_selection_rate_when_executable": float(
            selected_preferred.sum() / hierarchy_denominator
        ),
        "hierarchical_oracle_regret": float(hierarchy_regret.mean())
        if len(hierarchy_regret) else 0.0,
        "legacy_cross_tier_oracle_regret": float(legacy_regret.mean())
        if len(legacy_regret) else 0.0,
        "hard_veto_stop_rate": float((~any_hard).mean()) if apply_hard_veto else 0.0,
    }


def main():
    config, _sha, model_config, rollout_config, loss_config = trainer.load_config()
    _bundle_path, bundle, sequence_path, _authority, _gate = trainer.verify_data_authority(config)
    _source_path, source, _source_gate = trainer.verify_source(config)
    checkpoint = trainer.best_path(trainer.resolve(config["artifacts"]["output"]))
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = v4.build_model(model_config, rollout_config, source)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    dataset = P3HybridSequenceDatasetV4(
        sample_root=bundle["sample_root"], maps_root=bundle["maps_root"],
        split="validation", index_path=bundle["index"],
        index_splits=("train", "validation"), workers=8,
        expected_map_contract_aggregate_sha256=bundle["map_contract_aggregate_sha256"],
        expected_index_sha256=bundle["index_sha256"], modality="fusion",
        sequence_index_path=sequence_path,
    )
    loader = DataLoader(
        dataset, batch_size=64, shuffle=False, num_workers=8, persistent_workers=True,
        prefetch_factor=4, pin_memory=device.type == "cuda",
        collate_fn=p3_training_collate, worker_init_fn=p3_training_worker_init,
    )
    objective = DEPCarObjectiveV41(loss_config)
    keys = ("score", "safety", "viability", "hard", "viable", "cost")
    collected = {key: [] for key in keys}
    with torch.inference_mode():
        for host in loader:
            batch = v4.select_valid(host, device)
            if batch is None:
                continue
            output, losses = v4.forward_loss(
                model, objective, batch, trainer.STAGE, device.type == "cuda"
            )
            values = {
                "score": output.scores, "safety": output.safety_logits,
                "viability": output.viability_logits,
                "hard": losses["hard_feasible"], "viable": losses["viable"],
                "cost": losses["candidate_cost"],
            }
            for key, value in values.items():
                collected[key].append(value.detach().cpu().numpy())
    value = {key: np.concatenate(rows, axis=0) for key, rows in collected.items()}
    safety_risk = np.logaddexp(0.0, -value["safety"])
    viability_risk = np.logaddexp(0.0, -value["viability"])
    policies = []
    weights = (0.0, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
    for safety_weight in weights:
        for viability_weight in weights:
            score = (
                value["score"] + safety_weight * safety_risk
                + viability_weight * viability_risk
            )
            metric = policy_metrics(
                score, value["hard"], value["viable"], value["cost"],
                apply_hard_veto=True,
            )
            policies.append({
                "safety_weight": safety_weight,
                "viability_weight": viability_weight, **metric,
            })
    def violations(row):
        return (
            max(0.0, 0.95 - row["selected_hard_feasible_rate"]) / 0.05
            + max(0.0, 0.60 - row["selected_viable_rate"]) / 0.15
            + max(0.0, 0.90 - row["hierarchy_selection_rate_when_executable"]) / 0.10
            + max(0.0, row["hierarchical_oracle_regret"] - 0.20) / 0.20
        )
    policies.sort(key=lambda row: (
        violations(row), -row["hierarchy_selection_rate_when_executable"],
        row["hierarchical_oracle_regret"], -row["selected_viable_rate"],
    ))
    best = policies[0]
    joint_pass = violations(best) == 0.0
    report = {
        "schema": "DEPCarV41ExecutionParetoAuditV1",
        "status": "PASS" if joint_pass else "FAIL",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": trainer.sha256_file(checkpoint),
        "samples": int(len(value["score"])),
        "raw_neural_argmin": policy_metrics(
            value["score"], value["hard"], value["viable"], value["cost"],
            apply_hard_veto=False,
        ),
        "mandatory_hard_veto_argmin": policy_metrics(
            value["score"], value["hard"], value["viable"], value["cost"],
            apply_hard_veto=True,
        ),
        "best_joint_calibration": best,
        "joint_gates": {
            "minimum_selected_hard_feasible_rate": 0.95,
            "minimum_selected_viable_rate": 0.60,
            "minimum_hierarchy_selection_rate_when_executable": 0.90,
            "maximum_hierarchical_oracle_regret": 0.20,
        },
        "legacy_cross_tier_regret_is_qualification_gate": False,
        "hard_veto_is_mandatory_execution_semantics": True,
        "test_split_accessed": False,
        "production_qualified": False,
    }
    output = ROOT / "reports/p5_hybrid_sequence_v41_execution_pareto_audit.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if joint_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
