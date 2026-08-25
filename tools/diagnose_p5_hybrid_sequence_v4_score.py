#!/usr/bin/env python3
"""Diagnose V4 score hierarchy without changing any checkpoint."""

import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dep_car/src"))
sys.path.insert(0, str(ROOT / "tools"))

from dep_car.training.losses_v4 import DEPCarObjectiveV4
from dep_car.training.p4_dataset import p3_training_collate, p3_training_worker_init
from dep_car.training.v4_dataset import P3HybridSequenceDatasetV4
import train_dep_car_hybrid_sequence_v4 as trainer


def rates(scores, hard, viable, costs):
    selected = scores.argmin(axis=1)
    row = np.arange(len(scores))
    selected_hard = hard[row, selected]
    selected_viable = viable[row, selected]
    any_hard = hard.any(axis=1)
    oracle = np.where(hard, costs, np.inf).min(axis=1)
    regret = np.where(any_hard, np.maximum(0.0, costs[row, selected] - oracle), 0.0)
    return {
        "selected_hard_feasible_rate": float(selected_hard.mean()),
        "selected_viable_rate": float(selected_viable.mean()),
        "mean_oracle_regret": float(regret[any_hard].mean()) if any_hard.any() else 0.0,
    }


def classification(logits, labels):
    prediction = logits >= 0.0
    tp = int(np.logical_and(prediction, labels).sum())
    fp = int(np.logical_and(prediction, ~labels).sum())
    fn = int(np.logical_and(~prediction, labels).sum())
    tn = int(np.logical_and(~prediction, ~labels).sum())
    return {
        "accuracy": float((tp + tn) / max(1, tp + fp + fn + tn)),
        "precision": float(tp / max(1, tp + fp)),
        "recall": float(tp / max(1, tp + fn)),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def main():
    config, _sha, model_config, rollout_config, loss_config = trainer.load_config()
    _bundle_path, bundle, sequence_path, _authority, _gate = trainer.verify_data_authority(config)
    checkpoint = trainer._best(trainer.resolve(config["artifacts"]["score"]))
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    capacity = trainer._best(trainer.resolve(config["artifacts"]["capacity"]))
    _source_path, source, _source_gate = trainer.verify_source(capacity, trainer.STAGES[1], config)
    model = trainer.build_model(model_config, rollout_config, source)
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
        dataset, batch_size=64, shuffle=False, num_workers=8,
        pin_memory=device.type == "cuda", persistent_workers=True, prefetch_factor=4,
        collate_fn=p3_training_collate, worker_init_fn=p3_training_worker_init,
    )
    objective = DEPCarObjectiveV4(loss_config)
    collected = {key: [] for key in ("score", "safety", "viability", "hard", "viable", "cost")}
    with torch.inference_mode():
        for host in loader:
            batch = trainer.select_valid(host, device)
            if batch is None:
                continue
            output, losses = trainer.forward_loss(
                model, objective, batch, trainer.STAGES[1], device.type == "cuda"
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
    baseline = rates(value["score"], value["hard"], value["viable"], value["cost"])
    hard_risk = np.logaddexp(0.0, -value["safety"])
    viable_risk = np.logaddexp(0.0, -value["viability"])
    policies = []
    for safety_weight in (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0):
        for viability_weight in (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0):
            metric = rates(
                value["score"] + safety_weight * hard_risk + viability_weight * viable_risk,
                value["hard"], value["viable"], value["cost"],
            )
            policies.append({
                "safety_weight": safety_weight,
                "viability_weight": viability_weight,
                **metric,
            })
    policies.sort(key=lambda row: (
        row["selected_hard_feasible_rate"] < 0.95,
        row["selected_viable_rate"] < 0.60,
        -row["selected_viable_rate"],
        -row["selected_hard_feasible_rate"],
        row["mean_oracle_regret"],
    ))
    hard_mask_score = np.where(value["hard"], value["score"], np.inf)
    viable_mask_score = np.where(
        value["viable"], value["score"],
        np.where(value["hard"], value["score"] + 1.0e3, np.inf),
    )
    report = {
        "schema": "DEPCarV4ScoreHierarchyDiagnosticV1",
        "status": "DIAGNOSTIC_COMPLETE",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": trainer.sha256_file(checkpoint),
        "samples": int(len(value["score"])),
        "baseline": baseline,
        "safety_head": classification(value["safety"], value["hard"]),
        "viability_head": classification(value["viability"], value["viable"]),
        "best_static_logit_policy": policies[0],
        "oracle_hard_veto_then_score": rates(
            hard_mask_score, value["hard"], value["viable"], value["cost"]
        ),
        "oracle_viability_hierarchy_then_score": rates(
            viable_mask_score, value["hard"], value["viable"], value["cost"]
        ),
        "conclusion": "RETRAIN_SAFETY_VIABILITY_AND_SCORE_AS_HIERARCHICAL_V41",
        "test_split_accessed": False,
        "production_qualified": False,
    }
    output = ROOT / "reports/p5_hybrid_sequence_v4_score_hierarchy_diagnostic.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
