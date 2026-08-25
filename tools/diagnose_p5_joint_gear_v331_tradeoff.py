#!/usr/bin/env python3
"""Sweep the V3.3.1 gear-logit threshold over the formal validation split."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dep_car/src"))
sys.path.insert(0, str(ROOT / "tools"))

from dep_car.training.losses_v3 import DEPCarJointGearLossConfigV3
from dep_car.training.losses_v31 import DEPCarSequenceCorrectionConfigV31
from dep_car.training.losses_v33 import (
    DEPCarExplicitGearLossConfigV33,
    DEPCarObjectiveV33,
)
from dep_car.training.p4_dataset import p3_training_collate, p3_training_worker_init
from dep_car.training.score_dataset import P3JointGearDatasetV3
import train_dep_car_gear_selector_v331 as trainer


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default="models/dep_car/p5_joint_gear_v331/fusion_unilateral_safe_bank_correction.pth",
    )
    parser.add_argument(
        "--output", default="reports/p5_joint_gear_v331_threshold_tradeoff.json"
    )
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def metric_row(margin, threshold, teacher, required, no_forward, forward_progress, feasible):
    request = margin > threshold
    forward_safe = feasible[:, :15].any(axis=1)
    reverse_safe = feasible[:, 15:].any(axis=1)
    any_safe = forward_safe | reverse_safe
    requested_available = np.where(request, reverse_safe, forward_safe)
    fallback = (~requested_available) & np.where(request, forward_safe, reverse_safe)
    oracle_reverse = required & teacher
    oracle_forward = required & ~teacher
    return {
        "threshold": float(threshold),
        "requested_reverse_rate": float(request.mean()),
        "unnecessary_reverse_rate": float(
            (request & forward_progress).sum() / max(1, forward_progress.sum())
        ),
        "oracle_reverse_recall_within_required": float(
            (request & oracle_reverse).sum() / max(1, oracle_reverse.sum())
        ),
        "oracle_forward_false_reverse_rate_within_required": float(
            (request & oracle_forward).sum() / max(1, oracle_forward.sum())
        ),
        "no_hard_forward_reverse_selection_rate": float(
            (request & no_forward).sum() / max(1, no_forward.sum())
        ),
        "requested_bank_hard_available_rate": float(requested_available.mean()),
        "hard_safety_fallback_rate": float(fallback.mean()),
        "post_veto_selected_hard_feasible_rate": float(any_safe.mean()),
    }


def passes(row, acceptance):
    selector = acceptance["selector"]
    execution = acceptance["execution"]
    return (
        row["unnecessary_reverse_rate"]
        <= float(selector["maximum_unnecessary_reverse_rate"])
        and row["oracle_reverse_recall_within_required"]
        >= float(selector["minimum_oracle_reverse_recall_within_required"])
        and row["oracle_forward_false_reverse_rate_within_required"]
        <= float(selector["maximum_oracle_forward_false_reverse_rate_within_required"])
        and row["no_hard_forward_reverse_selection_rate"]
        >= float(selector["minimum_no_hard_forward_reverse_selection_rate"])
        and row["requested_bank_hard_available_rate"]
        >= float(execution["minimum_requested_bank_hard_available_rate"])
        and row["hard_safety_fallback_rate"]
        <= float(execution["maximum_hard_safety_fallback_rate"])
        and row["post_veto_selected_hard_feasible_rate"]
        >= float(execution["minimum_post_veto_selected_hard_feasible_rate"])
    )


def violation(row, acceptance):
    selector = acceptance["selector"]
    execution = acceptance["execution"]
    lower = (
        ("oracle_reverse_recall_within_required", selector["minimum_oracle_reverse_recall_within_required"]),
        ("no_hard_forward_reverse_selection_rate", selector["minimum_no_hard_forward_reverse_selection_rate"]),
        ("requested_bank_hard_available_rate", execution["minimum_requested_bank_hard_available_rate"]),
        ("post_veto_selected_hard_feasible_rate", execution["minimum_post_veto_selected_hard_feasible_rate"]),
    )
    upper = (
        ("unnecessary_reverse_rate", selector["maximum_unnecessary_reverse_rate"]),
        ("oracle_forward_false_reverse_rate_within_required", selector["maximum_oracle_forward_false_reverse_rate_within_required"]),
        ("hard_safety_fallback_rate", execution["maximum_hard_safety_fallback_rate"]),
    )
    return float(sum(max(0.0, float(target) - row[name]) / max(float(target), 1e-6) for name, target in lower)
                 + sum(max(0.0, row[name] - float(target)) / max(float(target), 1e-3) for name, target in upper))


def main():
    args = parse_args()
    if args.workers < 1 or not torch.cuda.is_available():
        raise RuntimeError("positive workers and CUDA are required")
    config, _config_sha, v33_config, v31_config, base_config, acceptance = trainer.load_config()
    _bundle_path, bundle, sequence_path, _authority, data_gate = trainer.v33.v31.v3.verify_data_authority(base_config)
    _source_path, source, _source_gate = trainer.verify_source(config, data_gate)
    checkpoint = trainer.resolve(args.checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    dataset = P3JointGearDatasetV3(
        bundle["sample_root"], bundle["maps_root"], split="validation",
        index_path=bundle["index"], index_splits=("train", "validation"),
        workers=args.workers,
        expected_map_contract_aggregate_sha256=bundle["map_contract_aggregate_sha256"],
        expected_index_sha256=bundle["index_sha256"], modality="fusion",
        sequence_index_path=sequence_path,
    )
    loader = DataLoader(
        dataset, batch_size=64, shuffle=False, num_workers=args.workers,
        pin_memory=True, persistent_workers=True, prefetch_factor=4,
        collate_fn=p3_training_collate, worker_init_fn=p3_training_worker_init,
    )
    device = torch.device("cuda")
    model = trainer.build_model(config, v33_config, base_config, data_gate, source)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.freeze_base(); model.to(device).eval()
    objective = DEPCarObjectiveV33(
        DEPCarJointGearLossConfigV3(**base_config["loss"]),
        DEPCarSequenceCorrectionConfigV31(**v31_config["correction"]),
        DEPCarExplicitGearLossConfigV33(**v33_config["selector_loss"]),
    )
    collected = {key: [] for key in ("margin", "teacher", "required", "no_forward", "forward_progress", "feasible")}
    with torch.inference_mode():
        for host in loader:
            batch = trainer.v33.v31.v3.select_valid(host, device)
            if batch is None:
                continue
            output, losses = trainer.v33.forward_loss(model, objective, batch, True)
            collected["margin"].append((output.gear_logits[:, 1] - output.gear_logits[:, 0]).cpu())
            collected["teacher"].append(losses["teacher_reverse"].cpu())
            collected["required"].append(losses["required_reverse"].cpu())
            collected["no_forward"].append(losses["no_hard_forward"].cpu())
            collected["forward_progress"].append(losses["forward_available"].cpu())
            collected["feasible"].append(losses["hard_feasible"].cpu())
    values = {key: torch.cat(parts).numpy().astype(bool if key != "margin" else np.float32)
              for key, parts in collected.items()}
    quantiles = np.linspace(0.0, 1.0, 2001)
    thresholds = np.unique(np.concatenate((np.quantile(values["margin"], quantiles), np.array([0.0]))))
    rows = [metric_row(values["margin"], threshold, values["teacher"], values["required"], values["no_forward"], values["forward_progress"], values["feasible"]) for threshold in thresholds]
    passing = [row for row in rows if passes(row, acceptance)]
    for row in rows:
        row["normalized_gate_violation"] = violation(row, acceptance)
    best = min(rows, key=lambda row: (row["normalized_gate_violation"], abs(row["threshold"])))
    report = {
        "schema": "DEPCarJointGearV331ThresholdTradeoffV1",
        "status": "PASSING_THRESHOLD_FOUND" if passing else "NO_PASSING_SCALAR_THRESHOLD",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": trainer.sha256_file(checkpoint),
        "samples": int(len(values["margin"])),
        "thresholds_evaluated": int(len(rows)),
        "zero_threshold": metric_row(values["margin"], 0.0, values["teacher"], values["required"], values["no_forward"], values["forward_progress"], values["feasible"]),
        "best_threshold": best,
        "passing_thresholds": passing[:20],
        "test_split_accessed": False,
    }
    output = trainer.resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
