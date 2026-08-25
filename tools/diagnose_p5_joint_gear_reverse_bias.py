#!/usr/bin/env python3
"""Sweep a constant reverse-bank score bias on the sealed validation split."""

from __future__ import annotations

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

from dep_car.model.dep_car_net_v3 import DEPCarNetV3
from dep_car.training.losses_v3 import DEPCarJointGearLossConfigV3
from dep_car.training.losses_v31 import (
    DEPCarObjectiveV31,
    DEPCarSequenceCorrectionConfigV31,
)
from dep_car.training.p4_dataset import p3_training_collate, p3_training_worker_init
from dep_car.training.score_dataset import P3JointGearDatasetV3
import train_dep_car_sequence_v32 as trainer


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--output", default="reports/p5_joint_gear_reverse_bias_diagnostic.json")
    value.add_argument("--workers", type=int, default=8)
    value.add_argument("--batch-size", type=int, default=64)
    value.add_argument("--minimum-bias", type=float, default=-0.30)
    value.add_argument("--maximum-bias", type=float, default=0.05)
    value.add_argument("--bias-step", type=float, default=0.0025)
    value.add_argument("--device", default="cuda")
    return value


def main(argv=None):
    args = parser().parse_args(argv)
    if min(args.workers, args.batch_size) < 1 or args.bias_step <= 0.0:
        raise SystemExit("workers, batch size and bias step must be positive")
    if args.minimum_bias >= args.maximum_bias:
        raise SystemExit("bias range is invalid")
    config, _config_sha, v31_config, base_config, acceptance = trainer.load_config()
    _bundle_path, bundle, sequence_path, _authority, data_gate = (
        trainer.v31.v3.verify_data_authority(base_config)
    )
    _source_path, source, source_gate = trainer.verify_source(config, data_gate)
    dataset = P3JointGearDatasetV3(
        bundle["sample_root"], bundle["maps_root"], split="validation",
        index_path=bundle["index"], index_splits=("train", "validation"),
        workers=args.workers,
        expected_map_contract_aggregate_sha256=bundle[
            "map_contract_aggregate_sha256"
        ],
        expected_index_sha256=bundle["index_sha256"], modality="fusion",
        sequence_index_path=sequence_path,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=args.device.startswith("cuda"),
        persistent_workers=args.workers > 0,
        prefetch_factor=int(config["training"]["prefetch_factor"]),
        collate_fn=p3_training_collate, worker_init_fn=p3_training_worker_init,
    )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    model = DEPCarNetV3()
    model.load_state_dict(source["model_state_dict"], strict=True)
    model.to(device).eval()
    objective = DEPCarObjectiveV31(
        DEPCarJointGearLossConfigV3(**base_config["loss"]),
        DEPCarSequenceCorrectionConfigV31(**v31_config["correction"]),
    )
    rows = {
        name: [] for name in (
            "forward_score", "reverse_score", "forward_hard", "reverse_hard",
            "forward_available", "forward_feasible", "reverse_feasible",
            "teacher_reverse",
        )
    }
    amp = bool(config["training"]["mixed_precision"]) and device.type == "cuda"
    with torch.inference_mode():
        for host in loader:
            batch = trainer.v31.v3.select_valid(host, device)
            if batch is None:
                continue
            output, losses = trainer.v31.forward_loss(model, objective, batch, amp)
            scores = output.scores.float()
            feasible = losses["hard_feasible"]
            forward_score, forward_index = scores[:, :15].min(dim=1)
            reverse_score, reverse_index = scores[:, 15:].min(dim=1)
            forward_hard = feasible[:, :15].gather(
                1, forward_index[:, None]
            ).squeeze(1)
            reverse_hard = feasible[:, 15:].gather(
                1, reverse_index[:, None]
            ).squeeze(1)
            effective = losses["candidate_cost"].masked_fill(~feasible, torch.inf)
            teacher_reverse = effective.argmin(dim=1) >= 15
            values = {
                "forward_score": forward_score,
                "reverse_score": reverse_score,
                "forward_hard": forward_hard,
                "reverse_hard": reverse_hard,
                "forward_available": losses["forward_available"],
                "forward_feasible": feasible[:, :15].any(dim=1),
                "reverse_feasible": feasible[:, 15:].any(dim=1),
                "teacher_reverse": teacher_reverse,
            }
            for name, value in values.items():
                rows[name].append(value.cpu())
    rows = {name: torch.cat(values) for name, values in rows.items()}
    required = ~rows["forward_available"] & rows["reverse_feasible"]
    no_hard_forward = required & ~rows["forward_feasible"]
    thresholds = acceptance["sequence"]
    bias_values = np.arange(
        args.minimum_bias,
        args.maximum_bias + 0.5 * args.bias_step,
        args.bias_step,
    )
    results = []
    for bias in bias_values:
        selected_reverse = rows["reverse_score"] + float(bias) < rows["forward_score"]
        selected_hard = torch.where(
            selected_reverse, rows["reverse_hard"], rows["forward_hard"]
        )
        oracle_reverse = required & rows["teacher_reverse"]
        oracle_forward = required & ~rows["teacher_reverse"]
        metrics = {
            "reverse_score_bias": float(bias),
            "selected_hard_feasible_rate": float(selected_hard.float().mean()),
            "unnecessary_reverse_rate": float(
                (selected_reverse & rows["forward_available"]).sum()
            ) / max(1, int(rows["forward_available"].sum())),
            "oracle_reverse_recall_within_required": float(
                (selected_reverse & oracle_reverse).sum()
            ) / max(1, int(oracle_reverse.sum())),
            "oracle_forward_false_reverse_rate_within_required": float(
                (selected_reverse & oracle_forward).sum()
            ) / max(1, int(oracle_forward.sum())),
            "no_hard_forward_reverse_selection_rate": float(
                (selected_reverse & no_hard_forward).sum()
            ) / max(1, int(no_hard_forward.sum())),
        }
        checks = {
            "selected_hard_feasible_rate": metrics["selected_hard_feasible_rate"]
            >= float(thresholds["minimum_selected_hard_feasible_rate"]),
            "unnecessary_reverse_rate": metrics["unnecessary_reverse_rate"]
            <= float(thresholds["maximum_unnecessary_reverse_rate"]),
            "oracle_reverse_recall_within_required": metrics[
                "oracle_reverse_recall_within_required"
            ] >= float(thresholds["minimum_oracle_reverse_recall_within_required"]),
            "oracle_forward_false_reverse_rate_within_required": metrics[
                "oracle_forward_false_reverse_rate_within_required"
            ] <= float(thresholds[
                "maximum_oracle_forward_false_reverse_rate_within_required"
            ]),
            "no_hard_forward_reverse_selection_rate": metrics[
                "no_hard_forward_reverse_selection_rate"
            ] >= float(thresholds[
                "minimum_no_hard_forward_reverse_selection_rate"
            ]),
        }
        results.append({
            **metrics,
            "gate_passed": all(checks.values()),
            "checks": {key: "PASS" if value else "FAIL" for key, value in checks.items()},
        })
    passing = [row for row in results if row["gate_passed"]]
    closest = min(
        results,
        key=lambda row: (
            sum(value == "FAIL" for value in row["checks"].values()),
            abs(row["reverse_score_bias"]),
        ),
    )
    report = {
        "schema": "DEPCarJointGearReverseBiasDiagnosticV1",
        "status": "PASSING_BIAS_EXISTS" if passing else "NO_PASSING_BIAS",
        "samples": len(rows["forward_score"]),
        "source_gate": source_gate,
        "data_authority_gate": data_gate,
        "minimum_bias": args.minimum_bias,
        "maximum_bias": args.maximum_bias,
        "bias_step": args.bias_step,
        "passing_bias_count": len(passing),
        "smallest_absolute_passing_bias": (
            min(passing, key=lambda row: abs(row["reverse_score_bias"]))
            if passing else None
        ),
        "closest": closest,
        "test_split_accessed": False,
        "production_qualified": False,
    }
    output = trainer.resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
