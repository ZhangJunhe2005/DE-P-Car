#!/usr/bin/env python3
"""Independently qualify one completed DEPCarNetV3 development stage."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dep_car/src"))
sys.path.insert(0, str(ROOT / "tools"))

from dep_car.model.dep_car_net_v3 import DEPCarNetV3
from dep_car.training.losses_v3 import DEPCarJointGearLossConfigV3, DEPCarObjectiveV3
from dep_car.training.p4_dataset import p3_training_collate, p3_training_worker_init
from dep_car.training.score_dataset import P3JointGearDatasetV3
import train_dep_car_joint_gear_v3 as trainer


ACCEPTANCE_CONTRACT = (
    ROOT / "dep_car/config/p5_joint_gear_v3_acceptance.yaml"
)


def load_acceptance_contract():
    raw = ACCEPTANCE_CONTRACT.read_bytes()
    value = yaml.safe_load(raw)
    if (
        value.get("schema") != "DEPCarJointGearV3AcceptanceContractV2"
        or value.get("revision") != 2
        or value.get("test_split_sealed") is not True
        or value.get("scope", {}).get("production_qualified") is not False
    ):
        raise RuntimeError("V3 acceptance contract is invalid")
    return value, trainer.sha256_file(ACCEPTANCE_CONTRACT)


def oracle_bank_metrics(model, objective, loader, stage, device, amp):
    """Compare learned bank choice with the exact joint-cost teacher.

    ``forward_available`` is only a progress screen; it is not proof that
    reverse is the lower-cost action.  The previous gate treated every frame
    failing that screen as mandatory reverse and therefore had a maximum
    attainable rate equal to the teacher's reverse prevalence.  This audit
    instead measures recall/false-positive rates against the same hard-safe
    counterfactual teacher used by score_ranking_loss.
    """

    names = (
        "required", "oracle_reverse_required", "selected_reverse_required",
        "bank_true_positive_required", "bank_false_negative_required",
        "bank_false_positive_required", "bank_true_negative_required",
        "required_no_hard_forward", "selected_reverse_no_hard_forward",
    )
    totals = {name: 0 for name in names}
    with torch.inference_mode():
        for host in loader:
            batch = trainer.select_valid(host, device)
            if batch is None:
                continue
            output, losses = trainer.forward_loss(
                model, objective, batch, stage, amp
            )
            feasible = losses["hard_feasible"]
            effective = losses["candidate_cost"].masked_fill(
                ~feasible, torch.inf
            )
            oracle_reverse = effective.argmin(dim=1) >= 15
            selected_reverse = output.scores.argmin(dim=1) >= 15
            hard_forward = feasible[:, :15].any(dim=1)
            required = (
                ~losses["forward_available"]
                & feasible[:, 15:].any(dim=1)
            )
            no_hard_forward = required & ~hard_forward
            totals["required"] += int(required.sum())
            totals["oracle_reverse_required"] += int(
                (required & oracle_reverse).sum()
            )
            totals["selected_reverse_required"] += int(
                (required & selected_reverse).sum()
            )
            totals["bank_true_positive_required"] += int(
                (required & selected_reverse & oracle_reverse).sum()
            )
            totals["bank_false_negative_required"] += int(
                (required & ~selected_reverse & oracle_reverse).sum()
            )
            totals["bank_false_positive_required"] += int(
                (required & selected_reverse & ~oracle_reverse).sum()
            )
            totals["bank_true_negative_required"] += int(
                (required & ~selected_reverse & ~oracle_reverse).sum()
            )
            totals["required_no_hard_forward"] += int(no_hard_forward.sum())
            totals["selected_reverse_no_hard_forward"] += int(
                (no_hard_forward & selected_reverse).sum()
            )
    required = max(1, totals["required"])
    oracle_reverse = max(1, totals["oracle_reverse_required"])
    oracle_forward = max(
        1, totals["required"] - totals["oracle_reverse_required"]
    )
    no_hard_forward = max(1, totals["required_no_hard_forward"])
    return {
        **totals,
        "legacy_required_reverse_selection_rate": (
            totals["selected_reverse_required"] / required
        ),
        "oracle_reverse_prevalence_within_required": (
            totals["oracle_reverse_required"] / required
        ),
        "oracle_reverse_recall_within_required": (
            totals["bank_true_positive_required"] / oracle_reverse
        ),
        "oracle_forward_false_reverse_rate_within_required": (
            totals["bank_false_positive_required"] / oracle_forward
        ),
        "no_hard_forward_reverse_selection_rate": (
            totals["selected_reverse_no_hard_forward"] / no_hard_forward
        ),
    }


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--stage", choices=trainer.STAGES, required=True)
    value.add_argument("--checkpoint")
    value.add_argument("--output")
    value.add_argument("--maximum-samples", type=int, default=0)
    value.add_argument("--batch-size", type=int, default=64)
    value.add_argument("--workers", type=int, default=8)
    value.add_argument("--device", default="cuda")
    return value


def main(argv=None):
    args = parser().parse_args(argv)
    if args.maximum_samples < 0 or min(args.batch_size, args.workers) < 1:
        raise SystemExit("sample limit must be non-negative; batch/workers must be positive")
    config, config_sha = trainer.load_config()
    acceptance_contract, acceptance_contract_sha = load_acceptance_contract()
    artifact = trainer.STAGE_ARTIFACT[args.stage]
    expected = trainer._best(trainer.resolve(config["artifacts"][artifact]))
    checkpoint = trainer.resolve(args.checkpoint) if args.checkpoint else expected
    formal = args.maximum_samples == 0
    if formal and checkpoint != expected:
        raise RuntimeError("formal audit requires the canonical best checkpoint")
    if args.output:
        output_path = trainer.resolve(args.output)
    elif formal:
        output_path = checkpoint.with_suffix(".acceptance.json")
    else:
        output_path = ROOT / (
            "reports/p5_joint_gear_v3_%s_smoke_acceptance.json" % artifact
        )
    bundle_path, bundle, sequence_path, _sequence_authority, data_gate = (
        trainer.verify_data_authority(config)
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    contract_path = trainer._verify_checkpoint_contract(checkpoint, payload)
    expected_epochs = int(config["training"][{
        trainer.STAGES[0]: "candidate_epochs",
        trainer.STAGES[1]: "score_epochs",
        trainer.STAGES[2]: "sequence_epochs",
    }[args.stage]])
    errors = []
    required = {
        "schema": "DEPCarJointGearV3CheckpointV1",
        "architecture_id": DEPCarNetV3.architecture_id,
        "objective_id": DEPCarObjectiveV3.objective_id,
        "training_stage": args.stage,
        "modality": "fusion",
        "artifact_role": "best",
        "run_completed": True,
        "partial_epoch": False,
        "training_config_sha256": config_sha,
        "trainer_sha256": trainer.sha256_file(trainer.TRAINER),
        "production_qualified": False,
    }
    errors.extend(
        "checkpoint_" + key
        for key, expected_value in required.items()
        if payload.get(key) != expected_value
    )
    if formal and payload.get("completed_epochs") != expected_epochs:
        errors.append("checkpoint_epochs")
    if payload.get("data_authority_gate") != data_gate:
        errors.append("checkpoint_data_authority")
    if errors:
        raise RuntimeError("V3 artifact identity failed: " + ",".join(errors))

    dataset = P3JointGearDatasetV3(
        bundle["sample_root"], bundle["maps_root"], split="validation",
        index_path=bundle["index"], index_splits=("train", "validation"),
        workers=args.workers,
        expected_map_contract_aggregate_sha256=bundle["map_contract_aggregate_sha256"],
        expected_index_sha256=bundle["index_sha256"], modality="fusion",
        sequence_index_path=sequence_path,
    )
    if args.maximum_samples and len(dataset) > args.maximum_samples:
        selected = np.linspace(
            0, len(dataset) - 1, args.maximum_samples, dtype=np.int64
        )
        dataset = Subset(dataset, selected.tolist())
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=args.device.startswith("cuda"),
        persistent_workers=args.workers > 0,
        prefetch_factor=int(config["training"]["prefetch_factor"])
        if args.workers > 0 else None,
        collate_fn=p3_training_collate, worker_init_fn=p3_training_worker_init,
    )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    model = DEPCarNetV3()
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device).eval()
    objective = DEPCarObjectiveV3(
        DEPCarJointGearLossConfigV3(**config["loss"])
    )
    metrics = trainer.epoch_loop(
        model, objective, loader, args.stage, device,
        bool(config["training"]["mixed_precision"]) and device.type == "cuda",
        progress_interval=int(config["training"]["progress_interval_steps"]),
    )
    thresholds = acceptance_contract[
        "candidate" if args.stage == trainer.STAGES[0]
        else "score" if args.stage == trainer.STAGES[1]
        else "sequence"
    ]
    checks = {
        "overall_zero_hard_feasible_rate": (
            metrics["zero_hard_feasible_rate"]
            <= float(acceptance_contract["candidate"]["maximum_zero_hard_feasible_rate"])
        ),
        "forward_bank_capable_rate": metrics["forward_bank_capable_rate"]
        >= float(acceptance_contract["candidate"]["minimum_forward_bank_capable_rate"]),
        "reverse_bank_capable_rate": metrics["reverse_bank_capable_rate"]
        >= float(acceptance_contract["candidate"]["minimum_reverse_bank_capable_rate"]),
    }
    if args.stage != trainer.STAGES[0]:
        oracle_metrics = oracle_bank_metrics(
            model, objective, loader, args.stage, device,
            bool(config["training"]["mixed_precision"])
            and device.type == "cuda",
        )
        metrics.update(oracle_metrics)
        checks.update({
            "selected_hard_feasible_rate": (
                metrics["selected_hard_feasible_rate"]
                >= float(thresholds["minimum_selected_hard_feasible_rate"])
            ),
            "unnecessary_reverse_rate": (
                metrics["unnecessary_reverse_rate"]
                <= float(thresholds["maximum_unnecessary_reverse_rate"])
            ),
            "oracle_reverse_recall_within_required": (
                metrics["oracle_reverse_recall_within_required"]
                >= float(thresholds[
                    "minimum_oracle_reverse_recall_within_required"
                ])
            ),
            "oracle_forward_false_reverse_rate_within_required": (
                metrics["oracle_forward_false_reverse_rate_within_required"]
                <= float(thresholds[
                    "maximum_oracle_forward_false_reverse_rate_within_required"
                ])
            ),
            "no_hard_forward_reverse_selection_rate": (
                metrics["no_hard_forward_reverse_selection_rate"]
                >= float(thresholds[
                    "minimum_no_hard_forward_reverse_selection_rate"
                ])
            ),
        })
    passed = all(checks.values())
    report = {
        "schema": "DEPCarJointGearV3AcceptanceV1",
        "status": "PASS" if passed else "FAIL",
        "gate_passed": passed,
        "errors": [] if passed else sorted(key for key, value in checks.items() if not value),
        "stage": args.stage,
        "scope": (
            acceptance_contract["scope"]["sequence"]
            if args.stage == trainer.STAGES[2]
            else acceptance_contract["scope"]["score"]
            if args.stage == trainer.STAGES[1]
            else "NEXT_TRAINING_STAGE_ONLY"
        ),
        "acceptance_contract": str(ACCEPTANCE_CONTRACT),
        "acceptance_contract_sha256": acceptance_contract_sha,
        "acceptance_contract_schema": acceptance_contract["schema"],
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": trainer.sha256_file(checkpoint),
        "checkpoint_contract": str(contract_path),
        "checkpoint_contract_sha256": trainer.sha256_file(contract_path),
        "data_authority_gate": data_gate,
        "metrics": metrics,
        "checks": {key: "PASS" if value else "FAIL" for key, value in checks.items()},
        "formal_population": formal,
        "population_samples": metrics["samples"],
        "test_split_accessed": False,
        "active_control_authorized": False,
        "production_qualified": False,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
