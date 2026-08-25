#!/usr/bin/env python3
"""Independently audit the V3.2 runtime-winner sequence checkpoint."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dep_car/src"))
sys.path.insert(0, str(ROOT / "tools"))

from dep_car.model.dep_car_net_v3 import DEPCarNetV3
from dep_car.training.losses_v3 import DEPCarJointGearLossConfigV3
from dep_car.training.losses_v31 import DEPCarSequenceCorrectionConfigV31
from dep_car.training.losses_v32 import (
    DEPCarObjectiveV32,
    DEPCarWinnerCorrectionConfigV32,
)
from dep_car.training.p4_dataset import p3_training_collate, p3_training_worker_init
from dep_car.training.score_dataset import P3JointGearDatasetV3
import train_dep_car_sequence_v32 as trainer


def parser():
    value = argparse.ArgumentParser(description=__doc__)
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
    config, config_sha, v31_config, base_config, acceptance_contract = (
        trainer.load_config()
    )
    expected = trainer._best(trainer.resolve(config["artifact"]["output"]))
    checkpoint = trainer.resolve(args.checkpoint) if args.checkpoint else expected
    formal = args.maximum_samples == 0
    if formal and checkpoint != expected:
        raise RuntimeError("formal V3.2 audit requires the canonical best checkpoint")
    if args.output:
        output_path = trainer.resolve(args.output)
    elif formal:
        output_path = checkpoint.with_suffix(".acceptance.json")
    else:
        output_path = ROOT / "reports/p5_joint_gear_v32_pilot_acceptance.json"

    _bundle_path, bundle, sequence_path, _authority, data_gate = (
        trainer.v31.v3.verify_data_authority(base_config)
    )
    source_path, _source, source_gate = trainer.verify_source(config, data_gate)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    contract_path = checkpoint.with_suffix(".contract.json")
    if not contract_path.is_file():
        raise RuntimeError("V3.2 checkpoint contract is missing")
    contract = trainer.read_json(contract_path)
    errors = []
    required = {
        "schema": trainer.CHECKPOINT_SCHEMA,
        "architecture_id": DEPCarNetV3.architecture_id,
        "objective_id": DEPCarObjectiveV32.objective_id,
        "training_stage": trainer.STAGE,
        "modality": "fusion",
        "artifact_role": "best",
        "run_completed": True,
        "partial_epoch": False,
        "source_checkpoint": str(source_path),
        "source_checkpoint_sha256": trainer.sha256_file(source_path),
        "training_config_sha256": config_sha,
        "trainer_sha256": trainer.sha256_file(trainer.TRAINER),
        "acceptance_contract_sha256": config["base_contract"][
            "acceptance_contract_sha256"
        ],
        "active_control_authorized": False,
        "production_qualified": False,
    }
    errors.extend(
        "checkpoint_" + key
        for key, expected_value in required.items()
        if payload.get(key) != expected_value
    )
    if formal and payload.get("completed_epochs") != int(config["training"]["epochs"]):
        errors.append("checkpoint_epochs")
    if payload.get("data_authority_gate") != data_gate:
        errors.append("checkpoint_data_authority")
    if payload.get("source_gate") != source_gate:
        errors.append("checkpoint_source_authority")
    if (
        contract.get("schema") != trainer.CONTRACT_SCHEMA
        or contract.get("checkpoint_sha256") != trainer.sha256_file(checkpoint)
        or contract.get("run_completed") is not True
        or contract.get("production_qualified") is not False
    ):
        errors.append("checkpoint_contract")
    if errors:
        raise RuntimeError("V3.2 artifact identity failed: " + ",".join(errors))

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
    objective = DEPCarObjectiveV32(
        DEPCarJointGearLossConfigV3(**base_config["loss"]),
        DEPCarSequenceCorrectionConfigV31(**v31_config["correction"]),
        DEPCarWinnerCorrectionConfigV32(**config["winner_correction"]),
    )
    metrics = trainer.v31.epoch_loop(
        model, objective, loader, device,
        bool(config["training"]["mixed_precision"]) and device.type == "cuda",
        progress_interval=int(config["training"]["progress_interval_steps"]),
    )
    checks = trainer.v31.sequence_acceptance_checks(metrics, acceptance_contract)
    passed = all(checks.values())
    report = {
        "schema": "DEPCarJointGearV32AcceptanceV1",
        "status": (
            "PASS" if passed else "FAIL" if formal else "DIAGNOSTIC_COMPLETE"
        ),
        "gate_passed": passed,
        "errors": [] if passed else sorted(
            key for key, value in checks.items() if not value
        ),
        "stage": trainer.STAGE,
        "scope": "P6_SHADOW_ONLY" if formal else "BOUNDED_GPU_DIAGNOSTIC_ONLY",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": trainer.sha256_file(checkpoint),
        "checkpoint_contract": str(contract_path),
        "checkpoint_contract_sha256": trainer.sha256_file(contract_path),
        "source_gate": source_gate,
        "data_authority_gate": data_gate,
        "training_config": str(trainer.CONFIG),
        "training_config_sha256": config_sha,
        "trainer_sha256": trainer.sha256_file(trainer.TRAINER),
        "acceptance_contract": str(
            trainer.resolve(config["base_contract"]["acceptance_contract"])
        ),
        "acceptance_contract_sha256": config["base_contract"][
            "acceptance_contract_sha256"
        ],
        "metrics": metrics,
        "checks": {
            key: "PASS" if value else "FAIL" for key, value in checks.items()
        },
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
    return 0 if passed or not formal else 1


if __name__ == "__main__":
    raise SystemExit(main())
