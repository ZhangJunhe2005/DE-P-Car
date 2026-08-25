#!/usr/bin/env python3
"""Independently audit a V4.1 hierarchical score checkpoint."""

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

from dep_car.model.dep_car_net_v4 import DEPCarNetV4
from dep_car.training.losses_v41 import DEPCarObjectiveV41
from dep_car.training.p4_dataset import p3_training_collate, p3_training_worker_init
from dep_car.training.v4_dataset import P3HybridSequenceDatasetV4
import train_dep_car_hybrid_sequence_v4 as v4
import train_dep_car_hybrid_sequence_v41 as trainer


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output")
    parser.add_argument("--maximum-samples", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.maximum_samples < 0 or min(args.batch_size, args.workers) < 1:
        raise SystemExit("V4.1 audit sample limit/batch/workers are invalid")
    config, config_sha, model_config, rollout_config, loss_config = trainer.load_config()
    _bundle_path, bundle, sequence_path, _authority, data_gate = trainer.verify_data_authority(config)
    source_path, source, source_gate = trainer.verify_source(config)
    checkpoint = trainer.resolve(args.checkpoint)
    canonical = trainer.best_path(trainer.resolve(config["artifacts"]["output"]))
    formal = args.maximum_samples == 0
    if formal and checkpoint != canonical:
        raise RuntimeError("formal V4.1 audit requires the canonical best checkpoint")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    contract_path = checkpoint.with_suffix(".contract.json")
    contract = trainer.read_json(contract_path)
    required = {
        "schema": trainer.CHECKPOINT_SCHEMA,
        "architecture_id": DEPCarNetV4.architecture_id,
        "objective_id": DEPCarObjectiveV41.objective_id,
        "training_stage": trainer.STAGE, "artifact_role": "best",
        "partial_epoch": False, "source_checkpoint": str(source_path),
        "source_checkpoint_sha256": trainer.sha256_file(source_path),
        "data_authority_gate": data_gate, "source_gate": source_gate,
        "training_config_sha256": config_sha,
        "trainer_sha256": trainer.sha256_file(trainer.TRAINER),
        "model_implementation_sha256": config["implementation"]["model_sha256"],
        "rollout_implementation_sha256": config["implementation"]["rollout_sha256"],
        "loss_implementation_sha256": config["implementation"]["loss_sha256"],
        "unified_hybrid_sequence": True,
        "hierarchical_score_correction": True,
        "high_level_gear_state_machine": False,
        "active_control_authorized": False, "production_qualified": False,
    }
    errors = [
        "checkpoint_" + key for key, expected in required.items()
        if payload.get(key) != expected
    ]
    if formal and (
        payload.get("completed_epochs") != int(config["training"]["epochs"])
        or payload.get("run_completed") is not True
    ):
        errors.append("checkpoint_completion")
    if (
        contract.get("schema") != trainer.CONTRACT_SCHEMA
        or contract.get("checkpoint_sha256") != trainer.sha256_file(checkpoint)
        or contract.get("run_completed") is not True
        or contract.get("active_control_authorized") is not False
        or contract.get("production_qualified") is not False
    ):
        errors.append("checkpoint_contract")
    if errors:
        raise RuntimeError("V4.1 artifact identity failed: " + ",".join(errors))
    dataset = P3HybridSequenceDatasetV4(
        sample_root=bundle["sample_root"], maps_root=bundle["maps_root"],
        split="validation", index_path=bundle["index"],
        index_splits=("train", "validation"), workers=args.workers,
        expected_map_contract_aggregate_sha256=bundle["map_contract_aggregate_sha256"],
        expected_index_sha256=bundle["index_sha256"], modality="fusion",
        sequence_index_path=sequence_path,
    )
    if args.maximum_samples and len(dataset) > args.maximum_samples:
        dataset = Subset(dataset, np.linspace(
            0, len(dataset) - 1, args.maximum_samples, dtype=np.int64
        ).tolist())
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
    model = v4.build_model(model_config, rollout_config, source)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.freeze_base(); model.to(device).eval()
    objective = DEPCarObjectiveV41(loss_config)
    metrics = v4.epoch_loop(
        model, objective, loader, trainer.STAGE, device,
        bool(config["training"]["mixed_precision"]) and device.type == "cuda",
        progress_interval=int(config["training"]["progress_interval_steps"]),
    )
    raw_checks = trainer.checks(metrics, config["qualification"])
    passed = all(raw_checks.values())
    report = {
        "schema": "DEPCarV41ScoreAcceptanceV1",
        "status": "PASS" if passed else "FAIL" if formal else "DIAGNOSTIC_COMPLETE",
        "gate_passed": passed,
        "errors": [] if passed else sorted(
            key for key, value in raw_checks.items() if not value
        ),
        "stage": trainer.STAGE,
        "scope": "NEXT_V41_CLOSED_LOOP_STAGE_ONLY" if formal
        else "BOUNDED_GPU_DIAGNOSTIC_ONLY",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": trainer.sha256_file(checkpoint),
        "checkpoint_contract": str(contract_path),
        "checkpoint_contract_sha256": trainer.sha256_file(contract_path),
        "source_gate": source_gate, "data_authority_gate": data_gate,
        "metrics": metrics,
        "checks": {
            key: "PASS" if value else "FAIL" for key, value in raw_checks.items()
        },
        "formal_population": formal, "population_samples": metrics["samples"],
        "candidate_generator_frozen": True,
        "joint_safety_viability_score_training": True,
        "unified_hybrid_sequence": True,
        "high_level_gear_state_machine": False,
        "test_split_accessed": False, "active_control_authorized": False,
        "production_qualified": False,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    output = trainer.resolve(args.output) if args.output else (
        checkpoint.with_suffix(".acceptance.json") if formal
        else ROOT / "reports/p5_hybrid_sequence_v41_pilot_acceptance.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if passed or not formal else 1


if __name__ == "__main__":
    raise SystemExit(main())
