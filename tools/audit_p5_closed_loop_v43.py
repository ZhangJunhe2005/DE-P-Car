#!/usr/bin/env python3
"""Independently audit V4.3 before any P6 shadow rollout."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dep_car/src")); sys.path.insert(0, str(ROOT / "tools"))
from dep_car.model.dep_car_net_v3 import DEPCarNetV3
from dep_car.model.dep_car_net_v43 import DEPCarNetV43
from dep_car.training.losses_v43 import DEPCarObjectiveV43
from dep_car.training.p4_dataset import p3_training_collate, p3_training_worker_init
from dep_car.training.v43_dataset import P3ClosedLoopSequenceDatasetV43
import train_dep_car_closed_loop_v43 as trainer
import train_dep_car_hybrid_sequence_v4 as v4


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--maximum-samples", type=int)
    args = parser.parse_args()
    if args.workers < 1: raise SystemExit("workers must be positive")
    config, config_sha, model_config, rollout_config, loss_config = trainer.load_config()
    authority, data_gate = trainer.verify_data(config); _source_path, _source, source_gate = trainer.verify_source(config)
    checkpoint = trainer.resolve(args.checkpoint); payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    contract_path = checkpoint.with_suffix(".contract.json"); contract = trainer.read_json(contract_path)
    bounded = args.maximum_samples is not None
    expected = trainer.best_path(trainer.resolve(config["artifacts"]["output"]))
    if (
        payload.get("schema") != trainer.CHECKPOINT_SCHEMA
        or payload.get("architecture_id") != DEPCarNetV43.architecture_id
        or payload.get("objective_id") != DEPCarObjectiveV43.objective_id
        or payload.get("training_stage") != trainer.STAGE
        or payload.get("run_completed") is not True
        or payload.get("partial_epoch") is not False
        or payload.get("continuous_sequence_authority") != trainer.SEQUENCE_AUTHORITY
        or payload.get("data_authority_gate", {}).get("authority_sha256") != data_gate["authority_sha256"]
        or contract.get("checkpoint_sha256") != trainer.sha256_file(checkpoint)
        or (not bounded and checkpoint != expected)
    ):
        raise RuntimeError("V4.3 checkpoint/contract identity differs")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA is unavailable")
    model = DEPCarNetV43(
        base_model=DEPCarNetV3(), sequence_config=model_config,
        rollout_config=rollout_config,
        residual_score_span=float(config["model"]["residual_score_span"]),
    )
    model.load_state_dict(payload["model_state_dict"], strict=True); model.freeze_base(); model.to(device).eval()
    dataset = P3ClosedLoopSequenceDatasetV43(
        sample_root=authority["sample_root"], maps_root=authority["maps_root"], split="validation",
        index_path=authority["training_index"], index_splits=("train", "validation"),
        workers=args.workers, expected_map_contract_aggregate_sha256=authority["map_contract_aggregate_sha256"],
        expected_index_sha256=authority["training_index_sha256"], modality="fusion",
        sequence_index_path=authority["sequence_index"], expected_sequence_index_sha256=authority["sequence_index_sha256"],
    )
    if args.maximum_samples:
        from torch.utils.data import Subset
        import numpy as np
        dataset = Subset(dataset, np.linspace(0, len(dataset)-1, min(len(dataset), args.maximum_samples), dtype=np.int64).tolist())
    loader_args = dict(batch_size=int(config["training"]["batch_size"]), shuffle=False, num_workers=args.workers, pin_memory=device.type == "cuda", persistent_workers=args.workers > 0, collate_fn=p3_training_collate, worker_init_fn=p3_training_worker_init)
    if args.workers > 0: loader_args["prefetch_factor"] = int(config["training"]["prefetch_factor"])
    loader = DataLoader(dataset, **loader_args)
    metrics = v4.epoch_loop(model, DEPCarObjectiveV43(loss_config), loader, trainer.STAGE, device, bool(config["training"]["mixed_precision"]) and device.type == "cuda", progress_interval=int(config["training"]["progress_interval_steps"]))
    raw_checks = trainer.checks(metrics, config["qualification"])
    checks = {key: "PASS" if value else "FAIL" for key, value in raw_checks.items()}
    passed = all(raw_checks.values())
    report = {
        "schema": "DEPCarV43AcceptanceV1", "status": "PASS" if passed else "FAIL",
        "gate_passed": passed, "errors": sorted(key for key, value in raw_checks.items() if not value),
        "scope": "P6_SHADOW_ONLY", "checkpoint": str(checkpoint),
        "checkpoint_sha256": trainer.sha256_file(checkpoint),
        "checkpoint_contract": str(contract_path), "checkpoint_contract_sha256": trainer.sha256_file(contract_path),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(), "checks": checks,
        "metrics": metrics, "population_samples": metrics["samples"], "formal_population": not bounded,
        "data_authority_gate": data_gate, "source_gate": source_gate,
        "continuous_sequence_authority": trainer.SEQUENCE_AUTHORITY,
        "closed_loop_gazebo_qualification_pending": True,
        "high_level_gear_state_machine": False, "active_control_authorized": False,
        "production_qualified": False, "test_split_accessed": False,
        "training_config_sha256": config_sha,
    }
    output = checkpoint.with_suffix(".acceptance.json")
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
