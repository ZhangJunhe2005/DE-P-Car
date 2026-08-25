#!/usr/bin/env python3
"""Train V3.2 runtime-winner-aligned sequence correction."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import yaml


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
import train_dep_car_sequence_v31 as v31


CONFIG = ROOT / "dep_car/config/p5_joint_gear_v32_winner_correction.yaml"
TRAINER = Path(__file__).resolve()
STAGE = "winner_sequence_correction"
CHECKPOINT_SCHEMA = "DEPCarJointGearV32CheckpointV1"
CONTRACT_SCHEMA = "DEPCarJointGearV32ArtifactContractV1"


resolve = v31.resolve
sha256_file = v31.sha256_file
read_json = v31.read_json
_best = v31._best


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--output")
    value.add_argument("--epochs", type=int)
    value.add_argument("--batch-size", type=int)
    value.add_argument("--workers", type=int)
    value.add_argument("--max-samples", type=int)
    value.add_argument("--max-steps", type=int)
    value.add_argument("--device", default="cuda")
    value.add_argument("--resume")
    value.add_argument("--dry-run", action="store_true")
    return value


def load_config():
    raw = CONFIG.read_bytes()
    config = yaml.safe_load(raw)
    if (
        config.get("schema") != "DEPCarJointGearV32TrainingContractV1"
        or config.get("architecture_id") != DEPCarNetV3.architecture_id
        or config.get("objective_id") != DEPCarObjectiveV32.objective_id
        or config.get("scope") != "fusion_only_runtime_winner_sequence_correction"
        or config.get("test_split_sealed") is not True
        or config.get("qualification", {}).get(
            "selection_uses_runtime_argmin_metrics"
        )
        is not True
        or config.get("qualification", {}).get(
            "active_control_authorized_by_training"
        )
        is not False
        or config.get("qualification", {}).get(
            "production_qualified_by_training"
        )
        is not False
    ):
        raise RuntimeError("V3.2 training contract identity is invalid")
    try:
        DEPCarWinnerCorrectionConfigV32(**config["winner_correction"]).validate()
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("V3.2 winner correction contract is invalid") from exc
    base_contract = config["base_contract"]
    frozen = (
        (base_contract["training_config"], base_contract["training_config_sha256"]),
        (base_contract["trainer"], base_contract["trainer_sha256"]),
        (base_contract["loss"], base_contract["loss_sha256"]),
        (
            base_contract["acceptance_contract"],
            base_contract["acceptance_contract_sha256"],
        ),
    )
    mismatch = [
        str(resolve(path))
        for path, expected in frozen
        if sha256_file(resolve(path)) != expected
    ]
    if mismatch:
        raise RuntimeError("V3.2 frozen V3.1 contract differs: " + ",".join(mismatch))
    v31_config, v31_sha, base_config, acceptance_contract = v31.load_config()
    if v31_sha != base_contract["training_config_sha256"]:
        raise RuntimeError("V3.2 did not load the exact V3.1 contract")
    return (
        config,
        hashlib.sha256(raw).hexdigest(),
        v31_config,
        base_config,
        acceptance_contract,
    )


def verify_source(config, data_gate):
    source = config["initialization"]
    path = resolve(source["checkpoint"])
    contract_path = resolve(source["checkpoint_contract"])
    acceptance_path = resolve(source["acceptance"])
    identities = (
        (sha256_file(path), source["checkpoint_sha256"]),
        (sha256_file(contract_path), source["checkpoint_contract_sha256"]),
        (sha256_file(acceptance_path), source["acceptance_sha256"]),
    )
    if any(actual != expected for actual, expected in identities):
        raise RuntimeError("V3.2 V3.1 source hash differs")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    contract = read_json(contract_path)
    acceptance = read_json(acceptance_path)
    base_contract = config["base_contract"]
    required_payload = {
        "schema": v31.CHECKPOINT_SCHEMA,
        "architecture_id": DEPCarNetV3.architecture_id,
        "objective_id": v31.DEPCarObjectiveV31.objective_id,
        "training_stage": v31.STAGE,
        "modality": "fusion",
        "artifact_role": "best",
        "run_completed": True,
        "partial_epoch": False,
        "training_config_sha256": base_contract["training_config_sha256"],
        "trainer_sha256": base_contract["trainer_sha256"],
        "active_control_authorized": False,
        "production_qualified": False,
    }
    mismatch = [
        key for key, expected in required_payload.items()
        if payload.get(key) != expected
    ]
    if payload.get("data_authority_gate") != data_gate:
        mismatch.append("data_authority_gate")
    allowed = sorted(source["allowed_failed_checks"])
    actual_failed = sorted(acceptance.get("errors", ()))
    passed_checks = {
        key for key, value in acceptance.get("checks", {}).items()
        if value == "PASS"
    }
    if (
        contract.get("schema") != v31.CONTRACT_SCHEMA
        or contract.get("checkpoint_sha256") != source["checkpoint_sha256"]
        or acceptance.get("schema") != "DEPCarJointGearV31AcceptanceV1"
        or acceptance.get("stage") != v31.STAGE
        or acceptance.get("status") != "FAIL"
        or acceptance.get("gate_passed") is not False
        or actual_failed != allowed
        or any(name in passed_checks for name in allowed)
        or acceptance.get("checkpoint_sha256") != source["checkpoint_sha256"]
        or acceptance.get("formal_population") is not True
        or acceptance.get("test_split_accessed") is not False
        or acceptance.get("active_control_authorized") is not False
        or acceptance.get("production_qualified") is not False
    ):
        mismatch.append("bounded_failed_source_authority")
    if mismatch:
        raise RuntimeError("V3.2 V3.1 source authority failed: " + ",".join(mismatch))
    gate = {
        "schema": "DEPCarJointGearV32SourceGateV1",
        "passed": True,
        "errors": [],
        "checkpoint": str(path),
        "checkpoint_sha256": sha256_file(path),
        "checkpoint_contract": str(contract_path),
        "checkpoint_contract_sha256": sha256_file(contract_path),
        "acceptance": str(acceptance_path),
        "acceptance_sha256": sha256_file(acceptance_path),
        "accepted_for_correction_continuation_only": True,
        "allowed_failed_checks": allowed,
        "p6_authorized_by_source": False,
        "test_split_accessed": False,
    }
    return path, payload, gate


def atomic_torch(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def configure_gear_specific(model):
    """Train only the V3 modules that express contextual gear preference."""

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    selected = tuple(model.transition_encoder.parameters()) + tuple(
        model.joint_energy_head.parameters()
    )
    for parameter in selected:
        parameter.requires_grad_(True)
    selected_ids = {id(parameter) for parameter in selected}
    if len(selected_ids) != len(selected):
        raise RuntimeError("V3.2 gear-specific parameter partition overlaps")
    return {
        "stage": STAGE,
        "candidate_trainable": 0,
        "generic_score_trainable": 0,
        "gear_specific_trainable": sum(
            parameter.numel() for parameter in selected
        ),
        "frozen_parameters": sum(
            parameter.numel() for parameter in model.parameters()
            if id(parameter) not in selected_ids
        ),
    }, selected


def write_artifact(path, model, optimizer, scaler, metadata, metrics, history):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": CHECKPOINT_SCHEMA,
        "architecture_id": DEPCarNetV3.architecture_id,
        "objective_id": DEPCarObjectiveV32.objective_id,
        "training_stage": STAGE,
        "modality": "fusion",
        "artifact_role": metadata["artifact_role"],
        "status": "TRAINED_UNQUALIFIED",
        "qualification_status": "UNQUALIFIED",
        "active_control_authorized": False,
        "production_qualified": False,
        "completed_epochs": metadata["completed_epochs"],
        "selected_epoch": metadata["selected_epoch"],
        "partial_epoch": False,
        "global_step": metadata["global_step"],
        "source_checkpoint": metadata["source_checkpoint"],
        "source_checkpoint_sha256": metadata["source_checkpoint_sha256"],
        "data_authority_gate": metadata["data_authority_gate"],
        "source_gate": metadata["source_gate"],
        "training_config_sha256": metadata["training_config_sha256"],
        "trainer_sha256": metadata["trainer_sha256"],
        "acceptance_contract_sha256": metadata["acceptance_contract_sha256"],
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "grad_scaler_state_dict": scaler.state_dict(),
        "metrics": metrics,
        "selection_gate": metadata["selection_gate"],
        "history": history,
    }
    atomic_torch(path, payload)
    contract = {
        key: payload[key]
        for key in (
            "architecture_id", "objective_id", "training_stage", "modality",
            "artifact_role", "status", "qualification_status",
            "active_control_authorized", "production_qualified",
            "completed_epochs", "selected_epoch", "partial_epoch",
            "global_step", "source_checkpoint", "source_checkpoint_sha256",
            "data_authority_gate", "source_gate", "training_config_sha256",
            "trainer_sha256", "acceptance_contract_sha256", "metrics",
            "selection_gate",
        )
    }
    contract.update({
        "schema": CONTRACT_SCHEMA,
        "checkpoint": str(path),
        "checkpoint_sha256": sha256_file(path),
    })
    path.with_suffix(".contract.json").write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def finalize_best(path, completed_epochs, global_step, history):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload.update({
        "completed_epochs": completed_epochs,
        "global_step": global_step,
        "run_completed": True,
        "partial_epoch": False,
        "history": history,
    })
    atomic_torch(path, payload)
    contract_path = path.with_suffix(".contract.json")
    contract = read_json(contract_path)
    contract.update({
        "checkpoint_sha256": sha256_file(path),
        "completed_epochs": completed_epochs,
        "global_step": global_step,
        "run_completed": True,
        "partial_epoch": False,
    })
    contract_path.write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main(argv=None):
    args = parser().parse_args(argv)
    config, config_sha, v31_config, base_config, acceptance_contract = load_config()
    training = config["training"]
    epochs = int(training["epochs"]) if args.epochs is None else args.epochs
    batch_size = int(training["batch_size"]) if args.batch_size is None else args.batch_size
    workers = int(training["workers"]) if args.workers is None else args.workers
    if min(epochs, batch_size, workers) < 1:
        raise SystemExit("epochs, batch size, and workers must be positive")
    if args.max_samples is not None and not 1 <= args.max_samples <= 1024:
        raise SystemExit("bounded max-samples must be in [1,1024]")
    if args.max_steps is not None and not 1 <= args.max_steps <= 64:
        raise SystemExit("bounded max-steps must be in [1,64]")

    _bundle_path, bundle, sequence_path, _authority, data_gate = (
        v31.v3.verify_data_authority(base_config)
    )
    source_path, source, source_gate = verify_source(config, data_gate)
    expected_output = resolve(config["artifact"]["output"])
    output = resolve(args.output) if args.output else expected_output
    bounded = args.max_samples is not None or args.max_steps is not None
    formal_parameters = (
        epochs == int(training["epochs"])
        and batch_size == int(training["batch_size"])
        and workers == int(training["workers"])
    )
    formal = bool(
        not bounded and formal_parameters and args.device == "cuda"
        and output == expected_output and data_gate["passed"]
        and source_gate["passed"]
    )
    if not args.dry_run and not formal and not bounded:
        raise RuntimeError("V3.2 run is neither formal nor a bounded diagnostic")
    if output == source_path:
        raise RuntimeError("V3.2 source and output must differ")
    if not args.dry_run and output.exists() and args.resume is None:
        raise RuntimeError("V3.2 output exists; use --resume with the canonical last artifact")
    if args.resume is not None and resolve(args.resume) != output:
        raise RuntimeError("V3.2 resume must point to the selected output")

    plan = {
        "schema": "DEPCarJointGearV32TrainingPlanV1",
        "status": (
            "DRY_RUN_READY" if args.dry_run and formal
            else "BOUNDED_DIAGNOSTIC_READY" if bounded else "READY"
        ),
        "stage": STAGE,
        "architecture_id": DEPCarNetV3.architecture_id,
        "objective_id": DEPCarObjectiveV32.objective_id,
        "source": str(source_path),
        "source_sha256": sha256_file(source_path),
        "source_scope": "CORRECTION_CONTINUATION_ONLY",
        "output": str(output),
        "epochs": epochs,
        "batch_size": batch_size,
        "workers": workers,
        "device": args.device,
        "bounded_smoke": bounded,
        "maximum_samples": args.max_samples,
        "maximum_steps": args.max_steps,
        "formal_training_authorized": formal,
        "data_authority_gate": data_gate,
        "source_gate": source_gate,
        "training_config": str(CONFIG),
        "training_config_sha256": config_sha,
        "trainer_sha256": sha256_file(TRAINER),
        "acceptance_contract": str(resolve(config["base_contract"]["acceptance_contract"])),
        "acceptance_contract_sha256": config["base_contract"]["acceptance_contract_sha256"],
        "selection_uses_runtime_argmin_metrics": True,
        "test_split_sealed": True,
        "active_control_authorized": False,
        "production_qualified": False,
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    random.seed(int(training["seed"]))
    np.random.seed(int(training["seed"]))
    torch.manual_seed(int(training["seed"]))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(training["seed"]))
    torch.set_num_threads(int(training["torch_threads"]))
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    model = DEPCarNetV3()
    model.load_state_dict(source["model_state_dict"], strict=True)
    model.to(device)
    ownership, selected_parameters = configure_gear_specific(model)
    optimizer = torch.optim.AdamW(
        selected_parameters,
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    amp = bool(training["mixed_precision"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    common = dict(
        sample_root=bundle["sample_root"], maps_root=bundle["maps_root"],
        index_path=bundle["index"], index_splits=("train", "validation"),
        workers=workers,
        expected_map_contract_aggregate_sha256=bundle[
            "map_contract_aggregate_sha256"
        ],
        expected_index_sha256=bundle["index_sha256"], modality="fusion",
        sequence_index_path=sequence_path,
    )
    train_data = P3JointGearDatasetV3(split="train", **common)
    validation_data = P3JointGearDatasetV3(split="validation", **common)
    if args.max_samples:
        train_selected = np.linspace(
            0, len(train_data) - 1,
            min(len(train_data), args.max_samples), dtype=np.int64,
        )
        validation_selected = np.linspace(
            0, len(validation_data) - 1,
            min(len(validation_data), args.max_samples), dtype=np.int64,
        )
        train_data = Subset(train_data, train_selected.tolist())
        validation_data = Subset(validation_data, validation_selected.tolist())
    loader_common = dict(
        batch_size=batch_size, num_workers=workers,
        pin_memory=device.type == "cuda", persistent_workers=workers > 0,
        prefetch_factor=int(training["prefetch_factor"]) if workers > 0 else None,
        collate_fn=p3_training_collate, worker_init_fn=p3_training_worker_init,
    )
    train_loader = DataLoader(train_data, shuffle=True, **loader_common)
    validation_loader = DataLoader(validation_data, shuffle=False, **loader_common)
    objective = DEPCarObjectiveV32(
        DEPCarJointGearLossConfigV3(**base_config["loss"]),
        DEPCarSequenceCorrectionConfigV31(**v31_config["correction"]),
        DEPCarWinnerCorrectionConfigV32(**config["winner_correction"]),
    )

    history, global_step, best_key, start_epoch = [], 0, None, 1
    if args.resume:
        resumed = torch.load(output, map_location="cpu", weights_only=True)
        required = {
            "schema": CHECKPOINT_SCHEMA,
            "architecture_id": DEPCarNetV3.architecture_id,
            "objective_id": DEPCarObjectiveV32.objective_id,
            "training_stage": STAGE,
            "artifact_role": "last",
            "source_checkpoint_sha256": plan["source_sha256"],
            "training_config_sha256": config_sha,
            "trainer_sha256": plan["trainer_sha256"],
        }
        mismatch = [
            key for key, expected in required.items()
            if resumed.get(key) != expected
        ]
        if mismatch:
            raise RuntimeError("V3.2 resume mismatch: " + ",".join(mismatch))
        completed = int(resumed.get("completed_epochs", 0))
        if not 1 <= completed < epochs:
            raise RuntimeError("V3.2 resume epoch is outside remaining run")
        model.load_state_dict(resumed["model_state_dict"], strict=True)
        optimizer.load_state_dict(resumed["optimizer_state_dict"])
        scaler.load_state_dict(resumed["grad_scaler_state_dict"])
        history = list(resumed.get("history", ()))
        global_step = int(resumed.get("global_step", 0))
        start_epoch = completed + 1
        keys = [
            v31.sequence_selection_key(row["validation"], acceptance_contract)
            for row in history
        ]
        best_key = min(keys) if keys else None
        model.to(device)

    if not args.resume:
        baseline_metrics = v31.epoch_loop(
            model, objective, validation_loader, device, amp,
            max_steps=args.max_steps,
            progress_interval=int(training["progress_interval_steps"]),
        )
        baseline_gate = v31.sequence_acceptance_checks(
            baseline_metrics, acceptance_contract
        )
        baseline_row = {
            "epoch": 0,
            "phase": "v31_failed_checkpoint_correction_baseline",
            "train": None,
            "validation": baseline_metrics,
            "selection_gate": {
                key: "PASS" if value else "FAIL"
                for key, value in baseline_gate.items()
            },
            "elapsed_s": baseline_metrics["elapsed_s"],
        }
        history.append(baseline_row)
        best_key = v31.sequence_selection_key(
            baseline_metrics, acceptance_contract
        )
        metadata = {
            "artifact_role": "best", "completed_epochs": 0,
            "selected_epoch": 0, "global_step": 0,
            "source_checkpoint": str(source_path),
            "source_checkpoint_sha256": plan["source_sha256"],
            "data_authority_gate": data_gate, "source_gate": source_gate,
            "training_config_sha256": config_sha,
            "trainer_sha256": plan["trainer_sha256"],
            "acceptance_contract_sha256": plan["acceptance_contract_sha256"],
            "selection_gate": baseline_row["selection_gate"],
        }
        write_artifact(
            _best(output), model, optimizer, scaler, metadata,
            baseline_metrics, history,
        )
        print(json.dumps(baseline_row, sort_keys=True), flush=True)

    for epoch in range(start_epoch, epochs + 1):
        started = time.time()
        train_metrics = v31.epoch_loop(
            model, objective, train_loader, device, amp,
            optimizer=optimizer, scaler=scaler, max_steps=args.max_steps,
            dropout=float(training["sensor_dropout_probability"]),
            progress_interval=int(training["progress_interval_steps"]),
            gradient_clip=float(training["gradient_clip"]),
        )
        validation_metrics = v31.epoch_loop(
            model, objective, validation_loader, device, amp,
            max_steps=args.max_steps,
            progress_interval=int(training["progress_interval_steps"]),
        )
        global_step += train_metrics["steps"]
        selection_gate = v31.sequence_acceptance_checks(
            validation_metrics, acceptance_contract
        )
        row = {
            "epoch": epoch,
            "train": train_metrics,
            "validation": validation_metrics,
            "selection_gate": {
                key: "PASS" if value else "FAIL"
                for key, value in selection_gate.items()
            },
            "elapsed_s": time.time() - started,
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        key = v31.sequence_selection_key(validation_metrics, acceptance_contract)
        metadata = {
            "artifact_role": "last", "completed_epochs": epoch,
            "selected_epoch": epoch, "global_step": global_step,
            "source_checkpoint": str(source_path),
            "source_checkpoint_sha256": plan["source_sha256"],
            "data_authority_gate": data_gate, "source_gate": source_gate,
            "training_config_sha256": config_sha,
            "trainer_sha256": plan["trainer_sha256"],
            "acceptance_contract_sha256": plan["acceptance_contract_sha256"],
            "selection_gate": row["selection_gate"],
        }
        write_artifact(
            output, model, optimizer, scaler, metadata,
            validation_metrics, history,
        )
        if best_key is None or key < best_key:
            best_key = key
            metadata["artifact_role"] = "best"
            write_artifact(
                _best(output), model, optimizer, scaler, metadata,
                validation_metrics, history,
            )

    best_path = _best(output)
    if not best_path.is_file():
        raise RuntimeError("V3.2 training ended without best checkpoint")
    finalize_best(best_path, epochs, global_step, history)
    print(json.dumps({
        **plan,
        "status": "COMPLETE",
        "global_step": global_step,
        "stage_ownership": ownership,
        "best_checkpoint": str(best_path),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
