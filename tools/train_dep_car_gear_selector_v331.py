#!/usr/bin/env python3
"""Fine-tune V3.3 on frames where exactly one gear bank is hard-safe."""

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

from dep_car.model.gear_selector_v33 import DEPCarGearSelectorV33
from dep_car.training.losses_v3 import DEPCarJointGearLossConfigV3
from dep_car.training.losses_v31 import DEPCarSequenceCorrectionConfigV31
from dep_car.training.losses_v33 import DEPCarExplicitGearLossConfigV33
from dep_car.training.losses_v331 import (
    DEPCarBankAvailabilityCorrectionConfigV331,
    DEPCarObjectiveV331,
)
from dep_car.training.p4_dataset import p3_training_collate, p3_training_worker_init
from dep_car.training.score_dataset import P3JointGearDatasetV3
import train_dep_car_gear_selector_v33 as v33


CONFIG = ROOT / "dep_car/config/p5_joint_gear_v331_unilateral_correction.yaml"
TRAINER = Path(__file__).resolve()
STAGE = "unilateral_safe_bank_correction"
CHECKPOINT_SCHEMA = "DEPCarJointGearV331CheckpointV1"
CONTRACT_SCHEMA = "DEPCarJointGearV331ArtifactContractV1"

resolve = v33.resolve
sha256_file = v33.sha256_file
read_json = v33.read_json
_best = v33._best


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
        config.get("schema") != "DEPCarJointGearV331TrainingContractV1"
        or config.get("architecture_id") != DEPCarGearSelectorV33.architecture_id
        or config.get("objective_id") != DEPCarObjectiveV331.objective_id
        or config.get("scope") != "fusion_only_unilateral_safe_bank_correction"
        or config.get("test_split_sealed") is not True
        or config.get("qualification", {}).get("only_explicit_selector_trainable") is not True
        or config.get("qualification", {}).get("candidate_and_score_frozen") is not True
        or config.get("qualification", {}).get(
            "selection_uses_all_v33_acceptance_metrics"
        ) is not True
        or config.get("qualification", {}).get(
            "active_control_authorized_by_training"
        ) is not False
        or config.get("qualification", {}).get(
            "production_qualified_by_training"
        ) is not False
    ):
        raise RuntimeError("V3.3.1 training contract identity is invalid")
    DEPCarBankAvailabilityCorrectionConfigV331(**config["correction"]).validate()
    base = config["base_contract"]
    frozen = (
        (base["training_config"], base["training_config_sha256"]),
        (base["trainer"], base["trainer_sha256"]),
        (base["model"], base["model_sha256"]),
        (base["loss"], base["loss_sha256"]),
        (base["acceptance_contract"], base["acceptance_contract_sha256"]),
        (config["implementation"]["loss"], config["implementation"]["loss_sha256"]),
        (
            config["diagnostic_evidence"]["report"],
            config["diagnostic_evidence"]["report_sha256"],
        ),
    )
    mismatched = [
        str(resolve(path)) for path, expected in frozen
        if sha256_file(resolve(path)) != expected
    ]
    if mismatched:
        raise RuntimeError("V3.3.1 frozen contract differs: " + ",".join(mismatched))
    v33_config, v33_sha, v31_config, base_config, acceptance = v33.load_config()
    if v33_sha != base["training_config_sha256"]:
        raise RuntimeError("V3.3.1 did not load the exact V3.3 contract")
    evidence = read_json(resolve(config["diagnostic_evidence"]["report"]))
    expected_evidence = config["diagnostic_evidence"]
    if (
        evidence.get("schema") != "DEPCarJointGearV33FallbackDiagnosticV1"
        or evidence.get("samples") != int(expected_evidence["validation_samples"])
        or evidence.get("fallback_samples") != int(expected_evidence["fallback_samples"])
        or evidence.get("counts_by_direction", {}).get(
            "REQUEST_REVERSE_ONLY_FORWARD_SAFE"
        ) != int(expected_evidence["request_reverse_only_forward_safe"])
        or evidence.get("counts_by_direction", {}).get(
            "REQUEST_FORWARD_ONLY_REVERSE_SAFE"
        ) != int(expected_evidence["request_forward_only_reverse_safe"])
        or evidence.get("counts_by_current_gear", {}).get("0")
        != int(expected_evidence["neutral_fallback_samples"])
        or evidence.get("test_split_accessed") is not False
    ):
        raise RuntimeError("V3.3.1 diagnostic evidence is inconsistent")
    return config, hashlib.sha256(raw).hexdigest(), v33_config, v31_config, base_config, acceptance


def verify_source(config, data_gate):
    source = config["initialization"]
    path = resolve(source["checkpoint"])
    contract_path = resolve(source["checkpoint_contract"])
    acceptance_path = resolve(source["acceptance"])
    for actual, expected in (
        (path, source["checkpoint_sha256"]),
        (contract_path, source["checkpoint_contract_sha256"]),
        (acceptance_path, source["acceptance_sha256"]),
    ):
        if sha256_file(actual) != expected:
            raise RuntimeError("V3.3.1 source hash differs: %s" % actual)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    contract = read_json(contract_path)
    acceptance = read_json(acceptance_path)
    base = config["base_contract"]
    required = {
        "schema": v33.CHECKPOINT_SCHEMA,
        "architecture_id": DEPCarGearSelectorV33.architecture_id,
        "objective_id": "dep_car_objective_v9_explicit_gear_selector_hard_veto",
        "training_stage": v33.STAGE,
        "modality": "fusion",
        "artifact_role": "best",
        "run_completed": True,
        "partial_epoch": False,
        "training_config_sha256": base["training_config_sha256"],
        "trainer_sha256": base["trainer_sha256"],
        "base_policy_frozen": True,
        "active_control_authorized": False,
        "production_qualified": False,
    }
    errors = [key for key, expected in required.items() if payload.get(key) != expected]
    allowed = sorted(source["allowed_failed_checks"])
    if payload.get("data_authority_gate") != data_gate:
        errors.append("data_authority_gate")
    if (
        contract.get("checkpoint_sha256") != source["checkpoint_sha256"]
        or acceptance.get("schema") != "DEPCarJointGearV33AcceptanceV1"
        or acceptance.get("stage") != v33.STAGE
        or acceptance.get("status") != "FAIL"
        or acceptance.get("gate_passed") is not False
        or sorted(acceptance.get("errors", ())) != allowed
        or acceptance.get("checkpoint_sha256") != source["checkpoint_sha256"]
        or acceptance.get("test_split_accessed") is not False
        or acceptance.get("active_control_authorized") is not False
        or acceptance.get("production_qualified") is not False
    ):
        errors.append("acceptance_lineage")
    if errors:
        raise RuntimeError("V3.3.1 source authority failed: " + ",".join(errors))
    gate = {
        "schema": "DEPCarJointGearV331SourceGateV1",
        "passed": True,
        "errors": [],
        "checkpoint": str(path),
        "checkpoint_sha256": sha256_file(path),
        "checkpoint_contract": str(contract_path),
        "checkpoint_contract_sha256": sha256_file(contract_path),
        "acceptance": str(acceptance_path),
        "acceptance_sha256": sha256_file(acceptance_path),
        "allowed_failed_checks": allowed,
        "candidate_and_score_frozen": True,
        "p6_authorized_by_source": False,
        "test_split_accessed": False,
    }
    return path, payload, gate


def build_model(config, v33_config, base_config, data_gate, source):
    _v31_path, v31_source, _v31_gate = v33.verify_source(v33_config, data_gate)
    model = v33.build_model(v33_config, v31_source)
    model.load_state_dict(source["model_state_dict"], strict=True)
    model.freeze_base()
    return model


def configure_selector_only(model):
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    selected = tuple(model.gear_selector.parameters())
    for parameter in selected:
        parameter.requires_grad_(True)
    selected_ids = {id(parameter) for parameter in selected}
    return {
        "stage": STAGE,
        "base_policy_trainable": 0,
        "explicit_selector_trainable": sum(p.numel() for p in selected),
        "frozen_parameters": sum(
            p.numel() for p in model.parameters() if id(p) not in selected_ids
        ),
    }, selected


def atomic_torch(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def write_artifact(path, model, optimizer, scaler, metadata, metrics, history):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": CHECKPOINT_SCHEMA,
        "architecture_id": DEPCarGearSelectorV33.architecture_id,
        "objective_id": DEPCarObjectiveV331.objective_id,
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
        "candidate_and_score_frozen": True,
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
            "completed_epochs", "selected_epoch", "partial_epoch", "global_step",
            "source_checkpoint", "source_checkpoint_sha256", "data_authority_gate",
            "source_gate", "training_config_sha256", "trainer_sha256",
            "acceptance_contract_sha256", "candidate_and_score_frozen",
            "metrics", "selection_gate",
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


def finalize_best(path, epochs, global_step, history):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload.update({
        "completed_epochs": epochs, "global_step": global_step,
        "run_completed": True, "partial_epoch": False, "history": history,
    })
    atomic_torch(path, payload)
    contract_path = path.with_suffix(".contract.json")
    contract = read_json(contract_path)
    contract.update({
        "checkpoint_sha256": sha256_file(path), "completed_epochs": epochs,
        "global_step": global_step, "run_completed": True, "partial_epoch": False,
    })
    contract_path.write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main(argv=None):
    args = parser().parse_args(argv)
    config, config_sha, v33_config, v31_config, base_config, acceptance = load_config()
    training = config["training"]
    epochs = int(training["epochs"]) if args.epochs is None else args.epochs
    batch_size = int(training["batch_size"]) if args.batch_size is None else args.batch_size
    workers = int(training["workers"]) if args.workers is None else args.workers
    if min(epochs, batch_size, workers) < 1:
        raise SystemExit("epochs, batch size, and workers must be positive")
    if args.max_samples is not None and not 1 <= args.max_samples <= 2048:
        raise SystemExit("bounded max-samples must be in [1,2048]")
    if args.max_steps is not None and not 1 <= args.max_steps <= 64:
        raise SystemExit("bounded max-steps must be in [1,64]")

    _bundle_path, bundle, sequence_path, _authority, data_gate = (
        v33.v31.v3.verify_data_authority(base_config)
    )
    source_path, source, source_gate = verify_source(config, data_gate)
    output_expected = resolve(config["artifact"]["output"])
    output = resolve(args.output) if args.output else output_expected
    bounded = args.max_samples is not None or args.max_steps is not None
    formal = bool(
        not bounded and epochs == int(training["epochs"])
        and batch_size == int(training["batch_size"])
        and workers == int(training["workers"])
        and args.device == "cuda" and output == output_expected
        and data_gate["passed"] and source_gate["passed"]
    )
    if not args.dry_run and not formal and not bounded:
        raise RuntimeError("V3.3.1 run is neither formal nor bounded diagnostic")
    if not args.dry_run and output.exists() and args.resume is None:
        raise RuntimeError("V3.3.1 output exists; use --resume")
    if args.resume and resolve(args.resume) != output:
        raise RuntimeError("V3.3.1 resume must point to canonical output")
    plan = {
        "schema": "DEPCarJointGearV331TrainingPlanV1",
        "status": "DRY_RUN_READY" if args.dry_run and formal else (
            "BOUNDED_DIAGNOSTIC_READY" if bounded else "READY"
        ),
        "stage": STAGE,
        "architecture_id": DEPCarGearSelectorV33.architecture_id,
        "objective_id": DEPCarObjectiveV331.objective_id,
        "source": str(source_path), "source_sha256": sha256_file(source_path),
        "output": str(output), "epochs": epochs, "batch_size": batch_size,
        "workers": workers, "device": args.device, "bounded_smoke": bounded,
        "maximum_samples": args.max_samples, "maximum_steps": args.max_steps,
        "formal_training_authorized": formal,
        "data_authority_gate": data_gate, "source_gate": source_gate,
        "diagnostic_evidence": config["diagnostic_evidence"],
        "training_config": str(CONFIG), "training_config_sha256": config_sha,
        "trainer_sha256": sha256_file(TRAINER),
        "acceptance_contract": str(resolve(config["base_contract"]["acceptance_contract"])),
        "acceptance_contract_sha256": config["base_contract"]["acceptance_contract_sha256"],
        "candidate_and_score_frozen": True,
        "test_split_sealed": True,
        "active_control_authorized": False,
        "production_qualified": False,
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    random.seed(int(training["seed"])); np.random.seed(int(training["seed"]))
    torch.manual_seed(int(training["seed"]))
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(int(training["seed"]))
    torch.set_num_threads(int(training["torch_threads"]))
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    model = build_model(config, v33_config, base_config, data_gate, source).to(device)
    ownership, selected = configure_selector_only(model)
    optimizer = torch.optim.AdamW(
        selected, lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    amp = bool(training["mixed_precision"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    common = dict(
        sample_root=bundle["sample_root"], maps_root=bundle["maps_root"],
        index_path=bundle["index"], index_splits=("train", "validation"),
        workers=workers,
        expected_map_contract_aggregate_sha256=bundle["map_contract_aggregate_sha256"],
        expected_index_sha256=bundle["index_sha256"], modality="fusion",
        sequence_index_path=sequence_path,
    )
    train_data = P3JointGearDatasetV3(split="train", **common)
    validation_data = P3JointGearDatasetV3(split="validation", **common)
    if args.max_samples:
        train_data = Subset(train_data, np.linspace(
            0, len(train_data) - 1, min(len(train_data), args.max_samples), dtype=np.int64
        ).tolist())
        validation_data = Subset(validation_data, np.linspace(
            0, len(validation_data) - 1,
            min(len(validation_data), args.max_samples), dtype=np.int64
        ).tolist())
    loader_common = dict(
        batch_size=batch_size, num_workers=workers,
        pin_memory=device.type == "cuda", persistent_workers=workers > 0,
        prefetch_factor=int(training["prefetch_factor"]) if workers > 0 else None,
        collate_fn=p3_training_collate, worker_init_fn=p3_training_worker_init,
    )
    train_loader = DataLoader(train_data, shuffle=True, **loader_common)
    validation_loader = DataLoader(validation_data, shuffle=False, **loader_common)
    objective = DEPCarObjectiveV331(
        DEPCarJointGearLossConfigV3(**base_config["loss"]),
        DEPCarSequenceCorrectionConfigV31(**v31_config["correction"]),
        DEPCarExplicitGearLossConfigV33(**v33_config["selector_loss"]),
        DEPCarBankAvailabilityCorrectionConfigV331(**config["correction"]),
    )

    history, global_step, best_key, start_epoch = [], 0, None, 1
    if args.resume:
        resumed = torch.load(output, map_location="cpu", weights_only=True)
        required = {
            "schema": CHECKPOINT_SCHEMA,
            "architecture_id": DEPCarGearSelectorV33.architecture_id,
            "objective_id": DEPCarObjectiveV331.objective_id,
            "training_stage": STAGE, "artifact_role": "last",
            "source_checkpoint_sha256": plan["source_sha256"],
            "training_config_sha256": config_sha,
            "trainer_sha256": plan["trainer_sha256"],
        }
        mismatch = [key for key, expected in required.items() if resumed.get(key) != expected]
        if mismatch: raise RuntimeError("V3.3.1 resume mismatch: " + ",".join(mismatch))
        completed = int(resumed.get("completed_epochs", 0))
        if not 1 <= completed < epochs: raise RuntimeError("invalid resume epoch")
        model.load_state_dict(resumed["model_state_dict"], strict=True)
        optimizer.load_state_dict(resumed["optimizer_state_dict"])
        scaler.load_state_dict(resumed["grad_scaler_state_dict"])
        history = list(resumed.get("history", ()))
        global_step = int(resumed.get("global_step", 0)); start_epoch = completed + 1
        keys = [v33.selection_key(row["validation"], acceptance) for row in history]
        best_key = min(keys) if keys else None
        model.to(device)

    def metadata(role, epoch, step, gate):
        return {
            "artifact_role": role, "completed_epochs": epoch,
            "selected_epoch": epoch, "global_step": step,
            "source_checkpoint": str(source_path),
            "source_checkpoint_sha256": plan["source_sha256"],
            "data_authority_gate": data_gate, "source_gate": source_gate,
            "training_config_sha256": config_sha,
            "trainer_sha256": plan["trainer_sha256"],
            "acceptance_contract_sha256": plan["acceptance_contract_sha256"],
            "selection_gate": gate,
        }

    if not args.resume:
        baseline = v33.epoch_loop(
            model, objective, validation_loader, device, amp,
            max_steps=args.max_steps,
            progress_interval=int(training["progress_interval_steps"]),
        )
        gate_raw = v33.acceptance_checks(baseline, acceptance)
        gate = {key: "PASS" if value else "FAIL" for key, value in gate_raw.items()}
        row = {"epoch": 0, "phase": "v33_failed_fallback_baseline", "train": None,
               "validation": baseline, "selection_gate": gate,
               "elapsed_s": baseline["elapsed_s"]}
        history.append(row); best_key = v33.selection_key(baseline, acceptance)
        write_artifact(_best(output), model, optimizer, scaler,
                       metadata("best", 0, 0, gate), baseline, history)
        print(json.dumps(row, sort_keys=True), flush=True)

    for epoch in range(start_epoch, epochs + 1):
        started = time.time()
        train_metrics = v33.epoch_loop(
            model, objective, train_loader, device, amp,
            optimizer=optimizer, scaler=scaler, max_steps=args.max_steps,
            dropout=float(training["sensor_dropout_probability"]),
            progress_interval=int(training["progress_interval_steps"]),
            gradient_clip=float(training["gradient_clip"]),
        )
        validation_metrics = v33.epoch_loop(
            model, objective, validation_loader, device, amp,
            max_steps=args.max_steps,
            progress_interval=int(training["progress_interval_steps"]),
        )
        global_step += train_metrics["steps"]
        gate_raw = v33.acceptance_checks(validation_metrics, acceptance)
        gate = {key: "PASS" if value else "FAIL" for key, value in gate_raw.items()}
        row = {"epoch": epoch, "train": train_metrics,
               "validation": validation_metrics, "selection_gate": gate,
               "elapsed_s": time.time() - started}
        history.append(row); print(json.dumps(row, sort_keys=True), flush=True)
        key = v33.selection_key(validation_metrics, acceptance)
        write_artifact(output, model, optimizer, scaler,
                       metadata("last", epoch, global_step, gate),
                       validation_metrics, history)
        if best_key is None or key < best_key:
            best_key = key
            write_artifact(_best(output), model, optimizer, scaler,
                           metadata("best", epoch, global_step, gate),
                           validation_metrics, history)
    best_path = _best(output)
    finalize_best(best_path, epochs, global_step, history)
    print(json.dumps({**plan, "status": "COMPLETE", "global_step": global_step,
                      "stage_ownership": ownership,
                      "best_checkpoint": str(best_path)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
