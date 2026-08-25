#!/usr/bin/env python3
"""Train the V4.1 hierarchical safety/viability score correction."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dep_car/src"))
sys.path.insert(0, str(ROOT / "tools"))

from dep_car.model.dep_car_net_v4 import DEPCarNetV4, HybridSequenceConfigV4
from dep_car.model.hybrid_sequence_rollout import HybridSequenceRolloutConfigV4
from dep_car.training.losses_v4 import DEPCarHybridSequenceLossConfigV4
from dep_car.training.losses_v41 import (
    DEPCarHierarchicalScoreLossConfigV41,
    DEPCarObjectiveV41,
)
from dep_car.training.p4_dataset import p3_training_collate, p3_training_worker_init
from dep_car.training.v4_dataset import P3HybridSequenceDatasetV4
import train_dep_car_hybrid_sequence_v4 as v4


CONFIG = ROOT / "dep_car/config/p5_hybrid_sequence_v41_score.yaml"
TRAINER = Path(__file__).resolve()
STAGE = DEPCarObjectiveV41.stage
CHECKPOINT_SCHEMA = "DEPCarV41ScoreCheckpointV1"
CONTRACT_SCHEMA = "DEPCarV41ScoreArtifactContractV1"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve(path):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def read_json(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError("expected JSON object: %s" % path)
    return value


def best_path(path):
    return path.with_name(path.stem + ".best.pth")


def load_config():
    raw = CONFIG.read_bytes()
    config = yaml.safe_load(raw)
    if (
        config.get("schema") != "DEPCarV41ScoreCorrectionContractV1"
        or config.get("architecture_id") != DEPCarNetV4.architecture_id
        or config.get("objective_id") != DEPCarObjectiveV41.objective_id
        or config.get("stage") != STAGE
        or config.get("test_split_sealed") is not True
    ):
        raise RuntimeError("V4.1 training contract identity is invalid")
    for name in ("model", "rollout", "loss", "dataset"):
        path = resolve(config["implementation"][name])
        if (
            not path.is_file()
            or sha256_file(path) != config["implementation"][name + "_sha256"]
        ):
            raise RuntimeError("V4.1 %s implementation hash differs" % name)
    diagnostic_path = resolve(config["diagnostic"]["report"])
    diagnostic = read_json(diagnostic_path)
    if (
        sha256_file(diagnostic_path) != config["diagnostic"]["report_sha256"]
        or diagnostic.get("status") != "DIAGNOSTIC_COMPLETE"
        or diagnostic.get("conclusion")
        != config["diagnostic"]["required_conclusion"]
        or diagnostic.get("test_split_accessed") is not False
    ):
        raise RuntimeError("V4.1 diagnostic authority differs")
    model_config = HybridSequenceConfigV4(**{
        key: config["model"][key] for key in (
            "candidates", "actions", "hidden_dim", "primitive_dim",
            "stage_dim", "template_logit_bias",
        )
    })
    rollout_config = HybridSequenceRolloutConfigV4(
        candidates=config["model"]["candidates"],
        actions=config["model"]["actions"],
        steps_per_action=config["model"]["steps_per_action"],
    )
    base_loss = DEPCarHybridSequenceLossConfigV4(**config["base_loss"])
    loss_config = DEPCarHierarchicalScoreLossConfigV41(
        base=base_loss, **config["hierarchical_loss"]
    )
    model_config.validate(); rollout_config.validate(); loss_config.validate()
    return (
        config, hashlib.sha256(raw).hexdigest(), model_config,
        rollout_config, loss_config,
    )


def verify_data_authority(config):
    v4_config, _sha, _model, _rollout, _loss = v4.load_config()
    bundle_path, bundle, sequence_path, authority_path, gate = v4.verify_data_authority(v4_config)
    if (
        sha256_file(resolve(config["implementation"]["dataset"]))
        != config["implementation"]["dataset_sha256"]
        or gate.get("passed") is not True
        or gate.get("test_split_sealed") is not True
    ):
        raise RuntimeError("V4.1 inherited data authority failed")
    value = dict(gate)
    value.update({
        "schema": "DEPCarV41DataGateV1",
        "inherited_v4_contract": str(v4.CONFIG),
        "continuous_sequence_authority": "FIRST_ACTION_ONLY",
        "later_action_supervision": "DIFFERENTIABLE_ROLLOUT_ROUTE_MAP",
    })
    return bundle_path, bundle, sequence_path, authority_path, value


def verify_source(config):
    source = config["source"]
    checkpoint = resolve(source["checkpoint"])
    contract_path = resolve(source["checkpoint_contract"])
    acceptance_path = resolve(source["acceptance"])
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    contract = read_json(contract_path)
    acceptance = read_json(acceptance_path)
    errors = []
    expected = (
        (checkpoint, source["checkpoint_sha256"]),
        (contract_path, source["checkpoint_contract_sha256"]),
        (acceptance_path, source["acceptance_sha256"]),
    )
    for path, digest in expected:
        if not path.is_file() or sha256_file(path) != digest:
            errors.append(path.name + "_sha256")
    if (
        payload.get("schema") != v4.CHECKPOINT_SCHEMA
        or payload.get("architecture_id") != DEPCarNetV4.architecture_id
        or payload.get("training_stage") != v4.STAGES[0]
        or payload.get("run_completed") is not True
        or payload.get("partial_epoch") is not False
        or contract.get("checkpoint_sha256") != sha256_file(checkpoint)
        or acceptance.get("schema") != "DEPCarV4AcceptanceV1"
        or acceptance.get("status") != "PASS"
        or acceptance.get("gate_passed") is not True
        or acceptance.get("checkpoint_sha256") != sha256_file(checkpoint)
        or acceptance.get("test_split_accessed") is not False
    ):
        errors.append("accepted_capacity_identity")
    if errors:
        raise RuntimeError("V4.1 source gate failed: " + ",".join(sorted(set(errors))))
    gate = {
        "schema": "DEPCarV41SourceGateV1", "passed": True,
        "checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_contract": str(contract_path),
        "checkpoint_contract_sha256": sha256_file(contract_path),
        "acceptance": str(acceptance_path),
        "acceptance_sha256": sha256_file(acceptance_path),
        "source_role": "ACCEPTED_V4_CANDIDATE_CAPACITY",
        "failed_v4_score_ignored": True, "test_split_accessed": False,
    }
    return checkpoint, payload, gate


def configure_model(model):
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    modules = (model.safety_head, model.viability_head, model.sequence_score_head)
    selected = []
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad_(True); selected.append(parameter)
    if any(parameter.requires_grad for parameter in model.base_model.parameters()):
        raise RuntimeError("V4.1 frozen perception base gained training authority")
    return tuple(selected), {
        "stage": STAGE,
        "candidate_generator_trainable": False,
        "safety_head_trainable": True,
        "viability_head_trainable": True,
        "score_head_trainable": True,
        "trainable_parameters": sum(parameter.numel() for parameter in selected),
        "high_level_gear_state_machine_trainable": False,
    }


def checks(metrics, qualification):
    return {
        "zero_hard_feasible_rate": metrics["zero_hard_feasible_rate"]
        <= float(qualification["maximum_zero_hard_feasible_rate"]),
        "selected_hard_feasible_rate": metrics["selected_hard_feasible_rate"]
        >= float(qualification["minimum_selected_hard_feasible_rate"]),
        "selected_viable_rate": metrics["selected_viable_rate"]
        >= float(qualification["minimum_selected_viable_rate"]),
        "mean_oracle_regret": metrics["mean_oracle_regret"]
        <= float(qualification["maximum_mean_oracle_regret"]),
    }


def selection_key(metrics, qualification):
    hard_target = float(qualification["minimum_selected_hard_feasible_rate"])
    viable_target = float(qualification["minimum_selected_viable_rate"])
    regret_target = float(qualification["maximum_mean_oracle_regret"])
    zero_target = float(qualification["maximum_zero_hard_feasible_rate"])
    deficit = (
        max(0.0, metrics["zero_hard_feasible_rate"] - zero_target) / zero_target
        + max(0.0, hard_target - metrics["selected_hard_feasible_rate"]) / 0.05
        + max(0.0, viable_target - metrics["selected_viable_rate"]) / 0.15
        + max(0.0, metrics["mean_oracle_regret"] - regret_target) / regret_target
    )
    passed = checks(metrics, qualification)
    return (
        sum(not value for value in passed.values()), deficit,
        -metrics["selected_viable_rate"], -metrics["selected_hard_feasible_rate"],
        metrics["mean_oracle_regret"], metrics["mean_loss"],
    )


def atomic_torch(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary); os.replace(temporary, path)


def write_artifact(path, model, optimizer, scaler, metadata, metrics, history):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": CHECKPOINT_SCHEMA,
        "architecture_id": DEPCarNetV4.architecture_id,
        "objective_id": DEPCarObjectiveV41.objective_id,
        "training_stage": STAGE, "artifact_role": metadata["artifact_role"],
        "status": "TRAINED_UNQUALIFIED", "qualification_status": "UNQUALIFIED",
        "active_control_authorized": False, "production_qualified": False,
        "completed_epochs": metadata["completed_epochs"],
        "selected_epoch": metadata["selected_epoch"], "partial_epoch": False,
        "run_completed": False, "global_step": metadata["global_step"],
        "source_checkpoint": metadata["source_checkpoint"],
        "source_checkpoint_sha256": metadata["source_checkpoint_sha256"],
        "data_authority_gate": metadata["data_authority_gate"],
        "source_gate": metadata["source_gate"],
        "training_config_sha256": metadata["training_config_sha256"],
        "trainer_sha256": metadata["trainer_sha256"],
        "model_implementation_sha256": metadata["model_implementation_sha256"],
        "rollout_implementation_sha256": metadata["rollout_implementation_sha256"],
        "loss_implementation_sha256": metadata["loss_implementation_sha256"],
        "unified_hybrid_sequence": True,
        "hierarchical_score_correction": True,
        "high_level_gear_state_machine": False,
        "metrics": metrics, "selection_gate": metadata["selection_gate"],
        "history": history, "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "grad_scaler_state_dict": scaler.state_dict(),
    }
    atomic_torch(path, payload)
    contract_keys = (
        "schema", "architecture_id", "objective_id", "training_stage",
        "artifact_role", "status", "qualification_status",
        "active_control_authorized", "production_qualified",
        "completed_epochs", "selected_epoch", "partial_epoch", "run_completed",
        "global_step", "source_checkpoint", "source_checkpoint_sha256",
        "data_authority_gate", "source_gate", "training_config_sha256",
        "trainer_sha256", "model_implementation_sha256",
        "rollout_implementation_sha256", "loss_implementation_sha256",
        "unified_hybrid_sequence", "hierarchical_score_correction",
        "high_level_gear_state_machine", "metrics", "selection_gate",
    )
    contract = {key: payload[key] for key in contract_keys}
    contract.update({
        "schema": CONTRACT_SCHEMA, "checkpoint": str(path),
        "checkpoint_sha256": sha256_file(path),
    })
    path.with_suffix(".contract.json").write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def finalize(path, epochs, global_step, history):
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


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--source", required=True)
    value.add_argument("--output", required=True)
    value.add_argument("--epochs", type=int)
    value.add_argument("--batch-size", type=int)
    value.add_argument("--workers", type=int)
    value.add_argument("--max-samples", type=int)
    value.add_argument("--max-steps", type=int)
    value.add_argument("--device", default="cuda")
    value.add_argument("--resume")
    value.add_argument("--dry-run", action="store_true")
    return value


def main(argv=None):
    args = parser().parse_args(argv)
    config, config_sha, model_config, rollout_config, loss_config = load_config()
    training = config["training"]
    epochs = int(training["epochs"]) if args.epochs is None else args.epochs
    batch_size = int(training["batch_size"]) if args.batch_size is None else args.batch_size
    workers = int(training["workers"]) if args.workers is None else args.workers
    if min(epochs, batch_size, workers) < 1:
        raise SystemExit("V4.1 epochs, batch size and workers must be positive")
    if args.max_samples is not None and not 1 <= args.max_samples <= 2048:
        raise SystemExit("V4.1 max-samples must be in [1,2048]")
    if args.max_steps is not None and not 1 <= args.max_steps <= 64:
        raise SystemExit("V4.1 max-steps must be in [1,64]")
    _bundle_path, bundle, sequence_path, _authority, data_gate = verify_data_authority(config)
    source_path, source, source_gate = verify_source(config)
    if resolve(args.source) != source_path:
        raise RuntimeError("V4.1 source differs from the frozen contract")
    output = resolve(args.output)
    expected_output = resolve(config["artifacts"]["output"])
    bounded = args.max_samples is not None or args.max_steps is not None
    formal = bool(
        not bounded and epochs == int(training["epochs"])
        and batch_size == int(training["batch_size"])
        and workers == int(training["workers"]) and args.device == "cuda"
        and output == expected_output
    )
    if not args.dry_run and not formal and not bounded:
        raise RuntimeError("V4.1 run is neither formal nor bounded diagnostic")
    if output == source_path:
        raise RuntimeError("V4.1 source and output must differ")
    plan = {
        "schema": "DEPCarV41ScoreTrainingPlanV1",
        "status": "DRY_RUN_READY" if args.dry_run and formal else "BOUNDED_DIAGNOSTIC_READY" if bounded else "READY",
        "stage": STAGE, "architecture_id": DEPCarNetV4.architecture_id,
        "objective_id": DEPCarObjectiveV41.objective_id,
        "source": str(source_path), "source_sha256": sha256_file(source_path),
        "output": str(output), "epochs": epochs, "batch_size": batch_size,
        "workers": workers, "device": args.device, "bounded_smoke": bounded,
        "maximum_samples": args.max_samples, "maximum_steps": args.max_steps,
        "formal_training_authorized": formal, "data_authority_gate": data_gate,
        "source_gate": source_gate, "training_config": str(CONFIG),
        "training_config_sha256": config_sha, "trainer_sha256": sha256_file(TRAINER),
        "candidate_generator_trainable": False,
        "joint_safety_viability_score_training": True,
        "unified_hybrid_sequence": True, "high_level_gear_state_machine": False,
        "test_split_sealed": True, "active_control_authorized": False,
        "production_qualified": False,
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True)); return 0
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if output.exists() and not args.resume:
        raise RuntimeError("V4.1 output exists; use --resume or preserve it explicitly")
    v4.seed_all(int(training["seed"])); torch.set_num_threads(int(training["torch_threads"]))
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    model = v4.build_model(model_config, rollout_config, source).to(device)
    selected, ownership = configure_model(model)
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
    train_data = P3HybridSequenceDatasetV4(split="train", **common)
    validation_data = P3HybridSequenceDatasetV4(split="validation", **common)
    if args.max_samples:
        train_data = Subset(train_data, np.linspace(
            0, len(train_data) - 1, min(len(train_data), args.max_samples), dtype=np.int64
        ).tolist())
        validation_data = Subset(validation_data, np.linspace(
            0, len(validation_data) - 1, min(len(validation_data), args.max_samples), dtype=np.int64
        ).tolist())
    loader_common = dict(
        batch_size=batch_size, num_workers=workers,
        pin_memory=device.type == "cuda", persistent_workers=workers > 0,
        prefetch_factor=int(training["prefetch_factor"]) if workers > 0 else None,
        collate_fn=p3_training_collate, worker_init_fn=p3_training_worker_init,
    )
    train_loader = DataLoader(train_data, shuffle=True, **loader_common)
    validation_loader = DataLoader(validation_data, shuffle=False, **loader_common)
    objective = DEPCarObjectiveV41(loss_config)
    history, global_step, best_key, start_epoch = [], 0, None, 1
    if args.resume:
        resumed = torch.load(output, map_location="cpu", weights_only=True)
        required = {
            "schema": CHECKPOINT_SCHEMA, "architecture_id": DEPCarNetV4.architecture_id,
            "training_stage": STAGE, "artifact_role": "last",
            "source_checkpoint_sha256": plan["source_sha256"],
            "training_config_sha256": config_sha,
            "trainer_sha256": plan["trainer_sha256"],
        }
        mismatch = [key for key, value in required.items() if resumed.get(key) != value]
        if mismatch:
            raise RuntimeError("V4.1 resume mismatch: " + ",".join(mismatch))
        completed = int(resumed.get("completed_epochs", 0))
        if not 1 <= completed < epochs:
            raise RuntimeError("V4.1 resume epoch is outside remaining run")
        model.load_state_dict(resumed["model_state_dict"], strict=True)
        optimizer.load_state_dict(resumed["optimizer_state_dict"])
        scaler.load_state_dict(resumed["grad_scaler_state_dict"])
        history = list(resumed.get("history", ()))
        global_step = int(resumed.get("global_step", 0)); start_epoch = completed + 1
        keys = [selection_key(row["validation"], config["qualification"]) for row in history]
        best_key = min(keys) if keys else None; model.to(device)
    for epoch in range(start_epoch, epochs + 1):
        started = time.time()
        train_metrics = v4.epoch_loop(
            model, objective, train_loader, STAGE, device, amp,
            optimizer=optimizer, scaler=scaler, max_steps=args.max_steps,
            dropout=float(training["sensor_dropout_probability"]),
            progress_interval=int(training["progress_interval_steps"]),
            gradient_clip=float(training["gradient_clip"]),
        )
        validation_metrics = v4.epoch_loop(
            model, objective, validation_loader, STAGE, device, amp,
            max_steps=args.max_steps,
            progress_interval=int(training["progress_interval_steps"]),
        )
        global_step += train_metrics["steps"]
        result = checks(validation_metrics, config["qualification"])
        gate = {key: "PASS" if value else "FAIL" for key, value in result.items()}
        row = {
            "epoch": epoch, "train": train_metrics,
            "validation": validation_metrics, "selection_gate": gate,
            "elapsed_s": time.time() - started,
        }
        history.append(row); print(json.dumps(row, sort_keys=True), flush=True)
        key = selection_key(validation_metrics, config["qualification"])
        metadata = {
            "artifact_role": "last", "completed_epochs": epoch,
            "selected_epoch": epoch, "global_step": global_step,
            "source_checkpoint": str(source_path),
            "source_checkpoint_sha256": plan["source_sha256"],
            "data_authority_gate": data_gate, "source_gate": source_gate,
            "training_config_sha256": config_sha,
            "trainer_sha256": plan["trainer_sha256"],
            "model_implementation_sha256": config["implementation"]["model_sha256"],
            "rollout_implementation_sha256": config["implementation"]["rollout_sha256"],
            "loss_implementation_sha256": config["implementation"]["loss_sha256"],
            "selection_gate": gate,
        }
        write_artifact(
            output, model, optimizer, scaler, metadata,
            validation_metrics, history,
        )
        if best_key is None or key < best_key:
            best_key = key; metadata["artifact_role"] = "best"
            write_artifact(
                best_path(output), model, optimizer, scaler, metadata,
                validation_metrics, history,
            )
    selected_path = best_path(output)
    if not selected_path.is_file():
        raise RuntimeError("V4.1 training ended without best checkpoint")
    finalize(selected_path, epochs, global_step, history)
    print(json.dumps({
        **plan, "status": "COMPLETE", "global_step": global_step,
        "stage_ownership": ownership, "best_checkpoint": str(selected_path),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
