#!/usr/bin/env python3
"""Train DEPCarNetV3 geometry, joint gear score, and recovery sequence stages."""

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

from dep_car.model.dep_car_net_v2 import DEPCarNetV2
from dep_car.model.dep_car_net_v3 import DEPCarNetV3
from dep_car.training.losses_v3 import (
    DEPCarJointGearLossConfigV3,
    DEPCarObjectiveV3,
)
from dep_car.training.p4_dataset import p3_training_collate, p3_training_worker_init
from dep_car.training.score_dataset import P3JointGearDatasetV3
from dep_car.training.stages import apply_sensor_dropout


CONFIG = ROOT / "dep_car/config/p5_joint_gear_v3_training.yaml"
TRAINER = Path(__file__).resolve()
STAGES = (
    "bidirectional_candidate_capacity",
    "joint_gear_score_calibration",
    "sequence_recovery_finetune",
)
STAGE_ARTIFACT = {
    STAGES[0]: "candidate",
    STAGES[1]: "score",
    STAGES[2]: "sequence",
}


def resolve(path):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value):
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_json(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError("expected JSON object: %s" % path)
    return value


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--stage", choices=STAGES, required=True)
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


def load_config():
    raw = CONFIG.read_bytes()
    config = yaml.safe_load(raw)
    if (
        config.get("schema") != "DEPCarJointGearV3TrainingContractV1"
        or config.get("architecture_id") != DEPCarNetV3.architecture_id
        or config.get("objective_id") != DEPCarObjectiveV3.objective_id
        or config.get("scope") != "fusion_only_joint_forward_reverse"
        or config.get("dataset", {}).get("test_split_sealed") is not True
    ):
        raise RuntimeError("V3 training contract identity is invalid")
    try:
        DEPCarJointGearLossConfigV3(**config["loss"]).validate()
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("V3 loss contract is invalid") from exc
    return config, hashlib.sha256(raw).hexdigest()


def verify_data_authority(config):
    dataset = config["dataset"]
    bundle_path = resolve(dataset["bundle_authority"])
    sequence_path = resolve(dataset["sequence_index"])
    authority_path = resolve(dataset["sequence_authority"])
    bundle = read_json(bundle_path)
    sequence = read_json(sequence_path)
    authority = read_json(authority_path)
    errors = []
    bundle_content = dict(bundle)
    bundle_claimed = bundle_content.pop("bundle_authority_sha256", None)
    if (
        bundle.get("schema") != "DEPCarP3V3BundleAuthorityV1"
        or bundle.get("status") != "SEALED"
        or bundle.get("test_npz_opened") is not False
        or bundle.get("test_map_yaml_or_png_opened") is not False
        or bundle_claimed != canonical_sha256(bundle_content)
    ):
        errors.append("bundle_authority")
    sequence_content = dict(sequence)
    sequence_claimed = sequence_content.pop("content_sha256", None)
    if (
        sequence.get("schema") != "DEPCarJointGearSequenceIndexV1"
        or sequence.get("bounded") is not False
        or sequence.get("test_split_opened") is not False
        or sequence_claimed != canonical_sha256(sequence_content)
    ):
        errors.append("sequence_index")
    authority_content = dict(authority)
    authority_claimed = authority_content.pop("authority_sha256", None)
    if (
        authority.get("schema") != "DEPCarJointGearSequenceAuthorityV1"
        or authority.get("status") != "PASS"
        or authority.get("errors") != []
        or authority.get("test_split_opened") is not False
        or authority_claimed != canonical_sha256(authority_content)
    ):
        errors.append("sequence_authority")
    identities = (
        (authority.get("index_file_sha256"), sha256_file(sequence_path)),
        (authority.get("index_content_sha256"), sequence_claimed),
        (authority.get("bundle_authority_file_sha256"), sha256_file(bundle_path)),
        (authority.get("bundle_authority_sha256"), bundle_claimed),
        (sequence.get("source_index_sha256"), bundle.get("index_sha256")),
        (sequence.get("source_content_aggregate_sha256"), bundle.get("content_aggregate_sha256")),
        (sequence.get("counts_by_split"), bundle.get("counts_by_split")),
    )
    if any(left != right for left, right in identities):
        errors.append("data_lineage")
    if errors:
        raise RuntimeError("V3 data authority failed: " + ",".join(errors))
    return bundle_path, bundle, sequence_path, authority_path, {
        "schema": "DEPCarJointGearV3DataGateV1",
        "passed": True,
        "errors": [],
        "bundle_authority": str(bundle_path),
        "bundle_authority_sha256": sha256_file(bundle_path),
        "sequence_index": str(sequence_path),
        "sequence_index_sha256": sha256_file(sequence_path),
        "sequence_authority": str(authority_path),
        "sequence_authority_sha256": sha256_file(authority_path),
        "samples": sequence["counts_by_split"],
        "test_split_sealed": True,
    }


def _best(path):
    return path.with_name(path.stem + ".best.pth")


def _verify_checkpoint_contract(path, payload):
    contract_path = path.with_suffix(".contract.json")
    if not contract_path.is_file():
        raise RuntimeError("source contract is missing: %s" % contract_path)
    contract = read_json(contract_path)
    if contract.get("checkpoint_sha256") != sha256_file(path):
        raise RuntimeError("source checkpoint/contract identity differs")
    if contract.get("architecture_id") != payload.get("architecture_id"):
        raise RuntimeError("source architecture contract differs")
    return contract_path


def verify_source(path, stage, config):
    path = resolve(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    contract_path = _verify_checkpoint_contract(path, payload)
    initialization = config["initialization"]
    gate = {
        "schema": "DEPCarJointGearV3SourceGateV1",
        "passed": True,
        "checkpoint": str(path),
        "checkpoint_sha256": sha256_file(path),
        "checkpoint_contract": str(contract_path),
        "checkpoint_contract_sha256": sha256_file(contract_path),
        "test_split_accessed": False,
    }
    if stage == STAGES[0]:
        acceptance_path = resolve(initialization["v2_score_acceptance"])
        acceptance = read_json(acceptance_path)
        if (
            path != resolve(initialization["v2_score"])
            or sha256_file(path) != initialization["v2_score_sha256"]
            or contract_path != resolve(initialization["v2_score_contract"])
            or sha256_file(contract_path) != initialization["v2_score_contract_sha256"]
            or sha256_file(acceptance_path) != initialization["v2_score_acceptance_sha256"]
            or payload.get("architecture_id") != DEPCarNetV2.architecture_id
            or payload.get("training_stage") != "score_calibration"
            or payload.get("modality") != "fusion"
            or payload.get("run_completed") is not True
            or acceptance.get("status") != "PASS"
            or acceptance.get("gate_passed") is not True
            or acceptance.get("checkpoint_sha256") != sha256_file(path)
            or acceptance.get("test_split_accessed") is not False
        ):
            raise RuntimeError("V2 Score initialization authority failed")
        gate.update({
            "source_architecture": DEPCarNetV2.architecture_id,
            "acceptance": str(acceptance_path),
            "acceptance_sha256": sha256_file(acceptance_path),
        })
    else:
        previous_stage = STAGES[STAGES.index(stage) - 1]
        previous_artifact = STAGE_ARTIFACT[previous_stage]
        expected = _best(resolve(config["artifacts"][previous_artifact]))
        acceptance_path = path.with_suffix(".acceptance.json")
        if path != expected or not acceptance_path.is_file():
            raise RuntimeError("V3 stage source is not the accepted canonical best")
        acceptance = read_json(acceptance_path)
        if (
            payload.get("schema") != "DEPCarJointGearV3CheckpointV1"
            or payload.get("architecture_id") != DEPCarNetV3.architecture_id
            or payload.get("training_stage") != previous_stage
            or payload.get("run_completed") is not True
            or payload.get("partial_epoch") is not False
            or acceptance.get("schema") != "DEPCarJointGearV3AcceptanceV1"
            or acceptance.get("stage") != previous_stage
            or acceptance.get("status") != "PASS"
            or acceptance.get("gate_passed") is not True
            or acceptance.get("checkpoint_sha256") != sha256_file(path)
            or acceptance.get("test_split_accessed") is not False
        ):
            raise RuntimeError("V3 previous-stage acceptance failed")
        gate.update({
            "source_architecture": DEPCarNetV3.architecture_id,
            "acceptance": str(acceptance_path),
            "acceptance_sha256": sha256_file(acceptance_path),
        })
    return path, payload, gate


def configure_stage(model, stage):
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    selected = (
        tuple(model.candidate_parameters())
        if stage == STAGES[0]
        else tuple(model.score_parameters())
    )
    for parameter in selected:
        parameter.requires_grad_(True)
    all_parameters = tuple(model.parameters())
    selected_ids = {id(parameter) for parameter in selected}
    if len(selected_ids) != len(selected):
        raise RuntimeError("V3 stage parameter partitions overlap internally")
    return {
        "stage": stage,
        "trainable_parameters": sum(parameter.numel() for parameter in selected),
        "frozen_parameters": sum(
            parameter.numel() for parameter in all_parameters
            if id(parameter) not in selected_ids
        ),
    }, selected


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_valid(host, device):
    valid = host["geometry_valid"].bool()
    if not bool(valid.any()):
        return None
    return {
        key: value[valid].to(device, non_blocking=True)
        for key, value in host.items()
        if key != "metadata"
    }


def forward_loss(model, objective, batch, stage, amp, dropout=0.0):
    modality = batch["modality_mask"]
    if dropout > 0.0:
        modality = apply_sensor_dropout(modality, dropout)
    with torch.autocast(
        device_type=batch["state"].device.type,
        dtype=torch.float16,
        enabled=amp,
    ):
        output = model(
            batch["depth"], batch["lidar_bev"], batch["state"],
            batch["current_gear"], batch["gear_history"],
            batch["route_pose"], batch["route_mask"], modality,
        )
    with torch.autocast(device_type=batch["state"].device.type, enabled=False):
        losses = objective(
            output,
            map_distance_field=batch["map_distance_field"].float(),
            map_resolution=batch["map_resolution"].float(),
            map_origin=batch["map_origin"].float(),
            chassis_to_map=batch["chassis_to_map"].float(),
            route=batch["route_pose"].float(),
            route_mask=batch["route_mask"],
            gear_history=batch["gear_history"].float(),
            sequence_gears=batch["sequence_gears"],
            sequence_mask=batch["sequence_mask"],
            stage=stage,
        )
    return output, losses


def epoch_loop(
    model, objective, loader, stage, device, amp, *, optimizer=None,
    scaler=None, max_steps=None, dropout=0.0, progress_interval=50,
    gradient_clip=5.0,
):
    training = optimizer is not None
    model.train(training and stage == STAGES[0])
    totals = torch.zeros(14, dtype=torch.float64, device=device)
    samples = steps = 0
    finite = torch.ones((), dtype=torch.bool, device=device)
    started = time.monotonic()
    context = torch.enable_grad if training else torch.inference_mode
    with context():
        for host in loader:
            if max_steps is not None and steps >= max_steps:
                break
            batch = select_valid(host, device)
            if batch is None:
                continue
            if training:
                optimizer.zero_grad(set_to_none=True)
            output, losses = forward_loss(
                model, objective, batch, stage, amp,
                dropout=dropout if training else 0.0,
            )
            finite.logical_and_(torch.isfinite(losses["total"].detach()))
            if training:
                scaler.scale(losses["total"]).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], gradient_clip
                )
                scaler.step(optimizer)
                scaler.update()
            count = len(batch["state"])
            feasible = losses["hard_feasible"]
            selected = output.scores.detach().argmin(dim=1)[:, None]
            selected_hard = feasible.gather(1, selected).squeeze(1)
            chosen_gear = output.candidate_gears.gather(1, selected).squeeze(1)
            forward_available = losses["forward_available"]
            reverse_feasible = feasible[:, 15:].any(dim=1)
            required_reverse = ~forward_available & reverse_feasible
            selected_reverse = chosen_gear < 0
            unnecessary_reverse = selected_reverse & forward_available
            selected_recovery = losses["forward_recovery_target"].gather(
                1, selected
            ).squeeze(1).bool()
            cost = losses["candidate_cost"]
            any_feasible = feasible.any(dim=1)
            oracle = cost.masked_fill(~feasible, torch.inf).amin(dim=1)
            chosen_cost = cost.gather(1, selected).squeeze(1)
            regret = torch.where(
                any_feasible,
                (chosen_cost - oracle).clamp_min(0.0),
                torch.zeros_like(chosen_cost),
            )
            totals += torch.stack((
                losses["total"].detach().double() * count,
                feasible.sum().double(),
                (~any_feasible).sum().double(),
                selected_hard.sum().double(),
                selected_reverse.sum().double(),
                unnecessary_reverse.sum().double(),
                forward_available.sum().double(),
                required_reverse.sum().double(),
                (selected_reverse & required_reverse).sum().double(),
                selected_recovery.sum().double(),
                regret.sum().double(),
                any_feasible.sum().double(),
                feasible[:, :15].any(dim=1).sum().double(),
                feasible[:, 15:].any(dim=1).sum().double(),
            ))
            samples += count
            steps += 1
            if training and steps % int(progress_interval) == 0:
                elapsed = max(1.0e-6, time.monotonic() - started)
                print(
                    "step=%d samples=%d loss=%.6f samples_per_s=%.1f" % (
                        steps, samples, float(losses["total"].detach()),
                        samples / elapsed,
                    ), flush=True,
                )
    if not bool(finite.cpu()):
        raise FloatingPointError("V3 objective became non-finite")
    values = totals.cpu().tolist()
    denominator = max(1, samples)
    return {
        "samples": samples,
        "steps": steps,
        "mean_loss": values[0] / denominator,
        "mean_feasible_candidates": values[1] / denominator,
        "zero_hard_feasible_rate": values[2] / denominator,
        "selected_hard_feasible_rate": values[3] / denominator,
        "selected_reverse_rate": values[4] / denominator,
        "unnecessary_reverse_rate": values[5] / max(1.0, values[6]),
        "forward_available_samples": int(values[6]),
        "required_reverse_samples": int(values[7]),
        "required_reverse_selection_rate": values[8] / max(1.0, values[7]),
        "selected_forward_recovery_rate": values[9] / denominator,
        "mean_oracle_regret": values[10] / max(1.0, values[11]),
        "forward_bank_capable_rate": values[12] / denominator,
        "reverse_bank_capable_rate": values[13] / denominator,
        "elapsed_s": time.monotonic() - started,
    }


def selection_key(stage, metrics):
    if stage == STAGES[0]:
        return (
            metrics["zero_hard_feasible_rate"],
            -metrics["forward_bank_capable_rate"],
            -metrics["reverse_bank_capable_rate"],
            metrics["mean_loss"],
        )
    return (
        -metrics["selected_hard_feasible_rate"],
        metrics["unnecessary_reverse_rate"],
        -metrics["required_reverse_selection_rate"],
        metrics["mean_oracle_regret"],
    )


def atomic_torch(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def write_artifact(path, model, optimizer, scaler, metadata, metrics, history):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "DEPCarJointGearV3CheckpointV1",
        "architecture_id": DEPCarNetV3.architecture_id,
        "objective_id": DEPCarObjectiveV3.objective_id,
        "training_stage": metadata["stage"],
        "modality": "fusion",
        "artifact_role": metadata["artifact_role"],
        "status": "TRAINED_UNQUALIFIED",
        "qualification_status": "UNQUALIFIED",
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
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "grad_scaler_state_dict": scaler.state_dict(),
        "metrics": metrics,
        "history": history,
    }
    atomic_torch(path, payload)
    contract = {
        key: payload[key]
        for key in (
            "schema", "architecture_id", "objective_id", "training_stage",
            "modality", "artifact_role", "status", "qualification_status",
            "production_qualified", "completed_epochs", "selected_epoch",
            "partial_epoch", "global_step", "source_checkpoint",
            "source_checkpoint_sha256", "data_authority_gate", "source_gate",
            "training_config_sha256", "trainer_sha256", "metrics",
        )
    }
    contract["schema"] = "DEPCarJointGearV3ArtifactContractV1"
    contract["checkpoint"] = str(path)
    contract["checkpoint_sha256"] = sha256_file(path)
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
    config, config_sha = load_config()
    training = config["training"]
    epoch_key = {
        STAGES[0]: "candidate_epochs",
        STAGES[1]: "score_epochs",
        STAGES[2]: "sequence_epochs",
    }[args.stage]
    epochs = int(training[epoch_key]) if args.epochs is None else args.epochs
    batch_size = int(training["batch_size"]) if args.batch_size is None else args.batch_size
    workers = int(training["workers"]) if args.workers is None else args.workers
    if min(epochs, batch_size, workers) < 1:
        raise SystemExit("epochs, batch size, and workers must be positive")
    if args.max_samples is not None and not 1 <= args.max_samples <= 1024:
        raise SystemExit("bounded max-samples must be in [1,1024]")
    if args.max_steps is not None and not 1 <= args.max_steps <= 64:
        raise SystemExit("bounded max-steps must be in [1,64]")
    bundle_path, bundle, sequence_path, sequence_authority, data_gate = (
        verify_data_authority(config)
    )
    source_path, source, source_gate = verify_source(args.source, args.stage, config)
    artifact_key = STAGE_ARTIFACT[args.stage]
    output = resolve(args.output)
    expected_output = resolve(config["artifacts"][artifact_key])
    bounded = args.max_samples is not None or args.max_steps is not None
    formal_parameters = (
        epochs == int(training[epoch_key])
        and batch_size == int(training["batch_size"])
        and workers == int(training["workers"])
    )
    formal = bool(
        not bounded and formal_parameters and args.device == "cuda"
        and output == expected_output and data_gate["passed"] and source_gate["passed"]
    )
    if not args.dry_run and not formal and not bounded:
        raise RuntimeError("V3 run is neither formal nor a bounded diagnostic")
    if output == source_path:
        raise RuntimeError("source and output must differ")
    if not args.dry_run and output.exists() and args.resume is None:
        raise RuntimeError("output exists; use --resume with the canonical last artifact")
    if args.resume is not None and resolve(args.resume) != output:
        raise RuntimeError("resume must point to the selected output")
    plan = {
        "schema": "DEPCarJointGearV3TrainingPlanV1",
        "status": "DRY_RUN_READY" if args.dry_run and formal else "BOUNDED_DIAGNOSTIC_READY" if bounded else "READY",
        "stage": args.stage,
        "architecture_id": DEPCarNetV3.architecture_id,
        "objective_id": DEPCarObjectiveV3.objective_id,
        "source": str(source_path),
        "source_sha256": sha256_file(source_path),
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
        "test_split_sealed": True,
        "production_qualified": False,
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    seed_all(int(training["seed"]))
    torch.set_num_threads(int(training["torch_threads"]))
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    model = DEPCarNetV3()
    if source.get("architecture_id") == DEPCarNetV2.architecture_id:
        model.initialize_from_v2(source["model_state_dict"])
    else:
        model.load_state_dict(source["model_state_dict"], strict=True)
    model.to(device)
    ownership, selected_parameters = configure_stage(model, args.stage)
    optimizer = torch.optim.AdamW(
        selected_parameters, lr=float(training["learning_rate"]),
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
        train_data = Subset(train_data, range(min(len(train_data), args.max_samples)))
        validation_data = Subset(validation_data, range(min(len(validation_data), args.max_samples)))
    loader_common = dict(
        batch_size=batch_size, num_workers=workers,
        pin_memory=device.type == "cuda", persistent_workers=workers > 0,
        prefetch_factor=int(training["prefetch_factor"]) if workers > 0 else None,
        collate_fn=p3_training_collate, worker_init_fn=p3_training_worker_init,
    )
    train_loader = DataLoader(train_data, shuffle=True, **loader_common)
    validation_loader = DataLoader(validation_data, shuffle=False, **loader_common)
    objective = DEPCarObjectiveV3(DEPCarJointGearLossConfigV3(**config["loss"]))
    history, global_step, best_key, start_epoch = [], 0, None, 1
    if args.resume:
        resumed = torch.load(output, map_location="cpu", weights_only=True)
        required = {
            "schema": "DEPCarJointGearV3CheckpointV1",
            "architecture_id": DEPCarNetV3.architecture_id,
            "training_stage": args.stage,
            "artifact_role": "last",
            "source_checkpoint_sha256": plan["source_sha256"],
            "training_config_sha256": config_sha,
            "trainer_sha256": plan["trainer_sha256"],
        }
        mismatch = [key for key, value in required.items() if resumed.get(key) != value]
        if mismatch:
            raise RuntimeError("V3 resume mismatch: " + ",".join(mismatch))
        completed = int(resumed.get("completed_epochs", 0))
        if not 1 <= completed < epochs:
            raise RuntimeError("V3 resume epoch is outside remaining run")
        model.load_state_dict(resumed["model_state_dict"], strict=True)
        optimizer.load_state_dict(resumed["optimizer_state_dict"])
        scaler.load_state_dict(resumed["grad_scaler_state_dict"])
        history = list(resumed.get("history", ()))
        global_step = int(resumed.get("global_step", 0))
        start_epoch = completed + 1
        keys = [selection_key(args.stage, row["validation"]) for row in history]
        best_key = min(keys) if keys else None
        model.to(device)
    for epoch in range(start_epoch, epochs + 1):
        started = time.time()
        train_metrics = epoch_loop(
            model, objective, train_loader, args.stage, device, amp,
            optimizer=optimizer, scaler=scaler, max_steps=args.max_steps,
            dropout=float(training["sensor_dropout_probability"]),
            progress_interval=int(training["progress_interval_steps"]),
            gradient_clip=float(training["gradient_clip"]),
        )
        validation_metrics = epoch_loop(
            model, objective, validation_loader, args.stage, device, amp,
            max_steps=args.max_steps,
            progress_interval=int(training["progress_interval_steps"]),
        )
        global_step += train_metrics["steps"]
        row = {"epoch": epoch, "train": train_metrics, "validation": validation_metrics, "elapsed_s": time.time() - started}
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        key = selection_key(args.stage, validation_metrics)
        metadata = {
            "stage": args.stage, "artifact_role": "last",
            "completed_epochs": epoch, "selected_epoch": epoch,
            "global_step": global_step, "source_checkpoint": str(source_path),
            "source_checkpoint_sha256": plan["source_sha256"],
            "data_authority_gate": data_gate, "source_gate": source_gate,
            "training_config_sha256": config_sha,
            "trainer_sha256": plan["trainer_sha256"],
        }
        write_artifact(output, model, optimizer, scaler, metadata, validation_metrics, history)
        if best_key is None or key < best_key:
            best_key = key
            metadata["artifact_role"] = "best"
            write_artifact(_best(output), model, optimizer, scaler, metadata, validation_metrics, history)
    best_path = _best(output)
    if not best_path.is_file():
        raise RuntimeError("V3 training ended without best checkpoint")
    finalize_best(best_path, epochs, global_step, history)
    print(json.dumps({**plan, "status": "COMPLETE", "global_step": global_step, "stage_ownership": ownership, "best_checkpoint": str(best_path)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
