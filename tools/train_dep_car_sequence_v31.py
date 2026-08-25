#!/usr/bin/env python3
"""Train the independent V3.1 teacher-balanced sequence correction stage."""

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
from dep_car.training.losses_v3 import DEPCarJointGearLossConfigV3, DEPCarObjectiveV3
from dep_car.training.losses_v31 import (
    DEPCarObjectiveV31,
    DEPCarSequenceCorrectionConfigV31,
)
from dep_car.training.p4_dataset import p3_training_collate, p3_training_worker_init
from dep_car.training.score_dataset import P3JointGearDatasetV3
from dep_car.training.stages import apply_sensor_dropout
import train_dep_car_joint_gear_v3 as v3


CONFIG = ROOT / "dep_car/config/p5_joint_gear_v31_sequence_correction.yaml"
TRAINER = Path(__file__).resolve()
STAGE = "sequence_correction"
CHECKPOINT_SCHEMA = "DEPCarJointGearV31CheckpointV1"
CONTRACT_SCHEMA = "DEPCarJointGearV31ArtifactContractV1"


def resolve(path):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError("expected JSON object: %s" % path)
    return value


def _best(path):
    return path.with_name(path.stem + ".best.pth")


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
        config.get("schema") != "DEPCarJointGearV31TrainingContractV1"
        or config.get("architecture_id") != DEPCarNetV3.architecture_id
        or config.get("objective_id") != DEPCarObjectiveV31.objective_id
        or config.get("scope") != "fusion_only_score_sequence_correction"
        or config.get("test_split_sealed") is not True
        or config.get("qualification", {}).get(
            "selection_uses_final_acceptance_metrics"
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
        raise RuntimeError("V3.1 training contract identity is invalid")
    try:
        DEPCarSequenceCorrectionConfigV31(**config["correction"]).validate()
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("V3.1 correction loss contract is invalid") from exc
    base_contract = config["base_contract"]
    frozen_files = (
        (base_contract["training_config"], base_contract["training_config_sha256"]),
        (base_contract["trainer"], base_contract["trainer_sha256"]),
        (
            base_contract["acceptance_contract"],
            base_contract["acceptance_contract_sha256"],
        ),
    )
    mismatched = [
        str(resolve(path))
        for path, expected in frozen_files
        if sha256_file(resolve(path)) != expected
    ]
    if mismatched:
        raise RuntimeError("V3.1 frozen base contract differs: " + ",".join(mismatched))
    base_config, base_config_sha = v3.load_config()
    if base_config_sha != base_contract["training_config_sha256"]:
        raise RuntimeError("V3.1 base training contract was not loaded exactly")
    acceptance = yaml.safe_load(
        resolve(base_contract["acceptance_contract"]).read_text(encoding="utf-8")
    )
    if (
        acceptance.get("schema") != "DEPCarJointGearV3AcceptanceContractV2"
        or acceptance.get("revision") != 2
        or acceptance.get("test_split_sealed") is not True
    ):
        raise RuntimeError("V3.1 final acceptance contract is invalid")
    return config, hashlib.sha256(raw).hexdigest(), base_config, acceptance


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
        raise RuntimeError("V3.1 accepted Score source hash differs")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    contract = read_json(contract_path)
    acceptance = read_json(acceptance_path)
    base_contract = config["base_contract"]
    required_payload = {
        "schema": "DEPCarJointGearV3CheckpointV1",
        "architecture_id": DEPCarNetV3.architecture_id,
        "objective_id": DEPCarObjectiveV3.objective_id,
        "training_stage": "joint_gear_score_calibration",
        "modality": "fusion",
        "artifact_role": "best",
        "run_completed": True,
        "partial_epoch": False,
        "training_config_sha256": base_contract["training_config_sha256"],
        "trainer_sha256": base_contract["trainer_sha256"],
        "production_qualified": False,
    }
    mismatched = [
        key for key, expected in required_payload.items()
        if payload.get(key) != expected
    ]
    if payload.get("data_authority_gate") != data_gate:
        mismatched.append("data_authority_gate")
    if (
        contract.get("checkpoint_sha256") != source["checkpoint_sha256"]
        or acceptance.get("schema") != "DEPCarJointGearV3AcceptanceV1"
        or acceptance.get("stage") != "joint_gear_score_calibration"
        or acceptance.get("status") != "PASS"
        or acceptance.get("gate_passed") is not True
        or acceptance.get("checkpoint_sha256") != source["checkpoint_sha256"]
        or acceptance.get("test_split_accessed") is not False
        or acceptance.get("active_control_authorized") is not False
        or acceptance.get("production_qualified") is not False
    ):
        mismatched.append("acceptance_lineage")
    if mismatched:
        raise RuntimeError("V3.1 Score source authority failed: " + ",".join(mismatched))
    gate = {
        "schema": "DEPCarJointGearV31SourceGateV1",
        "passed": True,
        "errors": [],
        "checkpoint": str(path),
        "checkpoint_sha256": sha256_file(path),
        "checkpoint_contract": str(contract_path),
        "checkpoint_contract_sha256": sha256_file(contract_path),
        "acceptance": str(acceptance_path),
        "acceptance_sha256": sha256_file(acceptance_path),
        "accepted_stage": "joint_gear_score_calibration",
        "failed_sequence_checkpoint_inherited": False,
        "test_split_accessed": False,
    }
    return path, payload, gate


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_score_only(model):
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    selected = tuple(model.score_parameters())
    for parameter in selected:
        parameter.requires_grad_(True)
    selected_ids = {id(parameter) for parameter in selected}
    if len(selected_ids) != len(selected):
        raise RuntimeError("V3.1 score parameter partition overlaps internally")
    return {
        "stage": STAGE,
        "candidate_trainable": 0,
        "score_trainable": sum(parameter.numel() for parameter in selected),
        "frozen_parameters": sum(
            parameter.numel() for parameter in model.parameters()
            if id(parameter) not in selected_ids
        ),
    }, selected


def forward_loss(model, objective, batch, amp, dropout=0.0):
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
            sequence_mask=batch["sequence_mask"],
        )
    return output, losses


_TOTAL_NAMES = (
    "weighted_loss", "base_score_loss", "bank_cross_entropy",
    "bank_direction_margin", "feasible_candidate_margin",
    "feasible_candidates", "zero_feasible", "selected_hard",
    "selected_reverse", "unnecessary_reverse", "forward_available",
    "required_reverse", "selected_reverse_required", "oracle_regret",
    "oracle_regret_count", "forward_bank_capable", "reverse_bank_capable",
    "oracle_reverse_required", "teacher_true_positive_required",
    "teacher_false_negative_required", "teacher_false_positive_required",
    "teacher_true_negative_required", "required_no_hard_forward",
    "selected_reverse_no_hard_forward", "multi_action",
    "multi_action_teacher_correct",
)


def epoch_loop(
    model, objective, loader, device, amp, *, optimizer=None, scaler=None,
    max_steps=None, dropout=0.0, progress_interval=50, gradient_clip=5.0,
):
    training = optimizer is not None
    model.eval()
    totals = torch.zeros(len(_TOTAL_NAMES), dtype=torch.float64, device=device)
    samples = steps = 0
    finite = torch.ones((), dtype=torch.bool, device=device)
    started = time.monotonic()
    context = torch.enable_grad if training else torch.inference_mode
    with context():
        for host in loader:
            if max_steps is not None and steps >= max_steps:
                break
            batch = v3.select_valid(host, device)
            if batch is None:
                continue
            if training:
                optimizer.zero_grad(set_to_none=True)
            output, losses = forward_loss(
                model, objective, batch, amp,
                dropout=dropout if training else 0.0,
            )
            finite.logical_and_(torch.isfinite(losses["total"].detach()))
            if training:
                scaler.scale(losses["total"]).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    float(gradient_clip),
                )
                scaler.step(optimizer)
                scaler.update()

            count = len(batch["state"])
            feasible = losses["hard_feasible"]
            selected_index = output.scores.detach().argmin(dim=1)
            selected = selected_index[:, None]
            selected_hard = feasible.gather(1, selected).squeeze(1)
            selected_reverse = selected_index >= 15
            forward_available = losses["forward_available"]
            reverse_feasible = feasible[:, 15:].any(dim=1)
            required = losses["required_reverse"]
            no_hard_forward = losses["no_hard_forward"]
            teacher_reverse = losses["teacher_reverse"]
            unnecessary_reverse = selected_reverse & forward_available
            any_feasible = feasible.any(dim=1)
            cost = losses["candidate_cost"]
            oracle = cost.masked_fill(~feasible, torch.inf).amin(dim=1)
            selected_cost = cost.gather(1, selected).squeeze(1)
            regret = torch.where(
                any_feasible,
                (selected_cost - oracle).clamp_min(0.0),
                torch.zeros_like(selected_cost),
            )
            multi_action = losses["multi_action"]
            selected_teacher = selected_reverse == teacher_reverse
            values = (
                losses["total"].detach().double() * count,
                losses["base_score_total"].double() * count,
                losses["bank_cross_entropy"].detach().double() * count,
                losses["bank_direction_margin"].detach().double() * count,
                losses["feasible_candidate_margin"].detach().double() * count,
                feasible.sum().double(),
                (~any_feasible).sum().double(),
                selected_hard.sum().double(),
                selected_reverse.sum().double(),
                unnecessary_reverse.sum().double(),
                forward_available.sum().double(),
                required.sum().double(),
                (required & selected_reverse).sum().double(),
                regret.sum().double(),
                any_feasible.sum().double(),
                feasible[:, :15].any(dim=1).sum().double(),
                reverse_feasible.sum().double(),
                (required & teacher_reverse).sum().double(),
                (required & selected_reverse & teacher_reverse).sum().double(),
                (required & ~selected_reverse & teacher_reverse).sum().double(),
                (required & selected_reverse & ~teacher_reverse).sum().double(),
                (required & ~selected_reverse & ~teacher_reverse).sum().double(),
                no_hard_forward.sum().double(),
                (no_hard_forward & selected_reverse).sum().double(),
                multi_action.sum().double(),
                (multi_action & selected_teacher).sum().double(),
            )
            totals += torch.stack(values)
            samples += count
            steps += 1
            if training and steps % int(progress_interval) == 0:
                elapsed = max(1.0e-6, time.monotonic() - started)
                print(
                    "step=%d samples=%d loss=%.6f samples_per_s=%.1f" % (
                        steps, samples, float(losses["total"].detach()),
                        samples / elapsed,
                    ),
                    flush=True,
                )
    if samples == 0:
        raise RuntimeError("V3.1 loader produced no valid samples")
    if not bool(finite.cpu()):
        raise FloatingPointError("V3.1 objective became non-finite")
    values = dict(zip(_TOTAL_NAMES, totals.cpu().tolist()))
    denominator = max(1, samples)
    oracle_reverse = max(1.0, values["oracle_reverse_required"])
    oracle_forward = max(
        1.0, values["required_reverse"] - values["oracle_reverse_required"]
    )
    return {
        "samples": samples,
        "steps": steps,
        "mean_loss": values["weighted_loss"] / denominator,
        "mean_base_score_loss": values["base_score_loss"] / denominator,
        "mean_bank_cross_entropy": values["bank_cross_entropy"] / denominator,
        "mean_bank_direction_margin": values["bank_direction_margin"] / denominator,
        "mean_feasible_candidate_margin": values["feasible_candidate_margin"] / denominator,
        "mean_feasible_candidates": values["feasible_candidates"] / denominator,
        "zero_hard_feasible_rate": values["zero_feasible"] / denominator,
        "selected_hard_feasible_rate": values["selected_hard"] / denominator,
        "selected_reverse_rate": values["selected_reverse"] / denominator,
        "unnecessary_reverse_rate": values["unnecessary_reverse"]
        / max(1.0, values["forward_available"]),
        "forward_available_samples": int(values["forward_available"]),
        "required_reverse_samples": int(values["required_reverse"]),
        "required_reverse_selection_rate": values["selected_reverse_required"]
        / max(1.0, values["required_reverse"]),
        "mean_oracle_regret": values["oracle_regret"]
        / max(1.0, values["oracle_regret_count"]),
        "forward_bank_capable_rate": values["forward_bank_capable"] / denominator,
        "reverse_bank_capable_rate": values["reverse_bank_capable"] / denominator,
        "oracle_reverse_required": int(values["oracle_reverse_required"]),
        "bank_true_positive_required": int(
            values["teacher_true_positive_required"]
        ),
        "bank_false_negative_required": int(
            values["teacher_false_negative_required"]
        ),
        "bank_false_positive_required": int(
            values["teacher_false_positive_required"]
        ),
        "bank_true_negative_required": int(
            values["teacher_true_negative_required"]
        ),
        "oracle_reverse_prevalence_within_required": values[
            "oracle_reverse_required"
        ] / max(1.0, values["required_reverse"]),
        "oracle_reverse_recall_within_required": values[
            "teacher_true_positive_required"
        ] / oracle_reverse,
        "oracle_forward_false_reverse_rate_within_required": values[
            "teacher_false_positive_required"
        ] / oracle_forward,
        "required_no_hard_forward": int(values["required_no_hard_forward"]),
        "selected_reverse_no_hard_forward": int(
            values["selected_reverse_no_hard_forward"]
        ),
        "no_hard_forward_reverse_selection_rate": values[
            "selected_reverse_no_hard_forward"
        ] / max(1.0, values["required_no_hard_forward"]),
        "multi_action_samples": int(values["multi_action"]),
        "multi_action_teacher_accuracy": values["multi_action_teacher_correct"]
        / max(1.0, values["multi_action"]),
        "elapsed_s": time.monotonic() - started,
    }


def sequence_acceptance_checks(metrics, acceptance_contract):
    sequence = acceptance_contract["sequence"]
    candidate = acceptance_contract["candidate"]
    return {
        "overall_zero_hard_feasible_rate": metrics["zero_hard_feasible_rate"]
        <= float(candidate["maximum_zero_hard_feasible_rate"]),
        "forward_bank_capable_rate": metrics["forward_bank_capable_rate"]
        >= float(candidate["minimum_forward_bank_capable_rate"]),
        "reverse_bank_capable_rate": metrics["reverse_bank_capable_rate"]
        >= float(candidate["minimum_reverse_bank_capable_rate"]),
        "selected_hard_feasible_rate": metrics["selected_hard_feasible_rate"]
        >= float(sequence["minimum_selected_hard_feasible_rate"]),
        "unnecessary_reverse_rate": metrics["unnecessary_reverse_rate"]
        <= float(sequence["maximum_unnecessary_reverse_rate"]),
        "oracle_reverse_recall_within_required": metrics[
            "oracle_reverse_recall_within_required"
        ] >= float(sequence["minimum_oracle_reverse_recall_within_required"]),
        "oracle_forward_false_reverse_rate_within_required": metrics[
            "oracle_forward_false_reverse_rate_within_required"
        ] <= float(sequence["maximum_oracle_forward_false_reverse_rate_within_required"]),
        "no_hard_forward_reverse_selection_rate": metrics[
            "no_hard_forward_reverse_selection_rate"
        ] >= float(sequence["minimum_no_hard_forward_reverse_selection_rate"]),
    }


def sequence_selection_key(metrics, acceptance_contract):
    """Select the closest final-acceptance epoch, not merely the safest one."""

    sequence = acceptance_contract["sequence"]
    checks = sequence_acceptance_checks(metrics, acceptance_contract)
    lower = (
        (
            "selected_hard_feasible_rate",
            "minimum_selected_hard_feasible_rate",
        ),
        (
            "oracle_reverse_recall_within_required",
            "minimum_oracle_reverse_recall_within_required",
        ),
        (
            "no_hard_forward_reverse_selection_rate",
            "minimum_no_hard_forward_reverse_selection_rate",
        ),
    )
    upper = (
        ("unnecessary_reverse_rate", "maximum_unnecessary_reverse_rate"),
        (
            "oracle_forward_false_reverse_rate_within_required",
            "maximum_oracle_forward_false_reverse_rate_within_required",
        ),
    )
    violation = 0.0
    for metric, threshold in lower:
        target = float(sequence[threshold])
        violation += max(0.0, target - float(metrics[metric])) / max(target, 1.0e-6)
    for metric, threshold in upper:
        target = float(sequence[threshold])
        violation += max(0.0, float(metrics[metric]) - target) / max(target, 1.0e-3)
    final_names = {name for pair in lower + upper for name in pair[:1]}
    failed_final = sum(
        1 for name, passed in checks.items()
        if name in final_names and not passed
    )
    return (
        failed_final,
        violation,
        -metrics["oracle_reverse_recall_within_required"],
        -metrics["no_hard_forward_reverse_selection_rate"],
        metrics["oracle_forward_false_reverse_rate_within_required"],
        metrics["unnecessary_reverse_rate"],
        -metrics["selected_hard_feasible_rate"],
        metrics["mean_oracle_regret"],
    )


def atomic_torch(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def write_artifact(path, model, optimizer, scaler, metadata, metrics, history):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": CHECKPOINT_SCHEMA,
        "architecture_id": DEPCarNetV3.architecture_id,
        "objective_id": DEPCarObjectiveV31.objective_id,
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
    config, config_sha, base_config, acceptance_contract = load_config()
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

    bundle_path, bundle, sequence_path, _authority, data_gate = (
        v3.verify_data_authority(base_config)
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
        raise RuntimeError("V3.1 run is neither formal nor a bounded diagnostic")
    if output == source_path:
        raise RuntimeError("V3.1 source and output must differ")
    if not args.dry_run and output.exists() and args.resume is None:
        raise RuntimeError("V3.1 output exists; use --resume with the canonical last artifact")
    if args.resume is not None and resolve(args.resume) != output:
        raise RuntimeError("V3.1 resume must point to the selected output")
    plan = {
        "schema": "DEPCarJointGearV31TrainingPlanV1",
        "status": (
            "DRY_RUN_READY" if args.dry_run and formal
            else "BOUNDED_DIAGNOSTIC_READY" if bounded else "READY"
        ),
        "stage": STAGE,
        "architecture_id": DEPCarNetV3.architecture_id,
        "objective_id": DEPCarObjectiveV31.objective_id,
        "source": str(source_path),
        "source_sha256": sha256_file(source_path),
        "failed_sequence_checkpoint_inherited": False,
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
        "selection_uses_final_acceptance_metrics": True,
        "test_split_sealed": True,
        "active_control_authorized": False,
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
    model.load_state_dict(source["model_state_dict"], strict=True)
    model.to(device)
    ownership, selected_parameters = configure_score_only(model)
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
        train_data = Subset(train_data, range(min(len(train_data), args.max_samples)))
        selected = np.linspace(
            0, len(validation_data) - 1,
            min(len(validation_data), args.max_samples), dtype=np.int64,
        )
        validation_data = Subset(validation_data, selected.tolist())
    loader_common = dict(
        batch_size=batch_size, num_workers=workers,
        pin_memory=device.type == "cuda", persistent_workers=workers > 0,
        prefetch_factor=int(training["prefetch_factor"]) if workers > 0 else None,
        collate_fn=p3_training_collate, worker_init_fn=p3_training_worker_init,
    )
    train_loader = DataLoader(train_data, shuffle=True, **loader_common)
    validation_loader = DataLoader(validation_data, shuffle=False, **loader_common)
    objective = DEPCarObjectiveV31(
        DEPCarJointGearLossConfigV3(**base_config["loss"]),
        DEPCarSequenceCorrectionConfigV31(**config["correction"]),
    )
    history, global_step, best_key, start_epoch = [], 0, None, 1
    if args.resume:
        resumed = torch.load(output, map_location="cpu", weights_only=True)
        required = {
            "schema": CHECKPOINT_SCHEMA,
            "architecture_id": DEPCarNetV3.architecture_id,
            "objective_id": DEPCarObjectiveV31.objective_id,
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
            raise RuntimeError("V3.1 resume mismatch: " + ",".join(mismatch))
        completed = int(resumed.get("completed_epochs", 0))
        if not 1 <= completed < epochs:
            raise RuntimeError("V3.1 resume epoch is outside remaining run")
        model.load_state_dict(resumed["model_state_dict"], strict=True)
        optimizer.load_state_dict(resumed["optimizer_state_dict"])
        scaler.load_state_dict(resumed["grad_scaler_state_dict"])
        history = list(resumed.get("history", ()))
        global_step = int(resumed.get("global_step", 0))
        start_epoch = completed + 1
        keys = [
            sequence_selection_key(row["validation"], acceptance_contract)
            for row in history
        ]
        best_key = min(keys) if keys else None
        model.to(device)

    if not args.resume:
        baseline_metrics = epoch_loop(
            model, objective, validation_loader, device, amp,
            max_steps=args.max_steps,
            progress_interval=int(training["progress_interval_steps"]),
        )
        baseline_gate = sequence_acceptance_checks(
            baseline_metrics, acceptance_contract
        )
        baseline_row = {
            "epoch": 0,
            "phase": "accepted_score_initialization_baseline",
            "train": None,
            "validation": baseline_metrics,
            "selection_gate": {
                key: "PASS" if value else "FAIL"
                for key, value in baseline_gate.items()
            },
            "elapsed_s": baseline_metrics["elapsed_s"],
        }
        history.append(baseline_row)
        best_key = sequence_selection_key(
            baseline_metrics, acceptance_contract
        )
        baseline_metadata = {
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
            _best(output), model, optimizer, scaler, baseline_metadata,
            baseline_metrics, history,
        )
        print(json.dumps(baseline_row, sort_keys=True), flush=True)

    for epoch in range(start_epoch, epochs + 1):
        started = time.time()
        train_metrics = epoch_loop(
            model, objective, train_loader, device, amp,
            optimizer=optimizer, scaler=scaler, max_steps=args.max_steps,
            dropout=float(training["sensor_dropout_probability"]),
            progress_interval=int(training["progress_interval_steps"]),
            gradient_clip=float(training["gradient_clip"]),
        )
        validation_metrics = epoch_loop(
            model, objective, validation_loader, device, amp,
            max_steps=args.max_steps,
            progress_interval=int(training["progress_interval_steps"]),
        )
        global_step += train_metrics["steps"]
        selection_gate = sequence_acceptance_checks(
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
        key = sequence_selection_key(validation_metrics, acceptance_contract)
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
        raise RuntimeError("V3.1 training ended without best checkpoint")
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
