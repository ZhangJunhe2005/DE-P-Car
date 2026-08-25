#!/usr/bin/env python3
"""Train the V3.3 explicit gear selector over a frozen V3.1 trajectory policy."""

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
from dep_car.model.gear_selector_v33 import (
    DEPCarGearSelectorV33,
    GearSelectorConfigV33,
)
from dep_car.training.losses_v3 import DEPCarJointGearLossConfigV3
from dep_car.training.losses_v31 import DEPCarSequenceCorrectionConfigV31
from dep_car.training.losses_v33 import (
    DEPCarExplicitGearLossConfigV33,
    DEPCarObjectiveV33,
)
from dep_car.training.p4_dataset import p3_training_collate, p3_training_worker_init
from dep_car.training.score_dataset import P3JointGearDatasetV3
from dep_car.training.stages import apply_sensor_dropout
import train_dep_car_sequence_v31 as v31


CONFIG = ROOT / "dep_car/config/p5_joint_gear_v33_explicit_selector.yaml"
TRAINER = Path(__file__).resolve()
STAGE = "explicit_gear_selector"
CHECKPOINT_SCHEMA = "DEPCarJointGearV33CheckpointV1"
CONTRACT_SCHEMA = "DEPCarJointGearV33ArtifactContractV1"

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
        config.get("schema") != "DEPCarJointGearV33TrainingContractV1"
        or config.get("architecture_id") != DEPCarGearSelectorV33.architecture_id
        or config.get("objective_id") != DEPCarObjectiveV33.objective_id
        or config.get("scope") != "fusion_only_explicit_gear_over_frozen_v31"
        or config.get("test_split_sealed") is not True
        or config.get("qualification", {}).get("base_policy_frozen") is not True
        or config.get("qualification", {}).get(
            "selection_uses_explicit_pre_veto_gear_metrics"
        ) is not True
        or config.get("qualification", {}).get(
            "hard_safety_fallback_is_reported"
        ) is not True
        or config.get("qualification", {}).get(
            "active_control_authorized_by_training"
        ) is not False
        or config.get("qualification", {}).get(
            "production_qualified_by_training"
        ) is not False
    ):
        raise RuntimeError("V3.3 training contract identity is invalid")
    GearSelectorConfigV33(**config["model"]).validate()
    DEPCarExplicitGearLossConfigV33(**config["selector_loss"]).validate()
    implementation = config["implementation"]
    for key in ("model", "loss"):
        if sha256_file(resolve(implementation[key])) != implementation[
            key + "_sha256"
        ]:
            raise RuntimeError("V3.3 %s implementation hash differs" % key)
    base = config["base_contract"]
    frozen = (
        (base["training_config"], base["training_config_sha256"]),
        (base["trainer"], base["trainer_sha256"]),
        (base["loss"], base["loss_sha256"]),
        (base["acceptance_contract"], base["acceptance_contract_sha256"]),
    )
    mismatched = [
        str(resolve(path))
        for path, expected in frozen
        if sha256_file(resolve(path)) != expected
    ]
    if mismatched:
        raise RuntimeError("V3.3 frozen base contract differs: " + ",".join(mismatched))
    v31_config, v31_sha, base_config, _old_acceptance = v31.load_config()
    if v31_sha != base["training_config_sha256"]:
        raise RuntimeError("V3.3 did not load the exact V3.1 training contract")
    acceptance = yaml.safe_load(
        resolve(base["acceptance_contract"]).read_text(encoding="utf-8")
    )
    if (
        acceptance.get("schema") != "DEPCarJointGearV33AcceptanceContractV1"
        or acceptance.get("revision") != 1
        or acceptance.get("test_split_sealed") is not True
        or acceptance.get("active_control_authorized_by_acceptance") is not False
        or acceptance.get("production_qualified_by_acceptance") is not False
    ):
        raise RuntimeError("V3.3 acceptance contract identity is invalid")
    return (
        config,
        hashlib.sha256(raw).hexdigest(),
        v31_config,
        base_config,
        acceptance,
    )


def verify_source(config, data_gate):
    source = config["initialization"]
    path = resolve(source["checkpoint"])
    contract_path = resolve(source["checkpoint_contract"])
    acceptance_path = resolve(source["acceptance"])
    for actual_path, expected in (
        (path, source["checkpoint_sha256"]),
        (contract_path, source["checkpoint_contract_sha256"]),
        (acceptance_path, source["acceptance_sha256"]),
    ):
        if sha256_file(actual_path) != expected:
            raise RuntimeError("V3.3 V3.1 source hash differs: %s" % actual_path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    contract = read_json(contract_path)
    acceptance = read_json(acceptance_path)
    required = {
        "schema": "DEPCarJointGearV31CheckpointV1",
        "architecture_id": DEPCarNetV3.architecture_id,
        "objective_id": "dep_car_objective_v7_sequence_teacher_balanced_feasibility_margin",
        "training_stage": "sequence_correction",
        "artifact_role": "best",
        "run_completed": True,
        "partial_epoch": False,
        "training_config_sha256": config["base_contract"]["training_config_sha256"],
        "trainer_sha256": config["base_contract"]["trainer_sha256"],
        "production_qualified": False,
    }
    errors = [key for key, value in required.items() if payload.get(key) != value]
    allowed = sorted(source["allowed_failed_checks"])
    if payload.get("data_authority_gate") != data_gate:
        errors.append("data_authority_gate")
    if (
        contract.get("checkpoint_sha256") != source["checkpoint_sha256"]
        or acceptance.get("schema") != "DEPCarJointGearV31AcceptanceV1"
        or acceptance.get("stage") != "sequence_correction"
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
        raise RuntimeError("V3.3 source authority failed: " + ",".join(errors))
    gate = {
        "schema": "DEPCarJointGearV33SourceGateV1",
        "passed": True,
        "errors": [],
        "checkpoint": str(path),
        "checkpoint_sha256": sha256_file(path),
        "checkpoint_contract": str(contract_path),
        "checkpoint_contract_sha256": sha256_file(contract_path),
        "acceptance": str(acceptance_path),
        "acceptance_sha256": sha256_file(acceptance_path),
        "accepted_for_explicit_selector_initialization_only": True,
        "allowed_failed_checks": allowed,
        "base_policy_frozen": True,
        "p6_authorized_by_source": False,
        "test_split_accessed": False,
    }
    return path, payload, gate


def build_model(config, source):
    base = DEPCarNetV3()
    base.load_state_dict(source["model_state_dict"], strict=True)
    model = DEPCarGearSelectorV33(
        base_model=base,
        selector_config=GearSelectorConfigV33(**config["model"]),
    )
    model.freeze_base()
    return model


def configure_selector_only(model):
    model.freeze_base()
    for parameter in model.gear_selector.parameters():
        parameter.requires_grad_(True)
    selected = tuple(model.selector_parameters())
    selected_ids = {id(parameter) for parameter in selected}
    if len(selected_ids) != len(selected):
        raise RuntimeError("V3.3 selector parameter partition overlaps")
    unexpected = [
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad and id(parameter) not in selected_ids
    ]
    if unexpected:
        raise RuntimeError("V3.3 unexpectedly unfrozen parameters: " + ",".join(unexpected))
    return {
        "stage": STAGE,
        "base_policy_trainable": 0,
        "explicit_selector_trainable": sum(p.numel() for p in selected),
        "frozen_parameters": sum(
            p.numel() for p in model.parameters() if id(p) not in selected_ids
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
    "loss", "cross_entropy", "margin", "frozen_base", "feasible_candidates",
    "zero_feasible", "forward_capable", "reverse_capable",
    "requested_reverse", "unnecessary_reverse", "forward_available",
    "required_reverse", "requested_reverse_required", "oracle_reverse_required",
    "true_positive_required", "false_negative_required",
    "false_positive_required", "true_negative_required", "no_hard_forward",
    "reverse_no_hard_forward", "requested_bank_available", "safety_fallback",
    "post_veto_selected_hard", "pre_veto_selected_hard", "oracle_regret",
    "oracle_regret_count", "multi_action", "multi_action_correct",
)


def epoch_loop(
    model, objective, loader, device, amp, *, optimizer=None, scaler=None,
    max_steps=None, dropout=0.0, progress_interval=50, gradient_clip=5.0,
):
    training = optimizer is not None
    model.train(training)
    totals = torch.zeros(len(_TOTAL_NAMES), dtype=torch.float64, device=device)
    samples = steps = 0
    finite = torch.ones((), dtype=torch.bool, device=device)
    started = time.monotonic()
    context = torch.enable_grad if training else torch.inference_mode
    with context():
        for host in loader:
            if max_steps is not None and steps >= max_steps:
                break
            batch = v31.v3.select_valid(host, device)
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
            feasible = losses["hard_feasible"].bool()
            any_feasible = feasible.any(dim=1)
            forward_capable = feasible[:, :15].any(dim=1)
            reverse_capable = feasible[:, 15:].any(dim=1)
            request = losses["requested_reverse"].bool()
            teacher = losses["teacher_reverse"].bool()
            required = losses["required_reverse"].bool()
            no_forward = losses["no_hard_forward"].bool()
            forward_available = losses["forward_available"].bool()
            selected = losses["selected_index"][:, None]
            selected_hard = feasible.gather(1, selected).squeeze(1)

            request_mask = torch.zeros_like(feasible)
            request_mask[:, :15] = ~request[:, None]
            request_mask[:, 15:] = request[:, None]
            raw_request_score = output.scores.detach().float().masked_fill(
                ~request_mask, torch.inf
            )
            raw_request_index = raw_request_score.argmin(dim=1)[:, None]
            pre_veto_hard = feasible.gather(1, raw_request_index).squeeze(1)
            cost = losses["candidate_cost"]
            oracle = cost.masked_fill(~feasible, torch.inf).amin(dim=1)
            selected_cost = cost.gather(1, selected).squeeze(1)
            regret = torch.where(
                any_feasible,
                (selected_cost - oracle).clamp_min(0.0),
                torch.zeros_like(selected_cost),
            )
            multi = losses["multi_action"].bool()
            values = (
                losses["total"].detach().double() * count,
                losses["explicit_gear_cross_entropy"].detach().double() * count,
                losses["explicit_gear_margin"].detach().double() * count,
                losses["frozen_base_total"].double() * count,
                feasible.sum().double(),
                (~any_feasible).sum().double(),
                forward_capable.sum().double(),
                reverse_capable.sum().double(),
                request.sum().double(),
                (request & forward_available).sum().double(),
                forward_available.sum().double(),
                required.sum().double(),
                (required & request).sum().double(),
                (required & teacher).sum().double(),
                (required & teacher & request).sum().double(),
                (required & teacher & ~request).sum().double(),
                (required & ~teacher & request).sum().double(),
                (required & ~teacher & ~request).sum().double(),
                no_forward.sum().double(),
                (no_forward & request).sum().double(),
                losses["requested_bank_available"].sum().double(),
                losses["hard_safety_fallback"].sum().double(),
                selected_hard.sum().double(),
                pre_veto_hard.sum().double(),
                regret.sum().double(),
                any_feasible.sum().double(),
                multi.sum().double(),
                (multi & (request == teacher)).sum().double(),
            )
            totals += torch.stack(values)
            samples += count
            steps += 1
            if training and steps % int(progress_interval) == 0:
                elapsed = max(1.0e-6, time.monotonic() - started)
                print(
                    "step=%d samples=%d loss=%.6f samples_per_s=%.1f" % (
                        steps, samples, float(losses["total"].detach()), samples / elapsed
                    ),
                    flush=True,
                )
    if samples == 0:
        raise RuntimeError("V3.3 loader produced no valid samples")
    if not bool(finite.cpu()):
        raise FloatingPointError("V3.3 objective became non-finite")
    value = dict(zip(_TOTAL_NAMES, totals.cpu().tolist()))
    n = max(1, samples)
    oracle_reverse = max(1.0, value["oracle_reverse_required"])
    oracle_forward = max(1.0, value["required_reverse"] - value["oracle_reverse_required"])
    return {
        "samples": samples,
        "steps": steps,
        "mean_loss": value["loss"] / n,
        "mean_explicit_gear_cross_entropy": value["cross_entropy"] / n,
        "mean_explicit_gear_margin": value["margin"] / n,
        "mean_frozen_base_loss": value["frozen_base"] / n,
        "mean_feasible_candidates": value["feasible_candidates"] / n,
        "zero_hard_feasible_rate": value["zero_feasible"] / n,
        "forward_bank_capable_rate": value["forward_capable"] / n,
        "reverse_bank_capable_rate": value["reverse_capable"] / n,
        "requested_reverse_rate": value["requested_reverse"] / n,
        "unnecessary_reverse_rate": value["unnecessary_reverse"]
        / max(1.0, value["forward_available"]),
        "forward_available_samples": int(value["forward_available"]),
        "required_reverse_samples": int(value["required_reverse"]),
        "required_reverse_selection_rate": value["requested_reverse_required"]
        / max(1.0, value["required_reverse"]),
        "oracle_reverse_required": int(value["oracle_reverse_required"]),
        "oracle_reverse_recall_within_required": value["true_positive_required"]
        / oracle_reverse,
        "oracle_forward_false_reverse_rate_within_required": value[
            "false_positive_required"
        ] / oracle_forward,
        "bank_true_positive_required": int(value["true_positive_required"]),
        "bank_false_negative_required": int(value["false_negative_required"]),
        "bank_false_positive_required": int(value["false_positive_required"]),
        "bank_true_negative_required": int(value["true_negative_required"]),
        "required_no_hard_forward": int(value["no_hard_forward"]),
        "selected_reverse_no_hard_forward": int(value["reverse_no_hard_forward"]),
        "no_hard_forward_reverse_selection_rate": value["reverse_no_hard_forward"]
        / max(1.0, value["no_hard_forward"]),
        "requested_bank_hard_available_rate": value["requested_bank_available"] / n,
        "hard_safety_fallback_rate": value["safety_fallback"] / n,
        "post_veto_selected_hard_feasible_rate": value["post_veto_selected_hard"] / n,
        "pre_veto_selected_hard_feasible_rate": value["pre_veto_selected_hard"] / n,
        "mean_oracle_regret": value["oracle_regret"]
        / max(1.0, value["oracle_regret_count"]),
        "multi_action_samples": int(value["multi_action"]),
        "multi_action_teacher_accuracy": value["multi_action_correct"]
        / max(1.0, value["multi_action"]),
        "elapsed_s": time.monotonic() - started,
    }


def acceptance_checks(metrics, contract):
    candidate = contract["candidate"]
    selector = contract["selector"]
    execution = contract["execution"]
    return {
        "overall_zero_hard_feasible_rate": metrics["zero_hard_feasible_rate"]
        <= float(candidate["maximum_zero_hard_feasible_rate"]),
        "forward_bank_capable_rate": metrics["forward_bank_capable_rate"]
        >= float(candidate["minimum_forward_bank_capable_rate"]),
        "reverse_bank_capable_rate": metrics["reverse_bank_capable_rate"]
        >= float(candidate["minimum_reverse_bank_capable_rate"]),
        "unnecessary_reverse_rate": metrics["unnecessary_reverse_rate"]
        <= float(selector["maximum_unnecessary_reverse_rate"]),
        "oracle_reverse_recall_within_required": metrics[
            "oracle_reverse_recall_within_required"
        ] >= float(selector["minimum_oracle_reverse_recall_within_required"]),
        "oracle_forward_false_reverse_rate_within_required": metrics[
            "oracle_forward_false_reverse_rate_within_required"
        ] <= float(selector["maximum_oracle_forward_false_reverse_rate_within_required"]),
        "no_hard_forward_reverse_selection_rate": metrics[
            "no_hard_forward_reverse_selection_rate"
        ] >= float(selector["minimum_no_hard_forward_reverse_selection_rate"]),
        "requested_bank_hard_available_rate": metrics[
            "requested_bank_hard_available_rate"
        ] >= float(execution["minimum_requested_bank_hard_available_rate"]),
        "hard_safety_fallback_rate": metrics["hard_safety_fallback_rate"]
        <= float(execution["maximum_hard_safety_fallback_rate"]),
        "post_veto_selected_hard_feasible_rate": metrics[
            "post_veto_selected_hard_feasible_rate"
        ] >= float(execution["minimum_post_veto_selected_hard_feasible_rate"]),
    }


def selection_key(metrics, contract):
    checks = acceptance_checks(metrics, contract)
    selector = contract["selector"]
    execution = contract["execution"]
    lower = (
        ("oracle_reverse_recall_within_required", selector["minimum_oracle_reverse_recall_within_required"]),
        ("no_hard_forward_reverse_selection_rate", selector["minimum_no_hard_forward_reverse_selection_rate"]),
        ("requested_bank_hard_available_rate", execution["minimum_requested_bank_hard_available_rate"]),
        ("post_veto_selected_hard_feasible_rate", execution["minimum_post_veto_selected_hard_feasible_rate"]),
    )
    upper = (
        ("unnecessary_reverse_rate", selector["maximum_unnecessary_reverse_rate"]),
        ("oracle_forward_false_reverse_rate_within_required", selector["maximum_oracle_forward_false_reverse_rate_within_required"]),
        ("hard_safety_fallback_rate", execution["maximum_hard_safety_fallback_rate"]),
    )
    violation = sum(
        max(0.0, float(target) - metrics[name]) / max(float(target), 1.0e-6)
        for name, target in lower
    ) + sum(
        max(0.0, metrics[name] - float(target)) / max(float(target), 1.0e-3)
        for name, target in upper
    )
    return (
        sum(not passed for passed in checks.values()),
        violation,
        -metrics["oracle_reverse_recall_within_required"],
        -metrics["no_hard_forward_reverse_selection_rate"],
        metrics["oracle_forward_false_reverse_rate_within_required"],
        metrics["hard_safety_fallback_rate"],
        metrics["unnecessary_reverse_rate"],
        metrics["mean_loss"],
    )


def atomic_torch(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def write_artifact(path, model, optimizer, scaler, metadata, metrics, history):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": CHECKPOINT_SCHEMA,
        "architecture_id": DEPCarGearSelectorV33.architecture_id,
        "base_architecture_id": DEPCarNetV3.architecture_id,
        "objective_id": DEPCarObjectiveV33.objective_id,
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
        "base_policy_frozen": True,
        "explicit_gear_logit_order": list(DEPCarGearSelectorV33.gear_logit_order),
        "model_implementation_sha256": metadata["model_implementation_sha256"],
        "loss_implementation_sha256": metadata["loss_implementation_sha256"],
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
            "architecture_id", "base_architecture_id", "objective_id",
            "training_stage", "modality", "artifact_role", "status",
            "qualification_status", "active_control_authorized",
            "production_qualified", "completed_epochs", "selected_epoch",
            "partial_epoch", "global_step", "source_checkpoint",
            "source_checkpoint_sha256", "data_authority_gate", "source_gate",
            "training_config_sha256", "trainer_sha256",
            "acceptance_contract_sha256", "base_policy_frozen",
            "explicit_gear_logit_order", "model_implementation_sha256",
            "loss_implementation_sha256", "metrics", "selection_gate",
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
    config, config_sha, v31_config, base_config, acceptance = load_config()
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
        and output == expected_output and data_gate["passed"] and source_gate["passed"]
    )
    if not args.dry_run and not formal and not bounded:
        raise RuntimeError("V3.3 run is neither formal nor a bounded diagnostic")
    if output == source_path:
        raise RuntimeError("V3.3 source and output must differ")
    if not args.dry_run and output.exists() and args.resume is None:
        raise RuntimeError("V3.3 output exists; use --resume with the canonical last artifact")
    if args.resume is not None and resolve(args.resume) != output:
        raise RuntimeError("V3.3 resume must point to the selected output")

    plan = {
        "schema": "DEPCarJointGearV33TrainingPlanV1",
        "status": "DRY_RUN_READY" if args.dry_run and formal else (
            "BOUNDED_DIAGNOSTIC_READY" if bounded else "READY"
        ),
        "stage": STAGE,
        "architecture_id": DEPCarGearSelectorV33.architecture_id,
        "base_architecture_id": DEPCarNetV3.architecture_id,
        "objective_id": DEPCarObjectiveV33.objective_id,
        "source": str(source_path),
        "source_sha256": sha256_file(source_path),
        "source_scope": "EXPLICIT_SELECTOR_INITIALIZATION_ONLY",
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
        "base_policy_frozen": True,
        "explicit_gear_logit_order": list(DEPCarGearSelectorV33.gear_logit_order),
        "hard_safety_fallback_reported_separately": True,
        "model_implementation_sha256": config["implementation"]["model_sha256"],
        "loss_implementation_sha256": config["implementation"]["loss_sha256"],
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
    model = build_model(config, source).to(device)
    ownership, selected = configure_selector_only(model)
    optimizer = torch.optim.AdamW(
        selected,
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
        train_idx = np.linspace(
            0, len(train_data) - 1, min(len(train_data), args.max_samples), dtype=np.int64
        )
        val_idx = np.linspace(
            0, len(validation_data) - 1,
            min(len(validation_data), args.max_samples), dtype=np.int64,
        )
        train_data = Subset(train_data, train_idx.tolist())
        validation_data = Subset(validation_data, val_idx.tolist())
    loader_common = dict(
        batch_size=batch_size, num_workers=workers,
        pin_memory=device.type == "cuda", persistent_workers=workers > 0,
        prefetch_factor=int(training["prefetch_factor"]) if workers > 0 else None,
        collate_fn=p3_training_collate, worker_init_fn=p3_training_worker_init,
    )
    train_loader = DataLoader(train_data, shuffle=True, **loader_common)
    validation_loader = DataLoader(validation_data, shuffle=False, **loader_common)
    objective = DEPCarObjectiveV33(
        DEPCarJointGearLossConfigV3(**base_config["loss"]),
        DEPCarSequenceCorrectionConfigV31(**v31_config["correction"]),
        DEPCarExplicitGearLossConfigV33(**config["selector_loss"]),
    )

    history, global_step, best_key, start_epoch = [], 0, None, 1
    if args.resume:
        resumed = torch.load(output, map_location="cpu", weights_only=True)
        required = {
            "schema": CHECKPOINT_SCHEMA,
            "architecture_id": DEPCarGearSelectorV33.architecture_id,
            "objective_id": DEPCarObjectiveV33.objective_id,
            "training_stage": STAGE,
            "artifact_role": "last",
            "source_checkpoint_sha256": plan["source_sha256"],
            "training_config_sha256": config_sha,
            "trainer_sha256": plan["trainer_sha256"],
            "base_policy_frozen": True,
        }
        mismatched = [key for key, expected in required.items() if resumed.get(key) != expected]
        if mismatched:
            raise RuntimeError("V3.3 resume mismatch: " + ",".join(mismatched))
        completed = int(resumed.get("completed_epochs", 0))
        if not 1 <= completed < epochs:
            raise RuntimeError("V3.3 resume epoch is outside remaining run")
        model.load_state_dict(resumed["model_state_dict"], strict=True)
        optimizer.load_state_dict(resumed["optimizer_state_dict"])
        scaler.load_state_dict(resumed["grad_scaler_state_dict"])
        history = list(resumed.get("history", ()))
        global_step = int(resumed.get("global_step", 0))
        start_epoch = completed + 1
        keys = [selection_key(row["validation"], acceptance) for row in history]
        best_key = min(keys) if keys else None
        model.to(device)

    if not args.resume:
        baseline = epoch_loop(
            model, objective, validation_loader, device, amp,
            max_steps=args.max_steps,
            progress_interval=int(training["progress_interval_steps"]),
        )
        gate = acceptance_checks(baseline, acceptance)
        row = {
            "epoch": 0,
            "phase": "random_explicit_selector_over_frozen_v31_baseline",
            "train": None,
            "validation": baseline,
            "selection_gate": {key: "PASS" if value else "FAIL" for key, value in gate.items()},
            "elapsed_s": baseline["elapsed_s"],
        }
        history.append(row)
        best_key = selection_key(baseline, acceptance)
        metadata = {
            "artifact_role": "best", "completed_epochs": 0,
            "selected_epoch": 0, "global_step": 0,
            "source_checkpoint": str(source_path),
            "source_checkpoint_sha256": plan["source_sha256"],
            "data_authority_gate": data_gate, "source_gate": source_gate,
            "training_config_sha256": config_sha,
            "trainer_sha256": plan["trainer_sha256"],
            "acceptance_contract_sha256": plan["acceptance_contract_sha256"],
            "model_implementation_sha256": plan["model_implementation_sha256"],
            "loss_implementation_sha256": plan["loss_implementation_sha256"],
            "selection_gate": row["selection_gate"],
        }
        write_artifact(_best(output), model, optimizer, scaler, metadata, baseline, history)
        print(json.dumps(row, sort_keys=True), flush=True)

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
        gate = acceptance_checks(validation_metrics, acceptance)
        row = {
            "epoch": epoch,
            "train": train_metrics,
            "validation": validation_metrics,
            "selection_gate": {key: "PASS" if value else "FAIL" for key, value in gate.items()},
            "elapsed_s": time.time() - started,
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        key = selection_key(validation_metrics, acceptance)
        metadata = {
            "artifact_role": "last", "completed_epochs": epoch,
            "selected_epoch": epoch, "global_step": global_step,
            "source_checkpoint": str(source_path),
            "source_checkpoint_sha256": plan["source_sha256"],
            "data_authority_gate": data_gate, "source_gate": source_gate,
            "training_config_sha256": config_sha,
            "trainer_sha256": plan["trainer_sha256"],
            "acceptance_contract_sha256": plan["acceptance_contract_sha256"],
            "model_implementation_sha256": plan["model_implementation_sha256"],
            "loss_implementation_sha256": plan["loss_implementation_sha256"],
            "selection_gate": row["selection_gate"],
        }
        write_artifact(output, model, optimizer, scaler, metadata, validation_metrics, history)
        if best_key is None or key < best_key:
            best_key = key
            metadata["artifact_role"] = "best"
            write_artifact(
                _best(output), model, optimizer, scaler, metadata,
                validation_metrics, history,
            )

    best_path = _best(output)
    if not best_path.is_file():
        raise RuntimeError("V3.3 training ended without best checkpoint")
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
