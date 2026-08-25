#!/usr/bin/env python3
"""Train DEPCarNetV4 capacity, score and closed-loop sequence stages."""

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
from dep_car.model.dep_car_net_v4 import DEPCarNetV4, HybridSequenceConfigV4
from dep_car.model.hybrid_sequence_rollout import HybridSequenceRolloutConfigV4
from dep_car.training.losses_v4 import (
    DEPCarHybridSequenceLossConfigV4,
    DEPCarObjectiveV4,
    sequence_target_tokens,
)
from dep_car.training.p4_dataset import p3_training_collate, p3_training_worker_init
from dep_car.training.stages import apply_sensor_dropout
from dep_car.training.v4_dataset import P3HybridSequenceDatasetV4
import train_dep_car_joint_gear_v3 as v3


CONFIG = ROOT / "dep_car/config/p5_hybrid_sequence_v4.yaml"
TRAINER = Path(__file__).resolve()
STAGES = (
    "hybrid_sequence_capacity",
    "hybrid_sequence_score",
    "closed_loop_sequence_finetune",
)
STAGE_ARTIFACT = {
    STAGES[0]: "capacity", STAGES[1]: "score", STAGES[2]: "closed_loop"
}
CHECKPOINT_SCHEMA = "DEPCarV4CheckpointV1"
CONTRACT_SCHEMA = "DEPCarV4ArtifactContractV1"


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
        config.get("schema") != "DEPCarV4TrainingContractV1"
        or config.get("architecture_id") != DEPCarNetV4.architecture_id
        or config.get("objective_id") != DEPCarObjectiveV4.objective_id
        or config.get("scope") != "fusion_only_unified_hybrid_gear_control_sequences"
        or config.get("test_split_sealed") is not True
    ):
        raise RuntimeError("V4 training contract identity is invalid")
    implementation = config["implementation"]
    for name in ("model", "rollout", "loss", "dataset"):
        path = resolve(implementation[name])
        if not path.is_file() or sha256_file(path) != implementation[name + "_sha256"]:
            raise RuntimeError("V4 %s implementation hash differs" % name)
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
    model_config.validate(); rollout_config.validate()
    loss_config = DEPCarHybridSequenceLossConfigV4(**config["loss"])
    loss_config.validate()
    return config, hashlib.sha256(raw).hexdigest(), model_config, rollout_config, loss_config


def verify_data_authority(config):
    bundle_path, bundle, sequence_path, authority_path, gate = v3.verify_data_authority(config)
    audit_path = resolve(config["dataset"]["hybrid_sequence_audit"])
    audit = read_json(audit_path)
    errors = []
    if (
        sha256_file(audit_path) != config["dataset"]["hybrid_sequence_audit_sha256"]
        or audit.get("schema") != "DEPCarP3V6HybridSequenceAuditV1"
        or audit.get("status") != "READY_FOR_V4_WEAK_SEQUENCE_BOOTSTRAP"
        or audit.get("errors") != []
        or audit.get("formal_v4_training_allowed") is not True
        or audit.get("test_split_opened") is not False
        or audit.get("sequence_index_sha256") != sha256_file(sequence_path)
        or audit.get("sequence_authority_sha256") != sha256_file(authority_path)
    ):
        errors.append("hybrid_sequence_audit")
    if errors:
        raise RuntimeError("V4 data authority failed: " + ",".join(errors))
    gate = dict(gate)
    gate.update({
        "schema": "DEPCarV4DataGateV1",
        "hybrid_sequence_audit": str(audit_path),
        "hybrid_sequence_audit_sha256": sha256_file(audit_path),
        "continuous_sequence_authority": "FIRST_ACTION_ONLY",
        "later_action_supervision": "DIFFERENTIABLE_ROLLOUT_ROUTE_MAP",
    })
    return bundle_path, bundle, sequence_path, authority_path, gate


def _verify_contract(path, payload):
    contract_path = path.with_suffix(".contract.json")
    if not contract_path.is_file():
        raise RuntimeError("V4 source contract is missing: %s" % contract_path)
    contract = read_json(contract_path)
    if contract.get("checkpoint_sha256") != sha256_file(path):
        raise RuntimeError("V4 source checkpoint/contract differs")
    if contract.get("architecture_id") != payload.get("architecture_id"):
        raise RuntimeError("V4 source architecture/contract differs")
    return contract_path, contract


def verify_source(path, stage, config):
    path = resolve(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    contract_path, _contract = _verify_contract(path, payload)
    if stage == STAGES[0]:
        initialization = config["initialization"]
        acceptance_path = resolve(initialization["acceptance"])
        acceptance = read_json(acceptance_path)
        expected_errors = sorted(initialization["allowed_failed_checks"])
        valid = (
            path == resolve(initialization["checkpoint"])
            and sha256_file(path) == initialization["checkpoint_sha256"]
            and contract_path == resolve(initialization["checkpoint_contract"])
            and sha256_file(contract_path) == initialization["checkpoint_contract_sha256"]
            and sha256_file(acceptance_path) == initialization["acceptance_sha256"]
            and payload.get("architecture_id") == DEPCarNetV3.architecture_id
            and payload.get("run_completed") is True
            and payload.get("partial_epoch") is False
            and sorted(acceptance.get("errors", ())) == expected_errors
            and acceptance.get("checkpoint_sha256") == sha256_file(path)
            and acceptance.get("test_split_accessed") is False
        )
        if not valid:
            raise RuntimeError("V4 frozen V3 geometry initialization failed")
        return path, payload, {
            "schema": "DEPCarV4SourceGateV1", "passed": True,
            "checkpoint": str(path), "checkpoint_sha256": sha256_file(path),
            "checkpoint_contract": str(contract_path),
            "checkpoint_contract_sha256": sha256_file(contract_path),
            "acceptance": str(acceptance_path),
            "acceptance_sha256": sha256_file(acceptance_path),
            "source_role": "FROZEN_PERCEPTION_AND_LOCAL_GEOMETRY_ONLY",
            "ignored_v3_gear_authority": True,
            "allowed_failed_checks": expected_errors,
            "test_split_accessed": False,
        }
    previous = STAGES[STAGES.index(stage) - 1]
    previous_output = _best(resolve(config["artifacts"][STAGE_ARTIFACT[previous]]))
    acceptance_path = previous_output.with_suffix(".acceptance.json")
    acceptance = read_json(acceptance_path) if acceptance_path.is_file() else {}
    if (
        path != previous_output
        or payload.get("schema") != CHECKPOINT_SCHEMA
        or payload.get("architecture_id") != DEPCarNetV4.architecture_id
        or payload.get("training_stage") != previous
        or payload.get("run_completed") is not True
        or payload.get("partial_epoch") is not False
        or acceptance.get("schema") != "DEPCarV4AcceptanceV1"
        or acceptance.get("stage") != previous
        or acceptance.get("status") != "PASS"
        or acceptance.get("gate_passed") is not True
        or acceptance.get("checkpoint_sha256") != sha256_file(path)
        or acceptance.get("test_split_accessed") is not False
    ):
        raise RuntimeError("V4 previous stage is not independently accepted")
    return path, payload, {
        "schema": "DEPCarV4SourceGateV1", "passed": True,
        "checkpoint": str(path), "checkpoint_sha256": sha256_file(path),
        "checkpoint_contract": str(contract_path),
        "checkpoint_contract_sha256": sha256_file(contract_path),
        "acceptance": str(acceptance_path),
        "acceptance_sha256": sha256_file(acceptance_path),
        "source_stage": previous, "test_split_accessed": False,
    }


def build_model(model_config, rollout_config, source):
    base = DEPCarNetV3()
    model = DEPCarNetV4(base_model=base, sequence_config=model_config, rollout_config=rollout_config)
    if source.get("architecture_id") == DEPCarNetV3.architecture_id:
        model.initialize_base(source["model_state_dict"])
    else:
        model.load_state_dict(source["model_state_dict"], strict=True)
        model.freeze_base()
    return model


def configure_stage(model, stage):
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if stage == STAGES[0]:
        selected = tuple(model.candidate_parameters())
    elif stage == STAGES[1]:
        selected = tuple(model.score_parameters())
    else:
        selected = tuple(model.all_v4_parameters())
    for parameter in selected:
        parameter.requires_grad_(True)
    if any(parameter.requires_grad for parameter in model.base_model.parameters()):
        raise RuntimeError("V4 frozen base gained training authority")
    selected_ids = {id(parameter) for parameter in selected}
    if len(selected_ids) != len(selected):
        raise RuntimeError("V4 stage parameter ownership overlaps internally")
    return {
        "stage": stage,
        "trainable_parameters": sum(parameter.numel() for parameter in selected),
        "frozen_base_parameters": sum(parameter.numel() for parameter in model.base_model.parameters()),
        "high_level_gear_state_machine_trainable": False,
    }, selected


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_valid(host, device):
    valid = host["geometry_valid"].bool()
    if not bool(valid.any()):
        return None
    return {
        key: value[valid].to(device, non_blocking=True)
        for key, value in host.items() if key != "metadata"
    }


def forward_loss(model, objective, batch, stage, amp, dropout=0.0):
    modality = batch["modality_mask"]
    if dropout > 0.0:
        modality = apply_sensor_dropout(modality, dropout)
    with torch.autocast(
        device_type=batch["state"].device.type, dtype=torch.float16, enabled=amp
    ):
        model_kwargs = {}
        if getattr(model, "requires_local_distance_field", False):
            model_kwargs["local_distance_field"] = batch["bev_distance_field"].float()
        output = model(
            batch["depth"], batch["lidar_bev"], batch["state"],
            batch["current_gear"], batch["gear_history"],
            batch["route_pose"], batch["route_mask"], modality,
            **model_kwargs,
        )
    with torch.autocast(device_type=batch["state"].device.type, enabled=False):
        optional = {}
        if "target_action_plan_pose" in batch:
            optional.update({
                "target_action_plan_pose": batch["target_action_plan_pose"].float(),
                "target_action_plan_mask": batch["target_action_plan_mask"],
            })
        losses = objective(
            output,
            map_distance_field=batch["map_distance_field"].float(),
            map_resolution=batch["map_resolution"].float(),
            map_origin=batch["map_origin"].float(),
            chassis_to_map=batch["chassis_to_map"].float(),
            route=batch["route_pose"].float(), route_mask=batch["route_mask"],
            sequence_gears=batch["sequence_gears"], sequence_mask=batch["sequence_mask"],
            target_first_action_pose=batch["target_first_action_pose"].float(),
            target_first_action_valid=batch["target_first_action_valid"],
            guidance_cost=batch["target_guidance_cost"].float(),
            requested_gear=batch["requested_gear"], stage=stage,
            **optional,
        )
    return output, losses


_TOTALS = (
    "loss", "feasible", "zero_feasible", "any_viable", "selected_hard",
    "selected_viable", "oracle_regret", "oracle_count", "first_coverage",
    "multi_rows", "multi_prefix", "recovery_rows", "recovery_selected",
    "selected_token_errors", "selected_token_count", "best_token_errors",
    "best_token_count", "shift_count", "reverse_distance",
    "initial_safe", "zero_feasible_safe_initial",
    "hard_available", "selected_hard_available",
    "viable_available", "selected_viable_available",
    "initial_unsafe", "unsafe_egress_available", "selected_egress_available",
    "navigation_available", "selected_navigation_available",
    "exact_sequence_available", "hard_exact_available",
    "viable_exact_available", "navigation_exact_available",
    "eligible_best_token_errors", "eligible_best_token_count",
    "raw_selected_hard_available", "raw_selected_viable_available",
    "raw_selected_egress_available", "raw_selected_navigation_available",
    "mandatory_guard_no_candidate", "mandatory_guard_stop",
    "exact_execution_available", "selected_exact_execution_available",
)


def epoch_loop(
    model, objective, loader, stage, device, amp, *, optimizer=None,
    scaler=None, max_steps=None, dropout=0.0, progress_interval=50,
    gradient_clip=5.0,
):
    training = optimizer is not None
    model.train(training)
    totals = torch.zeros(len(_TOTALS), dtype=torch.float64, device=device)
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
                    [parameter for parameter in model.parameters() if parameter.requires_grad],
                    float(gradient_clip),
                )
                scaler.step(optimizer); scaler.update()
            count = len(batch["state"])
            feasible = losses["hard_feasible"].bool()
            viable = losses["viable"].bool()
            any_feasible = feasible.any(dim=1)
            any_viable = viable.any(dim=1)
            raw_selected = output.scores.detach().argmin(dim=1)
            execution_allowed = losses.get("execution_allowed")
            if execution_allowed is None:
                execution_allowed = torch.ones_like(feasible)
            else:
                execution_allowed = execution_allowed.bool()
            execution_available = execution_allowed.any(dim=1)
            guarded_scores = output.scores.detach().masked_fill(
                ~execution_allowed, torch.inf
            )
            selected = guarded_scores.argmin(dim=1)
            selected_hard = feasible.gather(1, selected[:, None]).squeeze(1)
            selected_viable = viable.gather(1, selected[:, None]).squeeze(1)
            selected_hard &= execution_available
            selected_viable &= execution_available
            raw_selected_hard = feasible.gather(
                1, raw_selected[:, None]
            ).squeeze(1)
            raw_selected_viable = viable.gather(
                1, raw_selected[:, None]
            ).squeeze(1)
            initial_safe = losses.get(
                "initial_pose_safe",
                torch.ones(len(feasible), dtype=torch.bool, device=device),
            ).bool()
            egress = losses.get("egress", torch.zeros_like(feasible)).bool()
            navigation = losses.get("navigation_eligible", viable).bool()
            initial_unsafe = ~initial_safe
            any_egress = egress.any(dim=1)
            selected_egress = egress.gather(1, selected[:, None]).squeeze(1)
            any_navigation = navigation.any(dim=1)
            selected_navigation = navigation.gather(1, selected[:, None]).squeeze(1)
            selected_egress &= execution_available
            selected_navigation &= execution_available
            raw_selected_egress = egress.gather(
                1, raw_selected[:, None]
            ).squeeze(1)
            raw_selected_navigation = navigation.gather(
                1, raw_selected[:, None]
            ).squeeze(1)
            cost = losses["candidate_cost"]
            oracle = cost.masked_fill(~feasible, torch.inf).amin(dim=1)
            chosen = cost.gather(1, selected[:, None]).squeeze(1)
            regret = torch.where(any_feasible, (chosen - oracle).clamp_min(0.0), torch.zeros_like(chosen))
            target = sequence_target_tokens(batch["sequence_gears"], batch["sequence_mask"])
            predicted = output.gear_tokens
            first_target = batch["sequence_gears"][:, 0]
            first_coverage = (
                (output.action_gears[..., 0] == first_target[:, None])
                & output.action_mask[..., 0]
            ).any(dim=1)
            action_count = batch["sequence_mask"].sum(dim=1)
            multi = action_count >= 2
            prefix_length = torch.minimum(action_count, action_count.new_full(action_count.shape, 3))
            positions = torch.arange(target.shape[1], device=device)[None]
            prefix_mask = positions < prefix_length[:, None]
            prefix_match = ((predicted == target[:, None]) | ~prefix_mask[:, None]).all(dim=-1)
            multi_prefix = prefix_match.any(dim=1) & multi
            target_recovery = (
                (batch["sequence_gears"][:, :-1] < 0)
                & (batch["sequence_gears"][:, 1:] > 0)
                & batch["sequence_mask"][:, 1:]
            ).any(dim=1) & execution_available
            selected_tokens = predicted.gather(
                1, selected[:, None, None].expand(-1, 1, predicted.shape[2])
            ).squeeze(1)
            selected_recovery = (
                (selected_tokens[:, :-1] == 2) & (selected_tokens[:, 1:] == 1)
            ).any(dim=1)
            token_weights = batch["sequence_mask"].bool()
            # Include the first STOP padding token so termination is audited.
            stop_index = action_count.clamp(max=target.shape[1] - 1)
            token_weights = token_weights.clone()
            token_weights[torch.arange(count, device=device), stop_index] = True
            selected_error_per = (
                (selected_tokens != target) & token_weights
            ).sum(dim=1)
            selected_error = (
                selected_error_per * execution_available.to(selected_error_per)
            ).sum()
            all_error = ((predicted != target[:, None]) & token_weights[:, None]).sum(dim=-1)
            best_error = all_error.amin(dim=1).sum()
            exact_sequence = all_error == 0
            exact_execution = exact_sequence & execution_allowed
            any_exact_execution = exact_execution.any(dim=1)
            selected_exact_execution = exact_execution.gather(
                1, selected[:, None]
            ).squeeze(1)
            effective_navigation = torch.where(
                any_navigation[:, None], navigation, torch.ones_like(navigation)
            )
            eligible_best_error = all_error.masked_fill(
                ~effective_navigation, target.shape[1] + 1
            ).amin(dim=1)
            selected_token_count = (
                token_weights.sum(dim=1)
                * execution_available.to(token_weights)
            ).sum()
            all_token_count = token_weights.sum()
            values = (
                losses["total"].detach().double() * count,
                feasible.sum().double(), (~any_feasible).sum().double(),
                any_viable.sum().double(), selected_hard.sum().double(),
                selected_viable.sum().double(), regret.sum().double(),
                any_feasible.sum().double(), first_coverage.sum().double(),
                multi.sum().double(), multi_prefix.sum().double(),
                target_recovery.sum().double(),
                (target_recovery & selected_recovery).sum().double(),
                selected_error.double(), selected_token_count.double(),
                best_error.double(), all_token_count.double(),
                losses["shift_count"].mean(dim=1).sum().double(),
                losses["reverse_distance_m"].mean(dim=1).sum().double(),
                initial_safe.sum().double(),
                (initial_safe & ~any_feasible).sum().double(),
                any_feasible.sum().double(),
                (any_feasible & selected_hard).sum().double(),
                any_viable.sum().double(),
                (any_viable & selected_viable).sum().double(),
                initial_unsafe.sum().double(),
                (initial_unsafe & any_egress).sum().double(),
                (initial_unsafe & any_egress & selected_egress).sum().double(),
                any_navigation.sum().double(),
                (any_navigation & selected_navigation).sum().double(),
                exact_sequence.any(dim=1).sum().double(),
                (any_feasible & (feasible & exact_sequence).any(dim=1)).sum().double(),
                (any_viable & (viable & exact_sequence).any(dim=1)).sum().double(),
                (any_navigation & (navigation & exact_sequence).any(dim=1)).sum().double(),
                eligible_best_error.sum().double(),
                all_token_count.double(),
                (any_feasible & raw_selected_hard).sum().double(),
                (any_viable & raw_selected_viable).sum().double(),
                (initial_unsafe & any_egress & raw_selected_egress).sum().double(),
                (any_navigation & raw_selected_navigation).sum().double(),
                (~execution_available).sum().double(),
                (~execution_available).sum().double(),
                any_exact_execution.sum().double(),
                (any_exact_execution & selected_exact_execution).sum().double(),
            )
            totals += torch.stack(values)
            samples += count; steps += 1
            if training and steps % int(progress_interval) == 0:
                elapsed = max(1e-6, time.monotonic() - started)
                print(
                    "step=%d samples=%d loss=%.6f samples_per_s=%.1f" % (
                        steps, samples, float(losses["total"].detach()), samples / elapsed
                    ), flush=True,
                )
    if samples == 0 or not bool(finite.cpu()):
        raise FloatingPointError("V4 loader is empty or objective became non-finite")
    value = dict(zip(_TOTALS, totals.cpu().tolist()))
    n = max(1, samples)
    return {
        "samples": samples, "steps": steps,
        "mean_loss": value["loss"] / n,
        "mean_feasible_candidates": value["feasible"] / n,
        "zero_hard_feasible_rate": value["zero_feasible"] / n,
        "any_viable_rate": value["any_viable"] / n,
        "selected_hard_feasible_rate": value["selected_hard"] / n,
        "selected_viable_rate": value["selected_viable"] / n,
        "mean_oracle_regret": value["oracle_regret"] / max(1.0, value["oracle_count"]),
        "teacher_first_gear_coverage": value["first_coverage"] / n,
        "multi_action_samples": int(value["multi_rows"]),
        "multiaction_prefix_coverage": value["multi_prefix"] / max(1.0, value["multi_rows"]),
        "reverse_then_forward_samples": int(value["recovery_rows"]),
        "reverse_then_forward_coverage": value["recovery_selected"] / max(1.0, value["recovery_rows"]),
        "plan_gear_prefix_error_rate": value["selected_token_errors"] / max(1.0, value["selected_token_count"]),
        "best_of_15_gear_error_rate": value["best_token_errors"] / max(1.0, value["best_token_count"]),
        "mean_candidate_shift_count": value["shift_count"] / n,
        "mean_candidate_reverse_distance_m": value["reverse_distance"] / n,
        "initial_pose_hard_unsafe_rate": value["initial_unsafe"] / n,
        "zero_hard_feasible_rate_given_safe_initial_pose": value[
            "zero_feasible_safe_initial"
        ] / max(1.0, value["initial_safe"]),
        "selected_hard_feasible_rate_when_available": value[
            "selected_hard_available"
        ] / max(1.0, value["hard_available"]),
        "selected_viable_rate_when_available": value[
            "selected_viable_available"
        ] / max(1.0, value["viable_available"]),
        "unsafe_initial_egress_candidate_rate": value[
            "unsafe_egress_available"
        ] / max(1.0, value["initial_unsafe"]),
        "selected_egress_rate_when_available": value[
            "selected_egress_available"
        ] / max(1.0, value["unsafe_egress_available"]),
        "selected_navigation_eligible_rate_when_available": value[
            "selected_navigation_available"
        ] / max(1.0, value["navigation_available"]),
        "exact_sequence_candidate_rate": value["exact_sequence_available"] / n,
        "hard_feasible_exact_sequence_rate_when_hard_available": value[
            "hard_exact_available"
        ] / max(1.0, value["hard_available"]),
        "viable_exact_sequence_rate_when_viable_available": value[
            "viable_exact_available"
        ] / max(1.0, value["viable_available"]),
        "navigation_exact_sequence_rate_when_navigation_available": value[
            "navigation_exact_available"
        ] / max(1.0, value["navigation_available"]),
        "best_navigation_eligible_gear_error_rate": value[
            "eligible_best_token_errors"
        ] / max(1.0, value["eligible_best_token_count"]),
        "raw_selected_hard_feasible_rate_when_available": value[
            "raw_selected_hard_available"
        ] / max(1.0, value["hard_available"]),
        "raw_selected_viable_rate_when_available": value[
            "raw_selected_viable_available"
        ] / max(1.0, value["viable_available"]),
        "raw_selected_egress_rate_when_available": value[
            "raw_selected_egress_available"
        ] / max(1.0, value["unsafe_egress_available"]),
        "raw_selected_navigation_eligible_rate_when_available": value[
            "raw_selected_navigation_available"
        ] / max(1.0, value["navigation_available"]),
        "mandatory_guard_no_candidate_rate": value[
            "mandatory_guard_no_candidate"
        ] / n,
        "mandatory_guard_stop_rate_when_no_candidate": value[
            "mandatory_guard_stop"
        ] / max(1.0, value["mandatory_guard_no_candidate"]),
        "selected_exact_sequence_rate_when_available": value[
            "selected_exact_execution_available"
        ] / max(1.0, value["exact_execution_available"]),
        "exact_sequence_execution_available_rate": value[
            "exact_execution_available"
        ] / n,
        "elapsed_s": time.monotonic() - started,
    }


def acceptance_checks(stage, metrics, qualification):
    gate = qualification[STAGE_ARTIFACT[stage] if stage != STAGES[2] else "closed_loop"]
    common = {
        "zero_hard_feasible_rate": metrics["zero_hard_feasible_rate"] <= float(gate["maximum_zero_hard_feasible_rate"]),
    }
    if stage == STAGES[0]:
        common.update({
            "any_viable_rate": metrics["any_viable_rate"] >= float(gate["minimum_any_viable_rate"]),
            "teacher_first_gear_coverage": metrics["teacher_first_gear_coverage"] >= float(gate["minimum_teacher_first_gear_coverage"]),
            "multiaction_prefix_coverage": metrics["multiaction_prefix_coverage"] >= float(gate["minimum_multiaction_prefix_coverage"]),
        })
    elif stage == STAGES[1]:
        common.update({
            "selected_hard_feasible_rate": metrics["selected_hard_feasible_rate"] >= float(gate["minimum_selected_hard_feasible_rate"]),
            "selected_viable_rate": metrics["selected_viable_rate"] >= float(gate["minimum_selected_viable_rate"]),
            "mean_oracle_regret": metrics["mean_oracle_regret"] <= float(gate["maximum_mean_oracle_regret"]),
        })
    else:
        common.update({
            "selected_hard_feasible_rate": metrics["selected_hard_feasible_rate"] >= float(gate["minimum_selected_hard_feasible_rate"]),
            "selected_viable_rate": metrics["selected_viable_rate"] >= float(gate["minimum_selected_viable_rate"]),
            "reverse_then_forward_coverage": metrics["reverse_then_forward_coverage"] >= float(gate["minimum_reverse_then_forward_coverage"]),
            "plan_gear_prefix_error_rate": metrics["plan_gear_prefix_error_rate"] <= float(gate["maximum_plan_gear_prefix_error_rate"]),
        })
    return common


def selection_key(stage, metrics, qualification):
    checks = acceptance_checks(stage, metrics, qualification)
    return (
        sum(not value for value in checks.values()),
        metrics["zero_hard_feasible_rate"],
        -metrics["any_viable_rate"],
        -metrics["selected_hard_feasible_rate"],
        -metrics["selected_viable_rate"],
        metrics["plan_gear_prefix_error_rate"],
        metrics["mean_oracle_regret"],
        metrics["mean_loss"],
    )


def atomic_torch(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary); os.replace(temporary, path)


def write_artifact(path, model, optimizer, scaler, metadata, metrics, history):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": CHECKPOINT_SCHEMA, "architecture_id": DEPCarNetV4.architecture_id,
        "objective_id": DEPCarObjectiveV4.objective_id,
        "training_stage": metadata["stage"], "modality": "fusion",
        "artifact_role": metadata["artifact_role"], "status": "TRAINED_UNQUALIFIED",
        "qualification_status": "UNQUALIFIED", "active_control_authorized": False,
        "production_qualified": False, "completed_epochs": metadata["completed_epochs"],
        "selected_epoch": metadata["selected_epoch"], "partial_epoch": False,
        "global_step": metadata["global_step"], "source_checkpoint": metadata["source_checkpoint"],
        "source_checkpoint_sha256": metadata["source_checkpoint_sha256"],
        "data_authority_gate": metadata["data_authority_gate"],
        "source_gate": metadata["source_gate"],
        "training_config_sha256": metadata["training_config_sha256"],
        "trainer_sha256": metadata["trainer_sha256"],
        "model_implementation_sha256": metadata["model_implementation_sha256"],
        "rollout_implementation_sha256": metadata["rollout_implementation_sha256"],
        "loss_implementation_sha256": metadata["loss_implementation_sha256"],
        "unified_hybrid_sequence": True, "high_level_gear_state_machine": False,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "grad_scaler_state_dict": scaler.state_dict(),
        "metrics": metrics, "selection_gate": metadata["selection_gate"],
        "history": history,
    }
    atomic_torch(path, payload)
    keys = (
        "architecture_id", "objective_id", "training_stage", "modality",
        "artifact_role", "status", "qualification_status", "active_control_authorized",
        "production_qualified", "completed_epochs", "selected_epoch", "partial_epoch",
        "global_step", "source_checkpoint", "source_checkpoint_sha256",
        "data_authority_gate", "source_gate", "training_config_sha256", "trainer_sha256",
        "model_implementation_sha256", "rollout_implementation_sha256",
        "loss_implementation_sha256", "unified_hybrid_sequence",
        "high_level_gear_state_machine", "metrics", "selection_gate",
    )
    contract = {key: payload[key] for key in keys}
    contract.update({"schema": CONTRACT_SCHEMA, "checkpoint": str(path), "checkpoint_sha256": sha256_file(path)})
    path.with_suffix(".contract.json").write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def finalize_best(path, completed_epochs, global_step, history):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload.update({
        "completed_epochs": completed_epochs, "global_step": global_step,
        "run_completed": True, "partial_epoch": False, "history": history,
    })
    atomic_torch(path, payload)
    contract_path = path.with_suffix(".contract.json")
    contract = read_json(contract_path)
    contract.update({
        "checkpoint_sha256": sha256_file(path), "completed_epochs": completed_epochs,
        "global_step": global_step, "run_completed": True, "partial_epoch": False,
    })
    contract_path.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv=None):
    args = parser().parse_args(argv)
    config, config_sha, model_config, rollout_config, loss_config = load_config()
    training = config["training"]
    epoch_key = {STAGES[0]: "capacity_epochs", STAGES[1]: "score_epochs", STAGES[2]: "closed_loop_epochs"}[args.stage]
    epochs = int(training[epoch_key]) if args.epochs is None else args.epochs
    batch_size = int(training["batch_size"]) if args.batch_size is None else args.batch_size
    workers = int(training["workers"]) if args.workers is None else args.workers
    if min(epochs, batch_size, workers) < 1:
        raise SystemExit("V4 epochs, batch size and workers must be positive")
    if args.max_samples is not None and not 1 <= args.max_samples <= 2048:
        raise SystemExit("V4 max-samples must be in [1,2048]")
    if args.max_steps is not None and not 1 <= args.max_steps <= 64:
        raise SystemExit("V4 max-steps must be in [1,64]")
    bundle_path, bundle, sequence_path, _authority, data_gate = verify_data_authority(config)
    source_path, source, source_gate = verify_source(args.source, args.stage, config)
    output = resolve(args.output)
    expected = resolve(config["artifacts"][STAGE_ARTIFACT[args.stage]])
    bounded = args.max_samples is not None or args.max_steps is not None
    formal = bool(
        not bounded and epochs == int(training[epoch_key])
        and batch_size == int(training["batch_size"]) and workers == int(training["workers"])
        and args.device == "cuda" and output == expected
        and data_gate["passed"] and source_gate["passed"]
    )
    if not args.dry_run and not formal and not bounded:
        raise RuntimeError("V4 run is neither formal nor bounded diagnostic")
    if output == source_path:
        raise RuntimeError("V4 source and output must differ")
    if not args.dry_run and output.exists() and args.resume is None:
        raise RuntimeError("V4 output exists; use canonical --resume")
    if args.resume is not None and resolve(args.resume) != output:
        raise RuntimeError("V4 resume must name the canonical last artifact")
    plan = {
        "schema": "DEPCarV4TrainingPlanV1",
        "status": "DRY_RUN_READY" if args.dry_run and formal else "BOUNDED_DIAGNOSTIC_READY" if bounded else "READY",
        "stage": args.stage, "architecture_id": DEPCarNetV4.architecture_id,
        "objective_id": DEPCarObjectiveV4.objective_id,
        "source": str(source_path), "source_sha256": sha256_file(source_path),
        "output": str(output), "epochs": epochs, "batch_size": batch_size,
        "workers": workers, "device": args.device, "bounded_smoke": bounded,
        "maximum_samples": args.max_samples, "maximum_steps": args.max_steps,
        "formal_training_authorized": formal, "data_authority_gate": data_gate,
        "source_gate": source_gate, "training_config": str(CONFIG),
        "training_config_sha256": config_sha, "trainer_sha256": sha256_file(TRAINER),
        "unified_hybrid_sequence": True, "high_level_gear_state_machine": False,
        "test_split_sealed": True, "active_control_authorized": False,
        "production_qualified": False,
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True)); return 0
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    seed_all(int(training["seed"])); torch.set_num_threads(int(training["torch_threads"]))
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True; torch.set_float32_matmul_precision("high")
    model = build_model(model_config, rollout_config, source).to(device)
    ownership, selected = configure_stage(model, args.stage)
    learning_rate = float(training["closed_loop_learning_rate"] if args.stage == STAGES[2] else training["learning_rate"])
    optimizer = torch.optim.AdamW(selected, lr=learning_rate, weight_decay=float(training["weight_decay"]))
    amp = bool(training["mixed_precision"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    common = dict(
        sample_root=bundle["sample_root"], maps_root=bundle["maps_root"],
        index_path=bundle["index"], index_splits=("train", "validation"), workers=workers,
        expected_map_contract_aggregate_sha256=bundle["map_contract_aggregate_sha256"],
        expected_index_sha256=bundle["index_sha256"], modality="fusion",
        sequence_index_path=sequence_path,
    )
    train_data = P3HybridSequenceDatasetV4(split="train", **common)
    validation_data = P3HybridSequenceDatasetV4(split="validation", **common)
    if args.max_samples:
        train_data = Subset(train_data, np.linspace(0, len(train_data) - 1, min(len(train_data), args.max_samples), dtype=np.int64).tolist())
        validation_data = Subset(validation_data, np.linspace(0, len(validation_data) - 1, min(len(validation_data), args.max_samples), dtype=np.int64).tolist())
    loader_common = dict(
        batch_size=batch_size, num_workers=workers, pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        prefetch_factor=int(training["prefetch_factor"]) if workers > 0 else None,
        collate_fn=p3_training_collate, worker_init_fn=p3_training_worker_init,
    )
    train_loader = DataLoader(train_data, shuffle=True, **loader_common)
    validation_loader = DataLoader(validation_data, shuffle=False, **loader_common)
    objective = DEPCarObjectiveV4(loss_config)
    history, global_step, best_key, start_epoch = [], 0, None, 1
    if args.resume:
        resumed = torch.load(output, map_location="cpu", weights_only=True)
        required = {
            "schema": CHECKPOINT_SCHEMA, "architecture_id": DEPCarNetV4.architecture_id,
            "training_stage": args.stage, "artifact_role": "last",
            "source_checkpoint_sha256": plan["source_sha256"],
            "training_config_sha256": config_sha, "trainer_sha256": plan["trainer_sha256"],
        }
        mismatch = [key for key, value in required.items() if resumed.get(key) != value]
        if mismatch:
            raise RuntimeError("V4 resume mismatch: " + ",".join(mismatch))
        completed = int(resumed.get("completed_epochs", 0))
        if not 1 <= completed < epochs:
            raise RuntimeError("V4 resume epoch is outside remaining run")
        model.load_state_dict(resumed["model_state_dict"], strict=True)
        optimizer.load_state_dict(resumed["optimizer_state_dict"])
        scaler.load_state_dict(resumed["grad_scaler_state_dict"])
        history = list(resumed.get("history", ())); global_step = int(resumed.get("global_step", 0))
        start_epoch = completed + 1
        keys = [selection_key(args.stage, row["validation"], config["qualification"]) for row in history]
        best_key = min(keys) if keys else None; model.to(device)
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
            max_steps=args.max_steps, progress_interval=int(training["progress_interval_steps"]),
        )
        global_step += train_metrics["steps"]
        checks = acceptance_checks(args.stage, validation_metrics, config["qualification"])
        gate = {key: "PASS" if value else "FAIL" for key, value in checks.items()}
        row = {"epoch": epoch, "train": train_metrics, "validation": validation_metrics, "selection_gate": gate, "elapsed_s": time.time() - started}
        history.append(row); print(json.dumps(row, sort_keys=True), flush=True)
        key = selection_key(args.stage, validation_metrics, config["qualification"])
        metadata = {
            "stage": args.stage, "artifact_role": "last", "completed_epochs": epoch,
            "selected_epoch": epoch, "global_step": global_step,
            "source_checkpoint": str(source_path), "source_checkpoint_sha256": plan["source_sha256"],
            "data_authority_gate": data_gate, "source_gate": source_gate,
            "training_config_sha256": config_sha, "trainer_sha256": plan["trainer_sha256"],
            "model_implementation_sha256": config["implementation"]["model_sha256"],
            "rollout_implementation_sha256": config["implementation"]["rollout_sha256"],
            "loss_implementation_sha256": config["implementation"]["loss_sha256"],
            "selection_gate": gate,
        }
        write_artifact(output, model, optimizer, scaler, metadata, validation_metrics, history)
        if best_key is None or key < best_key:
            best_key = key; metadata["artifact_role"] = "best"
            write_artifact(_best(output), model, optimizer, scaler, metadata, validation_metrics, history)
    best_path = _best(output)
    if not best_path.is_file():
        raise RuntimeError("V4 training ended without best checkpoint")
    finalize_best(best_path, epochs, global_step, history)
    print(json.dumps({**plan, "status": "COMPLETE", "global_step": global_step, "stage_ownership": ownership, "best_checkpoint": str(best_path)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
