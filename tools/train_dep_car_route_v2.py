#!/usr/bin/env python3
"""Train route-conditioned Candidate Capacity and Score stages for V2."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dep_car/src"))

from dep_car.model.dep_car_net import DEPCarNetV1
from dep_car.model.dep_car_net_v2 import DEPCarNetV2
from dep_car.training.losses_v2 import DEPCarObjectiveV2, DEPCarRouteLossConfigV2
from dep_car.training.pilot import canonical_sha256
from dep_car.training.p4_dataset import p3_training_collate, p3_training_worker_init
from dep_car.training.score_dataset import P3ScoreTrainingDatasetV1
from dep_car.training.stages import (
    apply_sensor_dropout,
    build_optimizer,
    configure_training_stage,
)


CONFIG = ROOT / "dep_car/config/p5_route_v2_training.yaml"
TRAINER = Path(__file__).resolve()
V2_CANDIDATE_ACCEPTANCE_REPORT = (
    ROOT / "reports/p5_route_v2_fusion_candidate_acceptance.json"
)
V1_ARCHITECTURE = DEPCarNetV1.architecture_id
V2_ARCHITECTURE = DEPCarNetV2.architecture_id


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve(path):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("candidate_capacity", "score_calibration"), required=True)
    parser.add_argument("--modality", choices=("depth_only", "lidar_only", "fusion"), required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--authority", default="data/p3_v4/bundle_v1/bundle_authority.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def load_config():
    raw = CONFIG.read_bytes()
    config = yaml.safe_load(raw)
    if (
        config.get("schema") != "DEPCarRouteV2TrainingContractV1"
        or config.get("architecture_id") != V2_ARCHITECTURE
        or config.get("objective_id") != DEPCarObjectiveV2.objective_id
    ):
        raise RuntimeError("V2 training contract is invalid")
    authorization = config.get("authorization", {})
    if (
        authorization.get("status") != "APPLIED_EXPLICIT_FUSION_ONLY_APPROVAL"
        or authorization.get("formal_modalities") != ["fusion"]
        or set(authorization.get("diagnostic_only_modalities", ()))
        != {"depth_only", "lidar_only"}
        or authorization.get("test_split_sealed") is not True
    ):
        raise RuntimeError("V2 fusion-only authorization contract is invalid")
    training = config.get("training", {})
    dropout = training.get("sensor_dropout_probability")
    if (
        isinstance(dropout, bool)
        or not isinstance(dropout, (int, float))
        or not 0.0 <= float(dropout) < 1.0
    ):
        raise RuntimeError("V2 fusion sensor dropout probability is invalid")
    loss = config.get("loss", {})
    try:
        DEPCarRouteLossConfigV2(**loss).validate()
    except (TypeError, ValueError) as exc:
        raise RuntimeError("V2 route loss contract is invalid") from exc
    return config, hashlib.sha256(raw).hexdigest()


def read_json(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("expected a JSON object: %s" % path)
    return payload


def verify_authority(path, config):
    path = resolve(path)
    authority = read_json(path)
    authorization = config["authorization"]
    errors = []
    if (
        authority.get("schema") != "DEPCarP3V3BundleAuthorityV1"
        or authority.get("status") != "SEALED"
        or authority.get("test_npz_opened") is not False
        or authority.get("test_map_yaml_or_png_opened") is not False
    ):
        errors.append("bundle_not_sealed_development_authority")
    for key in (
        "sample_root", "maps_root", "index", "index_sha256",
        "map_contract_aggregate_sha256", "content_aggregate_sha256",
    ):
        if not authority.get(key):
            errors.append("bundle_field_missing_" + key)
    if path != resolve(config["dataset_authority"]):
        errors.append("bundle_path_not_training_contract")
    if path != resolve(authorization["bundle_authority"]):
        errors.append("bundle_path_not_authorization_contract")
    if sha256_file(path) != authorization.get("bundle_authority_file_sha256"):
        errors.append("bundle_file_sha256")
    claimed = authority.get("bundle_authority_sha256", "")
    content = dict(authority)
    content.pop("bundle_authority_sha256", None)
    if claimed != canonical_sha256(content) or claimed != authorization.get(
        "bundle_authority_sha256"
    ):
        errors.append("bundle_internal_sha256")

    if not errors:
        index_path = resolve(authority["index"])
        if sha256_file(index_path) != authority["index_sha256"]:
            errors.append("dataset_index_file_sha256")
        else:
            index = read_json(index_path)
            if (
                index.get("schema") != "P3TrainingIndexV2"
                or index.get("content_aggregate_sha256")
                != authority["content_aggregate_sha256"]
                or index.get("samples") != authority.get("samples")
                or index.get("counts_by_split") != authority.get("counts_by_split")
            ):
                errors.append("dataset_index_authority")

    reaudit_path = resolve(authorization["development_reaudit"])
    proposal_path = resolve(authorization["proposal"])
    if sha256_file(reaudit_path) != authorization.get("development_reaudit_sha256"):
        errors.append("development_reaudit_file_sha256")
    if sha256_file(proposal_path) != authorization.get("proposal_sha256"):
        errors.append("proposal_file_sha256")
    reaudit = read_json(reaudit_path)
    proposal = read_json(proposal_path)
    if (
        reaudit.get("schema") != "DEPCarP3DevelopmentReauditV3"
        or reaudit.get("status") != "PASS"
        or reaudit.get("errors") != []
        or reaudit.get("qualification_eligible") is not True
        or reaudit.get("validation_coverage_gate", {}).get("status") != "PASS"
        or reaudit.get("scope", {}).get("test_npz_opened") is not False
        or reaudit.get("scope", {}).get("test_map_yaml_or_png_opened") is not False
    ):
        errors.append("development_reaudit_not_qualifying")
    report_bundle = reaudit.get("bundle_authority", {})
    if (
        report_bundle.get("file_sha256") != sha256_file(path)
        or report_bundle.get("bundle_authority_sha256") != claimed
        or report_bundle.get("verified") is not True
    ):
        errors.append("development_reaudit_bundle_identity")
    gate_rows = reaudit.get("gates", {})
    mode_rows = gate_rows.get("configured_per_mode_zero_feasible_rate", {})
    if (
        gate_rows.get("configured_overall_zero_feasible_rate", {}).get("status")
        != "PASS"
        or gate_rows.get("configured_overall_median_feasible_candidates", {}).get(
            "status"
        ) != "PASS"
        or set(mode_rows) != {
            "NORMAL", "SHARP_TURN", "NARROW_CORRIDOR", "U_TURN",
            "DEAD_END_ESCAPE", "REVERSE_EXIT", "THREE_POINT_TURN",
        }
        or any(row.get("status") != "PASS" for row in mode_rows.values())
    ):
        errors.append("development_reaudit_geometry_gates")
    training_authority = reaudit.get("training_authority", {})
    for key in (
        "index_sha256", "content_aggregate_sha256",
        "map_contract_aggregate_sha256",
    ):
        if training_authority.get(key) != authority.get(key):
            errors.append("development_training_authority_" + key)
    if (
        training_authority.get("splits") != ["train", "validation"]
        or training_authority.get("test_split_used") is not False
    ):
        errors.append("development_training_authority_splits")

    if (
        proposal.get("schema") != "DEPCarP3V3TrainingAuthorityProposalV1"
        or proposal.get("status") != "READY_FOR_EXPLICIT_P5_CONFIG_APPROVAL"
        or proposal.get("formal_training_started") is not False
        or resolve(proposal.get("bundle_authority", "")) != path
        or proposal.get("bundle_authority_sha256") != sha256_file(path)
        or resolve(proposal.get("reaudit", "")) != reaudit_path
        or proposal.get("reaudit_sha256") != sha256_file(reaudit_path)
    ):
        errors.append("training_authority_proposal_identity")
    proposal_changes = proposal.get("training_yaml_changes", {})
    expected_changes = {
        "dataset.root": resolve(authority.get("sample_root", "")),
        "dataset.maps": resolve(authority.get("maps_root", "")),
        "dataset.index": resolve(authority.get("index", "")),
    }
    for key, expected in expected_changes.items():
        if resolve(proposal_changes.get(key, "")) != expected:
            errors.append("training_authority_proposal_" + key.replace(".", "_"))
    for key in ("content_aggregate_sha256", "map_contract_aggregate_sha256"):
        if proposal_changes.get("dataset." + key) != authority.get(key):
            errors.append("training_authority_proposal_" + key)
    if (
        proposal_changes.get("qualification.corrected_footprint_p3_status")
        != "PASS"
        or proposal_changes.get("qualification.p5_formal_training_allowed") is not True
        or proposal_changes.get("qualification.blocked_gates") != []
    ):
        errors.append("training_authority_proposal_qualification")

    gate = {
        "schema": "DEPCarRouteV2FormalTrainingAuthorityGateV1",
        "passed": not errors,
        "errors": sorted(set(errors)),
        "scope": "fusion_only_candidate_then_score",
        "bundle_authority": str(path),
        "bundle_authority_file_sha256": sha256_file(path),
        "bundle_authority_sha256": claimed,
        "development_reaudit": str(reaudit_path),
        "development_reaudit_sha256": sha256_file(reaudit_path),
        "proposal": str(proposal_path),
        "proposal_sha256": sha256_file(proposal_path),
        "samples": authority.get("counts_by_split"),
        "test_split_sealed": True,
    }
    if errors:
        raise RuntimeError(
            "V2 formal training authority failed: " + ",".join(gate["errors"])
        )
    return path, authority, gate


def _verify_v2_candidate_acceptance(
    path, payload, contract_path, acceptance_path, config
):
    """Fail closed unless Score consumes the fully audited canonical Candidate."""

    acceptance = read_json(acceptance_path)
    report_path = V2_CANDIDATE_ACCEPTANCE_REPORT.resolve()
    errors = []
    checkpoint_sha = sha256_file(path)
    contract_sha = sha256_file(contract_path) if contract_path.is_file() else None
    authority_path = resolve(config["dataset_authority"])
    expected_candidate = _expected_best(resolve(config["artifacts"]["candidate"]))
    if path != expected_candidate:
        errors.append("not_canonical_candidate_best")
    if not report_path.is_file():
        errors.append("formal_acceptance_report_missing")
    elif sha256_file(report_path) != sha256_file(acceptance_path):
        errors.append("formal_report_sidecar_identity")
    if (
        acceptance.get("schema") != "DEPCarRouteV2CandidateAcceptanceV1"
        or acceptance.get("status") != "PASS"
        or acceptance.get("gate_passed") is not True
        or acceptance.get("errors") != []
        or acceptance.get("smoke_limited") is not False
        or acceptance.get("test_split_accessed") is not False
    ):
        errors.append("formal_acceptance_status")
    if (
        acceptance.get("checkpoint_sha256") != checkpoint_sha
        or resolve(acceptance.get("checkpoint", "")) != path
        or acceptance.get("checkpoint_contract_sha256") != contract_sha
        or resolve(acceptance.get("checkpoint_contract", "")) != contract_path
    ):
        errors.append("formal_acceptance_checkpoint_identity")
    if (
        acceptance.get("dataset_authority_sha256") != sha256_file(authority_path)
        or resolve(acceptance.get("dataset_authority", "")) != authority_path
    ):
        errors.append("formal_acceptance_dataset_identity")
    if (
        acceptance.get("candidate_training_config_sha256")
        != payload.get("training_config_sha256")
        or acceptance.get("candidate_trainer_sha256")
        != payload.get("trainer_sha256")
    ):
        errors.append("formal_acceptance_training_lineage")
    if (
        acceptance.get("population")
        != "validation/SHARP_TURN/geometry_valid"
        or not isinstance(acceptance.get("population_frames_available"), int)
        or acceptance.get("population_frames_available") <= 0
        or acceptance.get("population_frames_selected")
        != acceptance.get("population_frames_available")
    ):
        errors.append("formal_acceptance_population")
    gates = acceptance.get("gates", {})
    if (
        set(gates) != {"capacity_capable_rate", "zero_hard_feasible_rate"}
        or any(row.get("status") != "PASS" for row in gates.values())
    ):
        errors.append("formal_acceptance_capacity_gates")
    if (
        payload.get("run_completed") is not True
        or payload.get("partial_epoch") is not False
        or payload.get("completed_epochs") != int(config["training"]["epochs"])
    ):
        errors.append("candidate_training_not_complete")
    if errors:
        raise RuntimeError(
            "V2 Candidate source acceptance failed: " + ",".join(sorted(set(errors)))
        )
    return {
        "schema": "DEPCarRouteV2ScoreSourceGateV1",
        "passed": True,
        "errors": [],
        "checkpoint": str(path),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_contract": str(contract_path),
        "checkpoint_contract_sha256": contract_sha,
        "acceptance_sidecar": str(acceptance_path),
        "acceptance_sidecar_sha256": sha256_file(acceptance_path),
        "formal_acceptance_report": str(report_path),
        "formal_acceptance_report_sha256": sha256_file(report_path),
        "population_frames": acceptance["population_frames_selected"],
        "capacity_capable_rate": acceptance["metrics"]["capacity_capable_rate"],
        "zero_hard_feasible_rate": acceptance["metrics"]["zero_hard_feasible_rate"],
        "test_split_accessed": False,
    }


def verify_source(path, stage, modality, config):
    path = resolve(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("modality") != modality:
        raise RuntimeError("source checkpoint modality mismatch")
    architecture = payload.get("architecture_id")
    source_gate = None
    if stage == "candidate_capacity":
        if architecture != V1_ARCHITECTURE or payload.get("training_stage") != "candidate_capacity":
            raise RuntimeError("V2 Candidate must initialize from an accepted V1 Candidate")
        acceptance_path = path.with_suffix(".candidate_acceptance.json")
        if not acceptance_path.is_file():
            raise RuntimeError("V1 Candidate acceptance sidecar is missing")
        acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
        initialization = config["initialization"]
        prefix = modality + "_candidate_v1"
        if (
            path != resolve(initialization[prefix])
            or sha256_file(path) != initialization[prefix + "_sha256"]
            or acceptance_path
            != resolve(initialization[prefix + "_acceptance"])
            or sha256_file(acceptance_path)
            != initialization[prefix + "_acceptance_sha256"]
        ):
            raise RuntimeError("V1 Candidate source differs from the fusion authorization")
        if (
            acceptance.get("status") != "PASS"
            or acceptance.get("gate_passed") is not True
            or acceptance.get("checkpoint_sha256") != sha256_file(path)
            or acceptance.get("training_stage") != "candidate_capacity"
            or acceptance.get("smoke_limited") is not False
        ):
            raise RuntimeError("V1 Candidate source was not accepted")
        source_gate = {
            "schema": "DEPCarRouteV2CandidateInitializationGateV1",
            "passed": True,
            "checkpoint": str(path),
            "checkpoint_sha256": sha256_file(path),
            "acceptance_sidecar": str(acceptance_path),
            "acceptance_sidecar_sha256": sha256_file(acceptance_path),
        }
    else:
        if architecture != V2_ARCHITECTURE or payload.get("training_stage") != "candidate_capacity":
            raise RuntimeError("V2 Score must initialize from a V2 Candidate checkpoint")
        if payload.get("completed_epochs", 0) < 1 or payload.get("partial_epoch") is not False:
            raise RuntimeError("V2 Candidate source is incomplete")
        acceptance_path = path.with_suffix(".candidate_acceptance.json")
        if not acceptance_path.is_file():
            raise RuntimeError("V2 Candidate acceptance sidecar is missing")
    contract_path = path.with_suffix(".contract.json")
    if not contract_path.is_file():
        raise RuntimeError("source checkpoint contract is missing")
    contract = read_json(contract_path)
    if contract.get("checkpoint_sha256") != sha256_file(path):
        raise RuntimeError("source checkpoint/contract hash mismatch")
    if stage == "score_calibration":
        source_gate = _verify_v2_candidate_acceptance(
            path, payload, contract_path, acceptance_path, config
        )
    return path, payload, source_gate


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@contextmanager
def amp_encoder_outputs_fp32(model, enabled):
    handles = []
    if enabled:
        def cast(_module, _arguments, output):
            return output.float()
        handles = [
            model.depth_encoder.register_forward_hook(cast),
            model.lidar_encoder.register_forward_hook(cast),
        ]
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def select_valid(host, device):
    valid = host["geometry_valid"].bool()
    if not bool(valid.any()):
        return None
    return {
        key: value[valid].to(device, non_blocking=True)
        for key, value in host.items()
        if key != "metadata"
    }


def set_stage_mode(model, stage, training):
    """Keep frozen Candidate BatchNorm buffers immutable during Score training."""

    if training and stage == "candidate_capacity":
        model.train()
    else:
        # eval() does not disable autograd.  Score parameters remain trainable,
        # while all Candidate BatchNorm running statistics stay frozen.
        model.eval()


def forward_loss(
    model, objective, batch, stage, amp, sensor_dropout_probability=0.0,
    apply_training_dropout=False,
):
    selected_mask = batch["modality_mask"]
    if apply_training_dropout and sensor_dropout_probability > 0.0:
        selected_mask = apply_sensor_dropout(
            selected_mask, float(sensor_dropout_probability)
        )
    with torch.autocast(
        device_type=batch["state"].device.type,
        dtype=torch.float16,
        enabled=amp,
    ):
        output = model(
            batch["depth"], batch["lidar_bev"], batch["state"],
            batch["requested_gear"], batch["route_pose"], batch["route_mask"],
            selected_mask,
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
            requested_gear=batch["requested_gear"],
            stage=stage,
        )
    return output, losses


def epoch_loop(
    model, objective, loader, stage, device, amp, optimizer=None, scaler=None,
    max_steps=None, sensor_dropout_probability=0.0, progress_interval=50,
):
    training = optimizer is not None
    set_stage_mode(model, stage, training)
    aggregate = {"samples": 0, "steps": 0}
    device_totals = torch.zeros(10, dtype=torch.float64, device=device)
    finite = torch.ones((), dtype=torch.bool, device=device)
    started = time.monotonic()
    context = torch.enable_grad if training else torch.inference_mode
    with context():
        for host in loader:
            if max_steps is not None and aggregate["steps"] >= max_steps:
                break
            batch = select_valid(host, device)
            if batch is None:
                continue
            if training:
                optimizer.zero_grad(set_to_none=True)
            output, losses = forward_loss(
                model, objective, batch, stage, amp,
                sensor_dropout_probability if training else 0.0,
                apply_training_dropout=training,
            )
            finite.logical_and_(
                torch.isfinite(losses["total"].detach())
                & torch.isfinite(output.scores.detach()).all()
                & torch.isfinite(losses["candidate_cost"]).all()
            )
            if training:
                scaler.scale(losses["total"]).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in model.parameters() if parameter.requires_grad],
                    5.0,
                )
                scaler.step(optimizer)
                scaler.update()
            count = len(batch["state"])
            feasible = losses["hard_feasible"]
            margin = losses["minimum_clearance"] >= objective.config.clearance_margin_m
            future = losses["future_minimum_clearance"] >= objective.config.clearance_margin_m
            selected = output.scores.detach().argmin(dim=1)
            selected_column = selected[:, None]
            selected_hard = feasible.gather(1, selected_column).squeeze(1)
            selected_margin = (
                margin.gather(1, selected_column).squeeze(1) & selected_hard
            )
            selected_future = (
                future.gather(1, selected_column).squeeze(1) & selected_margin
            )
            any_feasible = feasible.any(dim=1)
            candidate_cost = losses["candidate_cost"]
            oracle_cost = candidate_cost.masked_fill(~feasible, torch.inf).amin(dim=1)
            selected_cost = candidate_cost.gather(1, selected_column).squeeze(1)
            oracle_regret = torch.where(
                any_feasible,
                (selected_cost - oracle_cost).clamp_min(0.0),
                torch.zeros_like(selected_cost),
            )
            aggregate["samples"] += count
            aggregate["steps"] += 1
            device_totals += torch.stack((
                losses["total"].detach().to(torch.float64) * count,
                feasible.sum().to(torch.float64),
                (~feasible.any(dim=1)).sum().to(torch.float64),
                margin.any(dim=1).sum().to(torch.float64),
                (margin & future).any(dim=1).sum().to(torch.float64),
                selected_hard.sum().to(torch.float64),
                selected_margin.sum().to(torch.float64),
                selected_future.sum().to(torch.float64),
                oracle_regret.sum().to(torch.float64),
                any_feasible.sum().to(torch.float64),
            ))
            if training and aggregate["steps"] % int(progress_interval) == 0:
                elapsed = max(time.monotonic() - started, 1.0e-6)
                print(
                    "step=%d samples=%d loss=%.6f samples_per_s=%.1f" % (
                        aggregate["steps"], aggregate["samples"],
                        float(losses["total"].detach()),
                        aggregate["samples"] / elapsed,
                    ),
                    flush=True,
                )
    values = device_totals.cpu().tolist()
    if not bool(finite.cpu()):
        raise FloatingPointError("V2 objective became non-finite")
    samples = max(1, aggregate["samples"])
    aggregate.update({
        "mean_loss": values[0] / samples,
        "mean_feasible_candidates": values[1] / samples,
        "zero_feasible": int(values[2]),
        "zero_feasible_rate": values[2] / samples,
        "margin_capable": int(values[3]),
        "margin_capable_rate": values[3] / samples,
        "future_capable": int(values[4]),
        "future_capable_rate": values[4] / samples,
        "selected_hard_feasible": int(values[5]),
        "selected_hard_feasible_rate": values[5] / samples,
        "selected_margin_capable": int(values[6]),
        "selected_margin_capable_rate": values[6] / samples,
        "selected_future_capable": int(values[7]),
        "selected_future_capable_rate": values[7] / samples,
        "mean_oracle_regret": values[8] / max(1.0, values[9]),
        "oracle_regret_samples": int(values[9]),
        "elapsed_s": time.monotonic() - started,
    })
    return aggregate


def atomic_torch(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def write_artifact(path, model, optimizer, scaler, metadata, metrics, history):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "DEPCarRouteV2CheckpointV1",
        "architecture_id": V2_ARCHITECTURE,
        "training_stage": metadata["stage"],
        "modality": metadata["modality"],
        "artifact_role": metadata["artifact_role"],
        "status": "TRAINED_UNQUALIFIED",
        "qualification_status": "UNQUALIFIED",
        "production_qualified": False,
        "completed_epochs": metadata["completed_epochs"],
        "selected_epoch": metadata.get(
            "selected_epoch", metadata["completed_epochs"]
        ),
        "partial_epoch": False,
        "global_step": metadata["global_step"],
        "source_checkpoint_sha256": metadata["source_checkpoint_sha256"],
        "source_checkpoint": metadata["source_checkpoint"],
        "dataset_authority_sha256": metadata["dataset_authority_sha256"],
        "training_config_sha256": metadata["training_config_sha256"],
        "trainer_sha256": metadata["trainer_sha256"],
        "formal_training_authority_gate": metadata[
            "formal_training_authority_gate"
        ],
        "source_acceptance_gate": metadata["source_acceptance_gate"],
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "grad_scaler_state_dict": scaler.state_dict(),
        "metrics": metrics,
        "history": history,
    }
    atomic_torch(path, payload)
    contract = {
        "schema": "DEPCarRouteV2ArtifactContractV1",
        "architecture_id": V2_ARCHITECTURE,
        "objective_id": DEPCarObjectiveV2.objective_id,
        "checkpoint": str(path),
        "checkpoint_sha256": sha256_file(path),
        "training_stage": metadata["stage"],
        "modality": metadata["modality"],
        "artifact_role": metadata["artifact_role"],
        "status": "TRAINED_UNQUALIFIED",
        "production_qualified": False,
        "qualification_status": "UNQUALIFIED",
        "completed_epochs": metadata["completed_epochs"],
        "selected_epoch": metadata.get(
            "selected_epoch", metadata["completed_epochs"]
        ),
        "partial_epoch": False,
        "source_checkpoint_sha256": metadata["source_checkpoint_sha256"],
        "source_checkpoint": metadata["source_checkpoint"],
        "dataset_authority_sha256": metadata["dataset_authority_sha256"],
        "training_config_sha256": metadata["training_config_sha256"],
        "trainer_sha256": metadata["trainer_sha256"],
        "formal_training_authority_gate": metadata[
            "formal_training_authority_gate"
        ],
        "source_acceptance_gate": metadata["source_acceptance_gate"],
        "formal_dataset_authority_gate_passed": metadata[
            "formal_training_authority_gate"
        ]["passed"],
        "formal_training_contract_gate_passed": True,
        "metrics": metrics,
    }
    path.with_suffix(".contract.json").write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    path.with_suffix(".metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def finalize_best_artifact(path, *, completed_epochs, global_step, history):
    """Seal the selected epoch with evidence that the full run completed."""
    path = Path(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    selected_epoch = int(payload.get("selected_epoch", payload["completed_epochs"]))
    selected_global_step = int(payload.get("global_step", 0))
    payload.update({
        "completed_epochs": int(completed_epochs),
        "selected_epoch": selected_epoch,
        "selected_global_step": selected_global_step,
        "global_step": int(global_step),
        "run_completed": True,
        "partial_epoch": False,
        "history": list(history),
    })
    atomic_torch(path, payload)
    contract_path = path.with_suffix(".contract.json")
    contract = read_json(contract_path)
    contract.update({
        "checkpoint_sha256": sha256_file(path),
        "completed_epochs": int(completed_epochs),
        "selected_epoch": selected_epoch,
        "selected_global_step": selected_global_step,
        "global_step": int(global_step),
        "run_completed": True,
        "partial_epoch": False,
    })
    contract_path.write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def selection_key(stage, metrics):
    if stage == "candidate_capacity":
        return (
            metrics["zero_feasible_rate"],
            -metrics["future_capable_rate"],
            metrics["mean_loss"],
        )
    return (
        -metrics["selected_hard_feasible_rate"],
        -metrics["selected_future_capable_rate"],
        metrics["mean_oracle_regret"],
        metrics["mean_loss"],
    )


def _expected_best(path):
    return path.with_name(path.stem + ".best.pth")


def main(argv=None):
    args = build_parser().parse_args(argv)
    config, config_sha = load_config()
    training = config["training"]
    epochs = int(training["epochs"]) if args.epochs is None else args.epochs
    batch_size = (
        int(training["batch_size"]) if args.batch_size is None else args.batch_size
    )
    workers = args.workers if args.workers is not None else int(training["workers"])
    if min(epochs, batch_size, workers) < 1:
        raise SystemExit("epochs, batch size and workers must be positive")
    if args.max_samples is not None and args.max_samples < 1:
        raise SystemExit("max-samples must be positive")
    if args.max_steps is not None and args.max_steps < 1:
        raise SystemExit("max-steps must be positive")

    authority_path, authority, authority_gate = verify_authority(
        args.authority, config
    )
    source_path, source, source_gate = verify_source(
        args.source, args.stage, args.modality, config
    )
    output = resolve(args.output)
    if output == source_path:
        raise RuntimeError("source and output checkpoints must differ")
    bounded = args.max_samples is not None or args.max_steps is not None
    if bounded and (
        (args.max_samples or 0) > 1024 or (args.max_steps or 0) > 64
    ):
        raise RuntimeError("bounded V2 diagnostics are limited to 1024 samples/64 steps")
    formal_hyperparameters = (
        args.epochs is None and args.batch_size is None and args.workers is None
    )
    expected_output = resolve(config["artifacts"][
        "candidate" if args.stage == "candidate_capacity" else "score"
    ])
    expected_source = (
        resolve(config["initialization"]["fusion_candidate_v1"])
        if args.stage == "candidate_capacity"
        else _expected_best(resolve(config["artifacts"]["candidate"]))
    )
    formal_training_authorized = bool(
        authority_gate["passed"]
        and source_gate is not None
        and source_gate.get("passed") is True
        and args.modality in config["authorization"]["formal_modalities"]
        and not bounded
        and formal_hyperparameters
        and args.device == "cuda"
        and output == expected_output
        and source_path == expected_source
    )
    if args.modality != "fusion" and not bounded:
        raise RuntimeError(
            "depth-only and lidar-only are bounded diagnostics, not formal V2 training"
        )
    if not args.dry_run and not formal_training_authorized and not bounded:
        raise RuntimeError("V2 run is neither formally authorized nor bounded smoke")
    if not args.dry_run and output.exists() and args.resume is None:
        raise RuntimeError(
            "output already exists; pass --resume with the last checkpoint explicitly"
        )
    if args.resume is not None and resolve(args.resume) != output:
        raise RuntimeError("resume checkpoint must equal the selected last output")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    plan = {
        "schema": "DEPCarRouteV2TrainingPlanV1",
        "status": (
            "DRY_RUN_READY"
            if args.dry_run and formal_training_authorized
            else "BOUNDED_DIAGNOSTIC_READY"
            if bounded
            else "READY"
        ),
        "stage": args.stage,
        "modality": args.modality,
        "architecture_id": V2_ARCHITECTURE,
        "objective_id": DEPCarObjectiveV2.objective_id,
        "source": str(source_path),
        "source_sha256": sha256_file(source_path),
        "dataset_authority": str(authority_path),
        "dataset_authority_sha256": sha256_file(authority_path),
        "training_config": str(CONFIG.resolve()),
        "training_config_sha256": config_sha,
        "trainer_sha256": sha256_file(TRAINER),
        "output": str(output),
        "epochs": epochs,
        "batch_size": batch_size,
        "workers": workers,
        "device": args.device,
        "maximum_samples": args.max_samples,
        "maximum_steps": args.max_steps,
        "bounded_smoke": bounded,
        "formal_training_authorized": formal_training_authorized,
        "formal_training_authority_gate": authority_gate,
        "source_acceptance_gate": source_gate,
        "formal_modalities": config["authorization"]["formal_modalities"],
        "sensor_dropout_probability": (
            float(training["sensor_dropout_probability"])
            if args.modality == "fusion" else 0.0
        ),
        "prefetch_factor": int(training["prefetch_factor"]),
        "resume": str(resolve(args.resume)) if args.resume else None,
        "test_split_sealed": True,
        "production_qualified": False,
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0

    seed_all(int(training["seed"]))
    torch.set_num_threads(int(training["torch_threads"]))
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    model = DEPCarNetV2()
    if source.get("architecture_id") == V1_ARCHITECTURE:
        model.initialize_from_v1(source["model_state_dict"])
    else:
        model.load_state_dict(source["model_state_dict"], strict=True)
    model.to(device)
    ownership = configure_training_stage(model, args.stage)
    optimizer = build_optimizer(
        model, args.stage,
        float(training["learning_rate"]), float(training["weight_decay"]),
    )
    amp = bool(training["mixed_precision"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    common = dict(
        sample_root=authority["sample_root"], maps_root=authority["maps_root"],
        index_path=authority["index"], index_splits=("train", "validation"),
        workers=workers,
        expected_map_contract_aggregate_sha256=authority[
            "map_contract_aggregate_sha256"
        ],
        expected_index_sha256=authority["index_sha256"], modality=args.modality,
    )
    train_data = P3ScoreTrainingDatasetV1(split="train", **common)
    validation_data = P3ScoreTrainingDatasetV1(split="validation", **common)
    if args.max_samples:
        train_data = Subset(
            train_data, range(min(len(train_data), args.max_samples))
        )
        validation_data = Subset(
            validation_data, range(min(len(validation_data), args.max_samples))
        )
    loader_common = dict(
        batch_size=batch_size,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        prefetch_factor=int(training["prefetch_factor"]) if workers > 0 else None,
        collate_fn=p3_training_collate,
        worker_init_fn=p3_training_worker_init,
    )
    train_loader = DataLoader(train_data, shuffle=True, **loader_common)
    validation_loader = DataLoader(validation_data, shuffle=False, **loader_common)
    objective = DEPCarObjectiveV2(DEPCarRouteLossConfigV2(**config["loss"]))
    history, global_step, best_key, start_epoch = [], 0, None, 1
    if args.resume is not None:
        resumed = torch.load(output, map_location="cpu", weights_only=True)
        expected_resume = {
            "schema": "DEPCarRouteV2CheckpointV1",
            "architecture_id": V2_ARCHITECTURE,
            "training_stage": args.stage,
            "modality": args.modality,
            "artifact_role": "last",
            "partial_epoch": False,
            "dataset_authority_sha256": plan["dataset_authority_sha256"],
            "training_config_sha256": config_sha,
            "trainer_sha256": plan["trainer_sha256"],
            "source_checkpoint_sha256": plan["source_sha256"],
            "source_acceptance_gate": source_gate,
        }
        mismatches = [
            key for key, expected in expected_resume.items()
            if resumed.get(key) != expected
        ]
        if mismatches:
            raise RuntimeError("resume checkpoint mismatch: " + ",".join(mismatches))
        completed = int(resumed.get("completed_epochs", 0))
        if completed < 1 or completed >= epochs:
            raise RuntimeError("resume completed epoch is outside the remaining run")
        model.load_state_dict(resumed["model_state_dict"], strict=True)
        optimizer.load_state_dict(resumed["optimizer_state_dict"])
        scaler.load_state_dict(resumed["grad_scaler_state_dict"])
        history = list(resumed.get("history", ()))
        global_step = int(resumed.get("global_step", 0))
        start_epoch = completed + 1
        keys = [
            selection_key(args.stage, row["validation"])
            for row in history
            if isinstance(row, dict) and isinstance(row.get("validation"), dict)
        ]
        best_key = min(keys) if keys else None
        model.to(device)

    with amp_encoder_outputs_fp32(model, amp):
        for epoch in range(start_epoch, epochs + 1):
            started = time.time()
            train_metrics = epoch_loop(
                model, objective, train_loader, args.stage, device, amp,
                optimizer, scaler, args.max_steps,
                float(training["sensor_dropout_probability"])
                if args.modality == "fusion" else 0.0,
                int(training["progress_interval_steps"]),
            )
            validation_metrics = epoch_loop(
                model, objective, validation_loader, args.stage, device, amp,
                max_steps=args.max_steps,
                progress_interval=int(training["progress_interval_steps"]),
            )
            global_step += train_metrics["steps"]
            row = {
                "epoch": epoch,
                "train": train_metrics,
                "validation": validation_metrics,
                "elapsed_s": time.time() - started,
            }
            history.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
            key = selection_key(args.stage, validation_metrics)
            metadata = {
                "stage": args.stage,
                "modality": args.modality,
                "artifact_role": "last",
                "completed_epochs": epoch,
                "selected_epoch": epoch,
                "global_step": global_step,
                "source_checkpoint_sha256": plan["source_sha256"],
                "source_checkpoint": str(source_path),
                "dataset_authority_sha256": plan["dataset_authority_sha256"],
                "training_config_sha256": config_sha,
                "trainer_sha256": plan["trainer_sha256"],
                "formal_training_authority_gate": authority_gate,
                "source_acceptance_gate": source_gate,
            }
            write_artifact(
                output, model, optimizer, scaler, metadata,
                validation_metrics, history,
            )
            if best_key is None or key < best_key:
                best_key = key
                best_path = _expected_best(output)
                metadata["artifact_role"] = "best"
                write_artifact(
                    best_path, model, optimizer, scaler, metadata,
                    validation_metrics, history,
                )
    best_path = _expected_best(output)
    if not best_path.is_file():
        raise RuntimeError("V2 training completed without a best checkpoint")
    finalize_best_artifact(
        best_path,
        completed_epochs=epochs,
        global_step=global_step,
        history=history,
    )
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
