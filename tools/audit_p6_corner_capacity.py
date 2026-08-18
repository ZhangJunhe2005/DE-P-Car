#!/usr/bin/env python3
"""Audit whether the accepted Candidate bank can negotiate 90-degree turns.

The audit deliberately separates Candidate capacity from Score selection.  A
candidate is counted as corner-capable only when it is collision-free with a
positive operating margin, follows the route tube, and leaves at least one
safe same-gear primitive for the following planning cycle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dep_car/src"))
sys.path.insert(0, str(ROOT / "tools"))

from dep_car.model.dep_car_net import DEPCarNetV1
from dep_car.model.dep_car_net_v2 import DEPCarNetV2
from dep_car.training.losses import (
    DEPCarObjectiveV1,
    swept_map_footprint_clearance,
)
from dep_car.training.p4_dataset import p3_training_collate, p3_training_worker_init
from dep_car.training.score_dataset import P3ScoreTrainingDatasetV1
import train_dep_car_route_v2 as route_trainer


CONFIG = ROOT / "dep_car/config/p5_route_v2_candidate_acceptance.yaml"
REPORT = ROOT / "reports/p5_route_v2_fusion_candidate_acceptance.json"


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--checkpoint", required=True)
    value.add_argument(
        "--authority",
        default="data/p3_v4/bundle_v1/bundle_authority.json",
    )
    value.add_argument("--modality", choices=("depth_only", "lidar_only", "fusion"), default="fusion")
    value.add_argument("--output", default=str(REPORT))
    value.add_argument("--maximum-samples", type=int, default=0)
    value.add_argument("--batch-size", type=int, default=16)
    value.add_argument("--workers", type=int, default=8)
    value.add_argument("--device", default="cuda")
    return value


def _resolve(path):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_contract():
    raw = CONFIG.read_bytes()
    config = yaml.safe_load(raw)
    if (
        config.get("schema") != "DEPCarRouteV2CandidateAcceptanceContractV1"
        or config.get("architecture_id") != DEPCarNetV2.architecture_id
        or config.get("population", {}).get("split") != "validation"
        or config.get("population", {}).get("maneuver_mode") != "SHARP_TURN"
        or config.get("population", {}).get("geometry_valid_only") is not True
        or config.get("population", {}).get("maximum_samples") != 0
        or config.get("authority", {}).get("test_split_sealed") is not True
    ):
        raise RuntimeError("V2 Candidate acceptance contract is invalid")
    return config, hashlib.sha256(raw).hexdigest()


def _amp_output_fp32(_module, _arguments, output):
    return output.float()


def _future_viability(output, batch, model, clearance_margin):
    """Return safe next primitives and their current-body trajectories."""

    trajectories = output.trajectories.float()
    batch_size, candidates = trajectories.shape[:2]
    endpoint = trajectories[:, :, -1]
    next_state = batch["state"][:, None, :].expand(-1, candidates, -1).clone()
    next_state[..., 0] = endpoint[..., 4]
    next_state[..., 1] = 0.0
    next_state[..., 2] = endpoint[..., 5]
    next_state[..., 3] = 0.0
    next_state = next_state.reshape(batch_size * candidates, -1)
    next_gear = batch["requested_gear"][:, None].expand(-1, candidates).reshape(-1)
    zeros = next_state.new_zeros((batch_size * candidates, 15, 4))
    following = model.rollout(next_state, next_gear, zeros).trajectory

    parent = endpoint.reshape(batch_size * candidates, 6)
    yaw = parent[:, 3][:, None, None]
    cosine, sine = torch.cos(yaw), torch.sin(yaw)
    relative_x = following[..., 1]
    relative_y = following[..., 2]
    transformed = following.clone()
    transformed[..., 1] = parent[:, 1][:, None, None] + cosine * relative_x - sine * relative_y
    transformed[..., 2] = parent[:, 2][:, None, None] + sine * relative_x + cosine * relative_y
    transformed[..., 3] = parent[:, 3][:, None, None] + following[..., 3]

    repeat = lambda value: value[:, None].expand(-1, candidates, *value.shape[1:]).reshape(
        batch_size * candidates, *value.shape[1:]
    )
    clearance = swept_map_footprint_clearance(
        transformed,
        repeat(batch["map_distance_field"]),
        repeat(batch["map_resolution"]),
        repeat(batch["map_origin"]),
        repeat(batch["chassis_to_map"]),
    ).amin(dim=(-1, -2))
    safe = (clearance >= float(clearance_margin)).reshape(
        batch_size, candidates, 15
    )
    transformed = transformed.reshape(
        batch_size, candidates * 15, transformed.shape[-2], transformed.shape[-1]
    )
    return safe, transformed


def _route_tube_quality(trajectories, route, route_mask):
    """Measure one-cycle corridor adherence without demanding route completion."""

    positions = trajectories[..., 1:3]
    route_xy = route[..., :2]
    # P3 stores the future path beginning at the first expert sample, not the
    # current chassis origin.  The route corridor is continuous from the
    # current pose to that first sample; omitting this segment falsely calls a
    # valid short-horizon approach cross-track error.
    route_xy = torch.cat((route_xy.new_zeros((len(route_xy), 1, 2)), route_xy), dim=1)
    route_mask = torch.cat(
        (torch.ones((len(route_mask), 1), dtype=torch.bool, device=route_mask.device), route_mask),
        dim=1,
    )
    starts = route_xy[:, :-1]
    segments = route_xy[:, 1:] - starts
    lengths = torch.linalg.vector_norm(segments, dim=-1)
    valid = (route_mask[:, 1:] & route_mask[:, :-1]) & (lengths > 1.0e-5)
    delta = positions[..., None, :] - starts[:, None, None, :, :]
    projection = (
        (delta * segments[:, None, None, :, :]).sum(dim=-1)
        / lengths.square().clamp_min(1.0e-8)[:, None, None, :]
    ).clamp(0.0, 1.0)
    closest = starts[:, None, None, :, :] + projection[..., None] * segments[
        :, None, None, :, :
    ]
    pairwise = torch.linalg.vector_norm(
        positions[..., None, :] - closest, dim=-1
    ).masked_fill(~valid[:, None, None, :], 1.0e4)
    cross_track, nearest_segment = pairwise.min(dim=-1)
    mean_cross_track = cross_track[..., 1:].mean(dim=-1)
    lengths = lengths * valid.to(lengths)
    arc_start = torch.cat(
        (lengths.new_zeros((len(route), 1)), torch.cumsum(lengths, dim=1)[:, :-1]),
        dim=1,
    )
    segment_progress = arc_start[:, None, None, :] + projection * lengths[
        :, None, None, :
    ]
    nearest_progress = torch.gather(
        segment_progress, -1, nearest_segment[..., None]
    ).squeeze(-1)
    return (
        mean_cross_track,
        nearest_progress[..., -1] - nearest_progress[..., 0],
        nearest_progress[..., -1],
    )


def main(argv=None):
    args = parser().parse_args(argv)
    if args.maximum_samples < 0 or min(args.batch_size, args.workers) < 1:
        raise SystemExit(
            "sample limit must be non-negative; batch/workers must be positive"
        )
    contract, contract_sha = _load_contract()
    thresholds = contract["thresholds"]
    formal = args.maximum_samples == 0
    report_path = _resolve(args.output)
    if not formal and report_path == REPORT.resolve():
        report_path = REPORT.with_name(REPORT.stem + "_smoke.json")
    if formal and (
        args.modality != contract["execution"]["modality"]
        or args.batch_size != int(contract["execution"]["batch_size"])
        or args.workers != int(contract["execution"]["workers"])
        or args.device != contract["execution"]["device"]
        or report_path != REPORT.resolve()
    ):
        raise RuntimeError("formal V2 Candidate acceptance execution contract differs")

    training_config, _ = route_trainer.load_config()
    authority_path, authority, authority_gate = route_trainer.verify_authority(
        args.authority, training_config
    )
    if (
        authority_path != _resolve(contract["authority"]["bundle"])
        or _sha256(authority_path) != contract["authority"]["bundle_file_sha256"]
    ):
        raise RuntimeError("Candidate acceptance bundle authority differs")

    checkpoint = _resolve(args.checkpoint)
    expected_checkpoint = route_trainer._expected_best(
        _resolve(training_config["artifacts"]["candidate"])
    )
    checkpoint_sha = _sha256(checkpoint)
    checkpoint_payload = torch.load(
        checkpoint, map_location="cpu", weights_only=True
    )
    checkpoint_contract_path = checkpoint.with_suffix(".contract.json")
    checkpoint_contract = json.loads(
        checkpoint_contract_path.read_text(encoding="utf-8")
    )
    dry_run_path = _resolve(contract["authority"]["candidate_dry_run"])
    dry_run = json.loads(dry_run_path.read_text(encoding="utf-8"))
    artifact_errors = []
    expected_artifact = {
        "schema": "DEPCarRouteV2CheckpointV1",
        "architecture_id": DEPCarNetV2.architecture_id,
        "training_stage": "candidate_capacity",
        "modality": "fusion",
        "artifact_role": "best",
        "status": "TRAINED_UNQUALIFIED",
        "qualification_status": "UNQUALIFIED",
        "production_qualified": False,
        "completed_epochs": 40,
        "partial_epoch": False,
        "run_completed": True,
        "dataset_authority_sha256": _sha256(authority_path),
    }
    if checkpoint != expected_checkpoint:
        artifact_errors.append("checkpoint_noncanonical_path")
    artifact_errors.extend(
        "checkpoint_" + key
        for key, expected in expected_artifact.items()
        if checkpoint_payload.get(key) != expected
    )
    if (
        checkpoint_contract.get("schema") != "DEPCarRouteV2ArtifactContractV1"
        or checkpoint_contract.get("checkpoint_sha256") != checkpoint_sha
        or checkpoint_contract.get("formal_training_authority_gate", {}).get(
            "passed"
        ) is not True
        or checkpoint_contract.get("trainer_sha256")
        != checkpoint_payload.get("trainer_sha256")
        or checkpoint_contract.get("training_config_sha256")
        != checkpoint_payload.get("training_config_sha256")
    ):
        artifact_errors.append("checkpoint_contract_identity")
    if (
        dry_run.get("status") != "DRY_RUN_READY"
        or dry_run.get("formal_training_authorized") is not True
        or dry_run.get("trainer_sha256") != checkpoint_payload.get("trainer_sha256")
        or dry_run.get("training_config_sha256")
        != checkpoint_payload.get("training_config_sha256")
        or dry_run.get("dataset_authority_sha256") != _sha256(authority_path)
    ):
        artifact_errors.append("candidate_dry_run_identity")
    if artifact_errors:
        raise RuntimeError(
            "V2 Candidate artifact is not formally auditable: "
            + ",".join(sorted(set(artifact_errors)))
        )

    dataset = P3ScoreTrainingDatasetV1(
        authority["sample_root"],
        authority["maps_root"],
        split="validation",
        index_path=authority["index"],
        index_splits=("train", "validation"),
        workers=args.workers,
        expected_map_contract_aggregate_sha256=authority[
            "map_contract_aggregate_sha256"
        ],
        expected_index_sha256=authority["index_sha256"],
        modality=args.modality,
    )
    all_indices = [
        index
        for index, entry in enumerate(dataset.entries)
        if entry.get("maneuver_mode") == "SHARP_TURN"
    ]
    if args.maximum_samples and len(all_indices) > args.maximum_samples:
        selected = np.linspace(
            0, len(all_indices) - 1, args.maximum_samples, dtype=np.int64
        )
        indices = [all_indices[index] for index in selected]
    else:
        indices = all_indices
    if not indices:
        raise RuntimeError("validation split contains no SHARP_TURN samples")
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=str(args.device).startswith("cuda"),
        persistent_workers=args.workers > 0,
        prefetch_factor=4 if args.workers > 0 else None,
        collate_fn=p3_training_collate,
        worker_init_fn=p3_training_worker_init,
    )
    device = torch.device(args.device)
    model = DEPCarNetV2()
    model.load_state_dict(checkpoint_payload["model_state_dict"], strict=True)
    model.to(device).eval()
    handles = []
    if device.type == "cuda":
        handles = [
            model.depth_encoder.register_forward_hook(_amp_output_fp32),
            model.lidar_encoder.register_forward_hook(_amp_output_fp32),
        ]
    objective = DEPCarObjectiveV1()
    totals = {
        "samples": 0,
        "capacity_capable": 0,
        "selected_capable": 0,
        "zero_hard_feasible": 0,
        "zero_margin_feasible": 0,
        "route_tube_capable": 0,
        "followup_capable": 0,
    }
    regrets = []
    feasible_counts = []
    maximum_route_progress = []
    minimum_route_cross_track = []
    best_progress_inside_tube = []
    try:
        with torch.inference_mode():
            for host in loader:
                valid = host["geometry_valid"].bool()
                if not bool(valid.any()):
                    continue
                batch = {
                    key: value[valid].to(device, non_blocking=True)
                    for key, value in host.items()
                    if key != "metadata"
                }
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=device.type == "cuda",
                ):
                    output = model(
                        batch["depth"],
                        batch["lidar_bev"],
                        batch["state"],
                        batch["requested_gear"],
                        batch["route_pose"],
                        batch["route_mask"],
                        batch["modality_mask"],
                    )
                metrics = objective(
                    output,
                    map_distance_field=batch["map_distance_field"],
                    map_resolution=batch["map_resolution"],
                    map_origin=batch["map_origin"],
                    chassis_to_map=batch["chassis_to_map"],
                    route=batch["route_pose"],
                    route_mask=batch["route_mask"],
                    requested_gear=batch["requested_gear"],
                    geometry_valid=None,
                    stage="score_calibration",
                )
                future_safe, future_trajectories = _future_viability(
                    output, batch, model, thresholds["clearance_margin_m"]
                )
                mean_cross_track, route_progress, _ = _route_tube_quality(
                    output.trajectories,
                    batch["route_pose"],
                    batch["route_mask"],
                )
                future_cross_track, _, future_endpoint_progress = _route_tube_quality(
                    future_trajectories,
                    batch["route_pose"],
                    batch["route_mask"],
                )
                future_cross_track = future_cross_track.reshape(-1, 15, 15)
                future_endpoint_progress = future_endpoint_progress.reshape(-1, 15, 15)
                hard = metrics["hard_feasible"]
                margin = (
                    metrics["minimum_clearance"] >= thresholds["clearance_margin_m"]
                ) & ~metrics["kinematic_violation"]
                future_route_capable = (
                    future_safe
                    & (future_cross_track <= thresholds["route_tube_radius_m"])
                    & (
                        future_endpoint_progress
                        >= thresholds["minimum_route_progress_m"]
                    )
                )
                route_capable = (
                    mean_cross_track <= thresholds["route_tube_radius_m"]
                ) & future_route_capable.any(dim=2)
                capable = margin & route_capable
                selected_candidate = output.scores.argmin(dim=1)
                selected_capable = capable.gather(
                    1, selected_candidate[:, None]
                ).squeeze(1)
                oracle = metrics["candidate_cost"].masked_fill(
                    ~hard, 1.0e4
                ).amin(dim=1)
                chosen = metrics["candidate_cost"].gather(
                    1, selected_candidate[:, None]
                ).squeeze(1)
                count = len(selected_candidate)
                totals["samples"] += count
                totals["capacity_capable"] += int(capable.any(dim=1).sum().item())
                totals["selected_capable"] += int(selected_capable.sum().item())
                totals["zero_hard_feasible"] += int((~hard.any(dim=1)).sum().item())
                totals["zero_margin_feasible"] += int((~margin.any(dim=1)).sum().item())
                totals["route_tube_capable"] += int(
                    route_capable.any(dim=1).sum().item()
                )
                totals["followup_capable"] += int(
                    future_route_capable.any(dim=(1, 2)).sum().item()
                )
                regrets.extend((chosen - oracle).clamp_min(0.0).cpu().tolist())
                feasible_counts.extend(hard.sum(dim=1).cpu().tolist())
                maximum_route_progress.extend(
                    route_progress.amax(dim=1).cpu().tolist()
                )
                minimum_route_cross_track.extend(
                    mean_cross_track.amin(dim=1).cpu().tolist()
                )
                progress_in_tube = route_progress.masked_fill(
                    mean_cross_track > thresholds["route_tube_radius_m"], -1.0e4
                ).amax(dim=1)
                best_progress_inside_tube.extend(progress_in_tube.cpu().tolist())
    finally:
        for handle in handles:
            handle.remove()

    samples = totals["samples"]
    capacity_rate = totals["capacity_capable"] / max(1, samples)
    selected_rate = totals["selected_capable"] / max(1, samples)
    zero_hard_rate = totals["zero_hard_feasible"] / max(1, samples)
    capacity_pass = capacity_rate >= thresholds["minimum_capacity_capable_rate"]
    hard_pass = zero_hard_rate <= thresholds["maximum_zero_hard_feasible_rate"]
    gate_passed = bool(formal and capacity_pass and hard_pass)
    if capacity_pass and selected_rate + 0.10 < capacity_rate:
        conclusion = "CAPACITY_PASS_RANKING_FAIL"
    elif capacity_pass:
        conclusion = "CAPACITY_AND_RANKING_PASS"
    else:
        conclusion = "CAPACITY_FAIL"
    errors = []
    if not capacity_pass:
        errors.append("corner_capacity_capable_rate")
    if not hard_pass:
        errors.append("corner_zero_hard_feasible_rate")
    if not formal:
        errors.append("diagnostic_limited_candidate_audit")
    report = {
        "schema": "DEPCarRouteV2CandidateAcceptanceV1",
        "status": "PASS" if gate_passed else "SMOKE" if not formal else "FAIL",
        "gate_passed": gate_passed,
        "smoke_limited": not formal,
        "errors": errors,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "conclusion": conclusion,
        "modality": args.modality,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_contract": str(checkpoint_contract_path),
        "checkpoint_contract_sha256": _sha256(checkpoint_contract_path),
        "candidate_training_config_sha256": checkpoint_payload.get(
            "training_config_sha256"
        ),
        "candidate_trainer_sha256": checkpoint_payload.get("trainer_sha256"),
        "acceptance_contract": str(CONFIG.resolve()),
        "acceptance_contract_sha256": contract_sha,
        "acceptance_tool": str(Path(__file__).resolve()),
        "acceptance_tool_sha256": _sha256(__file__),
        "dataset_authority": str(authority_path),
        "dataset_authority_sha256": _sha256(authority_path),
        "formal_training_authority_gate": authority_gate,
        "population": "validation/SHARP_TURN/geometry_valid",
        "population_frames_available": len(all_indices),
        "population_frames_selected": len(indices),
        "test_split_accessed": False,
        "thresholds": dict(thresholds),
        "gates": {
            "capacity_capable_rate": {
                "observed": capacity_rate,
                "operator": ">=",
                "threshold": thresholds["minimum_capacity_capable_rate"],
                "status": "PASS" if capacity_pass else "FAIL",
            },
            "zero_hard_feasible_rate": {
                "observed": zero_hard_rate,
                "operator": "<=",
                "threshold": thresholds["maximum_zero_hard_feasible_rate"],
                "status": "PASS" if hard_pass else "FAIL",
            },
        },
        "metrics": {
            **totals,
            "capacity_capable_rate": capacity_rate,
            "selected_capable_rate": selected_rate,
            "zero_hard_feasible_rate": zero_hard_rate,
            "zero_margin_feasible_rate": totals["zero_margin_feasible"]
            / max(1, samples),
            "feasible_candidates_median": float(np.median(feasible_counts)),
            "oracle_regret_mean": float(np.mean(regrets)),
            "maximum_route_progress_m_percentiles": {
                name: float(np.percentile(maximum_route_progress, percentile))
                for name, percentile in (("p10", 10), ("p50", 50), ("p90", 90))
            },
            "minimum_route_cross_track_m_percentiles": {
                name: float(np.percentile(minimum_route_cross_track, percentile))
                for name, percentile in (("p10", 10), ("p50", 50), ("p90", 90))
            },
            "best_progress_inside_tube_m_percentiles": {
                name: float(np.percentile(best_progress_inside_tube, percentile))
                for name, percentile in (("p10", 10), ("p50", 50), ("p90", 90))
            },
        },
        "decision": (
            "preserve Candidate checkpoint; train route-conditioned Score V2"
            if conclusion == "CAPACITY_PASS_RANKING_FAIL"
            else "retrain Candidate and Score"
            if conclusion == "CAPACITY_FAIL"
            else "Candidate and inherited ranking both pass"
        ),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
    report_path.write_text(serialized, encoding="utf-8")
    if formal:
        checkpoint.with_suffix(".candidate_acceptance.json").write_text(
            serialized, encoding="utf-8"
        )
    print(serialized, end="")
    return 0 if report["status"] in ("PASS", "SMOKE") else 1


if __name__ == "__main__":
    raise SystemExit(main())
