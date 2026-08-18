#!/usr/bin/env python3
"""Qualify the formal Fusion V2 Score artifact for P6 shadow evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dep_car/src"))
sys.path.insert(0, str(ROOT / "tools"))

from dep_car.model.dep_car_net_v2 import DEPCarNetV2
from dep_car.training.losses_v2 import DEPCarObjectiveV2, DEPCarRouteLossConfigV2
from dep_car.training.p4_dataset import p3_training_collate, p3_training_worker_init
from dep_car.training.score_dataset import P3ScoreTrainingDatasetV1
import train_dep_car_route_v2 as trainer


CONTRACT = ROOT / "dep_car/config/p5_route_v2_score_acceptance.yaml"
REPORT = ROOT / "reports/p5_route_v2_fusion_score_shadow_acceptance.json"
REQUIRED_MODES = (
    "NORMAL", "SHARP_TURN", "NARROW_CORRIDOR", "U_TURN",
    "DEAD_END_ESCAPE", "REVERSE_EXIT", "THREE_POINT_TURN",
)


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


def load_contract():
    raw = CONTRACT.read_bytes()
    value = yaml.safe_load(raw)
    if (
        value.get("schema") != "DEPCarRouteV2ScoreShadowAcceptanceContractV1"
        or value.get("architecture_id") != DEPCarNetV2.architecture_id
        or value.get("population", {}).get("split") != "validation"
        or value.get("population", {}).get("geometry_valid_only") is not True
        or value.get("population", {}).get("maximum_samples") != 0
        or value.get("authority", {}).get("test_split_sealed") is not True
        or value.get("scope") != "P6_SHADOW_ONLY"
    ):
        raise RuntimeError("Score shadow acceptance contract is invalid")
    return value, hashlib.sha256(raw).hexdigest()


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument(
        "--checkpoint",
        default="models/dep_car/p5_route_v2/fusion_score_calibration.best.pth",
    )
    value.add_argument(
        "--authority", default="data/p3_v4/bundle_v1/bundle_authority.json"
    )
    value.add_argument("--output", default=str(REPORT))
    value.add_argument("--maximum-samples", type=int, default=0)
    value.add_argument("--batch-size", type=int, default=128)
    value.add_argument("--workers", type=int, default=8)
    value.add_argument("--device", default="cuda")
    return value


def verify_artifact(checkpoint, authority_path, config, config_sha):
    expected = trainer._expected_best(resolve(config["artifacts"]["score"]))
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    contract_path = checkpoint.with_suffix(".contract.json")
    contract = read_json(contract_path)
    errors = []
    required = {
        "schema": "DEPCarRouteV2CheckpointV1",
        "architecture_id": DEPCarNetV2.architecture_id,
        "training_stage": "score_calibration",
        "modality": "fusion",
        "artifact_role": "best",
        "status": "TRAINED_UNQUALIFIED",
        "qualification_status": "UNQUALIFIED",
        "production_qualified": False,
        "completed_epochs": int(config["training"]["epochs"]),
        "partial_epoch": False,
        "run_completed": True,
        "training_config_sha256": config_sha,
        "trainer_sha256": sha256_file(trainer.TRAINER),
        "dataset_authority_sha256": sha256_file(authority_path),
    }
    if checkpoint != expected:
        errors.append("checkpoint_noncanonical_path")
    errors.extend(
        "checkpoint_" + key
        for key, expected_value in required.items()
        if payload.get(key) != expected_value
    )
    if (
        contract.get("schema") != "DEPCarRouteV2ArtifactContractV1"
        or contract.get("checkpoint_sha256") != sha256_file(checkpoint)
        or contract.get("training_stage") != "score_calibration"
        or contract.get("artifact_role") != "best"
        or contract.get("run_completed") is not True
        or contract.get("source_acceptance_gate")
        != payload.get("source_acceptance_gate")
    ):
        errors.append("checkpoint_contract_identity")
    source = payload.get("source_acceptance_gate", {})
    source_checkpoint = resolve(source.get("checkpoint", ""))
    source_contract = resolve(source.get("checkpoint_contract", ""))
    source_acceptance = resolve(source.get("acceptance_sidecar", ""))
    source_report = resolve(source.get("formal_acceptance_report", ""))
    try:
        source_identity_valid = (
            source.get("schema") == "DEPCarRouteV2ScoreSourceGateV1"
            and source.get("passed") is True
            and source.get("errors") == []
            and source.get("test_split_accessed") is False
            and sha256_file(source_checkpoint) == source.get("checkpoint_sha256")
            and sha256_file(source_contract)
            == source.get("checkpoint_contract_sha256")
            and sha256_file(source_acceptance)
            == source.get("acceptance_sidecar_sha256")
            and sha256_file(source_report)
            == source.get("formal_acceptance_report_sha256")
        )
    except OSError:
        source_identity_valid = False
    if not source_identity_valid:
        errors.append("candidate_source_acceptance_identity")
    if payload.get("source_checkpoint_sha256") != source.get("checkpoint_sha256"):
        errors.append("candidate_source_checkpoint_lineage")
    if errors:
        raise RuntimeError(
            "formal Score artifact failed: " + ",".join(sorted(set(errors)))
        )
    return payload, contract_path, contract


def empty_totals():
    return defaultdict(float)


def add_metrics(totals, *, feasible, margin, future, scores, costs):
    selected = scores.argmin(dim=1)[:, None]
    selected_hard = feasible.gather(1, selected).squeeze(1)
    selected_margin = margin.gather(1, selected).squeeze(1) & selected_hard
    selected_future = future.gather(1, selected).squeeze(1) & selected_margin
    any_feasible = feasible.any(dim=1)
    oracle = costs.masked_fill(~feasible, torch.inf).amin(dim=1)
    chosen = costs.gather(1, selected).squeeze(1)
    regret = torch.where(
        any_feasible, (chosen - oracle).clamp_min(0.0), torch.zeros_like(chosen)
    )
    totals["samples"] += len(scores)
    totals["zero_feasible"] += int((~any_feasible).sum().item())
    totals["selected_hard"] += int(selected_hard.sum().item())
    totals["selected_margin"] += int(selected_margin.sum().item())
    totals["selected_future"] += int(selected_future.sum().item())
    totals["oracle_samples"] += int(any_feasible.sum().item())
    totals["oracle_regret_sum"] += float(regret.sum().item())


def summarize(totals):
    samples = max(1, int(totals["samples"]))
    oracle_samples = max(1, int(totals["oracle_samples"]))
    return {
        "samples": int(totals["samples"]),
        "zero_feasible_rate": totals["zero_feasible"] / samples,
        "selected_hard_feasible_rate": totals["selected_hard"] / samples,
        "selected_margin_capable_rate": totals["selected_margin"] / samples,
        "selected_future_capable_rate": totals["selected_future"] / samples,
        "mean_oracle_regret": totals["oracle_regret_sum"] / oracle_samples,
    }


def main(argv=None):
    args = parser().parse_args(argv)
    if args.maximum_samples < 0 or min(args.batch_size, args.workers) < 1:
        raise SystemExit("sample limit must be non-negative; batch/workers must be positive")
    acceptance, acceptance_sha = load_contract()
    execution = acceptance["execution"]
    formal = args.maximum_samples == 0
    output = resolve(args.output)
    if not formal and output == REPORT.resolve():
        output = REPORT.with_name(REPORT.stem + "_smoke.json")
    if formal and (
        args.batch_size != int(execution["batch_size"])
        or args.workers != int(execution["workers"])
        or args.device != execution["device"]
        or output != REPORT.resolve()
    ):
        raise RuntimeError("formal Score acceptance execution contract differs")

    config, config_sha = trainer.load_config()
    authority_path, authority, authority_gate = trainer.verify_authority(
        args.authority, config
    )
    if authority_path != resolve(acceptance["authority"]["bundle"]):
        raise RuntimeError("Score acceptance dataset authority differs")
    checkpoint = resolve(args.checkpoint)
    payload, contract_path, _ = verify_artifact(
        checkpoint, authority_path, config, config_sha
    )
    dataset = P3ScoreTrainingDatasetV1(
        authority["sample_root"], authority["maps_root"], split="validation",
        index_path=authority["index"], index_splits=("train", "validation"),
        workers=args.workers,
        expected_map_contract_aggregate_sha256=authority[
            "map_contract_aggregate_sha256"
        ],
        expected_index_sha256=authority["index_sha256"], modality="fusion",
    )
    indices = list(range(len(dataset)))
    if args.maximum_samples and len(indices) > args.maximum_samples:
        selected = np.linspace(0, len(indices) - 1, args.maximum_samples, dtype=np.int64)
        indices = [indices[index] for index in selected]
    loader = DataLoader(
        Subset(dataset, indices), batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=str(args.device).startswith("cuda"),
        persistent_workers=args.workers > 0,
        prefetch_factor=4 if args.workers > 0 else None,
        collate_fn=p3_training_collate, worker_init_fn=p3_training_worker_init,
    )
    device = torch.device(args.device)
    model = DEPCarNetV2()
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device).eval()
    objective = DEPCarObjectiveV2(DEPCarRouteLossConfigV2(**config["loss"]))
    totals = {"OVERALL": empty_totals()}
    for mode in REQUIRED_MODES:
        totals["MODE:" + mode] = empty_totals()
    for gear in ("FORWARD", "REVERSE"):
        totals["GEAR:" + gear] = empty_totals()
    with trainer.amp_encoder_outputs_fp32(model, device.type == "cuda"):
        with torch.inference_mode():
            for host in loader:
                valid = host["geometry_valid"].bool()
                if not bool(valid.any()):
                    continue
                metadata = [row for row, keep in zip(host["metadata"], valid) if bool(keep)]
                batch = trainer.select_valid(host, device)
                network, losses = trainer.forward_loss(
                    model, objective, batch, "score_calibration",
                    device.type == "cuda",
                )
                feasible = losses["hard_feasible"]
                margin = (
                    losses["minimum_clearance"] >= objective.config.clearance_margin_m
                )
                future = (
                    losses["future_minimum_clearance"] >= objective.config.clearance_margin_m
                )
                gears = batch["requested_gear"]
                group_names = ["OVERALL"]
                group_masks = [torch.ones(len(gears), dtype=torch.bool, device=device)]
                for mode in REQUIRED_MODES:
                    group_names.append("MODE:" + mode)
                    group_masks.append(torch.tensor(
                        [row["maneuver_mode"] == mode for row in metadata],
                        dtype=torch.bool, device=device,
                    ))
                group_names.extend(("GEAR:FORWARD", "GEAR:REVERSE"))
                group_masks.extend((gears > 0, gears < 0))
                for name, mask in zip(group_names, group_masks):
                    if not bool(mask.any()):
                        continue
                    add_metrics(
                        totals[name], feasible=feasible[mask], margin=margin[mask],
                        future=future[mask], scores=network.scores[mask],
                        costs=losses["candidate_cost"][mask],
                    )
    metrics = {name: summarize(row) for name, row in totals.items()}
    thresholds = acceptance["thresholds"]
    checks = {
        "overall_zero_feasible_rate": metrics["OVERALL"]["zero_feasible_rate"]
        <= thresholds["maximum_overall_zero_feasible_rate"],
        "overall_selected_hard_feasible_rate": metrics["OVERALL"]["selected_hard_feasible_rate"]
        >= thresholds["minimum_overall_selected_hard_feasible_rate"],
        "overall_selected_future_capable_rate": metrics["OVERALL"]["selected_future_capable_rate"]
        >= thresholds["minimum_overall_selected_future_capable_rate"],
        "overall_mean_oracle_regret": metrics["OVERALL"]["mean_oracle_regret"]
        <= thresholds["maximum_overall_mean_oracle_regret"],
        "sharp_turn_selected_hard_feasible_rate": metrics["MODE:SHARP_TURN"]["selected_hard_feasible_rate"]
        >= thresholds["minimum_sharp_turn_selected_hard_feasible_rate"],
        "sharp_turn_selected_future_capable_rate": metrics["MODE:SHARP_TURN"]["selected_future_capable_rate"]
        >= thresholds["minimum_sharp_turn_selected_future_capable_rate"],
        "three_point_turn_selected_hard_feasible_rate": metrics["MODE:THREE_POINT_TURN"]["selected_hard_feasible_rate"]
        >= thresholds["minimum_three_point_turn_selected_hard_feasible_rate"],
        "three_point_turn_selected_future_capable_rate": metrics["MODE:THREE_POINT_TURN"]["selected_future_capable_rate"]
        >= thresholds["minimum_three_point_turn_selected_future_capable_rate"],
        "reverse_exit_selected_hard_feasible_rate": metrics["MODE:REVERSE_EXIT"]["selected_hard_feasible_rate"]
        >= thresholds["minimum_reverse_exit_selected_hard_feasible_rate"],
        "reverse_exit_selected_future_capable_rate": metrics["MODE:REVERSE_EXIT"]["selected_future_capable_rate"]
        >= thresholds["minimum_reverse_exit_selected_future_capable_rate"],
        "required_population_coverage": all(
            metrics["MODE:" + mode]["samples"] > 0 for mode in REQUIRED_MODES
        ) and metrics["GEAR:FORWARD"]["samples"] > 0
        and metrics["GEAR:REVERSE"]["samples"] > 0,
    }
    errors = sorted(name for name, passed in checks.items() if not passed)
    if not formal:
        errors.append("diagnostic_limited_score_audit")
    gate_passed = bool(formal and not errors)
    report = {
        "schema": "DEPCarRouteV2ScoreShadowAcceptanceV1",
        "status": "PASS" if gate_passed else "SMOKE" if not formal else "FAIL",
        "gate_passed": gate_passed,
        "scope": "P6_SHADOW_ONLY",
        "production_qualified": False,
        "active_control_authorized": False,
        "errors": errors,
        "risk_flags": [
            "REVERSE_EXIT_SELECTED_HARD_FEASIBLE_BELOW_0_95"
        ] if metrics["MODE:REVERSE_EXIT"]["selected_hard_feasible_rate"] < 0.95 else [],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_contract": str(contract_path),
        "checkpoint_contract_sha256": sha256_file(contract_path),
        "candidate_source_gate": payload["source_acceptance_gate"],
        "dataset_authority": str(authority_path),
        "dataset_authority_sha256": sha256_file(authority_path),
        "formal_training_authority_gate": authority_gate,
        "training_config": str(trainer.CONFIG.resolve()),
        "training_config_sha256": config_sha,
        "trainer_sha256": sha256_file(trainer.TRAINER),
        "acceptance_contract": str(CONTRACT.resolve()),
        "acceptance_contract_sha256": acceptance_sha,
        "acceptance_tool": str(Path(__file__).resolve()),
        "acceptance_tool_sha256": sha256_file(__file__),
        "population_frames_available": len(dataset),
        "population_frames_selected": len(indices),
        "geometry_valid_frames": metrics["OVERALL"]["samples"],
        "test_split_accessed": False,
        "thresholds": thresholds,
        "checks": {name: "PASS" if passed else "FAIL" for name, passed in checks.items()},
        "metrics": metrics,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
    output.write_text(serialized, encoding="utf-8")
    if formal:
        checkpoint.with_suffix(".score_shadow_acceptance.json").write_text(
            serialized, encoding="utf-8"
        )
    print(serialized, end="")
    return 0 if report["status"] in ("PASS", "SMOKE") else 2


if __name__ == "__main__":
    raise SystemExit(main())
