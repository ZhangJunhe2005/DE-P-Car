#!/usr/bin/env python3
"""Issue V4.2 calibrated execution authority after an independent full audit."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dep_car/src"))
sys.path.insert(0, str(ROOT / "tools"))

from dep_car.model.dep_car_net_v3 import DEPCarNetV3
from dep_car.model.dep_car_net_v4 import HybridSequenceConfigV4
from dep_car.model.dep_car_net_v42 import DEPCarNetV42
from dep_car.model.hybrid_sequence_rollout import HybridSequenceRolloutConfigV4
from dep_car.training.losses_v41 import DEPCarObjectiveV41
from dep_car.training.p4_dataset import p3_training_collate, p3_training_worker_init
from dep_car.training.v4_dataset import P3HybridSequenceDatasetV4
import train_dep_car_hybrid_sequence_v4 as v4
import train_dep_car_hybrid_sequence_v41 as v41


CONFIG = ROOT / "dep_car/config/p5_hybrid_sequence_v42_execution.yaml"
AUDITOR = Path(__file__).resolve()


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
    raw = CONFIG.read_bytes()
    config = yaml.safe_load(raw)
    if (
        config.get("schema") != "DEPCarV42ExecutionQualificationContractV1"
        or config.get("architecture_id") != DEPCarNetV42.architecture_id
        or config.get("source_architecture_id") != DEPCarNetV42.source_architecture_id
        or config.get("test_split_sealed") is not True
    ):
        raise RuntimeError("V4.2 execution contract identity differs")
    for name in ("adapter", "model", "rollout", "loss", "dataset"):
        path = resolve(config["implementation"][name])
        if (
            not path.is_file()
            or sha256_file(path) != config["implementation"][name + "_sha256"]
        ):
            raise RuntimeError("V4.2 %s implementation hash differs" % name)
    if (
        float(config["calibration_authority"]["safety_risk_weight"])
        != DEPCarNetV42.safety_risk_weight
        or float(config["calibration_authority"]["viability_risk_weight"])
        != DEPCarNetV42.viability_risk_weight
        or config["calibration_authority"]["hard_veto_mandatory"] is not True
        or DEPCarNetV42.requires_mandatory_hard_veto is not True
    ):
        raise RuntimeError("V4.2 adapter/calibration constants differ")
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
    return config, hashlib.sha256(raw).hexdigest(), model_config, rollout_config


def verify_authorities(config):
    _v41_config, _sha, _model, _rollout, loss_config = v41.load_config()
    _bundle_path, bundle, sequence_path, _authority, data_gate = v41.verify_data_authority(_v41_config)
    source = config["source"]
    checkpoint = resolve(source["checkpoint"])
    contract_path = resolve(source["checkpoint_contract"])
    legacy_path = resolve(source["legacy_acceptance"])
    pareto_path = resolve(config["calibration_authority"]["pareto_audit"])
    paths = (
        (checkpoint, source["checkpoint_sha256"]),
        (contract_path, source["checkpoint_contract_sha256"]),
        (legacy_path, source["legacy_acceptance_sha256"]),
        (pareto_path, config["calibration_authority"]["pareto_audit_sha256"]),
    )
    errors = [path.name for path, digest in paths if not path.is_file() or sha256_file(path) != digest]
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    contract = read_json(contract_path)
    legacy = read_json(legacy_path)
    pareto = read_json(pareto_path)
    if (
        payload.get("schema") != v41.CHECKPOINT_SCHEMA
        or payload.get("architecture_id") != DEPCarNetV42.source_architecture_id
        or payload.get("run_completed") is not True
        or payload.get("partial_epoch") is not False
        or contract.get("checkpoint_sha256") != sha256_file(checkpoint)
        or legacy.get("status") != "FAIL"
        or sorted(legacy.get("errors", ()))
        != sorted(source["expected_legacy_failed_checks"])
        or legacy.get("checkpoint_sha256") != sha256_file(checkpoint)
        or pareto.get("schema") != "DEPCarV41ExecutionParetoAuditV1"
        or pareto.get("status") != "PASS"
        or pareto.get("checkpoint_sha256") != sha256_file(checkpoint)
        or pareto.get("hard_veto_is_mandatory_execution_semantics") is not True
        or pareto.get("test_split_accessed") is not False
    ):
        errors.append("source_or_pareto_identity")
    best = pareto.get("best_joint_calibration", {})
    if (
        best.get("safety_weight")
        != float(config["calibration_authority"]["safety_risk_weight"])
        or best.get("viability_weight")
        != float(config["calibration_authority"]["viability_risk_weight"])
    ):
        errors.append("pareto_calibration")
    if errors:
        raise RuntimeError("V4.2 authority failed: " + ",".join(sorted(set(errors))))
    source_gate = {
        "schema": "DEPCarV42SourceGateV1", "passed": True,
        "checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_contract": str(contract_path),
        "checkpoint_contract_sha256": sha256_file(contract_path),
        "legacy_acceptance": str(legacy_path),
        "legacy_acceptance_sha256": sha256_file(legacy_path),
        "legacy_failed_checks": sorted(legacy.get("errors", ())),
        "pareto_audit": str(pareto_path), "pareto_audit_sha256": sha256_file(pareto_path),
        "mandatory_hard_veto": True, "test_split_accessed": False,
    }
    return bundle, sequence_path, data_gate, checkpoint, payload, loss_config, source_gate


def execution_metrics(scores, hard, viable, costs):
    any_hard = hard.any(axis=1)
    any_viable = viable.any(axis=1)
    preferred = np.where(any_viable[:, None], viable, hard)
    effective = np.where(hard, scores, np.inf)
    effective[~any_hard] = scores[~any_hard]
    selected = effective.argmin(axis=1)
    row = np.arange(len(scores))
    selected_hard = hard[row, selected]
    selected_viable = viable[row, selected]
    selected_preferred = preferred[row, selected] & any_hard
    oracle = np.where(preferred, costs, np.inf).min(axis=1)
    chosen = costs[row, selected]
    regret_rows = selected_preferred
    regret = np.maximum(0.0, chosen[regret_rows] - oracle[regret_rows])
    hard_oracle = np.where(hard, costs, np.inf).min(axis=1)
    legacy = np.maximum(0.0, chosen[any_hard] - hard_oracle[any_hard])
    return {
        "samples": int(len(scores)),
        "hard_veto_stop_rate": float((~any_hard).mean()),
        "selected_hard_feasible_rate": float(selected_hard.mean()),
        "selected_viable_rate": float(selected_viable.mean()),
        "hierarchy_selection_rate_when_executable": float(
            selected_preferred.sum() / max(1, int(any_hard.sum()))
        ),
        "hierarchical_oracle_regret": float(regret.mean()) if len(regret) else 0.0,
        "legacy_cross_tier_oracle_regret": float(legacy.mean()) if len(legacy) else 0.0,
        "any_viable_rate": float(any_viable.mean()),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.workers < 1:
        raise SystemExit("V4.2 workers must be positive")
    config, config_sha, model_config, rollout_config = load_contract()
    bundle, sequence_path, data_gate, checkpoint, payload, loss_config, source_gate = verify_authorities(config)
    plan = {
        "schema": "DEPCarV42ExecutionAuditPlanV1",
        "status": "DRY_RUN_READY" if args.dry_run else "READY",
        "architecture_id": DEPCarNetV42.architecture_id,
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_sha256": sha256_file(checkpoint),
        "training_required": False, "mandatory_hard_veto": True,
        "safety_risk_weight": DEPCarNetV42.safety_risk_weight,
        "viability_risk_weight": DEPCarNetV42.viability_risk_weight,
        "data_authority_gate": data_gate, "source_gate": source_gate,
        "config": str(CONFIG), "config_sha256": config_sha,
        "auditor_sha256": sha256_file(AUDITOR),
        "test_split_sealed": True, "active_control_authorized": False,
        "production_qualified": False,
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True)); return 0
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    model = DEPCarNetV42(
        base_model=DEPCarNetV3(), sequence_config=model_config,
        rollout_config=rollout_config,
    )
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.freeze_base(); model.to(device).eval()
    dataset = P3HybridSequenceDatasetV4(
        sample_root=bundle["sample_root"], maps_root=bundle["maps_root"],
        split="validation", index_path=bundle["index"],
        index_splits=("train", "validation"), workers=args.workers,
        expected_map_contract_aggregate_sha256=bundle["map_contract_aggregate_sha256"],
        expected_index_sha256=bundle["index_sha256"], modality="fusion",
        sequence_index_path=sequence_path,
    )
    loader = DataLoader(
        dataset, batch_size=int(config["audit"]["batch_size"]), shuffle=False,
        num_workers=args.workers, pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
        prefetch_factor=int(config["audit"]["prefetch_factor"]),
        collate_fn=p3_training_collate, worker_init_fn=p3_training_worker_init,
    )
    objective = DEPCarObjectiveV41(loss_config)
    collected = {key: [] for key in ("score", "hard", "viable", "cost")}
    with torch.inference_mode():
        for host in loader:
            batch = v4.select_valid(host, device)
            if batch is None:
                continue
            output, losses = v4.forward_loss(
                model, objective, batch, v41.STAGE,
                bool(config["audit"]["mixed_precision"]) and device.type == "cuda",
            )
            values = {
                "score": output.scores, "hard": losses["hard_feasible"],
                "viable": losses["viable"], "cost": losses["candidate_cost"],
            }
            for key, value in values.items():
                collected[key].append(value.detach().cpu().numpy())
    value = {key: np.concatenate(rows, axis=0) for key, rows in collected.items()}
    metrics = execution_metrics(
        value["score"], value["hard"], value["viable"], value["cost"]
    )
    qualification = config["qualification"]
    raw_checks = {
        "hard_veto_stop_rate": metrics["hard_veto_stop_rate"]
        <= float(qualification["maximum_hard_veto_stop_rate"]),
        "selected_hard_feasible_rate": metrics["selected_hard_feasible_rate"]
        >= float(qualification["minimum_selected_hard_feasible_rate"]),
        "selected_viable_rate": metrics["selected_viable_rate"]
        >= float(qualification["minimum_selected_viable_rate"]),
        "hierarchy_selection_rate_when_executable": metrics["hierarchy_selection_rate_when_executable"]
        >= float(qualification["minimum_hierarchy_selection_rate_when_executable"]),
        "hierarchical_oracle_regret": metrics["hierarchical_oracle_regret"]
        <= float(qualification["maximum_hierarchical_oracle_regret"]),
    }
    passed = all(raw_checks.values())
    report = {
        **plan, "schema": "DEPCarV42ExecutionAcceptanceV1",
        "status": "PASS" if passed else "FAIL", "gate_passed": passed,
        "errors": [] if passed else sorted(
            key for key, value in raw_checks.items() if not value
        ),
        "metrics": metrics,
        "checks": {key: "PASS" if value else "FAIL" for key, value in raw_checks.items()},
        "legacy_cross_tier_oracle_regret_is_gate": False,
        "p6_shadow_authorized": bool(passed),
        "active_control_authorized": False, "production_qualified": False,
        "test_split_accessed": False,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    report_path = resolve(config["artifacts"]["report"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    authority = {
        "schema": "DEPCarV42ExecutionAuthorityV1",
        "status": "P6_SHADOW_AUTHORIZED" if passed else "BLOCKED",
        "architecture_id": DEPCarNetV42.architecture_id,
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_sha256": sha256_file(checkpoint),
        "adapter": str(resolve(config["implementation"]["adapter"])),
        "adapter_sha256": config["implementation"]["adapter_sha256"],
        "mandatory_hard_veto": True,
        "safety_risk_weight": DEPCarNetV42.safety_risk_weight,
        "viability_risk_weight": DEPCarNetV42.viability_risk_weight,
        "acceptance_report": str(report_path),
        "acceptance_report_sha256": sha256_file(report_path),
        "p6_shadow_authorized": bool(passed),
        "active_control_authorized": False, "production_qualified": False,
        "test_split_accessed": False,
    }
    authority_path = resolve(config["artifacts"]["authority"])
    authority_path.parent.mkdir(parents=True, exist_ok=True)
    authority_path.write_text(
        json.dumps(authority, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
