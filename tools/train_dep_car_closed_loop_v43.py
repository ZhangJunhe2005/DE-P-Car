#!/usr/bin/env python3
"""Train V4.3 on re-observed closed-loop DAgger sequences."""

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
sys.path.insert(0, str(ROOT / "dep_car/src")); sys.path.insert(0, str(ROOT / "tools"))
from dep_car.model.dep_car_net_v3 import DEPCarNetV3
from dep_car.model.dep_car_net_v4 import DEPCarNetV4, HybridSequenceConfigV4
from dep_car.model.dep_car_net_v43 import DEPCarNetV43
from dep_car.model.hybrid_sequence_rollout import HybridSequenceRolloutConfigV4
from dep_car.training.losses_v4 import DEPCarHybridSequenceLossConfigV4
from dep_car.training.losses_v43 import DEPCarClosedLoopLossConfigV43, DEPCarObjectiveV43
from dep_car.training.p4_dataset import p3_training_collate, p3_training_worker_init
from dep_car.training.v43_dataset import P3ClosedLoopSequenceDatasetV43
import train_dep_car_hybrid_sequence_v4 as v4


CONFIG = ROOT / "dep_car/config/p5_closed_loop_v43.yaml"
TRAINER = Path(__file__).resolve()
CHECKPOINT_SCHEMA = "DEPCarV43CheckpointV5"
CONTRACT_SCHEMA = "DEPCarV43ArtifactContractV5"
STAGES = DEPCarObjectiveV43.stages
STAGE = DEPCarObjectiveV43.selector_stage
DATA_AUTHORITY_SCHEMA = "DEPCarV43ClosedLoopDataAuthorityV2"
SEQUENCE_AUTHORITY = "REOBSERVED_STATE_EXACT_SIGNED_HYBRID_ASTAR_PLAN"


def resolve(value):
    path = Path(value)
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
        raise RuntimeError("expected JSON object: " + str(path))
    return value


def best_path(path):
    return path.with_name(path.stem + ".best.pth")


def load_config():
    raw = CONFIG.read_bytes(); config = yaml.safe_load(raw)
    if (
        config.get("schema") != "DEPCarV43ClosedLoopTrainingContractV4"
        or config.get("architecture_id") != DEPCarNetV43.architecture_id
        or config.get("objective_id") != DEPCarObjectiveV43.objective_id
        or config.get("test_split_sealed") is not True
    ):
        raise RuntimeError("V4.3 training contract identity differs")
    for name in (
        "model", "candidate_model", "rollout", "loss", "dataset",
        "score_dataset",
    ):
        path = resolve(config["implementation"][name])
        if not path.is_file() or sha256_file(path) != config["implementation"][name + "_sha256"]:
            raise RuntimeError("V4.3 implementation hash differs: " + name)
    model_config = HybridSequenceConfigV4(**{
        key: config["model"][key] for key in (
            "candidates", "actions", "hidden_dim", "primitive_dim",
            "stage_dim", "template_logit_bias",
        )
    })
    rollout_config = HybridSequenceRolloutConfigV4(
        candidates=config["model"]["candidates"], actions=config["model"]["actions"],
        steps_per_action=config["model"]["steps_per_action"],
    )
    base_loss = DEPCarHybridSequenceLossConfigV4(**config["base_loss"])
    loss_config = DEPCarClosedLoopLossConfigV43(base=base_loss, **config["closed_loop_loss"])
    model_config.validate(); rollout_config.validate(); loss_config.validate()
    return config, hashlib.sha256(raw).hexdigest(), model_config, rollout_config, loss_config


def verify_data(config):
    path = resolve(config["dataset"]["authority"]); authority = read_json(path)
    integrity_path = resolve(config["dataset"]["integrity_audit"])
    integrity = read_json(integrity_path)
    errors = []
    if (
        sha256_file(path) != config["dataset"]["authority_sha256"]
        or authority.get("schema") != DATA_AUTHORITY_SCHEMA
        or authority.get("status") != "PASS" or authority.get("errors") != []
        or authority.get("continuous_sequence_authority") != SEQUENCE_AUTHORITY
        or authority.get("runtime_hybrid_astar_dependency") is not False
        or authority.get("runtime_ground_truth_input") is not False
        or authority.get("test_split_opened") is not False
    ):
        errors.append("closed_loop_data_authority")
    if (
        sha256_file(integrity_path) != config["dataset"]["integrity_audit_sha256"]
        or integrity.get("schema") != "DEPCarV43IndependentIntegrityAuditV1"
        or integrity.get("status") != "PASS"
        or integrity.get("errors") != []
        or integrity.get("authority_file_sha256") != sha256_file(path)
        or integrity.get("test_split_opened") is not False
    ):
        errors.append("independent_integrity_audit")
    for key, authority_key in (
        ("training_index", "training_index_sha256"),
        ("sequence_index", "sequence_index_sha256"),
    ):
        artifact = resolve(authority[key])
        if not artifact.is_file() or sha256_file(artifact) != authority[authority_key]:
            errors.append(key)
    if errors:
        raise RuntimeError("V4.3 data gate failed: " + ",".join(errors))
    return authority, {
        "schema": "DEPCarV43DataGateV1", "passed": True,
        "authority": str(path), "authority_sha256": sha256_file(path),
        "samples": authority["samples"], "episodes": authority["episodes"],
        "continuous_sequence_authority": SEQUENCE_AUTHORITY,
        "integrity_audit": str(integrity_path),
        "integrity_audit_sha256": sha256_file(integrity_path),
        "test_split_sealed": True,
    }


def verify_source(config):
    source = config["source"]
    checkpoint = resolve(source["checkpoint"]); contract_path = resolve(source["checkpoint_contract"])
    guarded_path = resolve(source["guarded_authority"])
    capacity_path = resolve(source["capacity_diagnostic"])
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    contract = read_json(contract_path); guarded = read_json(guarded_path)
    capacity = read_json(capacity_path)
    if (
        sha256_file(checkpoint) != source["checkpoint_sha256"]
        or sha256_file(contract_path) != source["checkpoint_contract_sha256"]
        or sha256_file(guarded_path) != source["guarded_authority_sha256"]
        or payload.get("architecture_id") != DEPCarNetV4.architecture_id
        or payload.get("run_completed") is not True
        or contract.get("checkpoint_sha256") != sha256_file(checkpoint)
        or guarded.get("status") != "P6_GUARDED_SIMULATION_AUTHORIZED"
        or guarded.get("source_checkpoint_sha256") != sha256_file(checkpoint)
        or guarded.get("test_split_accessed") is not False
        or sha256_file(capacity_path) != source["capacity_diagnostic_sha256"]
        or capacity.get("schema") != "DEPCarV43CapacityDiagnosticV1"
        or capacity.get("status") != "PASS"
        or capacity.get("checkpoint_sha256") != sha256_file(checkpoint)
        or capacity.get("data_authority_gate", {}).get("authority_sha256")
        != sha256_file(resolve(config["dataset"]["authority"]))
        or capacity.get("test_split_accessed") is not False
    ):
        raise RuntimeError("V4.3 V4.2 guarded source gate failed")
    return checkpoint, payload, {
        "schema": "DEPCarV43SourceGateV1", "passed": True,
        "checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_contract": str(contract_path),
        "checkpoint_contract_sha256": sha256_file(contract_path),
        "guarded_authority": str(guarded_path),
        "guarded_authority_sha256": sha256_file(guarded_path),
        "capacity_diagnostic": str(capacity_path),
        "capacity_diagnostic_sha256": sha256_file(capacity_path),
        "candidate_decoder_role": "FROZEN_ACCEPTED_UNIFIED_SEQUENCE_CAPACITY",
        "source_role": "V42_GUARDED_FAILURE_POLICY_INITIALIZATION",
        "test_split_accessed": False,
    }


def configure_model(model, stage):
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if stage == DEPCarObjectiveV43.selector_stage:
        modules = (model.closed_loop_score_adapter,)
    else:
        raise ValueError("unknown V4.3 parameter stage: " + str(stage))
    selected = tuple(parameter for module in modules for parameter in module.parameters())
    for parameter in selected:
        parameter.requires_grad_(True)
    if any(parameter.requires_grad for parameter in model.base_model.parameters()):
        raise RuntimeError("V4.3 perception/local-geometry base gained training authority")
    return selected, {
        "frozen_perception_and_local_geometry": True,
        "hybrid_sequence_decoder_trainable": False,
        "candidate_geometry_trainable": False,
        "gear_sequence_trainable": False,
        "navigation_heads_trainable": False,
        "source_sequence_score_head_trainable": False,
        "closed_loop_residual_score_trainable": True,
        "high_level_gear_state_machine": False,
        "stage": stage,
        "trainable_parameters": sum(parameter.numel() for parameter in selected),
    }


def checks(metrics, qualification, stage=STAGE):
    result = {
        "initial_pose_hard_unsafe_rate_disclosed": metrics["initial_pose_hard_unsafe_rate"] <= float(qualification["maximum_initial_pose_hard_unsafe_rate"]),
        "zero_hard_feasible_rate_given_safe_initial_pose": metrics["zero_hard_feasible_rate_given_safe_initial_pose"] <= float(qualification["maximum_zero_hard_feasible_rate_given_safe_initial_pose"]),
        "unsafe_initial_egress_candidate_rate": metrics["unsafe_initial_egress_candidate_rate"] >= float(qualification["minimum_unsafe_initial_egress_candidate_rate"]),
        "best_of_15_gear_error_rate": metrics["best_of_15_gear_error_rate"] <= float(qualification["maximum_best_of_15_gear_error_rate"]),
        "multiaction_prefix_coverage": metrics["multiaction_prefix_coverage"] >= float(qualification["minimum_multiaction_prefix_coverage"]),
        "selected_hard_feasible_rate_when_available": metrics["selected_hard_feasible_rate_when_available"] >= float(qualification["minimum_selected_hard_feasible_rate_when_available"]),
        "selected_egress_rate_when_available": metrics["selected_egress_rate_when_available"] >= float(qualification["minimum_selected_egress_rate_when_available"]),
        "selected_navigation_eligible_rate_when_available": metrics["selected_navigation_eligible_rate_when_available"] >= float(qualification["minimum_selected_navigation_eligible_rate_when_available"]),
    }
    result.update({
        "plan_gear_prefix_error_rate": metrics["plan_gear_prefix_error_rate"] <= float(qualification["maximum_plan_gear_prefix_error_rate"]),
        "reverse_then_forward_coverage": metrics["reverse_then_forward_coverage"] >= float(qualification["minimum_reverse_then_forward_coverage"]),
        "mandatory_guard_stop_rate_when_no_candidate": metrics["mandatory_guard_stop_rate_when_no_candidate"] >= float(qualification["minimum_mandatory_guard_stop_rate_when_no_candidate"]),
        "selected_exact_sequence_rate_when_available": metrics["selected_exact_sequence_rate_when_available"] >= float(qualification["minimum_selected_exact_sequence_rate_when_available"]),
    })
    return result


def selection_key(metrics, qualification, stage=STAGE):
    result = checks(metrics, qualification, stage)
    common = (
        sum(not value for value in result.values()),
        -metrics["selected_exact_sequence_rate_when_available"],
        metrics["plan_gear_prefix_error_rate"],
        -metrics["reverse_then_forward_coverage"],
        -metrics["selected_navigation_eligible_rate_when_available"],
        -metrics["selected_egress_rate_when_available"],
        -metrics["selected_hard_feasible_rate_when_available"],
    )
    return common + (
        -metrics["selected_viable_rate_when_available"], metrics["mean_loss"],
    )


def atomic_torch(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp"); torch.save(payload, temporary); os.replace(temporary, path)


def write_artifact(path, model, optimizer, scaler, metadata, metrics, history):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": CHECKPOINT_SCHEMA, "architecture_id": DEPCarNetV43.architecture_id,
        "objective_id": DEPCarObjectiveV43.objective_id,
        "training_stage": metadata["stage"],
        "artifact_role": metadata["artifact_role"], "status": "TRAINED_UNQUALIFIED",
        "qualification_status": "UNQUALIFIED", "run_completed": False,
        "partial_epoch": False, "completed_epochs": metadata["completed_epochs"],
        "selected_epoch": metadata["selected_epoch"], "global_step": metadata["global_step"],
        "active_control_authorized": False, "production_qualified": False,
        "source_gate": metadata["source_gate"], "data_authority_gate": metadata["data_gate"],
        "training_config_sha256": metadata["config_sha"],
        "trainer_sha256": sha256_file(TRAINER),
        "implementation_sha256": metadata["implementation_sha256"],
        "continuous_sequence_authority": SEQUENCE_AUTHORITY,
        "high_level_gear_state_machine": False, "stage_ownership": metadata["ownership"],
        "phase_source_gate": metadata.get("phase_source_gate"),
        "metrics": metrics, "selection_gate": metadata["selection_gate"], "history": history,
        "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
        "grad_scaler_state_dict": scaler.state_dict(),
    }
    atomic_torch(path, payload)
    contract = {key: value for key, value in payload.items() if key not in (
        "model_state_dict", "optimizer_state_dict", "grad_scaler_state_dict", "history"
    )}
    contract.update({"schema": CONTRACT_SCHEMA, "checkpoint": str(path), "checkpoint_sha256": sha256_file(path)})
    path.with_suffix(".contract.json").write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def finalize(path, epochs, global_step, history):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload.update({"completed_epochs": epochs, "global_step": global_step, "run_completed": True, "history": history})
    atomic_torch(path, payload)
    contract_path = path.with_suffix(".contract.json"); contract = read_json(contract_path)
    contract.update({"checkpoint_sha256": sha256_file(path), "completed_epochs": epochs, "global_step": global_step, "run_completed": True})
    contract_path.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--workers", type=int); parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-steps", type=int); parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    config, config_sha, model_config, rollout_config, loss_config = load_config()
    training = config["training"]
    epochs = int(training["epochs"] if args.epochs is None else args.epochs)
    batch_size = int(training["batch_size"] if args.batch_size is None else args.batch_size)
    workers = int(training["workers"] if args.workers is None else args.workers)
    if min(epochs, batch_size, workers) < 1:
        raise SystemExit("V4.3 training sizes must be positive")
    if args.max_samples is not None and not 1 <= args.max_samples <= 2048: raise SystemExit("max-samples must be [1,2048]")
    if args.max_steps is not None and not 1 <= args.max_steps <= 64: raise SystemExit("max-steps must be [1,64]")
    authority, data_gate = verify_data(config); source_path, source, source_gate = verify_source(config)
    if resolve(args.source) != source_path: raise RuntimeError("V4.3 source differs from contract")
    output = resolve(args.output); expected = resolve(config["artifacts"]["output"])
    bounded = args.max_samples is not None or args.max_steps is not None
    formal = bool(
        not bounded
        and epochs == int(training["epochs"])
        and batch_size == int(training["batch_size"])
        and workers == int(training["workers"])
        and args.device == "cuda" and output == expected
    )
    if not args.dry_run and not formal and not bounded: raise RuntimeError("V4.3 run is neither formal nor bounded")
    if not args.dry_run and output.exists():
        raise RuntimeError("V4.3 output exists; preserve it before rerun")
    plan = {
        "schema": "DEPCarV43TrainingPlanV4", "status": "DRY_RUN_READY" if args.dry_run and formal else "BOUNDED_DIAGNOSTIC_READY" if bounded else "READY",
        "stages": list(STAGES), "architecture_id": DEPCarNetV43.architecture_id,
        "objective_id": DEPCarObjectiveV43.objective_id, "source": str(source_path),
        "source_sha256": sha256_file(source_path), "output": str(output),
        "epochs": epochs,
        "batch_size": batch_size, "workers": workers, "device": args.device,
        "bounded_smoke": bounded, "formal_training_authorized": formal,
        "data_authority_gate": data_gate, "source_gate": source_gate,
        "training_config": str(CONFIG), "training_config_sha256": config_sha,
        "trainer_sha256": sha256_file(TRAINER), "test_split_sealed": True,
        "active_control_authorized": False, "production_qualified": False,
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True)); return 0
    if args.device == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA is unavailable")
    v4.seed_all(int(training["seed"])); torch.set_num_threads(int(training["torch_threads"]))
    device = torch.device(args.device)
    if device.type == "cuda": torch.backends.cudnn.benchmark = True; torch.set_float32_matmul_precision("high")
    model = DEPCarNetV43(
        base_model=DEPCarNetV3(), sequence_config=model_config,
        rollout_config=rollout_config,
        residual_score_span=float(config["model"]["residual_score_span"]),
    )
    model.initialize_from_v4(source["model_state_dict"])
    model.freeze_base(); model.to(device)
    amp = bool(training["mixed_precision"]) and device.type == "cuda"
    common = dict(
        sample_root=authority["sample_root"], maps_root=authority["maps_root"],
        index_path=authority["training_index"], index_splits=("train", "validation"),
        workers=workers, expected_map_contract_aggregate_sha256=authority["map_contract_aggregate_sha256"],
        expected_index_sha256=authority["training_index_sha256"], modality="fusion",
        sequence_index_path=authority["sequence_index"],
        expected_sequence_index_sha256=authority["sequence_index_sha256"],
    )
    train_data = P3ClosedLoopSequenceDatasetV43(split="train", **common)
    validation_data = P3ClosedLoopSequenceDatasetV43(split="validation", **common)
    if args.max_samples:
        train_data = Subset(train_data, np.linspace(0, len(train_data)-1, min(len(train_data), args.max_samples), dtype=np.int64).tolist())
        validation_data = Subset(validation_data, np.linspace(0, len(validation_data)-1, min(len(validation_data), args.max_samples), dtype=np.int64).tolist())
    loader_args = dict(batch_size=batch_size, num_workers=workers, pin_memory=device.type == "cuda", persistent_workers=workers > 0, collate_fn=p3_training_collate, worker_init_fn=p3_training_worker_init)
    if workers > 0: loader_args["prefetch_factor"] = int(training["prefetch_factor"])
    train_loader = DataLoader(train_data, shuffle=True, **loader_args)
    validation_loader = DataLoader(validation_data, shuffle=False, **loader_args)
    objective = DEPCarObjectiveV43(loss_config)
    history = []; global_step = 0
    implementation_sha256 = {
        key: config["implementation"][key + "_sha256"]
        for key in (
            "model", "candidate_model", "rollout", "loss", "dataset",
            "score_dataset",
        )
    }
    ownership_by_stage = {}
    phase_outputs = ((DEPCarObjectiveV43.selector_stage, epochs, output),)
    for stage, phase_epochs, phase_output in phase_outputs:
        selected, ownership = configure_model(model, stage)
        ownership_by_stage[stage] = ownership
        optimizer = torch.optim.AdamW(
            selected, lr=float(training["learning_rate"]),
            weight_decay=float(training["weight_decay"]),
        )
        scaler = torch.amp.GradScaler(device.type, enabled=amp)
        best_key = None
        phase_history = []
        for epoch in range(1, phase_epochs + 1):
            started = time.time()
            train_metrics = v4.epoch_loop(
                model, objective, train_loader, stage, device, amp,
                optimizer=optimizer, scaler=scaler, max_steps=args.max_steps,
                dropout=float(training["sensor_dropout_probability"]),
                progress_interval=int(training["progress_interval_steps"]),
                gradient_clip=float(training["gradient_clip"]),
            )
            validation_metrics = v4.epoch_loop(
                model, objective, validation_loader, stage, device, amp,
                max_steps=args.max_steps,
                progress_interval=int(training["progress_interval_steps"]),
            )
            global_step += train_metrics["steps"]
            result = checks(validation_metrics, config["qualification"], stage)
            gate = {key: "PASS" if value else "FAIL" for key, value in result.items()}
            row = {
                "stage": stage, "epoch": epoch, "train": train_metrics,
                "validation": validation_metrics, "selection_gate": gate,
                "elapsed_s": time.time() - started,
            }
            history.append(row); phase_history.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
            metadata = {
                "stage": stage, "artifact_role": "last",
                "completed_epochs": epoch, "selected_epoch": epoch,
                "global_step": global_step, "source_gate": source_gate,
                "phase_source_gate": source_gate,
                "data_gate": data_gate, "config_sha": config_sha,
                "implementation_sha256": implementation_sha256,
                "selection_gate": gate, "ownership": ownership,
            }
            write_artifact(
                phase_output, model, optimizer, scaler, metadata,
                validation_metrics, history,
            )
            key = selection_key(validation_metrics, config["qualification"], stage)
            if best_key is None or key < best_key:
                best_key = key; metadata["artifact_role"] = "best"
                write_artifact(
                    best_path(phase_output), model, optimizer, scaler, metadata,
                    validation_metrics, history,
                )
        phase_best = best_path(phase_output)
        finalize(phase_best, phase_epochs, global_step, history)
        phase_payload = torch.load(phase_best, map_location="cpu", weights_only=True)
        model.load_state_dict(phase_payload["model_state_dict"], strict=True)
        model.to(device)
    print(json.dumps({
        **plan, "status": "COMPLETE", "global_step": global_step,
        "best_checkpoint": str(best_path(output)),
        "stage_ownership": ownership_by_stage,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
