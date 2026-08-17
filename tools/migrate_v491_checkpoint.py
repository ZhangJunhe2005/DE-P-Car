#!/usr/bin/env python3
"""Build the formal P4 initialization from the frozen DE-P depth backbone.

V4.9.1 did not train new policy weights; its runtime release uses the frozen
V4.8.3 checkpoint below.  Only the one-channel MobileNetV3 depth backbone has
the same physical meaning in :class:`DEPCarNetV1`.  UAV state/head tensors are
therefore outside the allowlist even when a tensor happens to have a compatible
shape.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
DEP_SOURCE_TREE = ROOT / "third_party" / "DE-P" / "DE-P"
DEFAULT_SOURCE = (
    DEP_SOURCE_TREE
    / "runs"
    / "route_a_static_yopo_v4_8_3_candidate_only_shakedown"
    / "20260813T050742Z-13178"
    / "checkpoints"
    / "best.pth"
)
DEFAULT_OUTPUT = ROOT / "models" / "dep_car" / "dep_car_net_v1_depth_v483_init.pth"
P3_ACCEPTANCE = ROOT / "reports" / "p3_pilot_acceptance.json"

SOURCE_CHECKPOINT_SHA256 = (
    "22e5c63c273d751c15479d70c99d9b85ad615b7b4c62063946a5b1683776ac60"
)
SOURCE_CHECKPOINT_VERSION = "mixed_scene_static_yopo_checkpoint_v1"
SOURCE_PREFIX = "network.image_backbone."
TARGET_PREFIX = "depth_encoder.image_backbone."
EXPECTED_TRANSFER_TENSORS = 246
INITIALIZATION_SEED = 49101
CHECKPOINT_VERSION = "dep_car_p4_depth_transfer_initialization_v1"
CONTRACT_SCHEMA = "DEPCarCheckpointContractV2"
STATUS = "INITIALIZATION_ONLY_RETRAINING_REQUIRED"

BACKBONE_SOURCE_FILES = (
    "policy/models/backbone.py",
    "policy/models/MobileNetV3.py",
    "policy/backbone_variant.py",
)

sys.path.insert(0, str(ROOT / "dep_car" / "src"))
from dep_car.core.occupancy import (
    FIVE_CIRCLE_FOOTPRINT_SCHEMA,
    FOOTPRINT_ALLOWANCE_SCHEMA,
    RUNTIME_HALF_DIAGONAL_MULTIPLIER,
    SWEPT_INTERPOLATION_SCHEMA,
    SWEPT_SUBSTEPS_PER_SEGMENT,
    TRAINING_ONE_DIAGONAL_MULTIPLIER,
    FootprintConfig,
)
from dep_car.core.state_contract import STATE_NORMALIZATION_SCALE
from dep_car.model.ackermann_rollout import (
    ACKERMANN_ROLLOUT_SCHEMA,
    LONGITUDINAL_LIMITS_FRAME,
    AckermannRolloutV1,
)
from dep_car.model.dep_car_net import DEPCarNetV1
from dep_car.model.implementation_contract import build_p4_implementation_contract
from dep_car.perception.bev import BEV_CHANNELS
from dep_car.training.losses import DEPCarLossConfig, DEPCarObjectiveV1
from dep_car.training.p4_dataset import SIGNED_SDF_SCHEMA


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    contiguous = value.detach().cpu().contiguous()
    return hashlib.sha256(contiguous.numpy().tobytes(order="C")).hexdigest()


def _repository_path(path: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(ROOT.resolve()))
    except ValueError:
        return str(resolved)


def load_authoritative_source(source_path: Path):
    """Validate identity before allowing unrestricted formal-checkpoint load."""

    source_path = Path(source_path).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"V4.8.3 source checkpoint is unavailable: {source_path}")
    # Pin one immutable byte snapshot before invoking pickle.  Hashing the path
    # and then reopening it for deserialization leaves a TOCTOU window in which
    # an unverified replacement could be executed while the old digest is
    # recorded in the migration provenance.
    source_bytes = source_path.read_bytes()
    actual_sha256 = hashlib.sha256(source_bytes).hexdigest()
    if actual_sha256 != SOURCE_CHECKPOINT_SHA256:
        raise ValueError(
            "source checkpoint SHA-256 mismatch: the P4 initializer only accepts "
            "the frozen V4.8.3 policy used by the V4.9.1 runtime"
        )

    # This checkpoint contains optimizer/RNG objects that weights_only cannot
    # decode.  The byte identity is checked above before pickle is opened.
    payload = torch.load(
        io.BytesIO(source_bytes), map_location="cpu", weights_only=False
    )
    if not isinstance(payload, Mapping):
        raise ValueError("source checkpoint payload must be a mapping")
    if payload.get("checkpoint_version") != SOURCE_CHECKPOINT_VERSION:
        raise ValueError("source checkpoint version is not the frozen formal policy")
    state = payload.get("model")
    if not isinstance(state, Mapping):
        raise ValueError("source checkpoint is missing its formal model state")
    if not all(isinstance(name, str) and torch.is_tensor(value) for name, value in state.items()):
        raise ValueError("source model state contains non-tensor entries")
    return payload, state, actual_sha256


def exact_backbone_transfer(source_state, target_state):
    """Copy the complete depth backbone under an exact, prefix-only allowlist."""

    source_backbone = {
        name: value for name, value in source_state.items()
        if name.startswith(SOURCE_PREFIX)
    }
    target_backbone = {
        name: value for name, value in target_state.items()
        if name.startswith(TARGET_PREFIX)
    }
    if len(source_backbone) != EXPECTED_TRANSFER_TENSORS:
        raise ValueError(
            f"source depth backbone tensor count is {len(source_backbone)}, "
            f"expected {EXPECTED_TRANSFER_TENSORS}"
        )
    if len(target_backbone) != EXPECTED_TRANSFER_TENSORS:
        raise ValueError(
            f"target depth backbone tensor count is {len(target_backbone)}, "
            f"expected {EXPECTED_TRANSFER_TENSORS}"
        )

    expected_targets = {
        TARGET_PREFIX + name[len(SOURCE_PREFIX):] for name in source_backbone
    }
    if expected_targets != set(target_backbone):
        missing = sorted(expected_targets.difference(target_backbone))
        unexpected = sorted(set(target_backbone).difference(expected_targets))
        raise ValueError(
            "depth backbone key mismatch; "
            f"missing={missing[:3]}, unexpected={unexpected[:3]}"
        )

    transfers = []
    with torch.no_grad():
        for source_name in sorted(source_backbone):
            source_value = source_backbone[source_name]
            target_name = TARGET_PREFIX + source_name[len(SOURCE_PREFIX):]
            target_value = target_state[target_name]
            if source_value.shape != target_value.shape:
                raise ValueError(
                    f"exact transfer shape mismatch for {source_name}: "
                    f"source={tuple(source_value.shape)}, target={tuple(target_value.shape)}"
                )
            if source_value.dtype != target_value.dtype:
                raise ValueError(
                    f"exact transfer dtype mismatch for {source_name}: "
                    f"source={source_value.dtype}, target={target_value.dtype}"
                )
            target_value.copy_(source_value)
            transfers.append({
                "source": source_name,
                "target": target_name,
                "shape": list(source_value.shape),
                "dtype": str(source_value.dtype).removeprefix("torch."),
                "tensor_sha256": tensor_sha256(source_value),
                "mode": "exact",
            })

    if len(transfers) != EXPECTED_TRANSFER_TENSORS:
        raise RuntimeError("formal migration did not transfer the complete backbone")
    if any(
        row["mode"] != "exact"
        or not row["source"].startswith(SOURCE_PREFIX)
        or not row["target"].startswith(TARGET_PREFIX)
        for row in transfers
    ):
        raise RuntimeError("partial or non-backbone migration is forbidden")
    return transfers


def backbone_source_contract(source_tree: Path):
    hashes = {}
    for relative in BACKBONE_SOURCE_FILES:
        path = Path(source_tree) / relative
        if not path.is_file():
            raise FileNotFoundError(f"backbone source file is unavailable: {path}")
        hashes[relative] = sha256_file(path)
    return {
        "files": hashes,
        "aggregate_sha256": canonical_sha256(hashes),
    }


def p3_provenance_contract():
    if not P3_ACCEPTANCE.is_file():
        raise FileNotFoundError(f"P3 acceptance report is unavailable: {P3_ACCEPTANCE}")
    acceptance = json.loads(P3_ACCEPTANCE.read_text(encoding="utf-8"))
    if acceptance.get("status") != "PASS":
        raise ValueError("P3 acceptance report has not passed")
    preprocessing_sha256 = acceptance.get("dataset", {}).get("preprocessing_sha256")
    task_manifest_sha256 = acceptance.get("task_manifest_sha256")
    for name, value in (
        ("P3 preprocessing", preprocessing_sha256),
        ("P3 task manifest", task_manifest_sha256),
    ):
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"{name} SHA-256 is missing or malformed")
    return {
        "p3_acceptance_schema": acceptance.get("schema"),
        "p3_acceptance_sha256": sha256_file(P3_ACCEPTANCE),
        "p3_preprocessing_sha256": preprocessing_sha256,
        "p3_task_manifest_sha256": task_manifest_sha256,
        "p3_authority_report_sha256": acceptance.get("authority_report_sha256"),
        "qualification_scope": acceptance.get("qualification_scope"),
    }


def build_contract(
    *,
    model,
    source_path,
    source_payload,
    source_sha256,
    checkpoint_sha256,
    transfers,
    source_tree,
):
    rollout = model.rollout.config
    footprint = FootprintConfig()
    transfer_manifest_sha256 = canonical_sha256(transfers)
    source_code = backbone_source_contract(source_tree)
    p3 = p3_provenance_contract()
    loss_config = asdict(DEPCarLossConfig())
    return {
        "schema": CONTRACT_SCHEMA,
        "contract_version": 2,
        "architecture_id": model.architecture_id,
        "checkpoint_version": CHECKPOINT_VERSION,
        "checkpoint_sha256": checkpoint_sha256,
        "status": STATUS,
        "production_qualified": False,
        "initialization_seed": INITIALIZATION_SEED,
        "implementation_contract": build_p4_implementation_contract(ROOT),
        "input_contract": {
            "depth_metric_and_validity_shape": [2, 480, 640],
            "depth_network_input_shape": [2, *model.config.depth_size],
            "depth_backbone_input_shape": [1, *model.config.depth_size],
            "depth_fields": ["metric_depth_normalized_by_10m", "validity"],
            "validity_fusion": "independent_learned_encoder_added_to_depth_backbone_feature",
            "depth_valid_range_m": [0.2, model.config.depth_max_m],
            "depth_normalization_divisor_m": model.config.depth_max_m,
            "invalid_depth_fill_normalized": 1.0,
            "lidar_bev_shape": [model.config.lidar_channels, *model.config.lidar_size],
            "lidar_bev_channels": list(BEV_CHANNELS),
            "lidar_bev_extent_m": model.config.lidar_extent_m,
            "modality_mask_shape": [2],
            "modality_mask_order": ["depth", "lidar"],
        },
        "state_contract": {
            "dimension": model.config.state_dim,
            "fields": list(model.state_fields),
            "normalization_scale": list(STATE_NORMALIZATION_SCALE),
        },
        "gear_contract": {
            "authority": "deterministic GearSupervisor",
            "input": "requested_gear",
            "allowed_values": [-1, 1],
            "one_constant_gear_per_candidate_bank": True,
            "network_predicts_gear": model.predicts_gear,
        },
        "output_contract": {
            "candidate_count": 15,
            "raw_residuals_shape": [15, 4],
            "residuals_shape": [15, 4],
            "residual_fields": list(AckermannRolloutV1.residual_names),
            "controls_shape": [15, 4],
            "control_fields": [
                "steering_mid", "steering_end", "signed_speed_end", "duration"
            ],
            "trajectory_shape": [15, rollout.steps, 6],
            "trajectory_fields": ["t", "x", "y", "yaw", "signed_speed", "steering"],
            "scores_shape": [15],
            "scores_nonnegative": True,
        },
        "lattice_contract": {
            "candidate_count": 15,
            "order": model.candidate_order,
            "forward_speed_anchors_mps": list(rollout.forward_speed_anchors_mps),
            "reverse_speed_anchors_mps": list(rollout.reverse_speed_anchors_mps),
            "steering_anchors_rad": list(rollout.steering_anchors_rad),
        },
        "rollout_contract": {
            "schema": ACKERMANN_ROLLOUT_SCHEMA,
            "config": asdict(rollout),
            "gear_conditioned": True,
            "differentiable": True,
            "longitudinal_limits_frame": LONGITUDINAL_LIMITS_FRAME,
            "directed_acceleration_definition": "requested_gear * signed_dv_dt",
        },
        "footprint_contract": {
            "schema": FIVE_CIRCLE_FOOTPRINT_SCHEMA,
            "length_m": footprint.length,
            "width_m": footprint.width,
            "safety_margin_m": footprint.safety_margin,
            "circle_count": footprint.circle_count,
            "circle_radius_m": footprint.circle_radius,
            "longitudinal_offsets_m": footprint.longitudinal_offsets.tolist(),
            "runtime_grid_allowance": "half_cell_diagonal",
            "differentiable_training_grid_allowance": "one_cell_diagonal",
            "allowance_schema": FOOTPRINT_ALLOWANCE_SCHEMA,
            "runtime_allowance_multiplier": RUNTIME_HALF_DIAGONAL_MULTIPLIER,
            "training_allowance_multiplier": TRAINING_ONE_DIAGONAL_MULTIPLIER,
            "sweep_interpolation_schema": SWEPT_INTERPOLATION_SCHEMA,
            "sweep_substeps_per_source_segment": SWEPT_SUBSTEPS_PER_SEGMENT,
            "source_trajectory_rows": rollout.steps,
            "safety_query_rows": (
                (rollout.steps - 1) * SWEPT_SUBSTEPS_PER_SEGMENT + 1
            ),
        },
        "training_contract": {
            "objective_id": DEPCarObjectiveV1.objective_id,
            "objective_revision": DEPCarObjectiveV1.objective_revision,
            "sdf_schema": SIGNED_SDF_SCHEMA,
            "loss_config": loss_config,
            "loss_config_sha256": canonical_sha256(loss_config),
            "stage_order": ["candidate_capacity", "score_calibration"],
        },
        "source": {
            "checkpoint_path": _repository_path(source_path),
            "checkpoint_sha256": source_sha256,
            "checkpoint_version": source_payload.get("checkpoint_version"),
            "checkpoint_git_commit": source_payload.get("git_commit"),
            "weight_release": "route-a-v4.8.3-candidate-only",
            "runtime_lineage": "route-a-v4.9.1-recovery-subgoal-v6",
            "network_weights_modified_by_v4_9_1": False,
            "backbone_source": source_code,
        },
        "transfer": {
            "mode": "exact_depth_backbone_only",
            "source_prefix": SOURCE_PREFIX,
            "target_prefix": TARGET_PREFIX,
            "tensor_count": len(transfers),
            "manifest_sha256": transfer_manifest_sha256,
            "partial_transfer_allowed": False,
            "head_transfer_allowed": False,
        },
        "dataset_provenance": p3,
    }


def migrate_checkpoint(source_path=DEFAULT_SOURCE, output_path=DEFAULT_OUTPUT):
    source_path = Path(source_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    source_payload, source_state, source_sha256 = load_authoritative_source(source_path)

    # fork_rng prevents callers from having their global RNG advanced while
    # still making every newly initialized P4 parameter reproducible.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(INITIALIZATION_SEED)
        model = DEPCarNetV1(source_tree=DEP_SOURCE_TREE)
    target_state = model.state_dict()
    transfers = exact_backbone_transfer(source_state, target_state)
    model.load_state_dict(target_state, strict=True)

    transfer_manifest_sha256 = canonical_sha256(transfers)
    implementation = build_p4_implementation_contract(ROOT)
    checkpoint_payload = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "architecture_id": model.architecture_id,
        "model_state_dict": model.state_dict(),
        "status": STATUS,
        "production_qualified": False,
        "initialization_seed": INITIALIZATION_SEED,
        "source_checkpoint_sha256": source_sha256,
        "source_weight_release": "route-a-v4.8.3-candidate-only",
        "source_runtime_lineage": "route-a-v4.9.1-recovery-subgoal-v6",
        "transfer_mode": "exact_depth_backbone_only",
        "transfer_manifest_sha256": transfer_manifest_sha256,
        "implementation_aggregate_sha256": implementation["aggregate_sha256"],
        "transfers": transfers,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Saving through a stream keeps PyTorch's internal archive root independent
    # of the destination filename, so the fixed seed produces byte-identical
    # initialization artifacts in staging and in the final models directory.
    with output_path.open("wb") as stream:
        torch.save(checkpoint_payload, stream)
    checkpoint_sha256 = sha256_file(output_path)

    contract = build_contract(
        model=model,
        source_path=source_path,
        source_payload=source_payload,
        source_sha256=source_sha256,
        checkpoint_sha256=checkpoint_sha256,
        transfers=transfers,
        source_tree=DEP_SOURCE_TREE,
    )
    contract_path = output_path.with_suffix(".contract.json")
    contract_path.write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output_path, contract_path, contract


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    checkpoint, contract_path, contract = migrate_checkpoint(args.source, args.output)
    print(json.dumps({
        "status": "PASS",
        "checkpoint": str(checkpoint),
        "contract": str(contract_path),
        "architecture_id": contract["architecture_id"],
        "checkpoint_sha256": contract["checkpoint_sha256"],
        "source_checkpoint_sha256": contract["source"]["checkpoint_sha256"],
        "transferred_parameter_tensors": contract["transfer"]["tensor_count"],
        "transfer_manifest_sha256": contract["transfer"]["manifest_sha256"],
        "production_qualified": contract["production_qualified"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
