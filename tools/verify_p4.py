#!/usr/bin/env python3
"""Run the P4 implementation gate without starting formal P5 training."""

import argparse
import hashlib
import io
import json
import os
import sys
import tempfile
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

# cuBLAS reads this setting before its first CUDA operation.  It does not make
# CUDA grid-sample backward deterministic, so PyTorch remains in warn-only
# mode below, but it removes avoidable GEMM workspace nondeterminism.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dep_car" / "src"))

from dep_car.model.checkpoint import verify_checkpoint
from dep_car.model.dep_car_net import DEPCarNetV1, DEPCarNetworkOutput
from dep_car.training.losses import DEPCarObjectiveV1, candidate_diversity_loss
from dep_car.training.metrics import candidate_batch_metrics
from dep_car.training.p4_dataset import P3TrainingDatasetV1, p3_training_collate
from dep_car.training.stages import configure_training_stage, modality_mask, parameter_partitions


MINIMUM_REAL_SAMPLES = 16
MINIMUM_REAL_MAPS = 4
CANDIDATE_MAXIMUM_RATIO = 0.20
SCORE_MAXIMUM_RATIO = 0.20
CANDIDATE_SMOKE_READOUT_LR = 5.0e-4
CANDIDATE_SMOKE_READOUT_PREFIXES = (
    "candidate_tower.",
    "candidate_head.",
    "gear_embedding.",
    "speed_embedding.",
    "steering_embedding.",
)
# The curated V3 view contains substantially harder sharp-turn/recovery
# samples than the original 9,290-frame smoke view.  The deterministic oracle
# was still improving at its former 1,500-step boundary and could remain a few
# millipoints above the trained network.  Keep the strict no-worse-than-network
# gate and give the independent optimizer enough budget to converge instead.
DIRECT_ORACLE_STEPS = 3000
DIRECT_ORACLE_RESTARTS = 8
DIRECT_ORACLE_SEED = 49004
DIRECT_ORACLE_MAXIMUM_NOISE_SCALE = 1.0
DIRECT_ORACLE_INSPECTION_INTERVAL = 25
DIRECT_ORACLE_MINIMUM_RELATIVE_IMPROVEMENT = 0.01
DIRECT_ORACLE_TERMINAL_RTOL = 1.0e-5
DIRECT_ORACLE_TERMINAL_ATOL = 1.0e-6


def configure_reproducibility(device, seed=49004):
    """Pin every deterministic CUDA/CPU control available to this smoke.

    The production objective contains CUDA ``grid_sample`` backward, for which
    PyTorch does not provide a deterministic implementation.  Warn-only mode
    keeps that exact production path while deterministic cuDNN and cuBLAS
    remove the much larger convolution/GEMM algorithm drift.
    """

    device = torch.device(device)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    return {
        "seed": seed,
        "deterministic_algorithms_enabled": (
            torch.are_deterministic_algorithms_enabled()
        ),
        "deterministic_algorithms_warn_only": (
            torch.is_deterministic_algorithms_warn_only_enabled()
        ),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "known_nondeterministic_exact_production_operator": (
            "cuda_grid_sample_backward" if device.type == "cuda" else None
        ),
    }


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_verified_checkpoint_payload(checkpoint_path, contract, map_location="cpu"):
    """Load a verified checkpoint from one immutable byte snapshot.

    ``verify_checkpoint`` validates the on-disk artifact and its sidecar before
    this helper is called.  The P4 smoke then takes one byte snapshot, hashes
    exactly those bytes against the returned contract and gives a ``BytesIO``
    view of the same snapshot to the restricted PyTorch loader.  It therefore
    cannot hash one path read and deserialize a different path read.
    """

    checkpoint_path = Path(checkpoint_path)
    checkpoint_bytes = checkpoint_path.read_bytes()
    observed_sha256 = hashlib.sha256(checkpoint_bytes).hexdigest()
    expected_sha256 = str(contract.get("checkpoint_sha256", ""))
    if observed_sha256 != expected_sha256:
        raise RuntimeError(
            "P4 initialization checkpoint bytes differ from verified contract"
        )
    payload = torch.load(
        io.BytesIO(checkpoint_bytes),
        map_location=map_location,
        weights_only=True,
    )
    if not isinstance(payload, dict) or not isinstance(
        payload.get("model_state_dict"), dict
    ):
        raise RuntimeError("P4 initialization checkpoint has no model_state_dict")
    return payload, {
        "status": "PASS",
        "checkpoint_sha256": observed_sha256,
        "expected_checkpoint_sha256": expected_sha256,
        "path_reads_after_contract_verification": 1,
        "deserialization_source": "BytesIO_of_the_hashed_byte_snapshot",
        "torch_load_weights_only": True,
    }


def training_authority_evidence(index_path, index_contract, training_path):
    """Bind the P4 smoke data view to the frozen training configuration."""

    index_path = Path(index_path)
    training_path = Path(training_path)
    training = yaml.safe_load(training_path.read_text(encoding="utf-8"))
    configured = training.get("dataset", {})
    entries = index_contract.get("entries")
    if not isinstance(entries, list):
        raise RuntimeError("training index has no entries list")
    content_sha256 = str(index_contract.get("content_aggregate_sha256", ""))
    configured_content_sha256 = str(
        configured.get("content_aggregate_sha256", "")
    )
    if not content_sha256 or content_sha256 != configured_content_sha256:
        raise RuntimeError(
            "training index content aggregate differs from training.yaml"
        )
    map_sha256 = str(configured.get("map_contract_aggregate_sha256", ""))
    if len(map_sha256) != 64:
        raise RuntimeError(
            "training.yaml has no frozen map contract aggregate SHA-256"
        )
    indexed_samples = int(index_contract.get("samples", -1))
    if indexed_samples != len(entries):
        raise RuntimeError("training index sample count differs from entries")
    splits = tuple(index_contract.get("splits", ()))
    if splits != ("train", "validation"):
        raise RuntimeError("P4 training index must contain only train/validation")
    return {
        "training_index_sha256": sha256_file(index_path),
        "training_index_content_aggregate_sha256": content_sha256,
        "expected_map_contract_aggregate_sha256": map_sha256,
        "expected_map_contract_aggregate_authority": (
            "frozen_training_yaml_configuration_value"
        ),
        "expected_map_contract_aggregate_source": (
            str(training_path.resolve())
            + "::dataset.map_contract_aggregate_sha256"
        ),
        "indexed_samples": indexed_samples,
        "indexed_splits": list(splits),
        "test_split_present": False,
    }


def configured_dataset_paths(training_path, project_root=ROOT):
    """Resolve the P4 smoke view from the same training authority as P5."""

    training_path = Path(training_path).resolve()
    project_root = Path(project_root).resolve()
    training = yaml.safe_load(training_path.read_text(encoding="utf-8"))
    configured = training.get("dataset", {})
    output = {}
    for name in ("root", "maps", "index"):
        raw = configured.get(name)
        if not isinstance(raw, str) or not raw.strip():
            raise RuntimeError("training.yaml dataset.%s is missing" % name)
        path = Path(raw)
        path = path.resolve() if path.is_absolute() else (project_root / path).resolve()
        if project_root != path and project_root not in path.parents:
            raise RuntimeError("training.yaml dataset.%s escapes project root" % name)
        output[name] = path
    return output


def bind_dataset_map_authority(training_authority, dataset_map_contract):
    """Require Dataset's selected-map scan to equal the frozen expectation."""

    evidence = dict(training_authority)
    if not isinstance(dataset_map_contract, dict):
        raise RuntimeError("P4 Dataset exposes no map contract")
    actual_sha256 = str(dataset_map_contract.get("aggregate_sha256", ""))
    expected_sha256 = evidence["expected_map_contract_aggregate_sha256"]
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            "P4 Dataset actual map aggregate differs from training.yaml"
        )
    evidence.update({
        "actual_map_contract_aggregate_sha256": actual_sha256,
        "actual_map_contract_schema": dataset_map_contract.get("schema"),
        "actual_map_count": dataset_map_contract.get("map_count"),
        "actual_equals_expected_map_contract_aggregate": True,
        "actual_map_contract_authority": (
            "P3TrainingDatasetV1_selected_index_maps_verified_bytes_and_semantics"
        ),
    })
    return evidence


def summarize_p3_development_reaudit(report_path, training_authority):
    """Extract P3 development evidence without making it a P4 smoke gate."""

    report_path = Path(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("schema") != "DEPCarP3DevelopmentReauditV3":
        raise RuntimeError("unexpected P3 development re-audit schema")
    statistics = report.get("statistics", {})
    overall = statistics.get("overall", {})
    by_mode = statistics.get("by_mode", {})
    if not isinstance(overall.get("new"), dict) or not isinstance(by_mode, dict):
        raise RuntimeError("P3 development re-audit lacks current-footprint metrics")
    mode_metrics = {}
    for mode, values in sorted(by_mode.items()):
        if not isinstance(values, dict) or not isinstance(values.get("new"), dict):
            raise RuntimeError("P3 development re-audit mode lacks new metrics")
        mode_metrics[mode] = {
            "samples": values.get("samples"),
            "new": values["new"],
        }

    sample_files_audited = int(report.get("sample_files_audited", -1))
    sample_files_discovered = int(report.get("sample_files_discovered", -1))
    expected_samples = int(training_authority["indexed_samples"])
    scope = report.get("scope", {})
    report_authority = report.get("training_authority", {})
    authority_matches = {
        "index_sha256": report_authority.get("index_sha256")
        == training_authority["training_index_sha256"],
        "content_aggregate_sha256": report_authority.get(
            "content_aggregate_sha256"
        )
        == training_authority["training_index_content_aggregate_sha256"],
        "map_contract_aggregate_sha256": report_authority.get(
            "map_contract_aggregate_sha256"
        )
        == training_authority["expected_map_contract_aggregate_sha256"],
    }
    test_access_evidence = {
        "test_split_used_for_tuning": scope.get("test_split_used_for_tuning"),
        "test_npz_opened": scope.get("test_npz_opened"),
        "test_map_yaml_or_png_opened": scope.get("test_map_yaml_or_png_opened"),
        "training_authority_test_split_used": report_authority.get(
            "test_split_used"
        ),
    }
    test_not_accessed = all(
        value is False for value in test_access_evidence.values()
    )
    complete_inventory = bool(
        sample_files_audited == expected_samples
        and sample_files_discovered == expected_samples
        and not report.get("sample_failures")
    )
    p5_gate_errors = []
    if report.get("status") != "PASS":
        p5_gate_errors.extend(
            "p3_reaudit:" + str(error) for error in report.get("errors", ())
        )
        if not report.get("errors"):
            p5_gate_errors.append("p3_reaudit_status_is_not_PASS")
    if report.get("qualification_eligible") is not True:
        p5_gate_errors.append("p3_reaudit_was_not_a_qualifying_full_run")
    if not complete_inventory:
        p5_gate_errors.append("p3_reaudit_did_not_cover_the_frozen_index")
    if not all(authority_matches.values()):
        p5_gate_errors.append("p3_reaudit_training_authority_mismatch")
    if not test_not_accessed:
        p5_gate_errors.append("p3_reaudit_accessed_or_used_sealed_test_data")
    return {
        "path": str(report_path.resolve()),
        "sha256": sha256_file(report_path),
        "schema": report["schema"],
        "status": report.get("status"),
        "sample_files_audited": sample_files_audited,
        "sample_files_discovered": sample_files_discovered,
        "expected_index_samples": expected_samples,
        "complete_frozen_index_audit": complete_inventory,
        "overall_new_metrics": overall["new"],
        "by_mode_new_metrics": mode_metrics,
        "training_authority_matches": authority_matches,
        "test_access_evidence": test_access_evidence,
        "test_not_accessed": test_not_accessed,
        "p5_gate_eligible": not p5_gate_errors,
        "p5_gate_errors": p5_gate_errors,
        "p4_implementation_status_is_independent": True,
    }


def canonical_sha256(value):
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def tensor_sha256(value):
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def verify_loss_contract(contract, training_path, objective):
    training_path = Path(training_path)
    training = yaml.safe_load(training_path.read_text(encoding="utf-8"))
    configured = training.get("loss", {})
    expected = contract.get("training_contract", {})
    expected_config = expected.get("loss_config", {})
    if not expected_config:
        raise RuntimeError("initialization contract has no loss configuration")
    missing = sorted(set(expected_config).difference(configured))
    if missing:
        raise RuntimeError("training.yaml lacks loss keys: " + ", ".join(missing))
    training_config = {key: configured[key] for key in expected_config}
    runtime_config = asdict(objective.config)
    expected_hash = str(expected.get("loss_config_sha256", ""))
    hashes = {
        "initialization_contract": expected_hash,
        "training_yaml": canonical_sha256(training_config),
        "runtime_objective": canonical_sha256(runtime_config),
    }
    if len(set(hashes.values())) != 1:
        raise RuntimeError("training/runtime loss config differs from initialization contract")
    identity = {
        "objective_id": objective.objective_id,
        "objective_revision": objective.objective_revision,
        "sdf_schema": configured.get("sdf_schema"),
    }
    expected_identity = {
        "objective_id": expected.get("objective_id"),
        "objective_revision": expected.get("objective_revision"),
        "sdf_schema": expected.get("sdf_schema"),
    }
    if identity != expected_identity:
        raise RuntimeError("training/runtime objective identity differs from initialization contract")
    return {
        "status": "PASS",
        "training_yaml": str(training_path.resolve()),
        "training_yaml_sha256": sha256_file(training_path),
        "loss_config_sha256": expected_hash,
        "loss_config_hashes": hashes,
        **identity,
    }


def move_batch(batch, device):
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def model_forward(model, batch, mode="fusion"):
    mask = batch["modality_mask"]
    if mode != "fusion":
        mask = modality_mask(len(batch["state"]), mode, device=batch["state"].device)
    return model(
        batch["depth"],
        batch["lidar_bev"],
        batch["state"],
        batch["requested_gear"],
        mask,
    )


def objective_forward(objective, output, batch, stage="candidate_capacity"):
    return objective(
        output,
        map_distance_field=batch["map_distance_field"],
        map_resolution=batch["map_resolution"],
        map_origin=batch["map_origin"],
        chassis_to_map=batch["chassis_to_map"],
        route=batch["route_pose"],
        route_mask=batch["route_mask"],
        requested_gear=batch["requested_gear"],
        geometry_valid=batch["geometry_valid"],
        stage=stage,
    )


def _spread_indices(indices, probes=4):
    """Choose deterministic quantiles without inspecting validation or test."""

    indices = tuple(indices)
    if len(indices) <= probes:
        return indices
    positions = np.linspace(0, len(indices) - 1, probes, dtype=np.int64)
    return tuple(indices[int(position)] for position in positions)


def _entry_drive_contract(dataset, index):
    """Read only the two scalar gear fields needed for stratification."""

    entry = dataset.entries[int(index)]
    path = (dataset.sample_root / entry["path"]).resolve()
    if dataset.sample_root not in path.parents or not path.is_file():
        raise RuntimeError("indexed train sample is unavailable: %s" % path)
    with np.load(str(path), allow_pickle=False) as data:
        current = int(data["current_gear"])
        requested = int(data["requested_gear"])
    if current not in (-1, 0, 1) or requested not in (-1, 1):
        raise RuntimeError("indexed train sample has an invalid gear contract")
    return requested, current != -requested


def _selection_descriptors(dataset, probes_per_mode_map=4):
    groups = defaultdict(list)
    for index, entry in enumerate(dataset.entries):
        if entry.get("split") != "train":
            raise RuntimeError("P4 real-data selector received a non-train index entry")
        groups[(entry["maneuver_mode"], entry["map_uuid"])].append(index)
    descriptors = []
    for (mode, map_uuid), indices in sorted(groups.items()):
        for index in _spread_indices(indices, probes_per_mode_map):
            requested, geometry_valid = _entry_drive_contract(dataset, index)
            if geometry_valid:
                descriptors.append({
                    "index": index,
                    "gear": requested,
                    "maneuver_mode": mode,
                    "map_uuid": map_uuid,
                    "sample_id": dataset.entries[index]["sample_id"],
                })
    return descriptors


def select_real_train_samples(dataset, count=MINIMUM_REAL_SAMPLES):
    """Select a deterministic, train-only, balanced and map-diverse batch."""

    count = int(count)
    if count < MINIMUM_REAL_SAMPLES:
        raise ValueError("P4 qualification requires at least 16 real train samples")
    descriptors = _selection_descriptors(dataset)
    if len(descriptors) < count:
        raise RuntimeError("too few geometry-valid train descriptors for P4 qualification")
    available_modes = sorted({row["maneuver_mode"] for row in descriptors})
    target_mode_count = min(count, len(available_modes))
    gear_targets = {-1: count // 2, 1: count - count // 2}
    selected = []
    remaining = list(descriptors)
    mode_counts = Counter()
    map_counts = Counter()
    gear_counts = Counter()

    def choose(required_mode=None):
        candidates = [
            row for row in remaining
            if gear_counts[row["gear"]] < gear_targets[row["gear"]]
            and (required_mode is None or row["maneuver_mode"] == required_mode)
        ]
        if not candidates:
            return None
        candidates.sort(key=lambda row: (
            0 if row["map_uuid"] not in map_counts else 1,
            map_counts[row["map_uuid"]],
            mode_counts[row["maneuver_mode"]],
            gear_counts[row["gear"]] / float(gear_targets[row["gear"]]),
            row["sample_id"],
        ))
        return candidates[0]

    # First maximize maneuver coverage.  The global gear quotas remain hard,
    # so a rare reverse/forward mode cannot consume the whole batch.
    for mode in available_modes:
        row = choose(mode)
        if row is not None:
            selected.append(row)
            remaining.remove(row)
            mode_counts[row["maneuver_mode"]] += 1
            map_counts[row["map_uuid"]] += 1
            gear_counts[row["gear"]] += 1
    while len(selected) < count:
        row = choose()
        if row is None:
            raise RuntimeError("could not fill balanced real train qualification batch")
        selected.append(row)
        remaining.remove(row)
        mode_counts[row["maneuver_mode"]] += 1
        map_counts[row["map_uuid"]] += 1
        gear_counts[row["gear"]] += 1

    if gear_counts != Counter(gear_targets):
        raise RuntimeError("real train qualification batch is not gear-balanced")
    if len(map_counts) < MINIMUM_REAL_MAPS:
        raise RuntimeError("real train qualification batch covers fewer than four maps")
    if len(mode_counts) < target_mode_count:
        raise RuntimeError("real train qualification batch did not maximize maneuver coverage")

    samples = [dataset[row["index"]] for row in selected]
    for row, sample in zip(selected, samples):
        metadata = sample["metadata"]
        if (
            metadata["split"] != "train"
            or metadata["sample_id"] != row["sample_id"]
            or metadata["map_uuid"] != row["map_uuid"]
            or metadata["maneuver_mode"] != row["maneuver_mode"]
            or int(sample["requested_gear"]) != row["gear"]
            or not bool(sample["geometry_valid"])
        ):
            raise RuntimeError("loaded train sample disagrees with its selection descriptor")
    return samples


def terminal_history(history, window=12):
    values = np.asarray(history, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise RuntimeError("tiny-overfit history is empty or non-finite")
    size = min(len(values), max(1, int(window)))
    initial = float(values[0])
    terminal_mean = float(values[-size:].mean())
    terminal_last = float(values[-1])
    return {
        "initial": initial,
        "terminal_window_size": size,
        "terminal_window_mean": terminal_mean,
        "terminal_last": terminal_last,
        "terminal_window_to_initial_ratio": terminal_mean / max(initial, 1.0e-12),
        "terminal_last_to_initial_ratio": terminal_last / max(initial, 1.0e-12),
        "minimum_observed": float(values.min()),
        "finite": True,
    }


def ranking_snapshot(output, objective_result):
    metrics = candidate_batch_metrics(output, objective_result)
    valid = ~metrics["zero_feasible"]
    valid_count = int(valid.sum())
    if valid_count < 1:
        raise RuntimeError("score calibration batch has no hard-feasible candidate")
    feasible = objective_result["hard_feasible"].bool()
    cost = objective_result["candidate_cost"]
    oracle_index = cost.masked_fill(~feasible, float("inf")).argmin(dim=1)
    selected = metrics["selected_index"]
    top1 = float((selected[valid] == oracle_index[valid]).float().mean())
    regret = metrics["oracle_regret"][valid]
    return {
        "valid_samples": valid_count,
        "zero_feasible_samples": int((~valid).sum()),
        "top1_oracle_accuracy": top1,
        "mean_oracle_regret": float(regret.mean()),
        "maximum_oracle_regret": float(regret.max()),
    }


def candidate_loss_breakdown(objective, output, objective_result):
    """Expose the exact weighted terms that sum to candidate loss."""

    config = objective.config
    weights = config.weights
    indices = torch.topk(
        objective_result["candidate_cost"],
        config.capacity_top_k,
        dim=1,
        largest=False,
    ).indices

    def selected_mean(values):
        return float(torch.gather(values, 1, indices).mean().detach())

    top_safety = selected_mean(objective_result["safety_per_candidate"])
    top_guidance = selected_mean(objective_result["guidance_per_candidate"])
    top_kinematic = selected_mean(objective_result["kinematic_per_candidate"])
    top_comfort = selected_mean(objective_result["comfort_per_candidate"])
    terms = {
        "top_k_safety": weights.safety * top_safety,
        "top_k_guidance": weights.guidance * top_guidance,
        "top_k_kinematic": weights.kinematic * top_kinematic,
        "top_k_comfort": weights.comfort * top_comfort,
        "all_candidate_safety": weights.safety * float(objective_result["safety"].detach()),
        "diversity": weights.diversity * float(objective_result["diversity"].detach()),
        "anchor": weights.anchor * float(objective_result["anchor"].detach()),
    }
    weighted_sum = sum(terms.values())
    total = float(objective_result["candidate"].detach())
    if not np.isclose(weighted_sum, total, rtol=1.0e-5, atol=1.0e-6):
        raise RuntimeError("candidate loss breakdown does not reconstruct total")
    return {
        "candidate_total": total,
        "weighted_terms": terms,
        "weighted_terms_sum": weighted_sum,
        "raw_top_k_terms": {
            "safety": top_safety,
            "guidance": top_guidance,
            "kinematic": top_kinematic,
            "comfort": top_comfort,
        },
    }


def candidate_loss_per_sample(objective, output, objective_result, batch):
    """Reconstruct the exact separable candidate loss for oracle selection.

    The direct oracle owns one free residual tensor per real sample.  Selecting
    one whole restart for the entire batch is therefore unnecessarily weak:
    different samples may converge in different deterministic restart basins.
    This helper exposes the same production terms before their final batch
    mean so the oracle can retain the best achieved residuals per sample.
    """

    geometry_valid = batch["geometry_valid"].bool()
    if not bool(torch.all(geometry_valid)):
        raise RuntimeError("P4 oracle batch contains invalid geometry")
    config = objective.config
    weights = config.weights
    top_k = torch.topk(
        objective_result["candidate_cost"],
        config.capacity_top_k,
        dim=1,
        largest=False,
    ).values.mean(dim=1)
    diversity = candidate_diversity_loss(
        output.trajectories,
        batch["requested_gear"],
        config.minimum_normalized_terminal_separation,
        config.forward_diversity_scales,
        config.reverse_diversity_scales,
    )
    anchor = output.residuals.square().mean(dim=(-1, -2))
    per_sample = (
        top_k
        + weights.safety
        * objective_result["safety_per_candidate"].mean(dim=1)
        + weights.diversity * diversity
        + weights.anchor * anchor
    )
    if not torch.allclose(
        per_sample.mean(), objective_result["candidate"], rtol=1.0e-5, atol=1.0e-6
    ):
        raise RuntimeError("per-sample oracle loss does not reconstruct candidate loss")
    return per_sample


def direct_residual_oracle(objective, model, batch, starting_raw):
    """Find an achieved per-sample residual floor under the production bounds.

    This auxiliary optimization bypasses the encoders but uses the exact same
    bounded rollout, map SDF, footprint and candidate objective.  Its lowest
    achieved loss is conservative for the reducible-gap gate: a lower floor
    makes the normalized network gap larger, never easier.
    """

    initial_raw = starting_raw.detach().clone()

    def evaluate(raw, objective_batch):
        rollout = model.rollout(
            objective_batch["state"], objective_batch["requested_gear"], raw
        )
        output = DEPCarNetworkOutput(
            raw_residuals=raw,
            residuals=rollout.residuals,
            controls=rollout.controls,
            trajectories=rollout.trajectory,
            scores=raw.new_zeros(raw.shape[:2]),
        )
        return output, objective_forward(objective, output, objective_batch)

    with torch.no_grad():
        _, initial_result = evaluate(initial_raw, batch)
        initial_loss = float(initial_result["total"])

    # Restart zero is the exact saved step-0 tensor.  The other deterministic
    # restarts explore basins around that tensor and never consume a trained
    # network output.  A stronger floor only makes the normalized network gate
    # harder, so this exploration cannot manufacture a pass.
    generator = torch.Generator(device=initial_raw.device).manual_seed(
        DIRECT_ORACLE_SEED
    )
    scales = torch.linspace(
        0.0,
        DIRECT_ORACLE_MAXIMUM_NOISE_SCALE,
        DIRECT_ORACLE_RESTARTS,
        device=initial_raw.device,
        dtype=initial_raw.dtype,
    ).view(-1, 1, 1, 1)
    population = initial_raw.unsqueeze(0).repeat(
        DIRECT_ORACLE_RESTARTS, 1, 1, 1
    )
    population = population + scales * torch.randn(
        population.shape,
        generator=generator,
        device=initial_raw.device,
        dtype=initial_raw.dtype,
    )
    repeated_keys = (
        "state", "requested_gear", "map_distance_field", "map_resolution",
        "map_origin", "chassis_to_map", "route_pose", "route_mask",
        "geometry_valid",
    )
    repeated_batch = {
        key: batch[key].repeat(
            (DIRECT_ORACLE_RESTARTS,) + (1,) * (batch[key].ndim - 1)
        )
        for key in repeated_keys
    }
    raw = torch.nn.Parameter(population.flatten(0, 1))
    optimizer = torch.optim.Adam((raw,), lr=1.0e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=DIRECT_ORACLE_STEPS, eta_min=1.0e-5
    )
    population_history = []
    best_loss = float("inf")
    best_raw = None
    best_per_sample_loss = None
    best_step = None
    restart_initial_losses = None
    restart_final_losses = None

    def inspect_restarts(step):
        nonlocal best_loss, best_raw, best_step, best_per_sample_loss
        shaped = raw.detach().reshape(
            DIRECT_ORACLE_RESTARTS, *initial_raw.shape
        )
        losses = []
        with torch.no_grad():
            for restart_raw in shaped:
                restart_output, restart_result = evaluate(restart_raw, batch)
                value = float(restart_result["total"])
                losses.append(value)
                per_sample = candidate_loss_per_sample(
                    objective, restart_output, restart_result, batch
                )
                if best_per_sample_loss is None:
                    best_per_sample_loss = per_sample.clone()
                    best_raw = restart_raw.clone()
                else:
                    improved = per_sample < best_per_sample_loss
                    best_per_sample_loss = torch.where(
                        improved, per_sample, best_per_sample_loss
                    )
                    best_raw[improved] = restart_raw[improved]
                combined = float(best_per_sample_loss.mean())
                if np.isfinite(combined) and combined < best_loss:
                    best_loss = combined
                    best_step = step
        return losses

    for step in range(DIRECT_ORACLE_STEPS + 1):
        _, result = evaluate(raw, repeated_batch)
        value = float(result["total"].detach())
        population_history.append(value)
        if step % DIRECT_ORACLE_INSPECTION_INTERVAL == 0 or step == DIRECT_ORACLE_STEPS:
            restart_losses = inspect_restarts(step)
            if restart_initial_losses is None:
                restart_initial_losses = restart_losses
            if step == DIRECT_ORACLE_STEPS:
                restart_final_losses = restart_losses
        if step == DIRECT_ORACLE_STEPS:
            break
        optimizer.zero_grad(set_to_none=True)
        result["total"].backward()
        optimizer.step()
        scheduler.step()
    if best_raw is None:
        raise RuntimeError("direct residual oracle produced no finite iterate")
    with torch.no_grad():
        best_output, best_result = evaluate(best_raw, batch)
    best_loss = float(best_result["total"])
    population_convergence = terminal_history(population_history, window=50)
    absolute_improvement = initial_loss - best_loss
    relative_improvement = absolute_improvement / max(
        abs(initial_loss), 1.0e-12
    )
    best_initial_restart_loss = min(restart_initial_losses)
    optimization_absolute_improvement = best_initial_restart_loss - best_loss
    optimization_relative_improvement = optimization_absolute_improvement / max(
        abs(best_initial_restart_loss), 1.0e-12
    )
    convergence = {
        "finite": bool(
            np.isfinite(initial_loss)
            and np.isfinite(population_history).all()
            and np.isfinite(best_loss)
        ),
        "initial": initial_loss,
        "optimizer": (
            "deterministic_multistart_Adam+CosineAnnealingLR+"
            "per_sample_best_achieved_restart"
        ),
        "steps": DIRECT_ORACLE_STEPS,
        "initial_learning_rate": 1.0e-2,
        "final_learning_rate": 1.0e-5,
        "starting_point": "network_step0_raw_residuals",
        "starting_raw_sha256": tensor_sha256(initial_raw),
        "input_contract": [
            "production_objective", "production_rollout", "real_train_batch",
            "network_step0_raw_residuals",
        ],
        "exact_production_objective_and_rollout": True,
        "restart_count": DIRECT_ORACLE_RESTARTS,
        "per_sample_restart_selection": True,
        "per_sample_residual_ownership": True,
        "restart_seed": DIRECT_ORACLE_SEED,
        "restart_noise_scales": [float(value) for value in scales.flatten().cpu()],
        "exact_step0_restart_present": bool(scales.flatten()[0] == 0),
        "inspection_interval": DIRECT_ORACLE_INSPECTION_INTERVAL,
        "restart_initial_losses": restart_initial_losses,
        "restart_final_losses": restart_final_losses,
        "exact_step0_restart_loss": restart_initial_losses[0],
        "best_initial_restart_loss": best_initial_restart_loss,
        "achieved_floor": best_loss,
        "best_step": best_step,
        "absolute_improvement": absolute_improvement,
        "relative_improvement": relative_improvement,
        "optimization_absolute_improvement": optimization_absolute_improvement,
        "optimization_relative_improvement": optimization_relative_improvement,
        "minimum_required_relative_improvement": (
            DIRECT_ORACLE_MINIMUM_RELATIVE_IMPROVEMENT
        ),
        "population_initial_mean": population_convergence["initial"],
        "population_minimum_observed": population_convergence["minimum_observed"],
        "population_terminal_window_mean": population_convergence["terminal_window_mean"],
        "population_terminal_last": population_convergence["terminal_last"],
        "population_terminal_last_minus_floor": (
            population_convergence["terminal_last"] - best_loss
        ),
    }
    return convergence, best_output, best_result


def direct_oracle_gate(
    convergence,
    candidate_terminal,
    expected_starting_raw_sha256,
    network_terminal_raw_sha256,
):
    """Require an independent, converged oracle that beats the network terminal.

    The oracle must start at the saved network step-0 residuals.  Matching the
    corresponding step-0 loss prevents a later network residual from silently
    becoming the oracle seed.  A failed optimizer therefore cannot manufacture
    a zero normalized gap by merely returning the trained network terminal.
    """

    oracle_initial = float(convergence["initial"])
    network_initial = float(candidate_terminal["initial"])
    network_terminal = float(candidate_terminal["terminal_last"])
    tolerance = DIRECT_ORACLE_TERMINAL_ATOL + (
        DIRECT_ORACLE_TERMINAL_RTOL * abs(network_terminal)
    )
    starts_at_network_step0 = bool(
        np.isclose(
            oracle_initial,
            network_initial,
            rtol=DIRECT_ORACLE_TERMINAL_RTOL,
            atol=DIRECT_ORACLE_TERMINAL_ATOL,
        )
    )
    exact_restart_matches_step0 = bool(
        np.isclose(
            float(convergence["exact_step0_restart_loss"]),
            network_initial,
            rtol=DIRECT_ORACLE_TERMINAL_RTOL,
            atol=DIRECT_ORACLE_TERMINAL_ATOL,
        )
    )
    significantly_converged = bool(
        float(convergence["relative_improvement"])
        >= DIRECT_ORACLE_MINIMUM_RELATIVE_IMPROVEMENT
        and float(convergence["optimization_relative_improvement"])
        >= DIRECT_ORACLE_MINIMUM_RELATIVE_IMPROVEMENT
    )
    independent_from_network_terminal = bool(
        convergence.get("starting_point") == "network_step0_raw_residuals"
        and convergence.get("starting_raw_sha256")
        == expected_starting_raw_sha256
        and expected_starting_raw_sha256 != network_terminal_raw_sha256
        and convergence.get("exact_step0_restart_present") is True
        and exact_restart_matches_step0
    )
    finite = bool(convergence.get("finite"))
    exact_production_path = bool(
        convergence.get("exact_production_objective_and_rollout")
    )
    no_worse_than_network_terminal = bool(
        float(convergence["achieved_floor"]) <= network_terminal + tolerance
    )
    errors = []
    if not starts_at_network_step0:
        errors.append("direct_residual_oracle_did_not_start_at_network_step0")
    if not independent_from_network_terminal:
        errors.append("direct_residual_oracle_is_not_independent_from_network_terminal")
    if not finite:
        errors.append("direct_residual_oracle_is_not_finite")
    if not exact_production_path:
        errors.append("direct_residual_oracle_did_not_use_production_path")
    if not significantly_converged:
        errors.append("direct_residual_oracle_did_not_significantly_converge")
    if not no_worse_than_network_terminal:
        errors.append("direct_residual_oracle_worse_than_network_terminal")
    return {
        "starts_at_network_step0": starts_at_network_step0,
        "exact_restart_matches_step0": exact_restart_matches_step0,
        "independent_from_network_terminal": independent_from_network_terminal,
        "finite": finite,
        "exact_production_path": exact_production_path,
        "expected_starting_raw_sha256": expected_starting_raw_sha256,
        "network_terminal_raw_sha256": network_terminal_raw_sha256,
        "significantly_converged": significantly_converged,
        "no_worse_than_network_terminal": no_worse_than_network_terminal,
        "network_terminal_loss": network_terminal,
        "network_comparison_tolerance": tolerance,
        "errors": errors,
    }


def score_teacher_entropy(objective, objective_result, geometry_valid):
    cost = objective_result["candidate_cost"]
    feasible = objective_result["hard_feasible"].bool()
    effective = torch.where(
        feasible.any(dim=1, keepdim=True), feasible, torch.ones_like(feasible)
    )
    teacher = cost.masked_fill(~effective, 1.0e4)
    probability = torch.softmax(
        -teacher / objective.config.score_temperature, dim=-1
    )
    entropy_per = -(
        probability * torch.log(probability.clamp_min(1.0e-30))
    ).sum(dim=1)
    weights = geometry_valid.to(entropy_per)
    entropy = (entropy_per * weights).sum() / weights.sum().clamp_min(1.0)
    return float(entropy.detach())


def gradient_evidence(model, groups):
    evidence = {}
    prefixes = (
        "depth_encoder", "lidar_encoder", "state_encoder",
        "speed_embedding", "steering_embedding", "candidate_head",
    )
    for prefix in prefixes:
        norm = sum(
            float(parameter.grad.detach().abs().sum())
            for name, parameter in model.named_parameters()
            if name.startswith(prefix) and parameter.grad is not None
        )
        evidence[prefix] = norm
    evidence["score_has_gradient"] = any(parameter.grad is not None for parameter in groups["score"])
    if any(not np.isfinite(value) or value <= 1.0e-8 for key, value in evidence.items() if key != "score_has_gradient"):
        raise RuntimeError("candidate objective did not reach every required P4 branch")
    if evidence["score_has_gradient"]:
        raise RuntimeError("candidate stage leaked gradients into the score partition")
    return evidence


def candidate_smoke_parameter_groups(model):
    """Freeze features after the separate all-branch gradient probe.

    P4 is a fixed-batch capacity smoke, not the formal P5 optimizer contract.
    Backpropagating every tiny-batch update through CUDA grid sampling,
    adaptive pooling and both encoders produced platform-dependent local
    plateaus.  The separate gradient probe still requires signal in every
    sensor/state/query branch.  This second gate then verifies the capacity of
    logical queries and the candidate tower/head on immutable features.
    """

    partitions = parameter_partitions(model)
    candidate_ids = {id(parameter) for parameter in partitions["candidate"]}
    score_ids = {id(parameter) for parameter in partitions["score"]}
    fixed_feature = []
    readout = []
    fixed_feature_names = []
    readout_names = []
    for name, parameter in model.named_parameters():
        if id(parameter) not in candidate_ids:
            continue
        if name.startswith(CANDIDATE_SMOKE_READOUT_PREFIXES):
            readout.append(parameter)
            readout_names.append(name)
            parameter.requires_grad_(True)
        else:
            fixed_feature.append(parameter)
            fixed_feature_names.append(name)
            parameter.requires_grad_(False)
    grouped_ids = {id(parameter) for parameter in fixed_feature + readout}
    if grouped_ids != candidate_ids or grouped_ids.intersection(score_ids):
        raise RuntimeError("candidate smoke optimizer ownership mismatch")
    if not fixed_feature or not readout:
        raise RuntimeError("candidate smoke optimizer has an empty parameter group")
    optimizer_groups = [
        {"params": readout, "lr": CANDIDATE_SMOKE_READOUT_LR},
    ]
    evidence = {
        "schema": "P4CandidateFixedFeatureReadoutCapacityV1",
        "purpose": (
            "fixed_feature_readout_capacity_smoke_not_formal_P5_optimizer"
        ),
        "readout_learning_rate": CANDIDATE_SMOKE_READOUT_LR,
        "fixed_feature_parameter_tensors": len(fixed_feature),
        "readout_parameter_tensors": len(readout),
        "fixed_feature_parameters": sum(
            parameter.numel() for parameter in fixed_feature
        ),
        "readout_parameters": sum(parameter.numel() for parameter in readout),
        "candidate_ownership_accounted": True,
        "fixed_feature_parameters_frozen": True,
        "readout_parameters_trainable": True,
        "score_parameters_excluded": True,
        "readout_prefixes": list(CANDIDATE_SMOKE_READOUT_PREFIXES),
        "fixed_feature_parameter_names_sha256": hashlib.sha256(
            "\n".join(sorted(fixed_feature_names)).encode("utf-8")
        ).hexdigest(),
        "readout_parameter_names_sha256": hashlib.sha256(
            "\n".join(sorted(readout_names)).encode("utf-8")
        ).hexdigest(),
    }
    return optimizer_groups, evidence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=240, help="fixed-feature candidate readout tiny-overfit updates")
    parser.add_argument("--score-steps", type=int, default=500)
    parser.add_argument("--samples", type=int, default=MINIMUM_REAL_SAMPLES)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "p4_model_implementation_acceptance.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.steps < 1 or args.score_steps < 1 or args.threads < 1:
        raise ValueError("candidate steps, score steps and threads must be positive")
    if args.samples < MINIMUM_REAL_SAMPLES:
        raise ValueError("P4 qualification requires at least 16 real train samples")
    torch.set_num_threads(args.threads)
    reproducibility = configure_reproducibility(args.device)
    started = time.monotonic()

    checkpoint = ROOT / "models" / "dep_car" / "dep_car_net_v1_depth_v483_init.pth"
    contract_path = checkpoint.with_suffix(".contract.json")
    contract = verify_checkpoint(checkpoint, contract_path, allow_untrained=True)
    payload, checkpoint_load = load_verified_checkpoint_payload(
        checkpoint, contract, map_location=args.device
    )
    training_path = ROOT / "dep_car" / "config" / "training.yaml"
    dataset_paths = configured_dataset_paths(training_path)
    training_index_path = dataset_paths["index"]
    training_index_contract = json.loads(
        training_index_path.read_text(encoding="utf-8")
    )
    training_authority = training_authority_evidence(
        training_index_path, training_index_contract, training_path
    )
    dataset = P3TrainingDatasetV1(
        dataset_paths["root"],
        dataset_paths["maps"],
        split="train",
        index_path=training_index_path,
        index_splits=("train", "validation"),
        workers=args.threads,
        expected_map_contract_aggregate_sha256=training_authority[
            "expected_map_contract_aggregate_sha256"
        ],
    )
    training_authority = bind_dataset_map_authority(
        training_authority, dataset.map_contract
    )
    p3_reaudit = summarize_p3_development_reaudit(
        ROOT / "reports" / "p3_development_reaudit_v3.json",
        training_authority,
    )
    selected = select_real_train_samples(dataset, args.samples)
    batch = move_batch(p3_training_collate(selected), torch.device(args.device))

    model = DEPCarNetV1().to(args.device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    groups = parameter_partitions(model)
    configure_training_stage(model, "candidate_capacity")
    objective = DEPCarObjectiveV1()
    loss_contract = verify_loss_contract(
        contract, training_path, objective
    )

    # Gate 1: prove that the exact production objective reaches every
    # candidate-owned branch.  No optimizer update is allowed, and the full
    # checkpoint (including BatchNorm buffers) is restored before gate 2.
    model.train()
    model.zero_grad(set_to_none=True)
    probe_output = model_forward(model, batch)
    probe_losses = objective_forward(objective, probe_output, batch)
    probe_losses["total"].backward()
    gradients = gradient_evidence(model, groups)
    gradient_probe = {
        "mode": "all_candidate_branches_train_mode",
        "optimizer_steps": 0,
        "loss": float(probe_losses["total"].detach()),
        "checkpoint_reloaded_after_probe": True,
        "batchnorm_buffers_restored_after_probe": True,
    }
    model.zero_grad(set_to_none=True)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    configure_training_stage(model, "candidate_capacity")

    # Gate 2: on immutable encoder/state features, verify that the logical
    # query embeddings and candidate tower/head can fit the real fixed batch.
    candidate_optimizer_groups, candidate_optimizer_contract = (
        candidate_smoke_parameter_groups(model)
    )
    optimizer = torch.optim.AdamW(candidate_optimizer_groups, weight_decay=0.0)

    model.eval()
    candidate_history = []
    initial_candidate_breakdown = None
    initial_raw_residuals = None
    for step in range(args.steps + 1):
        output = model_forward(model, batch)
        losses = objective_forward(objective, output, batch)
        candidate_history.append(float(losses["total"].detach()))
        if initial_candidate_breakdown is None:
            initial_candidate_breakdown = candidate_loss_breakdown(
                objective, output, losses
            )
            initial_raw_residuals = output.raw_residuals.detach().clone()
        if step == args.steps:
            break
        optimizer.zero_grad(set_to_none=True)
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad], 10.0
        )
        optimizer.step()

    candidate_terminal = terminal_history(candidate_history)
    final_candidate_breakdown = candidate_loss_breakdown(objective, output, losses)
    initial_raw_sha256 = tensor_sha256(initial_raw_residuals)
    network_terminal_raw_sha256 = tensor_sha256(output.raw_residuals)
    oracle_convergence, oracle_output, oracle_result = direct_residual_oracle(
        objective, model, batch, initial_raw_residuals
    )
    oracle_breakdown = candidate_loss_breakdown(
        objective, oracle_output, oracle_result
    )
    candidate_floor = oracle_convergence["achieved_floor"]
    candidate_reducible_initial = candidate_terminal["initial"] - candidate_floor
    candidate_reducible_terminal = (
        candidate_terminal["terminal_window_mean"] - candidate_floor
    )
    candidate_reducible_last = candidate_terminal["terminal_last"] - candidate_floor
    if candidate_reducible_initial <= 1.0e-8:
        raise RuntimeError("direct residual floor does not leave a positive candidate gap")
    candidate_ratio = candidate_reducible_terminal / candidate_reducible_initial
    candidate_last_ratio = candidate_reducible_last / candidate_reducible_initial
    qualification_errors = []
    oracle_gate = direct_oracle_gate(
        oracle_convergence,
        candidate_terminal,
        initial_raw_sha256,
        network_terminal_raw_sha256,
    )
    qualification_errors.extend(oracle_gate["errors"])
    if (
        candidate_ratio > CANDIDATE_MAXIMUM_RATIO
        or candidate_last_ratio > CANDIDATE_MAXIMUM_RATIO
    ):
        qualification_errors.append(
            "candidate_reducible_gap_ratio_exceeds_0_20"
        )
    candidate_terminal_drift = abs(
        candidate_terminal["terminal_last"]
        - candidate_terminal["terminal_window_mean"]
    ) / max(candidate_terminal["initial"], 1.0e-12)
    if candidate_terminal_drift > 0.01:
        qualification_errors.append("candidate_terminal_window_is_unstable")

    # Stage two starts from the actually overfit candidate branch, then freezes
    # every candidate parameter and BatchNorm buffer.  It must improve ranking
    # on the same finite real-data batch without any candidate gradient.
    configure_training_stage(model, "score_calibration")
    model.zero_grad(set_to_none=True)
    model.eval()
    score_optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=1.0e-3,
        weight_decay=0.0,
    )
    with torch.no_grad():
        initial_score_output = model_forward(model, batch)
        initial_score_losses = objective_forward(
            objective, initial_score_output, batch, stage="score_calibration"
        )
        initial_ranking = ranking_snapshot(initial_score_output, initial_score_losses)
        teacher_entropy = score_teacher_entropy(
            objective, initial_score_losses, batch["geometry_valid"]
        )

    score_history = []
    score_gradient_evidence = None
    for step in range(args.score_steps + 1):
        score_output = model_forward(model, batch)
        score_losses = objective_forward(
            objective, score_output, batch, stage="score_calibration"
        )
        score_history.append(float(score_losses["total"].detach()))
        if step == args.score_steps:
            break
        model.zero_grad(set_to_none=True)
        score_losses["total"].backward()
        candidate_with_gradient = sum(
            parameter.grad is not None for parameter in groups["candidate"]
        )
        score_gradient_norm = sum(
            float(parameter.grad.detach().abs().sum())
            for parameter in groups["score"] if parameter.grad is not None
        )
        if candidate_with_gradient:
            raise RuntimeError("score calibration leaked gradients into candidate parameters")
        if not np.isfinite(score_gradient_norm) or score_gradient_norm <= 1.0e-8:
            raise RuntimeError("score calibration did not reach the score partition")
        if score_gradient_evidence is None:
            score_gradient_evidence = {
                "candidate_parameters_with_gradient": candidate_with_gradient,
                "score_gradient_l1": score_gradient_norm,
            }
        torch.nn.utils.clip_grad_norm_(groups["score"], 10.0)
        score_optimizer.step()

    score_terminal = terminal_history(score_history)
    with torch.no_grad():
        final_score_output = model_forward(model, batch)
        final_score_losses = objective_forward(
            objective, final_score_output, batch, stage="score_calibration"
        )
        final_ranking = ranking_snapshot(final_score_output, final_score_losses)
    top1_improved = (
        final_ranking["top1_oracle_accuracy"]
        > initial_ranking["top1_oracle_accuracy"]
    )
    regret_improved = (
        final_ranking["mean_oracle_regret"]
        < initial_ranking["mean_oracle_regret"]
    )
    score_initial_excess = score_terminal["initial"] - teacher_entropy
    score_terminal_excess = score_terminal["terminal_window_mean"] - teacher_entropy
    score_last_excess = score_terminal["terminal_last"] - teacher_entropy
    if score_initial_excess <= 1.0e-8:
        raise RuntimeError("teacher entropy does not leave a positive score KL gap")
    score_excess_ratio = score_terminal_excess / score_initial_excess
    score_last_excess_ratio = score_last_excess / score_initial_excess
    if score_excess_ratio > SCORE_MAXIMUM_RATIO or score_last_excess_ratio > SCORE_MAXIMUM_RATIO:
        qualification_errors.append("score_reducible_gap_ratio_exceeds_0_20")
    if not top1_improved:
        qualification_errors.append("score_top1_oracle_accuracy_not_improved")
    if not regret_improved:
        qualification_errors.append("score_mean_oracle_regret_not_improved")

    model.eval()
    with torch.no_grad():
        modality_checks = {}
        for mode in ("depth_only", "lidar_only", "fusion"):
            value = model_forward(model, batch, mode)
            modality_checks[mode] = {
                "finite": all(bool(torch.isfinite(tensor).all()) for tensor in value),
                "trajectory_shape": list(value.trajectories.shape),
            }
        reference = model_forward(model, batch)
        with tempfile.TemporaryDirectory(prefix="dep_car_p4_roundtrip_") as temporary:
            path = Path(temporary) / "state.pth"
            torch.save(model.state_dict(), path)
            reloaded = DEPCarNetV1().to(args.device).eval()
            reloaded.load_state_dict(torch.load(path, map_location=args.device, weights_only=True), strict=True)
            restored = model_forward(reloaded, batch)
        maximum_roundtrip_error = max(
            float((left - right).abs().max()) for left, right in zip(reference, restored)
        )
    if not all(value["finite"] for value in modality_checks.values()):
        raise RuntimeError("a P4 ablation path produced NaN/Inf")
    if maximum_roundtrip_error > 1.0e-6:
        raise RuntimeError("checkpoint round-trip changed P4 inference")

    metadata = batch["metadata"]
    report = {
        "schema": "DEPCarP4ImplementationAcceptanceV3",
        "date": "2026-08-16",
        "status": "PASS" if not qualification_errors else "FAIL",
        "errors": qualification_errors,
        "scope": "P4 implementation/smoke qualification only; formal P5 training and P6 closed-loop qualification are not claimed",
        "architecture_id": model.architecture_id,
        "initialization_checkpoint_sha256": checkpoint_load[
            "checkpoint_sha256"
        ],
        "initialization_contract_sha256": sha256_file(contract_path),
        "initialization_checkpoint_load": checkpoint_load,
        "implementation_aggregate_sha256": contract["implementation_contract"][
            "aggregate_sha256"
        ],
        "transfer": contract["transfer"],
        "rollout_contract": contract["rollout_contract"],
        "footprint_contract": contract["footprint_contract"],
        "dataset": {
            **training_authority,
            "indexed_train_samples": len(dataset),
            "test_split_accessed": False,
            "smoke_samples": len(metadata),
            "smoke_map_uuids": sorted({item["map_uuid"] for item in metadata}),
            "smoke_maneuver_modes": dict(sorted(Counter(item["maneuver_mode"] for item in metadata).items())),
            "smoke_requested_gears": dict(sorted(Counter(int(value) for value in batch["requested_gear"].cpu().tolist()).items())),
            "sample_ids": [item["sample_id"] for item in metadata],
            "selection_contract": {
                "split": "train",
                "minimum_samples": MINIMUM_REAL_SAMPLES,
                "minimum_maps": MINIMUM_REAL_MAPS,
                "balanced_requested_gears": True,
                "maximize_available_maneuver_coverage": True,
                "validation_samples_selected": False,
            },
        },
        "p3_development_reaudit": p3_reaudit,
        "sealed_test_data": {
            "test_split_accessed_by_p4": False,
            "validation_raw_bytes_hashed_for_index_authority": True,
            "validation_npz_semantics_parsed": False,
            "validation_map_authority_opened": True,
            "p3_reaudit_reported_access": p3_reaudit[
                "test_access_evidence"
            ],
            "p3_reaudit_test_seal_verified": p3_reaudit["test_not_accessed"],
        },
        "candidate_capacity_tiny_overfit": {
            "steps": args.steps,
            "gradient_connectivity_probe": gradient_probe,
            "optimizer_contract": candidate_optimizer_contract,
            "history_evaluation": "terminal_window_mean_and_terminal_last; minimum is diagnostic only",
            **candidate_terminal,
            "raw_terminal_ratios_are_diagnostic_only": True,
            "direct_residual_oracle": oracle_convergence,
            "direct_residual_oracle_gate": oracle_gate,
            "initial_loss_breakdown": initial_candidate_breakdown,
            "terminal_loss_breakdown": final_candidate_breakdown,
            "oracle_loss_breakdown": oracle_breakdown,
            "initial_reducible_gap": candidate_reducible_initial,
            "terminal_window_reducible_gap": candidate_reducible_terminal,
            "terminal_last_reducible_gap": candidate_reducible_last,
            "terminal_window_reducible_gap_ratio": candidate_ratio,
            "terminal_last_reducible_gap_ratio": candidate_last_ratio,
            "required_maximum_reducible_gap_ratio": CANDIDATE_MAXIMUM_RATIO,
            "terminal_window_drift_ratio": candidate_terminal_drift,
        },
        "score_calibration_tiny_overfit": {
            "steps": args.score_steps,
            "learning_rate": 1.0e-3,
            "history_evaluation": "teacher-entropy-normalized terminal window and terminal last; raw minimum is diagnostic only",
            **score_terminal,
            "raw_terminal_ratios_are_diagnostic_only": True,
            "teacher_entropy_floor": teacher_entropy,
            "initial_reducible_kl_gap": score_initial_excess,
            "terminal_window_reducible_kl_gap": score_terminal_excess,
            "terminal_last_reducible_kl_gap": score_last_excess,
            "terminal_window_reducible_gap_ratio": score_excess_ratio,
            "terminal_last_reducible_gap_ratio": score_last_excess_ratio,
            "required_maximum_reducible_gap_ratio": SCORE_MAXIMUM_RATIO,
            "initial_ranking": initial_ranking,
            "final_ranking": final_ranking,
            "top1_oracle_accuracy_improved": top1_improved,
            "mean_oracle_regret_improved": regret_improved,
            "candidate_branch_frozen": True,
            "gradient_evidence": score_gradient_evidence,
        },
        "loss_contract": loss_contract,
        "candidate_gradient_evidence": gradients,
        "modalities": modality_checks,
        "checkpoint_roundtrip_max_abs_error": maximum_roundtrip_error,
        "execution": {
            "device": str(args.device),
            "torch_threads": args.threads,
            "elapsed_seconds": time.monotonic() - started,
            "reproducibility": reproducibility,
        },
        "production_qualified": False,
        "production_qualification_note": (
            "P4 is an implementation smoke only; P3/P5/P6 qualification is "
            "independent and remains required"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
