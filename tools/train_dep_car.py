#!/usr/bin/env python3
"""P5 two-stage trainer for the formal multimodal DE-P-Car policy.

This entry point deliberately cannot qualify a policy for deployment.  Every
checkpoint and sidecar written here is marked ``UNQUALIFIED``; P6/P8 own the
closed-loop qualification decision.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import math
import os
import random
import sys
import tempfile
import time
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import torch
import yaml
import cv2
from torch.utils.data import DataLoader, Subset


ROOT = Path(__file__).resolve().parents[1]
TRAINER_PATH = Path(__file__).resolve()
sys.path.insert(0, str(ROOT / "dep_car/src"))

from dep_car.model.checkpoint import (
    P4_ARCHITECTURE_ID,
    P4_CONTRACT_SCHEMA,
    sha256_file,
    verify_checkpoint,
)
from dep_car.model.implementation_contract import build_p4_implementation_contract
from dep_car.core.occupancy import (
    FIVE_CIRCLE_FOOTPRINT_SCHEMA,
    SWEPT_INTERPOLATION_SCHEMA,
    SWEPT_SUBSTEPS_PER_SEGMENT,
)
from dep_car.core.lattice import LatticeConfig
from dep_car.model.dep_car_net import DEPCarNetV1
from dep_car.training.losses import (
    DEPCarLossConfig,
    DEPCarLossWeights,
    DEPCarObjectiveV1,
)
from dep_car.training.metrics import CandidateMetricAccumulator, candidate_batch_metrics
from dep_car.training.pilot import PILOT_MANEUVER_MODES
from dep_car.training.p4_dataset import (
    EXPECTED_BEV_PREPROCESSING_SHA256,
    P3TrainingDatasetV1,
    TRAINING_INDEX_SCHEMA,
    TRAINING_VIEW_SCHEMA,
    load_training_index,
    p3_training_collate,
    p3_training_worker_init,
)
from dep_car.training.stages import (
    MODALITY_MODES,
    apply_sensor_dropout,
    configure_training_stage,
    modality_mask,
)


CHECKPOINT_SCHEMA = "DEPCarP5CheckpointV1"
# Training sidecars retain the strict P4 model contract so P6 can load them
# only with an explicit ``allow_untrained=True``.  Extra P5 execution fields do
# not weaken the architecture/input/rollout/transfer checks.
CONTRACT_SCHEMA = P4_CONTRACT_SCHEMA
CHECKPOINT_VERSION = "dep_car_p5_two_stage_unqualified_v1"
TRAINING_STAGES = ("candidate_capacity", "score_calibration")
DEFAULT_SAMPLE_ROOT = ROOT / "data/p3_pilot/run/samples"
DEFAULT_MAPS_ROOT = ROOT / "data/p3_pilot/maps"
DEFAULT_INDEX = ROOT / "data/p3_pilot/run/training_index.json"
DEFAULT_CANDIDATE_INITIALIZATION = (
    ROOT / "models/dep_car/dep_car_net_v1_depth_v483_init.pth"
)
DEFAULT_P3_FOOTPRINT_REAUDIT = ROOT / "reports/p3_development_reaudit_v3.json"
DEFAULT_TRAINING_CONFIG = ROOT / "dep_car/config/training.yaml"
EXPECTED_P3_FOOTPRINT_REAUDIT_SCHEMA = "DEPCarP3DevelopmentReauditV3"
SMOKE_MAX_STEPS = 10
SMOKE_MAX_SAMPLES = 32
FORMAL_INITIALIZATION_VERSION = "dep_car_p4_depth_transfer_initialization_v1"
FORMAL_INITIALIZATION_STATUS = "INITIALIZATION_ONLY_RETRAINING_REQUIRED"


class TrainingConfigurationError(ValueError):
    """Raised before training when stage or provenance constraints are unsafe."""


_EXPECTED_MODEL_STATE_SIGNATURE = None


def _model_state_signature() -> dict:
    global _EXPECTED_MODEL_STATE_SIGNATURE
    if _EXPECTED_MODEL_STATE_SIGNATURE is None:
        # State construction must not perturb the later reproducible training seed.
        with torch.random.fork_rng(devices=[]):
            expected = DEPCarNetV1().state_dict()
        _EXPECTED_MODEL_STATE_SIGNATURE = {
            name: (tuple(value.shape), value.dtype)
            for name, value in expected.items()
        }
    return _EXPECTED_MODEL_STATE_SIGNATURE


def _validate_model_state_dict_structure(state_dict: Mapping[str, Any]) -> None:
    expected = _model_state_signature()
    if set(state_dict) != set(expected):
        missing = sorted(set(expected).difference(state_dict))
        unknown = sorted(set(state_dict).difference(expected))
        raise TrainingConfigurationError(
            "checkpoint model_state_dict keys mismatch "
            f"(missing={missing[:3]}, unknown={unknown[:3]})"
        )
    mismatches = []
    for name, (shape, dtype) in expected.items():
        value = state_dict[name]
        if not torch.is_tensor(value) or tuple(value.shape) != shape or value.dtype != dtype:
            mismatches.append(name)
    if mismatches:
        raise TrainingConfigurationError(
            "checkpoint model_state_dict tensor signature mismatch: "
            + ", ".join(mismatches[:3])
        )


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _authority_path(value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise TrainingConfigurationError(f"training.yaml dataset {name} must be a path")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise TrainingConfigurationError(
            f"training.yaml dataset {name} must be project-relative without parent traversal"
        )
    path = (ROOT / relative).resolve()
    if path != ROOT.resolve() and ROOT.resolve() not in path.parents:
        raise TrainingConfigurationError(
            f"training.yaml dataset {name} escapes the project root"
        )
    return path


def _validate_candidate_acceptance_policy(
    policy: Mapping[str, Any], training: Mapping[str, Any]
) -> None:
    integer_fields = (
        "minimum_completed_epochs",
        "minimum_global_steps",
        "minimum_validation_frames_per_maneuver",
        "minimum_validation_frames_per_requested_gear",
        "minimum_validation_frames_per_candidate_context",
    )
    for name in integer_fields:
        value = policy.get(name)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise TrainingConfigurationError(
                f"training.yaml candidate acceptance {name} must be a positive integer"
            )
    if int(policy["minimum_completed_epochs"]) < int(training["epochs"]):
        raise TrainingConfigurationError(
            "candidate acceptance minimum_completed_epochs may not be below training.epochs"
        )

    numeric_ranges = {
        "maximum_validation_total_loss": (0.0, None, False, False),
        "maximum_validation_zero_feasible_rate": (0.0, 1.0, True, False),
        "minimum_validation_mean_feasible_candidates": (0.0, 15.0, False, True),
        "maximum_validation_kinematic_violation_rate": (0.0, 1.0, True, True),
        "minimum_validation_geometry_valid_fraction": (0.0, 1.0, False, True),
    }
    for name, (lower, upper, lower_inclusive, upper_inclusive) in numeric_ranges.items():
        value = policy.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TrainingConfigurationError(
                f"training.yaml candidate acceptance {name} must be numeric"
            )
        number = float(value)
        lower_ok = number >= lower if lower_inclusive else number > lower
        upper_ok = True if upper is None else (
            number <= upper if upper_inclusive else number < upper
        )
        if not math.isfinite(number) or not lower_ok or not upper_ok:
            bounds = (
                f"{'[' if lower_inclusive else '('}{lower},"
                f"{upper if upper is not None else 'inf'}"
                f"{']' if upper_inclusive else ')'}"
            )
            raise TrainingConfigurationError(
                f"training.yaml candidate acceptance {name} must be in {bounds}"
            )

    required_lists = {
        "required_maneuvers": set(PILOT_MANEUVER_MODES),
        "required_requested_gears": {"FORWARD", "REVERSE"},
        "required_candidate_contexts": {"MISSION", "RECOVERY"},
    }
    for name, expected in required_lists.items():
        values = policy.get(name)
        if (
            not isinstance(values, list)
            or not values
            or any(not isinstance(value, str) or not value for value in values)
            or len(values) != len(set(values))
            or set(values) != expected
        ):
            raise TrainingConfigurationError(
                f"training.yaml candidate acceptance {name} must contain exactly "
                + ", ".join(sorted(expected))
            )


def _training_config_authority() -> dict:
    path = DEFAULT_TRAINING_CONFIG.resolve()
    try:
        config_bytes = path.read_bytes()
        raw = yaml.safe_load(config_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise TrainingConfigurationError(f"cannot read training authority {path}: {exc}") from exc
    if not isinstance(raw, dict) or raw.get("schema") != "DEPCarTrainingContractV1":
        raise TrainingConfigurationError("training.yaml schema mismatch")
    dataset = raw.get("dataset", {})
    training = raw.get("training", {})
    loss = raw.get("loss", {})
    expected_dataset_keys = {
        "root", "maps", "index", "content_aggregate_sha256",
        "map_contract_aggregate_sha256", "schema", "contract_revision",
        "task_manifest_sha256", "required_sensor_authority",
        "lidar_bev_preprocessing_sha256", "split_authority", "workers",
    }
    if set(dataset) != expected_dataset_keys:
        raise TrainingConfigurationError(
            "training.yaml dataset keys are missing or unknown: "
            + ", ".join(
                sorted(set(dataset).symmetric_difference(expected_dataset_keys))
            )
        )
    authority_paths = {
        name: _authority_path(dataset[name], name)
        for name in ("root", "maps", "index")
    }
    for name in (
        "content_aggregate_sha256", "map_contract_aggregate_sha256",
        "task_manifest_sha256", "lidar_bev_preprocessing_sha256",
    ):
        if not _is_sha256(dataset.get(name)):
            raise TrainingConfigurationError(
                f"training.yaml dataset {name} must be a lowercase SHA-256"
            )
    if (
        dataset.get("schema") != "StaticAckermannSampleV2"
        or dataset.get("contract_revision") != 2
        or dataset.get("required_sensor_authority") != "urban_car_depth_vlp16_sim"
        or dataset.get("split_authority") != "map_uuid"
        or not isinstance(dataset.get("workers"), int)
        or isinstance(dataset.get("workers"), bool)
        or int(dataset.get("workers")) < 1
    ):
        raise TrainingConfigurationError("training.yaml dataset authority is invalid")
    expected_training_keys = {
        "default_stage", "stages", "modalities", "epochs", "batch_size",
        "learning_rate", "weight_decay", "gradient_clip",
        "sensor_dropout_probability", "mixed_precision", "seed",
        "torch_threads", "candidate_capacity", "score_calibration",
    }
    if set(training) != expected_training_keys:
        raise TrainingConfigurationError(
            "training.yaml training keys are missing or unknown: "
            + ", ".join(sorted(set(training).symmetric_difference(expected_training_keys)))
        )
    if training.get("stages") != list(TRAINING_STAGES):
        raise TrainingConfigurationError("training.yaml stage order mismatch")
    if set(training.get("modalities", ())) != set(MODALITY_MODES):
        raise TrainingConfigurationError("training.yaml modality set mismatch")
    integer_training_fields = ("epochs", "batch_size", "torch_threads")
    if any(
        not isinstance(training.get(name), int)
        or isinstance(training.get(name), bool)
        or int(training[name]) < 1
        for name in integer_training_fields
    ):
        raise TrainingConfigurationError(
            "training.yaml epochs, batch_size and torch_threads must be positive integers"
        )
    if not isinstance(training.get("seed"), int) or isinstance(
        training.get("seed"), bool
    ):
        raise TrainingConfigurationError("training.yaml seed must be an integer")
    for name in ("learning_rate", "weight_decay", "gradient_clip"):
        value = training.get(name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise TrainingConfigurationError(
                f"training.yaml {name} must be finite and non-negative"
            )
    if float(training["learning_rate"]) <= 0.0:
        raise TrainingConfigurationError("training.yaml learning_rate must be positive")
    dropout = training.get("sensor_dropout_probability")
    if (
        isinstance(dropout, bool)
        or not isinstance(dropout, (int, float))
        or not math.isfinite(float(dropout))
        or not 0.0 <= float(dropout) <= 1.0
    ):
        raise TrainingConfigurationError(
            "training.yaml sensor_dropout_probability must be in [0,1]"
        )
    if not isinstance(training.get("mixed_precision"), bool):
        raise TrainingConfigurationError(
            "training.yaml mixed_precision must be boolean"
        )
    candidate_partition = [
        "depth_encoder", "lidar_encoder", "state_encoder", "gear_embedding",
        "speed_embedding", "steering_embedding", "depth_missing_token",
        "lidar_missing_token", "candidate_tower", "candidate_head",
    ]
    score_partition = ["score_geometry_encoder", "score_tower", "score_head"]
    expected_stage_partitions = {
        "candidate_capacity": {
            "train": candidate_partition,
            "freeze": score_partition,
        },
        "score_calibration": {
            "train": score_partition,
            "freeze": candidate_partition,
        },
    }
    if any(
        training.get(stage) != expected
        for stage, expected in expected_stage_partitions.items()
    ):
        raise TrainingConfigurationError(
            "training.yaml stage parameter partitions do not match model ownership"
        )
    if loss.get("objective_id") != DEPCarObjectiveV1.objective_id:
        raise TrainingConfigurationError("training.yaml objective mismatch")
    expected_loss_metadata = {
        "objective_revision": DEPCarObjectiveV1.objective_revision,
        "sdf_schema": "SignedDistanceFieldV1KnownFreePositiveUnknownUnsafe",
        "footprint_schema": FIVE_CIRCLE_FOOTPRINT_SCHEMA,
        "sweep_interpolation_schema": SWEPT_INTERPOLATION_SCHEMA,
        "sweep_substeps_per_source_segment": SWEPT_SUBSTEPS_PER_SEGMENT,
        "runtime_grid_allowance": "half_cell_diagonal",
        "differentiable_sdf_allowance": "one_cell_diagonal",
    }
    if any(loss.get(name) != value for name, value in expected_loss_metadata.items()):
        raise TrainingConfigurationError("training.yaml safety geometry metadata mismatch")
    loss_field_names = {field.name for field in fields(DEPCarLossConfig)}
    metadata_names = {
        "objective_id", "objective_revision", "sdf_schema",
        "footprint_schema", "sweep_interpolation_schema",
        "sweep_substeps_per_source_segment", "runtime_grid_allowance",
        "differentiable_sdf_allowance",
    }
    expected_loss_keys = loss_field_names | metadata_names
    if set(loss) != expected_loss_keys:
        raise TrainingConfigurationError(
            "training.yaml loss keys are missing or unknown: "
            + ", ".join(sorted(set(loss).symmetric_difference(expected_loss_keys)))
        )
    try:
        weights = DEPCarLossWeights(**loss["weights"])
        loss_values = {
            name: loss[name] for name in loss_field_names.difference({"weights"})
        }
        for name in ("forward_diversity_scales", "reverse_diversity_scales"):
            loss_values[name] = tuple(loss_values[name])
        loss_config = DEPCarLossConfig(weights=weights, **loss_values)
    except (KeyError, TypeError, ValueError) as exc:
        raise TrainingConfigurationError("training.yaml loss contract is invalid") from exc
    loss_config.validate()
    candidate_acceptance = raw.get("qualification", {}).get(
        "candidate_acceptance", {}
    )
    expected_acceptance_keys = {
        "minimum_completed_epochs", "minimum_global_steps",
        "maximum_validation_total_loss", "maximum_validation_zero_feasible_rate",
        "minimum_validation_mean_feasible_candidates",
        "maximum_validation_kinematic_violation_rate",
        "minimum_validation_geometry_valid_fraction",
        "minimum_validation_frames_per_maneuver",
        "minimum_validation_frames_per_requested_gear",
        "minimum_validation_frames_per_candidate_context",
        "required_maneuvers", "required_requested_gears",
        "required_candidate_contexts",
    }
    if set(candidate_acceptance) != expected_acceptance_keys:
        raise TrainingConfigurationError(
            "training.yaml candidate acceptance keys are missing or unknown: "
            + ", ".join(
                sorted(set(candidate_acceptance).symmetric_difference(expected_acceptance_keys))
            )
        )
    _validate_candidate_acceptance_policy(candidate_acceptance, training)
    return {
        "path": path,
        "raw": raw,
        "dataset": dataset,
        "authority_paths": authority_paths,
        "training": training,
        "loss_config": loss_config,
        "file_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "semantic_sha256": _canonical_sha256(raw),
        "loss_config_sha256": _canonical_sha256(asdict(loss_config)),
    }


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _nonnegative_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be a non-negative integer")
    return parsed


def _probability(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("value must be in [0,1]")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    authority = _training_config_authority()
    training_defaults = authority["training"]
    parser = argparse.ArgumentParser(
        description=(
            "Train DEPCarNetV1 on the sealed P3 train/validation view. "
            "This command never reads the test split and never signs a policy."
        )
    )
    parser.add_argument(
        "--stage",
        choices=TRAINING_STAGES,
        default=training_defaults["default_stage"],
        help="candidate capacity must be trained before score calibration",
    )
    parser.add_argument(
        "--modality", choices=MODALITY_MODES, default="fusion"
    )
    parser.add_argument(
        "--data",
        "--sample-root",
        dest="data",
        type=Path,
        default=authority["authority_paths"]["root"],
        help="P3 NPZ sample root",
    )
    parser.add_argument(
        "--maps", type=Path, default=authority["authority_paths"]["maps"]
    )
    parser.add_argument(
        "--index", type=Path, default=authority["authority_paths"]["index"]
    )
    parser.add_argument("--output", type=Path, required=True)
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--init",
        type=Path,
        help=(
            "candidate stage: formal depth-v4.8.3 initialization; score stage: "
            "an explicitly trained candidate_capacity checkpoint"
        ),
    )
    source.add_argument(
        "--resume", type=Path, help="resume a checkpoint from the same stage/modality"
    )
    parser.add_argument("--epochs", type=_positive_integer, default=training_defaults["epochs"])
    parser.add_argument("--batch-size", type=_positive_integer, default=training_defaults["batch_size"])
    parser.add_argument("--learning-rate", type=float, default=training_defaults["learning_rate"])
    parser.add_argument("--weight-decay", type=float, default=training_defaults["weight_decay"])
    parser.add_argument(
        "--gradient-clip", type=float, default=training_defaults["gradient_clip"]
    )
    parser.add_argument(
        "--workers",
        type=_nonnegative_integer,
        default=authority["raw"]["dataset"]["workers"],
        help="DataLoader workers (formal default: 8)",
    )
    parser.add_argument(
        "--torch-threads",
        type=_positive_integer,
        default=training_defaults["torch_threads"],
        help="in-process CPU threads; DataLoader workers remain separate",
    )
    parser.add_argument("--seed", type=int, default=training_defaults["seed"])
    parser.add_argument(
        "--sensor-dropout-probability",
        type=_probability,
        default=training_defaults["sensor_dropout_probability"],
        help="fusion-stage training only; never removes both sensors",
    )
    parser.add_argument(
        "--max-samples",
        type=_positive_integer,
        help="cap each of train and validation for an explicitly unqualified smoke run",
    )
    parser.add_argument(
        "--max-steps",
        type=_positive_integer,
        help="cap optimizer steps for an explicitly unqualified smoke run",
    )
    parser.add_argument(
        "--device", default="auto", help="auto, cpu, cuda, or an explicit torch device"
    )
    amp = parser.add_mutually_exclusive_group()
    amp.add_argument("--amp", dest="amp", action="store_true")
    amp.add_argument("--no-amp", dest="amp", action="store_false")
    parser.set_defaults(amp=bool(training_defaults["mixed_precision"]))
    parser.add_argument(
        "--allow-smoke-source",
        action="store_true",
        help="score smoke only: accept a non-accepted candidate smoke checkpoint",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate split/index/checkpoint lineage and print the plan without training",
    )
    return parser


def _validate_numeric_arguments(args: argparse.Namespace) -> None:
    for name in ("learning_rate", "weight_decay", "gradient_clip"):
        value = float(getattr(args, name))
        if not np.isfinite(value) or value < 0.0:
            raise TrainingConfigurationError(f"--{name.replace('_', '-')} must be finite and non-negative")
    if args.learning_rate == 0.0:
        raise TrainingConfigurationError("--learning-rate must be positive")
    if args.output.suffix != ".pth":
        raise TrainingConfigurationError("--output must end in .pth")
    if args.allow_smoke_source and args.stage != "score_calibration":
        raise TrainingConfigurationError(
            "--allow-smoke-source is valid only for score_calibration"
        )
    if args.allow_smoke_source and not _explicit_smoke_run(args):
        raise TrainingConfigurationError(
            "--allow-smoke-source requires a permanently marked current smoke run"
        )


def _contract_path(checkpoint: Path) -> Path:
    return checkpoint.with_suffix(".contract.json")


def _candidate_acceptance_path(checkpoint: Path) -> Path:
    return checkpoint.with_suffix(".candidate_acceptance.json")


def _reject_output_source_collision(output: Path, source: Path) -> None:
    """Reject checkpoint or sidecar aliasing before any source deserialization."""

    output = Path(output).resolve()
    source = Path(source).resolve()
    output_paths = {path.resolve() for path in _artifact_paths(output).values()}
    output_paths.add(_candidate_acceptance_path(output).resolve())
    source_paths = {path.resolve() for path in _artifact_paths(source).values()}
    source_paths.add(_candidate_acceptance_path(source).resolve())
    collisions = sorted(str(path) for path in output_paths.intersection(source_paths))
    if collisions:
        raise TrainingConfigurationError(
            "--output artifacts must not overwrite the init/resume source or its "
            "sidecars (" + ", ".join(collisions) + ")"
        )


def _reject_existing_output_artifacts(output: Path) -> None:
    output = Path(output).resolve()
    paths = {path.resolve() for path in _artifact_paths(output).values()}
    paths.add(_candidate_acceptance_path(output).resolve())
    existing = sorted(str(path) for path in paths if path.exists())
    if existing:
        raise TrainingConfigurationError(
            "output artifacts already exist; refusing implicit overwrite ("
            + ", ".join(existing)
            + ")"
        )


def _load_json(path: Path) -> dict:
    value, _raw = _load_json_exact(path)
    return value


def _load_json_exact(path: Path) -> tuple:
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TrainingConfigurationError(f"cannot read JSON contract {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise TrainingConfigurationError(f"JSON contract root must be a mapping: {path}")
    return value, raw


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _map_contract_aggregate(
    maps_root: Path, index: Mapping[str, Any]
) -> dict:
    """Hash the PNG bytes and metric YAML semantics for indexed maps only."""

    maps_root = Path(maps_root).resolve()
    required = {str(entry.get("map_uuid", "")) for entry in index.get("entries", ())}
    if not required or "" in required:
        raise TrainingConfigurationError("training index has no map UUID authority")
    folders = {}
    # Generated map folders embed the first UUID octet.  Filter by that public
    # folder name before opening manifests so sealed test-map metadata is not
    # inspected merely to discover that it is outside the training index.
    required_prefixes = {map_uuid[:8] for map_uuid in required}
    candidate_folders = [
        path for path in maps_root.iterdir()
        if path.is_dir()
        and any(prefix in path.name for prefix in required_prefixes)
    ]
    for folder in sorted(candidate_folders):
        manifest_path = folder / "manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = _load_json(manifest_path)
        map_uuid = str(manifest.get("map_uuid", ""))
        if map_uuid not in required:
            # Do not open a sealed test map's YAML or PNG.
            continue
        if map_uuid in folders:
            raise TrainingConfigurationError(
                f"duplicate map UUID in map authority: {map_uuid}"
            )
        folders[map_uuid] = (folder.resolve(), manifest)
    missing = sorted(required.difference(folders))
    if missing:
        raise TrainingConfigurationError(
            "indexed maps are missing from map authority: " + ", ".join(missing[:3])
        )

    rows = []
    for map_uuid in sorted(required):
        folder, manifest = folders[map_uuid]
        yaml_path = folder / "map.yaml"
        try:
            yaml_bytes = yaml_path.read_bytes()
            metadata = yaml.safe_load(yaml_bytes.decode("utf-8"))
        except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
            raise TrainingConfigurationError(
                f"cannot read indexed map YAML {yaml_path}: {exc}"
            ) from exc
        if not isinstance(metadata, dict):
            raise TrainingConfigurationError(f"map YAML root must be a mapping: {yaml_path}")
        try:
            image_name = str(metadata["image"])
            image_path = (folder / image_name).resolve()
            origin = [float(value) for value in metadata["origin"]]
            semantic = {
                "image": image_name,
                "resolution": float(metadata["resolution"]),
                "origin": origin,
                "negate": int(metadata.get("negate", 0)),
                "occupied_thresh": float(metadata.get("occupied_thresh", 0.65)),
                "free_thresh": float(metadata.get("free_thresh", 0.196)),
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise TrainingConfigurationError(
                f"indexed map YAML semantics are invalid: {yaml_path}"
            ) from exc
        if (
            image_path.parent != folder
            or not image_path.is_file()
            or len(origin) < 3
            or not all(math.isfinite(value) for value in origin)
            or abs(float(origin[2])) > 1.0e-9
            or not math.isfinite(semantic["resolution"])
            or semantic["resolution"] <= 0.0
            or semantic["negate"] not in (0, 1)
            or not 0.0 <= semantic["free_thresh"] < semantic["occupied_thresh"] <= 1.0
        ):
            raise TrainingConfigurationError(
                f"indexed map contract is invalid: {folder}"
            )
        occupancy_sha256 = manifest.get("occupancy_sha256")
        if not _is_sha256(occupancy_sha256):
            raise TrainingConfigurationError(
                f"indexed map occupancy SHA-256 is invalid: {folder}"
            )
        try:
            image_bytes = image_path.read_bytes()
        except OSError as exc:
            raise TrainingConfigurationError(
                f"cannot read indexed map PNG {image_path}: {exc}"
            ) from exc
        pixels = cv2.imdecode(
            np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_GRAYSCALE
        )
        if pixels is None:
            raise TrainingConfigurationError(
                f"indexed map PNG is unreadable: {image_path}"
            )
        decoded_occupancy_sha256 = hashlib.sha256(
            np.asarray(pixels, dtype=np.uint8).tobytes()
        ).hexdigest()
        if decoded_occupancy_sha256 != occupancy_sha256:
            raise TrainingConfigurationError(
                f"indexed map decoded occupancy SHA-256 mismatch: {folder}"
            )
        rows.append({
            "map_uuid": map_uuid,
            "png_sha256": hashlib.sha256(image_bytes).hexdigest(),
            "occupancy_sha256": decoded_occupancy_sha256,
            "yaml": semantic,
        })
    return {
        "schema": "IndexedMapContractAggregateV1",
        "map_count": len(rows),
        "aggregate_sha256": _canonical_sha256(rows),
    }


def _dataset_authority_gate(
    args: argparse.Namespace,
    training_config: Mapping[str, Any],
    index: Mapping[str, Any],
    map_contract: Mapping[str, Any],
) -> dict:
    configured_paths = training_config["authority_paths"]
    actual_paths = {
        "root": args.data.resolve(),
        "maps": args.maps.resolve(),
        "index": args.index.resolve(),
    }
    path_mismatches = sorted(
        name for name in configured_paths
        if actual_paths[name] != configured_paths[name]
    )
    dataset = training_config["dataset"]
    errors = ["dataset_authority_path_" + name for name in path_mismatches]
    if index.get("content_aggregate_sha256") != dataset["content_aggregate_sha256"]:
        errors.append("dataset_authority_content_aggregate")
    if map_contract.get("aggregate_sha256") != dataset["map_contract_aggregate_sha256"]:
        errors.append("dataset_authority_map_contract_aggregate")
    return {
        "passed": not errors,
        "errors": errors,
        "path_overrides": path_mismatches,
        "configured_paths": {
            name: str(path) for name, path in configured_paths.items()
        },
        "actual_paths": {name: str(path) for name, path in actual_paths.items()},
        "expected_content_aggregate_sha256": dataset["content_aggregate_sha256"],
        "actual_content_aggregate_sha256": index.get("content_aggregate_sha256"),
        "expected_map_contract_aggregate_sha256": dataset[
            "map_contract_aggregate_sha256"
        ],
        "actual_map_contract_aggregate_sha256": map_contract.get("aggregate_sha256"),
        "map_count": map_contract.get("map_count"),
    }


def _validation_coverage_gate(
    index: Mapping[str, Any], training_config: Mapping[str, Any]
) -> dict:
    """Fail fast on coverage facts already sealed into the validation index.

    Requested gear was not part of every historical P3 index entry.  It stays
    deferred to the loader/acceptance metric gate unless every validation entry
    carries a valid scalar; no NPZ (and especially no sealed test NPZ) is opened
    to fill that historical gap.
    """

    policy = training_config["raw"]["qualification"]["candidate_acceptance"]
    entries = [
        entry for entry in index.get("entries", ())
        if entry.get("split") == "validation"
    ]
    maneuver_counts = Counter(str(entry.get("maneuver_mode", "UNKNOWN")) for entry in entries)
    context_counts = Counter(
        str(entry.get("candidate_context", "UNKNOWN")) for entry in entries
    )
    errors = []

    def required_rows(required, counts, minimum, prefix):
        rows = {}
        for name in required:
            observed = int(counts.get(name, 0))
            passed = observed >= int(minimum)
            rows[name] = {
                "frames": observed,
                "minimum_frames": int(minimum),
                "passed": passed,
            }
            if not passed:
                errors.append(
                    f"validation_{prefix}_{name}_frames_{observed}_lt_{int(minimum)}"
                )
        return rows

    maneuver_rows = required_rows(
        policy["required_maneuvers"],
        maneuver_counts,
        policy["minimum_validation_frames_per_maneuver"],
        "maneuver",
    )
    context_rows = required_rows(
        policy["required_candidate_contexts"],
        context_counts,
        policy["minimum_validation_frames_per_candidate_context"],
        "candidate_context",
    )
    allowed_contexts = set(policy["required_candidate_contexts"])
    unexpected_contexts = {
        name: int(count)
        for name, count in context_counts.items()
        if name not in allowed_contexts and int(count) > 0
    }
    errors.extend(
        f"validation_candidate_context_unexpected_{name}_frames_{count}"
        for name, count in sorted(unexpected_contexts.items())
    )

    normalized_gears = []
    for entry in entries:
        raw = entry.get("requested_gear")
        if raw in (1, "1", "FORWARD"):
            normalized_gears.append("FORWARD")
        elif raw in (-1, "-1", "REVERSE"):
            normalized_gears.append("REVERSE")
        else:
            normalized_gears.append(None)
    gear_available = bool(entries) and all(value is not None for value in normalized_gears)
    gear_counts = Counter(value for value in normalized_gears if value is not None)
    if gear_available:
        gear_rows = required_rows(
            policy["required_requested_gears"],
            gear_counts,
            policy["minimum_validation_frames_per_requested_gear"],
            "requested_gear",
        )
        gear_status = "CHECKED_FROM_INDEX"
    else:
        gear_rows = {
            name: {
                "frames": int(gear_counts.get(name, 0)),
                "minimum_frames": int(
                    policy["minimum_validation_frames_per_requested_gear"]
                ),
                "passed": None,
            }
            for name in policy["required_requested_gears"]
        }
        gear_status = "DEFERRED_TO_LOADER_AND_CANDIDATE_ACCEPTANCE"

    return {
        "schema": "DEPCarP5ValidationCoverageGateV1",
        "passed": not errors,
        "validation_frames": len(entries),
        "errors": sorted(errors),
        "maneuver": {
            "counts": dict(sorted(maneuver_counts.items())),
            "required": maneuver_rows,
        },
        "candidate_context": {
            "counts": dict(sorted(context_counts.items())),
            "required": context_rows,
            "unexpected": unexpected_contexts,
        },
        "requested_gear": {
            "status": gear_status,
            "indexed_frames": len(normalized_gears) - normalized_gears.count(None),
            "counts": dict(sorted(gear_counts.items())),
            "required": gear_rows,
        },
        "test_samples_opened": False,
    }


def _load_checkpoint(path: Path, checkpoint_bytes: Optional[bytes] = None) -> dict:
    if checkpoint_bytes is None:
        if not path.is_file():
            raise TrainingConfigurationError(f"checkpoint does not exist: {path}")
        try:
            checkpoint_bytes = path.read_bytes()
        except OSError as exc:
            raise TrainingConfigurationError(
                f"cannot read checkpoint bytes {path}: {exc}"
            ) from exc
    if not isinstance(checkpoint_bytes, bytes):
        raise TrainingConfigurationError("checkpoint loader requires exact bytes")
    try:
        payload = torch.load(
            io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=True
        )
    except Exception as exc:
        raise TrainingConfigurationError(f"cannot load checkpoint {path}: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("model_state_dict"), dict):
        raise TrainingConfigurationError(f"checkpoint has no model_state_dict: {path}")
    if not payload["model_state_dict"]:
        raise TrainingConfigurationError(f"checkpoint model_state_dict is empty: {path}")
    _validate_model_state_dict_structure(payload["model_state_dict"])
    if payload.get("architecture_id") != P4_ARCHITECTURE_ID:
        raise TrainingConfigurationError(
            "checkpoint architecture mismatch; the retired LiDAR/range-image model is forbidden"
        )
    if payload.get("production_qualified") is not False:
        raise TrainingConfigurationError(
            "training source must be explicitly production_qualified=false"
        )
    return payload


def _verify_sidecar_identity(
    path: Path, *, expected_schema: Optional[str] = None
) -> tuple:
    sidecar_path = _contract_path(path)
    if not sidecar_path.is_file():
        raise TrainingConfigurationError(f"checkpoint sidecar is missing: {sidecar_path}")
    contract, contract_bytes = _load_json_exact(sidecar_path)
    if contract.get("architecture_id") != P4_ARCHITECTURE_ID:
        raise TrainingConfigurationError("checkpoint sidecar architecture mismatch")
    if expected_schema is not None and contract.get("schema") != expected_schema:
        raise TrainingConfigurationError(
            f"checkpoint sidecar schema must be {expected_schema}"
        )
    try:
        checkpoint_bytes = path.read_bytes()
    except OSError as exc:
        raise TrainingConfigurationError(
            f"cannot read checkpoint bytes {path}: {exc}"
        ) from exc
    checkpoint_sha256 = hashlib.sha256(checkpoint_bytes).hexdigest()
    if contract.get("checkpoint_sha256") != checkpoint_sha256:
        raise TrainingConfigurationError("checkpoint SHA-256 does not match its sidecar")
    if contract.get("production_qualified") is not False:
        raise TrainingConfigurationError("training source sidecar must be explicitly unqualified")
    try:
        verified = verify_checkpoint(
            path,
            sidecar_path,
            allow_untrained=True,
            checkpoint_bytes=checkpoint_bytes,
            contract_document=contract,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        raise TrainingConfigurationError(
            f"checkpoint full identity verification failed: {exc}"
        ) from exc
    if verified != contract:
        raise TrainingConfigurationError("checkpoint verifier returned a different contract")
    return contract, checkpoint_bytes, hashlib.sha256(contract_bytes).hexdigest()


def _safe_candidate_artifact(checkpoint: Path, filename: Any) -> Path:
    if not isinstance(filename, str) or not filename:
        raise TrainingConfigurationError("candidate metrics artifact name is missing")
    path = (checkpoint.parent / filename).resolve()
    if path.parent != checkpoint.parent.resolve():
        raise TrainingConfigurationError("candidate artifact path escapes checkpoint folder")
    return path


def evaluate_candidate_acceptance(
    checkpoint: Path, *, authority: Optional[Mapping[str, Any]] = None
) -> dict:
    """Recompute the complete candidate gate from immutable candidate artifacts.

    This function is shared by the external acceptance CLI and the score-stage
    loader.  A handwritten ``PASS`` JSON therefore has no authority of its own.
    """

    checkpoint = Path(checkpoint).resolve()
    authority = _training_config_authority() if authority is None else authority
    thresholds = authority["raw"].get("qualification", {}).get(
        "candidate_acceptance", {}
    )
    contract_path = _contract_path(checkpoint)
    contract, checkpoint_bytes, contract_sha256 = _verify_sidecar_identity(
        checkpoint, expected_schema=P4_CONTRACT_SCHEMA
    )
    # Full identity verification hashes the file before trusted-local pickle
    # metadata is deserialized by the training-specific loader.
    payload = _load_checkpoint(checkpoint, checkpoint_bytes)
    errors = []

    def check(condition: bool, name: str) -> None:
        if not condition:
            errors.append(name)

    check(payload.get("schema") == CHECKPOINT_SCHEMA, "checkpoint_schema")
    check(payload.get("training_stage") == "candidate_capacity", "training_stage")
    check(contract.get("training_stage") == "candidate_capacity", "contract_stage")
    check(payload.get("status") == "TRAINED_UNQUALIFIED", "checkpoint_status")
    check(payload.get("qualification_status") == "UNQUALIFIED", "qualification_status")
    check(payload.get("production_qualified") is False, "checkpoint_qualification")
    check(not bool(payload.get("smoke_lineage", True)), "smoke_lineage")
    check(not bool(payload.get("partial_epoch", True)), "partial_epoch")
    training_run = contract.get("training_run", {})
    check(not bool(training_run.get("smoke_limited", True)), "contract_smoke")
    check(not bool(training_run.get("smoke_lineage", True)), "contract_smoke_lineage")
    check(not bool(training_run.get("partial_epoch", True)), "contract_partial")

    try:
        completed_epochs = int(payload.get("completed_epochs", 0))
        global_step = int(payload.get("global_step", 0))
    except (TypeError, ValueError):
        completed_epochs = global_step = -1
    check(
        completed_epochs >= int(thresholds["minimum_completed_epochs"]),
        "minimum_completed_epochs",
    )
    check(
        global_step >= int(thresholds["minimum_global_steps"]),
        "minimum_global_steps",
    )

    implementation = build_p4_implementation_contract(ROOT)
    expected_hashes = {
        "training_config_sha256": authority["file_sha256"],
        "loss_config_sha256": authority["loss_config_sha256"],
        "trainer_sha256": sha256_file(TRAINER_PATH),
        "implementation_aggregate_sha256": implementation["aggregate_sha256"],
    }
    for name, expected in expected_hashes.items():
        check(payload.get(name) == expected, name)
        check(training_run.get(name) == expected, "contract_" + name)
    check(
        contract.get("implementation_contract") == implementation,
        "implementation_contract",
    )

    dataset_authority = contract.get("dataset_authority_gate", {})
    check(dataset_authority.get("passed") is True, "dataset_authority_gate")
    check(
        contract.get("validation_coverage_gate", {}).get("passed") is True,
        "validation_coverage_gate",
    )
    check(contract.get("p3_footprint_gate", {}).get("passed") is True, "p3_footprint_gate")
    check(contract.get("index_content_gate", {}).get("passed") is True, "index_content_gate")
    check(
        contract.get("training_yaml_qualification_gate", {}).get("passed") is True,
        "contract_training_yaml_formal_gate",
    )
    qualification = authority["raw"].get("qualification", {})
    check(
        qualification.get("corrected_footprint_p3_status") == "PASS"
        and qualification.get("p5_formal_training_allowed") is True
        and not qualification.get("blocked_gates"),
        "training_yaml_formal_gate",
    )
    configured_dataset = authority["dataset"]
    dataset_provenance = contract.get("dataset_provenance", {})
    expected_index_path = authority["authority_paths"]["index"]
    current_index_sha256 = (
        sha256_file(expected_index_path) if expected_index_path.is_file() else None
    )
    check(
        payload.get("training_index_sha256") == current_index_sha256,
        "training_index_sha256",
    )
    check(
        payload.get("training_index_content_sha256")
        == configured_dataset["content_aggregate_sha256"],
        "training_index_content_sha256",
    )
    check(
        payload.get("map_contract_aggregate_sha256")
        == configured_dataset["map_contract_aggregate_sha256"],
        "map_contract_aggregate_sha256",
    )
    check(
        dataset_provenance.get("index_sha256") == current_index_sha256,
        "contract_training_index_sha256",
    )
    check(
        dataset_provenance.get("content_aggregate_sha256")
        == configured_dataset["content_aggregate_sha256"],
        "contract_content_aggregate_sha256",
    )
    check(
        dataset_provenance.get("map_contract_aggregate_sha256")
        == configured_dataset["map_contract_aggregate_sha256"],
        "contract_map_contract_aggregate_sha256",
    )

    artifacts = contract.get("artifacts", {})
    try:
        metrics_path = _safe_candidate_artifact(checkpoint, artifacts.get("metrics"))
    except TrainingConfigurationError:
        metrics_path = checkpoint.parent / "__missing_candidate_metrics__"
        errors.append("metrics_path")
    check(metrics_path.is_file(), "metrics_missing")
    if metrics_path.is_file():
        metrics_document, metrics_bytes = _load_json_exact(metrics_path)
        metrics_sha256 = hashlib.sha256(metrics_bytes).hexdigest()
    else:
        metrics_document = {}
        metrics_sha256 = None
    check(artifacts.get("metrics_sha256") == metrics_sha256, "metrics_sha256")
    check(metrics_document.get("schema") == "DEPCarP5TrainingMetricsV1", "metrics_schema")
    check(metrics_document.get("architecture_id") == P4_ARCHITECTURE_ID, "metrics_architecture")
    check(metrics_document.get("training_stage") == "candidate_capacity", "metrics_stage")
    check(metrics_document.get("modality") == payload.get("modality"), "metrics_modality")
    check(metrics_document.get("qualification_status") == "UNQUALIFIED", "metrics_qualification")
    check(metrics_document.get("production_qualified") is False, "metrics_production_qualified")
    check(metrics_document.get("completed_epochs") == completed_epochs, "metrics_completed_epochs")
    check(metrics_document.get("global_step") == global_step, "metrics_global_step")
    check(metrics_document.get("partial_epoch") is False, "metrics_partial_epoch")
    check(metrics_document.get("metrics") == payload.get("metrics"), "checkpoint_metrics_mismatch")

    validation = metrics_document.get("metrics", {}).get("validation", {})
    candidate = validation.get("candidate_metrics", {})
    overall = candidate.get("overall", {})
    observed = {
        "validation_total_loss": validation.get("total"),
        "validation_geometry_valid_fraction": validation.get("geometry_valid_fraction"),
        "validation_zero_feasible_rate": overall.get("zero_feasible_rate"),
        "validation_mean_feasible_candidates": overall.get("mean_feasible_candidates"),
        "validation_kinematic_violation_rate": overall.get("kinematic_violation_rate"),
    }

    def finite(name: str) -> bool:
        value = observed.get(name)
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
        )

    comparisons = (
        ("validation_total_loss", "maximum_validation_total_loss", lambda value, limit: value <= limit),
        ("validation_geometry_valid_fraction", "minimum_validation_geometry_valid_fraction", lambda value, limit: value >= limit),
        ("validation_zero_feasible_rate", "maximum_validation_zero_feasible_rate", lambda value, limit: value < limit),
        ("validation_mean_feasible_candidates", "minimum_validation_mean_feasible_candidates", lambda value, limit: value >= limit),
        ("validation_kinematic_violation_rate", "maximum_validation_kinematic_violation_rate", lambda value, limit: value <= limit),
    )
    for observed_name, threshold_name, comparator in comparisons:
        check(
            finite(observed_name)
            and comparator(float(observed[observed_name]), float(thresholds[threshold_name])),
            threshold_name,
        )

    coverage = {}
    groups = (
        ("maneuver", candidate.get("by_maneuver", {}), thresholds["required_maneuvers"], "minimum_validation_frames_per_maneuver"),
        ("requested_gear", candidate.get("by_requested_gear", {}), thresholds["required_requested_gears"], "minimum_validation_frames_per_requested_gear"),
        ("candidate_context", candidate.get("by_candidate_context", {}), thresholds["required_candidate_contexts"], "minimum_validation_frames_per_candidate_context"),
    )
    overall_frames = int(overall.get("frames", 0)) if isinstance(overall.get("frames", 0), int) else 0
    for group_name, rows, required, minimum_name in groups:
        if not isinstance(rows, dict):
            rows = {}
            errors.append(group_name + "_metrics_missing")
        coverage[group_name] = {}
        total_frames = 0
        for value, row in rows.items():
            frames = row.get("frames", 0) if isinstance(row, dict) else None
            if (
                not isinstance(frames, int)
                or isinstance(frames, bool)
                or frames < 0
            ):
                errors.append(f"{group_name}_coverage_invalid_{value}")
                continue
            total_frames += frames
            if (
                group_name == "candidate_context"
                and value not in set(required)
                and frames > 0
            ):
                errors.append(f"candidate_context_unexpected_{value}")
        for value in required:
            row = rows.get(value, {})
            frames = row.get("frames", 0)
            frames = int(frames) if isinstance(frames, int) and not isinstance(frames, bool) else 0
            coverage[group_name][value] = frames
            check(
                frames >= int(thresholds[minimum_name]),
                f"{group_name}_coverage_{value}",
            )
            group_checks = (
                (
                    "zero_feasible_rate",
                    "maximum_validation_zero_feasible_rate",
                    lambda observed_value, limit: observed_value < limit,
                ),
                (
                    "mean_feasible_candidates",
                    "minimum_validation_mean_feasible_candidates",
                    lambda observed_value, limit: observed_value >= limit,
                ),
                (
                    "kinematic_violation_rate",
                    "maximum_validation_kinematic_violation_rate",
                    lambda observed_value, limit: observed_value <= limit,
                ),
            )
            for metric_name, threshold_name, comparator in group_checks:
                metric = row.get(metric_name)
                valid_metric = (
                    isinstance(metric, (int, float))
                    and not isinstance(metric, bool)
                    and math.isfinite(float(metric))
                )
                check(
                    valid_metric
                    and comparator(float(metric), float(thresholds[threshold_name])),
                    f"{group_name}_{value}_{threshold_name}",
                )
        check(total_frames == overall_frames, group_name + "_coverage_total")

    errors = sorted(set(errors))
    result = {
        "schema": "DEPCarP5CandidateAcceptanceV1",
        "status": "PASS" if not errors else "FAIL",
        "gate_passed": not errors,
        "training_stage": "candidate_capacity",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint_bytes).hexdigest(),
        "contract_sha256": contract_sha256,
        "metrics_sha256": metrics_sha256,
        "training_config_sha256": authority["file_sha256"],
        "loss_config_sha256": authority["loss_config_sha256"],
        "trainer_sha256": sha256_file(TRAINER_PATH),
        "implementation_aggregate_sha256": implementation["aggregate_sha256"],
        "acceptance_tool_sha256": sha256_file(ROOT / "tools/accept_p5_candidate.py"),
        "training_index_sha256": payload.get("training_index_sha256"),
        "training_index_content_sha256": payload.get("training_index_content_sha256"),
        "map_contract_aggregate_sha256": dataset_provenance.get(
            "map_contract_aggregate_sha256"
        ),
        "smoke_limited": bool(payload.get("smoke_lineage", True)),
        "partial_epoch": bool(payload.get("partial_epoch", True)),
        "thresholds": copy.deepcopy(thresholds),
        "observed": observed,
        "coverage": coverage,
        "errors": errors,
    }
    result["evaluation_sha256"] = _canonical_sha256(result)
    return result


def _verify_candidate_acceptance(path: Path, contract_path: Path) -> dict:
    acceptance_path = _candidate_acceptance_path(path)
    if not acceptance_path.is_file():
        raise TrainingConfigurationError(
            "score_calibration requires an externally written candidate acceptance sidecar: "
            f"{acceptance_path}"
        )
    recorded = _load_json(acceptance_path)
    live = evaluate_candidate_acceptance(path)
    compared_fields = set(live).difference({"checkpoint"})
    mismatches = sorted(
        name for name in compared_fields if recorded.get(name) != live.get(name)
    )
    if not live["gate_passed"]:
        mismatches.extend("live:" + error for error in live["errors"])
    if recorded.get("status") != "PASS" or recorded.get("gate_passed") is not True:
        mismatches.append("recorded_gate_status")
    if mismatches:
        raise TrainingConfigurationError(
            "candidate acceptance gate failed live recomputation ("
            + ", ".join(sorted(set(mismatches)))
            + ")"
        )
    return recorded


def _inspect_source(args: argparse.Namespace) -> dict:
    """Resolve and validate the only legal source for a training stage."""

    if args.resume is not None:
        path = args.resume.resolve()
        _reject_output_source_collision(args.output, path)
        contract, checkpoint_bytes, contract_sha256 = _verify_sidecar_identity(
            path, expected_schema=CONTRACT_SCHEMA
        )
        payload = _load_checkpoint(path, checkpoint_bytes)
        if payload.get("schema") != CHECKPOINT_SCHEMA:
            raise TrainingConfigurationError("--resume requires a P5 training checkpoint")
        if (
            payload.get("status") != "TRAINED_UNQUALIFIED"
            or payload.get("qualification_status") != "UNQUALIFIED"
        ):
            raise TrainingConfigurationError("resume checkpoint is not an unqualified P5 artifact")
        if payload.get("training_stage") != args.stage:
            raise TrainingConfigurationError("--resume checkpoint stage does not match --stage")
        if payload.get("modality") != args.modality:
            raise TrainingConfigurationError("--resume checkpoint modality does not match --modality")
        if contract.get("training_stage") != args.stage:
            raise TrainingConfigurationError("resume sidecar stage mismatch")
        if contract.get("modality") != args.modality:
            raise TrainingConfigurationError("resume sidecar modality mismatch")
        if not isinstance(payload.get("optimizer_state_dict"), dict):
            raise TrainingConfigurationError("resume checkpoint has no optimizer state")
        return {
            "kind": "resume",
            "path": path,
            "payload": payload,
            "contract": contract,
            "checkpoint_sha256": hashlib.sha256(checkpoint_bytes).hexdigest(),
            "contract_sha256": contract_sha256,
        }

    if args.stage == "candidate_capacity":
        path = (args.init or DEFAULT_CANDIDATE_INITIALIZATION).resolve()
        _reject_output_source_collision(args.output, path)
        contract, checkpoint_bytes, contract_sha256 = _verify_sidecar_identity(
            path, expected_schema=P4_CONTRACT_SCHEMA
        )
        payload = _load_checkpoint(path, checkpoint_bytes)
        if payload.get("checkpoint_version") != FORMAL_INITIALIZATION_VERSION:
            raise TrainingConfigurationError(
                "candidate_capacity --init must be the formal depth-v4.8.3 P4 initialization; "
                "use --resume for a trained candidate checkpoint"
            )
        if payload.get("status") != FORMAL_INITIALIZATION_STATUS:
            raise TrainingConfigurationError("candidate initialization status is invalid")
        if contract.get("status") != FORMAL_INITIALIZATION_STATUS:
            raise TrainingConfigurationError("candidate initialization sidecar status is invalid")
        return {
            "kind": "formal_depth_v483_initialization",
            "path": path,
            "payload": payload,
            "contract": contract,
            "checkpoint_sha256": hashlib.sha256(checkpoint_bytes).hexdigest(),
            "contract_sha256": contract_sha256,
        }

    if args.init is None:
        raise TrainingConfigurationError(
            "score_calibration requires explicit --init PATH to a trained "
            "candidate_capacity checkpoint (or --resume for the same score run)"
        )
    path = args.init.resolve()
    _reject_output_source_collision(args.output, path)
    contract, checkpoint_bytes, contract_sha256 = _verify_sidecar_identity(
        path, expected_schema=CONTRACT_SCHEMA
    )
    payload = _load_checkpoint(path, checkpoint_bytes)
    if payload.get("schema") != CHECKPOINT_SCHEMA:
        raise TrainingConfigurationError(
            "score_calibration may not start from the transfer initialization"
        )
    if (
        payload.get("status") != "TRAINED_UNQUALIFIED"
        or payload.get("qualification_status") != "UNQUALIFIED"
    ):
        raise TrainingConfigurationError("score source is not an unqualified P5 checkpoint")
    if payload.get("training_stage") != "candidate_capacity":
        raise TrainingConfigurationError(
            "score_calibration --init must be a candidate_capacity checkpoint"
        )
    if contract.get("training_stage") != "candidate_capacity":
        raise TrainingConfigurationError("candidate checkpoint sidecar stage mismatch")
    if payload.get("modality") != args.modality or contract.get("modality") != args.modality:
        raise TrainingConfigurationError(
            "score_calibration modality must match its candidate checkpoint"
        )
    if int(payload.get("global_step", 0)) < 1:
        raise TrainingConfigurationError("candidate checkpoint contains no completed optimizer step")
    candidate_is_smoke = bool(
        payload.get("partial_epoch")
        or payload.get("smoke_lineage", True)
        or contract.get("training_run", {}).get("smoke_limited", True)
        or contract.get("training_run", {}).get("smoke_lineage", True)
    )
    acceptance = None
    if args.allow_smoke_source:
        if not candidate_is_smoke:
            raise TrainingConfigurationError(
                "--allow-smoke-source is unnecessary for a formal candidate source"
            )
    else:
        if candidate_is_smoke:
            raise TrainingConfigurationError(
                "score_calibration rejects smoke/partial candidate checkpoints; "
                "a one-step candidate is not an accepted source"
            )
        if int(payload.get("global_step", 0)) <= 1:
            raise TrainingConfigurationError(
                "score_calibration rejects an ordinary one-step candidate checkpoint"
            )
        if (
            contract.get("p3_footprint_gate", {}).get("passed") is not True
            or contract.get("index_content_gate", {}).get("passed") is not True
            or contract.get("dataset_authority_gate", {}).get("passed") is not True
            or contract.get("validation_coverage_gate", {}).get("passed") is not True
            or not isinstance(payload.get("training_index_content_sha256"), str)
        ):
            raise TrainingConfigurationError(
                "accepted candidate lacks PASS P3 footprint/content provenance gates"
            )
        acceptance = _verify_candidate_acceptance(path, _contract_path(path))
    return {
        "kind": (
            "candidate_smoke_checkpoint"
            if args.allow_smoke_source
            else "accepted_candidate_capacity_checkpoint"
        ),
        "path": path,
        "payload": payload,
        "contract": contract,
        "checkpoint_sha256": hashlib.sha256(checkpoint_bytes).hexdigest(),
        "contract_sha256": contract_sha256,
        "candidate_acceptance": acceptance,
        "candidate_acceptance_sha256": (
            sha256_file(_candidate_acceptance_path(path))
            if acceptance is not None
            else None
        ),
    }


def _resolve_device(requested: str) -> torch.device:
    requested = str(requested).strip().lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        device = torch.device(requested)
    except (RuntimeError, ValueError) as exc:
        raise TrainingConfigurationError(f"invalid --device {requested!r}") from exc
    if device.type == "cuda" and not torch.cuda.is_available():
        raise TrainingConfigurationError("CUDA was requested but is unavailable")
    if device.type not in ("cpu", "cuda"):
        raise TrainingConfigurationError("training supports only CPU or CUDA devices")
    return device


def _p3_footprint_gate(
    source_contract: Mapping[str, Any],
    *,
    index_sha256: str,
    content_aggregate_sha256: str,
    map_contract_aggregate_sha256: str,
) -> dict:
    """Read and authenticate the fixed corrected-footprint re-audit."""

    path = DEFAULT_P3_FOOTPRINT_REAUDIT.resolve()
    failures = []
    if not path.is_file():
        return {
            "path": str(path),
            "report_sha256": None,
            "status": "MISSING",
            "passed": False,
            "errors": ["p3_footprint_v2_reaudit_missing"],
        }
    try:
        report, report_bytes = _load_json_exact(path)
        report_sha256 = hashlib.sha256(report_bytes).hexdigest()
    except TrainingConfigurationError as exc:
        return {
            "path": str(path),
            "report_sha256": sha256_file(path),
            "status": "INVALID",
            "passed": False,
            "errors": [str(exc)],
        }
    if report.get("schema") != EXPECTED_P3_FOOTPRINT_REAUDIT_SCHEMA:
        failures.append("reaudit_schema_mismatch")
    scope = report.get("scope", {})
    if scope.get("geometry_or_gate_cli_overrides_available") is not False:
        failures.append("reaudit_exposes_geometry_or_gate_override")
    if scope.get("npz_files_modified") is not False:
        failures.append("reaudit_modified_p3_samples")
    if scope.get("test_split_used_for_tuning") is not False:
        failures.append("reaudit_used_test_for_tuning")
    if int(report.get("parallel_workers", 0)) < 8:
        failures.append("reaudit_parallel_workers_lt_8")
    discovered = int(report.get("sample_files_discovered", 0))
    audited = int(report.get("sample_files_audited", -1))
    if discovered < 1 or audited != discovered or report.get("sample_failures"):
        failures.append("reaudit_sample_coverage_incomplete")
    training_authority = report.get("training_authority", {})
    expected_training_authority = {
        "index_sha256": index_sha256,
        "content_aggregate_sha256": content_aggregate_sha256,
        "map_contract_aggregate_sha256": map_contract_aggregate_sha256,
        "splits": ["train", "validation"],
        "test_split_used": False,
    }
    for name, expected in expected_training_authority.items():
        if training_authority.get(name) != expected:
            failures.append("reaudit_training_authority_" + name)

    implementation = report.get("audit_implementation", {})
    authorities = (
        (
            ROOT / "tools/audit_p3_footprint_upgrade.py",
            implementation.get("tool_sha256"),
            "reaudit_tool_sha256_mismatch",
        ),
        (
            ROOT / "dep_car/src/dep_car/core/lattice.py",
            implementation.get("lattice_implementation_sha256"),
            "lattice_implementation_sha256_mismatch",
        ),
        (
            ROOT / "dep_car/src/dep_car/core/occupancy.py",
            implementation.get("occupancy_implementation_sha256"),
            "occupancy_implementation_sha256_mismatch",
        ),
        (
            ROOT / "dep_car/src/dep_car/core/state_contract.py",
            implementation.get("state_contract_implementation_sha256"),
            "state_contract_implementation_sha256_mismatch",
        ),
        (
            ROOT / "dep_car/src/dep_car/core/vehicle.py",
            implementation.get("vehicle_implementation_sha256"),
            "vehicle_implementation_sha256_mismatch",
        ),
        (
            ROOT / "dep_car/src/dep_car/training/losses.py",
            implementation.get("losses_implementation_sha256"),
            "losses_implementation_sha256_mismatch",
        ),
        (
            ROOT / "dep_car/src/dep_car/training/p4_dataset.py",
            implementation.get("p4_dataset_implementation_sha256"),
            "p4_dataset_implementation_sha256_mismatch",
        ),
    )
    for authority_path, claimed_hash, error in authorities:
        if not authority_path.is_file() or claimed_hash != sha256_file(authority_path):
            failures.append(error)
    current_implementation = build_p4_implementation_contract(ROOT)
    if (
        implementation.get("p4_implementation_schema")
        != current_implementation["schema"]
        or implementation.get("p4_implementation_aggregate_sha256")
        != current_implementation["aggregate_sha256"]
        or implementation.get("p4_implementation_files")
        != current_implementation["files"]
    ):
        failures.append("p4_implementation_aggregate_sha256_mismatch")

    rollout = dict(report.get("rollout_contract", {}))
    rollout_hash = rollout.pop("sha256", None)
    expected_rollout = {
        "schema": "CanonicalAckermannLattice3x5V2GearAlignedLimits",
        "candidate_count": 15,
        "candidate_order": "speed_major_3x5_steering_minor",
        "trajectory_fields": [
            "t", "x", "y", "yaw", "signed_speed", "steering"
        ],
        "lattice_config": asdict(LatticeConfig()),
        "state_source": "StaticAckermannSampleV2.vehicle_state",
        "state_fields_used": ["signed_speed", "steering"],
        "opposite_gear_policy": "zero_speed_before_regeneration",
        "all_other_speed_policy": (
            "project_raw_signed_speed_to_requested_gear_and_clip_P0_limits"
        ),
        "steering_policy": "clip_to_P0_operating_limit",
        "longitudinal_limit_frame": "requested_gear_aligned",
    }
    # ``asdict`` preserves tuples from LatticeConfig, while the signed audit
    # report necessarily round-trips them through JSON arrays.  Compare the
    # canonical JSON value, not Python's tuple/list implementation detail;
    # otherwise a valid re-audit can never unlock P5.
    expected_rollout = json.loads(json.dumps(expected_rollout))
    if (
        rollout_hash != _canonical_sha256(rollout)
        or rollout != expected_rollout
    ):
        failures.append("reaudit_rollout_contract_mismatch")

    provenance = report.get("P3_provenance", {})
    acceptance = ROOT / "reports/p3_pilot_acceptance.json"
    if (
        not acceptance.is_file()
        or provenance.get("acceptance_sha256") != sha256_file(acceptance)
    ):
        failures.append("p3_acceptance_sha256_mismatch")
    task_manifest = ROOT / "data/p3_pilot/task_manifest.json"
    if task_manifest.is_file():
        task_payload = _load_json(task_manifest)
        if provenance.get("task_manifest_sha256") != task_payload.get(
            "task_manifest_sha256"
        ):
            failures.append("p3_task_manifest_sha256_mismatch")
    else:
        failures.append("p3_task_manifest_missing")
    source_dataset = source_contract.get("dataset_provenance", {})
    for source_name, report_name in (
        ("p3_acceptance_sha256", "acceptance_sha256"),
        ("p3_task_manifest_sha256", "task_manifest_sha256"),
    ):
        expected = source_dataset.get(source_name)
        if expected is not None and expected != provenance.get(report_name):
            failures.append("source_" + source_name + "_mismatch")

    geometry = dict(report.get("geometry_contract", {}))
    geometry_hash = geometry.pop("sha256", None)
    if geometry_hash != _canonical_sha256(geometry):
        failures.append("reaudit_geometry_sha256_mismatch")
    source_footprint = source_contract.get("footprint_contract", {})
    for report_key, source_key in (
        ("length_m", "length_m"),
        ("width_m", "width_m"),
        ("safety_margin_m", "safety_margin_m"),
        ("circle_count", "circle_count"),
        ("circle_radius_m", "circle_radius_m"),
        ("longitudinal_offsets_m", "longitudinal_offsets_m"),
    ):
        if source_key in source_footprint and not np.allclose(
            geometry.get(report_key), source_footprint[source_key], atol=1.0e-12
        ):
            failures.append("reaudit_footprint_contract_mismatch")
            break

    statistics = report.get("statistics", {}).get("overall", {}).get("new", {})
    sharp_turn = (
        report.get("statistics", {})
        .get("by_mode", {})
        .get("SHARP_TURN", {})
        .get("new", {})
    )
    try:
        overall_zero = float(statistics.get("zero_feasible_rate"))
        overall_median = float(statistics.get("feasible_candidates_median"))
    except (TypeError, ValueError):
        overall_zero = overall_median = float("nan")
    if not math.isfinite(overall_zero) or not overall_zero < 0.10:
        failures.append("overall_zero_feasible_rate_lt_0_10")
    if not math.isfinite(overall_median) or overall_median < 2.0:
        failures.append("overall_median_feasible_candidates_ge_2")
    mode_statistics = report.get("statistics", {}).get("by_mode", {})
    for mode in PILOT_MANEUVER_MODES:
        try:
            mode_zero = float(
                mode_statistics.get(mode, {}).get("new", {}).get(
                    "zero_feasible_rate"
                )
            )
        except (TypeError, ValueError):
            mode_zero = float("nan")
        if not math.isfinite(mode_zero) or not mode_zero < 0.25:
            failures.append("mode_zero_feasible_rate_" + mode)

    report_errors = [str(value) for value in report.get("errors", ())]
    if report.get("status") != "PASS" or report_errors:
        failures.extend(report_errors or ["reaudit_status_not_pass"])
    failures = sorted(set(failures))
    return {
        "path": str(path),
        "report_sha256": report_sha256,
        "schema": report.get("schema"),
        "status": report.get("status", "INVALID"),
        "passed": not failures,
        "errors": failures,
        "samples_audited": audited,
        "overall_zero_feasible_rate": statistics.get("zero_feasible_rate"),
        "overall_median_feasible_candidates": statistics.get(
            "feasible_candidates_median"
        ),
        "sharp_turn_zero_feasible_rate": sharp_turn.get("zero_feasible_rate"),
        "geometry_sha256": geometry_hash,
        "rollout_sha256": rollout_hash,
    }


def _explicit_smoke_run(args: argparse.Namespace) -> bool:
    return bool(_smoke_reasons(args))


def _bounded_smoke_run(args: argparse.Namespace) -> bool:
    """Only a doubly capped run may bypass a failing formal authority gate."""

    return bool(
        _explicit_smoke_run(args)
        and args.max_steps is not None
        and int(args.max_steps) <= SMOKE_MAX_STEPS
        and args.max_samples is not None
        and int(args.max_samples) <= SMOKE_MAX_SAMPLES
    )


def _smoke_reasons(
    args: argparse.Namespace,
    training_config: Optional[Mapping[str, Any]] = None,
) -> list:
    reasons = []
    if args.max_samples is not None:
        reasons.append("max_samples")
    if args.max_steps is not None:
        reasons.append("max_steps")
    if (
        args.stage == "candidate_capacity"
        and args.resume is None
        and args.init is not None
        and args.init.resolve() != DEFAULT_CANDIDATE_INITIALIZATION.resolve()
    ):
        reasons.append("candidate_initialization_override")
    authority = _training_config_authority() if training_config is None else training_config
    frozen = authority["training"]
    actual_parameters = {
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "gradient_clip": float(args.gradient_clip),
        "sensor_dropout_probability": float(args.sensor_dropout_probability),
        "seed": int(args.seed),
        "mixed_precision": bool(args.amp),
        "workers": int(args.workers),
        "torch_threads": int(args.torch_threads),
    }
    frozen_parameters = {
        "epochs": int(frozen["epochs"]),
        "batch_size": int(frozen["batch_size"]),
        "learning_rate": float(frozen["learning_rate"]),
        "weight_decay": float(frozen["weight_decay"]),
        "gradient_clip": float(frozen["gradient_clip"]),
        "sensor_dropout_probability": float(
            frozen["sensor_dropout_probability"]
        ),
        "seed": int(frozen["seed"]),
        "mixed_precision": bool(frozen["mixed_precision"]),
        "workers": int(authority["dataset"]["workers"]),
        "torch_threads": int(frozen["torch_threads"]),
    }
    reasons.extend(
        "training_parameter_override_" + name
        for name, expected in frozen_parameters.items()
        if actual_parameters[name] != expected
    )
    actual = {
        "root": args.data.resolve(),
        "maps": args.maps.resolve(),
        "index": args.index.resolve(),
    }
    reasons.extend(
        "dataset_path_override_" + name
        for name, expected in authority["authority_paths"].items()
        if actual[name] != expected
    )
    return sorted(set(reasons))


def _index_content_gate(index: Mapping[str, Any]) -> dict:
    aggregate = index.get("content_aggregate_sha256") or index.get(
        "samples_content_sha256"
    )
    digest_names = ("content_sha256", "sample_sha256", "sha256")
    entry_digests = []
    for entry in index.get("entries", ()):
        digest = next((entry.get(name) for name in digest_names if entry.get(name)), None)
        entry_digests.append(digest)
    valid = (
        isinstance(aggregate, str)
        and len(aggregate) == 64
        and all(
            isinstance(value, str)
            and len(value) == 64
            and all(character in "0123456789abcdef" for character in value)
            for value in entry_digests
        )
    )
    return {
        "passed": bool(valid),
        "content_aggregate_sha256": aggregate,
        "entries_with_content_sha256": sum(value is not None for value in entry_digests),
        "entries": len(entry_digests),
        "errors": [] if valid else ["training_index_content_hash_authority_missing"],
    }


def _training_yaml_qualification_gate(training_config: Mapping[str, Any]) -> dict:
    qualification = training_config["raw"].get("qualification", {})
    blocked = [str(value) for value in qualification.get("blocked_gates", ())]
    passed = (
        qualification.get("corrected_footprint_p3_status") == "PASS"
        and qualification.get("p5_formal_training_allowed") is True
        and not blocked
    )
    return {
        "passed": passed,
        "corrected_footprint_p3_status": qualification.get(
            "corrected_footprint_p3_status"
        ),
        "p5_formal_training_allowed": qualification.get(
            "p5_formal_training_allowed"
        ),
        "blocked_gates": blocked,
        "errors": [] if passed else ["training_yaml_p5_formal_gate"] + blocked,
    }


def _require_training_gate(args: argparse.Namespace, plan: Mapping[str, Any]) -> None:
    # The validated plan is an immutable authorization snapshot.  Mutating an
    # argparse namespace after planning must never turn a formal plan into an
    # unrecorded smoke run (or cause old hyperparameters to be signed).
    if _training_argument_snapshot(args, plan["training_config"]) != plan[
        "argument_snapshot"
    ]:
        raise TrainingConfigurationError("arguments_changed_after_training_plan")
    permanent_smoke = bool(plan["permanent_smoke"])
    bounded_smoke = bool(_bounded_smoke_run(args))
    if bounded_smoke != bool(plan["bounded_smoke_authorized"]):
        raise TrainingConfigurationError("arguments_changed_after_training_plan")
    formal_pass = (
        plan["p3_footprint_gate"]["passed"]
        and plan["index_content_gate"]["passed"]
        and plan["dataset_authority_gate"]["passed"]
        and plan["validation_coverage_gate"]["passed"]
        and plan["training_yaml_qualification_gate"]["passed"]
        and not permanent_smoke
    )
    if formal_pass or bounded_smoke:
        return
    errors = (
        plan["p3_footprint_gate"]["errors"]
        + plan["index_content_gate"]["errors"]
        + plan["dataset_authority_gate"]["errors"]
        + plan["validation_coverage_gate"]["errors"]
        + plan["training_yaml_qualification_gate"]["errors"]
    )
    if permanent_smoke:
        errors = errors + [
            "smoke_run_requires_max_steps_le_%d_and_max_samples_le_%d"
            % (SMOKE_MAX_STEPS, SMOKE_MAX_SAMPLES)
        ]
    raise TrainingConfigurationError(
        "formal P5 training is blocked by P3 provenance gates: "
        + ", ".join(errors)
        + "; use --dry-run for inspection or an explicitly permanent-smoke run"
    )


def _effective_training_contract(args: argparse.Namespace) -> dict:
    return {
        "stage": args.stage,
        "modality": args.modality,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "gradient_clip": args.gradient_clip,
        "sensor_dropout_probability": (
            args.sensor_dropout_probability if args.modality == "fusion" else 0.0
        ),
        "amp_requested": bool(args.amp),
        "seed": args.seed,
        "torch_threads": args.torch_threads,
        "workers": args.workers,
        "epochs": args.epochs,
        "max_samples": args.max_samples,
        "max_steps": args.max_steps,
        "data": str(args.data.resolve()),
        "maps": str(args.maps.resolve()),
        "index": str(args.index.resolve()),
    }


def _source_argument_path(args: argparse.Namespace) -> Optional[str]:
    if args.resume is not None:
        return str(args.resume.resolve())
    if args.init is not None:
        return str(args.init.resolve())
    if args.stage == "candidate_capacity":
        return str(DEFAULT_CANDIDATE_INITIALIZATION.resolve())
    return None


def _training_argument_snapshot(
    args: argparse.Namespace, training_config: Mapping[str, Any]
) -> dict:
    return {
        "effective_training_contract": _effective_training_contract(args),
        "smoke_reasons": _smoke_reasons(args, training_config),
        "output": str(args.output.resolve()),
        "source": _source_argument_path(args),
        "stage": args.stage,
        "modality": args.modality,
        "allow_smoke_source": bool(args.allow_smoke_source),
        "device_request": str(args.device),
    }


def _validate_p5_source_provenance(
    args: argparse.Namespace,
    source: Mapping[str, Any],
    *,
    index_sha256: str,
    training_config: Mapping[str, Any],
) -> None:
    implementation = build_p4_implementation_contract(ROOT)
    if (
        source["contract"].get("implementation_contract") != implementation
        or source["payload"].get("implementation_aggregate_sha256")
        != implementation["aggregate_sha256"]
    ):
        raise TrainingConfigurationError(
            "source implementation aggregate does not match current P4/P5 code"
        )
    source_training = source["contract"].get("training_contract", {})
    expected_training = {
        "objective_id": DEPCarObjectiveV1.objective_id,
        "objective_revision": DEPCarObjectiveV1.objective_revision,
        "sdf_schema": "SignedDistanceFieldV1KnownFreePositiveUnknownUnsafe",
        "loss_config_sha256": training_config["loss_config_sha256"],
    }
    source_training_mismatches = [
        name for name, value in expected_training.items()
        if source_training.get(name) != value
    ]
    if source_training_mismatches:
        raise TrainingConfigurationError(
            "source objective/loss contract mismatch ("
            + ", ".join(source_training_mismatches)
            + ")"
        )
    if source["kind"] == "formal_depth_v483_initialization":
        return
    payload = source["payload"]
    required = {
        "training_index_sha256": index_sha256,
        "training_index_content_sha256": training_config["dataset"][
            "content_aggregate_sha256"
        ],
        "map_contract_aggregate_sha256": training_config["dataset"][
            "map_contract_aggregate_sha256"
        ],
        "training_config_sha256": training_config["file_sha256"],
        "loss_config_sha256": training_config["loss_config_sha256"],
        "trainer_sha256": sha256_file(TRAINER_PATH),
        "implementation_aggregate_sha256": implementation["aggregate_sha256"],
    }
    mismatches = [name for name, value in required.items() if payload.get(name) != value]
    if mismatches:
        raise TrainingConfigurationError(
            "P5 source provenance mismatch (" + ", ".join(mismatches) + ")"
        )
    if source["kind"] == "resume":
        expected = _effective_training_contract(args)
        if payload.get("effective_training_contract") != expected:
            raise TrainingConfigurationError(
                "resume effective training/optimizer hyperparameters mismatch"
            )


def build_training_plan(args: argparse.Namespace) -> dict:
    """Validate all read-only authority before allocating a model or workers."""

    _validate_numeric_arguments(args)
    training_config = _training_config_authority()
    sample_root = args.data.resolve()
    maps_root = args.maps.resolve()
    index_path = args.index.resolve()
    output = args.output.resolve()
    if not args.dry_run:
        _reject_existing_output_artifacts(output)
    if not sample_root.is_dir():
        raise TrainingConfigurationError(f"sample root does not exist: {sample_root}")
    if not maps_root.is_dir():
        raise TrainingConfigurationError(f"map root does not exist: {maps_root}")
    if not index_path.is_file():
        raise TrainingConfigurationError(f"training index does not exist: {index_path}")
    index, index_sha256 = load_training_index(
        index_path,
        sample_root=sample_root,
        maps_root=maps_root,
        splits=("train", "validation"),
        allow_test=False,
        return_index_sha256=True,
    )
    counts = index.get("counts_by_split", {})
    if set(counts) != {"train", "validation"}:
        raise TrainingConfigurationError(
            "training index must contain exactly train and validation counts"
        )
    if min(int(counts["train"]), int(counts["validation"])) < 1:
        raise TrainingConfigurationError("train and validation splits must both be non-empty")
    counted_entries = {
        split: sum(entry.get("split") == split for entry in index["entries"])
        for split in ("train", "validation")
    }
    if any(int(counts[name]) != counted_entries[name] for name in counted_entries):
        raise TrainingConfigurationError("training index split counts do not match its entries")
    if "test" in index.get("splits", ()):
        raise TrainingConfigurationError("test split must remain sealed during P5 training")
    map_contract = _map_contract_aggregate(maps_root, index)
    dataset_authority_gate = _dataset_authority_gate(
        args, training_config, index, map_contract
    )
    validation_coverage_gate = _validation_coverage_gate(index, training_config)
    smoke_reasons = _smoke_reasons(args, training_config)
    bounded_smoke_authorized = _bounded_smoke_run(args)
    argument_snapshot = _training_argument_snapshot(args, training_config)
    source = _inspect_source(args)
    _validate_p5_source_provenance(
        args,
        source,
        index_sha256=index_sha256,
        training_config=training_config,
    )
    footprint_gate = _p3_footprint_gate(
        source["contract"],
        index_sha256=index_sha256,
        content_aggregate_sha256=index.get("content_aggregate_sha256"),
        map_contract_aggregate_sha256=map_contract["aggregate_sha256"],
    )
    content_gate = _index_content_gate(index)
    yaml_qualification_gate = _training_yaml_qualification_gate(training_config)
    source_preprocessing = source["contract"].get("dataset_provenance", {}).get(
        "p3_preprocessing_sha256"
    )
    if source_preprocessing not in (None, index["bev_preprocessing_sha256"]):
        raise TrainingConfigurationError(
            "checkpoint and training index preprocessing SHA-256 disagree"
        )
    device = _resolve_device(args.device)
    amp_enabled = bool(args.amp and device.type == "cuda")
    implementation_snapshot = build_p4_implementation_contract(ROOT)
    return {
        "stage": args.stage,
        "modality": args.modality,
        "sample_root": sample_root,
        "maps_root": maps_root,
        "index_path": index_path,
        "index_sha256": index_sha256,
        "index": index,
        "map_contract": map_contract,
        "training_config": training_config,
        "trainer_sha256": sha256_file(TRAINER_PATH),
        "implementation_contract": implementation_snapshot,
        "effective_training_contract": _effective_training_contract(args),
        "argument_snapshot": argument_snapshot,
        "output": output,
        "source": source,
        "p3_footprint_gate": footprint_gate,
        "index_content_gate": content_gate,
        "dataset_authority_gate": dataset_authority_gate,
        "validation_coverage_gate": validation_coverage_gate,
        "training_yaml_qualification_gate": yaml_qualification_gate,
        "permanent_smoke": bool(smoke_reasons),
        "bounded_smoke_authorized": bounded_smoke_authorized,
        "smoke_reasons": smoke_reasons,
        "device": device,
        "amp_requested": bool(args.amp),
        "amp_enabled": amp_enabled,
        "splits": ("train", "validation"),
    }


def _public_plan(args: argparse.Namespace, plan: Mapping[str, Any]) -> dict:
    counts = plan["index"]["counts_by_split"]
    cap = args.max_samples
    formal_authorized = (
        plan["p3_footprint_gate"]["passed"]
        and plan["index_content_gate"]["passed"]
        and plan["dataset_authority_gate"]["passed"]
        and plan["validation_coverage_gate"]["passed"]
        and plan["training_yaml_qualification_gate"]["passed"]
        and not plan["permanent_smoke"]
    )
    return {
        "status": (
            "DRY_RUN_READY" if formal_authorized else "BLOCKED"
        ),
        "architecture_id": P4_ARCHITECTURE_ID,
        "stage": plan["stage"],
        "modality": plan["modality"],
        "sealed_test_split": True,
        "splits": list(plan["splits"]),
        "samples": {
            name: min(int(counts[name]), cap) if cap is not None else int(counts[name])
            for name in plan["splits"]
        },
        "workers": args.workers,
        "torch_threads": args.torch_threads,
        "device": str(plan["device"]),
        "amp_requested": plan["amp_requested"],
        "amp_enabled": plan["amp_enabled"],
        "epochs": args.epochs,
        "max_steps": args.max_steps,
        "source_kind": plan["source"]["kind"],
        "source": str(plan["source"]["path"]),
        "output": str(plan["output"]),
        "trainer_sha256": plan["trainer_sha256"],
        "training_config": {
            "path": str(plan["training_config"]["path"]),
            "file_sha256": plan["training_config"]["file_sha256"],
            "semantic_sha256": plan["training_config"]["semantic_sha256"],
            "loss_config_sha256": plan["training_config"][
                "loss_config_sha256"
            ],
        },
        "qualification_status": "UNQUALIFIED",
        "formal_training_authorized": formal_authorized,
        "bounded_smoke_authorized": plan["bounded_smoke_authorized"],
        "permanent_smoke": plan["permanent_smoke"],
        "smoke_reasons": plan["smoke_reasons"],
        "p3_footprint_gate": plan["p3_footprint_gate"],
        "index_content_gate": plan["index_content_gate"],
        "dataset_authority_gate": plan["dataset_authority_gate"],
        "validation_coverage_gate": plan["validation_coverage_gate"],
        "training_yaml_qualification_gate": plan[
            "training_yaml_qualification_gate"
        ],
    }


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except TypeError:  # pragma: no cover - compatibility with older Noetic envs
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _cap_dataset(dataset, maximum_samples: Optional[int]):
    if maximum_samples is None or len(dataset) <= maximum_samples:
        return dataset
    return Subset(dataset, range(maximum_samples))


def _make_loaders(args: argparse.Namespace, plan: Mapping[str, Any]):
    common = dict(
        sample_root=plan["sample_root"],
        maps_root=plan["maps_root"],
        index_path=plan["index_path"],
        index_splits=("train", "validation"),
        workers=args.workers or 1,
        allow_test=False,
        depth_dropout_probability=0.0,
        lidar_dropout_probability=0.0,
        augmentation_seed=args.seed,
        expected_map_contract_aggregate_sha256=plan["map_contract"][
            "aggregate_sha256"
        ],
        expected_index_sha256=plan["index_sha256"],
    )
    train_base = P3TrainingDatasetV1(split="train", **common)
    validation_base = P3TrainingDatasetV1(split="validation", **common)
    train = _cap_dataset(train_base, args.max_samples)
    validation = _cap_dataset(validation_base, args.max_samples)
    generator = torch.Generator().manual_seed(args.seed)
    loader_common = dict(
        batch_size=args.batch_size,
        num_workers=args.workers,
        collate_fn=p3_training_collate,
        worker_init_fn=p3_training_worker_init,
        pin_memory=plan["device"].type == "cuda",
        persistent_workers=args.workers > 0,
    )
    train_loader = DataLoader(train, shuffle=True, generator=generator, **loader_common)
    validation_loader = DataLoader(validation, shuffle=False, **loader_common)
    return train_base, validation_base, train_loader, validation_loader


def _select_valid_geometry(batch: Mapping[str, Any]) -> tuple:
    valid = batch["geometry_valid"].bool()
    total = int(valid.numel())
    selected = int(valid.sum())
    if selected == 0:
        return None, {}, selected, total
    keys = (
        "depth",
        "lidar_bev",
        "modality_mask",
        "state",
        "requested_gear",
        "route_pose",
        "route_mask",
        "map_distance_field",
        "map_resolution",
        "map_origin",
        "chassis_to_map",
        "geometry_valid",
    )
    indices = torch.nonzero(valid, as_tuple=False).flatten().tolist()
    metadata = [batch["metadata"][index] for index in indices]
    grouping = {
        "maneuver": tuple(row["maneuver_mode"] for row in metadata),
        "candidate_context": tuple(
            row.get("candidate_context", "UNKNOWN") for row in metadata
        ),
        "requested_gear": tuple(
            "FORWARD" if int(batch["requested_gear"][index]) > 0 else "REVERSE"
            for index in indices
        ),
    }
    return (
        {key: batch[key][valid] for key in keys},
        grouping,
        selected,
        total,
    )


def _to_device(batch: Mapping[str, torch.Tensor], device: torch.device) -> dict:
    return {
        key: value.to(device=device, non_blocking=True)
        for key, value in batch.items()
    }


def _batch_modality_mask(
    batch: Mapping[str, torch.Tensor],
    mode: str,
    dropout_probability: float,
    training: bool,
) -> torch.Tensor:
    selected = modality_mask(
        len(batch["state"]), mode, device=batch["state"].device, dtype=batch["state"].dtype
    )
    selected = selected * batch["modality_mask"].to(selected)
    if bool(torch.any(selected.sum(dim=1) < 1.0)):
        raise RuntimeError("requested ablation conflicts with an absent source modality")
    if training and mode == "fusion" and dropout_probability > 0.0:
        selected = apply_sensor_dropout(selected, dropout_probability)
    return selected


@contextmanager
def _amp_encoder_output_fp32(model: DEPCarNetV1, enabled: bool):
    """Keep AMP convolutions but join encoder features in FP32.

    Autocast emits FP16 encoder features while the learned missing-modality
    tokens intentionally remain FP32 parameters.  ``Tensor.index_copy`` does
    not perform dtype promotion, so every modality that executes an encoder
    would otherwise fail before its first optimizer step.  Forward hooks cast
    only the two encoder outputs back to FP32; convolution kernels still run
    under autocast, gradients cross the cast, and checkpoint parameter dtypes
    and the signed model implementation contract remain unchanged.
    """

    handles = []
    if enabled:
        def output_fp32(_module, _arguments, output):
            if not isinstance(output, torch.Tensor):
                raise RuntimeError("sensor encoder output must be a tensor")
            return output.float()

        handles = [
            model.depth_encoder.register_forward_hook(output_fp32),
            model.lidar_encoder.register_forward_hook(output_fp32),
        ]
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def _forward_objective(
    model: DEPCarNetV1,
    objective: DEPCarObjectiveV1,
    batch: Mapping[str, torch.Tensor],
    *,
    stage: str,
    mode: str,
    sensor_dropout_probability: float,
    training: bool,
    amp_enabled: bool,
) -> tuple:
    mask = _batch_modality_mask(
        batch, mode, sensor_dropout_probability, training
    )
    with _amp_encoder_output_fp32(model, amp_enabled):
        with torch.autocast(
            device_type=batch["state"].device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            output = model(
                batch["depth"],
                batch["lidar_bev"],
                batch["state"],
                batch["requested_gear"],
                modality_mask=mask,
            )
            losses = objective(
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
    return output, losses


class _MetricAccumulator:
    scalar_loss_names = (
        "total",
        "candidate",
        "score",
        "safety",
        "guidance",
        "kinematic",
        "comfort",
        "diversity",
        "anchor",
    )

    def __init__(self):
        self.weight = 0
        self.seen = 0
        self.valid = 0
        self.steps = 0
        self.values = {name: 0.0 for name in self.scalar_loss_names}
        self.minimum_clearance = 0.0
        self.candidate_metrics = CandidateMetricAccumulator()
        self.context_metrics = CandidateMetricAccumulator()
        self.gear_metrics = CandidateMetricAccumulator()

    def note_geometry(self, valid: int, seen: int) -> None:
        self.valid += valid
        self.seen += seen

    def update(
        self,
        output,
        losses: Mapping[str, torch.Tensor],
        batch_size: int,
        grouping,
    ) -> None:
        self.weight += batch_size
        self.steps += 1
        for name in self.scalar_loss_names:
            self.values[name] += float(losses[name].detach()) * batch_size
        clearance = losses["minimum_clearance"].detach()
        self.minimum_clearance += float(clearance.mean()) * batch_size
        metrics = candidate_batch_metrics(output, losses)
        self.candidate_metrics.update(metrics, grouping["maneuver"])
        self.context_metrics.update(metrics, grouping["candidate_context"])
        self.gear_metrics.update(metrics, grouping["requested_gear"])

    def result(self) -> dict:
        denominator = max(self.weight, 1)
        result = {name: value / denominator for name, value in self.values.items()}
        result.update(
            {
                "mean_candidate_minimum_clearance_m": self.minimum_clearance / denominator,
                "geometry_valid_fraction": self.valid / max(self.seen, 1),
                "samples_seen": self.seen,
                "samples_optimized": self.weight,
                "batches": self.steps,
            }
        )
        candidate_summary = self.candidate_metrics.compute()
        result["candidate_metrics"] = {
            "overall": candidate_summary["overall"],
            "by_maneuver": candidate_summary["by_maneuver"],
            "by_candidate_context": self.context_metrics.compute()["by_maneuver"],
            "by_requested_gear": self.gear_metrics.compute()["by_maneuver"],
        }
        return result


def _configure_stage_and_modality(model: DEPCarNetV1, stage: str, mode: str) -> dict:
    ownership = configure_training_stage(model, stage)
    frozen_partition = None
    if stage == "candidate_capacity" and mode == "depth_only":
        frozen_partition = "lidar"
        for parameter in model.lidar_encoder.parameters():
            parameter.requires_grad_(False)
        model.lidar_missing_token.requires_grad_(False)
    elif stage == "candidate_capacity" and mode == "lidar_only":
        frozen_partition = "depth"
        for parameter in model.depth_encoder.parameters():
            parameter.requires_grad_(False)
        model.depth_missing_token.requires_grad_(False)
    ownership = dict(ownership)
    ownership.update({
        "modality": mode,
        "frozen_sensor_partition": frozen_partition,
        "effective_trainable": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
    })
    return ownership


def _build_effective_optimizer(model, stage, mode, learning_rate, weight_decay):
    ownership = _configure_stage_and_modality(model, stage, mode)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise RuntimeError("selected stage/modality has no trainable parameters")
    optimizer = torch.optim.AdamW(
        parameters, lr=learning_rate, weight_decay=weight_decay
    )
    return optimizer, ownership


def _set_training_mode(
    model: DEPCarNetV1, stage: str, mode: str, training: bool
) -> None:
    if not training:
        model.eval()
    elif stage == "score_calibration":
        # The candidate partition contains BatchNorm buffers.  eval() freezes
        # them while score modules (which contain no BatchNorm/Dropout) remain
        # fully differentiable through their trainable parameters.
        model.eval()
    else:
        model.train()
        if mode == "depth_only":
            model.lidar_encoder.eval()
        elif mode == "lidar_only":
            model.depth_encoder.eval()


def _run_epoch(
    model,
    objective,
    loader,
    *,
    optimizer,
    scaler,
    args,
    plan,
    training,
    remaining_steps=None,
) -> tuple:
    _set_training_mode(model, plan["stage"], plan["modality"], training)
    accumulator = _MetricAccumulator()
    completed = True
    for raw_index, raw_batch in enumerate(loader):
        selected, grouping, valid_count, seen_count = _select_valid_geometry(
            raw_batch
        )
        accumulator.note_geometry(valid_count, seen_count)
        if selected is None:
            continue
        batch = _to_device(selected, plan["device"])
        if training:
            optimizer.zero_grad(set_to_none=True)
            output, losses = _forward_objective(
                model,
                objective,
                batch,
                stage=plan["stage"],
                mode=plan["modality"],
                sensor_dropout_probability=args.sensor_dropout_probability,
                training=True,
                amp_enabled=plan["amp_enabled"],
            )
            if not bool(torch.isfinite(losses["total"])):
                raise FloatingPointError("training loss became non-finite")
            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimizer)
            if args.gradient_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(
                    (parameter for parameter in model.parameters() if parameter.requires_grad),
                    args.gradient_clip,
                )
            scaler.step(optimizer)
            scaler.update()
        else:
            with torch.no_grad():
                output, losses = _forward_objective(
                    model,
                    objective,
                    batch,
                    stage=plan["stage"],
                    mode=plan["modality"],
                    sensor_dropout_probability=0.0,
                    training=False,
                    amp_enabled=plan["amp_enabled"],
                )
        accumulator.update(output, losses, valid_count, grouping)
        if training and remaining_steps is not None and accumulator.steps >= remaining_steps:
            completed = raw_index + 1 >= len(loader)
            break
    if accumulator.weight == 0:
        raise RuntimeError("epoch contains no geometry-valid samples")
    return accumulator.result(), accumulator.steps, completed


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, path)
        os.chmod(path, 0o644)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    os.close(descriptor)
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
        os.chmod(path, 0o644)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _artifact_paths(output: Path) -> dict:
    return {
        "checkpoint": output,
        "contract": output.with_suffix(".contract.json"),
        "history": output.with_suffix(".history.json"),
        "metrics": output.with_suffix(".metrics.json"),
    }


def _capture_rng_state() -> dict:
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": str(numpy_state[0]),
            "state": torch.from_numpy(
                np.asarray(numpy_state[1], dtype=np.uint32).copy()
            ),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        ),
    }


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    required = {"python", "numpy", "torch_cpu", "torch_cuda"}
    if not isinstance(state, Mapping) or not required.issubset(state):
        raise TrainingConfigurationError("resume checkpoint RNG state is incomplete")
    numpy_state = state["numpy"]
    if (
        not isinstance(numpy_state, Mapping)
        or set(numpy_state)
        != {"bit_generator", "state", "position", "has_gauss", "cached_gaussian"}
        or not isinstance(numpy_state.get("state"), torch.Tensor)
    ):
        raise TrainingConfigurationError("resume checkpoint NumPy RNG state is invalid")
    random.setstate(tuple(state["python"]))
    np.random.set_state((
        str(numpy_state["bit_generator"]),
        numpy_state["state"].detach().cpu().numpy().astype(np.uint32, copy=True),
        int(numpy_state["position"]),
        int(numpy_state["has_gauss"]),
        float(numpy_state["cached_gaussian"]),
    ))
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state["torch_cuda"]:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _optimizer_contract(optimizer) -> list:
    fields = ("lr", "weight_decay", "betas", "eps", "amsgrad", "maximize")
    return [
        {
            name: list(group[name]) if isinstance(group.get(name), tuple) else group.get(name)
            for name in fields
        }
        for group in optimizer.param_groups
    ]


def _write_artifacts(
    *,
    model,
    optimizer,
    scaler,
    args,
    plan,
    history,
    metrics,
    completed_epochs,
    global_step,
    partial_epoch,
) -> dict:
    paths = _artifact_paths(plan["output"])
    status = "TRAINED_UNQUALIFIED"
    implementation = build_p4_implementation_contract(ROOT)
    if (
        implementation != plan["implementation_contract"]
        or sha256_file(TRAINER_PATH) != plan["trainer_sha256"]
    ):
        raise TrainingConfigurationError(
            "trainer/P4 implementation changed after plan validation; refusing "
            "to sign artifacts for code that did not execute for the full run"
        )
    try:
        current_training_config_sha256 = sha256_file(
            plan["training_config"]["path"]
        )
        current_index_sha256 = sha256_file(plan["index_path"])
        current_map_contract = _map_contract_aggregate(
            plan["maps_root"], plan["index"]
        )
    except (OSError, TrainingConfigurationError) as exc:
        raise TrainingConfigurationError(
            f"training authority changed before artifact signing: {exc}"
        ) from exc
    if (
        current_training_config_sha256
        != plan["training_config"]["file_sha256"]
        or current_index_sha256 != plan["index_sha256"]
        or current_map_contract != plan["map_contract"]
    ):
        raise TrainingConfigurationError(
            "training config/index/map authority changed after plan validation"
        )
    smoke_lineage = bool(
        plan["permanent_smoke"]
        or plan["source"].get("kind") == "candidate_smoke_checkpoint"
        or plan["source"].get("payload", {}).get("smoke_lineage", False)
    )
    checkpoint = {
        "schema": CHECKPOINT_SCHEMA,
        "checkpoint_version": CHECKPOINT_VERSION,
        "architecture_id": P4_ARCHITECTURE_ID,
        "status": status,
        "qualification_status": "UNQUALIFIED",
        "production_qualified": False,
        "training_stage": plan["stage"],
        "modality": plan["modality"],
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "grad_scaler_state_dict": scaler.state_dict(),
        "completed_epochs": int(completed_epochs),
        "global_step": int(global_step),
        "partial_epoch": bool(partial_epoch),
        "history": list(history),
        "metrics": dict(metrics),
        "source_checkpoint_sha256": plan["source"]["checkpoint_sha256"],
        "training_index_sha256": plan["index_sha256"],
        "training_index_content_sha256": plan["index_content_gate"].get(
            "content_aggregate_sha256"
        ),
        "map_contract_aggregate_sha256": plan["map_contract"][
            "aggregate_sha256"
        ],
        "training_config_sha256": plan["training_config"]["file_sha256"],
        "training_config_semantic_sha256": plan["training_config"][
            "semantic_sha256"
        ],
        "loss_config_sha256": plan["training_config"]["loss_config_sha256"],
        "trainer_sha256": sha256_file(TRAINER_PATH),
        "implementation_aggregate_sha256": implementation["aggregate_sha256"],
        "effective_training_contract": plan["effective_training_contract"],
        "effective_optimizer_hyperparameters": _optimizer_contract(optimizer),
        "validation_coverage_gate": dict(plan["validation_coverage_gate"]),
        "smoke_lineage": smoke_lineage,
        "smoke_reasons": list(plan["smoke_reasons"]),
        "rng_state": _capture_rng_state(),
        "sampler_state": {
            "strategy": "epoch_seed_v1",
            "base_seed": int(args.seed),
            "next_epoch": int(completed_epochs),
            "partial_epoch": bool(partial_epoch),
        },
        "seed": int(args.seed),
    }
    _atomic_torch_save(paths["checkpoint"], checkpoint)
    history_document = {
        "schema": "DEPCarP5TrainingHistoryV1",
        "architecture_id": P4_ARCHITECTURE_ID,
        "training_stage": plan["stage"],
        "modality": plan["modality"],
        "qualification_status": "UNQUALIFIED",
        "production_qualified": False,
        "history": list(history),
    }
    metrics_document = {
        "schema": "DEPCarP5TrainingMetricsV1",
        "architecture_id": P4_ARCHITECTURE_ID,
        "training_stage": plan["stage"],
        "modality": plan["modality"],
        "qualification_status": "UNQUALIFIED",
        "production_qualified": False,
        "completed_epochs": int(completed_epochs),
        "global_step": int(global_step),
        "partial_epoch": bool(partial_epoch),
        "metrics": dict(metrics),
    }
    _atomic_write_json(paths["history"], history_document)
    _atomic_write_json(paths["metrics"], metrics_document)
    index = plan["index"]
    objective_config = asdict(plan["training_config"]["loss_config"])
    # Preserve the full formal P4 identity contract (inputs, state order,
    # rollout, footprint and exact v4.8.3 transfer).  This makes the artifact
    # usable in an explicitly unqualified P6 shadow run while the default ROS
    # loader still rejects it because production_qualified remains false.
    contract = copy.deepcopy(plan["source"]["contract"])
    contract.update({
        "schema": P4_CONTRACT_SCHEMA,
        "contract_version": 2,
        "architecture_id": P4_ARCHITECTURE_ID,
        "checkpoint_version": CHECKPOINT_VERSION,
        "checkpoint_sha256": sha256_file(paths["checkpoint"]),
        "status": status,
        "qualification_status": "UNQUALIFIED",
        "production_qualified": False,
        "training_stage": plan["stage"],
        "modality": plan["modality"],
    })
    contract["implementation_contract"] = implementation
    contract["training_contract"] = {
        "objective_id": DEPCarObjectiveV1.objective_id,
        "objective_revision": DEPCarObjectiveV1.objective_revision,
        "sdf_schema": "SignedDistanceFieldV1KnownFreePositiveUnknownUnsafe",
        "footprint_schema": FIVE_CIRCLE_FOOTPRINT_SCHEMA,
        "sweep_interpolation_schema": SWEPT_INTERPOLATION_SCHEMA,
        "sweep_substeps_per_source_segment": SWEPT_SUBSTEPS_PER_SEGMENT,
        "loss_config": objective_config,
        "loss_config_sha256": _canonical_sha256(objective_config),
        "stage_order": ["candidate_capacity", "score_calibration"],
    }
    footprint_contract = dict(contract.get("footprint_contract", {}))
    footprint_contract.update({
        "runtime_grid_allowance": "half_cell_diagonal",
        "differentiable_training_grid_allowance": "one_cell_diagonal",
    })
    contract["footprint_contract"] = footprint_contract
    dataset_provenance = dict(contract.get("dataset_provenance", {}))
    dataset_provenance.update({
        "index_schema": TRAINING_INDEX_SCHEMA,
        "training_view": TRAINING_VIEW_SCHEMA,
        "index_sha256": plan["index_sha256"],
        "content_aggregate_sha256": plan["index_content_gate"].get(
            "content_aggregate_sha256"
        ),
        "map_contract_schema": plan["map_contract"]["schema"],
        "map_contract_aggregate_sha256": plan["map_contract"][
            "aggregate_sha256"
        ],
        "map_contract_count": plan["map_contract"]["map_count"],
        "sample_root": str(plan["sample_root"]),
        "maps_root": str(plan["maps_root"]),
        "splits_used": ["train", "validation"],
        "test_split_used": False,
        "split_authority": "map_uuid",
        "counts_by_split": index["counts_by_split"],
        "bev_preprocessing_sha256": index["bev_preprocessing_sha256"],
        "p3_footprint_v2_reaudit_sha256": plan["p3_footprint_gate"][
            "report_sha256"
        ],
    })
    contract["dataset_provenance"] = dataset_provenance
    contract["objective_execution"] = {
        "objective_id": DEPCarObjectiveV1.objective_id,
        "objective_revision": DEPCarObjectiveV1.objective_revision,
        "sdf_schema": "SignedDistanceFieldV1KnownFreePositiveUnknownUnsafe",
        "footprint_schema": FIVE_CIRCLE_FOOTPRINT_SCHEMA,
        "sweep_interpolation_schema": SWEPT_INTERPOLATION_SCHEMA,
        "sweep_substeps_per_source_segment": SWEPT_SUBSTEPS_PER_SEGMENT,
        "geometry_authority": [
            "physical_state",
            "physical_route_pose",
            "map_distance_field",
            "chassis_to_map",
        ],
    }
    contract["training_source"] = {
        "kind": plan["source"]["kind"],
        "checkpoint": str(plan["source"]["path"]),
        "checkpoint_sha256": plan["source"]["checkpoint_sha256"],
        "contract_sha256": plan["source"]["contract_sha256"],
        "candidate_acceptance_sha256": plan["source"].get(
            "candidate_acceptance_sha256"
        ),
    }
    contract["training_run"] = {
        "epochs_requested": args.epochs,
        "completed_epochs": int(completed_epochs),
        "global_step": int(global_step),
        "partial_epoch": bool(partial_epoch),
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "gradient_clip": args.gradient_clip,
        "workers": args.workers,
        "torch_threads": args.torch_threads,
        "seed": args.seed,
        "sensor_dropout_probability": (
            args.sensor_dropout_probability if plan["modality"] == "fusion" else 0.0
        ),
        "amp_requested": plan["amp_requested"],
        "amp_enabled": plan["amp_enabled"],
        "device_type": plan["device"].type,
        "max_samples": args.max_samples,
        "max_steps": args.max_steps,
        "smoke_limited": plan["permanent_smoke"],
        "smoke_lineage": smoke_lineage,
        "smoke_reasons": list(plan["smoke_reasons"]),
        "formal_p3_footprint_gate_passed": plan["p3_footprint_gate"]["passed"],
        "formal_index_content_gate_passed": plan["index_content_gate"]["passed"],
        "formal_dataset_authority_gate_passed": plan[
            "dataset_authority_gate"
        ]["passed"],
        "formal_validation_coverage_gate_passed": plan[
            "validation_coverage_gate"
        ]["passed"],
        "formal_training_yaml_gate_passed": plan[
            "training_yaml_qualification_gate"
        ]["passed"],
        "training_config_sha256": plan["training_config"]["file_sha256"],
        "training_config_semantic_sha256": plan["training_config"][
            "semantic_sha256"
        ],
        "loss_config_sha256": plan["training_config"]["loss_config_sha256"],
        "trainer_sha256": sha256_file(TRAINER_PATH),
        "implementation_aggregate_sha256": implementation["aggregate_sha256"],
        "effective_training_contract": plan["effective_training_contract"],
        "effective_optimizer_hyperparameters": _optimizer_contract(optimizer),
    }
    contract["artifacts"] = {
        "history": paths["history"].name,
        "history_sha256": sha256_file(paths["history"]),
        "metrics": paths["metrics"].name,
        "metrics_sha256": sha256_file(paths["metrics"]),
    }
    contract["qualification"] = {
        "eligible_for_deployment": False,
        "reason": "P5 training does not perform P6/P8 closed-loop qualification",
    }
    contract["p3_footprint_gate"] = dict(plan["p3_footprint_gate"])
    contract["index_content_gate"] = dict(plan["index_content_gate"])
    contract["dataset_authority_gate"] = dict(plan["dataset_authority_gate"])
    contract["validation_coverage_gate"] = dict(
        plan["validation_coverage_gate"]
    )
    contract["training_yaml_qualification_gate"] = dict(
        plan["training_yaml_qualification_gate"]
    )
    _atomic_write_json(paths["contract"], contract)
    if plan["stage"] == "candidate_capacity":
        acceptance_path = _candidate_acceptance_path(paths["checkpoint"])
        _atomic_write_json(acceptance_path, {
            "schema": "DEPCarP5CandidateAcceptanceV1",
            "status": "SMOKE_SOURCE_ONLY" if smoke_lineage else "PENDING_EXTERNAL_ACCEPTANCE",
            "gate_passed": False,
            "training_stage": "candidate_capacity",
            "checkpoint_sha256": sha256_file(paths["checkpoint"]),
            "contract_sha256": sha256_file(paths["contract"]),
            "smoke_limited": smoke_lineage,
            "partial_epoch": bool(partial_epoch),
            "validation_metrics_sha256": sha256_file(paths["metrics"]),
        })
        paths["candidate_acceptance"] = acceptance_path
    return {name: str(path) for name, path in paths.items()}


def _restore_or_initialize(model, optimizer, scaler, plan):
    payload = plan["source"]["payload"]
    try:
        model.load_state_dict(payload["model_state_dict"], strict=True)
    except RuntimeError as exc:
        raise TrainingConfigurationError(
            f"checkpoint tensors do not match DEPCarNetV1: {exc}"
        ) from exc
    history = []
    completed_epochs = 0
    global_step = 0
    if plan["source"]["kind"] == "resume":
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        for group in optimizer.param_groups:
            if not np.isclose(group["lr"], plan["effective_training_contract"]["learning_rate"]):
                raise TrainingConfigurationError("resume optimizer learning rate mismatch")
            if not np.isclose(
                group["weight_decay"],
                plan["effective_training_contract"]["weight_decay"],
            ):
                raise TrainingConfigurationError("resume optimizer weight decay mismatch")
        scaler_state = payload.get("grad_scaler_state_dict")
        if isinstance(scaler_state, dict):
            scaler.load_state_dict(scaler_state)
        history = list(payload.get("history", ()))
        completed_epochs = int(payload.get("completed_epochs", 0))
        global_step = int(payload.get("global_step", 0))
        sampler_state = payload.get("sampler_state", {})
        if (
            sampler_state.get("strategy") != "epoch_seed_v1"
            or int(sampler_state.get("base_seed", -1))
            != int(plan["effective_training_contract"]["seed"])
        ):
            raise TrainingConfigurationError("resume sampler state mismatch")
        _restore_rng_state(payload.get("rng_state"))
        # A max-steps smoke checkpoint may end mid-epoch.  Resume restarts that
        # epoch with a deterministic sampler and replaces its partial history
        # record instead of reporting the same epoch twice.
        if payload.get("partial_epoch") and history and not history[-1].get("completed", True):
            history.pop()
    return history, completed_epochs, global_step


def run_training(args: argparse.Namespace, plan: Mapping[str, Any]) -> dict:
    _require_training_gate(args, plan)
    _reject_existing_output_artifacts(plan["output"])
    _seed_everything(args.seed)
    torch.set_num_threads(args.torch_threads)
    train_base, _, train_loader, validation_loader = _make_loaders(args, plan)
    model = DEPCarNetV1().to(plan["device"])
    optimizer, ownership = _build_effective_optimizer(
        model,
        plan["stage"],
        plan["modality"],
        args.learning_rate,
        args.weight_decay,
    )
    scaler = _make_grad_scaler(plan["amp_enabled"])
    history, completed_epochs, global_step = _restore_or_initialize(
        model, optimizer, scaler, plan
    )
    if completed_epochs >= args.epochs:
        raise TrainingConfigurationError(
            "--epochs must exceed the completed epoch count in --resume"
        )
    objective = DEPCarObjectiveV1(plan["training_config"]["loss_config"])
    last_metrics = {}
    partial_epoch = False
    steps_this_run = 0
    started = time.time()
    for epoch in range(completed_epochs, args.epochs):
        train_base.set_epoch(epoch)
        train_loader.generator.manual_seed(args.seed + epoch)
        torch.manual_seed(args.seed + epoch)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed + epoch)
        remaining_steps = None
        if args.max_steps is not None:
            remaining_steps = args.max_steps - steps_this_run
            if remaining_steps <= 0:
                partial_epoch = True
                break
        train_metrics, steps, train_completed = _run_epoch(
            model,
            objective,
            train_loader,
            optimizer=optimizer,
            scaler=scaler,
            args=args,
            plan=plan,
            training=True,
            remaining_steps=remaining_steps,
        )
        global_step += steps
        steps_this_run += steps
        validation_metrics, _, _ = _run_epoch(
            model,
            objective,
            validation_loader,
            optimizer=optimizer,
            scaler=scaler,
            args=args,
            plan=plan,
            training=False,
        )
        partial_epoch = not train_completed
        if train_completed:
            completed_epochs = epoch + 1
        record = {
            "epoch": epoch + 1,
            "completed": train_completed,
            "global_step": global_step,
            "train": train_metrics,
            "validation": validation_metrics,
        }
        history.append(record)
        last_metrics = {
            "train": train_metrics,
            "validation": validation_metrics,
            "trainable_parameters": ownership,
            "elapsed_seconds": time.time() - started,
        }
        print(json.dumps(record, sort_keys=True), flush=True)
        artifacts = _write_artifacts(
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            args=args,
            plan=plan,
            history=history,
            metrics=last_metrics,
            completed_epochs=completed_epochs,
            global_step=global_step,
            partial_epoch=partial_epoch,
        )
        if partial_epoch or (
            args.max_steps is not None and steps_this_run >= args.max_steps
        ):
            break
    if not last_metrics:
        raise TrainingConfigurationError("no new optimizer step was requested by this run")
    return {
        "status": "TRAINED_UNQUALIFIED",
        "qualification_status": "UNQUALIFIED",
        "stage": plan["stage"],
        "modality": plan["modality"],
        "completed_epochs": completed_epochs,
        "global_step": global_step,
        "partial_epoch": partial_epoch,
        "artifacts": artifacts,
    }


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        plan = build_training_plan(args)
        if args.dry_run:
            print(json.dumps(_public_plan(args, plan), indent=2, sort_keys=True))
            return 0
        result = run_training(args, plan)
    except (ValueError, OSError, RuntimeError, FloatingPointError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
