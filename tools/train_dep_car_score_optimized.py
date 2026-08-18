#!/usr/bin/env python3
"""Hash-isolated, higher-throughput P5 Score Head trainer.

The accepted Candidate Capacity artifacts remain bound to ``train_dep_car.py``
and the P4 implementation aggregate.  This entry point first asks that frozen
trainer to verify the Candidate and all P3/P4 authorities, then applies a
separately signed Score-only execution contract.  It never trains candidate
parameters and never qualifies a checkpoint for deployment.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
OPTIMIZED_TRAINER_PATH = Path(__file__).resolve()
BASE_TRAINER_PATH = ROOT / "tools/train_dep_car.py"
PERFORMANCE_CONFIG_PATH = ROOT / "dep_car/config/score_training_optimized.yaml"
SCORE_DATASET_PATH = ROOT / "dep_car/src/dep_car/training/score_dataset.py"
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(ROOT / "dep_car/src"))

import train_dep_car as base

from dep_car.model.checkpoint import sha256_file
from dep_car.model.implementation_contract import build_p4_implementation_contract
from dep_car.training.metrics import CandidateMetricAccumulator, candidate_batch_metrics
from dep_car.training.p4_dataset import (
    p3_training_collate,
    p3_training_worker_init,
)
from dep_car.training.score_dataset import (
    P3ScoreTrainingDatasetV1,
    SCORE_TRAINING_VIEW_REVISION,
    SCORE_TRAINING_VIEW_SCHEMA,
)


PERFORMANCE_SCHEMA = "DEPCarP5ScorePerformanceContractV1"
ATTESTATION_SCHEMA = "DEPCarP5ScoreExecutionAttestationV1"


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_performance_authority() -> dict:
    try:
        raw_bytes = PERFORMANCE_CONFIG_PATH.read_bytes()
        value = yaml.safe_load(raw_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise base.TrainingConfigurationError(
            "cannot read Score performance authority: %s" % exc
        ) from exc
    expected_root = {
        "schema",
        "revision",
        "status",
        "base_authority",
        "score_training",
        "data_loader",
        "metrics",
        "precision",
        "benchmark",
    }
    if not isinstance(value, dict) or set(value) != expected_root:
        raise base.TrainingConfigurationError(
            "Score performance authority keys/schema are invalid"
        )
    if (
        value.get("schema") != PERFORMANCE_SCHEMA
        or value.get("revision") != 1
        or value.get("status") != "AUTHORIZED_AFTER_ACCEPTED_CANDIDATE"
    ):
        raise base.TrainingConfigurationError(
            "Score performance authority status is invalid"
        )

    base_authority = value.get("base_authority", {})
    expected_base_keys = {
        "training_config",
        "training_config_sha256",
        "candidate_trainer",
        "candidate_trainer_sha256",
        "p4_implementation_aggregate_sha256",
    }
    if not isinstance(base_authority, dict) or set(base_authority) != expected_base_keys:
        raise base.TrainingConfigurationError(
            "Score base authority fields are invalid"
        )
    expected_paths = {
        "training_config": ROOT / "dep_car/config/training.yaml",
        "candidate_trainer": BASE_TRAINER_PATH,
    }
    for name, expected_path in expected_paths.items():
        relative = Path(str(base_authority.get(name, "")))
        if relative.is_absolute() or ".." in relative.parts:
            raise base.TrainingConfigurationError(
                "Score base authority path is unsafe: %s" % name
            )
        if (ROOT / relative).resolve() != expected_path.resolve():
            raise base.TrainingConfigurationError(
                "Score base authority path changed: %s" % name
            )
    checks = {
        "training_config_sha256": sha256_file(expected_paths["training_config"]),
        "candidate_trainer_sha256": sha256_file(expected_paths["candidate_trainer"]),
        "p4_implementation_aggregate_sha256": build_p4_implementation_contract(
            ROOT
        )["aggregate_sha256"],
    }
    mismatches = [
        name for name, actual in checks.items() if base_authority.get(name) != actual
    ]
    if mismatches:
        raise base.TrainingConfigurationError(
            "accepted Candidate authority changed ("
            + ", ".join(mismatches)
            + ")"
        )

    score = value.get("score_training", {})
    expected_score_keys = {
        "stage",
        "epochs",
        "batch_size",
        "learning_rate",
        "weight_decay",
        "gradient_clip",
        "sensor_dropout_probability",
        "mixed_precision",
        "device",
        "seed",
        "workers",
        "torch_threads",
    }
    if not isinstance(score, dict) or set(score) != expected_score_keys:
        raise base.TrainingConfigurationError(
            "Score training performance fields are invalid"
        )
    if score.get("stage") != "score_calibration" or score.get("device") != "cuda":
        raise base.TrainingConfigurationError(
            "Score performance stage/device must be score_calibration/cuda"
        )
    for name in ("epochs", "batch_size", "seed", "workers", "torch_threads"):
        if not isinstance(score.get(name), int) or isinstance(score.get(name), bool):
            raise base.TrainingConfigurationError(
                "Score performance %s must be an integer" % name
            )
    if min(score["epochs"], score["batch_size"], score["workers"], score["torch_threads"]) < 1:
        raise base.TrainingConfigurationError(
            "Score performance integer parameters must be positive"
        )
    for name in (
        "learning_rate",
        "weight_decay",
        "gradient_clip",
        "sensor_dropout_probability",
    ):
        number = score.get(name)
        if (
            isinstance(number, bool)
            or not isinstance(number, (int, float))
            or not math.isfinite(float(number))
            or float(number) < 0.0
        ):
            raise base.TrainingConfigurationError(
                "Score performance %s must be finite and non-negative" % name
            )
    if float(score["learning_rate"]) <= 0.0:
        raise base.TrainingConfigurationError(
            "Score performance learning rate must be positive"
        )
    if not 0.0 <= float(score["sensor_dropout_probability"]) <= 1.0:
        raise base.TrainingConfigurationError(
            "Score performance sensor dropout must be in [0,1]"
        )
    if not isinstance(score.get("mixed_precision"), bool):
        raise base.TrainingConfigurationError(
            "Score performance mixed_precision must be boolean"
        )

    loader = value.get("data_loader", {})
    if (
        loader.get("view_schema") != SCORE_TRAINING_VIEW_SCHEMA
        or loader.get("view_revision") != SCORE_TRAINING_VIEW_REVISION
        or loader.get("integrity_policy")
        != "full_sha256_preflight_then_full_stat_identity_per_access"
        or loader.get("modality_selective_decode") is not True
        or loader.get("unused_bev_distance_field") != "omitted"
        or not isinstance(loader.get("prefetch_factor"), int)
        or int(loader["prefetch_factor"]) < 1
        or not isinstance(loader.get("pin_memory_on_cuda"), bool)
        or not isinstance(loader.get("persistent_workers"), bool)
    ):
        raise base.TrainingConfigurationError(
            "Score data-loader performance authority is invalid"
        )
    metrics = value.get("metrics", {})
    if (
        metrics.get("transfer_policy")
        != "one_device_to_host_transfer_per_metric_per_epoch"
        or metrics.get("preserve_candidate_grouping")
        != ["overall", "maneuver", "candidate_context", "requested_gear"]
    ):
        raise base.TrainingConfigurationError(
            "Score metric transfer authority is invalid"
        )
    precision = value.get("precision", {})
    if precision != {
        "neural_encoders_and_towers": "AMP_when_enabled",
        "ackermann_rollout": "float32",
        "physical_objective": "float32",
        "hard_veto": "float32",
    }:
        raise base.TrainingConfigurationError(
            "Score precision authority is invalid"
        )
    benchmark = value.get("benchmark", {})
    batches = benchmark.get("allowed_smoke_batch_sizes")
    if (
        not isinstance(batches, list)
        or sorted(batches) != [16, 64, 128]
        or benchmark.get("maximum_samples") != base.SMOKE_MAX_SAMPLES
        or benchmark.get("maximum_steps") != base.SMOKE_MAX_STEPS
    ):
        raise base.TrainingConfigurationError(
            "Score benchmark authority is invalid"
        )
    return {
        "raw": value,
        "path": PERFORMANCE_CONFIG_PATH,
        "file_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "semantic_sha256": _canonical_sha256(value),
    }


def build_parser(authority=None):
    authority = _load_performance_authority() if authority is None else authority
    parser = base.build_parser()
    score = authority["raw"]["score_training"]
    parser.description = (
        "Train only the P5 Score Head using an accepted Candidate checkpoint "
        "and the isolated Score performance contract."
    )
    parser.set_defaults(
        stage="score_calibration",
        epochs=int(score["epochs"]),
        batch_size=int(score["batch_size"]),
        learning_rate=float(score["learning_rate"]),
        weight_decay=float(score["weight_decay"]),
        gradient_clip=float(score["gradient_clip"]),
        sensor_dropout_probability=float(score["sensor_dropout_probability"]),
        amp=bool(score["mixed_precision"]),
        device=str(score["device"]),
        seed=int(score["seed"]),
        workers=int(score["workers"]),
        torch_threads=int(score["torch_threads"]),
    )
    return parser


def _formal_parameter_mismatches(args, authority: Mapping[str, Any]) -> list:
    score = authority["raw"]["score_training"]
    actual = {
        "stage": args.stage,
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "gradient_clip": float(args.gradient_clip),
        "sensor_dropout_probability": float(args.sensor_dropout_probability),
        "mixed_precision": bool(args.amp),
        "device": str(args.device),
        "seed": int(args.seed),
        "workers": int(args.workers),
        "torch_threads": int(args.torch_threads),
    }
    return sorted(name for name, expected in score.items() if actual[name] != expected)


def _performance_attestation(authority: Mapping[str, Any]) -> dict:
    raw = authority["raw"]
    return {
        "schema": ATTESTATION_SCHEMA,
        "performance_config": str(PERFORMANCE_CONFIG_PATH),
        "performance_config_sha256": authority["file_sha256"],
        "performance_config_semantic_sha256": authority["semantic_sha256"],
        "base_candidate_trainer_sha256": sha256_file(BASE_TRAINER_PATH),
        "optimized_score_trainer_sha256": sha256_file(OPTIMIZED_TRAINER_PATH),
        "score_dataset_sha256": sha256_file(SCORE_DATASET_PATH),
        "p4_implementation_aggregate_sha256": build_p4_implementation_contract(
            ROOT
        )["aggregate_sha256"],
        "score_training_view_schema": SCORE_TRAINING_VIEW_SCHEMA,
        "score_training_view_revision": SCORE_TRAINING_VIEW_REVISION,
        "integrity_policy": raw["data_loader"]["integrity_policy"],
        "metric_transfer_policy": raw["metrics"]["transfer_policy"],
        "precision": copy.deepcopy(raw["precision"]),
    }


def _prepare_plan(args, authority):
    if args.stage != "score_calibration":
        raise base.TrainingConfigurationError(
            "optimized entry point supports only score_calibration"
        )
    if args.allow_smoke_source:
        raise base.TrainingConfigurationError(
            "optimized Score training requires an externally accepted formal Candidate"
        )
    base.TRAINER_PATH = (
        OPTIMIZED_TRAINER_PATH if args.resume is not None else BASE_TRAINER_PATH
    )
    plan = base.build_training_plan(args)
    if args.resume is None:
        acceptance = plan["source"].get("candidate_acceptance", {})
        if acceptance.get("status") != "PASS" or acceptance.get("gate_passed") is not True:
            raise base.TrainingConfigurationError(
                "optimized Score training requires a PASS Candidate acceptance"
            )
    else:
        recorded = plan["source"]["payload"].get("score_performance_contract")
        if recorded != _performance_attestation(authority):
            raise base.TrainingConfigurationError(
                "resume Score performance attestation does not match current code/authority"
            )

    bounded_smoke = args.max_samples is not None or args.max_steps is not None
    if bounded_smoke:
        benchmark = authority["raw"]["benchmark"]
        if args.max_samples is None or args.max_steps is None:
            raise base.TrainingConfigurationError(
                "optimized Score smoke requires both --max-samples and --max-steps"
            )
        if (
            int(args.max_samples) > int(benchmark["maximum_samples"])
            or int(args.max_steps) > int(benchmark["maximum_steps"])
            or int(args.batch_size) not in benchmark["allowed_smoke_batch_sizes"]
        ):
            raise base.TrainingConfigurationError(
                "optimized Score smoke exceeds its signed benchmark bounds"
            )
    else:
        mismatches = _formal_parameter_mismatches(args, authority)
        if mismatches:
            raise base.TrainingConfigurationError(
                "formal optimized Score parameters changed ("
                + ", ".join(mismatches)
                + ")"
            )
        # The frozen Candidate trainer treats batch-size 64 as an override of
        # its historical batch-size 16.  This separate authority is precisely
        # what signs that one formal execution change; all P3/P4 gates above
        # were still recomputed by the original trainer.
        expected_base_reasons = ["training_parameter_override_batch_size"]
        observed_base_reasons = base._smoke_reasons(
            args, plan["training_config"]
        )
        if observed_base_reasons != expected_base_reasons:
            raise base.TrainingConfigurationError(
                "unexpected difference from frozen Candidate training authority ("
                + ", ".join(observed_base_reasons)
                + ")"
            )
        plan["permanent_smoke"] = False
        plan["bounded_smoke_authorized"] = False
        plan["smoke_reasons"] = []

    plan["score_performance_authority"] = authority
    plan["score_performance_contract"] = _performance_attestation(authority)
    base.TRAINER_PATH = OPTIMIZED_TRAINER_PATH
    plan["trainer_sha256"] = sha256_file(OPTIMIZED_TRAINER_PATH)
    return plan


def _make_score_loaders(args, plan):
    authority = plan["score_performance_authority"]["raw"]
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
        modality=plan["modality"],
    )
    train_base = P3ScoreTrainingDatasetV1(split="train", **common)
    validation_base = P3ScoreTrainingDatasetV1(split="validation", **common)
    train = base._cap_dataset(train_base, args.max_samples)
    validation = base._cap_dataset(validation_base, args.max_samples)
    generator = torch.Generator().manual_seed(args.seed)
    loader_authority = authority["data_loader"]
    loader_common = dict(
        batch_size=args.batch_size,
        num_workers=args.workers,
        collate_fn=p3_training_collate,
        worker_init_fn=p3_training_worker_init,
        pin_memory=(
            plan["device"].type == "cuda"
            and bool(loader_authority["pin_memory_on_cuda"])
        ),
        persistent_workers=(
            args.workers > 0 and bool(loader_authority["persistent_workers"])
        ),
    )
    if args.workers > 0:
        loader_common["prefetch_factor"] = int(
            loader_authority["prefetch_factor"]
        )
    train_loader = DataLoader(
        train, shuffle=True, generator=generator, **loader_common
    )
    validation_loader = DataLoader(validation, shuffle=False, **loader_common)
    return train_base, validation_base, train_loader, validation_loader


class _EpochTransferMetricAccumulator:
    """Retain detached metrics on device and transfer once at epoch end."""

    scalar_loss_names = base._MetricAccumulator.scalar_loss_names

    def __init__(self):
        self.weight = 0
        self.seen = 0
        self.valid = 0
        self.steps = 0
        self.loss_sums = {name: None for name in self.scalar_loss_names}
        self.clearance_sum = None
        self.metric_chunks = defaultdict(list)
        self.grouping = {
            "maneuver": [],
            "candidate_context": [],
            "requested_gear": [],
        }

    def note_geometry(self, valid, seen):
        self.valid += int(valid)
        self.seen += int(seen)

    @staticmethod
    def _add(current, value):
        value = value.detach()
        return value.clone() if current is None else current + value

    def update(self, output, losses, batch_size, grouping):
        self.weight += int(batch_size)
        self.steps += 1
        for name in self.scalar_loss_names:
            self.loss_sums[name] = self._add(
                self.loss_sums[name], losses[name] * int(batch_size)
            )
        per_frame_clearance = losses["minimum_clearance"].detach().mean(dim=1)
        self.clearance_sum = self._add(
            self.clearance_sum, per_frame_clearance.sum()
        )
        metrics = candidate_batch_metrics(output, losses)
        for name, values in metrics.items():
            self.metric_chunks[name].append(values.detach())
        for name in self.grouping:
            self.grouping[name].extend(grouping[name])

    def result(self):
        if self.weight < 1 or self.clearance_sum is None:
            raise RuntimeError("no Score metrics were accumulated")
        scalar_names = list(self.scalar_loss_names) + ["minimum_clearance"]
        scalar_values = [self.loss_sums[name] for name in self.scalar_loss_names]
        scalar_values.append(self.clearance_sum)
        transferred = torch.stack(scalar_values).detach().cpu().tolist()
        scalar = dict(zip(scalar_names, transferred))
        result = {
            name: scalar[name] / float(self.weight)
            for name in self.scalar_loss_names
        }
        result.update(
            {
                "mean_candidate_minimum_clearance_m": scalar[
                    "minimum_clearance"
                ]
                / float(self.weight),
                "geometry_valid_fraction": self.valid / float(max(self.seen, 1)),
                "samples_seen": self.seen,
                "samples_optimized": self.weight,
                "batches": self.steps,
            }
        )
        metrics = {
            name: torch.cat(chunks, dim=0).cpu()
            for name, chunks in self.metric_chunks.items()
        }
        overall = CandidateMetricAccumulator()
        context = CandidateMetricAccumulator()
        gear = CandidateMetricAccumulator()
        overall.update(metrics, self.grouping["maneuver"])
        context.update(metrics, self.grouping["candidate_context"])
        gear.update(metrics, self.grouping["requested_gear"])
        summary = overall.compute()
        result["candidate_metrics"] = {
            "overall": summary["overall"],
            "by_maneuver": summary["by_maneuver"],
            "by_candidate_context": context.compute()["by_maneuver"],
            "by_requested_gear": gear.compute()["by_maneuver"],
        }
        return result


def _assert_finite_async(value):
    finite = torch.isfinite(value)
    if hasattr(torch, "_assert_async"):
        torch._assert_async(finite, "training loss became non-finite")
    elif not bool(finite):  # pragma: no cover - old Noetic PyTorch fallback
        raise FloatingPointError("training loss became non-finite")


def _run_score_epoch(
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
):
    base._set_training_mode(model, plan["stage"], plan["modality"], training)
    accumulator = _EpochTransferMetricAccumulator()
    completed = True
    for raw_index, raw_batch in enumerate(loader):
        selected, grouping, valid_count, seen_count = base._select_valid_geometry(
            raw_batch
        )
        accumulator.note_geometry(valid_count, seen_count)
        if selected is None:
            continue
        batch = base._to_device(selected, plan["device"])
        if training:
            optimizer.zero_grad(set_to_none=True)
            output, losses = base._forward_objective(
                model,
                objective,
                batch,
                stage=plan["stage"],
                mode=plan["modality"],
                sensor_dropout_probability=args.sensor_dropout_probability,
                training=True,
                amp_enabled=plan["amp_enabled"],
            )
            _assert_finite_async(losses["total"])
            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimizer)
            if args.gradient_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(
                    (
                        parameter
                        for parameter in model.parameters()
                        if parameter.requires_grad
                    ),
                    args.gradient_clip,
                )
            scaler.step(optimizer)
            scaler.update()
        else:
            with torch.no_grad():
                output, losses = base._forward_objective(
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


def _decorate_score_artifacts(paths, plan):
    checkpoint_path = Path(paths["checkpoint"])
    contract_path = Path(paths["contract"])
    payload = base._load_checkpoint(checkpoint_path)
    payload["score_performance_contract"] = copy.deepcopy(
        plan["score_performance_contract"]
    )
    base._atomic_torch_save(checkpoint_path, payload)
    contract = base._load_json(contract_path)
    contract["checkpoint_sha256"] = sha256_file(checkpoint_path)
    contract["score_performance_contract"] = copy.deepcopy(
        plan["score_performance_contract"]
    )
    contract.setdefault("dataset_provenance", {}).update(
        {
            "score_training_view": SCORE_TRAINING_VIEW_SCHEMA,
            "score_training_view_revision": SCORE_TRAINING_VIEW_REVISION,
            "score_integrity_policy": plan["score_performance_contract"][
                "integrity_policy"
            ],
        }
    )
    contract.setdefault("training_run", {}).update(
        {
            "score_performance_config_sha256": plan[
                "score_performance_contract"
            ]["performance_config_sha256"],
            "score_dataset_sha256": plan["score_performance_contract"][
                "score_dataset_sha256"
            ],
            "metric_transfer_policy": plan["score_performance_contract"][
                "metric_transfer_policy"
            ],
        }
    )
    base._atomic_write_json(contract_path, contract)
    return paths


@contextmanager
def _optimized_runtime(plan):
    original_loader = base._make_loaders
    original_epoch = base._run_epoch
    original_writer = base._write_artifacts
    original_trainer_path = base.TRAINER_PATH

    def write_artifacts(**kwargs):
        paths = original_writer(**kwargs)
        return _decorate_score_artifacts(paths, kwargs["plan"])

    base._make_loaders = _make_score_loaders
    base._run_epoch = _run_score_epoch
    base._write_artifacts = write_artifacts
    base.TRAINER_PATH = OPTIMIZED_TRAINER_PATH
    try:
        yield
    finally:
        base._make_loaders = original_loader
        base._run_epoch = original_epoch
        base._write_artifacts = original_writer
        base.TRAINER_PATH = original_trainer_path


def _public_plan(args, plan):
    public = base._public_plan(args, plan)
    formal = not plan["permanent_smoke"] and not _formal_parameter_mismatches(
        args, plan["score_performance_authority"]
    )
    public.update(
        {
            "status": "DRY_RUN_READY" if formal else "BOUNDED_SMOKE_READY",
            "formal_training_authorized": formal,
            "optimized_score_training": True,
            "trainer_sha256": plan["trainer_sha256"],
            "score_performance_contract": plan["score_performance_contract"],
            "batch_size": int(args.batch_size),
            "prefetch_factor": int(
                plan["score_performance_authority"]["raw"]["data_loader"][
                    "prefetch_factor"
                ]
            ),
        }
    )
    return public


def main(argv=None):
    authority = _load_performance_authority()
    parser = build_parser(authority)
    args = parser.parse_args(argv)
    try:
        plan = _prepare_plan(args, authority)
        if args.dry_run:
            print(json.dumps(_public_plan(args, plan), indent=2, sort_keys=True))
            return 0
        with _optimized_runtime(plan):
            result = base.run_training(args, plan)
    except (ValueError, OSError, RuntimeError, FloatingPointError) as exc:
        parser.error(str(exc))
    result["optimized_score_training"] = True
    result["score_performance_config_sha256"] = authority["file_sha256"]
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
