#!/usr/bin/env python3
"""Compare the frozen P5 and optimized Score data/forward paths on CUDA."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(ROOT / "dep_car/src"))

import train_dep_car as base

from dep_car.model.checkpoint import sha256_file
from dep_car.model.dep_car_net import DEPCarNetV1
from dep_car.training.losses import DEPCarObjectiveV1
from dep_car.training.p4_dataset import P3TrainingDatasetV1, p3_training_collate
from dep_car.training.score_dataset import P3ScoreTrainingDatasetV1


MODALITIES = ("depth_only", "lidar_only", "fusion")
CANDIDATE_ROOT = ROOT / "models/dep_car/p5_v2"


def candidate_path(modality):
    return CANDIDATE_ROOT / (modality + "_candidate_capacity.best.pth")


def _maximum_absolute(first, second):
    if first.dtype == torch.bool or not first.dtype.is_floating_point:
        return 0.0 if torch.equal(first, second) else float("inf")
    return float((first.float() - second.float()).abs().max().cpu())


def _compare_tensor_mapping(first, second, prefix=""):
    output = {}
    for name in sorted(set(first).intersection(second)):
        left = first[name]
        right = second[name]
        qualified = prefix + name
        if torch.is_tensor(left) and torch.is_tensor(right):
            if left.shape != right.shape or left.dtype != right.dtype:
                output[qualified] = float("inf")
            else:
                output[qualified] = _maximum_absolute(left, right)
        elif isinstance(left, dict) and isinstance(right, dict):
            output.update(
                _compare_tensor_mapping(left, right, prefix=qualified + ".")
            )
    return output


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports/p5_score_optimized_numerics.json",
    )
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.batch_size > 16:
        parser.error("--batch-size must be in [1,16]")
    if not torch.cuda.is_available():
        parser.error("CUDA is required")

    authority = base._training_config_authority()
    common = dict(
        sample_root=authority["authority_paths"]["root"],
        maps_root=authority["authority_paths"]["maps"],
        split="validation",
        index_path=authority["authority_paths"]["index"],
        index_splits=("train", "validation"),
        workers=authority["dataset"]["workers"],
        allow_test=False,
        depth_dropout_probability=0.0,
        lidar_dropout_probability=0.0,
        augmentation_seed=authority["training"]["seed"],
        expected_map_contract_aggregate_sha256=authority["dataset"][
            "map_contract_aggregate_sha256"
        ],
    )
    original_dataset = P3TrainingDatasetV1(**common)
    indices = list(range(min(args.batch_size, len(original_dataset))))
    original_items = [original_dataset[index] for index in indices]
    original_raw = p3_training_collate(original_items)
    device = torch.device("cuda")
    rows = {}
    tolerance = 1.0e-6

    for modality in MODALITIES:
        optimized_dataset = P3ScoreTrainingDatasetV1(
            modality=modality, **common
        )
        optimized_items = [optimized_dataset[index] for index in indices]
        optimized_raw = p3_training_collate(optimized_items)
        original_selected, original_grouping, _, _ = base._select_valid_geometry(
            original_raw
        )
        optimized_selected, optimized_grouping, _, _ = base._select_valid_geometry(
            optimized_raw
        )
        if original_selected is None or optimized_selected is None:
            parser.error("selected numerical batch has no geometry-valid samples")
        if original_grouping != optimized_grouping:
            parser.error("optimized grouping differs from frozen P4 grouping")

        used_inputs = (
            "state",
            "requested_gear",
            "geometry_valid",
            "route_pose",
            "route_mask",
            "map_distance_field",
            "map_resolution",
            "map_origin",
            "chassis_to_map",
        )
        input_differences = {
            name: _maximum_absolute(
                original_selected[name], optimized_selected[name]
            )
            for name in used_inputs
        }
        if modality != "lidar_only":
            input_differences["depth"] = _maximum_absolute(
                original_selected["depth"], optimized_selected["depth"]
            )
        if modality != "depth_only":
            input_differences["lidar_bev"] = _maximum_absolute(
                original_selected["lidar_bev"], optimized_selected["lidar_bev"]
            )

        checkpoint = candidate_path(modality)
        acceptance = base.evaluate_candidate_acceptance(checkpoint)
        if acceptance.get("status") != "PASS" or acceptance.get("gate_passed") is not True:
            parser.error("Candidate acceptance failed for " + modality)
        payload = base._load_checkpoint(checkpoint)
        model = DEPCarNetV1().to(device)
        model.load_state_dict(payload["model_state_dict"], strict=True)
        ownership = base._configure_stage_and_modality(
            model, "score_calibration", modality
        )
        model.eval()
        objective = DEPCarObjectiveV1(authority["loss_config"])
        frozen_batch = base._to_device(original_selected, device)
        optimized_batch = base._to_device(optimized_selected, device)
        with torch.no_grad():
            frozen_output, frozen_losses = base._forward_objective(
                model,
                objective,
                frozen_batch,
                stage="score_calibration",
                mode=modality,
                sensor_dropout_probability=0.0,
                training=False,
                amp_enabled=True,
            )
            optimized_output, optimized_losses = base._forward_objective(
                model,
                objective,
                optimized_batch,
                stage="score_calibration",
                mode=modality,
                sensor_dropout_probability=0.0,
                training=False,
                amp_enabled=True,
            )
        torch.cuda.synchronize()
        output_differences = {
            name: _maximum_absolute(left, right)
            for name, left, right in zip(
                frozen_output._fields, frozen_output, optimized_output
            )
        }
        loss_differences = _compare_tensor_mapping(
            frozen_losses, optimized_losses
        )
        maximum = max(
            list(input_differences.values())
            + list(output_differences.values())
            + list(loss_differences.values())
        )
        rows[modality] = {
            "status": "PASS" if maximum <= tolerance else "FAIL",
            "frames": int(len(optimized_batch["state"])),
            "input_max_abs": input_differences,
            "output_max_abs": output_differences,
            "loss_max_abs": loss_differences,
            "maximum_absolute_difference": maximum,
            "score_trainable_parameters": ownership["effective_trainable"],
            "candidate_acceptance_evaluation_sha256": acceptance[
                "evaluation_sha256"
            ],
            "candidate_checkpoint_sha256": sha256_file(checkpoint),
        }

    status = "PASS" if all(row["status"] == "PASS" for row in rows.values()) else "FAIL"
    report = {
        "schema": "DEPCarP5ScoreOptimizedNumericalEquivalenceV1",
        "status": status,
        "device": torch.cuda.get_device_name(0),
        "amp_enabled": True,
        "absolute_tolerance": tolerance,
        "test_split_opened": False,
        "indices": indices,
        "modalities": rows,
        "tool_sha256": sha256_file(Path(__file__)),
    }
    base._atomic_write_json(args.output.resolve(), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
