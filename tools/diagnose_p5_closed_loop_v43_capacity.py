#!/usr/bin/env python3
"""Measure the physical lower bound of V4.3 hard-feasibility gates."""

import argparse
import json
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dep_car/src"))
sys.path.insert(0, str(ROOT / "tools"))

from dep_car.model.dep_car_net_v3 import DEPCarNetV3
from dep_car.model.dep_car_net_v43 import DEPCarNetV43
from dep_car.training.losses import (
    swept_footprint_clearance,
    swept_map_footprint_clearance,
)
from dep_car.training.losses_v43 import DEPCarObjectiveV43
from dep_car.training.p4_dataset import p3_training_collate, p3_training_worker_init
from dep_car.training.v43_dataset import P3ClosedLoopSequenceDatasetV43
import train_dep_car_closed_loop_v43 as trainer
import train_dep_car_hybrid_sequence_v4 as v4


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.workers < 1:
        raise SystemExit("workers must be positive")

    config, _config_sha, model_config, rollout_config, loss_config = trainer.load_config()
    authority, data_gate = trainer.verify_data(config)
    source_path, source, source_gate = trainer.verify_source(config)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    model = DEPCarNetV43(
        base_model=DEPCarNetV3(),
        sequence_config=model_config,
        rollout_config=rollout_config,
        residual_score_span=float(config["model"]["residual_score_span"]),
    )
    model.initialize_from_v4(source["model_state_dict"])
    model.freeze_base()
    model.to(device).eval()

    dataset = P3ClosedLoopSequenceDatasetV43(
        sample_root=authority["sample_root"], maps_root=authority["maps_root"],
        split="validation", index_path=authority["training_index"],
        index_splits=("train", "validation"), workers=args.workers,
        expected_map_contract_aggregate_sha256=authority["map_contract_aggregate_sha256"],
        expected_index_sha256=authority["training_index_sha256"], modality="fusion",
        sequence_index_path=authority["sequence_index"],
        expected_sequence_index_sha256=authority["sequence_index_sha256"],
    )
    loader_args = dict(
        batch_size=int(config["training"]["batch_size"]), shuffle=False,
        num_workers=args.workers, pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0, collate_fn=p3_training_collate,
        worker_init_fn=p3_training_worker_init,
    )
    if args.workers > 0:
        loader_args["prefetch_factor"] = int(config["training"]["prefetch_factor"])
    loader = DataLoader(dataset, **loader_args)
    objective = DEPCarObjectiveV43(loss_config)

    counts = {
        "samples": 0, "initial_pose_hard_safe": 0,
        "any_hard_feasible": 0, "any_viable": 0,
        "no_hard_feasible_with_safe_initial_pose": 0,
        "selected_hard_feasible": 0, "selected_viable": 0,
        "initial_pose_hard_unsafe": 0, "unsafe_initial_with_egress_candidate": 0,
        "selected_egress_when_available": 0,
        "candidate_slots": 0,
        "global_hard_candidate_slots": 0,
        "local_hard_candidate_slots": 0,
        "local_and_global_hard_candidate_slots": 0,
        "global_egress_candidate_slots_on_unsafe_rows": 0,
        "local_egress_candidate_slots_on_unsafe_rows": 0,
        "local_and_global_egress_candidate_slots_on_unsafe_rows": 0,
        "unsafe_rows_with_local_global_egress_intersection": 0,
        "unsafe_rows_where_local_clearance_argmax_is_global_egress": 0,
    }
    amp = bool(config["training"]["mixed_precision"]) and device.type == "cuda"
    with torch.inference_mode():
        for host in loader:
            batch = v4.select_valid(host, device)
            if batch is None:
                continue
            output, losses = v4.forward_loss(
                model, objective, batch, trainer.STAGE, amp
            )
            clearance = swept_map_footprint_clearance(
                output.trajectories.float(),
                batch["map_distance_field"].float(),
                batch["map_resolution"].float(),
                batch["map_origin"].float(),
                batch["chassis_to_map"].float(),
            )
            initial_clearance = clearance[:, 0, 0].amin(dim=-1)
            initial = initial_clearance > 0.0
            future_floor = clearance[:, :, 1:].amin(dim=(-1, -2))
            terminal = clearance[:, :, -1].amin(dim=-1)
            egress = (
                (future_floor >= initial_clearance[:, None] - 0.03)
                & (
                    terminal
                    >= torch.maximum(
                        initial_clearance[:, None] + 0.05,
                        torch.zeros_like(terminal),
                    )
                )
            )
            local_clearance = swept_footprint_clearance(
                output.trajectories.float(),
                batch["bev_distance_field"].float(),
                extent_m=float(config["model"]["local_distance_extent_m"]),
            )
            local_initial_clearance = local_clearance[:, 0, 0].amin(dim=-1)
            local_future_floor = local_clearance[:, :, 1:].amin(dim=(-1, -2))
            local_terminal = local_clearance[:, :, -1].amin(dim=-1)
            local_egress = (
                (
                    local_future_floor
                    >= local_initial_clearance[:, None] - 0.03
                )
                & (
                    local_terminal
                    >= torch.maximum(
                        local_initial_clearance[:, None] + 0.05,
                        torch.zeros_like(local_terminal),
                    )
                )
            )
            unsafe = ~initial
            any_egress = egress.any(dim=1)
            feasible = losses["hard_feasible"].bool()
            local_hard = local_clearance.amin(dim=(-1, -2)) > 0.0
            viable = losses["viable"].bool()
            any_feasible = feasible.any(dim=1)
            any_viable = viable.any(dim=1)
            selected = output.scores.argmin(dim=1)
            selected_hard = feasible.gather(1, selected[:, None]).squeeze(1)
            selected_viable = viable.gather(1, selected[:, None]).squeeze(1)
            count = len(initial)
            counts["samples"] += count
            counts["initial_pose_hard_safe"] += int(initial.sum())
            counts["any_hard_feasible"] += int(any_feasible.sum())
            counts["any_viable"] += int(any_viable.sum())
            counts["no_hard_feasible_with_safe_initial_pose"] += int(
                (initial & ~any_feasible).sum()
            )
            counts["selected_hard_feasible"] += int(selected_hard.sum())
            counts["selected_viable"] += int(selected_viable.sum())
            counts["initial_pose_hard_unsafe"] += int(unsafe.sum())
            counts["unsafe_initial_with_egress_candidate"] += int(
                (unsafe & any_egress).sum()
            )
            selected_egress = egress.gather(1, selected[:, None]).squeeze(1)
            counts["selected_egress_when_available"] += int(
                (unsafe & any_egress & selected_egress).sum()
            )
            unsafe_rows = unsafe[:, None]
            local_global_egress = unsafe_rows & egress & local_egress
            counts["candidate_slots"] += int(feasible.numel())
            counts["global_hard_candidate_slots"] += int(feasible.sum())
            counts["local_hard_candidate_slots"] += int(local_hard.sum())
            counts["local_and_global_hard_candidate_slots"] += int(
                (local_hard & feasible).sum()
            )
            counts["global_egress_candidate_slots_on_unsafe_rows"] += int(
                (unsafe_rows & egress).sum()
            )
            counts["local_egress_candidate_slots_on_unsafe_rows"] += int(
                (unsafe_rows & local_egress).sum()
            )
            counts[
                "local_and_global_egress_candidate_slots_on_unsafe_rows"
            ] += int(local_global_egress.sum())
            counts["unsafe_rows_with_local_global_egress_intersection"] += int(
                local_global_egress.any(dim=1).sum()
            )
            local_best = local_terminal.argmax(dim=1)
            local_best_is_global_egress = egress.gather(
                1, local_best[:, None]
            ).squeeze(1)
            counts[
                "unsafe_rows_where_local_clearance_argmax_is_global_egress"
            ] += int((unsafe & any_egress & local_best_is_global_egress).sum())

    source_metrics = v4.epoch_loop(
        model, objective, loader, trainer.STAGE, device, amp,
        progress_interval=1000,
    )

    samples = max(1, counts["samples"])
    safe_initial = max(1, counts["initial_pose_hard_safe"])
    unsafe_with_egress = max(1, counts["unsafe_initial_with_egress_candidate"])
    report = {
        "schema": "DEPCarV43CapacityDiagnosticV1",
        "status": "PASS",
        "checkpoint": str(source_path),
        "checkpoint_sha256": trainer.sha256_file(source_path),
        "data_authority_gate": data_gate,
        "source_gate": source_gate,
        "counts": counts,
        "source_metrics": source_metrics,
        "rates": {
            "initial_pose_hard_unsafe_rate": 1.0 - counts["initial_pose_hard_safe"] / samples,
            "zero_hard_feasible_rate": 1.0 - counts["any_hard_feasible"] / samples,
            "zero_hard_feasible_rate_given_safe_initial_pose": counts[
                "no_hard_feasible_with_safe_initial_pose"
            ] / safe_initial,
            "any_viable_rate": counts["any_viable"] / samples,
            "selected_hard_feasible_rate": counts["selected_hard_feasible"] / samples,
            "selected_viable_rate": counts["selected_viable"] / samples,
            "unsafe_initial_egress_candidate_rate": counts[
                "unsafe_initial_with_egress_candidate"
            ] / max(1, counts["initial_pose_hard_unsafe"]),
            "selected_egress_rate_when_available": counts[
                "selected_egress_when_available"
            ] / unsafe_with_egress,
            "local_hard_precision_against_map": counts[
                "local_and_global_hard_candidate_slots"
            ] / max(1, counts["local_hard_candidate_slots"]),
            "local_hard_recall_against_map": counts[
                "local_and_global_hard_candidate_slots"
            ] / max(1, counts["global_hard_candidate_slots"]),
            "local_egress_precision_against_map_on_unsafe_rows": counts[
                "local_and_global_egress_candidate_slots_on_unsafe_rows"
            ] / max(1, counts["local_egress_candidate_slots_on_unsafe_rows"]),
            "local_egress_recall_against_map_on_unsafe_rows": counts[
                "local_and_global_egress_candidate_slots_on_unsafe_rows"
            ] / max(1, counts["global_egress_candidate_slots_on_unsafe_rows"]),
            "unsafe_row_local_global_egress_intersection_rate": counts[
                "unsafe_rows_with_local_global_egress_intersection"
            ] / unsafe_with_egress,
            "local_clearance_argmax_global_egress_rate": counts[
                "unsafe_rows_where_local_clearance_argmax_is_global_egress"
            ] / unsafe_with_egress,
        },
        "test_split_accessed": False,
    }
    output = ROOT / "reports/p5_closed_loop_v43_capacity_diagnostic.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
