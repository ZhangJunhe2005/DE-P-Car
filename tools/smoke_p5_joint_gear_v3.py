#!/usr/bin/env python3
"""Bounded CUDA/AMP gradient and throughput smoke test for DEPCarNetV3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dep_car/src"))
sys.path.insert(0, str(ROOT / "tools"))

from dep_car.model.dep_car_net_v3 import DEPCarNetV3
from dep_car.training.losses_v3 import DEPCarJointGearLossConfigV3, DEPCarObjectiveV3
import train_dep_car_joint_gear_v3 as trainer


def make_batch(batch, device):
    depth = torch.zeros(batch, 2, 96, 160, device=device)
    depth[:, 0] = 0.55
    depth[:, 1] = 1.0
    lidar = torch.zeros(batch, 6, 160, 160, device=device)
    lidar[:, 5] = 1.0
    state = torch.zeros(batch, 9, device=device)
    state[:, 0] = torch.linspace(-0.15, 0.35, batch, device=device)
    state[:, 2] = torch.linspace(-0.12, 0.12, batch, device=device)
    current = torch.where(
        torch.arange(batch, device=device) % 2 == 0,
        torch.ones(batch, dtype=torch.long, device=device),
        -torch.ones(batch, dtype=torch.long, device=device),
    )
    history = torch.zeros(batch, 6, device=device)
    history[:, 0] = current
    history[:, 4] = torch.arange(batch, device=device) % 4
    history[:, 5] = (torch.arange(batch, device=device) % 3 == 0).float()
    route = torch.zeros(batch, 24, 3, device=device)
    route[:, :, 0] = torch.linspace(0.0, 3.0, 24, device=device)
    route[:, :, 1] = 0.35 * torch.sin(
        torch.linspace(0.0, 1.5, 24, device=device)
    )
    route[:, :, 2] = 0.18
    route_mask = torch.ones(batch, 24, dtype=torch.bool, device=device)
    sequence = torch.tensor(
        [1, -1, 1, -1, 1, -1], dtype=torch.long, device=device
    )[None].expand(batch, -1).clone()
    sequence[1::2] *= -1
    sequence_mask = torch.ones(batch, 6, dtype=torch.bool, device=device)
    return {
        "depth": depth,
        "lidar_bev": lidar,
        "state": state,
        "current_gear": current,
        "gear_history": history,
        "route_pose": route,
        "route_mask": route_mask,
        "modality_mask": torch.ones(batch, 2, device=device),
        "map_distance_field": torch.full(
            (batch, 1, 220, 220), 4.0, device=device
        ),
        "map_resolution": torch.full((batch,), 0.05, device=device),
        "map_origin": torch.full((batch, 2), -5.5, device=device),
        "chassis_to_map": torch.eye(4, device=device)[None].expand(batch, -1, -1).clone(),
        "sequence_gears": sequence,
        "sequence_mask": sequence_mask,
    }


def calculate(model, objective, batch, stage, amp):
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
        output = model(
            batch["depth"], batch["lidar_bev"], batch["state"],
            batch["current_gear"], batch["gear_history"],
            batch["route_pose"], batch["route_mask"], batch["modality_mask"],
        )
    with torch.autocast(device_type="cuda", enabled=False):
        losses = objective(
            output,
            map_distance_field=batch["map_distance_field"],
            map_resolution=batch["map_resolution"],
            map_origin=batch["map_origin"],
            chassis_to_map=batch["chassis_to_map"],
            route=batch["route_pose"], route_mask=batch["route_mask"],
            gear_history=batch["gear_history"],
            sequence_gears=batch["sequence_gears"],
            sequence_mask=batch["sequence_mask"], stage=stage,
        )
    return output, losses


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=4)
    args = parser.parse_args(argv)
    if min(args.batch_size, args.steps) < 1 or args.batch_size > 32 or args.steps > 16:
        raise SystemExit("bounded smoke limits are batch<=32 and steps<=16")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    config = yaml.safe_load(
        trainer.CONFIG.read_text(encoding="utf-8")
    )
    source = torch.load(
        trainer.resolve(config["initialization"]["v2_score"]),
        map_location="cpu", weights_only=True,
    )
    device = torch.device("cuda")
    torch.manual_seed(int(config["training"]["seed"]))
    model = DEPCarNetV3()
    model.initialize_from_v2(source["model_state_dict"])
    model.to(device)
    objective = DEPCarObjectiveV3(
        DEPCarJointGearLossConfigV3(**config["loss"])
    )
    batch = make_batch(args.batch_size, device)
    stages = {}
    for stage in trainer.STAGES:
        ownership, selected = trainer.configure_stage(model, stage)
        optimizer = torch.optim.AdamW(selected, lr=1.0e-4)
        optimizer.zero_grad(set_to_none=True)
        output, losses = calculate(model, objective, batch, stage, True)
        losses["total"].backward()
        gradients = [
            parameter.grad for parameter in selected if parameter.grad is not None
        ]
        finite = bool(
            torch.isfinite(losses["total"]).all()
            and torch.isfinite(output.scores).all()
            and gradients
            and all(torch.isfinite(gradient).all() for gradient in gradients)
        )
        if not finite:
            raise FloatingPointError("non-finite V3 AMP gradient: %s" % stage)
        optimizer.step()
        stages[stage] = {
            **ownership,
            "loss": float(losses["total"].detach()),
            "gradient_tensors": len(gradients),
            "finite": True,
        }

    model.eval()
    with torch.inference_mode():
        fp32, _ = calculate(model, objective, batch, "joint_smoke", False)
        amp, _ = calculate(model, objective, batch, "joint_smoke", True)
    score_abs = float((fp32.scores - amp.scores).abs().max())
    trajectory_abs = float(
        (fp32.trajectories - amp.trajectories).abs().max()
    )
    if score_abs > 0.10 or trajectory_abs > 0.015:
        raise RuntimeError("V3 AMP/FP32 consistency tolerance failed")

    torch.cuda.reset_peak_memory_stats()
    for _ in range(2):
        with torch.inference_mode():
            calculate(model, objective, batch, "joint_smoke", True)
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(args.steps):
        with torch.inference_mode():
            calculate(model, objective, batch, "joint_smoke", True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    report = {
        "schema": "DEPCarJointGearV3CUDASmokeV1",
        "status": "PASS",
        "device": torch.cuda.get_device_name(0),
        "batch_size": args.batch_size,
        "steps": args.steps,
        "samples_per_second": args.batch_size * args.steps / elapsed,
        "milliseconds_per_batch": 1000.0 * elapsed / args.steps,
        "peak_memory_mib": torch.cuda.max_memory_allocated() / (1024.0 ** 2),
        "amp_fp32_max_score_abs": score_abs,
        "amp_fp32_max_trajectory_abs": trajectory_abs,
        "stages": stages,
        "test_split_accessed": False,
        "bounded_smoke": True,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
