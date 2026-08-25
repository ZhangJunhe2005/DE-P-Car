#!/usr/bin/env python3
"""Bounded CUDA/CPU smoke test for the V4.2 P6 ROS execution boundary."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dep_car/src"))

from dep_car.core.occupancy import OccupancyGrid2D
from dep_car.core.types import Candidate, Gear
from dep_car.runtime.p6_policy import P6PolicyRuntime
from dep_car.runtime.safety import evaluate_hybrid_sequence_candidate_bank


def resolve(path):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mode", choices=("shadow", "guarded"), default="shadow")
    args = parser.parse_args()
    checkpoint = resolve(
        "models/dep_car/p5_hybrid_sequence_v41/"
        "fusion_hierarchical_score.best.pth"
    )
    contract = checkpoint.with_suffix(".contract.json")
    authority = resolve(
        "models/dep_car/p5_hybrid_sequence_v42/"
        + (
            "fusion_guarded_simulation.authority.json"
            if args.mode == "guarded"
            else "fusion_calibrated_execution.authority.json"
        )
    )
    runtime = P6PolicyRuntime(
        checkpoint,
        contract,
        modality="fusion",
        device=args.device,
        mode=args.mode,
        p6_authority=authority,
    )
    route = np.zeros((80, 3), dtype=np.float32)
    route[:32, 0] = np.linspace(0.05, 3.0, 32)
    route_mask = np.zeros(80, dtype=bool)
    route_mask[:32] = True
    latencies = []
    for _ in range(3):
        output = runtime.infer(
            np.ones((2, 96, 160), dtype=np.float32),
            np.zeros((6, 160, 160), dtype=np.float32),
            np.asarray([0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32),
            route_pose=route,
            route_mask=route_mask,
            current_gear=0,
            gear_history=np.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        )
        latencies.append(float(runtime.last_latency_ms))
    candidates = []
    for candidate_id in range(15):
        active = output.action_gears[candidate_id][output.action_mask[candidate_id]]
        first = int(active[0]) if len(active) else 1
        value = Candidate(
            candidate_id=candidate_id,
            speed_anchor=float(output.controls[candidate_id, 0, 2]),
            steering_anchor=float(output.controls[candidate_id, 0, 1]),
            duration=float(output.controls[candidate_id, 0, 3]),
            trajectory=output.trajectories[candidate_id],
            gear=Gear.require_drive(first),
            learned_score=float(output.scores[candidate_id]),
        )
        value.action_gears = output.action_gears[candidate_id]
        value.action_mask = output.action_mask[candidate_id]
        value.action_durations = output.controls[candidate_id, :, 3]
        value.shift_required = output.shift_required[candidate_id]
        value.transition_duration = output.transition_duration[candidate_id]
        value.motion_gears = output.motion_gears[candidate_id]
        candidates.append(value)
    grid = OccupancyGrid2D(
        np.zeros((400, 400), dtype=np.int16), 0.05, (-10.0, -10.0)
    )
    result = evaluate_hybrid_sequence_candidate_bank(
        candidates, (2.0, 0.0), grid
    )
    report = {
        "schema": "DEPCarP6V42RuntimeSmokeV1",
        "status": "PASS" if result.executable else "FAIL",
        "architecture_id": runtime.architecture_id,
        "checkpoint_sha256": runtime.checkpoint_sha256,
        "authority_sha256": runtime.execution_authority_sha256,
        "device": str(runtime.device),
        "mode": args.mode,
        "cold_latency_ms": latencies[0],
        "steady_latency_ms": latencies[-1],
        "candidate_count": len(result.candidates),
        "hard_feasible_count": sum(value.feasible for value in result.candidates),
        "selected_candidate_id": (
            result.selected.candidate_id if result.selected is not None else -1
        ),
        "selected_action_gears": (
            result.selected.action_gears[result.selected.action_mask].tolist()
            if result.selected is not None
            else []
        ),
        "veto_reasons": {
            str(value.candidate_id): value.veto_reason
            for value in result.candidates
            if not value.feasible
        },
        "control_authorized": runtime.control_authorized,
        "mandatory_hard_veto_applied": True,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
