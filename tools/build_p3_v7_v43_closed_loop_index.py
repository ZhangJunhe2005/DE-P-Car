#!/usr/bin/env python3
"""Build and seal exact signed-plan labels at V4.3 re-observed states.

The guarded V4.2 policy visits the states.  At index time only, a deterministic
bidirectional Hybrid A* expert labels a complete macro gear sequence from each
visited pose to the end of the current route corridor.  Hybrid A* is therefore
an offline DAgger oracle, never a deployment input or controller.
"""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dep_car/src"))
sys.path.insert(0, str(ROOT / "tools"))
from dep_car.core.types import Gear
from dep_car.global_planner.hybrid_astar import HybridAStar, HybridAStarConfig
from dep_car.training.p4_dataset import (
    build_training_index,
    indexed_map_contract_aggregate,
    _load_map_catalog,
)
import audit_p3_footprint_upgrade as geometry


INDEX_SCHEMA = "DEPCarV43ClosedLoopSequenceIndexV2"
AUTHORITY_SCHEMA = "DEPCarV43ClosedLoopDataAuthorityV2"
SEQUENCE_AUTHORITY = "REOBSERVED_STATE_EXACT_SIGNED_HYBRID_ASTAR_PLAN"
ACTIONS = 6
_WORKER_MAPS = {}
_WORKER_MAPS_ROOT = None


def resolve(value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value):
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def wrap_angle(value):
    return float(math.atan2(math.sin(float(value)), math.cos(float(value))))


def _initialize_worker(maps_root):
    global _WORKER_MAPS_ROOT
    _WORKER_MAPS_ROOT = Path(maps_root)
    cv2.setNumThreads(1)


def _worker_grid(map_uuid):
    if map_uuid not in _WORKER_MAPS:
        specs, _contract = geometry.load_map_specs(
            _WORKER_MAPS_ROOT, {map_uuid}, expected_aggregate=None
        )
        _WORKER_MAPS[map_uuid] = geometry.grid_from_spec(specs[map_uuid])
    return _WORKER_MAPS[map_uuid]


def _pose_from_transform(matrix):
    return (
        float(matrix[0, 3]),
        float(matrix[1, 3]),
        math.atan2(float(matrix[1, 0]), float(matrix[0, 0])),
    )


def _body_to_map(pose, start):
    cosine, sine = math.cos(start[2]), math.sin(start[2])
    return (
        start[0] + cosine * float(pose[0]) - sine * float(pose[1]),
        start[1] + sine * float(pose[0]) + cosine * float(pose[1]),
        wrap_angle(start[2] + float(pose[2])),
    )


def _map_to_body(pose, start):
    dx, dy = float(pose.x) - start[0], float(pose.y) - start[1]
    cosine, sine = math.cos(start[2]), math.sin(start[2])
    return [
        cosine * dx + sine * dy,
        -sine * dx + cosine * dy,
        wrap_angle(float(pose.yaw) - start[2]),
    ]


def _bank_contract(data, prefix):
    trajectories = np.asarray(data["dagger_%s_trajectories" % prefix])
    feasible = np.asarray(data["dagger_%s_feasible" % prefix])
    clearance = np.asarray(data["dagger_%s_static_clearance" % prefix])
    guidance = np.asarray(data["dagger_%s_guidance_cost" % prefix])
    if (
        trajectories.ndim != 3
        or trajectories.shape[0] != 15
        or trajectories.shape[2] != 6
        or feasible.shape != (15,)
        or clearance.shape != (15,)
        or guidance.shape != (15,)
        or not np.all(np.isfinite(trajectories))
        or not np.all(np.isfinite(clearance))
        or not np.all(np.isfinite(guidance))
    ):
        raise RuntimeError("invalid preserved %s DAgger bank" % prefix)
    return int(np.sum(feasible.astype(bool)))


def _gear_runs(path, start):
    runs = []
    for pose in path:
        gear = int(pose.gear)
        if gear not in (-1, 1):
            continue
        if not runs or runs[-1]["gear"] != gear:
            runs.append({"gear": gear, "endpoint": _map_to_body(pose, start)})
        else:
            runs[-1]["endpoint"] = _map_to_body(pose, start)
    return runs


def _candidate_goals(local_path, start, fallback_goal, grid):
    goals = []
    # Prefer the farthest route-corridor pose that can statically contain the
    # vehicle.  Shorter fallbacks make labels available near a blocked route
    # terminus without changing the obstacle geometry seen by the expert.
    for index in range(len(local_path) - 1, -1, -max(1, len(local_path) // 8)):
        target = _body_to_map(local_path[index], start)
        if math.hypot(target[0] - start[0], target[1] - start[1]) < 0.30:
            continue
        valid, _reason, _clearance = HybridAStar.validate_goal_pose(grid, target)
        if valid:
            goals.append((target, "ROUTE_CORRIDOR", int(index)))
    fallback = tuple(float(value) for value in fallback_goal)
    valid, _reason, _clearance = HybridAStar.validate_goal_pose(grid, fallback)
    if valid:
        goals.append((fallback, "TASK_GOAL_FALLBACK", -1))
    return goals


def _expert_plan(job):
    entry, sample_root, task = job
    path = (Path(sample_root) / entry["path"]).resolve()
    with np.load(path, allow_pickle=False) as data:
        manifest = json.loads(str(data["manifest_json"]))
        metadata = manifest.get("metadata", {})
        matrix = np.asarray(
            manifest.get("transforms", {})
            .get("chassis_to_map", {})
            .get("matrix", ()), dtype=np.float64,
        )
        local_path = np.asarray(data["local_path"], dtype=np.float64)
        current = int(data["current_gear"])
        forward_count = _bank_contract(data, "forward")
        reverse_count = _bank_contract(data, "reverse")
    if (
        metadata.get("dagger_schema") != "DEPCarV43ClosedLoopObservationV1"
        or metadata.get("dagger_reobserved_state") is not True
        or metadata.get("dagger_ground_truth_used_for_offline_map_label_only") is not True
        or matrix.shape != (4, 4)
        or not np.all(np.isfinite(matrix))
        or local_path.ndim != 2
        or local_path.shape[1] != 3
        or not len(local_path)
        or not np.all(np.isfinite(local_path))
        or current not in (-1, 0, 1)
    ):
        raise RuntimeError("invalid V4.3 re-observed sample: " + str(path))
    start = _pose_from_transform(matrix)
    grid = _worker_grid(entry["map_uuid"])
    planner_config = HybridAStarConfig(
        step_length=0.20,
        goal_tolerance=0.24,
        goal_heading_tolerance=0.55,
        # Preserve the production contract's human-like forward preference.
        # Reverse remains fully available when it is geometrically necessary;
        # it merely cannot win a comparable route through lower cost.
        reverse_penalty=2.0,
        gear_switch_penalty=0.75,
        maximum_expansions=25000,
    )
    result = None
    attempts = []
    for goal, source, route_index in _candidate_goals(
        local_path, start, task["goal"], grid
    ):
        planner = HybridAStar(planner_config)
        expert_path = planner.plan(grid, start, goal)
        attempts.append({
            "source": source,
            "route_index": route_index,
            "status": planner.last_status,
            "expansions": planner.last_expansions,
        })
        if expert_path:
            result = (expert_path, goal, source, route_index, planner.last_expansions)
            break
    if result is None:
        gears = [0] * ACTIONS
        mask = [False] * ACTIONS
        endpoints = [[0.0, 0.0, 0.0] for _ in range(ACTIONS)]
        status = "STOP_NO_EXACT_SIGNED_PLAN"
        target_source = "NONE"
        expansions = sum(item["expansions"] for item in attempts)
        truncated = False
    else:
        expert_path, goal, target_source, route_index, expansions = result
        runs = _gear_runs(expert_path, start)
        truncated = len(runs) > ACTIONS
        runs = runs[:ACTIONS]
        gears = [row["gear"] for row in runs]
        endpoints = [row["endpoint"] for row in runs]
        mask = [True] * len(runs)
        gears += [0] * (ACTIONS - len(gears))
        endpoints += [[0.0, 0.0, 0.0] for _ in range(ACTIONS - len(endpoints))]
        mask += [False] * (ACTIONS - len(mask))
        status = "SUCCESS"
    return {
        "sample_id": entry["sample_id"],
        "task_id": entry["task_id"],
        "split": entry["split"],
        "map_uuid": entry["map_uuid"],
        "source_content_sha256": entry["content_sha256"],
        "current_gear": current,
        "history": [float(value) for value in metadata.get("dagger_gear_history", ())],
        "executed_gear": int(metadata.get("dagger_executed_gear", 0)),
        "model_raw_sequence": [
            int(value) for value in metadata.get("dagger_model_raw_sequence", ())
        ],
        "diagnostic_one_step_teacher_gear": int(
            metadata.get("dagger_teacher_gear", 0)
        ),
        "sequence_gears": gears,
        "sequence_mask": mask,
        "action_plan_endpoints_body": endpoints,
        "teacher_plan_status": status,
        "teacher_target_source": target_source,
        "teacher_plan_expansions": int(expansions),
        "teacher_plan_attempts": attempts,
        "teacher_plan_truncated": bool(truncated),
        "forward_bank_feasible": forward_count,
        "reverse_bank_feasible": reverse_count,
        "sequence_authority": SEQUENCE_AUTHORITY,
    }


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", default="data/p3_v7_v43/run/samples")
    parser.add_argument("--maps", default="data/p3_pilot/maps")
    parser.add_argument("--manifest", default="data/p3_v7_v43/task_manifest.json")
    parser.add_argument(
        "--collection-state", default="data/p3_v7_v43/run/collection_state.json"
    )
    parser.add_argument("--output", default="data/p3_v7_v43/index")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        raise SystemExit("workers must be positive")
    sample_root, maps_root, output = (
        resolve(args.samples), resolve(args.maps), resolve(args.output)
    )
    manifest_path, state_path = resolve(args.manifest), resolve(args.collection_state)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    state = json.loads(state_path.read_text(encoding="utf-8"))
    tasks = {row["task_id"]: row for row in manifest.get("tasks", ())}
    complete = {
        task_id for task_id in tasks
        if state.get("tasks", {}).get(task_id, {}).get("status") == "COMPLETE"
    }
    incomplete = sorted(set(tasks) - complete)
    if incomplete and not args.allow_incomplete:
        raise RuntimeError(
            "V4.3 DAgger tasks incomplete: " + ",".join(incomplete[:8])
        )
    training_index_path = output / "training_index.json"
    sequence_path = output / "closed_loop_sequence_index.json"
    reusable = {}
    if sequence_path.is_file():
        previous = json.loads(sequence_path.read_text(encoding="utf-8"))
        if (
            previous.get("schema") == INDEX_SCHEMA
            and previous.get("sequence_authority") == SEQUENCE_AUTHORITY
            and previous.get("test_split_opened") is False
        ):
            reusable = {
                row["sample_id"]: row for row in previous.get("rows", ())
                if row.get("sequence_authority") == SEQUENCE_AUTHORITY
            }
    training = build_training_index(
        sample_root,
        maps_root,
        training_index_path,
        splits=("train", "validation"),
        workers=args.workers,
    )
    entries = [entry for entry in training["entries"] if entry["task_id"] in complete]
    jobs, rows = [], []
    for entry in entries:
        previous = reusable.get(entry["sample_id"])
        if (
            previous is not None
            and previous.get("source_content_sha256") == entry["content_sha256"]
            and previous.get("task_id") == entry["task_id"]
            and previous.get("map_uuid") == entry["map_uuid"]
            and previous.get("split") == entry["split"]
        ):
            rows.append(previous)
        else:
            jobs.append((entry, str(sample_root), tasks[entry["task_id"]]))
    reused_rows = len(rows)
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_initialize_worker,
        initargs=(str(maps_root),),
    ) as executor:
        futures = {executor.submit(_expert_plan, job): job[0]["sample_id"] for job in jobs}
        for finished, future in enumerate(as_completed(futures), 1):
            rows.append(future.result())
            if finished == 1 or finished % 100 == 0 or finished == len(futures):
                print(
                    "V4.3 exact teacher recomputed %d/%d (reused=%d)"
                    % (finished, len(futures), reused_rows),
                    flush=True,
                    file=sys.stderr,
                )
    rows.sort(key=lambda row: row["sample_id"])
    sequence = {
        "schema": INDEX_SCHEMA,
        "sequence_actions": ACTIONS,
        "sequence_authority": SEQUENCE_AUTHORITY,
        "source_index": str(training_index_path),
        "source_index_sha256": sha256_file(training_index_path),
        "source_content_aggregate_sha256": training["content_aggregate_sha256"],
        "sample_root": str(sample_root),
        "maps_root": str(maps_root),
        "test_split_opened": False,
        "samples": len(rows),
        "counts_by_split": dict(
            sorted(Counter(row["split"] for row in rows).items())
        ),
        "rows": rows,
    }
    sequence["content_sha256"] = canonical_sha256(sequence)
    atomic_json(sequence_path, sequence)

    patterns = Counter()
    reverse = reverse_forward = multi = stop = truncated = 0
    one_step_disagreement = 0
    for row in rows:
        active = [
            gear for gear, keep in zip(row["sequence_gears"], row["sequence_mask"])
            if keep
        ]
        pattern = "-".join("F" if gear > 0 else "R" for gear in active) or "STOP"
        patterns[pattern] += 1
        multi += int(len(active) >= 2)
        reverse += int(-1 in active)
        reverse_forward += int(
            any(first < 0 and second > 0 for first, second in zip(active, active[1:]))
        )
        stop += int(not active)
        truncated += int(row["teacher_plan_truncated"])
        one_step_disagreement += int(
            bool(active)
            and active[0] != row["diagnostic_one_step_teacher_gear"]
        )
    maps = _load_map_catalog(maps_root, {row["map_uuid"] for row in rows})
    splits = set(sequence["counts_by_split"])
    formal = not incomplete and not args.allow_incomplete
    recovered_evidence = []
    for task_id in complete:
        task_state = state.get("tasks", {}).get(task_id, {})
        if task_state.get("recovered_extraction_without_gazebo"):
            evidence_path = Path(task_state.get("recovery_evidence", ""))
            if not evidence_path.is_absolute():
                evidence_path = ROOT / evidence_path
            recovered_evidence.append(
                evidence_path.is_file()
                and json.loads(evidence_path.read_text(encoding="utf-8")).get("status")
                == "RECOVERED_DAGGER_HARD_VETO_OBSERVATION"
            )
    gates = {
        "all_manifest_tasks_complete": not incomplete,
        "development_splits_present": splits == {"train", "validation"},
        "minimum_samples": len(rows) >= 800,
        "minimum_reverse_sequences": reverse >= 100,
        "minimum_reverse_then_forward_sequences": reverse_forward >= 30,
        "maximum_stop_rate": stop / max(1, len(rows)) <= 0.08,
        "maximum_truncated_rate": truncated / max(1, len(rows)) <= 0.02,
        "exact_plan_labels": all(
            row["sequence_authority"] == SEQUENCE_AUTHORITY for row in rows
        ),
        "two_candidate_banks_preserved": all(
            row["forward_bank_feasible"] >= 0 and row["reverse_bank_feasible"] >= 0
            for row in rows
        ),
        "recovered_tasks_have_explicit_evidence": all(recovered_evidence),
    }
    if args.allow_incomplete:
        # A diagnostic partial index reports every metric but cannot mint a
        # formal data authority regardless of the available sample counts.
        gates["all_manifest_tasks_complete"] = False
    implementation = {
        "expert": "dep_car/src/dep_car/global_planner/hybrid_astar.py",
        "occupancy": "dep_car/src/dep_car/core/occupancy.py",
        "indexer": "tools/build_p3_v7_v43_closed_loop_index.py",
        "extractor": "ros/dep_car_dataset/scripts/extract_multimodal_bag.py",
        "episode_runner": "ros/dep_car_dataset/scripts/run_pilot_episode.py",
        "collection_orchestrator": "ros/dep_car_dataset/scripts/run_pilot_collection.py",
        "collection_recovery": "tools/recover_p5_closed_loop_v43_collection.py",
    }
    for name, relative in tuple(implementation.items()):
        implementation[name + "_sha256"] = sha256_file(ROOT / relative)
    authority = {
        "schema": AUTHORITY_SCHEMA,
        "status": "PASS" if formal and all(gates.values()) else "FAIL",
        "errors": sorted(key for key, value in gates.items() if not value),
        "gates": gates,
        "implementation": implementation,
        "training_index": str(training_index_path),
        "training_index_sha256": sha256_file(training_index_path),
        "content_aggregate_sha256": training["content_aggregate_sha256"],
        "sequence_index": str(sequence_path),
        "sequence_index_sha256": sha256_file(sequence_path),
        "sequence_content_sha256": sequence["content_sha256"],
        "sample_root": str(sample_root),
        "maps_root": str(maps_root),
        "map_contract_aggregate_sha256": indexed_map_contract_aggregate(maps)[
            "aggregate_sha256"
        ],
        "task_manifest": str(manifest_path),
        "task_manifest_sha256": sha256_file(manifest_path),
        "collection_state": str(state_path),
        "collection_state_sha256": sha256_file(state_path),
        "samples": len(rows),
        "teacher_rows_reused": reused_rows,
        "teacher_rows_recomputed": len(jobs),
        "episodes": len(complete),
        "counts_by_split": sequence["counts_by_split"],
        "multi_action_samples": multi,
        "reverse_sequence_samples": reverse,
        "reverse_then_forward_samples": reverse_forward,
        "stop_samples": stop,
        "truncated_samples": truncated,
        "diagnostic_one_step_teacher_disagreements": one_step_disagreement,
        "sequence_patterns": dict(sorted(patterns.items())),
        "continuous_sequence_authority": SEQUENCE_AUTHORITY,
        "runtime_hybrid_astar_dependency": False,
        "runtime_ground_truth_input": False,
        "offline_teacher_ground_truth_map_label": True,
        "test_split_opened": False,
        "active_control_authorized": False,
        "production_qualified": False,
    }
    authority["authority_sha256"] = canonical_sha256(authority)
    authority_path = output / "closed_loop_data_authority.json"
    atomic_json(authority_path, authority)
    print(
        json.dumps({**authority, "authority": str(authority_path)}, indent=2, sort_keys=True)
    )
    return 0 if authority["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
