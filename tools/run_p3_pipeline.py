#!/usr/bin/env python3
"""Host-side entry point for preparing, collecting and auditing P3 Pilot data."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def canonical_sha256(payload):
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def reusable_task_manifest(path, config, maps_root):
    if not path.is_file():
        return False
    manifest = json.loads(path.read_text(encoding="utf-8"))
    content = dict(manifest)
    expected = content.pop("task_manifest_sha256", "")
    if expected != canonical_sha256(content):
        return False
    contract = manifest.get("generator_contract", {})
    if contract.get("partial") or contract.get("config_sha256") != canonical_sha256(config):
        return False
    tasks = manifest.get("tasks", [])
    if len(tasks) != int(config["tasks"]["total"]):
        return False
    map_uuids = {
        json.loads(path.read_text(encoding="utf-8"))["map_uuid"]
        for path in maps_root.glob("*/manifest.json")
    }
    return all(task.get("map_uuid") in map_uuids for task in tasks)


def run(command):
    print("+ " + " ".join(str(item) for item in command), flush=True)
    result = subprocess.run([str(item) for item in command])
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("prepare", "collect", "reextract", "audit", "all"), default="all")
    parser.add_argument("--config", type=Path, default=ROOT / "dep_car/config/p3_pilot.yaml")
    parser.add_argument("--root", type=Path, default=ROOT / "data/p3_pilot")
    parser.add_argument("--maximum-tasks", type=int, default=0)
    parser.add_argument("--task-id")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--rerun-complete", action="store_true")
    parser.add_argument("--rerun-all-complete", action="store_true")
    parser.add_argument("--retry-zero-feasible-rate-above", type=float)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="validate collection commands without launching Gazebo")
    parser.add_argument("--workers", type=int, default=8, help="parallel collection/audit workers")
    parser.add_argument("--startup-stagger", type=float, default=1.0)
    parser.add_argument("--allow-partial-tasks", action="store_true")
    parser.add_argument("--force-prepare", action="store_true", help="regenerate a valid existing task manifest")
    parser.add_argument("--skip-bag-hash", action="store_true", help="diagnostic audit only")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    maps = args.root / "maps"
    manifest = args.root / "task_manifest.json"
    run_root = args.root / "run"
    collection_state = run_root / "collection_state.json"

    if args.stage in ("prepare", "all"):
        map_config = config["maps"]
        run([
            "/usr/bin/python3", ROOT / "ros/dep_car_dataset/scripts/generate_static_maps.py",
            "--output", maps, "--type", map_config["type"], "--count", map_config["generated_count"],
            "--seed", map_config["generator_seed"], "--width", map_config["width"], "--height", map_config["height"],
            "--resolution", map_config["resolution_m"], "--corridor-radius", map_config["corridor_radius_cells"],
            "--iterations", map_config["iterations"], "--resume",
        ])
        if collection_state.is_file() and manifest.is_file():
            state = json.loads(collection_state.read_text(encoding="utf-8"))
            existing = json.loads(manifest.read_text(encoding="utf-8"))
            if state.get("task_manifest_sha256") != existing.get("task_manifest_sha256"):
                raise RuntimeError("collection state and existing task manifest already disagree")
            print("collection state exists; preserving the bound task manifest", flush=True)
        elif not args.force_prepare and reusable_task_manifest(manifest, config, maps):
            existing = json.loads(manifest.read_text(encoding="utf-8"))
            print(
                "valid task manifest already exists; preserving " + existing["task_manifest_sha256"],
                flush=True,
            )
        else:
            command = [
                "/usr/bin/python3", ROOT / "ros/dep_car_dataset/scripts/generate_pilot_tasks.py",
                "--config", args.config, "--maps", maps, "--output", manifest,
            ]
            if args.allow_partial_tasks:
                command.append("--allow-partial")
            run(command)

    if args.stage in ("collect", "all"):
        command = [
            "/usr/bin/python3", ROOT / "ros/dep_car_dataset/scripts/run_pilot_collection.py",
            "--config", args.config, "--manifest", manifest, "--work-root", run_root,
            "--workers", args.workers, "--startup-stagger", args.startup_stagger,
        ]
        if args.maximum_tasks:
            command.extend(("--maximum-tasks", args.maximum_tasks))
        if args.task_id:
            command.extend(("--task-id", args.task_id))
        if args.retry_failed:
            command.append("--retry-failed")
        if args.rerun_complete:
            command.append("--rerun-complete")
        if args.rerun_all_complete:
            command.append("--rerun-all-complete")
        if args.retry_zero_feasible_rate_above is not None:
            command.extend((
                "--retry-zero-feasible-rate-above",
                args.retry_zero_feasible_rate_above,
            ))
        if args.fail_fast:
            command.append("--fail-fast")
        if args.dry_run:
            command.append("--dry-run")
        run(command)

    if args.stage == "reextract":
        run([
            "/usr/bin/python3", ROOT / "ros/dep_car_dataset/scripts/reextract_pilot_samples.py",
            "--config", args.config, "--manifest", manifest, "--work-root", run_root,
            "--workers", args.workers,
        ])

    if args.stage in ("audit", "all") and not args.dry_run:
        command = [
            "/usr/bin/python3", ROOT / "tools/audit_p3_pilot_dataset.py", run_root / "samples",
            "--config", args.config, "--manifest", manifest,
            "--collection-state", collection_state, "--report", run_root / "p3_pilot_audit.json",
            "--workers", args.workers,
        ]
        if args.skip_bag_hash:
            command.append("--skip-bag-hash")
        run(command)


if __name__ == "__main__":
    main()
