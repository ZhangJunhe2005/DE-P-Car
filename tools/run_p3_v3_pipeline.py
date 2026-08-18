#!/usr/bin/env python3
"""Host-side staged entry point for P3 V3 reinforcement and sealing."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
YOPO_PYTHON = Path("/home/zjh/miniconda3/envs/yopo/bin/python")
SYSTEM_PYTHON = Path("/usr/bin/python3")


def run(command):
    command = [str(item) for item in command]
    print("+ " + " ".join(command), flush=True)
    result = subprocess.run(command)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def load_json_if_present(path):
    if not Path(path).is_file():
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        required=True,
        choices=(
            "validate", "prepare", "base-reextract", "collect",
            "recover-extraction", "qualify-invalid-goals", "bundle", "audit",
            "proposal", "status",
        ),
    )
    parser.add_argument(
        "--config", type=Path,
        default=ROOT / "dep_car/config/p3_v3_incremental.yaml",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--maximum-tasks", type=int, default=0)
    parser.add_argument("--maximum-samples", type=int, default=0)
    parser.add_argument("--task-id")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--startup-stagger", type=float, default=1.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.workers < 1:
        raise ValueError("workers must be positive")
    if args.maximum_tasks < 0 or args.maximum_samples < 0:
        raise ValueError("maximum task/sample limits cannot be negative")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    wave_id = str(config["wave"]["id"])
    configured_wave_root = Path(
        config["wave"].get("output_root", "data/p3_v3/waves")
    )
    wave_base = (
        configured_wave_root.resolve()
        if configured_wave_root.is_absolute()
        else (ROOT / configured_wave_root).resolve()
    )
    if ROOT.resolve() not in wave_base.parents:
        raise ValueError("wave output root must remain inside the project")
    wave_root = wave_base / wave_id
    wave_manifest = wave_root / "task_manifest.json"
    wave_run = wave_root / "run"
    base_root = ROOT / "data/p3_v3/base_reextracted"
    configured_bundle_root = Path(
        config.get("bundle", {}).get("output", "data/p3_v3/bundle_v1")
    )
    bundle_root = (
        configured_bundle_root.resolve()
        if configured_bundle_root.is_absolute()
        else (ROOT / configured_bundle_root).resolve()
    )
    if ROOT.resolve() not in bundle_root.parents:
        raise ValueError("bundle output must remain inside the project")
    configured_audit_report = Path(
        config.get("audit", {}).get(
            "report", "reports/p3_development_reaudit_v3.json"
        )
    )
    audit_report = (
        configured_audit_report.resolve()
        if configured_audit_report.is_absolute()
        else (ROOT / configured_audit_report).resolve()
    )
    if ROOT.resolve() not in audit_report.parents:
        raise ValueError("audit report must remain inside the project")
    configured_proposal = Path(
        config.get("audit", {}).get(
            "proposal", "reports/p3_v3_training_authority_proposal.json"
        )
    )
    proposal_output = (
        configured_proposal.resolve()
        if configured_proposal.is_absolute()
        else (ROOT / configured_proposal).resolve()
    )
    if ROOT.resolve() not in proposal_output.parents:
        raise ValueError("proposal report must remain inside the project")

    if args.stage == "validate":
        run([
            YOPO_PYTHON, ROOT / "tools/generate_p3_v3_wave.py",
            "--config", args.config,
            "--maps", ROOT / "data/p3_pilot/maps",
            "--output", wave_manifest,
            "--workers", args.workers,
            "--validate-only",
        ])
        return 0

    if args.stage == "prepare":
        run([
            YOPO_PYTHON, ROOT / "tools/generate_p3_v3_wave.py",
            "--config", args.config,
            "--maps", ROOT / "data/p3_pilot/maps",
            "--output", wave_manifest,
            "--workers", args.workers,
        ])
        return 0

    if args.stage == "base-reextract":
        command = [
            SYSTEM_PYTHON, ROOT / "tools/reextract_p3_v3_base.py",
            "--output", base_root,
            "--workers", args.workers,
        ]
        if args.maximum_tasks:
            command.extend(("--maximum-tasks", args.maximum_tasks))
        if args.retry_failed:
            command.append("--retry-failed")
        if args.dry_run:
            command.append("--dry-run")
        run(command)
        return 0

    if args.stage == "collect":
        if not wave_manifest.is_file():
            raise FileNotFoundError("run --stage prepare before collection")
        command = [
            SYSTEM_PYTHON,
            ROOT / "ros/dep_car_dataset/scripts/run_pilot_collection.py",
            "--config", args.config,
            "--manifest", wave_manifest,
            "--work-root", wave_run,
            "--workers", args.workers,
            "--startup-stagger", args.startup_stagger,
        ]
        if args.maximum_tasks:
            command.extend(("--maximum-tasks", args.maximum_tasks))
        if args.task_id:
            command.extend(("--task-id", args.task_id))
        if args.retry_failed:
            command.append("--retry-failed")
        if args.fail_fast:
            command.append("--fail-fast")
        if args.dry_run:
            command.append("--dry-run")
        run(command)
        return 0

    if args.stage == "recover-extraction":
        if not wave_manifest.is_file():
            raise FileNotFoundError("run --stage prepare before extraction recovery")
        command = [
            SYSTEM_PYTHON,
            ROOT / "tools/recover_p3_extraction.py",
            "--config", args.config,
            "--manifest", wave_manifest,
            "--work-root", wave_run,
            "--workers", args.workers,
        ]
        if args.maximum_tasks:
            command.extend(("--maximum-tasks", args.maximum_tasks))
        if args.dry_run:
            command.append("--dry-run")
        run(command)
        return 0

    if args.stage == "qualify-invalid-goals":
        command = [
            YOPO_PYTHON,
            ROOT / "tools/qualify_p3_invalid_goal_exclusions.py",
            "--manifest", wave_manifest,
            "--state", wave_run / "collection_state.json",
            "--maps", ROOT / "data/p3_pilot/maps",
        ]
        if args.dry_run:
            command.append("--dry-run")
        run(command)
        return 0

    if args.stage == "bundle":
        command = [
            YOPO_PYTHON, ROOT / "tools/build_p3_v3_bundle.py",
            "--config", args.config,
            "--maps", ROOT / "data/p3_pilot/maps",
            "--output", bundle_root,
            "--workers", args.workers,
        ]
        if args.dry_run:
            command.append("--dry-run")
        run(command)
        return 0

    if args.stage == "audit":
        report_path = (
            audit_report.with_name(audit_report.stem + "_smoke" + audit_report.suffix)
            if args.maximum_samples
            else audit_report
        )
        command = [
            YOPO_PYTHON, ROOT / "tools/audit_p3_v3_bundle.py",
            "--config", args.config,
            "--authority", bundle_root / "bundle_authority.json",
            "--report", report_path,
            "--workers", args.workers,
        ]
        if args.maximum_samples:
            command.extend(("--maximum-samples", args.maximum_samples))
        run(command)
        return 0

    if args.stage == "proposal":
        run([
            YOPO_PYTHON, ROOT / "tools/propose_p3_v3_training_authority.py",
            "--authority", bundle_root / "bundle_authority.json",
            "--reaudit", audit_report,
            "--output", proposal_output,
        ])
        return 0

    status = {
        "schema": "DEPCarP3V3PipelineStatusV1",
        "wave_manifest": load_json_if_present(wave_manifest),
        "wave_collection": load_json_if_present(wave_run / "collection_state.json"),
        "base_reextract": load_json_if_present(base_root / "reextract_state.json"),
        "bundle": load_json_if_present(bundle_root / "bundle_authority.json"),
        "development_reaudit": load_json_if_present(
            audit_report
        ),
    }
    summary = {
        "schema": status["schema"],
        "wave_manifest_present": status["wave_manifest"] is not None,
        "wave_tasks": len((status["wave_manifest"] or {}).get("tasks", ())),
        "wave_collection_state_present": status["wave_collection"] is not None,
        "base_reextract_status": (status["base_reextract"] or {}).get("status", "MISSING"),
        "bundle_status": (status["bundle"] or {}).get("status", "MISSING"),
        "reaudit_status": (status["development_reaudit"] or {}).get("status", "MISSING"),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
