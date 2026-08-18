#!/usr/bin/env python3
"""Aggregate P6 shadow episodes and issue simulation-only active authority."""

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dep_car/src"))
from dep_car.runtime.p6_contract import (
    P6_SHADOW_ACCEPTANCE_SCHEMA,
    build_p6_runtime_contract,
    canonical_sha256,
    sha256_file,
)


def resolve(value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def load_manifest(path):
    document = json.loads(path.read_text(encoding="utf-8"))
    content = dict(document)
    expected = content.pop("scenario_manifest_sha256", "")
    if canonical_sha256(content) != expected:
        raise ValueError("P6 scenario manifest identity mismatch")
    return document


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "dep_car/config/p6_static.yaml")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/p6_static/scenario_manifest.json")
    parser.add_argument("--reports", type=Path, default=ROOT / "data/p6_static/run/shadow")
    parser.add_argument("--modality", choices=("depth_only", "lidar_only", "fusion"), default="fusion")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    gates = config["shadow_gates"]
    manifest = load_manifest(args.manifest)
    checkpoint = manifest["checkpoints"][args.modality]
    runtime = build_p6_runtime_contract()
    cohort = gates["cohort"]
    expected = {
        row["scenario_id"]: row
        for row in manifest["scenarios"]
        if row["cohort"] == cohort
        and row.get("start_robustness", {}).get("status") == "PASS"
    }
    report_root = args.reports / args.modality
    reports = []
    errors = []
    ignored_stale_reports = []
    for path in sorted(report_root.glob("*.json")) if report_root.is_dir() else ():
        report = json.loads(path.read_text(encoding="utf-8"))
        scenario_id = report.get("scenario_id")
        if scenario_id not in expected:
            continue
        stale_reasons = []
        if report.get("policy_mode") != "shadow":
            stale_reasons.append("not_shadow")
        if report.get("observed_modality") != args.modality:
            stale_reasons.append("modality_mismatch")
        if report.get("checkpoint_sha256") != checkpoint["checkpoint_sha256"]:
            stale_reasons.append("checkpoint_mismatch")
        if report.get("scenario_manifest_sha256") != manifest["scenario_manifest_sha256"]:
            stale_reasons.append("manifest_mismatch")
        if report.get("runtime_implementation_sha256") != runtime["aggregate_sha256"]:
            stale_reasons.append("runtime_implementation_mismatch")
        if stale_reasons:
            ignored_stale_reports.append({
                "path": str(path.resolve()),
                "scenario_id": scenario_id,
                "reasons": stale_reasons,
            })
            continue
        report["_path"] = path
        reports.append(report)
    by_id = {report["scenario_id"]: report for report in reports}
    if len(reports) < int(gates["minimum_episodes"]):
        errors.append("minimum_episodes")
    required_modes = set(gates["required_maneuvers"])
    observed_modes = Counter(report.get("maneuver_mode") for report in reports)
    for mode in sorted(required_modes.difference(observed_modes)):
        errors.append("missing_maneuver_" + mode)
    if gates.get("require_all_episode_success", True):
        for report in reports:
            if report.get("status") != "SUCCESS":
                errors.append("episode_not_success:" + report["scenario_id"])
    collision_samples = sum(int(report.get("collision_samples", 0)) for report in reports)
    illegal_shifts = sum(int(report.get("illegal_shift_count", 0)) for report in reports)
    if collision_samples > int(gates["maximum_collision_samples"]):
        errors.append("collision_samples")
    if illegal_shifts > int(gates["maximum_illegal_shift_count"]):
        errors.append("illegal_shift_count")
    safe_messages = sum(int(report.get("safe_policy_messages", 0)) for report in reports)
    zero_messages = sum(int(report.get("zero_feasible_messages", 0)) for report in reports)
    zero_rate = zero_messages / max(1, safe_messages)
    if zero_rate > float(gates["maximum_zero_feasible_rate"]):
        errors.append("zero_feasible_rate")
    latency_p95 = max(
        (
            float(report["inference_latency_ms"]["p95"])
            for report in reports
            if report.get("inference_latency_ms", {}).get("p95") is not None
        ),
        default=float("inf"),
    )
    maximum_skew = max(
        (
            float(report["sensor_skew_s"]["maximum"])
            for report in reports
            if report.get("sensor_skew_s", {}).get("maximum") is not None
        ),
        default=float("inf"),
    )
    if latency_p95 > float(gates["maximum_inference_latency_p95_ms"]):
        errors.append("inference_latency_p95")
    if maximum_skew > float(gates["maximum_sensor_skew_s"]) + 1.0e-9:
        errors.append("sensor_skew")
    if gates.get("require_hard_safety", True) and any(
        int(report.get("hard_safety_samples", 0)) == 0 for report in reports
    ):
        errors.append("hard_safety_missing")
    if gates.get("forbid_learned_control_in_shadow", True):
        for report in reports:
            if int(report.get("command_source_counts", {}).get("dep_car_net_v1_active", 0)):
                errors.append("shadow_control_violation:" + report["scenario_id"])
    for report in reports:
        if report.get("policy_mode") != "shadow":
            errors.append("not_shadow:" + report["scenario_id"])
        if report.get("observed_modality") != args.modality:
            errors.append("modality_mismatch:" + report["scenario_id"])
        if report.get("checkpoint_sha256") != checkpoint["checkpoint_sha256"]:
            errors.append("checkpoint_mismatch:" + report["scenario_id"])
        if report.get("scenario_manifest_sha256") != manifest["scenario_manifest_sha256"]:
            errors.append("manifest_mismatch:" + report["scenario_id"])
    report_hashes = {
        str(report["_path"].resolve()): sha256_file(report["_path"])
        for report in reports
    }
    payload = {
        "schema": P6_SHADOW_ACCEPTANCE_SCHEMA,
        "status": "PASS" if not errors else "FAIL",
        "modality": args.modality,
        "checkpoint": checkpoint["checkpoint"],
        "checkpoint_sha256": checkpoint["checkpoint_sha256"],
        "checkpoint_contract": checkpoint["contract"],
        "checkpoint_contract_sha256": checkpoint["contract_sha256"],
        "scenario_manifest": str(args.manifest.resolve()),
        "scenario_manifest_sha256": manifest["scenario_manifest_sha256"],
        "cohort": cohort,
        "episodes": len(reports),
        "maneuver_counts": dict(observed_modes),
        "collision_samples": collision_samples,
        "illegal_shift_count": illegal_shifts,
        "safe_policy_messages": safe_messages,
        "zero_feasible_messages": zero_messages,
        "zero_feasible_rate": zero_rate,
        "maximum_episode_inference_latency_p95_ms": latency_p95 if math.isfinite(latency_p95) else None,
        "maximum_sensor_skew_s": maximum_skew if math.isfinite(maximum_skew) else None,
        "report_hashes": report_hashes,
        "report_aggregate_sha256": canonical_sha256(report_hashes),
        "runtime_implementation": runtime,
        "errors": sorted(set(errors)),
        "ignored_stale_reports": ignored_stale_reports,
        "authority_scope": "Gazebo P6 active simulation only; never P8 production deployment",
    }
    output = args.output or (ROOT / "reports" / ("p6_shadow_acceptance_%s.json" % args.modality))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps(payload, indent=2, sort_keys=True))
    raise SystemExit(0 if payload["status"] == "PASS" else 2)


if __name__ == "__main__":
    main()
