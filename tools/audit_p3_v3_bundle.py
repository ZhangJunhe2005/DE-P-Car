#!/usr/bin/env python3
"""Qualify a sealed P3 V3/V4 development bundle for P5.

This wrapper authenticates the dynamic bundle authority, then delegates all
candidate regeneration and continuous footprint geometry to the frozen P3 V3
audit implementation.  A derived curriculum may tighten (but never weaken)
the frozen geometry gates.  It emits the existing P5-compatible report schema.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dep_car" / "src"))
sys.path.insert(0, str(ROOT / "tools"))

import audit_p3_footprint_upgrade as geometry
from dep_car.training.pilot import canonical_sha256


BUNDLE_SCHEMA = "DEPCarP3V3BundleAuthorityV1"
CONFIG_SCHEMA = "DEPCarP3V3IncrementalConfigV1"
REPORT_SCHEMA = "DEPCarP3DevelopmentReauditV3"
DEFAULT_FORMAL_REPORT = ROOT / "reports/p3_development_reaudit_v3.json"
ALLOWED_CONTEXTS = ("MISSION", "RECOVERY")
GEAR_NAMES = {1: "FORWARD", -1: "REVERSE"}
CURATION_SCHEMA = "P3V3InitialPoseFeasibilityCurationV1"
CURATION_POLICY = "exclude_initial_pose_infeasible"
CURATION_EVALUATOR = "production_signed_SDF_training_footprint_at_t0"
GEOMETRY_GATE_KEYS = (
    "maximum_zero_feasible_rate",
    "maximum_per_mode_zero_feasible_rate",
    "minimum_median_feasible_candidates",
)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def load_config(path):
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("schema") != CONFIG_SCHEMA:
        raise ValueError("P3 V3 incremental config schema mismatch")
    gates = config.get("gates", {})
    frozen = {
        "maximum_zero_feasible_rate": geometry.OVERALL_ZERO_FEASIBLE_LIMIT,
        "maximum_per_mode_zero_feasible_rate": geometry.PER_MODE_ZERO_FEASIBLE_LIMIT,
        "minimum_median_feasible_candidates": geometry.MINIMUM_MEDIAN_FEASIBLE,
        "minimum_validation_frames_per_maneuver": 50,
        "minimum_validation_frames_per_requested_gear": 100,
        "minimum_validation_frames_per_candidate_context": 20,
        "required_candidate_contexts": ["MISSION", "RECOVERY"],
        "allowed_candidate_contexts": ["MISSION", "RECOVERY"],
        "required_maneuvers": list(geometry.PILOT_MANEUVER_MODES),
        "required_requested_gears": ["FORWARD", "REVERSE"],
    }
    if not isinstance(gates, dict) or set(gates) != set(frozen):
        raise ValueError("P3 qualification gate fields differ from the frozen contract")
    for name, expected in frozen.items():
        if name not in GEOMETRY_GATE_KEYS and gates[name] != expected:
            raise ValueError(
                "P3 non-geometry qualification gates differ from the frozen contract"
            )
    for name in GEOMETRY_GATE_KEYS:
        value = gates[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise ValueError("P3 geometry qualification gates must be finite numbers")
    if not 0.0 < float(gates["maximum_zero_feasible_rate"]) <= float(
        frozen["maximum_zero_feasible_rate"]
    ):
        raise ValueError("P3 overall zero-feasible gate weakens the frozen contract")
    if not 0.0 < float(gates["maximum_per_mode_zero_feasible_rate"]) <= float(
        frozen["maximum_per_mode_zero_feasible_rate"]
    ):
        raise ValueError("P3 per-mode zero-feasible gate weakens the frozen contract")
    if not float(frozen["minimum_median_feasible_candidates"]) <= float(
        gates["minimum_median_feasible_candidates"]
    ) <= float(geometry.EXPECTED_CANDIDATES):
        raise ValueError("P3 median-feasible gate weakens the frozen contract")
    curation = config.get("curation")
    if curation is not None and curation != {
        "schema": CURATION_SCHEMA,
        "policy": CURATION_POLICY,
        "evaluator": CURATION_EVALUATOR,
        "preserve_rejected_source_samples": True,
    }:
        raise ValueError("P3 V3 curation differs from the frozen contract")
    return config


def evaluate_configured_geometry_gates(statistics, configured):
    """Evaluate the sealed bundle against its configured, non-weakened gates."""
    overall = statistics.get("overall", {}).get("new", {})
    by_mode = statistics.get("by_mode", {})
    zero_threshold = float(configured["maximum_zero_feasible_rate"])
    mode_threshold = float(configured["maximum_per_mode_zero_feasible_rate"])
    median_threshold = float(configured["minimum_median_feasible_candidates"])
    overall_zero = overall.get("zero_feasible_rate")
    overall_median = overall.get("feasible_candidates_median")

    zero_pass = (
        isinstance(overall_zero, (int, float))
        and not isinstance(overall_zero, bool)
        and math.isfinite(float(overall_zero))
        and float(overall_zero) < zero_threshold
    )
    median_pass = (
        isinstance(overall_median, (int, float))
        and not isinstance(overall_median, bool)
        and math.isfinite(float(overall_median))
        and float(overall_median) >= median_threshold
    )
    checks = {
        "configured_overall_zero_feasible_rate": {
            "observed": overall_zero,
            "operator": "<",
            "threshold": zero_threshold,
            "status": "PASS" if zero_pass else "FAIL",
        },
        "configured_overall_median_feasible_candidates": {
            "observed": overall_median,
            "operator": ">=",
            "threshold": median_threshold,
            "status": "PASS" if median_pass else "FAIL",
        },
    }
    mode_checks = {}
    for mode in geometry.PILOT_MANEUVER_MODES:
        observed = by_mode.get(mode, {}).get("new", {}).get("zero_feasible_rate")
        passed = (
            isinstance(observed, (int, float))
            and not isinstance(observed, bool)
            and math.isfinite(float(observed))
            and float(observed) < mode_threshold
        )
        mode_checks[mode] = {
            "observed": observed,
            "operator": "<",
            "threshold": mode_threshold,
            "status": "PASS" if passed else "FAIL",
        }
    checks["configured_per_mode_zero_feasible_rate"] = mode_checks
    failures = [
        name
        for name, row in checks.items()
        if name != "configured_per_mode_zero_feasible_rate"
        and row["status"] != "PASS"
    ]
    failures.extend(
        "configured_mode_zero_feasible_rate_" + mode
        for mode, row in mode_checks.items()
        if row["status"] != "PASS"
    )
    return checks, failures


def verify_bundle_authority(path, config_path):
    raw = Path(path).read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    claimed = payload.get("bundle_authority_sha256", "")
    content = dict(payload)
    content.pop("bundle_authority_sha256", None)
    errors = []
    if payload.get("schema") != BUNDLE_SCHEMA or payload.get("status") != "SEALED":
        errors.append("bundle_authority_schema_or_status")
    if claimed != canonical_sha256(content):
        errors.append("bundle_authority_internal_sha256")
    if payload.get("config_sha256") != file_sha256(config_path):
        errors.append("bundle_config_sha256")
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    if bool(config.get("curation") is not None) != bool(payload.get("curation") is not None):
        errors.append("bundle_curation_presence")
    if payload.get("test_npz_opened") is not False:
        errors.append("bundle_opened_test_npz")
    if payload.get("test_map_yaml_or_png_opened") is not False:
        errors.append("bundle_opened_test_map")
    for source in payload.get("sources", ()):
        authority = Path(str(source.get("authority", "")))
        if not authority.is_file() or file_sha256(authority) != source.get("authority_sha256"):
            errors.append("bundle_source_authority_" + str(source.get("name", "unknown")))
        manifest_path = source.get("task_manifest")
        if manifest_path:
            manifest_path = Path(manifest_path)
            if (
                not manifest_path.is_file()
                or file_sha256(manifest_path) != source.get("task_manifest_file_sha256")
            ):
                errors.append("bundle_source_manifest_" + str(source.get("name", "unknown")))
        exclusion = source.get("collection_exclusion")
        if exclusion is not None:
            exclusion_path = Path(str(exclusion.get("authority", "")))
            try:
                exclusion_raw = exclusion_path.read_bytes()
                exclusion_payload = json.loads(exclusion_raw.decode("utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                errors.append(
                    "bundle_source_exclusion_" + str(source.get("name", "unknown"))
                )
            else:
                exclusion_content = dict(exclusion_payload)
                exclusion_claimed = exclusion_content.pop(
                    "exclusion_authority_sha256", ""
                )
                if (
                    hashlib.sha256(exclusion_raw).hexdigest()
                    != exclusion.get("authority_file_sha256")
                    or exclusion_payload.get("schema")
                    != "DEPCarP3CollectionExclusionAuthorityV1"
                    or exclusion_payload.get("status") != "PASS"
                    or exclusion_claimed != canonical_sha256(exclusion_content)
                    or exclusion_claimed
                    != exclusion.get("exclusion_authority_sha256")
                    or len(exclusion_payload.get("entries", ()))
                    != int(exclusion.get("excluded_tasks", -1))
                ):
                    errors.append(
                        "bundle_source_exclusion_"
                        + str(source.get("name", "unknown"))
                    )
    curation = payload.get("curation")
    if curation is not None:
        if (
            curation.get("schema") != CURATION_SCHEMA
            or curation.get("status") != "PASS"
            or curation.get("policy") != CURATION_POLICY
            or curation.get("evaluator") != CURATION_EVALUATOR
            or curation.get("preserve_rejected_source_samples") is not True
        ):
            errors.append("bundle_curation_contract")
        curation_path = Path(str(curation.get("authority", "")))
        try:
            curation_raw = curation_path.read_bytes()
            curation_authority = json.loads(curation_raw.decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            errors.append("bundle_curation_authority_unreadable")
        else:
            claimed = curation_authority.get("curation_authority_sha256", "")
            content = dict(curation_authority)
            content.pop("curation_authority_sha256", None)
            if (
                hashlib.sha256(curation_raw).hexdigest()
                != curation.get("authority_file_sha256")
                or claimed != canonical_sha256(content)
                or claimed != curation.get("curation_authority_sha256")
                or curation_authority.get("schema") != CURATION_SCHEMA
                or curation_authority.get("status") != "PASS"
                or curation_authority.get("source_npz_modified") is not False
                or curation_authority.get("test_npz_opened") is not False
                or curation_authority.get("test_map_yaml_or_png_opened") is not False
            ):
                errors.append("bundle_curation_authority_identity")
            source_count = int(curation_authority.get("source_samples_evaluated", -1))
            accepted = int(curation_authority.get("accepted_samples", -1))
            rejected = int(curation_authority.get("rejected_samples", -1))
            entries = curation_authority.get("rejected_entries", ())
            if (
                source_count < 1
                or accepted != int(payload.get("samples", -1))
                or rejected != len(entries)
                or accepted + rejected != source_count
                or any(row.get("reason") != "initial_pose_infeasible" for row in entries)
            ):
                errors.append("bundle_curation_counts")
        tool_path = Path(str(payload.get("tool", "")))
        if (
            not tool_path.is_file()
            or file_sha256(tool_path) != payload.get("tool_sha256")
        ):
            errors.append("bundle_curation_tool_identity")
    return payload, hashlib.sha256(raw).hexdigest(), sorted(set(errors))


def inspect_coverage_entry(arguments):
    entry, sample_root = arguments
    path = geometry._safe_indexed_sample_path(sample_root, entry)
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != entry.get("content_sha256"):
        raise RuntimeError("coverage sample content SHA-256 mismatch")
    with np.load(io.BytesIO(raw), allow_pickle=False) as data:
        manifest = json.loads(str(data["manifest_json"].item()))
        context = str(manifest.get("metadata", {}).get("candidate_context", "UNKNOWN"))
        requested_gear = int(np.asarray(data["requested_gear"]).item())
    if (
        manifest.get("map_uuid") != entry.get("map_uuid")
        or manifest.get("split") != entry.get("split")
        or manifest.get("maneuver_mode") != entry.get("maneuver_mode")
        or context != entry.get("candidate_context")
    ):
        raise RuntimeError("coverage sample/index identity mismatch")
    return {
        "split": entry["split"],
        "maneuver_mode": entry["maneuver_mode"],
        "candidate_context": context,
        "requested_gear": GEAR_NAMES.get(requested_gear, "INVALID"),
    }


def coverage_gate(entries, sample_root, gates, workers):
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="p3v3-coverage") as pool:
        rows = list(pool.map(
            inspect_coverage_entry,
            ((entry, sample_root) for entry in entries),
        ))
    validation = [row for row in rows if row["split"] == "validation"]
    contexts = Counter(row["candidate_context"] for row in validation)
    maneuvers = Counter(row["maneuver_mode"] for row in validation)
    gears = Counter(row["requested_gear"] for row in validation)
    errors = []
    allowed = set(gates["allowed_candidate_contexts"])
    for name, count in sorted(contexts.items()):
        if name not in allowed and count:
            errors.append(
                "validation_candidate_context_unexpected_%s_frames_%d" % (name, count)
            )
    minimum_context = int(gates["minimum_validation_frames_per_candidate_context"])
    for name in gates["required_candidate_contexts"]:
        count = int(contexts.get(name, 0))
        if count < minimum_context:
            errors.append(
                "validation_candidate_context_%s_frames_%d_lt_%d"
                % (name, count, minimum_context)
            )
    minimum_mode = int(gates["minimum_validation_frames_per_maneuver"])
    for name in gates["required_maneuvers"]:
        count = int(maneuvers.get(name, 0))
        if count < minimum_mode:
            errors.append(
                "validation_maneuver_%s_frames_%d_lt_%d" % (name, count, minimum_mode)
            )
    minimum_gear = int(gates["minimum_validation_frames_per_requested_gear"])
    for name in gates["required_requested_gears"]:
        count = int(gears.get(name, 0))
        if count < minimum_gear:
            errors.append(
                "validation_requested_gear_%s_frames_%d_lt_%d"
                % (name, count, minimum_gear)
            )
    if gears.get("INVALID", 0):
        errors.append("validation_requested_gear_invalid")
    return {
        "schema": "DEPCarP3V3ValidationCoverageGateV1",
        "status": "PASS" if not errors else "FAIL",
        "errors": sorted(errors),
        "validation_frames": len(validation),
        "candidate_context": dict(sorted(contexts.items())),
        "maneuver_mode": dict(sorted(maneuvers.items())),
        "requested_gear": dict(sorted(gears.items())),
        "test_npz_opened": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path,
        default=ROOT / "dep_car/config/p3_v3_incremental.yaml",
    )
    parser.add_argument(
        "--authority", type=Path,
        default=ROOT / "data/p3_v3/bundle_v1/bundle_authority.json",
    )
    parser.add_argument(
        "--report", type=Path,
        default=DEFAULT_FORMAL_REPORT,
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--maximum-samples", type=int, default=0,
        help="diagnostic only; a limited audit can never produce PASS",
    )
    args = parser.parse_args(argv)
    if args.workers != 8:
        raise ValueError("P3 V3 qualification requires exactly 8 workers")
    if args.maximum_samples < 0:
        raise ValueError("maximum-samples cannot be negative")
    if args.maximum_samples and args.report.resolve() == DEFAULT_FORMAL_REPORT.resolve():
        raise ValueError(
            "a limited audit cannot overwrite the formal P3 development report"
        )
    config = load_config(args.config)
    authority, authority_file_hash, authority_errors = verify_bundle_authority(
        args.authority, args.config
    )
    index_path = Path(authority["index"]).resolve()
    sample_root = Path(authority["sample_root"]).resolve()
    maps_root = Path(authority["maps_root"]).resolve()
    if (
        file_sha256(index_path) != authority.get("index_sha256")
        or not sample_root.is_dir()
        or not maps_root.is_dir()
    ):
        authority_errors.append("bundle_index_or_root_identity")
    index = geometry.load_index_authority(
        index_path,
        sample_root,
        maps_root,
        expected_index_sha256=authority.get("index_sha256"),
        expected_content_aggregate=authority.get("content_aggregate_sha256"),
        expected_split_counts=authority.get("counts_by_split"),
    )
    _specs, map_contract = geometry.load_map_specs(
        maps_root,
        {str(entry["map_uuid"]) for entry in index.entries},
        expected_aggregate=authority.get("map_contract_aggregate_sha256"),
    )
    if (
        len(index.entries) != int(authority.get("samples", -1))
        or map_contract["aggregate_sha256"]
        != authority.get("map_contract_aggregate_sha256")
    ):
        authority_errors.append("bundle_declared_counts_or_map_contract")
    coverage = coverage_gate(
        index.entries, sample_root, config["gates"], args.workers
    )
    inner = geometry.run_audit(
        index_path,
        sample_root,
        maps_root,
        maximum_samples=args.maximum_samples,
        workers=args.workers,
        enforce_frozen_authority=False,
    )
    limited = bool(args.maximum_samples)
    configured_geometry_gates, geometry_failures = (
        evaluate_configured_geometry_gates(inner["statistics"], config["gates"])
    )
    operational = []
    if inner.get("sample_failures"):
        operational.append("sample_reaudit_failure")
    if inner.get("sample_files_audited") != (
        min(args.maximum_samples, len(index.entries)) if limited else len(index.entries)
    ):
        operational.append("development_index_coverage_incomplete")
    if inner.get("errors") not in (
        ["diagnostic_partial_or_nonfrozen_audit"],
        [],
    ):
        operational.append("inner_audit_unexpected_error")
    errors = sorted(set(
        authority_errors
        + coverage["errors"]
        + operational
        + ([] if limited else geometry_failures)
    ))
    if limited:
        status = "SMOKE"
        errors = sorted(set(errors + ["diagnostic_limited_bundle_audit"]))
    else:
        status = "PASS" if not errors else "FAIL"

    report = dict(inner)
    report["baseline_geometry_gates"] = report["gates"]
    report["gates"] = configured_geometry_gates
    report.update({
        "schema": REPORT_SCHEMA,
        "status": status,
        "errors": errors,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "qualification_eligible": bool(not limited and not authority_errors),
        "bundle_authority": {
            "schema": BUNDLE_SCHEMA,
            "path": str(args.authority.resolve()),
            "file_sha256": authority_file_hash,
            "bundle_authority_sha256": authority.get("bundle_authority_sha256"),
            "bundle_id": authority.get("bundle_id"),
            "sample_inventory_sha256": authority.get("sample_inventory_sha256"),
            "source_authorities": authority.get("sources"),
            "curation": authority.get("curation"),
            "verified": not authority_errors,
        },
        "validation_coverage_gate": coverage,
        "bundle_audit_implementation": {
            "tool": str(Path(__file__).resolve()),
            "tool_sha256": file_sha256(__file__),
            "delegated_geometry_tool": str(Path(geometry.__file__).resolve()),
            "delegated_geometry_tool_sha256": file_sha256(geometry.__file__),
        },
        "qualification_gate_contract": {
            "policy": "frozen_P3V3_or_stricter",
            "configured": {
                name: config["gates"][name] for name in GEOMETRY_GATE_KEYS
            },
            "frozen_baseline": {
                "maximum_zero_feasible_rate": geometry.OVERALL_ZERO_FEASIBLE_LIMIT,
                "maximum_per_mode_zero_feasible_rate": (
                    geometry.PER_MODE_ZERO_FEASIBLE_LIMIT
                ),
                "minimum_median_feasible_candidates": (
                    geometry.MINIMUM_MEDIAN_FEASIBLE
                ),
            },
            "weakened": False,
        },
    })
    report["scope"].update({
        "sample_inventory": "sealed_P3V3BundleAuthority_entries_only",
        "test_npz_opened": False,
        "test_map_yaml_or_png_opened": False,
        "test_split_used_for_tuning": False,
        "npz_files_modified": False,
        "geometry_or_gate_cli_overrides_available": False,
        "maximum_samples_is_nonqualifying": True,
        "initial_pose_curation_applied": authority.get("curation") is not None,
        "initial_pose_infeasible_samples_excluded": int(
            (authority.get("curation") or {}).get("rejected_samples", 0)
        ),
    })
    atomic_json(args.report, report)
    overall = report["statistics"]["overall"]["new"]
    print(json.dumps({
        "status": status,
        "report": str(args.report.resolve()),
        "bundle_authority_sha256": authority.get("bundle_authority_sha256"),
        "samples": report["sample_files_audited"],
        "candidates": report["statistics"]["overall"]["candidates"],
        "zero_feasible_rate": overall.get("zero_feasible_rate"),
        "feasible_candidates_median": overall.get("feasible_candidates_median"),
        "validation_coverage": coverage,
        "errors": errors,
        "parallel_workers": args.workers,
        "test_npz_opened": False,
    }, indent=2, sort_keys=True))
    return 0 if status in ("PASS", "SMOKE") else 1


if __name__ == "__main__":
    raise SystemExit(main())
