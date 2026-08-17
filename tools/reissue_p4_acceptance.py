#!/usr/bin/env python3
"""Reissue the canonical P4 acceptance from authenticated P3/P4/P5 evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
MODALITIES = ("depth_only", "lidar_only", "fusion")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def read_json(path: Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected a JSON object: {path}")
    return payload


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def dotted_get(payload: dict, dotted_key: str):
    value = payload
    for component in dotted_key.split("."):
        if not isinstance(value, dict) or component not in value:
            raise RuntimeError(f"training configuration is missing {dotted_key}")
        value = value[component]
    return value


def resolve_project_path(value: str, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    try:
        path.relative_to(ROOT)
    except ValueError as exc:
        raise RuntimeError(f"{label} escapes the project root: {path}") from exc
    require(path.exists(), f"{label} does not exist: {path}")
    return path


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def index_summary(index: dict) -> dict:
    entries = index.get("entries")
    require(isinstance(entries, list), "training index entries are missing")
    split_counts = Counter()
    split_maps = defaultdict(set)
    for entry in entries:
        require(isinstance(entry, dict), "training index contains a non-object entry")
        split = entry.get("split")
        map_uuid = entry.get("map_uuid")
        require(split in {"train", "validation"}, f"unexpected split: {split}")
        require(isinstance(map_uuid, str) and map_uuid, "index entry has no map_uuid")
        split_counts[split] += 1
        split_maps[split].add(map_uuid)
    require(len(entries) == index.get("samples"), "training index sample count mismatch")
    return {
        "samples": len(entries),
        "counts_by_split": dict(sorted(split_counts.items())),
        "maps_by_split": {
            split: len(map_uuids) for split, map_uuids in sorted(split_maps.items())
        },
        "total_maps": len(set().union(*split_maps.values())),
    }


def validate_proposal_application(proposal: dict, training: dict) -> dict:
    require(
        proposal.get("schema") == "DEPCarP3V3TrainingAuthorityProposalV1",
        "proposal schema mismatch",
    )
    require(
        proposal.get("status") == "READY_FOR_EXPLICIT_P5_CONFIG_APPROVAL",
        "proposal is not ready for explicit approval",
    )
    require(proposal.get("formal_training_started") is False, "proposal started training")
    changes = proposal.get("training_yaml_changes")
    require(isinstance(changes, dict) and changes, "proposal has no training changes")
    mismatches = {
        key: {"expected": expected, "actual": dotted_get(training, key)}
        for key, expected in changes.items()
        if dotted_get(training, key) != expected
    }
    require(not mismatches, f"proposal is not applied exactly: {mismatches}")
    return {
        "status": "APPLIED_EXACTLY",
        "changes": changes,
        "mismatches": mismatches,
    }


def all_dry_run_gates_pass(dry_run: dict) -> bool:
    gate_names = (
        "p3_footprint_gate",
        "index_content_gate",
        "dataset_authority_gate",
        "validation_coverage_gate",
        "training_yaml_qualification_gate",
    )
    return all(dry_run.get(name, {}).get("passed") is True for name in gate_names)


def build_acceptance(
    *,
    proposal_path: Path,
    training_path: Path,
    p4_machine_path: Path,
    dry_run_paths: dict[str, Path],
) -> dict:
    proposal = read_json(proposal_path)
    training = yaml.safe_load(training_path.read_text(encoding="utf-8"))
    require(isinstance(training, dict), "training YAML is not a mapping")
    application = validate_proposal_application(proposal, training)

    bundle_path = resolve_project_path(proposal["bundle_authority"], "bundle authority")
    p3_path = resolve_project_path(proposal["reaudit"], "P3 re-audit")
    require(
        file_sha256(bundle_path) == proposal.get("bundle_authority_sha256"),
        "proposal bundle file hash mismatch",
    )
    require(
        file_sha256(p3_path) == proposal.get("reaudit_sha256"),
        "proposal P3 re-audit file hash mismatch",
    )
    bundle = read_json(bundle_path)
    p3 = read_json(p3_path)
    p4_machine = read_json(p4_machine_path)

    require(bundle.get("schema") == "DEPCarP3V3BundleAuthorityV1", "bundle schema mismatch")
    require(bundle.get("status") == "SEALED", "bundle is not sealed")
    require(p3.get("schema") == "DEPCarP3DevelopmentReauditV3", "P3 schema mismatch")
    require(p3.get("status") == "PASS" and not p3.get("errors"), "P3 re-audit is not PASS")
    require(
        p3.get("qualification_eligible") is True,
        "P3 re-audit is not qualification eligible",
    )
    require(
        p3.get("validation_coverage_gate", {}).get("status") == "PASS",
        "P3 validation coverage is not PASS",
    )
    require(
        p4_machine.get("schema") == "DEPCarP4ImplementationAcceptanceV3",
        "P4 machine report schema mismatch",
    )
    require(
        p4_machine.get("status") == "PASS" and not p4_machine.get("errors"),
        "P4 machine report is not PASS",
    )
    require(
        p4_machine.get("production_qualified") is False,
        "P4 smoke must not claim production qualification",
    )

    configured_index = resolve_project_path(
        dotted_get(training, "dataset.index"), "training index"
    )
    configured_root = resolve_project_path(
        dotted_get(training, "dataset.root"), "training sample root"
    )
    configured_maps = resolve_project_path(
        dotted_get(training, "dataset.maps"), "training maps root"
    )
    index = read_json(configured_index)
    counts = index_summary(index)
    index_hash = file_sha256(configured_index)
    content_hash = dotted_get(training, "dataset.content_aggregate_sha256")
    map_hash = dotted_get(training, "dataset.map_contract_aggregate_sha256")
    require(configured_index == Path(bundle["index"]).resolve(), "bundle/index path mismatch")
    require(configured_root == Path(bundle["sample_root"]).resolve(), "bundle/sample path mismatch")
    require(configured_maps == Path(bundle["maps_root"]).resolve(), "bundle/maps path mismatch")
    require(index_hash == bundle.get("index_sha256"), "bundle/index file hash mismatch")
    require(index.get("content_aggregate_sha256") == content_hash, "index content hash mismatch")
    require(bundle.get("content_aggregate_sha256") == content_hash, "bundle content hash mismatch")
    require(bundle.get("map_contract_aggregate_sha256") == map_hash, "bundle map hash mismatch")
    require(counts["samples"] == bundle.get("samples"), "bundle sample count mismatch")

    machine_dataset = p4_machine.get("dataset", {})
    machine_loss = p4_machine.get("loss_contract", {})
    require(machine_dataset.get("training_index_sha256") == index_hash, "P4/index hash mismatch")
    require(
        machine_dataset.get("training_index_content_aggregate_sha256") == content_hash,
        "P4/content hash mismatch",
    )
    require(
        machine_dataset.get("actual_map_contract_aggregate_sha256") == map_hash,
        "P4/map hash mismatch",
    )
    require(
        machine_loss.get("training_yaml_sha256") == file_sha256(training_path),
        "P4/training YAML hash mismatch",
    )
    embedded_p3 = p4_machine.get("p3_development_reaudit", {})
    require(embedded_p3.get("sha256") == file_sha256(p3_path), "P4/P3 hash mismatch")
    require(
        embedded_p3.get("status") == "PASS"
        and embedded_p3.get("p5_gate_eligible") is True
        and not embedded_p3.get("p5_gate_errors"),
        "P4 embedded P3 gate is not eligible",
    )
    require(
        p4_machine.get("sealed_test_data", {}).get("test_split_accessed_by_p4") is False,
        "P4 accessed sealed test data",
    )

    dry_runs = {}
    for modality in MODALITIES:
        path = dry_run_paths[modality]
        dry_run = read_json(path)
        require(dry_run.get("modality") == modality, f"{modality} dry-run identity mismatch")
        require(dry_run.get("status") == "DRY_RUN_READY", f"{modality} dry-run is not ready")
        require(
            dry_run.get("formal_training_authorized") is True,
            f"{modality} formal training is not authorized",
        )
        require(all_dry_run_gates_pass(dry_run), f"{modality} dry-run has a failed gate")
        require(dry_run.get("sealed_test_split") is True, f"{modality} test split is not sealed")
        require(dry_run.get("permanent_smoke") is False, f"{modality} is a smoke lineage")
        require(dry_run.get("bounded_smoke_authorized") is False, f"{modality} bounded smoke enabled")
        require(dry_run.get("samples") == counts["counts_by_split"], f"{modality} sample mismatch")
        require(
            dry_run.get("training_config", {}).get("file_sha256")
            == file_sha256(training_path),
            f"{modality} training configuration hash mismatch",
        )
        require(
            dry_run.get("trainer_sha256")
            == file_sha256(ROOT / "tools/train_dep_car.py"),
            f"{modality} trainer hash mismatch",
        )
        require(
            dry_run.get("dataset_authority_gate", {}).get(
                "actual_content_aggregate_sha256"
            )
            == content_hash,
            f"{modality} content authority mismatch",
        )
        require(
            dry_run.get("dataset_authority_gate", {}).get(
                "actual_map_contract_aggregate_sha256"
            )
            == map_hash,
            f"{modality} map authority mismatch",
        )
        dry_runs[modality] = {
            "report": relative(path),
            "report_sha256": file_sha256(path),
            "status": dry_run["status"],
            "device": dry_run.get("device"),
            "formal_training_authorized": True,
            "all_gates_passed": True,
            "sealed_test_split": True,
            "permanent_smoke": False,
            "bounded_smoke_authorized": False,
            "training_config_sha256": dry_run["training_config"][
                "file_sha256"
            ],
            "trainer_sha256": dry_run["trainer_sha256"],
            "validation_requested_gear_gate": dry_run.get(
                "validation_coverage_gate", {}
            ).get("requested_gear", {}).get("status"),
        }

    training_artifacts = sorted((ROOT / "models/dep_car/p5").glob("*.pth"))
    require(
        not training_artifacts,
        "P5 checkpoint artifacts already exist; cannot attest that training has not started: "
        + ", ".join(relative(path) for path in training_artifacts),
    )

    p3_metrics = p3.get("statistics", {}).get("overall", {}).get("new")
    if not p3_metrics:
        p3_metrics = p3.get("overall_new_metrics", {})
    candidate = p4_machine["candidate_capacity_tiny_overfit"]
    score = p4_machine["score_calibration_tiny_overfit"]
    oracle = candidate["direct_residual_oracle"]
    oracle_gate = candidate["direct_residual_oracle_gate"]
    require(not oracle_gate.get("errors"), "P4 direct residual oracle gate failed")
    require(oracle_gate.get("no_worse_than_network_terminal") is True, "P4 oracle is invalid")

    return {
        "schema": "DEPCarP4AcceptanceV2",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "scope": (
            "P0-P4 development acceptance and authenticated P5 entry readiness; "
            "formal P5 training and P6/P8 qualification are not claimed"
        ),
        "architecture_id": p4_machine.get("architecture_id"),
        "production_qualified": False,
        "p5_formal_training_allowed": True,
        "p5_formal_training_started": False,
        "errors": [],
        "proposal_application": {
            "proposal": relative(proposal_path),
            "proposal_sha256": file_sha256(proposal_path),
            **application,
            "training_yaml": relative(training_path),
            "training_yaml_sha256": file_sha256(training_path),
        },
        "development_dataset": {
            "bundle_authority": relative(bundle_path),
            "bundle_authority_sha256": file_sha256(bundle_path),
            "bundle_id": bundle.get("bundle_id"),
            "bundle_status": bundle.get("status"),
            "sample_root": relative(configured_root),
            "maps_root": relative(configured_maps),
            "index": relative(configured_index),
            "index_sha256": index_hash,
            "content_aggregate_sha256": content_hash,
            "map_contract_aggregate_sha256": map_hash,
            **counts,
            "curation": bundle.get("curation"),
            "test_accessed": False,
        },
        "p3_development_gate": {
            "report": relative(p3_path),
            "report_sha256": file_sha256(p3_path),
            "status": p3.get("status"),
            "errors": p3.get("errors"),
            "samples_audited": p3.get("sample_files_audited"),
            "overall_zero_feasible_rate": p3_metrics.get("zero_feasible_rate"),
            "overall_median_feasible_candidates": p3_metrics.get(
                "feasible_candidates_median"
            ),
            "all_mode_gates_passed": all(
                gate.get("status") == "PASS"
                for gate in p3.get("gates", {})
                .get("per_mode_zero_feasible_rate_lt_0_25", {})
                .values()
            ),
            "validation_coverage_status": p3.get("validation_coverage_gate", {}).get(
                "status"
            ),
            "test_accessed": False,
        },
        "p4_machine_verification": {
            "report": relative(p4_machine_path),
            "report_sha256": file_sha256(p4_machine_path),
            "status": p4_machine.get("status"),
            "errors": p4_machine.get("errors"),
            "device": p4_machine.get("execution", {}).get("device"),
            "torch_threads": p4_machine.get("execution", {}).get("torch_threads"),
            "elapsed_seconds": p4_machine.get("execution", {}).get(
                "elapsed_seconds"
            ),
            "smoke_samples": machine_dataset.get("smoke_samples"),
            "candidate_initial_loss": candidate.get("initial"),
            "candidate_terminal_loss": candidate.get("terminal_last"),
            "direct_residual_oracle_floor": oracle.get("achieved_floor"),
            "direct_residual_oracle_gate": oracle_gate,
            "score_initial_loss": score.get("initial"),
            "score_terminal_loss": score.get("terminal_last"),
            "checkpoint_roundtrip_max_abs_error": p4_machine.get(
                "checkpoint_roundtrip_max_abs_error"
            ),
            "modalities_passed": sorted(p4_machine.get("modalities", {})),
            "test_accessed": False,
        },
        "p5_entry_verification": {
            "status": "READY_TO_START_CANDIDATE_CAPACITY",
            "dry_runs": dry_runs,
            "formal_checkpoint_artifacts": [],
            "formal_training_started": False,
            "validation_requested_gear_gate": (
                "DEFERRED_TO_LOADER_AND_CANDIDATE_ACCEPTANCE"
            ),
            "required_stage_order": [
                "candidate_capacity",
                "candidate_acceptance",
                "score_calibration",
            ],
            "modalities": list(MODALITIES),
        },
        "next_stage_requirements": [
            "run candidate_capacity separately for depth_only, lidar_only, and fusion",
            "accept each candidate checkpoint before starting its score_calibration stage",
            "keep all P5 checkpoints UNQUALIFIED until P5 acceptance succeeds",
            "qualify fusion depth-missing and lidar-missing behavior before P6",
            "implement and qualify the DEPCarNetV1 ROS adapter before P6 shadow mode",
        ],
        "issuer": {
            "tool": relative(Path(__file__)),
            "tool_sha256": file_sha256(Path(__file__)),
        },
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--proposal",
        type=Path,
        default=ROOT / "reports/p3_v3_training_authority_proposal.json",
    )
    parser.add_argument(
        "--training",
        type=Path,
        default=ROOT / "dep_car/config/training.yaml",
    )
    parser.add_argument(
        "--p4-machine",
        type=Path,
        default=ROOT / "reports/p4_model_implementation_acceptance.json",
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "reports/p4_acceptance.json"
    )
    args = parser.parse_args(argv)
    dry_runs = {
        modality: ROOT / f"reports/p5_{modality}_candidate_dry_run.json"
        for modality in MODALITIES
    }
    payload = build_acceptance(
        proposal_path=args.proposal.resolve(),
        training_path=args.training.resolve(),
        p4_machine_path=args.p4_machine.resolve(),
        dry_run_paths=dry_runs,
    )
    atomic_json(args.output.resolve(), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
