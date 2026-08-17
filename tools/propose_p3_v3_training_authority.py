#!/usr/bin/env python3
"""Emit (but never apply) the training.yaml authority update after P3 V3 PASS."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path):
    path = Path(path).resolve()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--authority", type=Path,
        default=ROOT / "data/p3_v3/bundle_v1/bundle_authority.json",
    )
    parser.add_argument(
        "--reaudit", type=Path,
        default=ROOT / "reports/p3_development_reaudit_v3.json",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "reports/p3_v3_training_authority_proposal.json",
    )
    args = parser.parse_args(argv)
    authority = json.loads(args.authority.read_text(encoding="utf-8"))
    report = json.loads(args.reaudit.read_text(encoding="utf-8"))
    if authority.get("schema") != "DEPCarP3V3BundleAuthorityV1":
        raise ValueError("P3 V3 bundle authority schema mismatch")
    if report.get("schema") != "DEPCarP3DevelopmentReauditV3":
        raise ValueError("P3 V3 re-audit schema mismatch")
    if report.get("status") != "PASS" or report.get("errors"):
        raise RuntimeError("P3 V3 re-audit is not PASS")
    if report.get("validation_coverage_gate", {}).get("status") != "PASS":
        raise RuntimeError("P3 V3 validation coverage is not PASS")
    report_bundle = report.get("bundle_authority", {})
    if (
        report_bundle.get("file_sha256") != file_sha256(args.authority)
        or report_bundle.get("bundle_authority_sha256")
        != authority.get("bundle_authority_sha256")
    ):
        raise RuntimeError("P3 V3 report/bundle identity mismatch")
    training = report.get("training_authority", {})
    expected = {
        "index_sha256": authority.get("index_sha256"),
        "content_aggregate_sha256": authority.get("content_aggregate_sha256"),
        "map_contract_aggregate_sha256": authority.get(
            "map_contract_aggregate_sha256"
        ),
        "splits": ["train", "validation"],
        "test_split_used": False,
    }
    if any(training.get(key) != value for key, value in expected.items()):
        raise RuntimeError("P3 V3 report training authority mismatch")
    payload = {
        "schema": "DEPCarP3V3TrainingAuthorityProposalV1",
        "status": "READY_FOR_EXPLICIT_P5_CONFIG_APPROVAL",
        "applied": False,
        "formal_training_started": False,
        "bundle_authority": relative(args.authority),
        "bundle_authority_sha256": file_sha256(args.authority),
        "reaudit": relative(args.reaudit),
        "reaudit_sha256": file_sha256(args.reaudit),
        "training_yaml_changes": {
            "dataset.root": relative(authority["sample_root"]),
            "dataset.maps": relative(authority["maps_root"]),
            "dataset.index": relative(authority["index"]),
            "dataset.content_aggregate_sha256": authority[
                "content_aggregate_sha256"
            ],
            "dataset.map_contract_aggregate_sha256": authority[
                "map_contract_aggregate_sha256"
            ],
            "qualification.corrected_footprint_p3_status": "PASS",
            "qualification.p5_formal_training_allowed": True,
            "qualification.blocked_gates": [],
        },
        "approval_boundary": (
            "This proposal does not edit training.yaml and does not authorize or "
            "start P5 training."
        ),
    }
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
