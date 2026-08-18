from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import audit_p3_v3_bundle as audit
import build_p3_v3_bundle as builder


def test_curated_config_selects_frozen_initial_pose_policy():
    path = ROOT / "dep_car/config/p3_v3_curated.yaml"
    built = builder.load_config(path)
    audited = audit.load_config(path)
    assert built["curation"] == audited["curation"]
    assert built["curation"] == {
        "schema": builder.CURATION_SCHEMA,
        "policy": builder.CURATION_POLICY,
        "evaluator": builder.CURATION_EVALUATOR,
        "preserve_rejected_source_samples": True,
    }
    assert built["bundle"]["output"] == "data/p3_v3/bundle_v2_curated"


@pytest.mark.parametrize("loader", (builder.load_config, audit.load_config))
def test_curation_policy_cannot_be_weakened(tmp_path, loader):
    source = yaml.safe_load(
        (ROOT / "dep_car/config/p3_v3_curated.yaml").read_text(encoding="utf-8")
    )
    changed = deepcopy(source)
    changed["curation"]["evaluator"] = "point_robot"
    path = tmp_path / "changed.yaml"
    path.write_text(yaml.safe_dump(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="curation differs"):
        loader(path)


def test_legacy_v1_config_remains_readable_without_curation():
    path = ROOT / "dep_car/config/p3_v3_incremental.yaml"
    assert builder.load_config(path).get("curation") is None
    assert audit.load_config(path).get("curation") is None


def test_v4_config_may_tighten_but_not_weaken_geometry_gates(tmp_path):
    path = ROOT / "dep_car/config/p3_v4_corner_supplement.yaml"
    config = audit.load_config(path)
    assert config["gates"]["maximum_zero_feasible_rate"] == 0.08
    assert config["gates"]["maximum_per_mode_zero_feasible_rate"] == 0.20
    assert config["gates"]["minimum_median_feasible_candidates"] == 3

    for name, weakened in (
        ("maximum_zero_feasible_rate", 0.11),
        ("maximum_per_mode_zero_feasible_rate", 0.26),
        ("minimum_median_feasible_candidates", 1),
    ):
        changed = deepcopy(config)
        changed["gates"][name] = weakened
        changed_path = tmp_path / (name + ".yaml")
        changed_path.write_text(yaml.safe_dump(changed), encoding="utf-8")
        with pytest.raises(ValueError, match="weakens"):
            audit.load_config(changed_path)


def test_configured_geometry_gates_enforce_v4_thresholds():
    statistics = {
        "overall": {
            "new": {
                "zero_feasible_rate": 0.09,
                "feasible_candidates_median": 14.0,
            }
        },
        "by_mode": {
            mode: {"new": {"zero_feasible_rate": 0.10}}
            for mode in audit.geometry.PILOT_MANEUVER_MODES
        },
    }
    statistics["by_mode"]["SHARP_TURN"]["new"]["zero_feasible_rate"] = 0.21
    gates = {
        "maximum_zero_feasible_rate": 0.08,
        "maximum_per_mode_zero_feasible_rate": 0.20,
        "minimum_median_feasible_candidates": 3,
    }
    checks, failures = audit.evaluate_configured_geometry_gates(statistics, gates)
    assert checks["configured_overall_zero_feasible_rate"]["status"] == "FAIL"
    assert checks["configured_per_mode_zero_feasible_rate"]["SHARP_TURN"][
        "status"
    ] == "FAIL"
    assert "configured_overall_zero_feasible_rate" in failures
    assert "configured_mode_zero_feasible_rate_SHARP_TURN" in failures


def test_bundle_verifier_authenticates_curated_inventory(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(
        (ROOT / "dep_car/config/p3_v3_curated.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    rejected = {
        "path": "map/sample.npz",
        "sha256": "a" * 64,
        "source_name": "source",
        "reason": "initial_pose_infeasible",
    }
    curation = {
        "schema": audit.CURATION_SCHEMA,
        "status": "PASS",
        "qualification_authority": True,
        "policy": audit.CURATION_POLICY,
        "evaluator": audit.CURATION_EVALUATOR,
        "source_samples_evaluated": 3,
        "accepted_samples": 2,
        "rejected_samples": 1,
        "rejection_reason": "initial_pose_infeasible",
        "rejected_entries": [rejected],
        "preserve_rejected_source_samples": True,
        "source_npz_modified": False,
        "test_npz_opened": False,
        "test_map_yaml_or_png_opened": False,
    }
    curation["curation_authority_sha256"] = audit.canonical_sha256(curation)
    curation_path = tmp_path / "curation.json"
    curation_path.write_text(json.dumps(curation) + "\n", encoding="utf-8")
    bundle = {
        "schema": audit.BUNDLE_SCHEMA,
        "status": "SEALED",
        "config_sha256": audit.file_sha256(config),
        "tool": str((ROOT / "tools/build_p3_v3_bundle.py").resolve()),
        "tool_sha256": audit.file_sha256(ROOT / "tools/build_p3_v3_bundle.py"),
        "sources": [],
        "samples": 2,
        "test_npz_opened": False,
        "test_map_yaml_or_png_opened": False,
        "curation": {
            "schema": audit.CURATION_SCHEMA,
            "status": "PASS",
            "policy": audit.CURATION_POLICY,
            "evaluator": audit.CURATION_EVALUATOR,
            "authority": str(curation_path),
            "authority_file_sha256": hashlib.sha256(
                curation_path.read_bytes()
            ).hexdigest(),
            "curation_authority_sha256": curation[
                "curation_authority_sha256"
            ],
            "source_samples_evaluated": 3,
            "accepted_samples": 2,
            "rejected_samples": 1,
            "preserve_rejected_source_samples": True,
        },
    }
    bundle["bundle_authority_sha256"] = audit.canonical_sha256(bundle)
    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_text(json.dumps(bundle) + "\n", encoding="utf-8")
    _payload, _file_hash, errors = audit.verify_bundle_authority(
        bundle_path, config
    )
    assert errors == []

    curation_path.write_text("{}\n", encoding="utf-8")
    _payload, _file_hash, errors = audit.verify_bundle_authority(
        bundle_path, config
    )
    assert "bundle_curation_authority_identity" in errors
