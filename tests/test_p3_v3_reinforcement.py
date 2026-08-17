import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import audit_p3_footprint_upgrade as geometry
import audit_p3_v3_bundle as bundle_audit
import build_p3_v3_bundle as bundle_builder
import generate_p3_v3_wave as wave
import reextract_p3_v3_base as reextract
from dep_car.core.types import Gear
from dep_car.global_planner.hybrid_astar import HybridPathPose


VALIDATION_UUID = "4b5d966a-238d-57a3-b9d9-8c3141ed98ee"


def test_incremental_config_freezes_development_only_wave_and_gates():
    config = wave.load_config(ROOT / "dep_car/config/p3_v3_incremental.yaml")

    assert set(config["wave"]["selected_split_maps"]) == {"train", "validation"}
    assert sum(config["wave"]["task_quotas"]["train"].values()) == 144
    assert sum(config["wave"]["task_quotas"]["validation"].values()) == 36
    assert config["wave"]["workers"] == 8
    assert config["gates"]["maximum_zero_feasible_rate"] == 0.10
    assert config["gates"]["allowed_candidate_contexts"] == ["MISSION", "RECOVERY"]


def test_map_inventory_filters_test_before_geometry_is_opened(tmp_path):
    maps = tmp_path / "maps"
    maps.mkdir()
    rows = (
        ("train", "00000000-0000-0000-0000-000000000001"),
        ("validation", VALIDATION_UUID),
        ("test", "00000002-0000-0000-0000-000000000000"),
    )
    for name, map_uuid in rows:
        folder = maps / name
        folder.mkdir()
        (folder / "manifest.json").write_text(json.dumps({
            "map_uuid": map_uuid,
            "name": name,
            "occupancy_sha256": "a" * 64,
            "seed": 1,
        }), encoding="utf-8")
        # Test geometry is deliberately malformed; inventory selection must not
        # attempt to parse it.
        (folder / "map.yaml").write_text("not: [yaml", encoding="utf-8")

    inventory = wave.load_map_inventory(maps)

    assert {row["split"] for row in inventory} == {"train", "validation"}
    assert all(row["manifest"]["name"] != "test" for row in inventory)


def test_exact_route_preflight_accepts_open_space_and_rejects_unsafe_space():
    path = [
        HybridPathPose(0.0, 0.0, 0.0, Gear.FORWARD, 0.0),
        HybridPathPose(0.5, 0.0, 0.0, Gear.FORWARD, 0.1),
        HybridPathPose(1.0, 0.1, 0.1, Gear.FORWARD, 0.1),
    ]
    spec = SimpleNamespace(resolution_m=0.1, origin_xy=(-20.0, -20.0))
    preflight = {
        "route_checkpoints": 3,
        "minimum_feasible_candidates_per_probe": 2,
        "forward_speed_probes_mps": [0.0, 0.6],
        "reverse_speed_probes_mps": [0.0, -0.2],
        "require_every_probe": True,
    }
    open_sdf = geometry._signed_distance_field(
        np.ones((400, 400), dtype=bool), 0.1
    )[None, None]
    unsafe_sdf = geometry._signed_distance_field(
        np.zeros((400, 400), dtype=bool), 0.1
    )[None, None]

    accepted = wave.preflight_route(path, spec, open_sdf, preflight)
    rejected = wave.preflight_route(path, spec, unsafe_sdf, preflight)

    assert accepted["schema"] == wave.PREFLIGHT_SCHEMA
    assert accepted["route_checkpoints"] == 3
    assert accepted["minimum_feasible_candidates"] >= 2
    assert rejected is None


def test_proposal_selection_enforces_split_quotas_and_map_capacity():
    def proposal(split, mode, map_uuid, number):
        return {
            "proposal_id": "%s-%s-%d" % (map_uuid, mode, number),
            "map_uuid": map_uuid,
            "map_split": split,
            "maneuver_mode": mode,
        }

    results = []
    for split, mode, maps in (
        ("train", "NARROW_CORRIDOR", ("t1", "t2")),
        ("validation", "THREE_POINT_TURN", ("v1", "v2")),
    ):
        for map_uuid in maps:
            results.append({
                "split": split,
                "mode": mode,
                "accepted": [proposal(split, mode, map_uuid, index) for index in range(3)],
                "failures": {},
            })
    quotas = {
        "train": {"NARROW_CORRIDOR": 4},
        "validation": {"THREE_POINT_TURN": 2},
    }

    selected, deficits, _failures, map_counts = wave.select_proposals(
        results, quotas, maximum_per_map=2
    )

    assert not deficits
    assert len(selected) == 6
    assert max(map_counts.values()) == 2
    assert sum(row["map_split"] == "validation" for row in selected) == 2


def write_context_sample(path, context="MISSION", requested_gear=1):
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "map_uuid": path.parent.name,
        "split": "validation",
        "maneuver_mode": "NORMAL",
        "metadata": {
            "candidate_context": context,
            "source_bag_sha256": "b" * 64,
        },
    }
    np.savez_compressed(
        path,
        manifest_json=np.asarray(json.dumps(manifest, sort_keys=True)),
        requested_gear=np.asarray(requested_gear, dtype=np.int8),
    )
    return manifest


def test_bundle_copy_requires_known_context_and_independent_bytes(tmp_path):
    source = tmp_path / "source" / VALIDATION_UUID / "sample.npz"
    write_context_sample(source, context="RECOVERY", requested_gear=-1)
    output = tmp_path / "bundle"
    row = {
        "source_name": "wave",
        "source": source,
        "relative": Path(VALIDATION_UUID) / source.name,
        "split": "validation",
    }

    result = bundle_builder.inspect_and_copy(row, output)
    destination = output / "samples" / row["relative"]

    assert destination.is_file()
    assert destination.stat().st_ino != source.stat().st_ino
    assert result["candidate_context"] == "RECOVERY"
    assert result["requested_gear"] == -1
    assert result["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()


def test_coverage_gate_verifies_context_against_npz(tmp_path):
    sample_root = tmp_path / "samples"
    sample = sample_root / VALIDATION_UUID / "sample.npz"
    write_context_sample(sample, context="MISSION", requested_gear=1)
    entry = {
        "path": str(sample.relative_to(sample_root)),
        "content_sha256": hashlib.sha256(sample.read_bytes()).hexdigest(),
        "map_uuid": VALIDATION_UUID,
        "split": "validation",
        "maneuver_mode": "NORMAL",
        "candidate_context": "MISSION",
    }
    gates = {
        "allowed_candidate_contexts": ["MISSION", "RECOVERY"],
        "required_candidate_contexts": ["MISSION"],
        "minimum_validation_frames_per_candidate_context": 1,
        "required_maneuvers": ["NORMAL"],
        "minimum_validation_frames_per_maneuver": 1,
        "required_requested_gears": ["FORWARD"],
        "minimum_validation_frames_per_requested_gear": 1,
    }

    report = bundle_audit.coverage_gate([entry], sample_root, gates, workers=1)

    assert report["status"] == "PASS"
    assert report["candidate_context"] == {"MISSION": 1}
    assert report["requested_gear"] == {"FORWARD": 1}
    assert report["test_npz_opened"] is False


def test_reextract_reuse_is_content_addressed(tmp_path):
    output = tmp_path / "output"
    sample = output / "samples" / VALIDATION_UUID / "sample.npz"
    write_context_sample(sample)
    row = {
        "status": "COMPLETE",
        "samples": [{
            "path": str(sample.relative_to(output)),
            "size_bytes": sample.stat().st_size,
            "sha256": hashlib.sha256(sample.read_bytes()).hexdigest(),
        }],
    }

    assert reextract.task_result_is_reusable(row, output)
    with sample.open("ab") as stream:
        stream.write(b"tamper")
    assert not reextract.task_result_is_reusable(row, output)


def test_limited_bundle_audit_cannot_overwrite_formal_report():
    with pytest.raises(ValueError, match="cannot overwrite"):
        bundle_audit.main(["--maximum-samples", "1"])
