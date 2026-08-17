import hashlib
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import audit_p3_footprint_upgrade as audit
from dep_car.core.occupancy import (
    TRAINING_FOOTPRINT_ALLOWANCE,
    FootprintConfig,
    OccupancyGrid2D,
)


TRAIN_UUID = "00000000-0000-0000-0000-000000000000"
TEST_UUID = "00000002-0000-0000-0000-000000000000"


def candidate_bank(points=2):
    trajectories = np.zeros((audit.EXPECTED_CANDIDATES, points, 6), dtype=np.float64)
    trajectories[..., 0] = np.arange(points, dtype=np.float64)
    trajectories[..., 1] = np.arange(1, points + 1, dtype=np.float64)
    trajectories[..., 4] = 0.5
    trajectories[..., 5] = 0.1
    return trajectories


def write_map(maps_root, map_uuid=TRAIN_UUID, pixels=None):
    folder = maps_root / ("dep_car_map_0000_" + map_uuid[:8])
    folder.mkdir(parents=True)
    if pixels is None:
        pixels = np.full((200, 200), 255, dtype=np.uint8)
    image = folder / "map.png"
    assert cv2.imwrite(str(image), pixels)
    decoded = hashlib.sha256(np.asarray(pixels, dtype=np.uint8).tobytes()).hexdigest()
    (folder / "manifest.json").write_text(json.dumps({
        "schema": "DEPCarArenaStaticMapV1",
        "map_uuid": map_uuid,
        "occupancy_sha256": decoded,
    }), encoding="utf-8")
    (folder / "map.yaml").write_text(
        "image: map.png\n"
        "resolution: 0.05\n"
        "origin: [-5.0, -5.0, 0.0]\n"
        "negate: 0\n"
        "occupied_thresh: 0.65\n"
        "free_thresh: 0.196\n",
        encoding="utf-8",
    )
    return folder, decoded


def write_sample(sample_root, occupancy_sha256, map_uuid=TRAIN_UUID):
    folder = sample_root / map_uuid
    folder.mkdir(parents=True)
    path = folder / "sample-000000.npz"
    manifest = {
        "schema": audit.EXPECTED_SAMPLE_SCHEMA,
        "contract_revision": audit.EXPECTED_CONTRACT_REVISION,
        "sensor_authority": audit.EXPECTED_SENSOR_AUTHORITY,
        "map_uuid": map_uuid,
        "split": "train",
        "maneuver_mode": "NORMAL",
        "preprocessing": {
            "lidar_bev": {"sha256": audit.EXPECTED_BEV_PREPROCESSING_SHA256}
        },
        "metadata": {
            "formal_training_authority": True,
            "map_occupancy_sha256": occupancy_sha256,
            "sample_id": path.stem,
        },
        "transforms": {
            "chassis_to_map": {
                "source_frame": "chassis",
                "target_frame": "map",
                "matrix": np.eye(4).tolist(),
            }
        },
    }
    state = np.zeros(9, dtype=np.float32)
    np.savez_compressed(
        str(path),
        manifest_json=np.asarray(json.dumps(manifest, sort_keys=True)),
        vehicle_state=state,
        current_gear=np.asarray(1, dtype=np.int8),
        requested_gear=np.asarray(1, dtype=np.int8),
        # Old labels remain present only to exercise their diagnostic path.
        trajectories=np.full((15, 11, 6), np.nan, dtype=np.float32),
        feasible=np.ones(15, dtype=np.uint8),
        static_clearance=np.ones(15, dtype=np.float32),
    )
    stat = path.stat()
    entry = {
        "path": str(path.relative_to(sample_root)),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "content_sha256": audit.file_sha256(path),
        "map_uuid": map_uuid,
        "split": "train",
        "maneuver_mode": "NORMAL",
        "sample_id": path.stem,
        "task_id": "task-0",
        "candidate_context": "MISSION",
    }
    return path, entry


def write_index(index_path, sample_root, maps_root, entries):
    entries = list(entries)
    payload = {
        "schema": audit.EXPECTED_INDEX_SCHEMA,
        "training_view": audit.EXPECTED_TRAINING_VIEW,
        "content_hash_algorithm": audit.TRAINING_INDEX_CONTENT_HASH_ALGORITHM,
        "content_aggregate_schema": audit.TRAINING_INDEX_CONTENT_AGGREGATE_SCHEMA,
        "content_aggregate_sha256": audit.training_index_content_aggregate(entries),
        "created_at_unix": 0.0,
        "sample_root": str(sample_root.resolve()),
        "maps_root": str(maps_root.resolve()),
        "splits": ["train", "validation"],
        "workers": audit.EXPECTED_WORKERS,
        "sensor_authority": audit.EXPECTED_SENSOR_AUTHORITY,
        "bev_preprocessing_sha256": audit.EXPECTED_BEV_PREPROCESSING_SHA256,
        "samples": len(entries),
        "counts_by_split": {"train": len(entries)},
        "counts_by_mode": {"NORMAL": len(entries)},
        "entries": entries,
    }
    index_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


def development_fixture(tmp_path):
    sample_root = tmp_path / "samples"
    maps_root = tmp_path / "maps"
    sample_root.mkdir()
    maps_root.mkdir()
    _, occupancy_sha = write_map(maps_root)
    sample_path, entry = write_sample(sample_root, occupancy_sha)
    index_path = tmp_path / "training_index.json"
    payload = write_index(index_path, sample_root, maps_root, [entry])
    return sample_root, maps_root, sample_path, entry, index_path, payload


def test_candidate_bank_transform_applies_map_from_chassis_pose():
    trajectories = candidate_bank()
    transform = np.array(
        (
            (0.0, -1.0, 0.0, 3.0),
            (1.0, 0.0, 0.0, 4.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )

    world = audit.transform_candidate_bank_to_map(trajectories, transform)

    np.testing.assert_allclose(world[..., 1], 3.0, atol=1e-12)
    np.testing.assert_allclose(world[:, :, 2], ((5.0, 6.0),) * 15, atol=1e-12)
    np.testing.assert_allclose(world[..., 3], math.pi / 2.0, atol=1e-12)
    np.testing.assert_array_equal(
        world[..., (0, 4, 5)], trajectories[..., (0, 4, 5)]
    )


def test_index_is_sole_inventory_and_sealed_test_files_are_not_opened(tmp_path):
    sample_root, maps_root, _, entry, index_path, payload = development_fixture(tmp_path)
    sealed_sample = sample_root / TEST_UUID / "must-not-open.npz"
    sealed_sample.parent.mkdir()
    sealed_sample.write_bytes(b"not an npz")
    sealed_map = maps_root / ("dep_car_map_9999_" + TEST_UUID[:8])
    sealed_map.mkdir()
    (sealed_map / "manifest.json").write_text("not json", encoding="utf-8")
    (sealed_map / "map.yaml").write_text("not: [valid", encoding="utf-8")
    (sealed_map / "map.png").write_bytes(b"not a png")

    authority = audit.load_index_authority(
        index_path,
        sample_root,
        maps_root,
        expected_index_sha256=None,
        expected_content_aggregate=payload["content_aggregate_sha256"],
        expected_split_counts={"train": 1},
    )
    specs, contract = audit.load_map_specs(
        maps_root, {entry["map_uuid"]}, expected_aggregate=None
    )

    assert [row["path"] for row in authority.entries] == [entry["path"]]
    assert set(specs) == {TRAIN_UUID}
    assert contract["schema"] == "IndexedMapContractAggregateV1"
    assert contract["map_count"] == 1
    assert contract["aggregate_sha256"] == audit.canonical_hash(contract["rows"])


def test_rotated_ros_map_origin_is_rejected_until_world_to_cell_supports_it(tmp_path):
    sample_root, maps_root, _, entry, _, _ = development_fixture(tmp_path)
    del sample_root
    folder = next(path for path in maps_root.iterdir() if path.is_dir())
    yaml_path = folder / "map.yaml"
    yaml_path.write_text(
        yaml_path.read_text(encoding="utf-8").replace(
            "origin: [-5.0, -5.0, 0.0]",
            "origin: [-5.0, -5.0, 0.1]",
        ),
        encoding="utf-8",
    )

    with pytest.raises(audit.DevelopmentReauditError, match="map contract is invalid"):
        audit.load_map_specs(
            maps_root, {entry["map_uuid"]}, expected_aggregate=None
        )


def test_indexed_sample_content_tamper_fails_before_npz_load(tmp_path):
    sample_root, maps_root, sample_path, entry, _, _ = development_fixture(tmp_path)
    specs, _ = audit.load_map_specs(
        maps_root, {entry["map_uuid"]}, expected_aggregate=None
    )
    audit.initialize_worker(specs)
    with sample_path.open("ab") as stream:
        stream.write(b"tampered")

    row = audit.inspect_indexed_sample(entry, sample_root)

    assert "failure" in row
    assert "content SHA-256 mismatch" in row["failure"]


def test_regeneration_uses_gear_aligned_reverse_acceleration_limit():
    state = np.zeros(9, dtype=np.float64)
    state[0] = 0.0

    trajectories = audit.regenerate_candidate_bank(state, -1, -1)

    dt = trajectories[0, 1, 0] - trajectories[0, 0, 0]
    signed_dv = trajectories[0, 1, 4] - trajectories[0, 0, 4]
    assert trajectories.shape == (15, 11, 6)
    assert signed_dv / dt == pytest.approx(-1.5)
    assert trajectories[0, 1, 4] == pytest.approx(-0.15)


def test_opposite_gear_regeneration_zeros_recorded_speed_first():
    state = np.zeros(9, dtype=np.float64)
    state[0] = 1.7

    trajectories = audit.regenerate_candidate_bank(state, 1, -1)

    assert np.all(trajectories[:, 0, 4] == 0.0)
    assert trajectories[0, 1, 4] == pytest.approx(-0.15)


@pytest.mark.parametrize(
    "raw_speed,requested_gear,expected",
    ((0.4, 1, 0.4), (-0.3, -1, -0.3), (-0.3, 1, 0.0), (0.4, -1, 0.0)),
)
def test_neutral_projects_speed_to_requested_gear_like_p4_rollout(
    raw_speed, requested_gear, expected
):
    state = np.zeros(9, dtype=np.float64)
    state[0] = raw_speed

    trajectories = audit.regenerate_candidate_bank(state, 0, requested_gear)

    assert np.allclose(trajectories[:, 0, 4], expected)


def test_reaudit_always_selects_frozen_training_one_diagonal_policy():
    class RecordingGrid:
        def __init__(self):
            self.policies = []

        def swept_footprint_clearance(
            self, trajectory, footprint, allowance_policy
        ):
            self.policies.append(allowance_policy)
            return True, 1.0

    grid = RecordingGrid()
    feasible, clearance = audit.reaudit_candidate_bank(
        candidate_bank(points=3), np.eye(4), grid, FootprintConfig()
    )

    assert np.all(feasible)
    assert np.all(clearance == 1.0)
    assert grid.policies == [TRAINING_FOOTPRINT_ALLOWANCE] * 15
    contract = audit.footprint_contract()
    assert contract["allowance_policy"] == "training_one_diagonal"
    assert contract["substeps_per_original_segment"] == 16
    assert contract["schema"] == audit.FIVE_CIRCLE_FOOTPRINT_SCHEMA


def test_exact_evaluator_has_bitwise_production_loss_semantics():
    known_free = np.ones((120, 120), dtype=bool)
    known_free[60, 60] = False
    resolution = 0.05
    origin = (-3.0, -3.0)
    sdf = audit._signed_distance_field(known_free, resolution)
    trajectories = candidate_bank(points=3)
    trajectories[..., 1] -= 1.5
    transform = np.eye(4, dtype=np.float32)

    feasible, clearance = audit.p5_exact_candidate_bank(
        trajectories, transform, sdf, resolution, origin
    )
    with torch.no_grad():
        production = audit.swept_map_footprint_clearance(
            torch.from_numpy(trajectories.astype(np.float32))[None],
            torch.from_numpy(sdf.astype(np.float32))[None, None],
            torch.tensor([resolution], dtype=torch.float32),
            torch.tensor([origin], dtype=torch.float32),
            torch.from_numpy(transform)[None],
        ).amin(dim=(-1, -2))[0]

    np.testing.assert_array_equal(feasible, production.numpy() > 0.0)
    np.testing.assert_array_equal(
        clearance.astype(np.float32), production.clamp_min(0.0).numpy()
    )


def test_continuous_sweep_detects_obstacle_between_original_rows():
    data = np.zeros((120, 120), dtype=np.int16)
    resolution = 0.05
    origin = (-3.0, -3.0)
    obstacle = np.floor((np.asarray((0.0, 0.0)) - origin) / resolution).astype(int)
    data[obstacle[1], obstacle[0]] = 100
    grid = OccupancyGrid2D(data, resolution, origin)
    trajectories = np.zeros((15, 2, 6), dtype=np.float64)
    trajectories[:, 0, 1] = -0.8
    trajectories[:, 1, 0] = 0.1
    trajectories[:, 1, 1] = 0.8

    for endpoint in (trajectories[0, :1], trajectories[0, 1:]):
        endpoint_ok, _ = grid.swept_footprint_clearance(
            endpoint,
            allowance_policy=TRAINING_FOOTPRINT_ALLOWANCE,
        )
        assert endpoint_ok
    feasible, clearance = audit.reaudit_candidate_bank(
        trajectories, np.eye(4), grid
    )
    known_free = data == 0
    sdf = audit._signed_distance_field(known_free, resolution)
    exact_feasible, exact_clearance = audit.p5_exact_candidate_bank(
        trajectories, np.eye(4), sdf, resolution, origin
    )

    assert not np.any(feasible)
    assert np.all(clearance == 0.0)
    assert not np.any(exact_feasible)
    assert np.all(exact_clearance == 0.0)


def test_old_nan_trajectories_are_not_candidate_authority(tmp_path):
    sample_root, maps_root, _, entry, _, _ = development_fixture(tmp_path)
    specs, _ = audit.load_map_specs(
        maps_root, {entry["map_uuid"]}, expected_aggregate=None
    )
    audit.initialize_worker(specs)

    row = audit.inspect_indexed_sample(entry, sample_root)

    assert "failure" not in row
    assert row["new_feasible_count"] == 15
    assert row["legacy_available"] is True


def test_maximum_samples_can_never_produce_pass(tmp_path):
    sample_root, maps_root, _, _, index_path, _ = development_fixture(tmp_path)

    report = audit.run_audit(
        index_path,
        sample_root,
        maps_root,
        maximum_samples=1,
        workers=1,
        enforce_frozen_authority=False,
    )

    assert report["status"] == "SMOKE"
    assert report["qualification_eligible"] is False
    assert report["sample_files_audited"] == 1
    assert "diagnostic_partial_or_nonfrozen_audit" in report["errors"]
    assert report["training_authority"] == {
        "schema": "P3TrainingAuthorityV1",
        "index_sha256": audit.file_sha256(index_path),
        "content_aggregate_sha256": report["development_authority"][
            "content_aggregate_sha256"
        ],
        "map_contract_aggregate_sha256": report["development_authority"][
            "map_contract_aggregate_sha256"
        ],
        "splits": ["train", "validation"],
        "test_split_used": False,
    }
    assert report["P3_provenance"]["task_manifest_internal_hash_verified"] is True
    assert report["audit_implementation"][
        "p4_implementation_aggregate_sha256"
    ] == audit.build_p4_implementation_contract(ROOT)["aggregate_sha256"]


def test_p3_gate_boundaries_are_frozen_and_strict():
    overall = {
        "new": {
            "zero_feasible_rate": audit.OVERALL_ZERO_FEASIBLE_LIMIT,
            "feasible_candidates_median": audit.MINIMUM_MEDIAN_FEASIBLE,
        }
    }
    modes = {
        mode: {"new": {"zero_feasible_rate": 0.0}}
        for mode in audit.PILOT_MANEUVER_MODES
    }
    modes[audit.PILOT_MANEUVER_MODES[0]]["new"]["zero_feasible_rate"] = (
        audit.PER_MODE_ZERO_FEASIBLE_LIMIT
    )

    checks, failures = audit.evaluate_gates(overall, modes)

    assert checks["overall_zero_feasible_rate_lt_0_10"]["status"] == "FAIL"
    assert checks["overall_median_feasible_candidates_ge_2"]["status"] == "PASS"
    assert (
        checks["per_mode_zero_feasible_rate_lt_0_25"]
        [audit.PILOT_MANEUVER_MODES[0]]["status"]
        == "FAIL"
    )
    assert "overall_zero_feasible_rate_lt_0_10" in failures


@pytest.mark.parametrize(
    "arguments",
    (
        ["--samples", "/tmp/samples"],
        ["--maps", "/tmp/maps"],
        ["--index", "/tmp/index.json"],
        ["--workers", "1"],
        ["--safety-margin", "0.0"],
    ),
)
def test_dataset_geometry_gate_and_worker_overrides_are_not_exposed(arguments):
    with pytest.raises(SystemExit):
        audit.parse_args(arguments)
