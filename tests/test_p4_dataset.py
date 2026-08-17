import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
import yaml

import dep_car.training.p4_dataset as p4_dataset_module
from dep_car.core.types import Candidate, Gear
from dep_car.core.occupancy import FootprintConfig
from dep_car.core.vehicle import STEERING_OPERATING_LIMIT_RAD
from dep_car.model.dep_car_net import DEPCarNetV1
from dep_car.perception.bev import LidarBEVConfig, lidar_bev_preprocessing_contract
from dep_car.training.dataset import map_split, save_multimodal_sample
from dep_car.training.p4_dataset import (
    P3TrainingDataError,
    P3TrainingDatasetV1,
    TRAINING_INDEX_SCHEMA,
    _load_map_catalog,
    build_training_index,
    indexed_map_contract_aggregate,
    load_training_index,
    p3_training_collate,
    training_index_content_aggregate,
)
from dep_car.training.losses import swept_map_footprint_clearance


def uuid_for_split(target):
    for index in range(10000):
        value = "p4-synthetic-map-%04d" % index
        if map_split(value) == target:
            return value
    raise AssertionError("unable to find a deterministic synthetic map UUID")


def make_map(root, map_uuid, *, occupied_landmark=None, unknown_landmark=None):
    folder = root / ("map_" + map_uuid)
    folder.mkdir(parents=True)
    image = np.full((21, 21), 255, dtype=np.uint8)
    image[[0, -1], :] = 0
    image[:, [0, -1]] = 0
    if occupied_landmark is not None:
        image[occupied_landmark] = 0
    if unknown_landmark is not None:
        image[unknown_landmark] = 128
    assert cv2.imwrite(str(folder / "map.png"), image)
    occupancy_hash = hashlib.sha256(image.tobytes()).hexdigest()
    (folder / "manifest.json").write_text(json.dumps({
        "schema": "DEPCarArenaStaticMapV1",
        "map_uuid": map_uuid,
        "name": folder.name,
        "occupancy_sha256": occupancy_hash,
    }), encoding="utf-8")
    (folder / "map.yaml").write_text(yaml.safe_dump({
        "image": "map.png",
        "resolution": 0.1,
        "origin": [-1.05, -1.05, 0.0],
        "negate": 0,
        "occupied_thresh": 0.65,
        "free_thresh": 0.196,
    }), encoding="utf-8")
    return occupancy_hash


def make_candidates(gear, steps):
    speeds = (0.2, 0.35, 0.5) if gear == Gear.REVERSE else (0.6, 1.2, 2.0)
    steering = (-0.52, -0.26, 0.0, 0.26, 0.52)
    duration = 0.1 * (steps - 1)
    output = []
    candidate_id = 0
    for magnitude in speeds:
        speed = float(gear) * magnitude
        for delta in steering:
            time_axis = np.linspace(0.0, duration, steps, dtype=np.float64)
            trajectory = np.zeros((steps, 6), dtype=np.float64)
            trajectory[:, 0] = time_axis
            trajectory[:, 1] = speed * time_axis
            trajectory[:, 3] = 0.2 * delta * time_axis
            trajectory[:, 4] = speed
            trajectory[:, 5] = delta
            output.append(Candidate(
                candidate_id,
                speed,
                delta,
                duration,
                trajectory,
                gear=gear,
                feasible=candidate_id != 14,
                static_clearance=0.4 if candidate_id != 14 else 0.0,
                guidance_cost=0.1 * candidate_id,
            ))
            candidate_id += 1
    return output


def save_sample(
    path,
    map_uuid,
    occupancy_hash,
    *,
    short_route=False,
    steps=13,
    chassis_to_map=None,
    longitudinal_acceleration=-0.4,
    requested_gear=Gear.REVERSE,
):
    preprocessing = lidar_bev_preprocessing_contract(LidarBEVConfig())
    bev = np.zeros((6, 160, 160), dtype=np.float32)
    bev[5] = 1.0
    bev[0, 80, 90] = 1.0
    bev[1, 80, 90] = 0.5
    bev[2, 80, 90] = 0.05
    bev[3, 80, 90] = 1.30
    bev[4, 80, 90] = 0.5
    if short_route:
        local_path = np.asarray([[0.1, 0.0, 0.0]], dtype=np.float32)
        local_gears = np.asarray([1], dtype=np.int8)
        current_gear = Gear.NEUTRAL
        steering = 0.0
    else:
        local_path = np.asarray([
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [-0.1, 0.0, 1.5],
            [-0.2, 0.0, 3.0],
            [0.1, 0.0, 0.0],
        ], dtype=np.float32)
        local_gears = np.asarray([0, 1, -1, -1, -1, 1], dtype=np.int8)
        current_gear = Gear.FORWARD
        steering = 0.9
    state = np.asarray([
        0.2, longitudinal_acceleration, steering, 0.1,
        -0.6, 0.0, 0.0, 1.0, 8.0,
    ], dtype=np.float32)
    chassis_to_map = (
        np.eye(4, dtype=np.float32)
        if chassis_to_map is None
        else np.asarray(chassis_to_map, dtype=np.float32)
    )
    save_multimodal_sample(
        path,
        map_uuid=map_uuid,
        depth_metric=np.full((480, 640), 5.0, dtype=np.float32),
        depth_validity=np.ones((480, 640), dtype=np.uint8),
        lidar_points=None,
        lidar_bev=bev,
        imu_measurement=np.asarray([0, 0, 0, 1, 0, 0, 0.1, -0.4, 0, 9.81], dtype=np.float32),
        vehicle_state=state,
        current_gear=current_gear,
        requested_gear=requested_gear,
        local_path=local_path,
        local_path_gears=local_gears,
        subgoal_body=(-0.6, 0.0),
        candidates=make_candidates(requested_gear, steps),
        timestamps={"lidar": 1.0, "depth": 1.01, "odom": 1.0, "joint_state": 1.0, "imu": 1.0, "candidates": 1.02},
        transforms={
            name: {
                "matrix": (
                    chassis_to_map.tolist()
                    if name == "chassis_to_map"
                    else np.eye(4).tolist()
                ),
                "measurement_stamp": 1.0,
                "source_frame": source,
                "target_frame": target,
            }
            for name, source, target in (
                ("lidar_to_chassis", "velodyne", "chassis"),
                ("camera_to_chassis", "camera_depth_optical_frame", "chassis"),
                ("chassis_to_map", "chassis", "map"),
            )
        },
        raw_authority={
            "kind": "rosbag_reference",
            "bag_path": "synthetic.bag",
            "bag_sha256": "0" * 64,
            "messages": {},
        },
        preprocessing={"lidar_bev": preprocessing},
        interpolation={},
        lidar_timing={
            "model": "gazebo_ray_instantaneous_snapshot",
            "scan_period_s": 0.1,
            "per_point_time_available": False,
            "deskew_applied": False,
        },
        metadata={
            "sample_id": path.stem,
            "pilot_task_id": "synthetic-task",
            "formal_training_authority": True,
            "map_occupancy_sha256": occupancy_hash,
            "candidate_context": "RECOVERY",
        },
        maneuver_mode="THREE_POINT_TURN",
    )


@pytest.fixture()
def synthetic_p3(tmp_path):
    sample_root = tmp_path / "samples"
    maps_root = tmp_path / "maps"
    map_uuid = uuid_for_split("train")
    occupancy_hash = make_map(maps_root, map_uuid)
    folder = sample_root / map_uuid
    folder.mkdir(parents=True)
    save_sample(folder / "sample-000.npz", map_uuid, occupancy_hash, steps=13)
    save_sample(folder / "sample-001.npz", map_uuid, occupancy_hash, short_route=True, steps=11)
    # A deliberately unreadable sealed sample proves that a train-only index
    # chooses folders by map UUID before opening any test NPZ.
    test_uuid = uuid_for_split("test")
    make_map(maps_root, test_uuid)
    sealed = sample_root / test_uuid
    sealed.mkdir(parents=True)
    (sealed / "must-not-be-opened.npz").write_bytes(b"sealed-test-authority")
    return sample_root, maps_root, map_uuid


def test_index_is_reusable_and_map_split_authoritative(synthetic_p3, tmp_path):
    sample_root, maps_root, map_uuid = synthetic_p3
    index_path = tmp_path / "training_index.json"
    payload = build_training_index(
        sample_root, maps_root, index_path, splits=("train",), workers=8
    )
    assert payload["samples"] == 2
    assert payload["schema"] == TRAINING_INDEX_SCHEMA == "P3TrainingIndexV2"
    assert payload["workers"] == 8
    assert payload["content_hash_algorithm"] == "sha256"
    assert payload["content_aggregate_sha256"] == training_index_content_aggregate(
        payload["entries"]
    )
    assert all(len(entry["content_sha256"]) == 64 for entry in payload["entries"])
    assert payload["counts_by_split"] == {"train": 2}
    assert {entry["map_uuid"] for entry in payload["entries"]} == {map_uuid}
    before = index_path.stat().st_mtime_ns
    loaded = load_training_index(
        index_path,
        sample_root=sample_root,
        maps_root=maps_root,
        splits=("train",),
    )
    assert loaded == payload
    assert index_path.stat().st_mtime_ns == before


def test_index_content_aggregate_is_stable_across_rebuild_metadata(
    synthetic_p3, tmp_path
):
    sample_root, maps_root, _ = synthetic_p3
    first = build_training_index(
        sample_root,
        maps_root,
        tmp_path / "first.json",
        splits=("train",),
        workers=8,
    )
    second = build_training_index(
        sample_root,
        maps_root,
        tmp_path / "second.json",
        splits=("train",),
        workers=8,
    )
    assert first["content_aggregate_sha256"] == second["content_aggregate_sha256"]
    reversed_entries = list(reversed(first["entries"]))
    assert training_index_content_aggregate(reversed_entries) == first[
        "content_aggregate_sha256"
    ]


def test_index_load_hashes_bytes_and_rejects_same_size_restored_mtime_tamper(
    synthetic_p3, tmp_path
):
    sample_root, maps_root, _ = synthetic_p3
    index_path = tmp_path / "training_index.json"
    payload = build_training_index(
        sample_root, maps_root, index_path, splits=("train",), workers=8
    )
    # The first load performs an actual byte hash and seeds only a process-local
    # verified cache.  A later edit that restores size+mtime must miss that cache
    # because inode/ctime/sample-stat authority is part of the cache key.
    load_training_index(
        index_path,
        sample_root=sample_root,
        maps_root=maps_root,
        splits=("train",),
        workers=8,
    )
    sample_path = sample_root / payload["entries"][0]["path"]
    original_stat = sample_path.stat()
    with sample_path.open("r+b") as stream:
        stream.seek(-1, os.SEEK_END)
        original = stream.read(1)
        stream.seek(-1, os.SEEK_END)
        stream.write(bytes((original[0] ^ 0x01,)))
    os.utime(
        sample_path,
        ns=(int(original_stat.st_atime_ns), int(original_stat.st_mtime_ns)),
    )
    assert sample_path.stat().st_size == original_stat.st_size
    assert sample_path.stat().st_mtime_ns == original_stat.st_mtime_ns
    with pytest.raises(P3TrainingDataError, match="content hash mismatch"):
        load_training_index(
            index_path,
            sample_root=sample_root,
            maps_root=maps_root,
            splits=("train",),
            workers=8,
        )


def test_index_load_reuses_only_a_previously_verified_in_process_cache(
    synthetic_p3, tmp_path, monkeypatch
):
    sample_root, maps_root, _ = synthetic_p3
    index_path = tmp_path / "training_index.json"
    build_training_index(
        sample_root, maps_root, index_path, splits=("train",), workers=8
    )
    load_training_index(
        index_path,
        sample_root=sample_root,
        maps_root=maps_root,
        splits=("train",),
        workers=8,
    )

    def unexpected_rehash(_path):
        raise AssertionError("unchanged, already-verified content was rehashed")

    monkeypatch.setattr(p4_dataset_module, "_sha256_file", unexpected_rehash)
    load_training_index(
        index_path,
        sample_root=sample_root,
        maps_root=maps_root,
        splits=("train",),
        workers=8,
    )


def test_training_view_shapes_normalization_and_authority(synthetic_p3, tmp_path):
    sample_root, maps_root, _ = synthetic_p3
    dataset = P3TrainingDatasetV1(
        sample_root,
        maps_root,
        split="train",
        index_path=tmp_path / "training_index.json",
        index_splits=("train",),
        workers=2,
    )
    sample = dataset[0]
    assert sample["depth"].shape == (2, 96, 160)
    assert torch.allclose(sample["depth"][0], torch.full((96, 160), 0.5))
    assert torch.all(sample["depth"][1] == 1.0)
    assert sample["lidar_bev"].shape == (6, 160, 160)
    assert float(sample["lidar_bev"].min()) >= 0.0
    assert float(sample["lidar_bev"].max()) <= 1.0
    assert sample["modality_mask"].tolist() == [1.0, 1.0]
    assert sample["route"].shape == (80, 5)
    assert int(sample["route_mask"].sum()) == 3
    assert torch.all(sample["route_gears"][sample["route_mask"]] == -1)
    assert sample["steering_clamped"].item() is True
    assert sample["geometry_valid"].item() is False
    assert sample["shift_speed_zeroed"].item() is True
    assert sample["state"][0].item() == 0.0
    assert sample["state"][1].item() == 0.0
    assert sample["vehicle_state_raw"][0].item() == pytest.approx(0.2)
    assert sample["state"][2].item() == pytest.approx(STEERING_OPERATING_LIMIT_RAD)
    assert sample["state"][8].item() == pytest.approx(1.0 / 0.55)
    assert sample["state_normalized"][2].item() == pytest.approx(1.0)
    assert sample["state_normalized"][8].item() == pytest.approx(1.0)
    torch.testing.assert_close(
        sample["state_normalized"],
        DEPCarNetV1.normalize_state(sample["state"]),
        atol=0.0,
        rtol=0.0,
    )
    assert sample["candidate_pose"].shape == (15, 15, 4)
    assert sample["candidate_time_mask"].shape == (15, 15)
    assert int(sample["candidate_time_mask"].sum()) == 15 * 13
    assert torch.all(sample["candidate_pose"][:, 13:] == 0.0)
    assert sample["bev_distance_field"].shape == (1, 160, 160)
    assert sample["bev_distance_field"][0, 80, 90].item() < 0.0
    assert sample["bev_distance_field"][0, 80, 80].item() > 0.0
    assert sample["map_distance_field"].shape == (1, 21, 21)
    assert sample["map_occupancy"].dtype == torch.bool
    assert sample["metadata"]["candidate_context"] == "RECOVERY"


@pytest.mark.parametrize(
    "requested_gear,raw_acceleration,expected",
    (
        (Gear.FORWARD, 3.0, 1.5),
        (Gear.FORWARD, -3.0, -2.0),
        (Gear.REVERSE, -3.0, -1.5),
        (Gear.REVERSE, 3.0, 2.0),
    ),
)
def test_training_state_clips_acceleration_in_requested_gear_frame(
    tmp_path, requested_gear, raw_acceleration, expected
):
    sample_root = tmp_path / "samples"
    maps_root = tmp_path / "maps"
    map_uuid = uuid_for_split("train")
    occupancy_hash = make_map(maps_root, map_uuid)
    folder = sample_root / map_uuid
    folder.mkdir(parents=True)
    save_sample(
        folder / "acceleration.npz",
        map_uuid,
        occupancy_hash,
        short_route=True,
        steps=11,
        longitudinal_acceleration=raw_acceleration,
        requested_gear=requested_gear,
    )
    dataset = P3TrainingDatasetV1(
        sample_root,
        maps_root,
        split="train",
        index_path=tmp_path / "training_index.json",
        index_splits=("train",),
        workers=1,
    )

    sample = dataset[0]
    assert sample["geometry_valid"].item() is True
    assert sample["state"][1].item() == pytest.approx(expected)


def test_dataset_getitem_rechecks_exact_indexed_bytes_after_initial_validation(
    synthetic_p3, tmp_path
):
    sample_root, maps_root, _ = synthetic_p3
    dataset = P3TrainingDatasetV1(
        sample_root,
        maps_root,
        split="train",
        index_path=tmp_path / "training_index.json",
        index_splits=("train",),
        workers=2,
    )
    sample_path = sample_root / dataset.entries[0]["path"]
    original_stat = sample_path.stat()
    with sample_path.open("r+b") as stream:
        stream.seek(-1, os.SEEK_END)
        value = stream.read(1)
        stream.seek(-1, os.SEEK_END)
        stream.write(bytes((value[0] ^ 0x01,)))
    os.utime(
        sample_path,
        ns=(int(original_stat.st_atime_ns), int(original_stat.st_mtime_ns)),
    )

    with pytest.raises(P3TrainingDataError, match="content changed"):
        dataset[0]


def test_dataset_rejects_map_contract_changed_after_trainer_snapshot(
    synthetic_p3, tmp_path
):
    sample_root, maps_root, _ = synthetic_p3
    index_path = tmp_path / "training_index.json"
    build_training_index(
        sample_root, maps_root, index_path, splits=("train",), workers=2
    )
    expected = indexed_map_contract_aggregate(
        _load_map_catalog(maps_root, {uuid_for_split("train")})
    )["aggregate_sha256"]
    image_path = next(
        folder / "map.png"
        for folder in maps_root.iterdir()
        if uuid_for_split("train") in folder.name
    )
    # Appending an ignored trailing byte preserves decoded occupancy but changes
    # the raw PNG identity that the trainer sealed into its plan.
    image_path.write_bytes(image_path.read_bytes() + b"\x00")
    with pytest.raises(P3TrainingDataError, match="map contract changed"):
        P3TrainingDatasetV1(
            sample_root,
            maps_root,
            split="train",
            index_path=index_path,
            index_splits=("train",),
            workers=2,
            expected_map_contract_aggregate_sha256=expected,
        )


def test_dataset_rejects_index_metadata_changed_after_trainer_snapshot(
    synthetic_p3, tmp_path
):
    sample_root, maps_root, _ = synthetic_p3
    index_path = tmp_path / "training_index.json"
    build_training_index(
        sample_root, maps_root, index_path, splits=("train",), workers=2
    )
    expected = hashlib.sha256(index_path.read_bytes()).hexdigest()
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    payload["created_at_unix"] = float(payload["created_at_unix"]) + 1.0
    index_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(P3TrainingDataError, match="index changed"):
        P3TrainingDatasetV1(
            sample_root,
            maps_root,
            split="train",
            index_path=index_path,
            index_splits=("train",),
            workers=2,
            expected_index_sha256=expected,
        )


def test_map_tensor_read_rechecks_bound_png_bytes(synthetic_p3, tmp_path):
    sample_root, maps_root, map_uuid = synthetic_p3
    dataset = P3TrainingDatasetV1(
        sample_root,
        maps_root,
        split="train",
        index_path=tmp_path / "training_index.json",
        index_splits=("train",),
        workers=2,
    )
    image_path = dataset.maps[map_uuid].image
    image_path.write_bytes(image_path.read_bytes() + b"\x00")
    with pytest.raises(P3TrainingDataError, match="PNG bytes changed"):
        dataset._map_tensors(map_uuid)


def test_map_tensor_read_rechecks_bound_yaml_semantics(synthetic_p3, tmp_path):
    sample_root, maps_root, map_uuid = synthetic_p3
    dataset = P3TrainingDatasetV1(
        sample_root,
        maps_root,
        split="train",
        index_path=tmp_path / "training_index.json",
        index_splits=("train",),
        workers=2,
    )
    yaml_path = dataset.maps[map_uuid].yaml_path
    metadata = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    metadata["resolution"] = 0.2
    yaml_path.write_text(yaml.safe_dump(metadata), encoding="utf-8")
    with pytest.raises(P3TrainingDataError, match="YAML semantics changed"):
        dataset._map_tensors(map_uuid)


def test_png_lower_left_flip_and_chassis_to_map_reach_asymmetric_landmark(tmp_path):
    sample_root = tmp_path / "samples"
    maps_root = tmp_path / "maps"
    map_uuid = uuid_for_split("train")
    # PNG row 3 becomes ROS lower-left row 17.  Mid-gray is trinary unknown and
    # must be unsafe as well, rather than silently becoming free.
    occupancy_hash = make_map(
        maps_root,
        map_uuid,
        occupied_landmark=(3, 15),
        unknown_landmark=(6, 6),
    )
    folder = sample_root / map_uuid
    folder.mkdir(parents=True)
    chassis_to_map = np.asarray(
        (
            (0.0, -1.0, 0.0, 0.4),
            (1.0, 0.0, 0.0, 0.5),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
        dtype=np.float32,
    )
    save_sample(
        folder / "landmark.npz",
        map_uuid,
        occupancy_hash,
        short_route=True,
        chassis_to_map=chassis_to_map,
    )
    dataset = P3TrainingDatasetV1(
        sample_root,
        maps_root,
        split="train",
        index_path=tmp_path / "landmark_index.json",
        index_splits=("train",),
        workers=1,
    )
    sample = dataset[0]
    occupancy = sample["map_occupancy"]
    signed = sample["map_distance_field"][0]
    assert occupancy[17, 15]
    assert not occupancy[3, 15]
    assert occupancy[14, 6]  # flipped trinary-unknown landmark
    assert signed[17, 15] < 0.0
    assert signed[14, 6] < 0.0

    # R(+90deg) * [0.2,-0.1] + [0.4,0.5] reaches world [0.5,0.7],
    # the centre of the flipped occupied PNG cell.  Body [-1.2,-0.1]
    # reaches the unflipped/wrong world y=-0.7 location and must remain free.
    trajectory = torch.zeros(1, 2, 1, 6)
    trajectory[0, 0, 0, 1:3] = torch.tensor([0.2, -0.1])
    trajectory[0, 1, 0, 1:3] = torch.tensor([-1.2, -0.1])
    tiny_footprint = FootprintConfig(
        length=0.01, width=0.01, safety_margin=0.0, circle_count=1
    )
    clearance = swept_map_footprint_clearance(
        trajectory,
        sample["map_distance_field"][None],
        sample["map_resolution"][None],
        sample["map_origin"][None],
        sample["chassis_to_map"][None],
        footprint=tiny_footprint,
    )
    assert clearance[0, 0, 0, 0] < 0.0
    assert clearance[0, 1, 0, 0] > 0.0


def test_map_yaml_trinary_threshold_contract_fails_closed(tmp_path):
    maps_root = tmp_path / "maps"
    map_uuid = uuid_for_split("train")
    make_map(maps_root, map_uuid)
    yaml_path = next(maps_root.glob("*/map.yaml"))
    metadata = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    metadata["negate"] = 2
    metadata["free_thresh"] = 0.8
    metadata["occupied_thresh"] = 0.2
    yaml_path.write_text(yaml.safe_dump(metadata), encoding="utf-8")
    with pytest.raises(P3TrainingDataError, match="metric metadata"):
        _load_map_catalog(maps_root)


def test_map_yaml_nonzero_origin_yaw_fails_closed(tmp_path):
    maps_root = tmp_path / "maps"
    map_uuid = uuid_for_split("train")
    make_map(maps_root, map_uuid)
    yaml_path = next(maps_root.glob("*/map.yaml"))
    metadata = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    metadata["origin"][2] = 0.01
    yaml_path.write_text(yaml.safe_dump(metadata), encoding="utf-8")
    with pytest.raises(P3TrainingDataError, match="origin yaw must be zero"):
        _load_map_catalog(maps_root)


def test_short_wrong_gear_route_uses_subgoal_fallback_and_collates(synthetic_p3, tmp_path):
    sample_root, maps_root, _ = synthetic_p3
    dataset = P3TrainingDatasetV1(
        sample_root,
        maps_root,
        split="train",
        index_path=tmp_path / "training_index.json",
        index_splits=("train",),
    )
    first, second = dataset[0], dataset[1]
    assert int(second["route_mask"].sum()) == 2
    torch.testing.assert_close(second["route_pose"][0], torch.zeros(3))
    torch.testing.assert_close(second["route_pose"][1, :2], torch.tensor([-0.6, 0.0]))
    assert second["geometry_valid"].item() is True
    batch = p3_training_collate([first, second])
    assert batch["depth"].shape == (2, 2, 96, 160)
    assert batch["candidate_pose"].shape == (2, 15, 15, 4)
    assert len(batch["metadata"]) == 2


def test_modality_dropout_is_mutually_exclusive_and_test_is_sealed(synthetic_p3, tmp_path):
    sample_root, maps_root, _ = synthetic_p3
    common = dict(
        sample_root=sample_root,
        maps_root=maps_root,
        split="train",
        index_path=tmp_path / "training_index.json",
        index_splits=("train",),
    )
    depth_missing = P3TrainingDatasetV1(
        **common, depth_dropout_probability=1.0
    )[0]
    assert depth_missing["modality_mask"].tolist() == [0.0, 1.0]
    assert torch.all(depth_missing["depth"] == 0.0)
    lidar_missing = P3TrainingDatasetV1(
        **common, lidar_dropout_probability=1.0
    )[0]
    assert lidar_missing["modality_mask"].tolist() == [1.0, 0.0]
    assert torch.all(lidar_missing["lidar_bev"] == 0.0)
    with pytest.raises(P3TrainingDataError, match="sum to <= 1"):
        P3TrainingDatasetV1(
            **common,
            depth_dropout_probability=0.6,
            lidar_dropout_probability=0.6,
        )
    with pytest.raises(P3TrainingDataError, match="test split is sealed"):
        P3TrainingDatasetV1(
            sample_root,
            maps_root,
            split="test",
            index_path=tmp_path / "test_index.json",
        )
