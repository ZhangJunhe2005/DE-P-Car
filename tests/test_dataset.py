import numpy as np

from dep_car.core.lattice import AckermannLattice
from dep_car.core.types import Gear, VehicleState
from dep_car.training.dataset import audit_multimodal_sample, map_split, save_multimodal_sample


def test_map_split_is_stable_and_grouped():
    assert map_split("same-map") == map_split("same-map")
    assert map_split("same-map") in {"train", "validation", "test"}


def test_multimodal_v2_round_trip_and_skew_audit(tmp_path):
    path = tmp_path / "map" / "sample.npz"
    candidates = AckermannLattice().generate(VehicleState(), gear=Gear.REVERSE)
    manifest = save_multimodal_sample(
        path,
        map_uuid="map-v2",
        depth_metric=np.full((48, 64), 3.0, dtype=np.float32),
        depth_validity=np.ones((48, 64), dtype=np.uint8),
        lidar_points=np.asarray([[1.0, 0.0, 0.2, 4.0, 0.0]], dtype=np.float32),
        lidar_bev=np.zeros((6, 160, 160), dtype=np.float32),
        imu_measurement=np.asarray([0, 0, 0, 1, 0, 0, 0, 0, 0, 9.81], dtype=np.float32),
        vehicle_state=np.zeros(9, dtype=np.float32),
        current_gear=Gear.NEUTRAL,
        requested_gear=Gear.REVERSE,
        local_path=np.asarray([[0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], dtype=np.float32),
        local_path_gears=np.asarray([0, -1], dtype=np.int8),
        subgoal_body=(-1.0, 0.0),
        candidates=candidates,
        timestamps={"lidar": 10.0, "depth": 10.03, "odom": 10.01, "joint_state": 10.02, "candidates": 10.05},
        transforms={
            name: {
                "matrix": np.eye(4).tolist(), "measurement_stamp": 10.0,
                "source_frame": source, "target_frame": target,
            }
            for name, source, target in (
                ("lidar_to_chassis", "velodyne", "chassis"),
                ("camera_to_chassis", "camera_depth_optical_frame", "chassis"),
                ("chassis_to_map", "chassis", "map"),
            )
        },
        raw_authority={"kind": "embedded_diagnostic"},
        preprocessing={"lidar_bev": {"sha256": "0" * 64}},
        interpolation={
            name: {"method": "approximate_time_nearest", "target_stamp": 10.0, "source_stamps": [10.0]}
            for name in ("odom", "joint_state", "imu")
        },
        lidar_timing={
            "model": "gazebo_ray_instantaneous_snapshot", "scan_period_s": 0.1,
            "per_point_time_available": False, "deskew_applied": False,
        },
    )
    assert manifest["schema"] == "StaticAckermannSampleV2"
    assert manifest["contract_revision"] == 2
    assert audit_multimodal_sample(path) == []
