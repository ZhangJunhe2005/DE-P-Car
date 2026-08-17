"""Versioned, map-group-safe static sample persistence."""

import hashlib
import json
from pathlib import Path

import numpy as np


SCHEMA_VERSION = "StaticAckermannSampleV1"
MULTIMODAL_SCHEMA_VERSION = "StaticAckermannSampleV2"
MULTIMODAL_CONTRACT_REVISION = 2
MANEUVER_MODES = {
    "NORMAL", "SHARP_TURN", "NARROW_CORRIDOR", "U_TURN",
    "DEAD_END_ESCAPE", "REVERSE_EXIT", "THREE_POINT_TURN",
    "GEAR_SHIFT", "RECOVERY",
}


def map_split(map_uuid, train=80, validation=10):
    bucket = int(hashlib.sha256(map_uuid.encode("utf-8")).hexdigest()[:8], 16) % 100
    if bucket < train:
        return "train"
    if bucket < train + validation:
        return "validation"
    return "test"


def save_sample(path, *, map_uuid, range_image, validity_mask, vehicle_state, subgoal_body, candidates, metadata=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    candidate_array = np.stack([candidate.trajectory for candidate in candidates])
    manifest = {
        "schema": SCHEMA_VERSION,
        "map_uuid": map_uuid,
        "split": map_split(map_uuid),
        "metadata": metadata or {},
    }
    np.savez_compressed(
        str(path),
        range_image=np.asarray(range_image, dtype=np.float32),
        validity_mask=np.asarray(validity_mask, dtype=np.float32),
        vehicle_state=np.asarray(vehicle_state, dtype=np.float32),
        subgoal_body=np.asarray(subgoal_body, dtype=np.float32),
        trajectories=candidate_array.astype(np.float32),
        feasible=np.asarray([candidate.feasible for candidate in candidates], dtype=np.uint8),
        static_clearance=np.asarray([candidate.static_clearance for candidate in candidates], dtype=np.float32),
        guidance_cost=np.asarray([candidate.guidance_cost for candidate in candidates], dtype=np.float32),
        manifest_json=np.asarray(json.dumps(manifest, sort_keys=True)),
    )
    return manifest


def _candidate_arrays(candidates):
    candidates = tuple(candidates)
    if not candidates:
        raise ValueError("a multimodal sample requires candidate authority")
    trajectories = [np.asarray(candidate.trajectory, dtype=np.float32) for candidate in candidates]
    if len({trajectory.shape for trajectory in trajectories}) != 1:
        raise ValueError("all candidate trajectories must share one shape")
    gears = np.asarray([int(candidate.gear) for candidate in candidates], dtype=np.int8)
    if any(gear not in (-1, 1) for gear in gears) or len(set(gears.tolist())) != 1:
        raise ValueError("one drive gear must remain constant across the candidate bank")
    return {
        "trajectories": np.stack(trajectories),
        "candidate_gear": gears,
        "feasible": np.asarray([candidate.feasible for candidate in candidates], dtype=np.uint8),
        "static_clearance": np.asarray([candidate.static_clearance for candidate in candidates], dtype=np.float32),
        "guidance_cost": np.asarray([candidate.guidance_cost for candidate in candidates], dtype=np.float32),
    }


def save_multimodal_sample(
    path,
    *,
    map_uuid,
    depth_metric,
    depth_validity,
    lidar_points,
    lidar_bev,
    imu_measurement,
    vehicle_state,
    current_gear,
    requested_gear,
    local_path,
    local_path_gears,
    subgoal_body,
    candidates,
    timestamps,
    transforms,
    raw_authority,
    preprocessing,
    interpolation,
    lidar_timing,
    metadata=None,
    maneuver_mode="NORMAL",
):
    """Persist one revision-2 depth/LiDAR/IMU Ackermann training sample.

    Formal samples normally keep raw LiDAR in the referenced rosbag and store
    only derived BEV here.  Diagnostic online captures may embed raw points.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    depth_metric = np.asarray(depth_metric, dtype=np.float32)
    depth_validity = np.asarray(depth_validity, dtype=np.uint8)
    lidar_points = np.empty((0, 5), dtype=np.float32) if lidar_points is None else np.asarray(lidar_points, dtype=np.float32)
    lidar_bev = np.asarray(lidar_bev, dtype=np.float32)
    imu_measurement = np.asarray(imu_measurement, dtype=np.float32)
    vehicle_state = np.asarray(vehicle_state, dtype=np.float32)
    local_path = np.asarray(local_path, dtype=np.float32)
    local_path_gears = np.asarray(local_path_gears, dtype=np.int8)
    if depth_metric.ndim != 2 or depth_validity.shape != depth_metric.shape:
        raise ValueError("depth metric and validity must share a 2-D shape")
    if lidar_points.ndim != 2 or lidar_points.shape[1] < 3:
        raise ValueError("raw LiDAR must have shape [N,>=3]")
    if lidar_bev.ndim != 3 or lidar_bev.shape[0] != 6:
        raise ValueError("LiDAR BEV must have shape [6,H,W]")
    if imu_measurement.shape != (10,):
        raise ValueError("IMU measurement must be [qx,qy,qz,qw,wx,wy,wz,ax,ay,az]")
    if vehicle_state.shape != (9,):
        raise ValueError("DE-P-Car V2 vehicle/route state must contain nine continuous values")
    if local_path.ndim != 2 or local_path.shape[1] != 3 or local_path_gears.shape != (len(local_path),):
        raise ValueError("local path must be [N,3] with one gear per pose")
    if int(requested_gear) not in (-1, 1):
        raise ValueError("requested gear must be forward or reverse")
    if int(current_gear) not in (-1, 0, 1):
        raise ValueError("current gear is invalid")
    if maneuver_mode not in MANEUVER_MODES:
        raise ValueError("unknown maneuver mode: " + str(maneuver_mode))
    timestamps = {str(key): float(value) for key, value in timestamps.items()}
    if "lidar" not in timestamps:
        raise ValueError("LiDAR is the required sample timestamp anchor")
    skews = {key: abs(value - timestamps["lidar"]) for key, value in timestamps.items() if key != "lidar"}
    candidate_arrays = _candidate_arrays(candidates)
    if set(candidate_arrays["candidate_gear"].tolist()) != {int(requested_gear)}:
        raise ValueError("candidate bank gear must equal the supervisor-requested gear")
    manifest = {
        "schema": MULTIMODAL_SCHEMA_VERSION,
        "contract_revision": MULTIMODAL_CONTRACT_REVISION,
        "map_uuid": map_uuid,
        "split": map_split(map_uuid),
        "sensor_authority": "urban_car_depth_vlp16_sim",
        "timestamp_anchor": "lidar",
        "timestamps": timestamps,
        "skew_s": skews,
        "transforms": transforms,
        "raw_authority": raw_authority,
        "preprocessing": preprocessing,
        "interpolation": interpolation,
        "lidar_timing": lidar_timing,
        "maneuver_mode": maneuver_mode,
        "metadata": metadata or {},
    }
    np.savez_compressed(
        str(path),
        depth_metric=depth_metric,
        depth_validity=depth_validity,
        lidar_points=lidar_points,
        lidar_bev=lidar_bev,
        imu_measurement=imu_measurement,
        vehicle_state=vehicle_state,
        current_gear=np.asarray(int(current_gear), dtype=np.int8),
        requested_gear=np.asarray(int(requested_gear), dtype=np.int8),
        local_path=local_path,
        local_path_gears=local_path_gears,
        subgoal_body=np.asarray(subgoal_body, dtype=np.float32),
        manifest_json=np.asarray(json.dumps(manifest, sort_keys=True)),
        **candidate_arrays,
    )
    return manifest


def audit_multimodal_sample(path, maximum_skews=None):
    """Return a list of contract violations; an empty list means PASS."""

    maximum_skews = maximum_skews or {"depth": 0.05, "candidates": 0.15}
    errors = []
    try:
        with np.load(str(path), allow_pickle=False) as data:
            manifest = json.loads(str(data["manifest_json"]))
            if manifest.get("schema") != MULTIMODAL_SCHEMA_VERSION:
                errors.append("schema")
            if manifest.get("contract_revision") != MULTIMODAL_CONTRACT_REVISION:
                errors.append("contract_revision")
            if data["depth_metric"].shape != data["depth_validity"].shape:
                errors.append("depth_shape")
            if data["lidar_points"].ndim != 2 or data["lidar_points"].shape[1] < 3:
                errors.append("lidar_points_shape")
            if data["lidar_bev"].ndim != 3 or data["lidar_bev"].shape[0] != 6:
                errors.append("lidar_bev_shape")
            if data["imu_measurement"].shape != (10,):
                errors.append("imu_shape")
            if data["vehicle_state"].shape != (9,):
                errors.append("vehicle_state_shape")
            if data["local_path"].shape[0] != data["local_path_gears"].shape[0]:
                errors.append("route_shape")
            if data["trajectories"].shape[0] != 15:
                errors.append("candidate_count")
            if int(data["requested_gear"]) not in (-1, 1):
                errors.append("requested_gear")
            if set(data["candidate_gear"].tolist()) != {int(data["requested_gear"])}:
                errors.append("candidate_gear_condition")
            if manifest.get("maneuver_mode") not in MANEUVER_MODES:
                errors.append("maneuver_mode")
            raw = manifest.get("raw_authority", {})
            if raw.get("kind") == "embedded_diagnostic":
                if len(data["lidar_points"]) == 0:
                    errors.append("embedded_raw_missing")
            elif raw.get("kind") == "rosbag_reference":
                messages = raw.get("messages", {})
                if len(raw.get("bag_sha256", "")) != 64 or not all(name in messages for name in ("lidar", "depth", "imu")):
                    errors.append("rosbag_authority")
                for message in messages.values():
                    if not all(key in message for key in ("topic", "message_index", "timestamp")):
                        errors.append("raw_message_reference")
                        break
            else:
                errors.append("raw_authority")
            preprocessing = manifest.get("preprocessing", {}).get("lidar_bev", {})
            if len(preprocessing.get("sha256", "")) != 64:
                errors.append("preprocessing_hash")
            timing = manifest.get("lidar_timing", {})
            if not all(key in timing for key in ("model", "scan_period_s", "per_point_time_available", "deskew_applied")):
                errors.append("lidar_timing")
            required_transforms = ("lidar_to_chassis", "camera_to_chassis", "chassis_to_map")
            for name in required_transforms:
                transform = manifest.get("transforms", {}).get(name, {})
                if not all(key in transform for key in ("matrix", "measurement_stamp", "source_frame", "target_frame")):
                    errors.append("transform_" + name)
            capture_mode = raw.get("kind")
            interpolation = manifest.get("interpolation", {})
            for name in ("odom", "joint_state", "imu"):
                detail = interpolation.get(name, {})
                if capture_mode == "rosbag_reference" and detail.get("method") not in ("linear", "linear+slerp", "exact"):
                    errors.append("interpolation_" + name)
                if capture_mode == "rosbag_reference" and len(detail.get("source_stamps", [])) != 2:
                    errors.append("interpolation_sources_" + name)
            for name, maximum in maximum_skews.items():
                if manifest.get("skew_s", {}).get(name, float("inf")) > maximum:
                    errors.append("skew_" + name)
    except Exception as exc:
        errors.append("unreadable:" + type(exc).__name__)
    return errors
