"""Lean, provenance-preserving data view for P5 score calibration.

The accepted Candidate checkpoints are bound to :mod:`p4_dataset`, so that
module must remain byte-for-byte unchanged.  Score calibration needs only the
sensor/state/route/map tensors consumed by the frozen candidate bank and the
score objective.  This view therefore reuses the already verified P3 index and
map authority while avoiding legacy labels, raw points, candidate trajectories
and the unused LiDAR BEV distance transform on every epoch.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch

from dep_car.core.state_contract import (
    ACCELERATION_LIMIT_MPS2,
    DECELERATION_LIMIT_MPS2,
    FORWARD_SPEED_LIMIT_MPS,
    REVERSE_SPEED_LIMIT_MPS,
    YAW_RATE_SCALE_RADPS,
)
from dep_car.core.vehicle import STEERING_OPERATING_LIMIT_RAD
from dep_car.training.dataset import (
    MULTIMODAL_CONTRACT_REVISION,
    MULTIMODAL_SCHEMA_VERSION,
    map_split,
)
from dep_car.training.p4_dataset import (
    DEPTH_MAXIMUM_M,
    P3TrainingDataError,
    P3TrainingDatasetV1,
    _active_same_gear_route,
    _normalize_bev,
    _reference_curvature,
    _stat_identity,
)


SCORE_TRAINING_VIEW_SCHEMA = "P3ScoreTrainingDatasetV1"
SCORE_TRAINING_VIEW_REVISION = 1
SCORE_MODALITIES = ("depth_only", "lidar_only", "fusion")


class P3ScoreTrainingDatasetV1(P3TrainingDatasetV1):
    """Minimal tensor view used only by the optimized Score Head trainer.

    ``P3TrainingDatasetV1.__init__`` performs the full SHA-256 verification of
    the sealed development index.  Immediately afterwards we freeze each
    sample's full stat identity (device/inode/size/mtime/ctime).  Every access
    checks that identity before and after parsing, so epochs can avoid hashing
    the same immutable NPZ bytes repeatedly without weakening run-time change
    detection.
    """

    def __init__(self, *args, modality="fusion", **kwargs):
        modality = str(modality)
        if modality not in SCORE_MODALITIES:
            raise P3TrainingDataError(
                "score modality must be depth_only, lidar_only or fusion"
            )
        super().__init__(*args, **kwargs)
        self.modality = modality
        self._sample_identities = {}
        for entry in self.entries:
            path = (self.sample_root / entry["path"]).resolve()
            if self.sample_root not in path.parents or not path.is_file():
                raise P3TrainingDataError(
                    "indexed score sample is unavailable: %s" % path
                )
            stat = path.stat()
            if (
                int(stat.st_size) != int(entry["size_bytes"])
                or int(stat.st_mtime_ns) != int(entry["mtime_ns"])
            ):
                raise P3TrainingDataError(
                    "indexed score sample changed after verification: %s" % path
                )
            self._sample_identities[str(path)] = _stat_identity(stat)
        self._zero_depth = torch.zeros((2, 96, 160), dtype=torch.float32)
        self._zero_lidar = torch.zeros((6, 160, 160), dtype=torch.float32)

    def _sample_path_and_identity(self, entry):
        path = (self.sample_root / entry["path"]).resolve()
        if self.sample_root not in path.parents or not path.is_file():
            raise P3TrainingDataError("indexed score sample is unavailable: %s" % path)
        expected = self._sample_identities.get(str(path))
        try:
            actual = _stat_identity(path.stat())
        except OSError as exc:
            raise P3TrainingDataError(
                "unable to stat indexed score sample: %s" % path
            ) from exc
        if expected is None or actual != expected:
            raise P3TrainingDataError("indexed score sample changed: %s" % path)
        return path, expected

    def __getitem__(self, index):
        entry = self.entries[int(index)]
        path, expected_identity = self._sample_path_and_identity(entry)
        try:
            archive = np.load(str(path), allow_pickle=False)
        except Exception as exc:
            raise P3TrainingDataError(
                "unable to open indexed score sample %s: %s" % (path, exc)
            ) from exc

        try:
            with archive as data:
                manifest = json.loads(str(data["manifest_json"]))
                map_uuid = str(manifest.get("map_uuid", ""))
                if (
                    manifest.get("schema") != MULTIMODAL_SCHEMA_VERSION
                    or manifest.get("contract_revision")
                    != MULTIMODAL_CONTRACT_REVISION
                    or manifest.get("split") != self.split
                    or self.split != map_split(map_uuid)
                    or map_uuid != entry["map_uuid"]
                ):
                    raise P3TrainingDataError(
                        "score sample/index provenance mismatch: %s" % path
                    )
                if manifest.get("sensor_authority") != self.expected_sensor_authority:
                    raise P3TrainingDataError(
                        "score sample sensor authority changed: %s" % path
                    )
                preprocessing = manifest.get("preprocessing", {}).get(
                    "lidar_bev", {}
                )
                if preprocessing.get("sha256") != self.expected_preprocessing_sha256:
                    raise P3TrainingDataError(
                        "score sample BEV contract changed: %s" % path
                    )

                if self.modality in ("depth_only", "fusion"):
                    depth = np.asarray(data["depth_metric"], dtype=np.float32)
                    validity_raw = np.asarray(data["depth_validity"])
                    if depth.shape != (480, 640) or validity_raw.shape != depth.shape:
                        raise P3TrainingDataError(
                            "depth must be [480,640]: %s" % path
                        )
                    if not np.all(np.isfinite(depth)) or not np.all(
                        np.isin(validity_raw, (0, 1))
                    ):
                        raise P3TrainingDataError(
                            "depth contains invalid values: %s" % path
                        )
                    validity = validity_raw.astype(np.float32)
                    normalized = np.where(
                        validity > 0.5,
                        np.clip(depth, 0.0, DEPTH_MAXIMUM_M) / DEPTH_MAXIMUM_M,
                        1.0,
                    ).astype(np.float32)
                    normalized = cv2.resize(
                        normalized, (160, 96), interpolation=cv2.INTER_NEAREST
                    )
                    validity = cv2.resize(
                        validity, (160, 96), interpolation=cv2.INTER_NEAREST
                    )
                    depth_tensor = torch.from_numpy(
                        np.stack((normalized, validity)).astype(np.float32)
                    )
                else:
                    depth_tensor = self._zero_depth

                if self.modality in ("lidar_only", "fusion"):
                    bev_raw = np.asarray(data["lidar_bev"], dtype=np.float32)
                    if bev_raw.shape != (6, 160, 160) or not np.all(
                        np.isfinite(bev_raw)
                    ):
                        raise P3TrainingDataError(
                            "LiDAR BEV must be finite [6,160,160]: %s" % path
                        )
                    bev_resolution = float(
                        preprocessing.get("bev", {}).get("resolution", 0.0)
                    )
                    if bev_resolution <= 0.0:
                        raise P3TrainingDataError(
                            "BEV resolution is invalid: %s" % path
                        )
                    lidar_tensor = torch.from_numpy(
                        _normalize_bev(bev_raw, preprocessing)
                    )
                else:
                    lidar_tensor = self._zero_lidar

                state_raw = np.asarray(data["vehicle_state"], dtype=np.float32)
                subgoal = np.asarray(data["subgoal_body"], dtype=np.float32)
                if (
                    state_raw.shape != (9,)
                    or subgoal.shape != (2,)
                    or not np.all(np.isfinite(state_raw))
                    or not np.all(np.isfinite(subgoal))
                ):
                    raise P3TrainingDataError(
                        "vehicle state/subgoal shape is invalid: %s" % path
                    )
                if not np.allclose(state_raw[4:6], subgoal, atol=1.0e-5):
                    raise P3TrainingDataError(
                        "state and subgoal disagree: %s" % path
                    )
                current_gear = int(data["current_gear"])
                requested_gear = int(data["requested_gear"])
                if current_gear not in (-1, 0, 1) or requested_gear not in (-1, 1):
                    raise P3TrainingDataError("gear value is invalid: %s" % path)

                local_path = np.asarray(data["local_path"], dtype=np.float32)
                route_gears = np.asarray(data["local_path_gears"], dtype=np.int8)
                if (
                    local_path.ndim != 2
                    or local_path.shape[1] != 3
                    or len(local_path) == 0
                    or route_gears.shape != (len(local_path),)
                    or not np.all(np.isfinite(local_path))
                    or not np.all(np.isin(route_gears, (-1, 0, 1)))
                ):
                    raise P3TrainingDataError("local route is invalid: %s" % path)
                heading_error = math.atan2(float(state_raw[6]), float(state_raw[7]))
                active_route, _ = _active_same_gear_route(
                    local_path,
                    route_gears,
                    requested_gear,
                    subgoal,
                    heading_error,
                )
                if len(active_route) > self.maximum_route_points:
                    active_route = active_route[: self.maximum_route_points]

                state_physical = state_raw.copy()
                state_physical[0] = np.clip(
                    state_physical[0],
                    -REVERSE_SPEED_LIMIT_MPS,
                    FORWARD_SPEED_LIMIT_MPS,
                )
                directed_acceleration = np.clip(
                    requested_gear * state_physical[1],
                    -DECELERATION_LIMIT_MPS2,
                    ACCELERATION_LIMIT_MPS2,
                )
                state_physical[1] = requested_gear * directed_acceleration
                state_physical[2] = np.clip(
                    state_physical[2],
                    -STEERING_OPERATING_LIMIT_RAD,
                    STEERING_OPERATING_LIMIT_RAD,
                )
                state_physical[3] = np.clip(
                    state_physical[3], -YAW_RATE_SCALE_RADPS, YAW_RATE_SCALE_RADPS
                )
                state_physical[6:8] = np.clip(state_physical[6:8], -1.0, 1.0)
                state_physical[8] = _reference_curvature(active_route)
                geometry_valid = current_gear != -requested_gear
                if not geometry_valid:
                    state_physical[0] = 0.0
                    state_physical[1] = 0.0

                route_pose = np.zeros(
                    (self.maximum_route_points, 3), dtype=np.float32
                )
                route_mask = np.zeros(self.maximum_route_points, dtype=bool)
                route_pose[: len(active_route)] = active_route
                route_mask[: len(active_route)] = True

                transform = np.asarray(
                    manifest.get("transforms", {})
                    .get("chassis_to_map", {})
                    .get("matrix", ()),
                    dtype=np.float32,
                )
                if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
                    raise P3TrainingDataError(
                        "chassis_to_map transform is invalid: %s" % path
                    )
        except P3TrainingDataError:
            raise
        except Exception as exc:
            raise P3TrainingDataError(
                "unable to parse indexed score sample %s: %s" % (path, exc)
            ) from exc

        try:
            final_identity = _stat_identity(path.stat())
        except OSError as exc:
            raise P3TrainingDataError(
                "indexed score sample vanished after parsing: %s" % path
            ) from exc
        if final_identity != expected_identity:
            raise P3TrainingDataError(
                "indexed score sample changed while parsing: %s" % path
            )

        modality_mask = {
            "depth_only": (1.0, 0.0),
            "lidar_only": (0.0, 1.0),
            "fusion": (1.0, 1.0),
        }[self.modality]
        metadata = manifest.get("metadata", {})
        raw_context = str(metadata.get("candidate_context", "UNKNOWN"))
        context_known = raw_context in ("MISSION", "RECOVERY")
        map_tensors = self._map_tensors(map_uuid)
        return {
            "depth": depth_tensor,
            "lidar_bev": lidar_tensor,
            "modality_mask": torch.tensor(modality_mask, dtype=torch.float32),
            "state": torch.from_numpy(state_physical.astype(np.float32)),
            "requested_gear": torch.tensor(requested_gear, dtype=torch.int64),
            "geometry_valid": torch.tensor(geometry_valid, dtype=torch.bool),
            "route_pose": torch.from_numpy(route_pose),
            "route_mask": torch.from_numpy(route_mask),
            "map_distance_field": map_tensors["map_distance_field"],
            "map_resolution": map_tensors["map_resolution"],
            "map_origin": map_tensors["map_origin"],
            "chassis_to_map": torch.from_numpy(transform),
            "metadata": {
                "schema": SCORE_TRAINING_VIEW_SCHEMA,
                "revision": SCORE_TRAINING_VIEW_REVISION,
                "path": str(path),
                "sample_id": entry["sample_id"],
                "task_id": entry["task_id"],
                "map_uuid": map_uuid,
                "split": self.split,
                "maneuver_mode": manifest["maneuver_mode"],
                "candidate_context": raw_context if context_known else "UNKNOWN",
                "candidate_context_known": context_known,
                "preprocessing_sha256": self.expected_preprocessing_sha256,
                "loaded_modality": self.modality,
            },
        }


__all__ = [
    "P3ScoreTrainingDatasetV1",
    "SCORE_MODALITIES",
    "SCORE_TRAINING_VIEW_REVISION",
    "SCORE_TRAINING_VIEW_SCHEMA",
]
