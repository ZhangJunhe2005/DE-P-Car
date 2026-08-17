#!/usr/bin/env python3
"""Development-only P3 -> P5 candidate-capacity re-audit.

The immutable ``P3TrainingIndexV2`` is the *only* sample inventory used by
this program.  In particular, the sample tree is never globbed and the sealed
test split is never opened.  Candidate trajectories and feasibility labels in
the old NPZ files are not qualification authorities: the current canonical
3x5 Ackermann lattice is regenerated from the recorded physical state, then
checked against the indexed map authority with the frozen P5 training
footprint policy.

This program intentionally exposes no geometry, rollout, allowance, gate, or
dataset-path overrides.  ``--maximum-samples`` is diagnostic only and can
never produce PASS.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INDEX = ROOT / "data" / "p3_pilot" / "run" / "training_index.json"
DEFAULT_SAMPLES = ROOT / "data" / "p3_pilot" / "run" / "samples"
DEFAULT_MAPS = ROOT / "data" / "p3_pilot" / "maps"
DEFAULT_REPORT = ROOT / "reports" / "p3_development_reaudit_v3.json"
DEFAULT_P3_ACCEPTANCE = ROOT / "reports" / "p3_pilot_acceptance.json"
DEFAULT_TASK_MANIFEST = ROOT / "data" / "p3_pilot" / "task_manifest.json"

REPORT_SCHEMA = "DEPCarP3DevelopmentReauditV3"
EXPECTED_INDEX_SCHEMA = "P3TrainingIndexV2"
EXPECTED_TRAINING_VIEW = "P3TrainingDatasetV1"
EXPECTED_SAMPLE_SCHEMA = "StaticAckermannSampleV2"
EXPECTED_CONTRACT_REVISION = 2
EXPECTED_SENSOR_AUTHORITY = "urban_car_depth_vlp16_sim"
EXPECTED_BEV_PREPROCESSING_SHA256 = (
    "89be32ba8f15fce9ff85332070c4da74668d4853a7419dd986a0d8bcafb3fb7b"
)
EXPECTED_INDEX_SHA256 = (
    "962242859a862c87123e201ed506afa3a0018c4c551ffcd68e469ab4a890494d"
)
EXPECTED_CONTENT_AGGREGATE_SHA256 = (
    "e7bbe901877bae81e04117a99c0c935087c3098372cf60b48358857a466ff1c2"
)
EXPECTED_MAP_CONTRACT_AGGREGATE_SHA256 = (
    "9d0251f764c1983a5ff73db67af481ab891b913b7984cd4de4cd967e774d1fa2"
)
EXPECTED_SPLIT_COUNTS = {"train": 8268, "validation": 1022}
EXPECTED_CANDIDATES = 15
EXPECTED_WORKERS = 8
OVERALL_ZERO_FEASIBLE_LIMIT = 0.10
PER_MODE_ZERO_FEASIBLE_LIMIT = 0.25
MINIMUM_MEDIAN_FEASIBLE = 2.0

sys.path.insert(0, str(ROOT / "dep_car" / "src"))
from dep_car.core.lattice import AckermannLattice, LatticeConfig
from dep_car.core.occupancy import (
    FIVE_CIRCLE_FOOTPRINT_SCHEMA,
    FOOTPRINT_ALLOWANCE_SCHEMA,
    SWEPT_INTERPOLATION_SCHEMA,
    SWEPT_SUBSTEPS_PER_SEGMENT,
    TRAINING_FOOTPRINT_ALLOWANCE,
    TRAINING_ONE_DIAGONAL_MULTIPLIER,
    FootprintConfig,
    OccupancyGrid2D,
)
from dep_car.core.state_contract import (
    FORWARD_SPEED_LIMIT_MPS,
    REVERSE_SPEED_LIMIT_MPS,
)
from dep_car.core.types import Gear, VehicleState
from dep_car.core.vehicle import STEERING_OPERATING_LIMIT_RAD
from dep_car.model.implementation_contract import build_p4_implementation_contract
from dep_car.training.dataset import map_split
from dep_car.training.p4_dataset import (
    TRAINING_INDEX_CONTENT_AGGREGATE_SCHEMA,
    TRAINING_INDEX_CONTENT_HASH_ALGORITHM,
    _ros_map_known_free,
    _signed_distance_field,
    training_index_content_aggregate,
)
from dep_car.training.losses import swept_map_footprint_clearance
from dep_car.training.pilot import PILOT_MANEUVER_MODES


class DevelopmentReauditError(ValueError):
    """Raised when a frozen P3 development authority is inconsistent."""


@dataclass(frozen=True)
class MapSpec:
    map_uuid: str
    split: str
    image: str
    resolution_m: float
    origin_xy: tuple
    negate: int
    occupied_threshold: float
    free_threshold: float
    decoded_occupancy_sha256: str
    png_sha256: str


@dataclass(frozen=True)
class IndexAuthority:
    payload: dict
    entries: tuple
    index_sha256: str
    content_aggregate_sha256: str
    counts_by_split: dict
    counts_by_mode: dict


_WORKER_GRIDS = {}
_WORKER_SPECS = {}
_WORKER_MAP_DISTANCE_FIELDS = {}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bytes_sha256(value):
    return hashlib.sha256(value).hexdigest()


def canonical_hash(value):
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return bytes_sha256(payload)


def is_sha256(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def load_p3_provenance(
    acceptance_path=DEFAULT_P3_ACCEPTANCE,
    task_manifest_path=DEFAULT_TASK_MANIFEST,
):
    """Authenticate the P3 acceptance file and task-manifest internal hash."""

    acceptance_path = Path(acceptance_path).resolve()
    task_manifest_path = Path(task_manifest_path).resolve()
    try:
        acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
        task_manifest = json.loads(task_manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DevelopmentReauditError("P3 provenance is unreadable: %s" % exc) from exc
    if (
        acceptance.get("schema") != "DEPCarP3PilotAcceptanceV1"
        or acceptance.get("status") != "PASS"
    ):
        raise DevelopmentReauditError("P3 pilot acceptance is not authoritative PASS")
    internal_hash = task_manifest.get("task_manifest_sha256")
    task_content = dict(task_manifest)
    task_content.pop("task_manifest_sha256", None)
    if (
        task_manifest.get("schema") != "DEPCarPilotTaskManifestV1"
        or not is_sha256(internal_hash)
        or canonical_hash(task_content) != internal_hash
    ):
        raise DevelopmentReauditError("P3 task-manifest internal hash is invalid")
    if acceptance.get("task_manifest_sha256") != internal_hash:
        raise DevelopmentReauditError("P3 acceptance/task-manifest identity mismatch")
    return {
        "acceptance_path": _project_relative(acceptance_path),
        "acceptance_sha256": file_sha256(acceptance_path),
        "task_manifest_path": _project_relative(task_manifest_path),
        "task_manifest_file_sha256": file_sha256(task_manifest_path),
        "task_manifest_sha256": internal_hash,
        "task_manifest_internal_hash_verified": True,
    }


def _safe_indexed_sample_path(sample_root, entry):
    sample_root = Path(sample_root).resolve()
    relative = Path(str(entry.get("path", "")))
    map_uuid = str(entry.get("map_uuid", ""))
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or len(relative.parts) != 2
        or relative.suffix != ".npz"
        or not map_uuid
        or relative.parts[0] != map_uuid
    ):
        raise DevelopmentReauditError("unsafe sample path in training index")
    path = (sample_root / relative).resolve()
    expected_parent = (sample_root / map_uuid).resolve()
    if sample_root not in path.parents or path.parent != expected_parent:
        raise DevelopmentReauditError("indexed sample escapes its map folder")
    return path


def load_index_authority(
    index_path,
    sample_root,
    maps_root,
    *,
    expected_index_sha256=EXPECTED_INDEX_SHA256,
    expected_content_aggregate=EXPECTED_CONTENT_AGGREGATE_SHA256,
    expected_split_counts=EXPECTED_SPLIT_COUNTS,
):
    """Validate the immutable development index without enumerating samples."""

    index_path = Path(index_path).resolve()
    sample_root = Path(sample_root).resolve()
    maps_root = Path(maps_root).resolve()
    try:
        raw = index_path.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DevelopmentReauditError(
            "unable to read P3 training index: %s" % exc
        ) from exc
    index_sha256 = bytes_sha256(raw)
    if expected_index_sha256 is not None and index_sha256 != expected_index_sha256:
        raise DevelopmentReauditError("P3 training index SHA-256 mismatch")

    fixed = {
        "schema": EXPECTED_INDEX_SCHEMA,
        "training_view": EXPECTED_TRAINING_VIEW,
        "content_hash_algorithm": TRAINING_INDEX_CONTENT_HASH_ALGORITHM,
        "content_aggregate_schema": TRAINING_INDEX_CONTENT_AGGREGATE_SCHEMA,
        "sample_root": str(sample_root),
        "maps_root": str(maps_root),
        "splits": ["train", "validation"],
        "workers": EXPECTED_WORKERS,
        "sensor_authority": EXPECTED_SENSOR_AUTHORITY,
        "bev_preprocessing_sha256": EXPECTED_BEV_PREPROCESSING_SHA256,
    }
    mismatches = [key for key, expected in fixed.items() if payload.get(key) != expected]
    if mismatches:
        raise DevelopmentReauditError(
            "P3 training index contract mismatch: " + ", ".join(mismatches)
        )
    entries = payload.get("entries")
    if not isinstance(entries, list) or not entries:
        raise DevelopmentReauditError("P3 training index is empty")
    if payload.get("samples") != len(entries):
        raise DevelopmentReauditError("P3 training index sample count is inconsistent")

    seen = set()
    normalized_entries = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise DevelopmentReauditError("P3 training index entry is not a mapping")
        path = _safe_indexed_sample_path(sample_root, entry)
        relative = str(Path(str(entry["path"])))
        if relative in seen:
            raise DevelopmentReauditError("duplicate path in P3 training index")
        seen.add(relative)
        map_uuid = str(entry.get("map_uuid", ""))
        split = str(entry.get("split", ""))
        mode = str(entry.get("maneuver_mode", ""))
        if split not in ("train", "validation") or split != map_split(map_uuid):
            raise DevelopmentReauditError("indexed sample is outside development splits")
        if mode not in PILOT_MANEUVER_MODES:
            raise DevelopmentReauditError("indexed maneuver mode is not in the frozen pilot set")
        if not is_sha256(entry.get("content_sha256")):
            raise DevelopmentReauditError("indexed sample content SHA-256 is invalid")
        try:
            size_bytes = int(entry["size_bytes"])
            mtime_ns = int(entry["mtime_ns"])
            stat = path.stat()
        except (KeyError, TypeError, ValueError, OSError) as exc:
            raise DevelopmentReauditError(
                "indexed sample stat authority is invalid: %s" % relative
            ) from exc
        if (
            not path.is_file()
            or size_bytes < 0
            or mtime_ns < 0
            or int(stat.st_size) != size_bytes
            or int(stat.st_mtime_ns) != mtime_ns
        ):
            raise DevelopmentReauditError(
                "indexed sample stat mismatch: %s" % relative
            )
        normalized_entries.append(dict(entry))

    counts_by_split = dict(sorted(Counter(
        str(entry["split"]) for entry in normalized_entries
    ).items()))
    counts_by_mode = dict(sorted(Counter(
        str(entry["maneuver_mode"]) for entry in normalized_entries
    ).items()))
    if payload.get("counts_by_split") != counts_by_split:
        raise DevelopmentReauditError("P3 index split counts are internally inconsistent")
    if payload.get("counts_by_mode") != counts_by_mode:
        raise DevelopmentReauditError("P3 index maneuver counts are internally inconsistent")
    if expected_split_counts is not None and counts_by_split != dict(expected_split_counts):
        raise DevelopmentReauditError("P3 development split counts changed")
    content_aggregate = training_index_content_aggregate(normalized_entries)
    if payload.get("content_aggregate_sha256") != content_aggregate:
        raise DevelopmentReauditError("P3 index content aggregate is inconsistent")
    if (
        expected_content_aggregate is not None
        and content_aggregate != expected_content_aggregate
    ):
        raise DevelopmentReauditError("P3 development content aggregate changed")
    return IndexAuthority(
        payload=payload,
        entries=tuple(normalized_entries),
        index_sha256=index_sha256,
        content_aggregate_sha256=content_aggregate,
        counts_by_split=counts_by_split,
        counts_by_mode=counts_by_mode,
    )


def _selected_map_folder(maps_root, map_uuid):
    """Resolve a generated map from its folder name without opening other maps."""

    suffix = "_" + str(map_uuid)[:8]
    matches = [
        path for path in Path(maps_root).iterdir()
        if path.is_dir() and path.name.endswith(suffix)
    ]
    if len(matches) != 1:
        raise DevelopmentReauditError(
            "indexed map folder resolution is not unique: %s" % map_uuid
        )
    return matches[0].resolve()


def load_map_specs(
    maps_root,
    selected_map_uuids,
    *,
    expected_aggregate=EXPECTED_MAP_CONTRACT_AGGREGATE_SHA256,
):
    """Load YAML/PNG semantics for indexed maps only.

    The aggregate rows intentionally match ``IndexedMapContractAggregateV1``
    in ``tools/train_dep_car.py`` byte-for-byte.
    """

    maps_root = Path(maps_root).resolve()
    selected = {str(value) for value in selected_map_uuids}
    if not maps_root.is_dir() or not selected:
        raise DevelopmentReauditError("indexed map authority is empty")
    specs = {}
    aggregate_rows = []
    for map_uuid in sorted(selected):
        folder = _selected_map_folder(maps_root, map_uuid)
        manifest_path = folder / "manifest.json"
        yaml_path = folder / "map.yaml"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            metadata = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, yaml.YAMLError) as exc:
            raise DevelopmentReauditError(
                "indexed map authority is unreadable: %s" % folder
            ) from exc
        if not isinstance(metadata, dict) or manifest.get("map_uuid") != map_uuid:
            raise DevelopmentReauditError("indexed map manifest/YAML identity mismatch")
        try:
            image_name = str(metadata["image"])
            image = (folder / image_name).resolve()
            origin = [float(value) for value in metadata["origin"]]
            semantic = {
                "image": image_name,
                "resolution": float(metadata["resolution"]),
                "origin": origin,
                "negate": int(metadata.get("negate", 0)),
                "occupied_thresh": float(metadata.get("occupied_thresh", 0.65)),
                "free_thresh": float(metadata.get("free_thresh", 0.196)),
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise DevelopmentReauditError(
                "indexed map YAML semantics are invalid: %s" % yaml_path
            ) from exc
        if (
            image.parent != folder
            or not image.is_file()
            or len(origin) < 3
            or not all(math.isfinite(value) for value in origin)
            # The current P4/P5 world-to-cell implementation consumes only
            # origin x/y.  Accepting a rotated ROS map here would silently
            # audit different geometry from the one encoded by map.yaml.
            or abs(origin[2]) > 1.0e-9
            or not math.isfinite(semantic["resolution"])
            or semantic["resolution"] <= 0.0
            or semantic["negate"] not in (0, 1)
            or not 0.0 <= semantic["free_thresh"] < semantic["occupied_thresh"] <= 1.0
        ):
            raise DevelopmentReauditError("indexed map contract is invalid: %s" % folder)
        claimed = manifest.get("occupancy_sha256")
        if not is_sha256(claimed):
            raise DevelopmentReauditError("indexed map occupancy hash is invalid")
        try:
            image_bytes = image.read_bytes()
        except OSError as exc:
            raise DevelopmentReauditError(
                "indexed map PNG is unreadable: %s" % image
            ) from exc
        png_sha256 = bytes_sha256(image_bytes)
        pixels = cv2.imdecode(
            np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_GRAYSCALE
        )
        if pixels is None:
            raise DevelopmentReauditError("indexed map PNG is unreadable: %s" % image)
        decoded = bytes_sha256(np.asarray(pixels, dtype=np.uint8).tobytes())
        if decoded != claimed:
            raise DevelopmentReauditError("indexed map decoded occupancy hash mismatch")
        spec = MapSpec(
            map_uuid=map_uuid,
            split=map_split(map_uuid),
            image=str(image),
            resolution_m=semantic["resolution"],
            origin_xy=(origin[0], origin[1]),
            negate=semantic["negate"],
            occupied_threshold=semantic["occupied_thresh"],
            free_threshold=semantic["free_thresh"],
            decoded_occupancy_sha256=decoded,
            png_sha256=png_sha256,
        )
        specs[map_uuid] = spec
        aggregate_rows.append({
            "map_uuid": map_uuid,
            "png_sha256": png_sha256,
            "occupancy_sha256": decoded,
            "yaml": semantic,
        })
    aggregate = canonical_hash(aggregate_rows)
    if expected_aggregate is not None and aggregate != expected_aggregate:
        raise DevelopmentReauditError("indexed map contract aggregate changed")
    return specs, {
        "schema": "IndexedMapContractAggregateV1",
        "map_count": len(aggregate_rows),
        "aggregate_sha256": aggregate,
        "rows": aggregate_rows,
    }


def grid_from_spec(spec):
    pixels = cv2.imread(str(spec.image), cv2.IMREAD_GRAYSCALE)
    if pixels is None:
        raise DevelopmentReauditError("indexed map image became unreadable")
    decoded = bytes_sha256(np.asarray(pixels, dtype=np.uint8).tobytes())
    if decoded != spec.decoded_occupancy_sha256 or file_sha256(spec.image) != spec.png_sha256:
        raise DevelopmentReauditError("indexed map changed after authority validation")
    probability = np.asarray(pixels, dtype=np.float32) / 255.0
    if not spec.negate:
        probability = 1.0 - probability
    occupied = probability > spec.occupied_threshold
    known_free = probability < spec.free_threshold
    data = np.full(pixels.shape, -1, dtype=np.int16)
    data[known_free] = 0
    data[occupied] = 100
    return OccupancyGrid2D(
        np.flipud(data), spec.resolution_m, spec.origin_xy,
        unknown_is_occupied=True,
    )


def map_signed_sdf_from_spec(spec):
    """Build the exact P5 map SDF through the production preprocessing code."""

    pixels = cv2.imread(str(spec.image), cv2.IMREAD_GRAYSCALE)
    if pixels is None:
        raise DevelopmentReauditError("indexed map image became unreadable")
    decoded = bytes_sha256(np.asarray(pixels, dtype=np.uint8).tobytes())
    if decoded != spec.decoded_occupancy_sha256 or file_sha256(spec.image) != spec.png_sha256:
        raise DevelopmentReauditError("indexed map changed before SDF construction")
    known_free = np.flipud(_ros_map_known_free(pixels, spec)).copy()
    distance = _signed_distance_field(known_free, spec.resolution_m)
    return torch.from_numpy(np.asarray(distance, dtype=np.float32)[None, None])


def initialize_worker(specs):
    global _WORKER_GRIDS, _WORKER_SPECS, _WORKER_MAP_DISTANCE_FIELDS
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    _WORKER_SPECS = dict(specs)
    _WORKER_GRIDS = {
        map_uuid: grid_from_spec(spec) for map_uuid, spec in specs.items()
    }
    _WORKER_MAP_DISTANCE_FIELDS = {
        map_uuid: map_signed_sdf_from_spec(spec)
        for map_uuid, spec in specs.items()
    }


def canonical_vehicle_state(vehicle_state, current_gear, requested_gear):
    """Return the executable boundary state for the requested drive gear."""

    raw = np.asarray(vehicle_state, dtype=np.float64)
    if raw.shape != (9,) or not np.all(np.isfinite(raw)):
        raise DevelopmentReauditError("vehicle_state must be finite [9]")
    try:
        requested = Gear.require_drive(requested_gear)
        current = Gear(int(current_gear))
    except (TypeError, ValueError) as exc:
        raise DevelopmentReauditError("sample gear is invalid") from exc
    speed = float(raw[0])
    if current == -requested:
        speed = 0.0
    elif requested == Gear.FORWARD:
        speed = float(np.clip(speed, 0.0, FORWARD_SPEED_LIMIT_MPS))
    else:
        speed = float(np.clip(speed, -REVERSE_SPEED_LIMIT_MPS, 0.0))
    steering = float(np.clip(
        raw[2], -STEERING_OPERATING_LIMIT_RAD, STEERING_OPERATING_LIMIT_RAD
    ))
    return VehicleState(speed=speed, steering=steering), requested


def regenerate_candidate_bank(vehicle_state, current_gear, requested_gear):
    state, requested = canonical_vehicle_state(
        vehicle_state, current_gear, requested_gear
    )
    candidates = AckermannLattice().generate(state, gear=requested)
    if len(candidates) != EXPECTED_CANDIDATES:
        raise DevelopmentReauditError("canonical lattice did not produce 15 candidates")
    trajectories = np.stack([
        np.asarray(candidate.trajectory, dtype=np.float64) for candidate in candidates
    ])
    if trajectories.shape != (EXPECTED_CANDIDATES, 11, 6):
        raise DevelopmentReauditError("canonical lattice trajectory shape changed")
    return trajectories


def transform_candidate_bank_to_map(trajectories, chassis_to_map):
    trajectories = np.asarray(trajectories, dtype=np.float64)
    transform = np.asarray(chassis_to_map, dtype=np.float64)
    if (
        trajectories.ndim != 3
        or trajectories.shape[0] != EXPECTED_CANDIDATES
        or trajectories.shape[2] < 4
        or not np.all(np.isfinite(trajectories))
    ):
        raise DevelopmentReauditError("candidate bank must be finite [15,T,>=4]")
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise DevelopmentReauditError("chassis_to_map must be finite [4,4]")
    if not np.allclose(transform[3], (0.0, 0.0, 0.0, 1.0), atol=1.0e-5):
        raise DevelopmentReauditError("chassis_to_map is not homogeneous")

    output = trajectories.copy()
    points = np.concatenate((
        trajectories[..., 1:3],
        np.zeros((*trajectories.shape[:2], 1), dtype=np.float64),
        np.ones((*trajectories.shape[:2], 1), dtype=np.float64),
    ), axis=-1)
    output[..., 1:3] = np.einsum("ij,bcj->bci", transform, points)[..., :2]
    headings = np.stack((
        np.cos(trajectories[..., 3]),
        np.sin(trajectories[..., 3]),
        np.zeros(trajectories.shape[:2], dtype=np.float64),
    ), axis=-1)
    world_headings = np.einsum("ij,bcj->bci", transform[:3, :3], headings)
    output[..., 3] = np.arctan2(world_headings[..., 1], world_headings[..., 0])
    return output


def reaudit_candidate_bank(trajectories, chassis_to_map, grid, footprint=None):
    """Floor-cell conservative diagnostic under the one-diagonal policy."""

    footprint = FootprintConfig() if footprint is None else footprint
    world = transform_candidate_bank_to_map(trajectories, chassis_to_map)
    feasible = np.zeros(EXPECTED_CANDIDATES, dtype=bool)
    clearance = np.zeros(EXPECTED_CANDIDATES, dtype=np.float64)
    for index, candidate in enumerate(world):
        feasible[index], clearance[index] = grid.swept_footprint_clearance(
            candidate,
            footprint,
            allowance_policy=TRAINING_FOOTPRINT_ALLOWANCE,
        )
    return feasible, clearance


def p5_exact_candidate_bank(
    trajectories,
    chassis_to_map,
    map_distance_field,
    map_resolution,
    map_origin,
    footprint=None,
):
    """Evaluate candidates with the exact differentiable P5 map objective.

    This delegates interpolation, five-circle centre generation, map-frame
    transform, bilinear SDF sampling, outside-map padding, and one-cell-
    diagonal inflation to the production loss implementation.  Qualification
    is based on its strictly-positive minimum clearance.
    """

    footprint = FootprintConfig() if footprint is None else footprint
    values = np.asarray(trajectories, dtype=np.float32)
    transform = np.asarray(chassis_to_map, dtype=np.float32)
    if (
        values.ndim != 3
        or values.shape[0] != EXPECTED_CANDIDATES
        or values.shape[2] < 4
        or not np.all(np.isfinite(values))
    ):
        raise DevelopmentReauditError("candidate bank must be finite [15,T,>=4]")
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise DevelopmentReauditError("chassis_to_map must be finite [4,4]")
    distance = torch.as_tensor(map_distance_field, dtype=torch.float32)
    if distance.ndim == 2:
        distance = distance[None, None]
    elif distance.ndim == 3:
        distance = distance[None]
    if distance.ndim != 4 or distance.shape[:2] != (1, 1):
        raise DevelopmentReauditError("map SDF must represent one [1,1,H,W] map")
    with torch.no_grad():
        swept = swept_map_footprint_clearance(
            torch.from_numpy(values)[None],
            distance,
            torch.tensor([float(map_resolution)], dtype=torch.float32),
            torch.tensor([list(map_origin)], dtype=torch.float32),
            torch.from_numpy(transform)[None],
            footprint=footprint,
        )
        minimum = swept.amin(dim=(-1, -2))[0]
        feasible = minimum > 0.0
        clearance = minimum.clamp_min(0.0)
    return feasible.cpu().numpy(), clearance.cpu().numpy().astype(np.float64)


def inspect_indexed_sample(entry, sample_root):
    """Hash, load, regenerate, and evaluate one explicitly indexed sample."""

    path = _safe_indexed_sample_path(sample_root, entry)
    relative = str(Path(str(entry["path"])))
    try:
        sample_bytes = path.read_bytes()
        if bytes_sha256(sample_bytes) != entry.get("content_sha256"):
            raise DevelopmentReauditError("indexed sample content SHA-256 mismatch")
        if len(sample_bytes) != int(entry.get("size_bytes", -1)):
            raise DevelopmentReauditError("indexed sample byte count mismatch")
        with np.load(io.BytesIO(sample_bytes), allow_pickle=False) as data:
            manifest = json.loads(str(data["manifest_json"]))
            map_uuid = str(manifest.get("map_uuid", ""))
            split = str(manifest.get("split", ""))
            mode = str(manifest.get("maneuver_mode", ""))
            if (
                manifest.get("schema") != EXPECTED_SAMPLE_SCHEMA
                or manifest.get("contract_revision") != EXPECTED_CONTRACT_REVISION
                or manifest.get("sensor_authority") != EXPECTED_SENSOR_AUTHORITY
            ):
                raise DevelopmentReauditError("sample multimodal contract mismatch")
            if (
                map_uuid != entry.get("map_uuid")
                or split != entry.get("split")
                or mode != entry.get("maneuver_mode")
                or map_uuid not in _WORKER_GRIDS
                or split != _WORKER_SPECS[map_uuid].split
                or split != map_split(map_uuid)
            ):
                raise DevelopmentReauditError("sample/index/map identity mismatch")
            preprocessing = manifest.get("preprocessing", {}).get("lidar_bev", {})
            if preprocessing.get("sha256") != EXPECTED_BEV_PREPROCESSING_SHA256:
                raise DevelopmentReauditError("sample BEV preprocessing identity mismatch")
            metadata = manifest.get("metadata", {})
            if not bool(metadata.get("formal_training_authority", False)):
                raise DevelopmentReauditError("sample is not formal P3 authority")
            if (
                metadata.get("map_occupancy_sha256")
                != _WORKER_SPECS[map_uuid].decoded_occupancy_sha256
            ):
                raise DevelopmentReauditError("sample/map occupancy identity mismatch")
            sample_id = str(metadata.get("sample_id", path.stem))
            if str(entry.get("sample_id", sample_id)) != sample_id:
                raise DevelopmentReauditError("sample/index sample_id mismatch")
            transform_contract = manifest.get("transforms", {}).get(
                "chassis_to_map", {}
            )
            if (
                transform_contract.get("source_frame") != "chassis"
                or transform_contract.get("target_frame") != "map"
            ):
                raise DevelopmentReauditError("chassis_to_map frame contract mismatch")
            transform = np.asarray(transform_contract.get("matrix", ()), dtype=np.float64)
            vehicle_state = np.asarray(data["vehicle_state"], dtype=np.float64)
            current_gear = int(data["current_gear"])
            requested_gear = int(data["requested_gear"])

            # Historical labels are diagnostic only.  Their absence or shape
            # cannot alter the regenerated qualification result.
            legacy_feasible = None
            if "feasible" in data.files:
                legacy = np.asarray(data["feasible"])
                if legacy.shape == (EXPECTED_CANDIDATES,) and np.all(
                    np.isin(legacy, (0, 1))
                ):
                    legacy_feasible = legacy.astype(bool)

        trajectories = regenerate_candidate_bank(
            vehicle_state, current_gear, requested_gear
        )
        feasible, clearance = p5_exact_candidate_bank(
            trajectories,
            transform,
            _WORKER_MAP_DISTANCE_FIELDS[map_uuid],
            _WORKER_SPECS[map_uuid].resolution_m,
            _WORKER_SPECS[map_uuid].origin_xy,
        )
        runtime_feasible, runtime_clearance = reaudit_candidate_bank(
            trajectories,
            transform,
            _WORKER_GRIDS[map_uuid],
        )
        row = {
            "path": relative,
            "split": split,
            "mode": mode,
            "map_uuid": map_uuid,
            "requested_gear": requested_gear,
            "new_feasible_count": int(np.sum(feasible)),
            "new_zero": not bool(np.any(feasible)),
            "new_clearance": clearance.tolist(),
            "new_feasible": feasible.tolist(),
            "runtime_conservative_feasible_count": int(np.sum(runtime_feasible)),
            "runtime_conservative_zero": not bool(np.any(runtime_feasible)),
            "runtime_conservative_clearance": runtime_clearance.tolist(),
            "runtime_conservative_feasible": runtime_feasible.tolist(),
            "legacy_available": legacy_feasible is not None,
        }
        if legacy_feasible is not None:
            row["legacy_feasible_count"] = int(np.sum(legacy_feasible))
            row["legacy_zero"] = not bool(np.any(legacy_feasible))
            row["legacy_feasible"] = legacy_feasible.tolist()
        return row
    except Exception as exc:
        return {
            "path": relative,
            "failure": type(exc).__name__ + ": " + str(exc),
        }


def _inspect_entry_task(arguments):
    entry, sample_root = arguments
    return inspect_indexed_sample(entry, sample_root)


def distribution(values):
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if not len(finite):
        return {
            "count": 0,
            "minimum": None,
            "p10": None,
            "median": None,
            "mean": None,
            "p90": None,
            "maximum": None,
        }
    return {
        "count": int(len(finite)),
        "minimum": float(np.min(finite)),
        "p10": float(np.percentile(finite, 10)),
        "median": float(np.median(finite)),
        "mean": float(np.mean(finite)),
        "p90": float(np.percentile(finite, 90)),
        "maximum": float(np.max(finite)),
    }


class AuditAccumulator:
    def __init__(self):
        self.samples = 0
        self.new_zero = 0
        self.new_counts = []
        self.new_clearance = []
        self.new_feasible_clearance = []
        self.runtime_zero = 0
        self.runtime_counts = []
        self.runtime_clearance = []
        self.legacy_samples = 0
        self.legacy_zero = 0
        self.legacy_counts = []

    def add(self, row):
        self.samples += 1
        self.new_zero += int(row["new_zero"])
        self.new_counts.append(row["new_feasible_count"])
        clearance = np.asarray(row["new_clearance"], dtype=np.float64)
        feasible = np.asarray(row["new_feasible"], dtype=bool)
        self.new_clearance.extend(clearance.tolist())
        self.new_feasible_clearance.extend(clearance[feasible].tolist())
        self.runtime_zero += int(row["runtime_conservative_zero"])
        self.runtime_counts.append(row["runtime_conservative_feasible_count"])
        self.runtime_clearance.extend(row["runtime_conservative_clearance"])
        if row.get("legacy_available"):
            self.legacy_samples += 1
            self.legacy_zero += int(row["legacy_zero"])
            self.legacy_counts.append(row["legacy_feasible_count"])

    def summary(self):
        if self.samples:
            zero_rate = self.new_zero / self.samples
            median = float(np.median(self.new_counts))
            mean = float(np.mean(self.new_counts))
        else:
            zero_rate = None
            median = None
            mean = None
        legacy = {
            "diagnostic_only": True,
            "samples_with_valid_legacy_label": self.legacy_samples,
            "zero_feasible_samples": self.legacy_zero,
            "zero_feasible_rate": (
                self.legacy_zero / self.legacy_samples if self.legacy_samples else None
            ),
            "feasible_candidates_median": (
                float(np.median(self.legacy_counts)) if self.legacy_counts else None
            ),
        }
        runtime_conservative = {
            "diagnostic_only": True,
            "authority": "OccupancyGrid2D_floor_cell_training_one_diagonal",
            "zero_feasible_samples": self.runtime_zero,
            "zero_feasible_rate": (
                self.runtime_zero / self.samples if self.samples else None
            ),
            "feasible_candidates_median": (
                float(np.median(self.runtime_counts)) if self.runtime_counts else None
            ),
            "feasible_candidates_mean": (
                float(np.mean(self.runtime_counts)) if self.runtime_counts else None
            ),
            "clearance_m": distribution(self.runtime_clearance),
        }
        return {
            "samples": self.samples,
            "candidates": self.samples * EXPECTED_CANDIDATES,
            "new": {
                "authority": (
                    "regenerated_current_AckermannLattice_plus_"
                    "production_signed_SDF_bilinear_swept_map_footprint_clearance"
                ),
                "zero_feasible_samples": self.new_zero,
                "zero_feasible_rate": zero_rate,
                "feasible_candidates_median": median,
                "feasible_candidates_mean": mean,
            },
            "runtime_conservative": runtime_conservative,
            "legacy": legacy,
            "clearance_m": {
                "new_all_candidates": distribution(self.new_clearance),
                "new_feasible_candidates": distribution(self.new_feasible_clearance),
            },
        }


def footprint_contract():
    footprint = FootprintConfig()
    payload = {
        "schema": FIVE_CIRCLE_FOOTPRINT_SCHEMA,
        "length_m": footprint.length,
        "width_m": footprint.width,
        "safety_margin_m": footprint.safety_margin,
        "circle_count": footprint.circle_count,
        "circle_radius_m": footprint.circle_radius,
        "longitudinal_offsets_m": footprint.longitudinal_offsets.tolist(),
        "interpolation_schema": SWEPT_INTERPOLATION_SCHEMA,
        "substeps_per_original_segment": SWEPT_SUBSTEPS_PER_SEGMENT,
        "allowance_schema": FOOTPRINT_ALLOWANCE_SCHEMA,
        "allowance_policy": TRAINING_FOOTPRINT_ALLOWANCE.value,
        "grid_cell_diagonal_multiplier": TRAINING_ONE_DIAGONAL_MULTIPLIER,
        "collision_rule": "minimum_clearance_strictly_greater_than_zero",
        "qualification_evaluator": (
            "dep_car.training.losses.swept_map_footprint_clearance"
        ),
        "qualification_map_field": (
            "SignedDistanceFieldV1KnownFreePositiveUnknownUnsafe"
        ),
        "qualification_sampling": "torch_grid_sample_bilinear_align_corners_false",
        "runtime_conservative_diagnostic_evaluator": (
            "OccupancyGrid2D.swept_footprint_clearance"
        ),
    }
    payload["sha256"] = canonical_hash(payload)
    return payload


def rollout_contract():
    config = LatticeConfig()
    payload = {
        "schema": "CanonicalAckermannLattice3x5V2GearAlignedLimits",
        "candidate_count": EXPECTED_CANDIDATES,
        "candidate_order": "speed_major_3x5_steering_minor",
        "trajectory_fields": [
            "t", "x", "y", "yaw", "signed_speed", "steering"
        ],
        "lattice_config": asdict(config),
        "state_source": "StaticAckermannSampleV2.vehicle_state",
        "state_fields_used": ["signed_speed", "steering"],
        "opposite_gear_policy": "zero_speed_before_regeneration",
        "all_other_speed_policy": (
            "project_raw_signed_speed_to_requested_gear_and_clip_P0_limits"
        ),
        "steering_policy": "clip_to_P0_operating_limit",
        "longitudinal_limit_frame": "requested_gear_aligned",
    }
    payload["sha256"] = canonical_hash(payload)
    return payload


def evaluate_gates(overall, by_mode):
    overall_zero = overall.get("new", {}).get("zero_feasible_rate")
    overall_median = overall.get("new", {}).get("feasible_candidates_median")
    overall_zero_pass = (
        isinstance(overall_zero, (int, float))
        and math.isfinite(float(overall_zero))
        and float(overall_zero) < OVERALL_ZERO_FEASIBLE_LIMIT
    )
    overall_median_pass = (
        isinstance(overall_median, (int, float))
        and math.isfinite(float(overall_median))
        and float(overall_median) >= MINIMUM_MEDIAN_FEASIBLE
    )
    checks = {
        "overall_zero_feasible_rate_lt_0_10": {
            "observed": overall_zero,
            "operator": "<",
            "threshold": OVERALL_ZERO_FEASIBLE_LIMIT,
            "status": "PASS" if overall_zero_pass else "FAIL",
        },
        "overall_median_feasible_candidates_ge_2": {
            "observed": overall_median,
            "operator": ">=",
            "threshold": MINIMUM_MEDIAN_FEASIBLE,
            "status": "PASS" if overall_median_pass else "FAIL",
        },
    }
    mode_checks = {}
    for mode in PILOT_MANEUVER_MODES:
        observed = by_mode.get(mode, {}).get("new", {}).get("zero_feasible_rate")
        passed = (
            isinstance(observed, (int, float))
            and math.isfinite(float(observed))
            and float(observed) < PER_MODE_ZERO_FEASIBLE_LIMIT
        )
        mode_checks[mode] = {
            "observed": observed,
            "operator": "<",
            "threshold": PER_MODE_ZERO_FEASIBLE_LIMIT,
            "status": "PASS" if passed else "FAIL",
        }
    checks["per_mode_zero_feasible_rate_lt_0_25"] = mode_checks
    failures = [
        name for name, row in checks.items()
        if name != "per_mode_zero_feasible_rate_lt_0_25"
        and row["status"] != "PASS"
    ]
    failures.extend(
        "mode_zero_feasible_rate_" + mode
        for mode, row in mode_checks.items() if row["status"] != "PASS"
    )
    return checks, failures


def _project_relative(path):
    path = Path(path).resolve()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def run_audit(
    index_path,
    sample_root,
    maps_root,
    *,
    maximum_samples=0,
    workers=EXPECTED_WORKERS,
    enforce_frozen_authority=True,
):
    if isinstance(maximum_samples, bool) or int(maximum_samples) < 0:
        raise DevelopmentReauditError("maximum-samples cannot be negative")
    if int(workers) < 1:
        raise DevelopmentReauditError("workers must be positive")
    index = load_index_authority(
        index_path,
        sample_root,
        maps_root,
        expected_index_sha256=(
            EXPECTED_INDEX_SHA256 if enforce_frozen_authority else None
        ),
        expected_content_aggregate=(
            EXPECTED_CONTENT_AGGREGATE_SHA256 if enforce_frozen_authority else None
        ),
        expected_split_counts=(
            EXPECTED_SPLIT_COUNTS if enforce_frozen_authority else None
        ),
    )
    selected_entries = (
        index.entries[:int(maximum_samples)] if maximum_samples else index.entries
    )
    indexed_maps = {str(entry["map_uuid"]) for entry in index.entries}
    selected_maps = {str(entry["map_uuid"]) for entry in selected_entries}
    specs, map_contract = load_map_specs(
        maps_root,
        indexed_maps,
        expected_aggregate=(
            EXPECTED_MAP_CONTRACT_AGGREGATE_SHA256
            if enforce_frozen_authority
            else None
        ),
    )
    worker_specs = {map_uuid: specs[map_uuid] for map_uuid in selected_maps}

    overall_accumulator = AuditAccumulator()
    by_split = defaultdict(AuditAccumulator)
    by_mode = defaultdict(AuditAccumulator)
    maps_seen = defaultdict(set)
    sample_failures = {}
    with ProcessPoolExecutor(
        max_workers=int(workers),
        initializer=initialize_worker,
        initargs=(worker_specs,),
    ) as executor:
        tasks = ((entry, str(Path(sample_root).resolve())) for entry in selected_entries)
        results = executor.map(_inspect_entry_task, tasks, chunksize=8)
        for number, row in enumerate(results, 1):
            if "failure" in row:
                sample_failures[row["path"]] = row["failure"]
            else:
                overall_accumulator.add(row)
                by_split[row["split"]].add(row)
                by_mode[row["mode"]].add(row)
                maps_seen[row["split"]].add(row["map_uuid"])
            if number % 512 == 0 or number == len(selected_entries):
                print(
                    "reaudited %d/%d indexed development samples with %d workers"
                    % (number, len(selected_entries), workers),
                    file=sys.stderr,
                    flush=True,
                )

    overall = overall_accumulator.summary()
    split_summaries = {
        split: by_split[split].summary() for split in ("train", "validation")
    }
    mode_summaries = {
        mode: by_mode[mode].summary() for mode in PILOT_MANEUVER_MODES
    }
    gates, gate_failures = evaluate_gates(overall, mode_summaries)
    limited = bool(maximum_samples)
    operational_errors = []
    if sample_failures or overall["samples"] != len(selected_entries):
        operational_errors.append("sample_reaudit_failure")
    if file_sha256(index_path) != index.index_sha256:
        operational_errors.append("training_index_changed_during_audit")
    if not limited and len(selected_entries) != sum(index.counts_by_split.values()):
        operational_errors.append("development_index_coverage_incomplete")
    if enforce_frozen_authority and int(workers) != EXPECTED_WORKERS:
        operational_errors.append("qualification_requires_exactly_8_workers")

    if operational_errors:
        status = "FAIL"
        errors = sorted(set(operational_errors + ([] if limited else gate_failures)))
    elif limited or not enforce_frozen_authority:
        status = "SMOKE"
        errors = ["diagnostic_partial_or_nonfrozen_audit"]
    else:
        errors = sorted(set(gate_failures))
        status = "PASS" if not errors else "FAIL"

    geometry = footprint_contract()
    rollout = rollout_contract()
    implementation_paths = {
        "lattice": ROOT / "dep_car/src/dep_car/core/lattice.py",
        "occupancy": ROOT / "dep_car/src/dep_car/core/occupancy.py",
        "state_contract": ROOT / "dep_car/src/dep_car/core/state_contract.py",
        "vehicle": ROOT / "dep_car/src/dep_car/core/vehicle.py",
        "losses": ROOT / "dep_car/src/dep_car/training/losses.py",
        "p4_dataset": ROOT / "dep_car/src/dep_car/training/p4_dataset.py",
    }
    p4_implementation = build_p4_implementation_contract(ROOT)
    implementation = {
        "tool": _project_relative(__file__),
        "tool_sha256": file_sha256(__file__),
        "lattice_implementation": _project_relative(implementation_paths["lattice"]),
        "lattice_implementation_sha256": file_sha256(implementation_paths["lattice"]),
        "occupancy_implementation": _project_relative(implementation_paths["occupancy"]),
        "occupancy_implementation_sha256": file_sha256(implementation_paths["occupancy"]),
        "state_contract_implementation": _project_relative(
            implementation_paths["state_contract"]
        ),
        "state_contract_implementation_sha256": file_sha256(
            implementation_paths["state_contract"]
        ),
        "vehicle_implementation": _project_relative(implementation_paths["vehicle"]),
        "vehicle_implementation_sha256": file_sha256(implementation_paths["vehicle"]),
        "losses_implementation": _project_relative(implementation_paths["losses"]),
        "losses_implementation_sha256": file_sha256(implementation_paths["losses"]),
        "p4_dataset_implementation": _project_relative(
            implementation_paths["p4_dataset"]
        ),
        "p4_dataset_implementation_sha256": file_sha256(
            implementation_paths["p4_dataset"]
        ),
        "p4_implementation_schema": p4_implementation["schema"],
        "p4_implementation_aggregate_sha256": p4_implementation[
            "aggregate_sha256"
        ],
        "p4_implementation_files": p4_implementation["files"],
    }
    development_authority = {
        "schema": "P3DevelopmentAuthorityV1",
        "splits": ["train", "validation"],
        "index_path": _project_relative(index_path),
        "index_sha256": index.index_sha256,
        "content_aggregate_schema": TRAINING_INDEX_CONTENT_AGGREGATE_SCHEMA,
        "content_aggregate_sha256": index.content_aggregate_sha256,
        "counts_by_split": index.counts_by_split,
        "sample_count": len(index.entries),
        "map_contract_schema": map_contract["schema"],
        "map_contract_aggregate_sha256": map_contract["aggregate_sha256"],
        "map_count": map_contract["map_count"],
    }
    training_authority = {
        "schema": "P3TrainingAuthorityV1",
        "index_sha256": index.index_sha256,
        "content_aggregate_sha256": index.content_aggregate_sha256,
        "map_contract_aggregate_sha256": map_contract["aggregate_sha256"],
        "splits": ["train", "validation"],
        "test_split_used": False,
    }
    p3_provenance = load_p3_provenance()
    payload = {
        "schema": REPORT_SCHEMA,
        "status": status,
        "errors": errors,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "purpose": "P3_train_validation_development_qualification_for_P5",
            "splits": ["train", "validation"],
            "sample_inventory": "P3TrainingIndexV2.entries_only",
            "sample_tree_enumerated": False,
            "test_npz_opened": False,
            "test_map_yaml_or_png_opened": False,
            "test_split_used_for_tuning": False,
            "old_trajectory_or_feasibility_used_as_authority": False,
            "npz_files_modified": False,
            "geometry_or_gate_cli_overrides_available": False,
            "maximum_samples_is_nonqualifying": True,
        },
        "parallel_workers": int(workers),
        "qualification_eligible": bool(
            not limited and enforce_frozen_authority and int(workers) == EXPECTED_WORKERS
        ),
        "sample_files_discovered": len(index.entries),
        "sample_files_audited": len(selected_entries),
        "sample_failures": sample_failures,
        "development_authority": development_authority,
        "training_authority": training_authority,
        "index_authority": {
            "schema": EXPECTED_INDEX_SCHEMA,
            "path": _project_relative(index_path),
            "index_sha256": index.index_sha256,
            "expected_index_sha256": (
                EXPECTED_INDEX_SHA256 if enforce_frozen_authority else None
            ),
            "content_aggregate_schema": TRAINING_INDEX_CONTENT_AGGREGATE_SCHEMA,
            "content_aggregate_sha256": index.content_aggregate_sha256,
            "expected_content_aggregate_sha256": (
                EXPECTED_CONTENT_AGGREGATE_SHA256
                if enforce_frozen_authority else None
            ),
            "entries_with_verified_content_sha256": (
                len(selected_entries) - len(sample_failures)
            ),
            "counts_by_split": index.counts_by_split,
            "counts_by_mode": index.counts_by_mode,
        },
        "indexed_map_authority": map_contract,
        "maps_by_split": {
            split: len(values) for split, values in sorted(maps_seen.items())
        },
        "rollout_contract": rollout,
        "geometry_contract": geometry,
        "statistics": {
            "overall": overall,
            "by_split": split_summaries,
            "by_mode": mode_summaries,
        },
        "gates": gates,
        "P3_provenance": p3_provenance,
        "audit_implementation": implementation,
    }
    return payload


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--maximum-samples",
        type=int,
        default=0,
        help="diagnostic limit; any nonzero value permanently marks the report SMOKE",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    payload = run_audit(
        DEFAULT_INDEX,
        DEFAULT_SAMPLES,
        DEFAULT_MAPS,
        maximum_samples=args.maximum_samples,
        workers=EXPECTED_WORKERS,
        enforce_frozen_authority=True,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    overall = payload["statistics"]["overall"]["new"]
    print(json.dumps({
        "status": payload["status"],
        "report": str(args.report.resolve()),
        "samples": payload["sample_files_audited"],
        "candidates": payload["statistics"]["overall"]["candidates"],
        "zero_feasible_rate": overall["zero_feasible_rate"],
        "feasible_candidates_median": overall["feasible_candidates_median"],
        "errors": payload["errors"],
    }, indent=2, sort_keys=True))
    raise SystemExit(0 if payload["status"] in ("PASS", "SMOKE") else 1)


if __name__ == "__main__":
    main()
