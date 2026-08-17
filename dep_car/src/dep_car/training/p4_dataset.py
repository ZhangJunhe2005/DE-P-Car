"""Strict P4 training view over the immutable P3 multimodal samples.

The P3 ``StaticAckermannSampleV2`` files remain the provenance authority.  This
module derives the fixed-shape, normalized tensors used by P4 without rewriting
those files.  In particular it keeps legacy candidate labels separate from the
map/BEV distance-field authority used by differentiable losses.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import tempfile
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import cv2
import numpy as np
import torch
import yaml
from torch.utils.data import Dataset

from dep_car.core.vehicle import (
    CALIBRATED_GUARANTEED_CURVATURE_PER_M,
    STEERING_OPERATING_LIMIT_RAD,
)
from dep_car.core.state_contract import (
    ACCELERATION_LIMIT_MPS2,
    DECELERATION_LIMIT_MPS2,
    FORWARD_SPEED_LIMIT_MPS,
    REVERSE_SPEED_LIMIT_MPS,
    STATE_NORMALIZATION_SCALE,
    SUBGOAL_SCALE_M,
    YAW_RATE_SCALE_RADPS,
)
from dep_car.training.dataset import (
    MANEUVER_MODES,
    MULTIMODAL_CONTRACT_REVISION,
    MULTIMODAL_SCHEMA_VERSION,
    map_split,
)


TRAINING_VIEW_SCHEMA = "P3TrainingDatasetV1"
TRAINING_INDEX_SCHEMA = "P3TrainingIndexV2"
TRAINING_INDEX_CONTENT_HASH_ALGORITHM = "sha256"
TRAINING_INDEX_CONTENT_AGGREGATE_SCHEMA = "SortedRelativePathSha256PairsV1"
SIGNED_SDF_SCHEMA = "SignedDistanceFieldV1KnownFreePositiveUnknownUnsafe"
EXPECTED_SENSOR_AUTHORITY = "urban_car_depth_vlp16_sim"
EXPECTED_BEV_PREPROCESSING_SHA256 = (
    "89be32ba8f15fce9ff85332070c4da74668d4853a7419dd986a0d8bcafb3fb7b"
)
DEPTH_MAXIMUM_M = 10.0


# Loading an index is a fail-closed content verification operation.  A process
# may reuse a successful verification only while both the index identity and
# every sample's full stat identity remain unchanged.  In particular, ctime
# and inode are included so restoring only size/mtime cannot hit this cache.
_INDEX_CONTENT_VALIDATION_CACHE = {}
_INDEX_CONTENT_VALIDATION_CACHE_LOCK = threading.Lock()
_INDEX_CONTENT_VALIDATION_CACHE_MAXIMUM = 8


class P3TrainingDataError(ValueError):
    """Raised when P3 provenance cannot produce the frozen P4 view."""


@dataclass(frozen=True)
class MapAuthority:
    map_uuid: str
    split: str
    folder: Path
    yaml_path: Path
    image: Path
    image_name: str
    resolution_m: float
    origin: Tuple[float, ...]
    origin_xy: Tuple[float, float]
    occupancy_sha256: str
    png_sha256: str
    yaml_semantic_sha256: str
    negate: int
    occupied_threshold: float
    free_threshold: float


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_sha256(value) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _read_exact_bytes(path: Path, description: str) -> bytes:
    try:
        return Path(path).read_bytes()
    except OSError as exc:
        raise P3TrainingDataError(
            "unable to read %s %s: %s" % (description, path, exc)
        ) from exc


def _decode_grayscale_png(image_bytes: bytes, path: Path) -> np.ndarray:
    encoded = np.frombuffer(image_bytes, dtype=np.uint8)
    pixels = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
    if pixels is None:
        raise P3TrainingDataError("map image is unreadable: %s" % path)
    return np.asarray(pixels, dtype=np.uint8)


def _map_yaml_semantics(metadata: Mapping, yaml_path: Path) -> dict:
    if not isinstance(metadata, dict):
        raise P3TrainingDataError("map.yaml root must be a mapping: %s" % yaml_path)
    try:
        image_name = str(metadata["image"])
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
        raise P3TrainingDataError(
            "map metric metadata is invalid: %s" % yaml_path
        ) from exc
    if (
        not image_name
        or len(origin) < 3
        or not all(np.isfinite(value) for value in origin)
        or abs(float(origin[2])) > 1.0e-9
        or not np.isfinite(semantic["resolution"])
        or semantic["resolution"] <= 0.0
        or semantic["negate"] not in (0, 1)
        or not np.isfinite(semantic["occupied_thresh"])
        or not np.isfinite(semantic["free_thresh"])
        or not 0.0
        <= semantic["free_thresh"]
        < semantic["occupied_thresh"]
        <= 1.0
    ):
        raise P3TrainingDataError(
            "map metric metadata is invalid (origin yaw must be zero): %s"
            % yaml_path
        )
    return semantic


def _map_contract_row(authority: MapAuthority) -> dict:
    return {
        "map_uuid": authority.map_uuid,
        "png_sha256": authority.png_sha256,
        "occupancy_sha256": authority.occupancy_sha256,
        "yaml": {
            "image": authority.image_name,
            "resolution": authority.resolution_m,
            "origin": list(authority.origin),
            "negate": authority.negate,
            "occupied_thresh": authority.occupied_threshold,
            "free_thresh": authority.free_threshold,
        },
    }


def indexed_map_contract_aggregate(maps: Mapping[str, MapAuthority]) -> dict:
    """Return the same closed map identity consumed by the P5 trainer.

    The identity covers exact PNG bytes, decoded grayscale occupancy and all
    ROS YAML values that affect metric map interpretation.
    """

    rows = [_map_contract_row(maps[map_uuid]) for map_uuid in sorted(maps)]
    if not rows:
        raise P3TrainingDataError("indexed map contract is empty")
    return {
        "schema": "IndexedMapContractAggregateV1",
        "map_count": len(rows),
        "aggregate_sha256": _canonical_sha256(rows),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as stream:
            for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise P3TrainingDataError("unable to hash sample %s: %s" % (path, exc)) from exc
    return digest.hexdigest()


def _stat_identity(stat_result) -> tuple:
    return (
        int(stat_result.st_dev),
        int(stat_result.st_ino),
        int(stat_result.st_size),
        int(stat_result.st_mtime_ns),
        int(stat_result.st_ctime_ns),
    )


def _is_sha256(value) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def training_index_content_aggregate(entries: Sequence[Mapping]) -> str:
    """Hash only sorted ``(relative path, content SHA-256)`` pairs.

    The aggregate intentionally excludes timestamps, absolute roots, worker
    counts, and all derived metadata.  It therefore identifies sample bytes
    stably across rebuilds and workspace relocation.
    """

    pairs = []
    for entry in entries:
        relative = str(entry.get("path", ""))
        content_sha256 = entry.get("content_sha256")
        if not relative or not _is_sha256(content_sha256):
            raise P3TrainingDataError(
                "content aggregate requires path and lowercase SHA-256 per entry"
            )
        pairs.append((relative, content_sha256))
    pairs.sort()
    canonical = json.dumps(
        pairs, ensure_ascii=True, separators=(",", ":")
    ).encode("utf-8")
    return _sha256_bytes(canonical)


def _read_npz_manifest(path: Path) -> dict:
    try:
        with np.load(str(path), allow_pickle=False) as data:
            return json.loads(str(data["manifest_json"]))
    except Exception as exc:
        raise P3TrainingDataError(
            "unable to read sample manifest %s: %s" % (path, exc)
        ) from exc


def _load_map_catalog(
    maps_root: Path, selected_map_uuids: Iterable[str] = None
) -> Dict[str, MapAuthority]:
    maps_root = Path(maps_root).resolve()
    if not maps_root.is_dir():
        raise P3TrainingDataError("map root does not exist: %s" % maps_root)
    selected = (
        None
        if selected_map_uuids is None
        else {str(value) for value in selected_map_uuids}
    )
    if selected is not None and not selected:
        raise P3TrainingDataError("selected map authority set is empty")
    folders = sorted(path for path in maps_root.iterdir() if path.is_dir())
    if selected is not None:
        # Locate selected development maps from the UUID suffix embedded in
        # generated folder names.  Do not open test-map YAML/PNG merely to
        # discover that they are outside the training view.
        prefixes = {value[:8] for value in selected}
        folders = [
            folder
            for folder in folders
            if any(prefix in folder.name for prefix in prefixes)
        ]
    output: Dict[str, MapAuthority] = {}
    for folder in folders:
        manifest_path = folder / "manifest.json"
        yaml_path = folder / "map.yaml"
        if not manifest_path.is_file():
            continue
        if not yaml_path.is_file():
            raise P3TrainingDataError("map.yaml is missing: %s" % folder)
        try:
            manifest = json.loads(
                _read_exact_bytes(manifest_path, "map manifest").decode("utf-8")
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise P3TrainingDataError(
                "map manifest is invalid: %s" % manifest_path
            ) from exc
        map_uuid = str(manifest.get("map_uuid", ""))
        if selected is not None and map_uuid not in selected:
            continue
        if not map_uuid:
            raise P3TrainingDataError("map UUID is missing: %s" % manifest_path)
        split = map_split(map_uuid)
        yaml_bytes = _read_exact_bytes(yaml_path, "map YAML")
        try:
            metadata = yaml.safe_load(yaml_bytes.decode("utf-8"))
        except (UnicodeDecodeError, yaml.YAMLError) as exc:
            raise P3TrainingDataError("map.yaml is invalid: %s" % yaml_path) from exc
        semantic = _map_yaml_semantics(metadata, yaml_path)
        image = (folder / semantic["image"]).resolve()
        if image.parent != folder.resolve() or not image.is_file():
            raise P3TrainingDataError("map image is outside its authority folder: %s" % image)
        image_bytes = _read_exact_bytes(image, "map PNG")
        pixels = _decode_grayscale_png(image_bytes, image)
        actual_hash = _sha256_bytes(np.asarray(pixels, dtype=np.uint8).tobytes())
        claimed_hash = str(manifest.get("occupancy_sha256", ""))
        if actual_hash != claimed_hash:
            raise P3TrainingDataError("map occupancy hash mismatch: %s" % folder)
        if map_uuid in output:
            raise P3TrainingDataError("duplicate map UUID: %s" % map_uuid)
        origin = tuple(semantic["origin"])
        resolution = float(semantic["resolution"])
        negate = int(semantic["negate"])
        occupied_threshold = float(semantic["occupied_thresh"])
        free_threshold = float(semantic["free_thresh"])
        output[map_uuid] = MapAuthority(
            map_uuid=map_uuid,
            split=split,
            folder=folder.resolve(),
            yaml_path=yaml_path.resolve(),
            image=image,
            image_name=semantic["image"],
            resolution_m=resolution,
            origin=origin,
            origin_xy=(float(origin[0]), float(origin[1])),
            occupancy_sha256=claimed_hash,
            png_sha256=_sha256_bytes(image_bytes),
            yaml_semantic_sha256=_canonical_sha256(semantic),
            negate=negate,
            occupied_threshold=occupied_threshold,
            free_threshold=free_threshold,
        )
    if selected is not None and set(output) != selected:
        missing = sorted(selected.difference(output))
        raise P3TrainingDataError(
            "selected map authorities are missing: %s" % ", ".join(missing[:3])
        )
    if not output:
        raise P3TrainingDataError("no map authorities found under %s" % maps_root)
    return output


def _validate_requested_splits(splits: Iterable[str], allow_test: bool) -> Tuple[str, ...]:
    values = tuple(dict.fromkeys(str(value) for value in splits))
    if not values or any(value not in ("train", "validation", "test") for value in values):
        raise P3TrainingDataError("splits must be drawn from train/validation/test")
    if "test" in values and not allow_test:
        raise P3TrainingDataError(
            "test split is sealed; pass allow_test=True only for final evaluation"
        )
    return values


def _index_one_sample(
    path: Path,
    sample_root: Path,
    maps: Mapping[str, MapAuthority],
    expected_sensor_authority: str,
    expected_preprocessing_sha256: str,
) -> dict:
    try:
        initial_stat = path.stat()
    except OSError as exc:
        raise P3TrainingDataError("unable to stat sample %s: %s" % (path, exc)) from exc
    manifest = _read_npz_manifest(path)
    if manifest.get("schema") != MULTIMODAL_SCHEMA_VERSION:
        raise P3TrainingDataError("sample schema mismatch: %s" % path)
    if manifest.get("contract_revision") != MULTIMODAL_CONTRACT_REVISION:
        raise P3TrainingDataError("sample contract revision mismatch: %s" % path)
    map_uuid = str(manifest.get("map_uuid", ""))
    if map_uuid not in maps:
        raise P3TrainingDataError("sample map UUID has no authority: %s" % path)
    split = str(manifest.get("split", ""))
    if split != map_split(map_uuid) or split != maps[map_uuid].split:
        raise P3TrainingDataError("sample split is not map-UUID authoritative: %s" % path)
    if path.parent.name != map_uuid:
        raise P3TrainingDataError("sample folder does not equal map UUID: %s" % path)
    if manifest.get("sensor_authority") != expected_sensor_authority:
        raise P3TrainingDataError("sensor authority mismatch: %s" % path)
    preprocessing_hash = (
        manifest.get("preprocessing", {}).get("lidar_bev", {}).get("sha256", "")
    )
    if preprocessing_hash != expected_preprocessing_sha256:
        raise P3TrainingDataError("BEV preprocessing hash mismatch: %s" % path)
    metadata = manifest.get("metadata", {})
    if not bool(metadata.get("formal_training_authority", False)):
        raise P3TrainingDataError("sample is not formal training authority: %s" % path)
    if metadata.get("map_occupancy_sha256") != maps[map_uuid].occupancy_sha256:
        raise P3TrainingDataError("sample/map occupancy hash mismatch: %s" % path)
    mode = str(manifest.get("maneuver_mode", ""))
    if mode not in MANEUVER_MODES:
        raise P3TrainingDataError("invalid maneuver mode: %s" % path)
    content_sha256 = _sha256_file(path)
    try:
        stat = path.stat()
    except OSError as exc:
        raise P3TrainingDataError("unable to stat sample %s: %s" % (path, exc)) from exc
    if _stat_identity(initial_stat) != _stat_identity(stat):
        raise P3TrainingDataError("sample changed while indexing: %s" % path)
    return {
        "path": str(path.relative_to(sample_root)),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "content_sha256": content_sha256,
        "map_uuid": map_uuid,
        "split": split,
        "maneuver_mode": mode,
        "sample_id": str(metadata.get("sample_id", path.stem)),
        "task_id": str(metadata.get("pilot_task_id", "")),
        "candidate_context": str(metadata.get("candidate_context", "UNKNOWN")),
    }


def build_training_index(
    sample_root,
    maps_root,
    output_path,
    *,
    splits=("train", "validation"),
    workers=8,
    allow_test=False,
    expected_sensor_authority=EXPECTED_SENSOR_AUTHORITY,
    expected_preprocessing_sha256=EXPECTED_BEV_PREPROCESSING_SHA256,
):
    """Build an atomic, content-authoritative index without opening sealed samples.

    Map folders determine the split before any NPZ is opened, so the default
    train/validation build does not inspect test sample contents.
    """

    sample_root = Path(sample_root).resolve()
    maps_root = Path(maps_root).resolve()
    output_path = Path(output_path).resolve()
    splits = _validate_requested_splits(splits, allow_test)
    workers = int(workers)
    if workers < 1:
        raise P3TrainingDataError("workers must be positive")
    if not sample_root.is_dir():
        raise P3TrainingDataError("sample root does not exist: %s" % sample_root)

    paths = []
    selected_map_uuids = set()
    for folder in sorted(path for path in sample_root.iterdir() if path.is_dir()):
        # The sample folder is the full map UUID, so its split can be decided
        # before opening any NPZ or any sealed test-map geometry.
        if map_split(folder.name) not in splits:
            continue
        folder_paths = sorted(folder.glob("*.npz"))
        if folder_paths:
            selected_map_uuids.add(folder.name)
            paths.extend(folder_paths)
    if not paths:
        raise P3TrainingDataError("no samples found for splits %s" % (splits,))
    maps = _load_map_catalog(maps_root, selected_map_uuids)

    def inspect(path):
        return _index_one_sample(
            path,
            sample_root,
            maps,
            expected_sensor_authority,
            expected_preprocessing_sha256,
        )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        entries = list(pool.map(inspect, paths))
    entries.sort(key=lambda item: (item["split"], item["map_uuid"], item["path"]))
    split_counts = Counter(item["split"] for item in entries)
    mode_counts = Counter(item["maneuver_mode"] for item in entries)
    payload = {
        "schema": TRAINING_INDEX_SCHEMA,
        "training_view": TRAINING_VIEW_SCHEMA,
        "content_hash_algorithm": TRAINING_INDEX_CONTENT_HASH_ALGORITHM,
        "content_aggregate_schema": TRAINING_INDEX_CONTENT_AGGREGATE_SCHEMA,
        "content_aggregate_sha256": training_index_content_aggregate(entries),
        "created_at_unix": time.time(),
        "sample_root": str(sample_root),
        "maps_root": str(maps_root),
        "splits": list(splits),
        "workers": workers,
        "sensor_authority": expected_sensor_authority,
        "bev_preprocessing_sha256": expected_preprocessing_sha256,
        "samples": len(entries),
        "counts_by_split": dict(sorted(split_counts.items())),
        "counts_by_mode": dict(sorted(mode_counts.items())),
        "entries": entries,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=output_path.name + ".", suffix=".tmp", dir=str(output_path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary_name, output_path)
        os.chmod(output_path, 0o644)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return payload


def load_training_index(
    path,
    *,
    sample_root,
    maps_root,
    splits=("train", "validation"),
    workers=8,
    allow_test=False,
    expected_sensor_authority=EXPECTED_SENSOR_AUTHORITY,
    expected_preprocessing_sha256=EXPECTED_BEV_PREPROCESSING_SHA256,
    expected_index_sha256=None,
    return_index_sha256=False,
):
    path = Path(path).resolve()
    sample_root = Path(sample_root).resolve()
    maps_root = Path(maps_root).resolve()
    splits = _validate_requested_splits(splits, allow_test)
    workers = int(workers)
    if workers < 1:
        raise P3TrainingDataError("workers must be positive")
    try:
        index_stat_before = path.stat()
        index_bytes = path.read_bytes()
        actual_index_sha256 = _sha256_bytes(index_bytes)
        if expected_index_sha256 is not None:
            if not _is_sha256(expected_index_sha256):
                raise P3TrainingDataError(
                    "expected index SHA-256 must be a lowercase SHA-256"
                )
            if actual_index_sha256 != expected_index_sha256:
                raise P3TrainingDataError(
                    "training index changed after trainer validation: %s" % path
                )
        payload = json.loads(index_bytes.decode("utf-8"))
        index_stat_after = path.stat()
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise P3TrainingDataError("unable to read training index %s: %s" % (path, exc)) from exc
    if _stat_identity(index_stat_before) != _stat_identity(index_stat_after):
        raise P3TrainingDataError("training index changed while loading: %s" % path)
    expected = {
        "schema": TRAINING_INDEX_SCHEMA,
        "training_view": TRAINING_VIEW_SCHEMA,
        "content_hash_algorithm": TRAINING_INDEX_CONTENT_HASH_ALGORITHM,
        "content_aggregate_schema": TRAINING_INDEX_CONTENT_AGGREGATE_SCHEMA,
        "sample_root": str(sample_root),
        "maps_root": str(maps_root),
        "splits": list(splits),
        "sensor_authority": expected_sensor_authority,
        "bev_preprocessing_sha256": expected_preprocessing_sha256,
    }
    mismatches = [key for key, value in expected.items() if payload.get(key) != value]
    if mismatches:
        raise P3TrainingDataError(
            "training index contract mismatch (%s): %s"
            % (", ".join(mismatches), path)
        )
    entries = payload.get("entries", [])
    if not isinstance(entries, list) or payload.get("samples") != len(entries) or not entries:
        raise P3TrainingDataError("training index is empty or truncated: %s" % path)
    declared_aggregate = payload.get("content_aggregate_sha256")
    if not _is_sha256(declared_aggregate):
        raise P3TrainingDataError("training index content aggregate is missing or invalid")

    seen_paths = set()
    files = []
    sample_stats = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise P3TrainingDataError("training index entry is not a mapping")
        relative = Path(str(entry.get("path", "")))
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or len(relative.parts) != 2
            or relative.suffix != ".npz"
        ):
            raise P3TrainingDataError("unsafe sample path in training index")
        relative_text = str(relative)
        if relative_text in seen_paths:
            raise P3TrainingDataError("duplicate sample path in training index")
        seen_paths.add(relative_text)
        split = str(entry.get("split", ""))
        map_uuid = str(entry.get("map_uuid", ""))
        maneuver_mode = str(entry.get("maneuver_mode", ""))
        if split not in splits:
            raise P3TrainingDataError("unexpected split in training index")
        if not map_uuid or relative.parts[0] != map_uuid or map_split(map_uuid) != split:
            raise P3TrainingDataError("sample split/path is not map-UUID authoritative")
        if maneuver_mode not in MANEUVER_MODES:
            raise P3TrainingDataError("invalid maneuver mode in training index")
        content_sha256 = entry.get("content_sha256")
        if not _is_sha256(content_sha256):
            raise P3TrainingDataError("invalid sample content SHA-256 in training index")
        try:
            size_bytes = int(entry["size_bytes"])
            mtime_ns = int(entry["mtime_ns"])
        except (KeyError, TypeError, ValueError) as exc:
            raise P3TrainingDataError("invalid sample stat authority in training index") from exc
        if size_bytes < 0 or mtime_ns < 0:
            raise P3TrainingDataError("invalid sample stat authority in training index")

        sample_path = (sample_root / relative).resolve()
        authoritative_folder = (sample_root / map_uuid).resolve()
        if (
            sample_root not in sample_path.parents
            or sample_path.parent != authoritative_folder
            or sample_path.parent.name != map_uuid
        ):
            raise P3TrainingDataError("sample path escapes its map authority folder")
        try:
            stat = sample_path.stat()
        except OSError as exc:
            raise P3TrainingDataError(
                "indexed sample is unavailable: %s" % sample_path
            ) from exc
        if not sample_path.is_file():
            raise P3TrainingDataError("indexed sample is unavailable: %s" % sample_path)
        if int(stat.st_size) != size_bytes or int(stat.st_mtime_ns) != mtime_ns:
            raise P3TrainingDataError("indexed sample stat mismatch: %s" % sample_path)
        identity = _stat_identity(stat)
        files.append((sample_path, relative_text, content_sha256, identity))
        sample_stats.append((relative_text, identity))

    actual_split_counts = dict(sorted(Counter(str(entry["split"]) for entry in entries).items()))
    actual_mode_counts = dict(
        sorted(Counter(str(entry.get("maneuver_mode", "")) for entry in entries).items())
    )
    if payload.get("counts_by_split") != actual_split_counts:
        raise P3TrainingDataError("training index split counts are inconsistent")
    if payload.get("counts_by_mode") != actual_mode_counts:
        raise P3TrainingDataError("training index maneuver counts are inconsistent")
    if training_index_content_aggregate(entries) != declared_aggregate:
        raise P3TrainingDataError("training index content aggregate mismatch")

    sample_stats_sha256 = _sha256_bytes(json.dumps(
        sample_stats, ensure_ascii=True, separators=(",", ":")
    ).encode("utf-8"))
    cache_key = (
        str(path),
        _stat_identity(index_stat_after),
        declared_aggregate,
        sample_stats_sha256,
    )
    with _INDEX_CONTENT_VALIDATION_CACHE_LOCK:
        content_already_verified = cache_key in _INDEX_CONTENT_VALIDATION_CACHE
    if not content_already_verified:
        def verify_content(file_authority):
            sample_path, _relative_text, expected_sha256, expected_identity = file_authority
            try:
                stat_before = sample_path.stat()
            except OSError as exc:
                raise P3TrainingDataError(
                    "indexed sample is unavailable: %s" % sample_path
                ) from exc
            if _stat_identity(stat_before) != expected_identity:
                raise P3TrainingDataError("indexed sample changed before hashing: %s" % sample_path)
            actual_sha256 = _sha256_file(sample_path)
            try:
                stat_after = sample_path.stat()
            except OSError as exc:
                raise P3TrainingDataError(
                    "indexed sample is unavailable: %s" % sample_path
                ) from exc
            if _stat_identity(stat_after) != expected_identity:
                raise P3TrainingDataError("indexed sample changed while hashing: %s" % sample_path)
            if actual_sha256 != expected_sha256:
                raise P3TrainingDataError("indexed sample content hash mismatch: %s" % sample_path)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(verify_content, files))
        for sample_path, _relative_text, _expected_sha256, expected_identity in files:
            try:
                current_identity = _stat_identity(sample_path.stat())
            except OSError as exc:
                raise P3TrainingDataError(
                    "indexed sample is unavailable: %s" % sample_path
                ) from exc
            if current_identity != expected_identity:
                raise P3TrainingDataError(
                    "indexed sample changed during content validation: %s" % sample_path
                )
        with _INDEX_CONTENT_VALIDATION_CACHE_LOCK:
            if len(_INDEX_CONTENT_VALIDATION_CACHE) >= _INDEX_CONTENT_VALIDATION_CACHE_MAXIMUM:
                _INDEX_CONTENT_VALIDATION_CACHE.clear()
            _INDEX_CONTENT_VALIDATION_CACHE[cache_key] = True
    try:
        final_index_identity = _stat_identity(path.stat())
    except OSError as exc:
        raise P3TrainingDataError("training index became unavailable: %s" % path) from exc
    if final_index_identity != _stat_identity(index_stat_after):
        raise P3TrainingDataError("training index changed during validation: %s" % path)
    if return_index_sha256:
        return payload, actual_index_sha256
    return payload


def load_or_build_training_index(
    path,
    *,
    sample_root,
    maps_root,
    splits=("train", "validation"),
    workers=8,
    rebuild=False,
    allow_test=False,
    expected_sensor_authority=EXPECTED_SENSOR_AUTHORITY,
    expected_preprocessing_sha256=EXPECTED_BEV_PREPROCESSING_SHA256,
    expected_index_sha256=None,
):
    path = Path(path)
    if path.is_file() and not rebuild:
        return load_training_index(
            path,
            sample_root=sample_root,
            maps_root=maps_root,
            splits=splits,
            workers=workers,
            allow_test=allow_test,
            expected_sensor_authority=expected_sensor_authority,
            expected_preprocessing_sha256=expected_preprocessing_sha256,
            expected_index_sha256=expected_index_sha256,
        )
    if expected_index_sha256 is not None:
        raise P3TrainingDataError(
            "cannot rebuild an index while enforcing a trainer-sealed index SHA-256"
        )
    return build_training_index(
        sample_root,
        maps_root,
        path,
        splits=splits,
        workers=workers,
        allow_test=allow_test,
        expected_sensor_authority=expected_sensor_authority,
        expected_preprocessing_sha256=expected_preprocessing_sha256,
    )


def _wrap_angle(value):
    return math.atan2(math.sin(float(value)), math.cos(float(value)))


def _active_same_gear_route(path, gears, requested_gear, subgoal, heading_error):
    indices = np.flatnonzero(gears == int(requested_gear))
    if len(indices):
        start = int(indices[0])
        stop = start
        while stop < len(gears) and int(gears[stop]) == int(requested_gear):
            stop += 1
        active = path[start:stop].copy()
    else:
        active = np.empty((0, 3), dtype=np.float32)
    if len(active) < 2:
        active = np.asarray(
            [[0.0, 0.0, 0.0], [float(subgoal[0]), float(subgoal[1]), heading_error]],
            dtype=np.float32,
        )
    active[:, 2] = np.asarray([_wrap_angle(value) for value in active[:, 2]], dtype=np.float32)
    return active, np.full(len(active), int(requested_gear), dtype=np.int8)


def _reference_curvature(active_route):
    route = np.asarray(active_route, dtype=np.float32)
    if len(route) < 2:
        return 0.0
    maximum_end = min(len(route) - 1, 2)
    for end in range(maximum_end, 0, -1):
        distance = float(np.linalg.norm(route[end, :2] - route[0, :2]))
        if distance > 1e-3:
            yaw_delta = _wrap_angle(route[end, 2] - route[0, 2])
            return float(np.clip(
                yaw_delta / distance,
                -CALIBRATED_GUARANTEED_CURVATURE_PER_M,
                CALIBRATED_GUARANTEED_CURVATURE_PER_M,
            ))
    return 0.0


def _signed_distance_field(known_free, resolution_m):
    """Return a metric SDF with known free positive and unsafe space negative.

    Unsafe means either occupied or unknown.  Using a second distance transform
    inside unsafe regions is essential: an unsigned obstacle-distance field is
    flat at zero inside a thick obstacle and therefore cannot provide a gradient
    that moves a colliding candidate back toward known free space.
    """

    known_free = np.asarray(known_free, dtype=bool)
    resolution_m = float(resolution_m)
    if known_free.ndim != 2 or not np.isfinite(resolution_m) or resolution_m <= 0.0:
        raise P3TrainingDataError("signed distance-field geometry is invalid")
    unsafe = ~known_free
    maximum_distance = float(np.hypot(*known_free.shape) * resolution_m)
    if not np.any(known_free):
        return np.full(known_free.shape, -maximum_distance, dtype=np.float32)
    if not np.any(unsafe):
        return np.full(known_free.shape, maximum_distance, dtype=np.float32)
    distance_in_free = cv2.distanceTransform(
        known_free.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE
    )
    distance_in_unsafe = cv2.distanceTransform(
        unsafe.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE
    )
    return ((distance_in_free - distance_in_unsafe) * resolution_m).astype(np.float32)


def _ros_map_known_free(pixels, authority):
    """Apply ROS trinary-map thresholds; unknown is deliberately not free."""

    pixels = np.asarray(pixels, dtype=np.float32)
    occupancy_probability = pixels / 255.0
    if not authority.negate:
        occupancy_probability = 1.0 - occupancy_probability
    # ROS trinary semantics use strict comparisons.  Values between the two
    # thresholds are unknown and join occupied cells in the unsafe set.
    return occupancy_probability < float(authority.free_threshold)


def _bev_distance_field(bev, resolution_m):
    occupied = np.asarray(bev[0] >= 0.5, dtype=bool)
    observed = np.asarray(bev[5] >= 0.5, dtype=bool)
    known_free = (~occupied) & observed
    # Unobserved BEV cells remain unsafe.  This is the frozen P0 policy:
    # unknown space is occupied except where a LiDAR ray established free space.
    return _signed_distance_field(known_free, resolution_m)


def _normalize_bev(bev, preprocessing):
    output = np.asarray(bev, dtype=np.float32).copy()
    output[0] = np.clip(output[0], 0.0, 1.0)
    output[1] = np.clip(output[1], 0.0, 1.0)
    obstacle = preprocessing.get("obstacle_filter", {})
    minimum = float(obstacle.get("minimum_height", 0.05))
    maximum = float(obstacle.get("maximum_height", 1.30))
    if maximum <= minimum:
        raise P3TrainingDataError("invalid BEV height normalization interval")
    occupied = output[0] >= 0.5
    for channel in (2, 3):
        normalized = np.clip((output[channel] - minimum) / (maximum - minimum), 0.0, 1.0)
        output[channel] = np.where(occupied, normalized, 0.0)
    output[4] = np.clip(output[4], 0.0, 1.0)
    output[5] = np.clip(output[5], 0.0, 1.0)
    return output


class P3TrainingDatasetV1(Dataset):
    """Fixed-shape P4 tensors derived from one map-isolated P3 split."""

    def __init__(
        self,
        sample_root,
        maps_root,
        *,
        split="train",
        index_path=None,
        index_splits=None,
        workers=8,
        rebuild_index=False,
        allow_test=False,
        maximum_route_points=80,
        maximum_candidate_steps=15,
        depth_dropout_probability=0.0,
        lidar_dropout_probability=0.0,
        augmentation_seed=49004,
        expected_sensor_authority=EXPECTED_SENSOR_AUTHORITY,
        expected_preprocessing_sha256=EXPECTED_BEV_PREPROCESSING_SHA256,
        expected_map_contract_aggregate_sha256=None,
        expected_index_sha256=None,
    ):
        self.sample_root = Path(sample_root).resolve()
        self.maps_root = Path(maps_root).resolve()
        self.split = str(split)
        _validate_requested_splits((self.split,), allow_test)
        if index_splits is None:
            index_splits = ("test",) if self.split == "test" else ("train", "validation")
        self.index_splits = _validate_requested_splits(index_splits, allow_test)
        if self.split not in self.index_splits:
            raise P3TrainingDataError("dataset split is absent from the requested index")
        default_name = "test_index.json" if self.split == "test" else "training_index.json"
        self.index_path = Path(index_path or (self.sample_root.parent / default_name)).resolve()
        self.maximum_route_points = int(maximum_route_points)
        self.maximum_candidate_steps = int(maximum_candidate_steps)
        if self.maximum_route_points < 2 or self.maximum_candidate_steps < 2:
            raise P3TrainingDataError("padding limits are too small")
        self.depth_dropout_probability = float(depth_dropout_probability)
        self.lidar_dropout_probability = float(lidar_dropout_probability)
        if (
            self.depth_dropout_probability < 0.0
            or self.lidar_dropout_probability < 0.0
            or self.depth_dropout_probability + self.lidar_dropout_probability > 1.0
        ):
            raise P3TrainingDataError(
                "modality dropout probabilities must be non-negative and sum to <= 1"
            )
        self.augmentation_seed = int(augmentation_seed)
        self.epoch = 0
        self.expected_sensor_authority = expected_sensor_authority
        self.expected_preprocessing_sha256 = expected_preprocessing_sha256
        payload = load_or_build_training_index(
            self.index_path,
            sample_root=self.sample_root,
            maps_root=self.maps_root,
            splits=self.index_splits,
            workers=workers,
            rebuild=rebuild_index,
            allow_test=allow_test,
            expected_sensor_authority=expected_sensor_authority,
            expected_preprocessing_sha256=expected_preprocessing_sha256,
            expected_index_sha256=expected_index_sha256,
        )
        self.entries = [entry for entry in payload["entries"] if entry["split"] == self.split]
        if not self.entries:
            raise P3TrainingDataError("index contains no %s samples" % self.split)
        # Both train and validation instances bind the complete development map
        # set from the same sealed index.  This makes a single trainer-computed
        # aggregate authoritative for both datasets and never discovers/opens
        # test maps outside that index.
        self.maps = _load_map_catalog(
            self.maps_root, {entry["map_uuid"] for entry in payload["entries"]}
        )
        self.map_contract = indexed_map_contract_aggregate(self.maps)
        if expected_map_contract_aggregate_sha256 is not None:
            expected_map_contract_aggregate_sha256 = str(
                expected_map_contract_aggregate_sha256
            )
            if not _is_sha256(expected_map_contract_aggregate_sha256):
                raise P3TrainingDataError(
                    "expected map contract aggregate must be a lowercase SHA-256"
                )
            if (
                self.map_contract["aggregate_sha256"]
                != expected_map_contract_aggregate_sha256
            ):
                raise P3TrainingDataError(
                    "indexed map contract changed after trainer validation"
                )
        self.expected_map_contract_aggregate_sha256 = (
            expected_map_contract_aggregate_sha256
        )
        self.expected_index_sha256 = expected_index_sha256
        self._map_cache = {}

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_map_cache"] = {}
        return state

    def __len__(self):
        return len(self.entries)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def _dropout_modality(self, sample_id):
        if self.depth_dropout_probability + self.lidar_dropout_probability == 0.0:
            return "none"
        token = "%d:%d:%s" % (self.augmentation_seed, self.epoch, sample_id)
        value = int(hashlib.sha256(token.encode("utf-8")).hexdigest()[:16], 16) / float(16 ** 16)
        if value < self.depth_dropout_probability:
            return "depth"
        if value < self.depth_dropout_probability + self.lidar_dropout_probability:
            return "lidar"
        return "none"

    def _map_tensors(self, map_uuid):
        cached = self._map_cache.get(map_uuid)
        if cached is not None:
            return cached
        authority = self.maps[map_uuid]
        # Re-read exact bytes at the consumption boundary.  Decoding the byte
        # sequence whose raw SHA was checked avoids a validate(path)->read(path)
        # race, while the decoded hash protects the occupancy representation.
        yaml_bytes = _read_exact_bytes(authority.yaml_path, "map YAML")
        try:
            metadata = yaml.safe_load(yaml_bytes.decode("utf-8"))
        except (UnicodeDecodeError, yaml.YAMLError) as exc:
            raise P3TrainingDataError(
                "map YAML became invalid: %s" % authority.yaml_path
            ) from exc
        semantic = _map_yaml_semantics(metadata, authority.yaml_path)
        if _canonical_sha256(semantic) != authority.yaml_semantic_sha256:
            raise P3TrainingDataError(
                "map YAML semantics changed after validation: %s"
                % authority.yaml_path
            )
        image_path = (authority.folder / semantic["image"]).resolve()
        if image_path != authority.image:
            raise P3TrainingDataError(
                "map image authority changed after validation: %s" % authority.yaml_path
            )
        image_bytes = _read_exact_bytes(image_path, "map PNG")
        if _sha256_bytes(image_bytes) != authority.png_sha256:
            raise P3TrainingDataError(
                "map PNG bytes changed after validation: %s" % image_path
            )
        image = _decode_grayscale_png(image_bytes, image_path)
        if _sha256_bytes(image.tobytes()) != authority.occupancy_sha256:
            raise P3TrainingDataError(
                "map decoded occupancy changed after validation: %s" % image_path
            )
        # PNG row zero is the top of the image, whereas ROS occupancy row zero
        # starts at the map origin (lower left).  Threshold before flipping and
        # treat both occupied and trinary-unknown pixels as unsafe.
        known_free = np.flipud(_ros_map_known_free(image, authority)).copy()
        unsafe = ~known_free
        distance = _signed_distance_field(known_free, authority.resolution_m)
        cached = {
            "map_occupancy": torch.from_numpy(unsafe.astype(np.bool_)),
            "map_distance_field": torch.from_numpy(distance[None]),
            "map_resolution": torch.tensor(authority.resolution_m, dtype=torch.float32),
            "map_origin": torch.tensor(authority.origin_xy, dtype=torch.float32),
        }
        self._map_cache[map_uuid] = cached
        return cached

    def __getitem__(self, index):
        entry = self.entries[int(index)]
        path = (self.sample_root / entry["path"]).resolve()
        if self.sample_root not in path.parents or not path.is_file():
            raise P3TrainingDataError("indexed sample is unavailable: %s" % path)
        stat = path.stat()
        if (
            int(stat.st_size) != int(entry["size_bytes"])
            or int(stat.st_mtime_ns) != int(entry["mtime_ns"])
        ):
            raise P3TrainingDataError("indexed sample changed: %s" % path)
        try:
            sample_bytes = path.read_bytes()
        except OSError as exc:
            raise P3TrainingDataError("unable to read indexed sample: %s" % path) from exc
        if _sha256_bytes(sample_bytes) != entry["content_sha256"]:
            raise P3TrainingDataError(
                "indexed sample content changed after index validation: %s" % path
            )

        # Parse the exact byte sequence that was hashed above.  This closes the
        # validation-to-read race without trusting a restorable mtime.
        with np.load(io.BytesIO(sample_bytes), allow_pickle=False) as data:
            manifest = json.loads(str(data["manifest_json"]))
            map_uuid = str(manifest.get("map_uuid", ""))
            if (
                manifest.get("schema") != MULTIMODAL_SCHEMA_VERSION
                or manifest.get("contract_revision") != MULTIMODAL_CONTRACT_REVISION
                or manifest.get("split") != self.split
                or self.split != map_split(map_uuid)
                or map_uuid != entry["map_uuid"]
            ):
                raise P3TrainingDataError("sample/index provenance mismatch: %s" % path)
            if manifest.get("sensor_authority") != self.expected_sensor_authority:
                raise P3TrainingDataError("sample sensor authority changed: %s" % path)
            preprocessing = manifest.get("preprocessing", {}).get("lidar_bev", {})
            if preprocessing.get("sha256") != self.expected_preprocessing_sha256:
                raise P3TrainingDataError("sample BEV contract changed: %s" % path)

            depth = np.asarray(data["depth_metric"], dtype=np.float32)
            depth_validity = np.asarray(data["depth_validity"])
            if depth.shape != (480, 640) or depth_validity.shape != depth.shape:
                raise P3TrainingDataError("depth must be [480,640]: %s" % path)
            if not np.all(np.isfinite(depth)) or not np.all(np.isin(depth_validity, (0, 1))):
                raise P3TrainingDataError("depth contains invalid values: %s" % path)
            valid = depth_validity.astype(np.float32)
            normalized_depth = np.where(
                valid > 0.5,
                np.clip(depth, 0.0, DEPTH_MAXIMUM_M) / DEPTH_MAXIMUM_M,
                1.0,
            ).astype(np.float32)
            normalized_depth = cv2.resize(normalized_depth, (160, 96), interpolation=cv2.INTER_NEAREST)
            valid = cv2.resize(valid, (160, 96), interpolation=cv2.INTER_NEAREST)
            depth_tensor = np.stack((normalized_depth, valid)).astype(np.float32)

            lidar_points = np.asarray(data["lidar_points"])
            if lidar_points.ndim != 2 or lidar_points.shape[1] != 5:
                raise P3TrainingDataError("LiDAR point reference shape is invalid: %s" % path)
            bev_raw = np.asarray(data["lidar_bev"], dtype=np.float32)
            if bev_raw.shape != (6, 160, 160) or not np.all(np.isfinite(bev_raw)):
                raise P3TrainingDataError("LiDAR BEV must be finite [6,160,160]: %s" % path)
            bev_resolution = float(preprocessing.get("bev", {}).get("resolution", 0.0))
            if bev_resolution <= 0.0:
                raise P3TrainingDataError("BEV resolution is invalid: %s" % path)
            bev_distance = _bev_distance_field(bev_raw, bev_resolution)
            bev_tensor = _normalize_bev(bev_raw, preprocessing)

            state_raw = np.asarray(data["vehicle_state"], dtype=np.float32)
            subgoal = np.asarray(data["subgoal_body"], dtype=np.float32)
            if state_raw.shape != (9,) or subgoal.shape != (2,) or not np.all(np.isfinite(state_raw)):
                raise P3TrainingDataError("vehicle state/subgoal shape is invalid: %s" % path)
            if not np.allclose(state_raw[4:6], subgoal, atol=1e-5):
                raise P3TrainingDataError("state and subgoal disagree: %s" % path)
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
            active_route, active_gears = _active_same_gear_route(
                local_path, route_gears, requested_gear, subgoal, heading_error
            )
            if len(active_route) > self.maximum_route_points:
                active_route = active_route[: self.maximum_route_points]
                active_gears = active_gears[: self.maximum_route_points]
            curvature = _reference_curvature(active_route)
            steering_clamped = abs(float(state_raw[2])) > STEERING_OPERATING_LIMIT_RAD
            state_physical = state_raw.copy()
            state_physical[0] = np.clip(
                state_physical[0], -REVERSE_SPEED_LIMIT_MPS, FORWARD_SPEED_LIMIT_MPS
            )
            # Acceleration/braking limits are asymmetric in the selected drive
            # direction, not in signed chassis-x coordinates.  In reverse a
            # negative dv/dt is propulsion (+1.5 limit), while positive dv/dt
            # is braking (-2.0 directed limit).
            directed_acceleration = np.clip(
                requested_gear * state_physical[1],
                -DECELERATION_LIMIT_MPS2,
                ACCELERATION_LIMIT_MPS2,
            )
            state_physical[1] = requested_gear * directed_acceleration
            state_physical[2] = np.clip(
                state_physical[2], -STEERING_OPERATING_LIMIT_RAD, STEERING_OPERATING_LIMIT_RAD
            )
            state_physical[3] = np.clip(
                state_physical[3], -YAW_RATE_SCALE_RADPS, YAW_RATE_SCALE_RADPS
            )
            state_physical[6:8] = np.clip(state_physical[6:8], -1.0, 1.0)
            state_physical[8] = curvature
            geometry_valid = current_gear != -requested_gear
            shift_speed_zeroed = not geometry_valid
            if shift_speed_zeroed:
                # This bank becomes executable only after GearSupervisor has
                # completed stop-before-shift.  Feed rollout that executable
                # boundary state while preserving state_raw for audit.
                state_physical[0] = 0.0
                state_physical[1] = 0.0
            state_normalized = np.clip(
                state_physical / np.asarray(STATE_NORMALIZATION_SCALE, dtype=np.float32),
                -2.0,
                2.0,
            )

            route_pose = np.zeros((self.maximum_route_points, 3), dtype=np.float32)
            route_features = np.zeros((self.maximum_route_points, 5), dtype=np.float32)
            route_gear_pad = np.zeros(self.maximum_route_points, dtype=np.int8)
            route_mask = np.zeros(self.maximum_route_points, dtype=bool)
            route_count = len(active_route)
            route_pose[:route_count] = active_route
            route_features[:route_count, 0:2] = np.clip(
                active_route[:, 0:2] / SUBGOAL_SCALE_M, -1.0, 1.0
            )
            route_features[:route_count, 2] = np.sin(active_route[:, 2])
            route_features[:route_count, 3] = np.cos(active_route[:, 2])
            route_features[:route_count, 4] = active_gears.astype(np.float32)
            route_gear_pad[:route_count] = active_gears
            route_mask[:route_count] = True

            trajectories = np.asarray(data["trajectories"], dtype=np.float32)
            if (
                trajectories.ndim != 3
                or trajectories.shape[0] != 15
                or trajectories.shape[2] != 6
                or trajectories.shape[1] > self.maximum_candidate_steps
                or not np.all(np.isfinite(trajectories))
            ):
                raise P3TrainingDataError("candidate trajectory shape is invalid: %s" % path)
            if np.any(np.diff(trajectories[:, :, 0], axis=1) < -1e-6):
                raise P3TrainingDataError("candidate time is not monotonic: %s" % path)
            candidate_gears = np.asarray(data["candidate_gear"], dtype=np.int8)
            feasible = np.asarray(data["feasible"])
            clearance = np.asarray(data["static_clearance"], dtype=np.float32)
            guidance = np.asarray(data["guidance_cost"], dtype=np.float32)
            if (
                candidate_gears.shape != (15,)
                or set(candidate_gears.tolist()) != {requested_gear}
                or feasible.shape != (15,)
                or clearance.shape != (15,)
                or guidance.shape != (15,)
                or not np.all(np.isin(feasible, (0, 1)))
                or not np.all(np.isfinite(clearance))
                or not np.all(np.isfinite(guidance))
            ):
                raise P3TrainingDataError("candidate authority is invalid: %s" % path)
            candidate_steps = trajectories.shape[1]
            candidate_pose = np.zeros((15, self.maximum_candidate_steps, 4), dtype=np.float32)
            candidate_time_mask = np.zeros((15, self.maximum_candidate_steps), dtype=bool)
            candidate_pose[:, :candidate_steps] = trajectories[:, :, :4]
            candidate_time_mask[:, :candidate_steps] = True
            candidate_anchor = trajectories[:, 0, 4:6].copy()
            candidate_duration = trajectories[:, -1, 0].copy()

            transform = np.asarray(
                manifest.get("transforms", {}).get("chassis_to_map", {}).get("matrix", ()),
                dtype=np.float32,
            )
            if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
                raise P3TrainingDataError("chassis_to_map transform is invalid: %s" % path)

        dropout = self._dropout_modality(entry["sample_id"])
        modality_mask = np.ones(2, dtype=np.float32)
        if dropout == "depth":
            depth_tensor.fill(0.0)
            modality_mask[0] = 0.0
        elif dropout == "lidar":
            bev_tensor.fill(0.0)
            modality_mask[1] = 0.0
        if modality_mask.sum() < 1.0:
            raise P3TrainingDataError("both sensor modalities may not be absent")

        metadata = manifest.get("metadata", {})
        raw_context = str(metadata.get("candidate_context", "UNKNOWN"))
        context_known = raw_context in ("MISSION", "RECOVERY")
        context = raw_context if context_known else "UNKNOWN"
        output = {
            "depth": torch.from_numpy(depth_tensor),
            "lidar_bev": torch.from_numpy(bev_tensor),
            "modality_mask": torch.from_numpy(modality_mask),
            # Rollout and physical losses consume ``state``.  Encoder-only
            # consumers may use the explicitly named normalized view.
            "state": torch.from_numpy(state_physical.astype(np.float32)),
            "state_normalized": torch.from_numpy(state_normalized.astype(np.float32)),
            "vehicle_state_physical": torch.from_numpy(state_physical.astype(np.float32)),
            "vehicle_state_raw": torch.from_numpy(state_raw.astype(np.float32)),
            "subgoal_body": torch.from_numpy(subgoal.astype(np.float32)),
            "current_gear": torch.tensor(current_gear, dtype=torch.int64),
            "requested_gear": torch.tensor(requested_gear, dtype=torch.int64),
            "geometry_valid": torch.tensor(geometry_valid, dtype=torch.bool),
            "shift_speed_zeroed": torch.tensor(shift_speed_zeroed, dtype=torch.bool),
            "steering_clamped": torch.tensor(steering_clamped, dtype=torch.bool),
            "route": torch.from_numpy(route_features),
            "route_pose": torch.from_numpy(route_pose),
            "route_gears": torch.from_numpy(route_gear_pad.astype(np.int64)),
            "route_mask": torch.from_numpy(route_mask),
            "candidate_pose": torch.from_numpy(candidate_pose),
            "candidate_time_mask": torch.from_numpy(candidate_time_mask),
            "candidate_anchor": torch.from_numpy(candidate_anchor),
            "candidate_duration": torch.from_numpy(candidate_duration),
            "candidate_gear": torch.from_numpy(candidate_gears.astype(np.int64)),
            "feasible": torch.from_numpy(feasible.astype(bool)),
            "static_clearance": torch.from_numpy(clearance),
            "legacy_guidance_cost": torch.from_numpy(guidance),
            "bev_distance_field": torch.from_numpy(bev_distance[None]),
            "chassis_to_map": torch.from_numpy(transform),
            "metadata": {
                "schema": TRAINING_VIEW_SCHEMA,
                "path": str(path),
                "sample_id": entry["sample_id"],
                "task_id": entry["task_id"],
                "map_uuid": map_uuid,
                "split": self.split,
                "maneuver_mode": manifest["maneuver_mode"],
                "candidate_context": context,
                "candidate_context_known": context_known,
                "preprocessing_sha256": self.expected_preprocessing_sha256,
            },
        }
        output.update(self._map_tensors(map_uuid))
        return output


def p3_training_collate(batch: Sequence[Mapping[str, object]]):
    """Collate the fixed P4 view while retaining provenance as a list."""

    if not batch:
        raise P3TrainingDataError("cannot collate an empty batch")
    tensor_keys = [key for key in batch[0] if key != "metadata"]
    output = {}
    for key in tensor_keys:
        values = [item[key] for item in batch]
        if not all(torch.is_tensor(value) for value in values):
            raise P3TrainingDataError("non-tensor field in P4 batch: %s" % key)
        try:
            output[key] = torch.stack(values, dim=0)
        except RuntimeError as exc:
            raise P3TrainingDataError("cannot stack P4 field %s" % key) from exc
    if torch.any(output["modality_mask"].sum(dim=1) < 1.0):
        raise P3TrainingDataError("collated batch contains a double-modality failure")
    output["metadata"] = [dict(item["metadata"]) for item in batch]
    return output


def p3_training_worker_init(_worker_id):
    """Prevent 8 DataLoader workers from each spawning an OpenCV thread pool."""

    cv2.setNumThreads(1)
    torch.set_num_threads(1)


__all__ = [
    "P3TrainingDataError",
    "P3TrainingDatasetV1",
    "TRAINING_INDEX_SCHEMA",
    "TRAINING_INDEX_CONTENT_AGGREGATE_SCHEMA",
    "TRAINING_INDEX_CONTENT_HASH_ALGORITHM",
    "TRAINING_VIEW_SCHEMA",
    "SIGNED_SDF_SCHEMA",
    "build_training_index",
    "load_or_build_training_index",
    "load_training_index",
    "training_index_content_aggregate",
    "p3_training_collate",
    "p3_training_worker_init",
]
