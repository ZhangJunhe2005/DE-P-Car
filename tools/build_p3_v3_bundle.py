#!/usr/bin/env python3
"""Materialize and seal an immutable base+wave P3 V3 development bundle."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import sys
import tempfile
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dep_car" / "src"))
sys.path.insert(0, str(ROOT / "tools"))

import audit_p3_footprint_upgrade as geometry
from dep_car.training.dataset import map_split
from dep_car.training.p4_dataset import (
    EXPECTED_BEV_PREPROCESSING_SHA256,
    EXPECTED_SENSOR_AUTHORITY,
    load_or_build_training_index,
)
from dep_car.training.pilot import canonical_sha256


CONFIG_SCHEMA = "DEPCarP3V3IncrementalConfigV1"
BUNDLE_SCHEMA = "DEPCarP3V3BundleAuthorityV1"
ALLOWED_SPLITS = ("train", "validation")
ALLOWED_CONTEXTS = ("MISSION", "RECOVERY")
CURATION_SCHEMA = "P3V3InitialPoseFeasibilityCurationV1"
CURATION_POLICY = "exclude_initial_pose_infeasible"
CURATION_EVALUATOR = "production_signed_SDF_training_footprint_at_t0"


def resolve(value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def load_config(path):
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("schema") != CONFIG_SCHEMA:
        raise ValueError("P3 V3 incremental config schema mismatch")
    bundle = config.get("bundle", {})
    if bundle.get("materialization") != "copy":
        raise ValueError("P3 V3 qualification bundle requires independent file copies")
    if not bundle.get("sources"):
        raise ValueError("bundle sources are empty")
    curation = config.get("curation")
    if curation is not None:
        expected = {
            "schema": CURATION_SCHEMA,
            "policy": CURATION_POLICY,
            "evaluator": CURATION_EVALUATOR,
            "preserve_rejected_source_samples": True,
        }
        if curation != expected:
            raise ValueError("P3 V3 curation differs from the frozen policy")
    return config


def verify_internal_manifest(path):
    raw = Path(path).read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    claimed = payload.get("task_manifest_sha256", "")
    content = dict(payload)
    content.pop("task_manifest_sha256", None)
    if claimed != canonical_sha256(content):
        raise ValueError("task manifest internal SHA-256 mismatch: " + str(path))
    return payload, hashlib.sha256(raw).hexdigest()


def validate_source(source):
    name = str(source.get("name", "")).strip()
    sample_root = resolve(source["samples"])
    authority_path = resolve(source["authority"])
    if not name or not sample_root.is_dir() or not authority_path.is_file():
        raise FileNotFoundError("bundle source is unavailable: " + name)
    authority_bytes = authority_path.read_bytes()
    authority = json.loads(authority_bytes.decode("utf-8"))
    task_manifest_path = source.get("task_manifest")
    if not task_manifest_path and authority.get("status") != "PASS":
        raise RuntimeError("bundle source authority is not PASS: " + name)
    evidence = {
        "name": name,
        "sample_root": str(sample_root),
        "authority": str(authority_path),
        "authority_sha256": hashlib.sha256(authority_bytes).hexdigest(),
        "authority_schema": authority.get("schema"),
    }
    if task_manifest_path:
        task_manifest_path = resolve(task_manifest_path)
        manifest, manifest_file_hash = verify_internal_manifest(task_manifest_path)
        if authority.get("task_manifest_sha256") != manifest["task_manifest_sha256"]:
            raise RuntimeError("wave collection state/manifest mismatch: " + name)
        task_ids = {
            row["task_id"] for row in manifest.get("tasks", ())
            if row.get("map_split") in ALLOWED_SPLITS
        }
        incomplete = sorted(
            task_id for task_id in task_ids
            if authority.get("tasks", {}).get(task_id, {}).get("status") != "COMPLETE"
        )
        if incomplete:
            raise RuntimeError(
                "wave source has incomplete development tasks: "
                + ",".join(incomplete[:5])
                + "; repair them with --stage collect --retry-failed before bundling"
            )
        evidence.update({
            "task_manifest": str(task_manifest_path),
            "task_manifest_file_sha256": manifest_file_hash,
            "task_manifest_sha256": manifest["task_manifest_sha256"],
            "complete_development_tasks": len(task_ids),
        })
    return sample_root, evidence


def enumerate_source(name, sample_root):
    rows = []
    for folder in sorted(path for path in sample_root.iterdir() if path.is_dir()):
        split = map_split(folder.name)
        if split not in ALLOWED_SPLITS:
            # Split is known from the folder UUID; test NPZ files are not listed
            # or opened.
            continue
        for path in sorted(folder.glob("*.npz")):
            rows.append({
                "source_name": name,
                "source": path.resolve(),
                "relative": Path(folder.name) / path.name,
                "split": split,
            })
    if not rows:
        raise RuntimeError("bundle source has no development samples: " + name)
    return rows


def inspect_and_copy(row, output_root):
    source = row["source"]
    relative = row["relative"]
    source_bytes = source.read_bytes()
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    destination = output_root / "samples" / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        if file_sha256(destination) != source_hash:
            raise RuntimeError("existing bundle destination differs: " + str(relative))
    else:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=destination.name + ".", suffix=".tmp", dir=str(destination.parent)
        )
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(source_bytes)
                stream.flush()
                os.fsync(stream.fileno())
            shutil.copystat(source, temporary_name, follow_symlinks=True)
            os.replace(temporary_name, destination)
        except Exception:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
            raise
    if file_sha256(destination) != source_hash:
        raise RuntimeError("bundle copy verification failed: " + str(relative))
    with np.load(io.BytesIO(source_bytes), allow_pickle=False) as data:
        manifest = json.loads(str(data["manifest_json"].item()))
        context = str(manifest.get("metadata", {}).get("candidate_context", "UNKNOWN"))
        requested_gear = int(np.asarray(data["requested_gear"]).item())
    if (
        manifest.get("map_uuid") != relative.parts[0]
        or manifest.get("split") != row["split"]
        or context not in ALLOWED_CONTEXTS
        or requested_gear not in (-1, 1)
    ):
        raise RuntimeError("bundle sample authority is incomplete: " + str(relative))
    return {
        "source_name": row["source_name"],
        "path": str(relative),
        "sha256": source_hash,
        "size_bytes": len(source_bytes),
        "split": row["split"],
        "map_uuid": manifest["map_uuid"],
        "maneuver_mode": manifest.get("maneuver_mode"),
        "candidate_context": context,
        "requested_gear": requested_gear,
    }


def initialize_curation_worker(specs):
    geometry.initialize_worker(specs)


def inspect_initial_pose(row):
    """Classify one development sample without changing its source bytes."""

    source = row["source"]
    relative = row["relative"]
    source_bytes = source.read_bytes()
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    with np.load(io.BytesIO(source_bytes), allow_pickle=False) as data:
        manifest = json.loads(str(data["manifest_json"].item()))
        metadata = manifest.get("metadata", {})
        context = str(metadata.get("candidate_context", "UNKNOWN"))
        map_uuid = str(manifest.get("map_uuid", ""))
        mode = str(manifest.get("maneuver_mode", ""))
        current_gear = int(np.asarray(data["current_gear"]).item())
        requested_gear = int(np.asarray(data["requested_gear"]).item())
        vehicle_state = np.asarray(data["vehicle_state"], dtype=np.float64)
        transform_contract = manifest.get("transforms", {}).get(
            "chassis_to_map", {}
        )
        transform = np.asarray(transform_contract.get("matrix", ()), dtype=np.float64)
    if (
        map_uuid != relative.parts[0]
        or manifest.get("split") != row["split"]
        or context not in ALLOWED_CONTEXTS
        or requested_gear not in (-1, 1)
        or transform_contract.get("source_frame") != "chassis"
        or transform_contract.get("target_frame") != "map"
        or map_uuid not in geometry._WORKER_SPECS
    ):
        raise RuntimeError("curation sample authority is incomplete: " + str(relative))
    trajectories = geometry.regenerate_candidate_bank(
        vehicle_state, current_gear, requested_gear
    )
    initial_feasible, _ = geometry.p5_exact_candidate_bank(
        trajectories[:, :1, :],
        transform,
        geometry._WORKER_MAP_DISTANCE_FIELDS[map_uuid],
        geometry._WORKER_SPECS[map_uuid].resolution_m,
        geometry._WORKER_SPECS[map_uuid].origin_xy,
    )
    if not bool(np.all(initial_feasible == initial_feasible[0])):
        raise RuntimeError("candidate t=0 footprint is not invariant")
    return {
        "accepted": bool(initial_feasible[0]),
        "source_name": row["source_name"],
        "path": str(relative),
        "sha256": source_hash,
        "size_bytes": len(source_bytes),
        "split": row["split"],
        "map_uuid": map_uuid,
        "maneuver_mode": mode,
        "candidate_context": context,
        "requested_gear": requested_gear,
        "reason": "" if bool(initial_feasible[0]) else "initial_pose_infeasible",
    }


def curate_source_rows(source_rows, maps_root, workers):
    map_uuids = {str(row["relative"].parts[0]) for row in source_rows}
    specs, _contract = geometry.load_map_specs(
        maps_root, map_uuids, expected_aggregate=None
    )
    inspected = []
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=initialize_curation_worker,
        initargs=(specs,),
    ) as pool:
        for number, result in enumerate(
            pool.map(inspect_initial_pose, source_rows, chunksize=8), 1
        ):
            inspected.append(result)
            if number % 256 == 0 or number == len(source_rows):
                print(
                    "curated %d/%d source samples with %d workers"
                    % (number, len(source_rows), workers),
                    file=sys.stderr,
                    flush=True,
                )
    decisions = {row["path"]: row for row in inspected}
    if len(decisions) != len(source_rows):
        raise RuntimeError("curation did not produce one unique decision per source sample")
    accepted = [row for row in source_rows if decisions[str(row["relative"])]["accepted"]]
    rejected = sorted(
        (row for row in inspected if not row["accepted"]), key=lambda row: row["path"]
    )
    return accepted, rejected, inspected


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path,
        default=ROOT / "dep_car/config/p3_v3_incremental.yaml",
    )
    parser.add_argument(
        "--maps", type=Path, default=ROOT / "data/p3_pilot/maps"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "data/p3_v3/bundle_v1"
    )
    parser.add_argument("--workers", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    bundle_config = config["bundle"]
    workers = int(bundle_config["workers"] if args.workers is None else args.workers)
    if workers < 1 or workers > (os.cpu_count() or 1):
        raise ValueError("workers must fit the visible CPU-thread count")
    maps_root = args.maps.resolve()
    if not maps_root.is_dir():
        raise FileNotFoundError(maps_root)

    source_rows = []
    source_evidence = []
    seen_names = set()
    for source in bundle_config["sources"]:
        sample_root, evidence = validate_source(source)
        if evidence["name"] in seen_names:
            raise ValueError("duplicate bundle source name")
        seen_names.add(evidence["name"])
        rows = enumerate_source(evidence["name"], sample_root)
        evidence["development_samples"] = len(rows)
        source_evidence.append(evidence)
        source_rows.extend(rows)
    collisions = Counter(str(row["relative"]) for row in source_rows)
    collisions = sorted(path for path, count in collisions.items() if count != 1)
    if collisions:
        raise RuntimeError("bundle sample-path collision: " + ",".join(collisions[:5]))
    source_rows.sort(key=lambda row: (str(row["relative"]), row["source_name"]))
    if args.dry_run:
        print(json.dumps({
            "status": "DRY_RUN_PASS",
            "bundle_id": bundle_config["id"],
            "sources": source_evidence,
            "development_samples": len(source_rows),
            "curation": config.get("curation"),
            "curation_evaluation_deferred": config.get("curation") is not None,
            "parallel_workers": workers,
            "test_npz_opened": False,
        }, indent=2, sort_keys=True))
        return 0

    authority_path = args.output / "bundle_authority.json"
    if authority_path.exists():
        raise RuntimeError(
            "bundle is already sealed; use a new output directory instead of overwriting it"
        )
    args.output.mkdir(parents=True, exist_ok=True)
    rejected = []
    curation_payload = None
    if config.get("curation") is not None:
        original_count = len(source_rows)
        source_rows, rejected, inspected = curate_source_rows(
            source_rows, maps_root, workers
        )
        accepted_paths = {str(row["relative"]) for row in source_rows}
        unexpected_existing = sorted(
            str(path.relative_to(args.output / "samples"))
            for path in (args.output / "samples").glob("*/*.npz")
            if str(path.relative_to(args.output / "samples")) not in accepted_paths
        ) if (args.output / "samples").is_dir() else []
        if unexpected_existing:
            raise RuntimeError(
                "curated output contains a rejected or unknown sample: "
                + ",".join(unexpected_existing[:5])
            )
        rejected_counts = {
            "by_source": dict(sorted(Counter(row["source_name"] for row in rejected).items())),
            "by_split": dict(sorted(Counter(row["split"] for row in rejected).items())),
            "by_mode": dict(sorted(Counter(row["maneuver_mode"] for row in rejected).items())),
            "by_candidate_context": dict(sorted(Counter(row["candidate_context"] for row in rejected).items())),
            "by_requested_gear": dict(sorted(Counter(
                "FORWARD" if row["requested_gear"] == 1 else "REVERSE"
                for row in rejected
            ).items())),
        }
        curation_payload = {
            "schema": CURATION_SCHEMA,
            "status": "PASS",
            "qualification_authority": True,
            "policy": CURATION_POLICY,
            "evaluator": CURATION_EVALUATOR,
            "source_samples_evaluated": original_count,
            "accepted_samples": len(source_rows),
            "rejected_samples": len(rejected),
            "rejection_reason": "initial_pose_infeasible",
            "rejected_counts": rejected_counts,
            "source_inventory_sha256": canonical_sha256(sorted(
                [row["path"], row["sha256"], row["source_name"], row["accepted"]]
                for row in inspected
            )),
            "rejected_entries": rejected,
            "preserve_rejected_source_samples": True,
            "source_npz_modified": False,
            "test_npz_opened": False,
            "test_map_yaml_or_png_opened": False,
        }
        curation_payload["curation_authority_sha256"] = canonical_sha256(
            curation_payload
        )
        curation_path = args.output / "curation_authority.json"
        atomic_json(curation_path, curation_payload)
        curation_payload = {
            "schema": CURATION_SCHEMA,
            "status": "PASS",
            "policy": CURATION_POLICY,
            "evaluator": CURATION_EVALUATOR,
            "authority": str(curation_path.resolve()),
            "authority_file_sha256": file_sha256(curation_path),
            "curation_authority_sha256": json.loads(
                curation_path.read_text(encoding="utf-8")
            )["curation_authority_sha256"],
            "source_samples_evaluated": original_count,
            "accepted_samples": len(source_rows),
            "rejected_samples": len(rejected),
            "preserve_rejected_source_samples": True,
        }

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="p3v3-bundle") as pool:
        materialized = []
        for number, result in enumerate(
            pool.map(lambda row: inspect_and_copy(row, args.output), source_rows), 1
        ):
            materialized.append(result)
            if number % 512 == 0 or number == len(source_rows):
                print(
                    "materialized %d/%d accepted samples with %d workers"
                    % (number, len(source_rows), workers),
                    file=sys.stderr,
                    flush=True,
                )
    materialized.sort(key=lambda row: row["path"])
    sample_inventory_sha256 = canonical_sha256([
        [row["path"], row["sha256"], row["source_name"]]
        for row in materialized
    ])

    index_path = args.output / "training_index.json"
    index = load_or_build_training_index(
        index_path,
        sample_root=args.output / "samples",
        maps_root=maps_root,
        splits=ALLOWED_SPLITS,
        workers=workers,
        rebuild=True,
        allow_test=False,
        expected_sensor_authority=EXPECTED_SENSOR_AUTHORITY,
        expected_preprocessing_sha256=EXPECTED_BEV_PREPROCESSING_SHA256,
    )
    # Freeze the otherwise wall-clock-dependent descriptor so rebuilding an
    # identical bundle at the same path produces the same index bytes.
    index["created_at_unix"] = 0.0
    atomic_json(index_path, index)
    index_sha256 = file_sha256(index_path)
    selected_maps = {str(entry["map_uuid"]) for entry in index["entries"]}
    _specs, map_contract = geometry.load_map_specs(
        maps_root, selected_maps, expected_aggregate=None
    )
    contexts = Counter(row["candidate_context"] for row in materialized)
    validation_contexts = Counter(
        row["candidate_context"] for row in materialized
        if row["split"] == "validation"
    )
    validation_gears = Counter(
        "FORWARD" if row["requested_gear"] == 1 else "REVERSE"
        for row in materialized if row["split"] == "validation"
    )
    validation_modes = Counter(
        str(row["maneuver_mode"]) for row in materialized
        if row["split"] == "validation"
    )
    payload = {
        "schema": BUNDLE_SCHEMA,
        "status": "SEALED",
        "created_at_unix": time.time(),
        "bundle_id": bundle_config["id"],
        "config": str(args.config.resolve()),
        "config_sha256": file_sha256(args.config),
        "tool": str(Path(__file__).resolve()),
        "tool_sha256": file_sha256(__file__),
        "materialization": "independent_verified_copy",
        "sources": source_evidence,
        "sample_inventory_schema": "SortedPathContentSha256AndSourceV1",
        "sample_inventory_sha256": sample_inventory_sha256,
        "sample_root": str((args.output / "samples").resolve()),
        "maps_root": str(maps_root),
        "index": str(index_path.resolve()),
        "index_sha256": index_sha256,
        "content_aggregate_sha256": index["content_aggregate_sha256"],
        "map_contract_aggregate_sha256": map_contract["aggregate_sha256"],
        "samples": len(materialized),
        "counts_by_split": index["counts_by_split"],
        "counts_by_mode": index["counts_by_mode"],
        "counts_by_candidate_context": dict(sorted(contexts.items())),
        "validation_coverage": {
            "candidate_context": dict(sorted(validation_contexts.items())),
            "requested_gear": dict(sorted(validation_gears.items())),
            "maneuver_mode": dict(sorted(validation_modes.items())),
        },
        "parallel_workers": workers,
        "test_npz_opened": False,
        "test_map_yaml_or_png_opened": False,
    }
    if curation_payload is not None:
        payload["curation"] = curation_payload
    payload["bundle_authority_sha256"] = canonical_sha256(payload)
    atomic_json(authority_path, payload)
    print(json.dumps({
        "status": "PASS",
        "authority": str(authority_path.resolve()),
        "bundle_authority_sha256": payload["bundle_authority_sha256"],
        "samples": payload["samples"],
        "counts_by_split": payload["counts_by_split"],
        "validation_coverage": payload["validation_coverage"],
        "index_sha256": payload["index_sha256"],
        "content_aggregate_sha256": payload["content_aggregate_sha256"],
        "map_contract_aggregate_sha256": payload["map_contract_aggregate_sha256"],
        "parallel_workers": workers,
        "test_npz_opened": False,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
