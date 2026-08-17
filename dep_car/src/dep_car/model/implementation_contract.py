"""Content identity for the executable P4 model/training implementation.

Checkpoint provenance must identify more than the entry-point script.  A
change to rollout physics, state normalization, map/SDF preprocessing or loss
semantics can alter a policy without changing any tensor shape.  This module
defines the reviewed, closed file set whose aggregate is embedded in every P4
initialization and P5 training artifact.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional


P4_IMPLEMENTATION_SCHEMA = "DEPCarP4ImplementationAggregateV1"
P4_IMPLEMENTATION_FILES = (
    "dep_car/src/dep_car/core/lattice.py",
    "dep_car/src/dep_car/core/occupancy.py",
    "dep_car/src/dep_car/core/state_contract.py",
    "dep_car/src/dep_car/core/types.py",
    "dep_car/src/dep_car/core/vehicle.py",
    "dep_car/src/dep_car/model/ackermann_rollout.py",
    "dep_car/src/dep_car/model/checkpoint.py",
    "dep_car/src/dep_car/model/dep_car_net.py",
    "dep_car/src/dep_car/model/implementation_contract.py",
    "dep_car/src/dep_car/model/lidar_dep.py",
    "dep_car/src/dep_car/model/symmetry.py",
    "dep_car/src/dep_car/training/dataset.py",
    "dep_car/src/dep_car/training/losses.py",
    "dep_car/src/dep_car/training/metrics.py",
    "dep_car/src/dep_car/training/p4_dataset.py",
    "dep_car/src/dep_car/training/stages.py",
    # DEPCarNetV1 imports its depth backbone from this pinned source tree at
    # runtime.  Keeping these files only in migration metadata would miss a
    # same-shape implementation change during later P5 construction/loading.
    "third_party/DE-P/DE-P/policy/backbone_variant.py",
    "third_party/DE-P/DE-P/policy/models/MobileNetV3.py",
    "third_party/DE-P/DE-P/policy/models/backbone.py",
)


def project_root() -> Path:
    return Path(__file__).resolve().parents[4]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_p4_implementation_contract(root: Optional[Path] = None) -> dict:
    root = project_root() if root is None else Path(root).resolve()
    files = {}
    for relative in P4_IMPLEMENTATION_FILES:
        path = (root / relative).resolve()
        if root not in path.parents or not path.is_file():
            raise ValueError("P4 implementation authority is missing: %s" % relative)
        files[relative] = sha256_file(path)
    return {
        "schema": P4_IMPLEMENTATION_SCHEMA,
        "files": files,
        "aggregate_sha256": canonical_sha256(files),
    }


def verify_p4_implementation_contract(
    contract: dict, root: Optional[Path] = None
) -> dict:
    if not isinstance(contract, dict):
        raise ValueError("P4 implementation contract must be a mapping")
    expected = build_p4_implementation_contract(root)
    if contract.get("schema") != P4_IMPLEMENTATION_SCHEMA:
        raise ValueError("P4 implementation schema mismatch")
    if contract.get("files") != expected["files"]:
        raise ValueError("P4 implementation file SHA-256 mismatch")
    if contract.get("aggregate_sha256") != expected["aggregate_sha256"]:
        raise ValueError("P4 implementation aggregate SHA-256 mismatch")
    return expected


__all__ = [
    "P4_IMPLEMENTATION_FILES",
    "P4_IMPLEMENTATION_SCHEMA",
    "build_p4_implementation_contract",
    "verify_p4_implementation_contract",
]
