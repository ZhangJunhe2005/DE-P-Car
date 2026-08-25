#!/usr/bin/env python3
"""Seal the V3 joint-gear sequence view without opening test samples."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INDEX_SCHEMA = "DEPCarJointGearSequenceIndexV1"
AUTHORITY_SCHEMA = "DEPCarJointGearSequenceAuthorityV1"


def resolve(path):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value):
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_json(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError("expected JSON object: %s" % path)
    return value


def audit(index_path, bundle_path, output_path):
    index_path = resolve(index_path)
    bundle_path = resolve(bundle_path)
    output_path = resolve(output_path)
    index = read_json(index_path)
    bundle = read_json(bundle_path)
    errors = []
    claimed = index.get("content_sha256")
    content = dict(index)
    content.pop("content_sha256", None)
    if index.get("schema") != INDEX_SCHEMA:
        errors.append("index_schema")
    if index.get("bounded") is not False:
        errors.append("bounded_index")
    if index.get("test_split_opened") is not False:
        errors.append("test_split_opened")
    if index.get("sequence_actions") != 6:
        errors.append("sequence_actions")
    if claimed != canonical_sha256(content):
        errors.append("index_internal_sha256")
    if bundle.get("schema") != "DEPCarP3V3BundleAuthorityV1":
        errors.append("bundle_schema")
    if bundle.get("status") != "SEALED":
        errors.append("bundle_not_sealed")
    if bundle.get("test_npz_opened") is not False:
        errors.append("bundle_test_npz_opened")
    if bundle.get("test_map_yaml_or_png_opened") is not False:
        errors.append("bundle_test_map_opened")
    source_index = resolve(index.get("source_index", ""))
    bundle_index = resolve(bundle.get("index", ""))
    if source_index != bundle_index:
        errors.append("source_index_path")
    try:
        source_sha = sha256_file(source_index)
    except OSError:
        source_sha = None
        errors.append("source_index_missing")
    if source_sha != index.get("source_index_sha256"):
        errors.append("source_index_sha256")
    if source_sha != bundle.get("index_sha256"):
        errors.append("bundle_index_sha256")
    if index.get("source_content_aggregate_sha256") != bundle.get(
        "content_aggregate_sha256"
    ):
        errors.append("source_content_aggregate_sha256")
    if resolve(index.get("sample_root", "")) != resolve(
        bundle.get("sample_root", "")
    ):
        errors.append("sample_root")

    rows = index.get("rows", ())
    if not isinstance(rows, list) or len(rows) != index.get("samples"):
        errors.append("row_count")
        rows = []
    sample_ids = [row.get("sample_id") for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        errors.append("duplicate_sample_id")
    split_counts = Counter(row.get("split") for row in rows)
    if set(split_counts) != {"train", "validation"}:
        errors.append("development_splits")
    if dict(sorted(split_counts.items())) != index.get("counts_by_split"):
        errors.append("split_counts")
    if index.get("counts_by_split") != bundle.get("counts_by_split"):
        errors.append("bundle_split_coverage")
    invalid_rows = 0
    reverse_rows = 0
    multi_action_rows = 0
    alternating_rows = 0
    for row in rows:
        history = row.get("history", ())
        gears = row.get("sequence_gears", ())
        mask = row.get("sequence_mask", ())
        valid_gears = [gear for gear, keep in zip(gears, mask) if keep]
        if (
            len(history) != 6
            or len(gears) != 6
            or len(mask) != 6
            or any(gear not in (-1, 0, 1) for gear in gears)
            or any((gear == 0) == bool(keep) for gear, keep in zip(gears, mask))
            or any(a == b for a, b in zip(valid_gears, valid_gears[1:]))
            or not valid_gears
        ):
            invalid_rows += 1
            continue
        reverse_rows += int(-1 in valid_gears)
        multi_action_rows += int(len(valid_gears) >= 2)
        alternating_rows += int(len(valid_gears) >= 4)
    if invalid_rows:
        errors.append("invalid_rows")
    if reverse_rows < 1000:
        errors.append("insufficient_reverse_sequences")
    if multi_action_rows < 500:
        errors.append("insufficient_multi_action_sequences")

    authority = {
        "schema": AUTHORITY_SCHEMA,
        "status": "PASS" if not errors else "FAIL",
        "errors": sorted(set(errors)),
        "index": str(index_path),
        "index_file_sha256": sha256_file(index_path),
        "index_content_sha256": claimed,
        "bundle_authority": str(bundle_path),
        "bundle_authority_file_sha256": sha256_file(bundle_path),
        "bundle_authority_sha256": bundle.get("bundle_authority_sha256"),
        "source_index": str(source_index),
        "source_index_sha256": source_sha,
        "source_content_aggregate_sha256": index.get(
            "source_content_aggregate_sha256"
        ),
        "samples": len(rows),
        "counts_by_split": dict(sorted(split_counts.items())),
        "reverse_sequence_samples": reverse_rows,
        "multi_action_samples_ge2": multi_action_rows,
        "multi_action_samples_ge4": alternating_rows,
        "sequence_actions": 6,
        "test_split_opened": False,
        "production_qualified": False,
    }
    authority["authority_sha256"] = canonical_sha256(authority)
    if errors:
        raise RuntimeError("V3 sequence audit failed: " + ",".join(authority["errors"]))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(authority, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, output_path)
    return authority


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--index", default="data/p3_v5/joint_gear_sequence_index.json"
    )
    parser.add_argument(
        "--bundle", default="data/p3_v4/bundle_v1/bundle_authority.json"
    )
    parser.add_argument(
        "--output", default="data/p3_v5/joint_gear_sequence_authority.json"
    )
    args = parser.parse_args(argv)
    print(json.dumps(audit(args.index, args.bundle, args.output), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
