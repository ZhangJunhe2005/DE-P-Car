#!/usr/bin/env python3
"""Build or validate the reusable P4 index over P3 formal samples."""

import argparse
import json
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dep_car" / "src"))

from dep_car.training.p4_dataset import load_or_build_training_index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--samples",
        type=Path,
        default=ROOT / "data" / "p3_pilot" / "run" / "samples",
    )
    parser.add_argument(
        "--maps",
        type=Path,
        default=ROOT / "data" / "p3_pilot" / "maps",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "p3_pilot" / "run" / "training_index.json",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "validation", "test"),
        default=("train", "validation"),
        help="Defaults to train+validation; test remains sealed.",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument(
        "--allow-test",
        action="store_true",
        help="Required when creating a test-only final-evaluation index.",
    )
    args = parser.parse_args()
    existed_before = args.output.is_file()
    started = time.monotonic()
    payload = load_or_build_training_index(
        args.output,
        sample_root=args.samples,
        maps_root=args.maps,
        splits=args.splits,
        workers=args.workers,
        rebuild=args.rebuild,
        allow_test=args.allow_test,
    )
    elapsed_seconds = time.monotonic() - started
    print(json.dumps({
        "status": "PASS",
        "index": str(args.output.resolve()),
        "schema": payload["schema"],
        "samples": payload["samples"],
        "splits": payload["splits"],
        "counts_by_split": payload["counts_by_split"],
        "counts_by_mode": payload["counts_by_mode"],
        "content_hash_algorithm": payload["content_hash_algorithm"],
        "content_aggregate_sha256": payload["content_aggregate_sha256"],
        "content_verified": True,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "workers": payload["workers"],
        "reused": existed_before and not args.rebuild,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
