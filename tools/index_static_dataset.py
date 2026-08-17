#!/usr/bin/env python3
"""Create a machine-readable index without mixing map UUIDs across splits."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("root", type=Path); args = parser.parse_args()
    counts = Counter(); maps = defaultdict(set); authorities = Counter(); samples = []
    for path in sorted(args.root.glob("*/*.npz")):
        with np.load(str(path), allow_pickle=False) as data:
            manifest = json.loads(str(data["manifest_json"]))
        split = manifest["split"]; map_uuid = manifest["map_uuid"]; authority = manifest.get("metadata", {}).get("sensor_authority", "unknown")
        counts[split] += 1; maps[split].add(map_uuid); authorities[authority] += 1
        samples.append({"path": str(path.relative_to(args.root)), "map_uuid": map_uuid, "split": split, "sensor_authority": authority})
    overlap = (maps["train"] & maps["validation"]) | (maps["train"] & maps["test"]) | (maps["validation"] & maps["test"])
    payload = {"schema": "StaticAckermannDatasetIndexV1", "samples": len(samples), "sample_counts": dict(counts), "map_counts": {key: len(value) for key, value in maps.items()}, "sensor_authorities": dict(authorities), "map_split_overlap": sorted(overlap), "production_training_ready": len(overlap) == 0 and authorities.get("urban_car_vlp16", 0) > 0, "entries": samples}
    (args.root / "manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"); print(json.dumps(payload, indent=2))


if __name__ == "__main__": main()

