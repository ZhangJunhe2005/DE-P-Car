#!/usr/bin/env python3
"""Break down every formal V3.3 hard-safety bank fallback."""

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dep_car/src"))
sys.path.insert(0, str(ROOT / "tools"))

from dep_car.training.losses_v3 import DEPCarJointGearLossConfigV3
from dep_car.training.losses_v31 import DEPCarSequenceCorrectionConfigV31
from dep_car.training.losses_v33 import (
    DEPCarExplicitGearLossConfigV33,
    DEPCarObjectiveV33,
)
from dep_car.training.p4_dataset import p3_training_collate, p3_training_worker_init
from dep_car.training.score_dataset import P3JointGearDatasetV3
import train_dep_car_gear_selector_v33 as trainer


def main():
    config, _config_sha, v31_config, base_config, _acceptance = trainer.load_config()
    _bundle_path, bundle, sequence_path, _authority, data_gate = (
        trainer.v31.v3.verify_data_authority(base_config)
    )
    _source_path, source, _source_gate = trainer.verify_source(config, data_gate)
    checkpoint = trainer._best(trainer.resolve(config["artifact"]["output"]))
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    dataset = P3JointGearDatasetV3(
        bundle["sample_root"], bundle["maps_root"], split="validation",
        index_path=bundle["index"], index_splits=("train", "validation"),
        workers=8,
        expected_map_contract_aggregate_sha256=bundle[
            "map_contract_aggregate_sha256"
        ],
        expected_index_sha256=bundle["index_sha256"], modality="fusion",
        sequence_index_path=sequence_path,
    )
    loader = DataLoader(
        dataset, batch_size=64, shuffle=False, num_workers=8,
        pin_memory=True, persistent_workers=True, prefetch_factor=4,
        collate_fn=p3_training_collate, worker_init_fn=p3_training_worker_init,
    )
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    model = trainer.build_model(config, source)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.freeze_base()
    model.to(device).eval()
    objective = DEPCarObjectiveV33(
        DEPCarJointGearLossConfigV3(**base_config["loss"]),
        DEPCarSequenceCorrectionConfigV31(**v31_config["correction"]),
        DEPCarExplicitGearLossConfigV33(**config["selector_loss"]),
    )

    direction = Counter()
    maneuver = Counter()
    context = Counter()
    current_gear = Counter()
    target_relation = Counter()
    tasks = Counter()
    fallback_samples = []
    total = 0
    with torch.inference_mode():
        for host in loader:
            valid = host["geometry_valid"].bool()
            indices = torch.nonzero(valid, as_tuple=False).flatten().tolist()
            batch = trainer.v31.v3.select_valid(host, device)
            if batch is None:
                continue
            output, losses = trainer.forward_loss(model, objective, batch, True)
            fallback = losses["hard_safety_fallback"].cpu().bool()
            request = losses["requested_reverse"].cpu().bool()
            teacher = losses["teacher_reverse"].cpu().bool()
            feasible = losses["hard_feasible"].cpu().bool()
            gears = batch["current_gear"].cpu().tolist()
            total += len(fallback)
            for local_index in torch.nonzero(fallback, as_tuple=False).flatten().tolist():
                metadata = host["metadata"][indices[local_index]]
                reverse_request = bool(request[local_index])
                forward_safe = bool(feasible[local_index, :15].any())
                reverse_safe = bool(feasible[local_index, 15:].any())
                key = (
                    "REQUEST_REVERSE_ONLY_FORWARD_SAFE"
                    if reverse_request and forward_safe and not reverse_safe
                    else "REQUEST_FORWARD_ONLY_REVERSE_SAFE"
                    if not reverse_request and reverse_safe and not forward_safe
                    else "UNEXPECTED_FALLBACK_GEOMETRY"
                )
                direction[key] += 1
                maneuver[str(metadata.get("maneuver_mode", "UNKNOWN"))] += 1
                context[str(metadata.get("candidate_context", "UNKNOWN"))] += 1
                current_gear[str(int(gears[local_index]))] += 1
                target_relation[
                    "MATCH_TEACHER" if bool(request[local_index]) == bool(teacher[local_index])
                    else "DISAGREE_TEACHER"
                ] += 1
                tasks[str(metadata.get("task_id", ""))] += 1
                if len(fallback_samples) < 64:
                    fallback_samples.append({
                        "sample_id": metadata.get("sample_id"),
                        "task_id": metadata.get("task_id"),
                        "map_uuid": metadata.get("map_uuid"),
                        "maneuver_mode": metadata.get("maneuver_mode"),
                        "candidate_context": metadata.get("candidate_context"),
                        "current_gear": int(gears[local_index]),
                        "fallback_type": key,
                        "requested_gear": "REVERSE" if reverse_request else "FORWARD",
                        "teacher_gear": "REVERSE" if bool(teacher[local_index]) else "FORWARD",
                    })
    fallback_count = sum(direction.values())
    report = {
        "schema": "DEPCarJointGearV33FallbackDiagnosticV1",
        "status": "DIAGNOSTIC_COMPLETE",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": trainer.sha256_file(checkpoint),
        "samples": total,
        "fallback_samples": fallback_count,
        "hard_safety_fallback_rate": fallback_count / max(1, total),
        "counts_by_direction": dict(sorted(direction.items())),
        "counts_by_maneuver_mode": dict(sorted(maneuver.items())),
        "counts_by_candidate_context": dict(sorted(context.items())),
        "counts_by_current_gear": dict(sorted(current_gear.items())),
        "counts_by_teacher_relation": dict(sorted(target_relation.items())),
        "top_task_ids": [
            {"task_id": task, "count": count} for task, count in tasks.most_common(20)
        ],
        "bounded_examples": fallback_samples,
        "test_split_accessed": False,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    output_path = ROOT / "reports/p5_joint_gear_v33_fallback_diagnostic.json"
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
