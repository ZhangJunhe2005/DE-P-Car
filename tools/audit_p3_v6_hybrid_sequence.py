#!/usr/bin/env python3
"""Audit whether the sealed P3 sequence view can bootstrap DEPCarNetV4."""

from collections import Counter
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "data/p3_v5/joint_gear_sequence_index.json"
AUTHORITY = ROOT / "data/p3_v5/joint_gear_sequence_authority.json"
OUTPUT = ROOT / "reports/p3_v6_hybrid_sequence_audit.json"


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


def main():
    index = json.loads(INDEX.read_text(encoding="utf-8"))
    authority = json.loads(AUTHORITY.read_text(encoding="utf-8"))
    errors = []
    content = dict(index)
    claimed = content.pop("content_sha256", None)
    authority_content = dict(authority)
    authority_claimed = authority_content.pop("authority_sha256", None)
    if (
        index.get("schema") != "DEPCarJointGearSequenceIndexV1"
        or index.get("bounded") is not False
        or index.get("test_split_opened") is not False
        or claimed != canonical_sha256(content)
    ):
        errors.append("sequence_index_authority")
    if (
        authority.get("schema") != "DEPCarJointGearSequenceAuthorityV1"
        or authority.get("status") != "PASS"
        or authority.get("errors") != []
        or authority.get("test_split_opened") is not False
        or authority_claimed != canonical_sha256(authority_content)
        or authority.get("index_file_sha256") != sha256_file(INDEX)
    ):
        errors.append("sequence_acceptance_authority")

    patterns = Counter()
    split = Counter()
    starts = Counter()
    action_counts = Counter()
    invalid = 0
    multi_switch = reverse = forward_recovery = 0
    rows = index.get("rows", ())
    for row in rows:
        gears = list(row.get("sequence_gears", ()))
        mask = list(row.get("sequence_mask", ()))
        active = [int(gear) for gear, keep in zip(gears, mask) if keep]
        valid = (
            len(gears) == 6
            and len(mask) == 6
            and bool(active)
            and all(gear in (-1, 1) for gear in active)
            and all(active[i] != active[i - 1] for i in range(1, len(active)))
            and mask == ([True] * len(active) + [False] * (6 - len(active)))
            and all(int(gears[i]) == 0 for i in range(len(active), 6))
        )
        if not valid:
            invalid += 1
            continue
        name = "-".join("F" if gear > 0 else "R" for gear in active)
        patterns[name] += 1
        split[str(row.get("split"))] += 1
        starts["FORWARD" if active[0] > 0 else "REVERSE"] += 1
        action_counts[str(len(active))] += 1
        reverse += int(any(gear < 0 for gear in active))
        multi_switch += int(len(active) >= 3)
        forward_recovery += int(
            any(active[i] < 0 and active[i + 1] > 0 for i in range(len(active) - 1))
        )
    if invalid or len(rows) != int(index.get("samples", -1)):
        errors.append("row_contract")
    required_fields = {
        "sequence_gears": True,
        "sequence_mask": True,
        "history": True,
        "action_duration_s": False,
        "action_controls": False,
        "action_terminal_pose": False,
        "terminal_route_alignment": False,
    }
    report = {
        "schema": "DEPCarP3V6HybridSequenceAuditV1",
        "status": "READY_FOR_V4_WEAK_SEQUENCE_BOOTSTRAP" if not errors else "FAIL",
        "errors": errors,
        "sequence_index": str(INDEX),
        "sequence_index_sha256": sha256_file(INDEX),
        "sequence_authority": str(AUTHORITY),
        "sequence_authority_sha256": sha256_file(AUTHORITY),
        "samples": len(rows),
        "counts_by_split": dict(sorted(split.items())),
        "counts_by_action_count": dict(sorted(action_counts.items())),
        "counts_by_start_gear": dict(sorted(starts.items())),
        "reverse_sequence_samples": reverse,
        "multi_switch_sequence_samples": multi_switch,
        "reverse_then_forward_samples": forward_recovery,
        "sequence_patterns": dict(sorted(patterns.items())),
        "available_supervision": required_fields,
        "v4_contract": {
            "maximum_macro_actions": 6,
            "gear_tokens_are_weak_episode_supervision": True,
            "first_action_geometry_uses_sealed_p3_candidate_trajectories": True,
            "later_action_geometry_uses_differentiable_rollout_route_and_map_losses": True,
            "continuous_teacher_claim_allowed": False,
            "test_split_sealed": True,
        },
        "formal_v4_training_allowed": not errors,
        "p6_active_control_authorized": False,
        "production_qualified": False,
        "test_split_opened": False,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
