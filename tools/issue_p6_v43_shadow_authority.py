#!/usr/bin/env python3
"""Bind the accepted V4.3 checkpoint to a simulation P6-shadow runtime."""

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dep_car/src"))

from dep_car.runtime.p6_contract import (
    V43_ARCHITECTURE_ID,
    V43_SHADOW_AUTHORITY_SCHEMA,
    build_v43_shadow_runtime_contract,
    sha256_file,
    verify_v43_training_acceptance,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=(
            "models/dep_car/p5_closed_loop_v43/"
            "fusion_p6_shadow.authority.json"
        ),
    )
    args = parser.parse_args()
    checkpoint = (
        ROOT
        / "models/dep_car/p5_closed_loop_v43/"
        "fusion_closed_loop_sequence.best.pth"
    ).resolve()
    contract = checkpoint.with_suffix(".contract.json")
    acceptance = checkpoint.with_suffix(".acceptance.json")
    verify_v43_training_acceptance(checkpoint, contract, acceptance)
    document = {
        "schema": V43_SHADOW_AUTHORITY_SCHEMA,
        "status": "P6_SHADOW_AUTHORIZED",
        "architecture_id": V43_ARCHITECTURE_ID,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_contract": str(contract),
        "checkpoint_contract_sha256": sha256_file(contract),
        "acceptance_report": str(acceptance),
        "acceptance_report_sha256": sha256_file(acceptance),
        "runtime_implementation": build_v43_shadow_runtime_contract(),
        "mandatory_full_sequence_hard_veto": True,
        "model_selects_first_gear": True,
        "model_selects_complete_gear_sequence": True,
        "runtime_executes_learned_sequence_prefix": False,
        "legacy_turnaround_state_machine_enabled": False,
        "deterministic_shadow_control": True,
        "route_confidence_safe_cruise": True,
        "physical_speed_and_acceleration_limits_unchanged": True,
        "model_control_authorized": False,
        "active_control_authorized": False,
        "physical_vehicle_authorized": False,
        "production_qualified": False,
        "test_split_accessed": False,
        "authorization_boundary": (
            "P6 shadow comparison only. V4.3 publishes complete learned "
            "sequences, but deterministic control remains authoritative and "
            "the current occupancy hard-veto is always applied. P6 V4.3.2 "
            "may only promote existing speed tiers inside a live verified "
            "route prefix; physical speed and acceleration limits are frozen."
        ),
    }
    output = Path(args.output)
    if not output.is_absolute():
        output = (ROOT / output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    result = dict(document)
    result["authority"] = str(output)
    result["authority_sha256"] = sha256_file(output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
