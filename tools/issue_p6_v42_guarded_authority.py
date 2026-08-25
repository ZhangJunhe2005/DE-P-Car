#!/usr/bin/env python3
"""Issue a simulation-only V4.2 learned-control authority."""

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dep_car/src"))

from dep_car.runtime.p6_contract import (
    V42_EXECUTION_ARCHITECTURE_ID,
    V42_GUARDED_AUTHORITY_SCHEMA,
    build_v42_guarded_runtime_contract,
    sha256_file,
    verify_v42_execution_authority,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=(
            "models/dep_car/p5_hybrid_sequence_v42/"
            "fusion_guarded_simulation.authority.json"
        ),
    )
    args = parser.parse_args()
    checkpoint = (
        ROOT
        / "models/dep_car/p5_hybrid_sequence_v41/"
        "fusion_hierarchical_score.best.pth"
    ).resolve()
    contract = checkpoint.with_suffix(".contract.json")
    shadow = (
        ROOT
        / "models/dep_car/p5_hybrid_sequence_v42/"
        "fusion_calibrated_execution.authority.json"
    ).resolve()
    verify_v42_execution_authority(
        shadow,
        checkpoint_path=checkpoint,
        checkpoint_contract_path=contract,
    )
    document = {
        "schema": V42_GUARDED_AUTHORITY_SCHEMA,
        "status": "P6_GUARDED_SIMULATION_AUTHORIZED",
        "architecture_id": V42_EXECUTION_ARCHITECTURE_ID,
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_contract": str(contract),
        "checkpoint_contract_sha256": sha256_file(contract),
        "shadow_execution_authority": str(shadow),
        "shadow_execution_authority_sha256": sha256_file(shadow),
        "runtime_implementation": build_v42_guarded_runtime_contract(),
        "mandatory_full_sequence_hard_veto": True,
        "model_selects_first_gear": True,
        "model_selects_complete_gear_sequence": True,
        "runtime_executes_learned_sequence_prefix": True,
        "legacy_turnaround_state_machine_enabled": False,
        "deterministic_motion_fallback": False,
        "gazebo_simulation_only": True,
        "active_control_authorized": True,
        "physical_vehicle_authorized": False,
        "production_qualified": False,
        "test_split_accessed": False,
        "authorization_boundary": (
            "Development Gazebo only. Every learned sequence is re-vetoed "
            "against live static/dynamic occupancy; no physical or production authority."
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
