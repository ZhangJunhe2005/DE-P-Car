#!/usr/bin/env python3
"""Fail-closed acceptance gate between P5 candidate and score training.

Thresholds come only from ``dep_car/config/training.yaml``.  This tool exposes
no threshold, provenance, smoke, or geometry override on its CLI.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import train_dep_car as trainer


SCHEMA = "DEPCarP5CandidateAcceptanceV1"


def evaluate_candidate(checkpoint, *, authority=None) -> dict:
    checkpoint = Path(checkpoint).resolve()
    result = trainer.evaluate_candidate_acceptance(
        checkpoint, authority=authority
    )
    result["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    trainer._atomic_write_json(trainer._candidate_acceptance_path(checkpoint), result)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args(argv)
    try:
        result = evaluate_candidate(args.checkpoint)
    except (ValueError, OSError, RuntimeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
