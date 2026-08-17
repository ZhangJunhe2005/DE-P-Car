import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import diagnose_p3_v3_feasibility as diagnosis


def row(mode="NORMAL", zero=False, count=5):
    return {
        "source": "wave01",
        "split": "train",
        "maneuver_mode": mode,
        "candidate_context": "MISSION",
        "requested_gear": "FORWARD",
        "map_uuid": "map",
        "task_id": "task",
        "feasible_candidates": 0 if zero else count,
        "zero_feasible": zero,
    }


def test_strict_repair_math_respects_open_thresholds():
    assert diagnosis.strict_zero_removals_needed(100, 10, 0.10) == 1
    assert diagnosis.strict_feasible_additions_needed(100, 10, 0.10) == 1
    assert diagnosis.strict_zero_removals_needed(100, 9, 0.10) == 0
    assert diagnosis.strict_feasible_additions_needed(100, 9, 0.10) == 0


def test_grouped_diagnosis_reports_exact_zero_rates():
    rows = [row(zero=True), row(zero=False), row("SHARP_TURN", zero=True)]

    overall = diagnosis.summary(rows)
    by_mode = diagnosis.grouped(rows, ("maneuver_mode",))

    assert overall["samples"] == 3
    assert overall["zero_feasible_samples"] == 2
    assert overall["zero_feasible_rate"] == 2 / 3
    assert by_mode[0]["maneuver_mode"] == "SHARP_TURN"
    assert by_mode[0]["zero_feasible_rate"] == 1.0
