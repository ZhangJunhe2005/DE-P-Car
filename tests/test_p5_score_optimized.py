import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]


def load_score_trainer():
    path = ROOT / "tools/train_dep_car_score_optimized.py"
    spec = importlib.util.spec_from_file_location("score_optimized_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def assert_nested_close(left, right):
    assert type(left) is type(right)
    if isinstance(left, dict):
        assert set(left) == set(right)
        for key in left:
            assert_nested_close(left[key], right[key])
    elif isinstance(left, list):
        assert len(left) == len(right)
        for first, second in zip(left, right):
            assert_nested_close(first, second)
    elif isinstance(left, float):
        assert left == pytest.approx(right, abs=1.0e-6)
    else:
        assert left == right


def metric_batch(offset=0.0):
    batch = 4
    candidates = 15
    scores = torch.linspace(0.0, 1.0, batch * candidates).reshape(
        batch, candidates
    )
    clearance = torch.linspace(0.2, 1.2, batch * candidates).reshape(
        batch, candidates
    )
    candidate_cost = torch.linspace(1.2, 0.2, batch * candidates).reshape(
        batch, candidates
    )
    violation = torch.zeros((batch, candidates), dtype=torch.bool)
    violation[1, -1] = True
    feasible = (clearance > 0.0) & ~violation
    losses = {
        name: torch.tensor(float(index + 1) / 10.0 + offset)
        for index, name in enumerate(
            (
                "total",
                "candidate",
                "score",
                "safety",
                "guidance",
                "kinematic",
                "comfort",
                "diversity",
                "anchor",
            )
        )
    }
    losses.update(
        {
            "minimum_clearance": clearance,
            "candidate_cost": candidate_cost,
            "kinematic_violation": violation,
            "kinematic_violation_by_constraint": {"speed": violation},
            "hard_feasible": feasible,
        }
    )
    grouping = {
        "maneuver": ("NORMAL", "SHARP_TURN", "NORMAL", "REVERSE_EXIT"),
        "candidate_context": ("MISSION", "MISSION", "RECOVERY", "MISSION"),
        "requested_gear": ("FORWARD", "FORWARD", "REVERSE", "REVERSE"),
    }
    return SimpleNamespace(scores=scores), losses, grouping


def test_performance_authority_preserves_candidate_hashes():
    trainer = load_score_trainer()
    authority = trainer._load_performance_authority()
    attestation = trainer._performance_attestation(authority)
    assert attestation["base_candidate_trainer_sha256"] == authority["raw"][
        "base_authority"
    ]["candidate_trainer_sha256"]
    assert attestation["p4_implementation_aggregate_sha256"] == authority["raw"][
        "base_authority"
    ]["p4_implementation_aggregate_sha256"]
    assert attestation["score_training_view_schema"] == "P3ScoreTrainingDatasetV1"


def test_formal_score_defaults_are_exactly_authorized(tmp_path):
    trainer = load_score_trainer()
    authority = trainer._load_performance_authority()
    parser = trainer.build_parser(authority)
    args = parser.parse_args(["--output", str(tmp_path / "score.pth")])
    assert trainer._formal_parameter_mismatches(args, authority) == []
    args.batch_size = 128
    assert trainer._formal_parameter_mismatches(args, authority) == ["batch_size"]


def test_epoch_transfer_metrics_match_original_accumulator():
    trainer = load_score_trainer()
    original = trainer.base._MetricAccumulator()
    optimized = trainer._EpochTransferMetricAccumulator()
    for offset in (0.0, 0.25):
        output, losses, grouping = metric_batch(offset)
        original.note_geometry(4, 5)
        optimized.note_geometry(4, 5)
        original.update(output, losses, 4, grouping)
        optimized.update(output, losses, 4, grouping)
    assert_nested_close(optimized.result(), original.result())
