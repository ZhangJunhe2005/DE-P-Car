from types import SimpleNamespace

import torch

from dep_car.training.metrics import CandidateMetricAccumulator, candidate_batch_metrics


def test_metrics_apply_hard_veto_before_lower_cost_score_ranking():
    scores = torch.tensor([[0.0, 10.0, 20.0] + [30.0] * 12, [0.0] * 15])
    clearance = torch.tensor([[-1.0, 0.2, 0.1] + [-1.0] * 12, [-1.0] * 15])
    cost = torch.tensor([[0.0, 2.0, 1.0] + [5.0] * 12, [float(i) for i in range(15)]])
    output = SimpleNamespace(scores=scores)
    metrics = candidate_batch_metrics(
        output, {"minimum_clearance": clearance, "candidate_cost": cost}
    )
    assert metrics["selected_index"].tolist() == [1, -1]
    assert metrics["zero_feasible"].tolist() == [False, True]
    assert metrics["oracle_cost"].tolist() == [1.0, 0.0]
    assert metrics["oracle_regret"][0].item() == 1.0
    assert torch.isnan(metrics["oracle_regret"][1])


def test_explicit_kinematic_veto_wins_over_a_low_learned_score():
    scores = torch.tensor([[0.0, 10.0] + [20.0] * 13])
    clearance = torch.ones(1, 15)
    cost = torch.tensor([[0.0, 1.0] + [2.0] * 13])
    violation = torch.zeros(1, 15, dtype=torch.bool)
    violation[:, 0] = True
    output = SimpleNamespace(scores=scores)
    metrics = candidate_batch_metrics(
        output,
        {
            "minimum_clearance": clearance,
            "candidate_cost": cost,
            "kinematic_violation": violation,
            "hard_feasible": (clearance > 0.0) & ~violation,
        },
    )
    assert metrics["selected_index"].item() == 1
    assert metrics["kinematic_violation_count"].item() == 1


def test_streaming_metrics_keep_maneuver_buckets():
    accumulator = CandidateMetricAccumulator()
    values = {
        "feasible_count": torch.tensor([2, 0]),
        "zero_feasible": torch.tensor([False, True]),
        "best_clearance": torch.tensor([0.4, -0.1]),
        "oracle_cost": torch.tensor([1.0, 2.0]),
        "selected_index": torch.tensor([3, -1]),
        "selected_cost": torch.tensor([1.2, float("nan")]),
        "oracle_regret": torch.tensor([0.2, float("nan")]),
        "kinematic_violation_count": torch.tensor([0, 1]),
    }
    accumulator.update(values, ["NORMAL", "REVERSE_EXIT"])
    summary = accumulator.compute()
    assert summary["overall"]["frames"] == 2
    assert summary["overall"]["zero_feasible_rate"] == 0.5
    assert set(summary["by_maneuver"]) == {"NORMAL", "REVERSE_EXIT"}
