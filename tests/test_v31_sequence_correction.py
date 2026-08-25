import importlib.util
from pathlib import Path

import torch

from dep_car.training.losses_v31 import (
    DEPCarSequenceCorrectionConfigV31,
    sequence_correction_terms,
)


def _mask(batch=1, actions=6):
    return torch.ones(batch, actions, dtype=torch.bool)


def test_no_hard_forward_forces_reverse_teacher_and_gradient_direction():
    scores = torch.zeros(1, 30, requires_grad=True)
    cost = torch.cat((torch.zeros(1, 15), torch.ones(1, 15)), dim=1)
    feasible = torch.zeros(1, 30, dtype=torch.bool)
    feasible[:, 15:] = True
    terms = sequence_correction_terms(
        scores, cost, feasible, torch.tensor([False]), _mask()
    )
    assert bool(terms["teacher_reverse"][0])
    assert bool(terms["required_reverse"][0])
    assert bool(terms["no_hard_forward"][0])
    terms["bank_cross_entropy"].backward()
    # Gradient descent lowers the desired reverse-bank scores and raises the
    # unavailable forward-bank scores.
    assert scores.grad[0, 15:].mean() > 0.0
    assert scores.grad[0, :15].mean() < 0.0


def test_open_forward_remains_a_forward_teacher_example():
    scores = torch.zeros(1, 30, requires_grad=True)
    cost = torch.cat((torch.zeros(1, 15), torch.ones(1, 15)), dim=1)
    feasible = torch.ones(1, 30, dtype=torch.bool)
    terms = sequence_correction_terms(
        scores, cost, feasible, torch.tensor([True]), _mask()
    )
    assert not bool(terms["teacher_reverse"][0])
    assert not bool(terms["required_reverse"][0])
    terms["bank_cross_entropy"].backward()
    assert scores.grad[0, :15].mean() > 0.0
    assert scores.grad[0, 15:].mean() < 0.0


def test_feasibility_margin_pushes_unsafe_winner_above_safe_candidate():
    scores = torch.full((1, 30), 2.0, requires_grad=True)
    with torch.no_grad():
        scores[0, 0] = 0.0
        scores[0, 15] = -0.2
    cost = torch.zeros(1, 30)
    feasible = torch.zeros(1, 30, dtype=torch.bool)
    feasible[0, 0] = True
    terms = sequence_correction_terms(
        scores, cost, feasible, torch.tensor([True]), _mask()
    )
    assert terms["feasible_candidate_margin"] > 0.0
    terms["feasible_candidate_margin"].backward()
    assert scores.grad[0, 0] > 0.0
    assert scores.grad[0, 15] < 0.0


def test_multi_action_weight_is_incremental_not_a_reverse_penalty():
    config = DEPCarSequenceCorrectionConfigV31(
        teacher_reverse_extra_weight=0.5,
        no_hard_forward_extra_weight=1.0,
        multi_action_extra_weight=0.25,
    )
    scores = torch.zeros(2, 30)
    cost = torch.cat((torch.zeros(2, 15), torch.ones(2, 15)), dim=1)
    feasible = torch.ones(2, 30, dtype=torch.bool)
    mask = torch.tensor(
        [[True, True, False, False, False, False], [True] * 6]
    )
    terms = sequence_correction_terms(
        scores, cost, feasible, torch.tensor([True, True]), mask, config
    )
    assert not bool(terms["teacher_reverse"].any())
    assert terms["sample_weight"][1] > terms["sample_weight"][0]


def test_best_epoch_selection_prioritizes_final_sequence_gates():
    path = Path(__file__).resolve().parents[1] / "tools/train_dep_car_sequence_v31.py"
    spec = importlib.util.spec_from_file_location("sequence_v31_trainer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    acceptance = {
        "candidate": {
            "maximum_zero_hard_feasible_rate": 0.08,
            "minimum_forward_bank_capable_rate": 0.80,
            "minimum_reverse_bank_capable_rate": 0.60,
        },
        "sequence": {
            "minimum_selected_hard_feasible_rate": 0.95,
            "maximum_unnecessary_reverse_rate": 0.05,
            "minimum_oracle_reverse_recall_within_required": 0.95,
            "maximum_oracle_forward_false_reverse_rate_within_required": 0.03,
            "minimum_no_hard_forward_reverse_selection_rate": 0.92,
        },
    }
    common = {
        "zero_hard_feasible_rate": 0.005,
        "forward_bank_capable_rate": 0.87,
        "reverse_bank_capable_rate": 0.85,
        "unnecessary_reverse_rate": 0.01,
        "oracle_forward_false_reverse_rate_within_required": 0.02,
        "mean_oracle_regret": 0.04,
    }
    passing = {
        **common,
        "selected_hard_feasible_rate": 0.951,
        "oracle_reverse_recall_within_required": 0.951,
        "no_hard_forward_reverse_selection_rate": 0.921,
    }
    safer_but_failed = {
        **common,
        "selected_hard_feasible_rate": 0.99,
        "oracle_reverse_recall_within_required": 0.90,
        "no_hard_forward_reverse_selection_rate": 0.86,
    }
    assert module.sequence_selection_key(
        passing, acceptance
    ) < module.sequence_selection_key(safer_but_failed, acceptance)
