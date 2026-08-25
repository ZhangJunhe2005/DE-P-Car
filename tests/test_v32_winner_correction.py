import torch

from dep_car.training.losses_v32 import (
    DEPCarWinnerCorrectionConfigV32,
    winner_aligned_terms,
)


def test_winner_alignment_matches_runtime_argmin_not_bank_population():
    scores = torch.full((1, 30), 10.0, requires_grad=True)
    with torch.no_grad():
        scores[0, 0] = 0.0
        scores[0, 15:] = 0.1
    terms = winner_aligned_terms(
        scores, torch.tensor([True]), torch.tensor([False]),
        torch.ones(1, 30, dtype=torch.bool),
    )
    # Fifteen moderately good reverse candidates can win a soft aggregate,
    # but runtime still executes the single forward score at 0.0.  V3.2 must
    # classify this exact runtime decision as a reverse-teacher error.
    assert not bool(terms["winner_predicted_reverse"][0])
    assert bool(terms["winner_misclassified"][0])
    terms["winner_cross_entropy"].backward()
    assert scores.grad[0, 0] < 0.0
    assert scores.grad[0, 15:].sum() > 0.0


def test_no_hard_forward_uses_larger_winner_margin():
    config = DEPCarWinnerCorrectionConfigV32(
        teacher_reverse_margin=0.10,
        no_hard_forward_margin=0.25,
    )
    reverse_teacher = torch.tensor([True, True])
    scores = torch.zeros(2, 30)
    ordinary = winner_aligned_terms(
        scores, reverse_teacher, torch.tensor([False, True]),
        torch.ones(2, 30, dtype=torch.bool), config,
    )
    assert ordinary["winner_margin"] > config.teacher_reverse_margin


def test_forward_teacher_is_preserved_with_asymmetric_margin():
    scores = torch.zeros(1, 30, requires_grad=True)
    terms = winner_aligned_terms(
        scores, torch.tensor([False]), torch.tensor([False]),
        torch.ones(1, 30, dtype=torch.bool),
    )
    terms["winner_margin"].backward()
    # Gradient descent lowers the best forward score and raises the best
    # reverse score, preserving the anti-unnecessary-reverse constraint.
    assert scores.grad[0, :15].sum() > 0.0
    assert scores.grad[0, 15:].sum() < 0.0


def test_winner_correction_prefers_safe_member_of_target_gear_bank():
    scores = torch.full((1, 30), 2.0, requires_grad=True)
    with torch.no_grad():
        scores[0, 0] = -0.2  # unsafe raw forward winner
        scores[0, 1] = 0.0   # safe forward winner
        scores[0, 15] = 0.1
    feasible = torch.ones(1, 30, dtype=torch.bool)
    feasible[0, 0] = False
    terms = winner_aligned_terms(
        scores, torch.tensor([False]), torch.tensor([False]), feasible
    )
    assert terms["safe_winner_margin"] > 0.0
    (terms["winner_margin"] + terms["safe_winner_margin"]).backward()
    assert scores.grad[0, 1] > 0.0
    assert scores.grad[0, 0] < 0.0
