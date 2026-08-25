import torch

from dep_car.training.losses_v331 import unilateral_bank_availability_terms


def test_unilateral_bank_correction_is_symmetric_across_gears():
    logits = torch.tensor([[0.0, 1.0], [1.0, 0.0]], requires_grad=True)
    feasible = torch.zeros(2, 30, dtype=torch.bool)
    feasible[0, 0] = True   # only forward is safe; model wrongly requests reverse
    feasible[1, 15] = True  # only reverse is safe; model wrongly requests forward
    terms = unilateral_bank_availability_terms(logits, feasible)
    assert terms["unilateral_bank_misclassified"].tolist() == [True, True]
    (terms["unilateral_bank_cross_entropy"] + terms["unilateral_bank_margin"]).backward()
    assert logits.grad[0, 0] < 0.0 and logits.grad[0, 1] > 0.0
    assert logits.grad[1, 0] > 0.0 and logits.grad[1, 1] < 0.0


def test_both_or_neither_safe_do_not_receive_unilateral_gradient():
    logits = torch.zeros(2, 2, requires_grad=True)
    feasible = torch.zeros(2, 30, dtype=torch.bool)
    feasible[0, 0] = True
    feasible[0, 15] = True
    terms = unilateral_bank_availability_terms(logits, feasible)
    assert not bool(terms["unilateral_bank_safe"].any())
    loss = terms["unilateral_bank_cross_entropy"] + terms["unilateral_bank_margin"]
    loss.backward()
    assert torch.all(logits.grad == 0.0)
