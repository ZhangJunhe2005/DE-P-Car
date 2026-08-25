from types import SimpleNamespace

import torch

from dep_car.model.dep_car_net import DEPCarNetConfig
from dep_car.model.dep_car_net_v3 import DEPCarNetV3
from dep_car.model.gear_selector_v33 import DEPCarGearSelectorV33
from dep_car.training.losses_v33 import (
    explicit_gear_terms,
    select_with_hard_veto_v33,
)


def test_explicit_selector_has_independent_forward_reverse_logits_and_gradients():
    logits = torch.tensor([[0.4, -0.2], [0.5, -0.1]], requires_grad=True)
    terms = explicit_gear_terms(
        logits,
        teacher_reverse=torch.tensor([False, True]),
        required_reverse=torch.tensor([False, True]),
        no_hard_forward=torch.tensor([False, True]),
        multi_action=torch.tensor([False, True]),
    )
    assert terms["requested_reverse"].tolist() == [False, False]
    loss = terms["explicit_gear_cross_entropy"] + terms["explicit_gear_margin"]
    loss.backward()
    # The second sample's reverse target directly increases the explicit
    # REVERSE logit; it no longer has to distort one of 30 trajectory scores.
    assert logits.grad[1, 1] < 0.0
    assert logits.grad[1, 0] > 0.0


def test_hard_veto_stays_separate_and_reports_bank_fallback():
    scores = torch.arange(30, dtype=torch.float32)[None].repeat(3, 1)
    feasible = torch.zeros(3, 30, dtype=torch.bool)
    feasible[0, 2] = True
    feasible[1, 17] = True
    result = select_with_hard_veto_v33(
        scores, feasible, torch.tensor([False, False, True])
    )
    assert result["selected_index"].tolist() == [2, 17, 0]
    assert result["requested_bank_available"].tolist() == [True, False, False]
    assert result["hard_safety_fallback"].tolist() == [False, True, False]
    assert result["no_safe_candidate"].tolist() == [False, False, True]


def test_selector_features_are_reflection_invariant_and_detached():
    model = DEPCarGearSelectorV33(
        base_model=DEPCarNetV3(
            config=DEPCarNetConfig(enforce_reflection_equivariance=False)
        )
    )
    batch = 2
    trajectories = torch.zeros(batch, 30, 15, 6, requires_grad=True)
    trajectories.data[..., 1] = torch.linspace(0.0, 2.0, 15)
    output = SimpleNamespace(
        scores=torch.linspace(0.0, 3.0, 30)[None].repeat(batch, 1),
        forward_recovery_value=torch.full((batch, 30), 0.5),
        trajectories=trajectories,
    )
    state = torch.zeros(batch, 9)
    state[0, (2, 3, 5, 6, 8)] = torch.tensor([0.2, 0.3, 0.4, 0.5, 0.1])
    state[1] = state[0]
    state[1, (2, 3, 5, 6, 8)] *= -1.0
    route = torch.zeros(batch, 8, 3)
    route[..., 0] = torch.linspace(0.2, 2.0, 8)
    route_mask = torch.ones(batch, 8, dtype=torch.bool)
    history = torch.zeros(batch, 6)
    features = model.selector_features(
        output,
        state,
        torch.ones(batch, dtype=torch.long),
        history,
        torch.ones(batch, 2),
        route,
        route_mask,
    )
    assert features.shape == (batch, 51)
    assert not features.requires_grad
    assert torch.allclose(features[0], features[1], atol=1.0e-6)


def test_only_explicit_selector_parameters_are_trainable():
    model = DEPCarGearSelectorV33(
        base_model=DEPCarNetV3(
            config=DEPCarNetConfig(enforce_reflection_equivariance=False)
        )
    )
    model.freeze_base()
    assert sum(p.numel() for p in model.selector_parameters()) == 5576
    assert all(not p.requires_grad for p in model.base_model.parameters())
    assert all(p.requires_grad for p in model.gear_selector.parameters())
