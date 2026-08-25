"""Explicit gear-selector objective and hard-veto execution contract for V3.3."""

from dataclasses import dataclass
import math

import torch
from torch.nn import functional as F

from .losses_v3 import DEPCarJointGearLossConfigV3
from .losses_v31 import (
    DEPCarObjectiveV31,
    DEPCarSequenceCorrectionConfigV31,
)


@dataclass(frozen=True)
class DEPCarExplicitGearLossConfigV33:
    cross_entropy_weight: float = 1.0
    teacher_reverse_extra_weight: float = 1.0
    required_reverse_extra_weight: float = 0.75
    no_hard_forward_extra_weight: float = 2.5
    multi_action_extra_weight: float = 0.5
    misclassified_extra_weight: float = 1.0
    forward_margin: float = 0.15
    reverse_margin: float = 0.25
    no_hard_forward_margin: float = 0.45
    margin_weight: float = 0.75
    label_smoothing: float = 0.01

    def validate(self):
        positive = (
            self.cross_entropy_weight,
            self.forward_margin,
            self.reverse_margin,
            self.no_hard_forward_margin,
            self.margin_weight,
        )
        nonnegative = (
            self.teacher_reverse_extra_weight,
            self.required_reverse_extra_weight,
            self.no_hard_forward_extra_weight,
            self.multi_action_extra_weight,
            self.misclassified_extra_weight,
            self.label_smoothing,
        )
        if any(not math.isfinite(float(v)) or float(v) <= 0.0 for v in positive):
            raise ValueError("V3.3 positive loss parameters must be finite and positive")
        if any(not math.isfinite(float(v)) or float(v) < 0.0 for v in nonnegative):
            raise ValueError("V3.3 sample weights must be finite and non-negative")
        if not 0.0 <= float(self.label_smoothing) < 0.2:
            raise ValueError("V3.3 label smoothing must be in [0,0.2)")
        if self.no_hard_forward_margin < self.reverse_margin:
            raise ValueError("V3.3 no-forward margin must be at least reverse margin")


def select_with_hard_veto_v33(scores, hard_feasible, requested_reverse):
    """Select within the requested bank, then apply a visible safety fallback.

    The returned ``fallback`` flag is part of the acceptance contract.  Gear
    classification metrics must be computed before fallback so the safety
    layer cannot conceal selector mistakes.
    """

    if scores.ndim != 2 or scores.shape[1] != 30:
        raise ValueError("V3.3 scores must have shape [B,30]")
    if hard_feasible.shape != scores.shape:
        raise ValueError("V3.3 hard_feasible must match scores")
    if requested_reverse.shape != scores.shape[:1]:
        raise ValueError("V3.3 requested_reverse must have shape [B]")
    feasible = hard_feasible.bool()
    request = requested_reverse.bool()
    forward_available = feasible[:, :15].any(dim=1)
    reverse_available = feasible[:, 15:].any(dim=1)
    requested_available = torch.where(request, reverse_available, forward_available)
    other_available = torch.where(request, forward_available, reverse_available)
    execute_reverse = torch.where(
        requested_available,
        request,
        torch.where(other_available, ~request, request),
    )
    bank_mask = torch.zeros_like(feasible)
    bank_mask[:, :15] = ~execute_reverse[:, None]
    bank_mask[:, 15:] = execute_reverse[:, None]
    safe_score = scores.float().masked_fill(~(feasible & bank_mask), torch.inf)
    selected = safe_score.argmin(dim=1)
    no_safe = ~feasible.any(dim=1)
    selected = torch.where(no_safe, scores.float().argmin(dim=1), selected)
    fallback = ~requested_available & other_available
    return {
        "selected_index": selected,
        "requested_bank_available": requested_available,
        "executed_reverse": execute_reverse,
        "hard_safety_fallback": fallback,
        "no_safe_candidate": no_safe,
    }


def explicit_gear_terms(
    gear_logits,
    teacher_reverse,
    required_reverse,
    no_hard_forward,
    multi_action,
    config=DEPCarExplicitGearLossConfigV33(),
):
    config.validate()
    if gear_logits.ndim != 2 or gear_logits.shape[1] != 2:
        raise ValueError("V3.3 gear_logits must have shape [B,2]")
    expected = gear_logits.shape[:1]
    for name, value in (
        ("teacher_reverse", teacher_reverse),
        ("required_reverse", required_reverse),
        ("no_hard_forward", no_hard_forward),
        ("multi_action", multi_action),
    ):
        if value.shape != expected:
            raise ValueError("V3.3 %s must have shape [B]" % name)

    logits = gear_logits.float()
    target = teacher_reverse.detach().bool().long()
    predicted = logits.argmax(dim=1)
    misclassified = predicted != target
    sample_weight = logits.new_ones(len(logits))
    sample_weight += target.to(sample_weight) * float(
        config.teacher_reverse_extra_weight
    )
    sample_weight += required_reverse.detach().bool().to(sample_weight) * float(
        config.required_reverse_extra_weight
    )
    sample_weight += no_hard_forward.detach().bool().to(sample_weight) * float(
        config.no_hard_forward_extra_weight
    )
    sample_weight += multi_action.detach().bool().to(sample_weight) * float(
        config.multi_action_extra_weight
    )
    sample_weight += misclassified.detach().to(sample_weight) * float(
        config.misclassified_extra_weight
    )
    ce_per = F.cross_entropy(
        logits,
        target,
        reduction="none",
        label_smoothing=float(config.label_smoothing),
    )
    cross_entropy = (ce_per * sample_weight).sum() / sample_weight.sum().clamp_min(1.0)

    target_logit = logits.gather(1, target[:, None]).squeeze(1)
    other_logit = logits.gather(1, (1 - target)[:, None]).squeeze(1)
    margin = torch.where(
        target.bool(),
        logits.new_full((len(logits),), float(config.reverse_margin)),
        logits.new_full((len(logits),), float(config.forward_margin)),
    )
    margin = torch.where(
        no_hard_forward.detach().bool(),
        logits.new_full((len(logits),), float(config.no_hard_forward_margin)),
        margin,
    )
    margin_per = F.relu(margin + other_logit - target_logit)
    direction_margin = (
        margin_per * sample_weight
    ).sum() / sample_weight.sum().clamp_min(1.0)
    return {
        "explicit_gear_cross_entropy": cross_entropy,
        "explicit_gear_margin": direction_margin,
        "requested_reverse": predicted.bool().detach(),
        "gear_target_reverse": target.bool().detach(),
        "gear_misclassified": misclassified.detach(),
        "gear_sample_weight": sample_weight.detach(),
    }


class DEPCarObjectiveV33:
    objective_id = "dep_car_objective_v9_explicit_gear_selector_hard_veto"
    objective_revision = 9
    stage = "explicit_gear_selector"

    def __init__(
        self,
        base_config=DEPCarJointGearLossConfigV3(),
        sequence_config=DEPCarSequenceCorrectionConfigV31(),
        selector_config=DEPCarExplicitGearLossConfigV33(),
    ):
        base_config.validate()
        sequence_config.validate()
        selector_config.validate()
        self.base = DEPCarObjectiveV31(base_config, sequence_config)
        self.config = selector_config

    def __call__(self, output, **kwargs):
        # Base outputs are frozen; this call supplies the counterfactual
        # hard-feasibility teacher and all independent audit diagnostics.
        base = self.base(output, **kwargs)
        explicit = explicit_gear_terms(
            output.gear_logits,
            base["teacher_reverse"],
            base["required_reverse"],
            base["no_hard_forward"],
            base["multi_action"],
            self.config,
        )
        total = (
            float(self.config.cross_entropy_weight)
            * explicit["explicit_gear_cross_entropy"]
            + float(self.config.margin_weight)
            * explicit["explicit_gear_margin"]
        )
        execution = select_with_hard_veto_v33(
            output.scores.detach(),
            base["hard_feasible"],
            explicit["requested_reverse"],
        )
        return {
            **base,
            **explicit,
            **execution,
            "total": total,
            "frozen_base_total": base["total"].detach(),
        }


__all__ = [
    "DEPCarExplicitGearLossConfigV33",
    "DEPCarObjectiveV33",
    "explicit_gear_terms",
    "select_with_hard_veto_v33",
]
