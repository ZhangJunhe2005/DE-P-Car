"""V3.3.1 correction for frames where only one gear bank is hard-safe."""

from dataclasses import dataclass
import math

import torch
from torch.nn import functional as F

from .losses_v3 import DEPCarJointGearLossConfigV3
from .losses_v31 import DEPCarSequenceCorrectionConfigV31
from .losses_v33 import (
    DEPCarExplicitGearLossConfigV33,
    DEPCarObjectiveV33,
)


@dataclass(frozen=True)
class DEPCarBankAvailabilityCorrectionConfigV331:
    base_selector_weight: float = 0.35
    unilateral_cross_entropy_weight: float = 2.0
    unilateral_misclassified_extra_weight: float = 3.0
    unilateral_margin: float = 0.60
    unilateral_margin_weight: float = 1.5

    def validate(self):
        values = (
            self.base_selector_weight,
            self.unilateral_cross_entropy_weight,
            self.unilateral_margin,
            self.unilateral_margin_weight,
        )
        if any(not math.isfinite(float(v)) or float(v) <= 0.0 for v in values):
            raise ValueError("V3.3.1 correction weights and margin must be positive")
        if (
            not math.isfinite(float(self.unilateral_misclassified_extra_weight))
            or float(self.unilateral_misclassified_extra_weight) < 0.0
        ):
            raise ValueError("V3.3.1 misclassification weight must be non-negative")


def unilateral_bank_availability_terms(
    gear_logits,
    hard_feasible,
    config=DEPCarBankAvailabilityCorrectionConfigV331(),
):
    """Supervise the safe gear directly when exactly one bank is available.

    This target is symmetric.  It corrects both pointless reverse requests
    when only forward is safe and premature forward requests when only reverse
    is safe, so multi-leg turns are not weakened by a blanket reverse cost.
    """

    config.validate()
    if gear_logits.ndim != 2 or gear_logits.shape[1] != 2:
        raise ValueError("V3.3.1 gear_logits must have shape [B,2]")
    if hard_feasible.ndim != 2 or hard_feasible.shape[1] != 30:
        raise ValueError("V3.3.1 hard_feasible must have shape [B,30]")
    if hard_feasible.shape[0] != gear_logits.shape[0]:
        raise ValueError("V3.3.1 candidate and logit batches must match")

    feasible = hard_feasible.detach().bool()
    forward_available = feasible[:, :15].any(dim=1)
    reverse_available = feasible[:, 15:].any(dim=1)
    unilateral = forward_available ^ reverse_available
    target_reverse = reverse_available & ~forward_available
    target = target_reverse.long()
    logits = gear_logits.float()
    predicted_reverse = logits.argmax(dim=1).bool()
    misclassified = unilateral & (predicted_reverse != target_reverse)
    sample_weight = logits.new_ones(len(logits))
    sample_weight += misclassified.to(sample_weight) * float(
        config.unilateral_misclassified_extra_weight
    )
    valid_weight = sample_weight * unilateral.to(sample_weight)

    cross_entropy_per = F.cross_entropy(logits, target, reduction="none")
    cross_entropy = (
        cross_entropy_per * valid_weight
    ).sum() / valid_weight.sum().clamp_min(1.0)
    selected = logits.gather(1, target[:, None]).squeeze(1)
    other = logits.gather(1, (1 - target)[:, None]).squeeze(1)
    margin_per = F.relu(float(config.unilateral_margin) + other - selected)
    margin = (margin_per * valid_weight).sum() / valid_weight.sum().clamp_min(1.0)
    return {
        "unilateral_bank_cross_entropy": cross_entropy,
        "unilateral_bank_margin": margin,
        "unilateral_bank_safe": unilateral.detach(),
        "unilateral_safe_reverse": target_reverse.detach(),
        "unilateral_bank_misclassified": misclassified.detach(),
        "unilateral_bank_correct": (unilateral & ~misclassified).detach(),
        "unilateral_sample_weight": sample_weight.detach(),
    }


class DEPCarObjectiveV331:
    objective_id = "dep_car_objective_v10_unilateral_safe_bank_correction"
    objective_revision = 10
    stage = "unilateral_safe_bank_correction"

    def __init__(
        self,
        base_config=DEPCarJointGearLossConfigV3(),
        sequence_config=DEPCarSequenceCorrectionConfigV31(),
        selector_config=DEPCarExplicitGearLossConfigV33(),
        correction_config=DEPCarBankAvailabilityCorrectionConfigV331(),
    ):
        correction_config.validate()
        self.base = DEPCarObjectiveV33(
            base_config, sequence_config, selector_config
        )
        self.config = correction_config

    def __call__(self, output, **kwargs):
        base = self.base(output, **kwargs)
        correction = unilateral_bank_availability_terms(
            output.gear_logits,
            base["hard_feasible"],
            self.config,
        )
        total = (
            float(self.config.base_selector_weight) * base["total"]
            + float(self.config.unilateral_cross_entropy_weight)
            * correction["unilateral_bank_cross_entropy"]
            + float(self.config.unilateral_margin_weight)
            * correction["unilateral_bank_margin"]
        )
        return {
            **base,
            **correction,
            "total": total,
            "v33_selector_total": base["total"].detach(),
        }


__all__ = [
    "DEPCarBankAvailabilityCorrectionConfigV331",
    "DEPCarObjectiveV331",
    "unilateral_bank_availability_terms",
]
