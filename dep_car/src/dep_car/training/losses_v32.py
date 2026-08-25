"""Winner-aligned gear correction for DEPCarNetV3.

V3.1 calibrated a smooth aggregate of all candidates in each gear bank.  The
runtime, however, executes the single lowest-score candidate.  V3.2 closes
that objective/decision gap by supervising the exact bank winners while
retaining V3.1's safety, forward-protection, and sequence losses.
"""

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
class DEPCarWinnerCorrectionConfigV32:
    winner_temperature: float = 0.10
    winner_cross_entropy_weight: float = 1.00
    teacher_reverse_extra_weight: float = 1.00
    no_hard_forward_extra_weight: float = 2.00
    misclassified_extra_weight: float = 2.00
    teacher_forward_margin: float = 0.06
    teacher_reverse_margin: float = 0.12
    no_hard_forward_margin: float = 0.20
    winner_margin_weight: float = 1.00
    safe_winner_margin: float = 0.20
    safe_winner_margin_weight: float = 2.00

    def validate(self):
        positive = (
            self.winner_temperature,
            self.winner_cross_entropy_weight,
            self.teacher_forward_margin,
            self.teacher_reverse_margin,
            self.no_hard_forward_margin,
            self.winner_margin_weight,
            self.safe_winner_margin,
            self.safe_winner_margin_weight,
        )
        nonnegative = (
            self.teacher_reverse_extra_weight,
            self.no_hard_forward_extra_weight,
            self.misclassified_extra_weight,
        )
        if any(
            not math.isfinite(float(value)) or float(value) <= 0.0
            for value in positive
        ):
            raise ValueError("V3.2 temperatures, margins, and weights must be positive")
        if any(
            not math.isfinite(float(value)) or float(value) < 0.0
            for value in nonnegative
        ):
            raise ValueError("V3.2 sample-weight increments must be non-negative")
        if self.no_hard_forward_margin < self.teacher_reverse_margin:
            raise ValueError("V3.2 no-forward margin must be at least the reverse margin")


def winner_aligned_terms(
    scores,
    teacher_reverse,
    no_hard_forward,
    hard_feasible,
    config=DEPCarWinnerCorrectionConfigV32(),
):
    """Supervise the same per-bank minimum used by runtime ``argmin``."""

    config.validate()
    if scores.ndim != 2 or scores.shape[1] != 30:
        raise ValueError("V3.2 scores must have shape [B,30]")
    if teacher_reverse.shape != scores.shape[:1]:
        raise ValueError("V3.2 teacher_reverse must have shape [B]")
    if no_hard_forward.shape != scores.shape[:1]:
        raise ValueError("V3.2 no_hard_forward must have shape [B]")
    if hard_feasible.shape != scores.shape:
        raise ValueError("V3.2 hard_feasible must match scores")
    score32 = scores.float()
    feasible = hard_feasible.detach().bool()
    best_forward = score32[:, :15].amin(dim=1)
    best_reverse = score32[:, 15:].amin(dim=1)
    safe_forward = score32[:, :15].masked_fill(
        ~feasible[:, :15], torch.inf
    ).amin(dim=1)
    safe_reverse = score32[:, 15:].masked_fill(
        ~feasible[:, 15:], torch.inf
    ).amin(dim=1)
    safe_bank_winner = torch.stack((safe_forward, safe_reverse), dim=1)
    raw_bank_winner = torch.stack((best_forward, best_reverse), dim=1)
    target = teacher_reverse.detach().bool().long()
    predicted_reverse = best_reverse < best_forward
    misclassified = predicted_reverse != target.bool()
    target_safe = safe_bank_winner.gather(1, target[:, None]).squeeze(1)
    target_valid = torch.isfinite(target_safe)

    sample_weight = score32.new_ones(len(score32))
    sample_weight = sample_weight + (
        target.to(sample_weight) * float(config.teacher_reverse_extra_weight)
    )
    sample_weight = sample_weight + (
        no_hard_forward.detach().bool().to(sample_weight)
        * float(config.no_hard_forward_extra_weight)
    )
    sample_weight = sample_weight + (
        misclassified.detach().to(sample_weight)
        * float(config.misclassified_extra_weight)
    )
    # A missing safe bank is deliberately worse than its raw winner, while
    # remaining finite so cross entropy cannot create inf/NaN gradients.
    finite_bank = torch.where(
        torch.isfinite(safe_bank_winner),
        safe_bank_winner,
        raw_bank_winner.detach() + 1.0,
    )
    logits = -finite_bank / float(config.winner_temperature)
    cross_entropy_per = F.cross_entropy(logits, target, reduction="none")
    valid_weight = sample_weight * target_valid.to(sample_weight)
    winner_cross_entropy = (
        cross_entropy_per * valid_weight
    ).sum() / valid_weight.sum().clamp_min(1.0)

    desired = torch.where(target_valid, target_safe, torch.zeros_like(target_safe))
    other = raw_bank_winner.gather(1, (1 - target)[:, None]).squeeze(1)
    margin = torch.where(
        target.bool(),
        score32.new_full((len(score32),), float(config.teacher_reverse_margin)),
        score32.new_full((len(score32),), float(config.teacher_forward_margin)),
    )
    margin = torch.where(
        no_hard_forward.detach().bool(),
        score32.new_full((len(score32),), float(config.no_hard_forward_margin)),
        margin,
    )
    winner_margin_per = F.relu(margin + desired - other)
    winner_margin = (
        winner_margin_per * valid_weight
    ).sum() / valid_weight.sum().clamp_min(1.0)

    best_safe = score32.masked_fill(~feasible, torch.inf).amin(dim=1)
    best_unsafe = score32.masked_fill(feasible, torch.inf).amin(dim=1)
    safety_valid = feasible.any(dim=1) & (~feasible).any(dim=1)
    best_safe = torch.where(safety_valid, best_safe, torch.zeros_like(best_safe))
    best_unsafe = torch.where(
        safety_valid, best_unsafe, torch.zeros_like(best_unsafe)
    )
    safe_winner_margin_per = F.relu(
        float(config.safe_winner_margin) + best_safe - best_unsafe
    )
    safe_winner_margin = (
        safe_winner_margin_per[safety_valid].mean()
        if bool(safety_valid.any())
        else score32.sum() * 0.0
    )
    return {
        "winner_cross_entropy": winner_cross_entropy,
        "winner_margin": winner_margin,
        "safe_winner_margin": safe_winner_margin,
        "winner_predicted_reverse": predicted_reverse.detach(),
        "winner_misclassified": misclassified.detach(),
        "winner_sample_weight": sample_weight.detach(),
        "best_forward_score": best_forward.detach(),
        "best_reverse_score": best_reverse.detach(),
    }


class DEPCarObjectiveV32:
    objective_id = "dep_car_objective_v8_runtime_argmin_winner_aligned_gear"
    objective_revision = 8
    stage = "winner_sequence_correction"

    def __init__(
        self,
        base_config=DEPCarJointGearLossConfigV3(),
        sequence_config=DEPCarSequenceCorrectionConfigV31(),
        winner_config=DEPCarWinnerCorrectionConfigV32(),
    ):
        base_config.validate()
        sequence_config.validate()
        winner_config.validate()
        self.base = DEPCarObjectiveV31(base_config, sequence_config)
        self.config = winner_config

    def __call__(self, output, **kwargs):
        base = self.base(output, **kwargs)
        winner = winner_aligned_terms(
            output.scores,
            base["teacher_reverse"],
            base["no_hard_forward"],
            base["hard_feasible"],
            self.config,
        )
        # V3.2 does not alter the accepted recovery-value head.  The
        # transition encoder is shared with that frozen head, so subtract its
        # loss before backpropagating the winner-only correction.
        recovery_weight = float(self.base.base.config.recovery_head_weight)
        v31_without_recovery = base["total"] - recovery_weight * base["recovery"]
        total = (
            v31_without_recovery
            + float(self.config.winner_cross_entropy_weight)
            * winner["winner_cross_entropy"]
            + float(self.config.winner_margin_weight) * winner["winner_margin"]
            + float(self.config.safe_winner_margin_weight)
            * winner["safe_winner_margin"]
        )
        return {
            **base,
            **winner,
            "total": total,
            "v31_total": base["total"].detach(),
            "v31_without_recovery": v31_without_recovery.detach(),
        }


__all__ = [
    "DEPCarObjectiveV32",
    "DEPCarWinnerCorrectionConfigV32",
    "winner_aligned_terms",
]
