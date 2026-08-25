"""Hierarchical safety/viability score correction for DEPCarNetV4.1."""

from dataclasses import dataclass
import math

import torch
from torch.nn import functional as F

from .losses import score_ranking_loss
from .losses_v4 import DEPCarHybridSequenceLossConfigV4, DEPCarObjectiveV4


@dataclass(frozen=True)
class DEPCarHierarchicalScoreLossConfigV41:
    base: DEPCarHybridSequenceLossConfigV4 = DEPCarHybridSequenceLossConfigV4()
    safety_calibration_weight: float = 2.0
    viability_calibration_weight: float = 3.0
    hard_hierarchy_weight: float = 4.0
    viability_hierarchy_weight: float = 5.0
    within_level_ranking_weight: float = 1.0
    hard_score_margin: float = 0.50
    viability_score_margin: float = 0.35
    score_temperature: float = 0.25

    def validate(self):
        self.base.validate()
        values = (
            self.safety_calibration_weight, self.viability_calibration_weight,
            self.hard_hierarchy_weight, self.viability_hierarchy_weight,
            self.within_level_ranking_weight, self.hard_score_margin,
            self.viability_score_margin, self.score_temperature,
        )
        if any(not math.isfinite(float(value)) or float(value) <= 0.0 for value in values):
            raise ValueError("V4.1 hierarchy weights and margins must be positive")


def balanced_binary_loss(logits, target):
    """Per-batch class-balanced BCE without changing the decision threshold."""

    target = target.detach().bool()
    value = target.to(logits)
    positive = value.sum()
    negative = value.numel() - positive
    positive_weight = value.numel() / (2.0 * positive.clamp_min(1.0))
    negative_weight = value.numel() / (2.0 * negative.clamp_min(1.0))
    weight = torch.where(target, positive_weight, negative_weight).detach()
    return F.binary_cross_entropy_with_logits(logits, value, weight=weight)


def hierarchy_margin_loss(scores, preferred, margin):
    """Put the best preferred candidate ahead of every lower-level option."""

    preferred = preferred.detach().bool()
    has_preferred = preferred.any(dim=1)
    has_other = (~preferred).any(dim=1)
    best_preferred = scores.masked_fill(~preferred, torch.inf).amin(dim=1)
    best_other = scores.masked_fill(preferred, torch.inf).amin(dim=1)
    valid = has_preferred & has_other
    # Do not form inf-inf (or multiply inf by zero) on all-positive/all-negative
    # rows.  Such rows contain no pairwise ranking information.
    delta = torch.where(
        valid, float(margin) + best_preferred - best_other,
        torch.zeros_like(best_preferred),
    )
    raw = F.relu(delta)
    return (raw * valid.to(raw)).sum() / valid.sum().clamp_min(1)


class DEPCarObjectiveV41:
    objective_id = "dep_car_objective_v12_hierarchical_safety_viability_score"
    stage = "hierarchical_sequence_score_correction"

    def __init__(self, config=DEPCarHierarchicalScoreLossConfigV41()):
        config.validate()
        self.config = config
        self.base_objective = DEPCarObjectiveV4(config.base)

    def __call__(self, output, *, stage=stage, **kwargs):
        if stage != self.stage:
            raise ValueError("unknown V4.1 stage: %s" % stage)
        # Reuse the frozen V4 physical labels and candidate costs.  Its total
        # is intentionally discarded: V4.1 corrects only the three ranking
        # heads and never changes candidate geometry or gear/control decoding.
        base = self.base_objective(
            output, stage="hybrid_sequence_score", **kwargs
        )
        hard = base["hard_feasible"].detach().bool()
        viable = base["viable"].detach().bool()
        any_viable = viable.any(dim=1, keepdim=True)
        any_hard = hard.any(dim=1, keepdim=True)
        fallback = torch.ones_like(hard)
        preferred = torch.where(
            any_viable, viable, torch.where(any_hard, hard, fallback)
        )
        cfg = self.config
        safety_loss = balanced_binary_loss(output.safety_logits, hard)
        viability_loss = balanced_binary_loss(output.viability_logits, viable)
        hard_margin = hierarchy_margin_loss(
            output.scores, hard, cfg.hard_score_margin
        )
        viable_margin = hierarchy_margin_loss(
            output.scores, preferred, cfg.viability_score_margin
        )
        within_level = score_ranking_loss(
            output.scores, base["candidate_cost"], feasible=preferred,
            temperature=float(cfg.score_temperature),
        ).mean()
        total = (
            float(cfg.safety_calibration_weight) * safety_loss
            + float(cfg.viability_calibration_weight) * viability_loss
            + float(cfg.hard_hierarchy_weight) * hard_margin
            + float(cfg.viability_hierarchy_weight) * viable_margin
            + float(cfg.within_level_ranking_weight) * within_level
        )
        result = dict(base)
        result.update({
            "total": total,
            "v41_safety_calibration": safety_loss,
            "v41_viability_calibration": viability_loss,
            "v41_hard_hierarchy": hard_margin,
            "v41_viability_hierarchy": viable_margin,
            "v41_within_level_ranking": within_level,
            "v41_preferred": preferred.detach(),
        })
        return result


__all__ = [
    "DEPCarHierarchicalScoreLossConfigV41", "DEPCarObjectiveV41",
    "balanced_binary_loss", "hierarchy_margin_loss",
]
