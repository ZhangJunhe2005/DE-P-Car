"""Teacher-balanced sequence correction for the V3 joint-gear policy.

V3.1 deliberately leaves candidate generation and the accepted V3 score
objective untouched.  It adds a small, score-only correction that gives the
counterfactual hard-feasibility teacher enough influence on frames where
reverse is genuinely required, while retaining all ordinary forward frames as
negative examples.  Consequently reverse is a manoeuvre primitive rather than
an unconditional penalty or reward.
"""

from dataclasses import dataclass
import math

import torch
from torch.nn import functional as F

from .losses_v3 import (
    DEPCarJointGearLossConfigV3,
    DEPCarObjectiveV3,
)


@dataclass(frozen=True)
class DEPCarSequenceCorrectionConfigV31:
    bank_temperature: float = 0.25
    bank_cross_entropy_weight: float = 0.75
    teacher_reverse_extra_weight: float = 0.75
    no_hard_forward_extra_weight: float = 1.25
    multi_action_extra_weight: float = 0.35
    bank_direction_margin: float = 0.08
    bank_direction_margin_weight: float = 0.50
    feasible_candidate_margin: float = 0.10
    feasible_candidate_margin_weight: float = 0.75
    multi_action_minimum_actions: int = 4

    def validate(self):
        positive = (
            self.bank_temperature,
            self.bank_cross_entropy_weight,
            self.bank_direction_margin,
            self.bank_direction_margin_weight,
            self.feasible_candidate_margin,
            self.feasible_candidate_margin_weight,
        )
        nonnegative = (
            self.teacher_reverse_extra_weight,
            self.no_hard_forward_extra_weight,
            self.multi_action_extra_weight,
        )
        if any(
            not math.isfinite(float(value)) or float(value) <= 0.0
            for value in positive
        ):
            raise ValueError("V3.1 temperatures, margins, and loss weights must be positive")
        if any(
            not math.isfinite(float(value)) or float(value) < 0.0
            for value in nonnegative
        ):
            raise ValueError("V3.1 sample-weight increments must be non-negative")
        if self.multi_action_minimum_actions < 2:
            raise ValueError("V3.1 multi-action threshold must be at least two")


def _soft_bank_energy(scores, temperature):
    """Return a smooth minimum score for the forward and reverse banks."""

    value = scores.float()
    tau = float(temperature)
    return torch.stack(
        (
            -tau * torch.logsumexp(-value[:, :15] / tau, dim=1),
            -tau * torch.logsumexp(-value[:, 15:] / tau, dim=1),
        ),
        dim=1,
    )


def sequence_correction_terms(
    scores,
    candidate_cost,
    hard_feasible,
    forward_available,
    sequence_mask,
    config=DEPCarSequenceCorrectionConfigV31(),
):
    """Build counterfactual gear supervision and safety margins.

    The teacher first compares the cheapest hard-safe candidate in each bank.
    If no hard-safe forward candidate exists while reverse is available, the
    target is unconditionally reverse.  Frames with no feasible candidate in
    either bank retain the lower raw-cost bank but do not contribute a bank
    direction margin.  This avoids inventing a safe action where none exists.
    """

    config.validate()
    if scores.ndim != 2 or scores.shape[1] != 30:
        raise ValueError("V3.1 scores must have shape [B,30]")
    if candidate_cost.shape != scores.shape or hard_feasible.shape != scores.shape:
        raise ValueError("V3.1 candidate tensors must match scores")
    if forward_available.shape != scores.shape[:1]:
        raise ValueError("V3.1 forward_available must have shape [B]")
    if sequence_mask.ndim != 2 or sequence_mask.shape[0] != scores.shape[0]:
        raise ValueError("V3.1 sequence_mask must have shape [B,A]")

    detached_cost = candidate_cost.detach().float()
    feasible = hard_feasible.detach().bool()
    feasible_cost = detached_cost.masked_fill(~feasible, torch.inf)
    forward_feasible = feasible[:, :15].any(dim=1)
    reverse_feasible = feasible[:, 15:].any(dim=1)
    forward_oracle = feasible_cost[:, :15].amin(dim=1)
    reverse_oracle = feasible_cost[:, 15:].amin(dim=1)
    raw_forward = detached_cost[:, :15].amin(dim=1)
    raw_reverse = detached_cost[:, 15:].amin(dim=1)
    neither_feasible = ~forward_feasible & ~reverse_feasible
    forward_oracle = torch.where(forward_feasible, forward_oracle, raw_forward)
    reverse_oracle = torch.where(reverse_feasible, reverse_oracle, raw_reverse)
    teacher_reverse = (
        (~forward_feasible & reverse_feasible)
        | (forward_feasible & reverse_feasible & (reverse_oracle < forward_oracle))
        | (neither_feasible & (raw_reverse < raw_forward))
    )
    target = teacher_reverse.long()

    required_reverse = (~forward_available.detach().bool()) & reverse_feasible
    no_hard_forward = required_reverse & ~forward_feasible
    action_count = sequence_mask.detach().bool().sum(dim=1)
    multi_action = action_count >= int(config.multi_action_minimum_actions)

    sample_weight = scores.new_ones(scores.shape[0], dtype=torch.float32)
    sample_weight = sample_weight + (
        teacher_reverse.to(sample_weight) * float(config.teacher_reverse_extra_weight)
    )
    sample_weight = sample_weight + (
        no_hard_forward.to(sample_weight) * float(config.no_hard_forward_extra_weight)
    )
    sample_weight = sample_weight + (
        multi_action.to(sample_weight) * float(config.multi_action_extra_weight)
    )

    bank_energy = _soft_bank_energy(scores, config.bank_temperature)
    bank_logits = -bank_energy / float(config.bank_temperature)
    bank_cross_entropy_per = F.cross_entropy(
        bank_logits, target, reduction="none"
    )
    bank_cross_entropy = (
        bank_cross_entropy_per * sample_weight
    ).sum() / sample_weight.sum().clamp_min(1.0)

    selected_energy = bank_energy.gather(1, target[:, None]).squeeze(1)
    other_energy = bank_energy.gather(1, (1 - target)[:, None]).squeeze(1)
    bank_margin_valid = forward_feasible | reverse_feasible
    bank_direction_margin_per = F.relu(
        float(config.bank_direction_margin) + selected_energy - other_energy
    )
    bank_direction_margin = (
        bank_direction_margin_per[bank_margin_valid].mean()
        if bool(bank_margin_valid.any())
        else scores.float().sum() * 0.0
    )

    score32 = scores.float()
    best_feasible_score = score32.masked_fill(~feasible, torch.inf).amin(dim=1)
    best_infeasible_score = score32.masked_fill(feasible, torch.inf).amin(dim=1)
    feasibility_margin_valid = feasible.any(dim=1) & (~feasible).any(dim=1)
    best_feasible_score = torch.where(
        feasibility_margin_valid, best_feasible_score, torch.zeros_like(best_feasible_score)
    )
    best_infeasible_score = torch.where(
        feasibility_margin_valid, best_infeasible_score, torch.zeros_like(best_infeasible_score)
    )
    feasible_candidate_margin_per = F.relu(
        float(config.feasible_candidate_margin)
        + best_feasible_score
        - best_infeasible_score
    )
    feasible_candidate_margin = (
        feasible_candidate_margin_per[feasibility_margin_valid].mean()
        if bool(feasibility_margin_valid.any())
        else scores.float().sum() * 0.0
    )

    return {
        "bank_cross_entropy": bank_cross_entropy,
        "bank_direction_margin": bank_direction_margin,
        "feasible_candidate_margin": feasible_candidate_margin,
        "teacher_reverse": teacher_reverse.detach(),
        "required_reverse": required_reverse.detach(),
        "no_hard_forward": no_hard_forward.detach(),
        "multi_action": multi_action.detach(),
        "forward_feasible": forward_feasible.detach(),
        "reverse_feasible": reverse_feasible.detach(),
        "bank_energy": bank_energy.detach(),
        "sample_weight": sample_weight.detach(),
    }


class DEPCarObjectiveV31:
    objective_id = "dep_car_objective_v7_sequence_teacher_balanced_feasibility_margin"
    objective_revision = 7
    stage = "sequence_correction"

    def __init__(
        self,
        base_config=DEPCarJointGearLossConfigV3(),
        correction_config=DEPCarSequenceCorrectionConfigV31(),
    ):
        base_config.validate()
        correction_config.validate()
        self.base = DEPCarObjectiveV3(base_config)
        self.config = correction_config

    def __call__(
        self,
        output,
        *,
        map_distance_field,
        map_resolution,
        map_origin,
        chassis_to_map,
        route,
        route_mask,
        gear_history,
        sequence_mask,
    ):
        base = self.base(
            output,
            map_distance_field=map_distance_field,
            map_resolution=map_resolution,
            map_origin=map_origin,
            chassis_to_map=chassis_to_map,
            route=route,
            route_mask=route_mask,
            gear_history=gear_history,
            sequence_gears=None,
            sequence_mask=None,
            stage="joint_gear_score_calibration",
        )
        correction = sequence_correction_terms(
            output.scores,
            base["candidate_cost"],
            base["hard_feasible"],
            base["forward_available"],
            sequence_mask,
            self.config,
        )
        total = (
            base["total"]
            + float(self.config.bank_cross_entropy_weight)
            * correction["bank_cross_entropy"]
            + float(self.config.bank_direction_margin_weight)
            * correction["bank_direction_margin"]
            + float(self.config.feasible_candidate_margin_weight)
            * correction["feasible_candidate_margin"]
        )
        return {
            **base,
            **correction,
            "total": total,
            "base_score_total": base["total"].detach(),
        }


__all__ = [
    "DEPCarObjectiveV31",
    "DEPCarSequenceCorrectionConfigV31",
    "sequence_correction_terms",
]
