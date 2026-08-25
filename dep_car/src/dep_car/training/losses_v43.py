"""Re-observed closed-loop selector objective for DEPCarNetV4.3.

The V4 candidate decoder already covers the exact teacher gear prefixes on the
closed-loop DAgger population.  V4.3 therefore freezes candidate geometry and
learns which complete signed manoeuvre to execute.  In particular, a state
whose *initial* footprint is already inside the conservative hard boundary is
not labelled impossible: candidates that monotonically leave that boundary
form an explicit egress class.
"""

from dataclasses import dataclass
import math

import torch
from torch.nn import functional as F

from .losses_v4 import (
    DEPCarHybridSequenceLossConfigV4,
    DEPCarObjectiveV4,
    sequence_target_tokens,
)
from .losses import score_ranking_loss, swept_map_footprint_clearance


@dataclass(frozen=True)
class DEPCarClosedLoopLossConfigV43:
    base: DEPCarHybridSequenceLossConfigV4 = DEPCarHybridSequenceLossConfigV4()
    ranking_weight: float = 2.0
    selected_sequence_weight: float = 3.0
    reverse_forward_weight: float = 3.0
    action_plan_geometry_weight: float = 1.0
    safety_head_weight: float = 1.0
    eligible_margin_weight: float = 2.0
    egress_ranking_weight: float = 8.0
    egress_margin_weight: float = 8.0
    eligible_margin: float = 0.40
    score_temperature: float = 0.20
    egress_clearance_tolerance_m: float = 0.03
    egress_terminal_gain_m: float = 0.05

    def validate(self):
        self.base.validate()
        values = (
            self.ranking_weight,
            self.selected_sequence_weight,
            self.reverse_forward_weight, self.action_plan_geometry_weight,
            self.safety_head_weight,
            self.eligible_margin_weight,
            self.egress_ranking_weight, self.egress_margin_weight,
            self.eligible_margin, self.score_temperature,
            self.egress_clearance_tolerance_m,
            self.egress_terminal_gain_m,
        )
        if any(not math.isfinite(float(value)) or float(value) <= 0.0 for value in values):
            raise ValueError("V4.3 loss weights must be positive and finite")


def per_candidate_sequence_ce(gear_logits, sequence_gears, sequence_mask):
    target = sequence_target_tokens(sequence_gears, sequence_mask)
    expanded = target[:, None].expand(-1, gear_logits.shape[1], -1)
    raw = F.cross_entropy(
        gear_logits.reshape(-1, 3), expanded.reshape(-1), reduction="none"
    ).reshape(gear_logits.shape[:3])
    active_weight = 0.40 + sequence_mask[:, None].to(raw)
    return (raw * active_weight).sum(dim=-1) / active_weight.sum(dim=-1)


def score_margin(scores, preferred, margin):
    preferred = preferred.bool()
    valid = preferred.any(dim=1) & (~preferred).any(dim=1)
    best = scores.masked_fill(~preferred, torch.inf).amin(dim=1)
    other = scores.masked_fill(preferred, torch.inf).amin(dim=1)
    delta = torch.where(valid, float(margin) + best - other, torch.zeros_like(best))
    return F.relu(delta).sum() / valid.sum().clamp_min(1)


def _row_normalize(value):
    value = value.float()
    shifted = value - value.amin(dim=1, keepdim=True)
    return shifted / shifted.std(dim=1, keepdim=True, unbiased=False).clamp_min(0.10)


def _balanced_candidate_bce(logits, target, valid_rows=None):
    target = target.bool()
    raw = F.binary_cross_entropy_with_logits(
        logits, target.to(logits), reduction="none"
    )
    positive = target.sum(dim=1).clamp_min(1)
    negative = (~target).sum(dim=1).clamp_min(1)
    per_row = 0.5 * (
        (raw * target.to(raw)).sum(dim=1) / positive
        + (raw * (~target).to(raw)).sum(dim=1) / negative
    )
    if valid_rows is None:
        return per_row.mean()
    valid_rows = valid_rows.bool()
    return (per_row * valid_rows.to(per_row)).sum() / valid_rows.sum().clamp_min(1)


def closed_loop_navigation_masks(
    output, base, *, map_distance_field, map_resolution, map_origin,
    chassis_to_map, clearance_tolerance_m, terminal_gain_m,
):
    """Build a lexicographic feasible/viable/egress training authority."""

    clearance = swept_map_footprint_clearance(
        output.trajectories.float(), map_distance_field.float(),
        map_resolution.float(), map_origin.float(), chassis_to_map.float(),
    )
    initial_clearance = clearance[:, 0, 0].amin(dim=-1)
    initial_pose_safe = initial_clearance > 0.0
    future_floor = clearance[:, :, 1:].amin(dim=(-1, -2))
    terminal = clearance[:, :, -1].amin(dim=-1)
    egress = (
        (future_floor >= initial_clearance[:, None] - float(clearance_tolerance_m))
        & (
            terminal
            >= torch.maximum(
                initial_clearance[:, None] + float(terminal_gain_m),
                torch.zeros_like(terminal),
            )
        )
    )
    hard = base["hard_feasible"].bool()
    viable = base["viable"].bool()
    any_hard = hard.any(dim=1)
    any_viable = viable.any(dim=1)
    unsafe_egress = (~initial_pose_safe)[:, None] & egress
    any_egress = unsafe_egress.any(dim=1)
    eligible = torch.where(
        any_viable[:, None], viable,
        torch.where(
            any_hard[:, None], hard,
            torch.where(any_egress[:, None], unsafe_egress, torch.zeros_like(hard)),
        ),
    )
    return initial_pose_safe, egress, eligible


def mandatory_execution_mask(initial_pose_safe, hard_feasible, egress):
    """Return the candidate-level hard-safety permission mask.

    This mask never chooses forward or reverse.  In an ordinary state it
    admits every hard-feasible hybrid sequence.  If the initial footprint is
    already inside the conservative boundary, it admits every monotonic
    egress sequence.  With no admissible sequence it is empty, which means
    STOP rather than silently executing an unsafe candidate.
    """

    initial_pose_safe = initial_pose_safe.bool()
    hard_feasible = hard_feasible.bool()
    egress = egress.bool()
    if hard_feasible.shape != egress.shape:
        raise ValueError("V4.3 hard-feasible and egress masks must match")
    if initial_pose_safe.shape != hard_feasible.shape[:1]:
        raise ValueError("V4.3 initial-pose mask shape differs")
    any_hard = hard_feasible.any(dim=1)
    any_egress = egress.any(dim=1)
    return torch.where(
        initial_pose_safe[:, None],
        torch.where(
            any_hard[:, None], hard_feasible,
            torch.zeros_like(hard_feasible),
        ),
        torch.where(
            any_egress[:, None], egress, torch.zeros_like(egress)
        ),
    )


def action_plan_geometry_loss(
    output, sequence_gears, sequence_mask, target_pose, steps_per_action=5
):
    """Match every expert macro endpoint, not only the first bank endpoint."""

    if target_pose.shape != sequence_gears.shape + (3,):
        raise ValueError("V4.3 expert action endpoints must have shape [B,6,3]")
    indices = torch.arange(
        1, sequence_gears.shape[1] + 1, device=target_pose.device
    ) * int(steps_per_action)
    if int(indices[-1]) >= output.trajectories.shape[2]:
        raise ValueError("V4.3 action endpoints exceed rollout horizon")
    predicted = output.trajectories[:, :, indices, 1:4].float()
    raw = predicted - target_pose[:, None].float()
    delta = torch.stack(
        (
            raw[..., 0], raw[..., 1],
            torch.atan2(torch.sin(raw[..., 2]), torch.cos(raw[..., 2])),
        ), dim=-1,
    )
    distance = (delta / delta.new_tensor((1.0, 0.6, 1.0))).square().mean(dim=-1)
    active = sequence_mask[:, None].bool()
    per_candidate = (distance * active.to(distance)).sum(dim=-1) / active.sum(
        dim=-1
    ).clamp_min(1)
    gear_match = (
        (~active)
        | (
            output.action_mask
            & (output.action_gears == sequence_gears[:, None])
        )
    ).all(dim=-1)
    per_candidate = per_candidate + (~gear_match).to(per_candidate) * 4.0
    non_stop = sequence_mask.any(dim=1)
    best = per_candidate.amin(dim=1)
    return (
        (best * non_stop.to(best)).sum() / non_stop.sum().clamp_min(1),
        per_candidate,
    )


class DEPCarObjectiveV43:
    objective_id = "dep_car_objective_v19_guarded_contextual_exact_closed_loop_selector"
    selector_stage = "dagger_guarded_closed_loop_sequence_selector"
    stages = (selector_stage,)
    stage = selector_stage

    def __init__(self, config=DEPCarClosedLoopLossConfigV43()):
        config.validate()
        self.config = config
        self.base_objective = DEPCarObjectiveV4(config.base)

    def __call__(
        self, output, *, sequence_gears, sequence_mask,
        target_action_plan_pose, target_action_plan_mask,
        stage=stage, **kwargs
    ):
        if stage not in self.stages:
            raise ValueError("unknown V4.3 training stage: " + str(stage))
        base = self.base_objective(
            output, sequence_gears=sequence_gears, sequence_mask=sequence_mask,
            stage="hybrid_sequence_score", **kwargs
        )
        cfg = self.config
        if not torch.equal(sequence_mask.bool(), target_action_plan_mask.bool()):
            raise ValueError("V4.3 sequence and action-plan masks differ")
        per_candidate = per_candidate_sequence_ce(
            output.gear_logits, sequence_gears, sequence_mask
        )
        _action_geometry, action_geometry_per = action_plan_geometry_loss(
            output, sequence_gears, sequence_mask, target_action_plan_pose
        )
        initial_safe, egress, eligible = closed_loop_navigation_masks(
            output, base,
            map_distance_field=kwargs["map_distance_field"],
            map_resolution=kwargs["map_resolution"],
            map_origin=kwargs["map_origin"],
            chassis_to_map=kwargs["chassis_to_map"],
            clearance_tolerance_m=cfg.egress_clearance_tolerance_m,
            terminal_gain_m=cfg.egress_terminal_gain_m,
        )
        hard = base["hard_feasible"].bool()
        execution_allowed = mandatory_execution_mask(initial_safe, hard, egress)
        execution_available = execution_allowed.any(dim=1)
        target_tokens = sequence_target_tokens(sequence_gears, sequence_mask)
        token_positions = torch.arange(
            target_tokens.shape[1], device=target_tokens.device
        )[None]
        target_length = sequence_mask.sum(dim=1)
        token_weight = token_positions < target_length[:, None]
        stop_index = target_length.clamp(max=target_tokens.shape[1] - 1)
        token_weight = token_weight.clone()
        token_weight[
            torch.arange(len(token_weight), device=token_weight.device),
            stop_index,
        ] = True
        exact_sequence = (
            (output.gear_tokens == target_tokens[:, None])
            | ~token_weight[:, None]
        ).all(dim=-1)
        exact_allowed = exact_sequence & execution_allowed
        any_exact_allowed = exact_allowed.any(dim=1)
        # The full signed Hybrid-A* plan is the authority for a multi-leg
        # manoeuvre.  The legacy one-cycle ``viable`` predicate is retained as
        # a diagnostic, but cannot reject a necessary intermediate reverse
        # leg merely because that leg temporarily reduces forward progress.
        teacher_preferred = torch.where(
            any_exact_allowed[:, None], exact_allowed, execution_allowed
        )
        imitation_allowed = execution_allowed
        imitation_available = imitation_allowed.any(dim=1)
        selection_logits = (-output.scores / float(cfg.score_temperature)).masked_fill(
            ~imitation_allowed, -torch.inf
        )
        selection_probability = torch.softmax(selection_logits, dim=1)
        selection_probability = torch.where(
            imitation_available[:, None], selection_probability,
            torch.zeros_like(selection_probability),
        )
        selected_sequence_per = (selection_probability * per_candidate).sum(dim=1)
        selected_sequence = (
            selected_sequence_per * imitation_available.to(selected_sequence_per)
        ).sum() / imitation_available.sum().clamp_min(1)

        reverse_forward = (
            (sequence_gears[:, :-1] < 0)
            & (sequence_gears[:, 1:] > 0)
            & sequence_mask[:, :-1] & sequence_mask[:, 1:]
        ).any(dim=1) & imitation_available
        if bool(reverse_forward.any()):
            recovery = selected_sequence_per[reverse_forward].mean()
        else:
            recovery = selected_sequence.new_zeros(())
        quality = (
            0.25 * _row_normalize(base["candidate_cost"])
            + 3.0 * _row_normalize(per_candidate.detach())
            + 1.5 * _row_normalize(action_geometry_per.detach())
        )
        ranking = score_ranking_loss(
            output.scores, quality, feasible=execution_allowed,
            temperature=float(cfg.score_temperature),
        ).mean()
        selected_geometry_per = (
            selection_probability * _row_normalize(action_geometry_per.detach())
        ).sum(dim=1)
        selected_geometry = (
            selected_geometry_per * imitation_available.to(selected_geometry_per)
        ).sum() / imitation_available.sum().clamp_min(1)
        safety_head = _balanced_candidate_bce(output.safety_logits, hard)
        eligible_margin = score_margin(
            output.scores, teacher_preferred, cfg.eligible_margin
        )
        egress_rows = (~initial_safe) & egress.any(dim=1)
        if bool(egress_rows.any()):
            egress_ranking = score_ranking_loss(
                output.scores[egress_rows], quality[egress_rows],
                feasible=egress[egress_rows],
                temperature=float(cfg.score_temperature),
            ).mean()
            egress_margin = score_margin(
                output.scores[egress_rows], egress[egress_rows],
                cfg.eligible_margin,
            )
        else:
            egress_ranking = output.scores.new_zeros(())
            egress_margin = output.scores.new_zeros(())
        total = (
            float(cfg.ranking_weight) * ranking
            + float(cfg.selected_sequence_weight) * selected_sequence
            + float(cfg.reverse_forward_weight) * recovery
            + float(cfg.action_plan_geometry_weight) * selected_geometry
            + float(cfg.eligible_margin_weight) * eligible_margin
            + float(cfg.egress_ranking_weight) * egress_ranking
            + float(cfg.egress_margin_weight) * egress_margin
        )
        result = dict(base)
        result.update({
            "total": total,
            "v43_ranking": ranking,
            "v43_selected_sequence": selected_sequence,
            "v43_reverse_forward": recovery,
            "v43_selected_action_geometry": selected_geometry,
            "v43_safety_head": safety_head,
            "v43_eligible_margin": eligible_margin,
            "v43_egress_ranking": egress_ranking,
            "v43_egress_margin": egress_margin,
            "v43_per_candidate_sequence_ce": per_candidate.detach(),
            "v43_action_plan_geometry_per_candidate": action_geometry_per.detach(),
            "initial_pose_safe": initial_safe.detach(),
            "egress": egress.detach(),
            "legacy_route_viable": eligible.detach(),
            "navigation_eligible": teacher_preferred.detach(),
            "exact_sequence_execution_allowed": exact_allowed.detach(),
            "execution_allowed": execution_allowed.detach(),
            "execution_available": execution_available.detach(),
        })
        return result


__all__ = [
    "DEPCarClosedLoopLossConfigV43", "DEPCarObjectiveV43",
    "mandatory_execution_mask",
]
