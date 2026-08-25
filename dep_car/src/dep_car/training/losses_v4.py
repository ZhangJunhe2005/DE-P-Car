"""Complete hybrid-sequence objective for DEPCarNetV4."""

from dataclasses import dataclass
import math

import torch
from torch.nn import functional as F

from dep_car.core.occupancy import FootprintConfig
from dep_car.model.dep_car_net_v2 import corridor_candidate_relations

from .losses import (
    DEPCarLossConfig,
    comfort_loss,
    score_ranking_loss,
    swept_map_footprint_loss,
)
from .losses_v2 import route_tube_loss_v2
from .losses_v3 import (
    bidirectional_kinematic_components,
    bidirectional_kinematic_loss,
)


@dataclass(frozen=True)
class DEPCarHybridSequenceLossConfigV4:
    base: DEPCarLossConfig = DEPCarLossConfig()
    route_tube_radius_m: float = 0.35
    route_outer_tube_radius_m: float = 0.55
    minimum_sequence_progress_m: float = 0.20
    maximum_terminal_heading_error_rad: float = 0.80
    clearance_margin_m: float = 0.06
    sequence_actions: int = 6
    capacity_top_k: int = 3
    safety_weight: float = 4.0
    route_weight: float = 2.0
    kinematic_weight: float = 2.0
    comfort_weight: float = 0.02
    sequence_gear_weight: float = 1.5
    first_action_geometry_weight: float = 1.0
    diversity_weight: float = 0.15
    safety_head_weight: float = 0.75
    viability_head_weight: float = 0.75
    score_weight: float = 1.0
    conditional_shift_weight: float = 0.08
    conditional_reverse_distance_weight: float = 0.05
    score_temperature: float = 0.25

    def validate(self):
        self.base.validate()
        if not 0.0 < self.route_tube_radius_m < self.route_outer_tube_radius_m:
            raise ValueError("V4 route-tube radii are invalid")
        if self.sequence_actions != 6 or not 1 <= self.capacity_top_k <= 15:
            raise ValueError("V4 sequence/capacity dimensions are invalid")
        positive = (
            self.minimum_sequence_progress_m,
            self.maximum_terminal_heading_error_rad,
            self.clearance_margin_m,
            self.safety_weight,
            self.route_weight,
            self.kinematic_weight,
            self.comfort_weight,
            self.sequence_gear_weight,
            self.first_action_geometry_weight,
            self.diversity_weight,
            self.safety_head_weight,
            self.viability_head_weight,
            self.score_weight,
            self.conditional_shift_weight,
            self.conditional_reverse_distance_weight,
            self.score_temperature,
        )
        if any(not math.isfinite(float(value)) or float(value) <= 0.0 for value in positive):
            raise ValueError("V4 loss weights and thresholds must be positive")


def sequence_target_tokens(sequence_gears, sequence_mask):
    if sequence_gears.shape != sequence_mask.shape or sequence_gears.ndim != 2:
        raise ValueError("V4 sequence targets must share shape [B,6]")
    if not bool(torch.all((sequence_gears == -1) | (sequence_gears == 0) | (sequence_gears == 1))):
        raise ValueError("V4 sequence gears must contain -1,0,+1")
    mask = sequence_mask.bool()
    if bool(torch.any((sequence_gears == 0) & mask)) or bool(torch.any((sequence_gears != 0) & ~mask)):
        raise ValueError("V4 sequence mask and padding differ")
    return torch.where(
        sequence_gears > 0,
        sequence_gears.new_ones(sequence_gears.shape),
        torch.where(sequence_gears < 0, sequence_gears.new_full(sequence_gears.shape, 2), sequence_gears.new_zeros(sequence_gears.shape)),
    )


def best_of_k_sequence_gear_loss(gear_logits, sequence_gears, sequence_mask):
    if gear_logits.ndim != 4 or gear_logits.shape[-1] != 3:
        raise ValueError("V4 gear logits must have shape [B,K,6,3]")
    target = sequence_target_tokens(sequence_gears, sequence_mask)
    if target.shape != (gear_logits.shape[0], gear_logits.shape[2]):
        raise ValueError("V4 target and decoder action dimensions differ")
    expanded = target[:, None, :].expand(-1, gear_logits.shape[1], -1)
    per_action = F.cross_entropy(
        gear_logits.reshape(-1, 3), expanded.reshape(-1), reduction="none"
    ).reshape(gear_logits.shape[:3])
    # STOP padding is a real end-of-plan token.  Active macro actions receive
    # extra weight so a short sequence cannot win by predicting STOP early.
    weights = 0.35 + sequence_mask[:, None, :].to(per_action)
    per_candidate = (per_action * weights).sum(dim=-1) / weights.sum(dim=-1)
    return per_candidate.amin(dim=1).mean(), per_candidate.detach()


def _path_distance(trajectories):
    delta = trajectories[..., 1:, 1:3] - trajectories[..., :-1, 1:3]
    return torch.linalg.vector_norm(delta, dim=-1)


def _sequence_diversity(trajectories):
    scale = trajectories.new_tensor((2.5, 1.5, 1.5))
    terminal = trajectories[..., -1, 1:4] / scale
    pairwise = torch.cdist(terminal, terminal)
    upper = torch.triu(
        torch.ones((trajectories.shape[1], trajectories.shape[1]), dtype=torch.bool, device=trajectories.device),
        diagonal=1,
    )
    return F.relu(0.15 - pairwise[:, upper]).square().mean(dim=-1)


def _first_action_geometry_loss(output, target_pose, target_valid, guidance_cost, requested_gear):
    if target_pose.ndim != 3 or target_pose.shape[1:] != (15, 3):
        raise ValueError("V4 first-action teacher must contain 15 endpoint poses")
    effective = target_valid.bool()
    any_feasible = effective.any(dim=1)
    # Zero-feasible P3 frames are valid safety-negative examples.  They do not
    # invent a continuous geometry teacher; their map/route safety loss still
    # trains V4 and STOP remains available through sequence supervision.
    selection_mask = torch.where(
        any_feasible[:, None], effective, torch.ones_like(effective)
    )
    target_index = guidance_cost.float().masked_fill(~selection_mask, torch.inf).argmin(dim=1)
    target = target_pose[
        torch.arange(len(target_pose), device=target_pose.device), target_index
    ].detach()
    # First V4 action ends after five rollout samples (row zero is the origin).
    predicted = output.trajectories[..., 5, 1:4]
    raw_delta = predicted - target[:, None, :]
    delta = torch.stack(
        (
            raw_delta[..., 0],
            raw_delta[..., 1],
            torch.atan2(torch.sin(raw_delta[..., 2]), torch.cos(raw_delta[..., 2])),
        ),
        dim=-1,
    )
    distance = (delta / delta.new_tensor((1.0, 0.6, 1.0))).square().mean(dim=-1)
    matching = output.action_mask[..., 0] & (
        output.action_gears[..., 0] == requested_gear[:, None]
    )
    distance = distance.masked_fill(~matching, 1.0e3)
    per_sample = distance.amin(dim=1)
    valid_weight = any_feasible.to(per_sample)
    loss = (per_sample * valid_weight).sum() / valid_weight.sum().clamp_min(1.0)
    return loss, distance.detach()


class DEPCarObjectiveV4:
    objective_id = "dep_car_objective_v11_complete_hybrid_sequence"
    objective_revision = 11
    stages = (
        "hybrid_sequence_capacity",
        "hybrid_sequence_score",
        "closed_loop_sequence_finetune",
        "joint_smoke",
    )

    def __init__(self, config=DEPCarHybridSequenceLossConfigV4(), footprint=FootprintConfig()):
        config.validate()
        self.config = config
        self.footprint = footprint

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
        sequence_gears,
        sequence_mask,
        target_first_action_pose,
        target_first_action_valid,
        guidance_cost,
        requested_gear,
        stage="hybrid_sequence_capacity",
    ):
        if stage not in self.stages:
            raise ValueError("unknown V4 training stage: %s" % stage)
        cfg = self.config
        safe_per, minimum_clearance = swept_map_footprint_loss(
            output.trajectories.float(), map_distance_field.float(),
            map_resolution.float(), map_origin.float(), chassis_to_map.float(),
            footprint=self.footprint,
            temperature_m=cfg.base.safety_temperature_m,
            cvar_fraction=cfg.base.safety_cvar_fraction,
            mean_weight=cfg.base.safety_mean_weight,
            cvar_weight=cfg.base.safety_cvar_weight,
            worst_weight=cfg.base.safety_worst_weight,
        )
        route_cfg = type(
            "RouteConfig", (), {
                "route_tube_radius_m": cfg.route_tube_radius_m,
                "route_outer_tube_radius_m": cfg.route_outer_tube_radius_m,
                "minimum_one_cycle_progress_m": cfg.minimum_sequence_progress_m,
            },
        )()
        route_per = route_tube_loss_v2(
            output.trajectories.float(), route.float(), route_mask, route_cfg
        )
        kinematic_per = bidirectional_kinematic_loss(
            output.trajectories.float(), output.motion_gears, cfg
        )
        components = bidirectional_kinematic_components(
            output.trajectories.float(), output.motion_gears,
            lateral_acceleration_limit_mps2=cfg.base.maximum_lateral_acceleration_mps2,
        )
        violation = torch.stack(tuple(components.values()), dim=0).any(dim=0)
        hard_feasible = (minimum_clearance.detach() > 0.0) & ~violation
        relation = corridor_candidate_relations(
            output.trajectories.float(), route.float(), route_mask
        )
        progress = relation[..., 3] * 3.0
        heading_error = torch.atan2(relation[..., 4], relation[..., 5]).abs()
        cross_track = relation[..., 2] * 3.0
        viable = (
            hard_feasible
            & (progress >= float(cfg.minimum_sequence_progress_m))
            & (heading_error <= float(cfg.maximum_terminal_heading_error_rad))
            & (cross_track <= float(cfg.route_outer_tube_radius_m))
        )
        comfort_per = comfort_loss(output.trajectories.float())
        path_segments = _path_distance(output.trajectories.float())
        reverse_segment = output.motion_gears[..., 1:] < 0
        reverse_distance = (path_segments * reverse_segment.to(path_segments)).sum(dim=-1)
        shift_count = output.shift_required.float().sum(dim=-1)
        # Efficiency terms affect ranking only after geometric success.  They
        # never turn necessary reverse motion into an infeasible label.
        success_gate = viable.detach().to(path_segments)
        efficiency = success_gate * (
            float(cfg.conditional_shift_weight) * shift_count
            + float(cfg.conditional_reverse_distance_weight) * reverse_distance
        )
        candidate_cost = (
            float(cfg.safety_weight) * safe_per
            + float(cfg.route_weight) * route_per
            + float(cfg.kinematic_weight) * kinematic_per
            + float(cfg.comfort_weight) * comfort_per
            + efficiency
        )
        sequence_loss, sequence_per = best_of_k_sequence_gear_loss(
            output.gear_logits, sequence_gears, sequence_mask
        )
        first_geometry, first_geometry_per = _first_action_geometry_loss(
            output, target_first_action_pose, target_first_action_valid,
            guidance_cost, requested_gear
        )
        top = torch.topk(
            candidate_cost, int(cfg.capacity_top_k), dim=1, largest=False
        ).values.mean(dim=1)
        capacity_loss = (
            top.mean()
            + float(cfg.safety_weight) * safe_per.mean()
            + float(cfg.kinematic_weight) * kinematic_per.mean()
            + float(cfg.sequence_gear_weight) * sequence_loss
            + float(cfg.first_action_geometry_weight) * first_geometry
            + float(cfg.diversity_weight) * _sequence_diversity(output.trajectories).mean()
        )
        score_loss = score_ranking_loss(
            output.scores, candidate_cost.detach(), feasible=hard_feasible,
            temperature=float(cfg.score_temperature),
        ).mean()
        safety_head_loss = F.binary_cross_entropy_with_logits(
            output.safety_logits, hard_feasible.to(output.safety_logits)
        )
        viability_head_loss = F.binary_cross_entropy_with_logits(
            output.viability_logits, viable.to(output.viability_logits)
        )
        ranking_loss = (
            float(cfg.score_weight) * score_loss
            + float(cfg.safety_head_weight) * safety_head_loss
            + float(cfg.viability_head_weight) * viability_head_loss
        )
        if stage == "hybrid_sequence_capacity":
            total = capacity_loss + float(cfg.safety_head_weight) * safety_head_loss + float(cfg.viability_head_weight) * viability_head_loss
        elif stage == "hybrid_sequence_score":
            total = ranking_loss
        elif stage == "closed_loop_sequence_finetune":
            total = 0.5 * capacity_loss + ranking_loss
        else:
            total = capacity_loss + ranking_loss
        selected = output.scores.argmin(dim=1)
        selected_hard = hard_feasible.gather(1, selected[:, None]).squeeze(1)
        selected_viable = viable.gather(1, selected[:, None]).squeeze(1)
        return {
            "total": total,
            "capacity": capacity_loss,
            "score": score_loss,
            "sequence_gear": sequence_loss,
            "first_action_geometry": first_geometry,
            "safety_head": safety_head_loss,
            "viability_head": viability_head_loss,
            "candidate_cost": candidate_cost.detach(),
            "minimum_clearance": minimum_clearance.detach(),
            "hard_feasible": hard_feasible.detach(),
            "viable": viable.detach(),
            "selected_hard_feasible": selected_hard.detach(),
            "selected_viable": selected_viable.detach(),
            "route_progress_m": progress.detach(),
            "terminal_heading_error_rad": heading_error.detach(),
            "sequence_candidate_loss": sequence_per,
            "first_geometry_candidate_loss": first_geometry_per,
            "kinematic_violation": violation.detach(),
            "kinematic_violation_by_constraint": {
                key: value.detach() for key, value in components.items()
            },
            "shift_count": shift_count.detach(),
            "reverse_distance_m": reverse_distance.detach(),
        }


__all__ = [
    "DEPCarHybridSequenceLossConfigV4",
    "DEPCarObjectiveV4",
    "best_of_k_sequence_gear_loss",
    "sequence_target_tokens",
]
