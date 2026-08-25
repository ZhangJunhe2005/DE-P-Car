"""Joint-gear objective for DEPCarNetV3.

Reverse motion is not assigned a blanket negative label.  Its effort and
switch costs are strong only while a hard-safe forward-progress candidate is
already available.  When forward progress is blocked, reverse candidates can
earn recovery credit by unlocking a hard-safe forward continuation.  This
preserves multi-point-turn capacity while suppressing purposeless gear chatter.
"""

from dataclasses import dataclass
import math

import torch
from torch.nn import functional as F

from dep_car.core.occupancy import FootprintConfig
from dep_car.core.state_contract import (
    ACCELERATION_LIMIT_MPS2,
    DECELERATION_LIMIT_MPS2,
    FORWARD_SPEED_LIMIT_MPS,
    REVERSE_SPEED_LIMIT_MPS,
)
from dep_car.core.vehicle import (
    PLANNER_ROLLOUT_WHEELBASE_M,
    STEERING_OPERATING_LIMIT_RAD,
)
from dep_car.model.bidirectional_rollout import BidirectionalAckermannRolloutV3
from dep_car.model.dep_car_net_v2 import corridor_candidate_relations

from .losses import (
    DEPCarLossConfig,
    _valid_mean,
    comfort_loss,
    score_ranking_loss,
    swept_map_footprint_clearance,
    swept_map_footprint_loss,
)
from .losses_v2 import constant_control_future, route_tube_loss_v2


@dataclass(frozen=True)
class DEPCarJointGearLossConfigV3:
    base: DEPCarLossConfig = DEPCarLossConfig()
    route_tube_radius_m: float = 0.30
    route_outer_tube_radius_m: float = 0.45
    minimum_one_cycle_progress_m: float = 0.05
    clearance_margin_m: float = 0.08
    future_horizon_s: float = 0.60
    future_steps: int = 7
    clearance_margin_weight: float = 2.0
    future_viability_weight: float = 1.5
    switch_effort_weight: float = 0.35
    reverse_distance_weight: float = 0.30
    repeated_switch_weight: float = 0.20
    unproductive_reverse_weight: float = 0.50
    recovery_credit_weight: float = 0.45
    recovery_head_weight: float = 0.50
    forward_available_progress_m: float = 0.12
    forward_probe_progress_m: float = 0.08
    blocked_reverse_burden_fraction: float = 0.10
    sequence_gear_weight: float = 0.25
    sequence_actions: int = 6

    def validate(self):
        self.base.validate()
        if not 0.0 < self.route_tube_radius_m < self.route_outer_tube_radius_m:
            raise ValueError("V3 route tube radii are invalid")
        positive = (
            self.clearance_margin_m,
            self.future_horizon_s,
            self.clearance_margin_weight,
            self.future_viability_weight,
            self.switch_effort_weight,
            self.reverse_distance_weight,
            self.repeated_switch_weight,
            self.unproductive_reverse_weight,
            self.recovery_credit_weight,
            self.recovery_head_weight,
            self.forward_available_progress_m,
            self.forward_probe_progress_m,
            self.sequence_gear_weight,
        )
        if any(not math.isfinite(float(value)) or value <= 0.0 for value in positive):
            raise ValueError("V3 loss weights and margins must be finite and positive")
        if not 0.0 <= self.blocked_reverse_burden_fraction < 1.0:
            raise ValueError("blocked reverse burden fraction must be in [0,1)")
        if self.future_steps < 2 or self.sequence_actions < 4:
            raise ValueError("V3 horizons are too short for multi-point turns")


def bidirectional_kinematic_components(
    trajectories,
    motion_gears,
    *,
    steering_limit_rad=STEERING_OPERATING_LIMIT_RAD,
    steering_rate_limit_rad_s=0.75,
    forward_speed_limit_mps=FORWARD_SPEED_LIMIT_MPS,
    reverse_speed_limit_mps=REVERSE_SPEED_LIMIT_MPS,
    acceleration_limit_mps2=ACCELERATION_LIMIT_MPS2,
    deceleration_limit_mps2=DECELERATION_LIMIT_MPS2,
    lateral_acceleration_limit_mps2=3.0,
    tolerance=1.0e-6,
):
    if motion_gears.shape != trajectories.shape[:-1]:
        raise ValueError("motion_gears must match trajectory [B,C,T]")
    if not bool(torch.all((motion_gears == -1) | (motion_gears == 1))):
        raise ValueError("motion_gears must contain only drive gears")
    time = trajectories[..., 0]
    speed = trajectories[..., 4]
    steering = trajectories[..., 5]
    dt = (time[..., 1:] - time[..., :-1]).clamp_min(1.0e-5)
    acceleration = (speed[..., 1:] - speed[..., :-1]) / dt
    steering_rate = (steering[..., 1:] - steering[..., :-1]) / dt
    row_gear = motion_gears.to(speed)
    segment_gear = row_gear[..., :-1]
    directed_acceleration = segment_gear * acceleration
    speed_limit = torch.where(
        row_gear > 0,
        speed.new_tensor(forward_speed_limit_mps),
        speed.new_tensor(reverse_speed_limit_mps),
    )
    lateral = speed.square() * torch.tan(steering).abs() / float(
        PLANNER_ROLLOUT_WHEELBASE_M
    )
    return {
        "opposite_motion": (row_gear * speed < -tolerance).any(dim=-1),
        "speed_limit": (speed.abs() > speed_limit + tolerance).any(dim=-1),
        "steering_limit": (
            steering.abs() > steering_limit_rad + tolerance
        ).any(dim=-1),
        "steering_rate": (
            steering_rate.abs() > steering_rate_limit_rad_s + tolerance
        ).any(dim=-1),
        "acceleration": (
            directed_acceleration > acceleration_limit_mps2 + tolerance
        ).any(dim=-1),
        "deceleration": (
            directed_acceleration < -deceleration_limit_mps2 - tolerance
        ).any(dim=-1),
        "lateral_acceleration": (
            lateral > lateral_acceleration_limit_mps2 + tolerance
        ).any(dim=-1),
    }


def bidirectional_kinematic_loss(trajectories, motion_gears, config):
    time = trajectories[..., 0]
    speed = trajectories[..., 4]
    steering = trajectories[..., 5]
    dt = (time[..., 1:] - time[..., :-1]).clamp_min(1.0e-5)
    acceleration = (speed[..., 1:] - speed[..., :-1]) / dt
    steering_rate = (steering[..., 1:] - steering[..., :-1]) / dt
    row_gear = motion_gears.to(speed)
    segment_gear = row_gear[..., :-1]
    directed = segment_gear * acceleration
    speed_limit = torch.where(
        row_gear > 0,
        speed.new_tensor(FORWARD_SPEED_LIMIT_MPS),
        speed.new_tensor(REVERSE_SPEED_LIMIT_MPS),
    )
    lateral = speed.square() * torch.tan(steering).abs() / float(
        PLANNER_ROLLOUT_WHEELBASE_M
    )
    operating = 1.0 - config.base.kinematic_safety_margin_fraction
    return (
        F.relu(-row_gear * speed).square().mean(dim=-1)
        + F.relu(speed.abs() - operating * speed_limit).square().mean(dim=-1)
        + F.relu(
            steering.abs() - operating * STEERING_OPERATING_LIMIT_RAD
        ).square().mean(dim=-1)
        + F.relu(
            steering_rate.abs() - operating * 0.75
        ).square().mean(dim=-1)
        + F.relu(
            directed - operating * ACCELERATION_LIMIT_MPS2
        ).square().mean(dim=-1)
        + F.relu(
            -directed - operating * DECELERATION_LIMIT_MPS2
        ).square().mean(dim=-1)
        + F.relu(
            lateral
            - operating * config.base.maximum_lateral_acceleration_mps2
        ).square().mean(dim=-1)
    )


def bank_diversity_loss(trajectories, config):
    """Apply normalized repulsion inside each bank, never across gear banks."""

    base = config.base
    rows = []
    for bank, scales in (
        (trajectories[:, :15], base.forward_diversity_scales),
        (trajectories[:, 15:], base.reverse_diversity_scales),
    ):
        scale = bank.new_tensor(scales)
        terminal = bank[..., -1, 1:4] / scale[None, None]
        pairwise = torch.cdist(terminal, terminal)
        upper = torch.triu(
            torch.ones((15, 15), dtype=torch.bool, device=bank.device),
            diagonal=1,
        )
        rows.append(
            F.relu(
                base.minimum_normalized_terminal_separation
                - pairwise[:, upper]
            ).square().mean(dim=-1)
        )
    return 0.5 * (rows[0] + rows[1])


def _path_distance(trajectories):
    delta = trajectories[..., 1:, 1:3] - trajectories[..., :-1, 1:3]
    return torch.linalg.vector_norm(delta, dim=-1).sum(dim=-1)


def conditional_gear_costs(
    *,
    candidate_gears,
    shift_required,
    hard_feasible,
    route_progress_m,
    path_distance_m,
    forward_recovery_target,
    gear_history,
    config,
):
    """Return progress-aware gear costs and diagnostics.

    A reverse action is cheap when no safe forward-progress action exists.  It
    earns additional credit when a forward probe becomes possible afterwards.
    Thus R-F-R-F sequences remain valid; only reverse/switch actions without a
    measurable change in recovery potential accumulate a strong burden.
    """

    forward = candidate_gears > 0
    reverse = ~forward
    forward_progress = (
        forward
        & hard_feasible
        & (route_progress_m >= config.forward_available_progress_m)
    )
    forward_available = forward_progress.any(dim=1)
    burden_gate = torch.where(
        forward_available,
        path_distance_m.new_ones(len(path_distance_m)),
        path_distance_m.new_full(
            (len(path_distance_m),),
            float(config.blocked_reverse_burden_fraction),
        ),
    )[:, None]
    reverse_effort = (
        reverse.to(path_distance_m)
        * path_distance_m
        * burden_gate
        * float(config.reverse_distance_weight)
    )
    shift_effort = (
        shift_required.to(path_distance_m)
        * burden_gate
        * float(config.switch_effort_weight)
    )
    recent_switches = gear_history[:, 4].clamp_min(0.0)[:, None]
    repeated_switch = (
        shift_required.to(path_distance_m)
        * recent_switches
        * burden_gate
        * float(config.repeated_switch_weight)
    )
    unproductive_reverse = (
        reverse.to(path_distance_m)
        * (1.0 - forward_recovery_target)
        * burden_gate
        * float(config.unproductive_reverse_weight)
    )
    recovery_credit = (
        reverse.to(path_distance_m)
        * forward_recovery_target
        * (~forward_available)[:, None].to(path_distance_m)
        * float(config.recovery_credit_weight)
    )
    total = (
        reverse_effort
        + shift_effort
        + repeated_switch
        + unproductive_reverse
        - recovery_credit
    )
    return total, {
        "forward_available": forward_available,
        "reverse_effort": reverse_effort,
        "shift_effort": shift_effort,
        "repeated_switch": repeated_switch,
        "unproductive_reverse": unproductive_reverse,
        "recovery_credit": recovery_credit,
    }


def _compose_from_endpoints(parent, continuation):
    endpoint = parent[..., -1, :]
    px, py, pyaw = endpoint[..., 1], endpoint[..., 2], endpoint[..., 3]
    x, y = continuation[..., 1], continuation[..., 2]
    cosine = torch.cos(pyaw)[..., None, None]
    sine = torch.sin(pyaw)[..., None, None]
    result = continuation.clone()
    result[..., 0] = continuation[..., 0] + endpoint[..., None, None, 0]
    result[..., 1] = px[..., None, None] + cosine * x - sine * y
    result[..., 2] = py[..., None, None] + sine * x + cosine * y
    result[..., 3] = pyaw[..., None, None] + continuation[..., 3]
    return result


def forward_recovery_probe_target(
    output,
    *,
    map_distance_field,
    map_resolution,
    map_origin,
    chassis_to_map,
    route,
    route_mask,
    footprint,
    clearance_margin_m,
    minimum_progress_m,
):
    """Test three low-speed forward continuations from every candidate end."""

    trajectories = output.trajectories.detach()
    batch, candidates = trajectories.shape[:2]
    endpoint = trajectories[..., -1, :]
    flat_state = endpoint.new_zeros((batch * candidates, 9))
    flat_state[:, 0] = endpoint[..., 4].reshape(-1)
    flat_state[:, 2] = endpoint[..., 5].reshape(-1)
    flat_state[:, 3] = (
        flat_state[:, 0]
        * torch.tan(flat_state[:, 2])
        / float(PLANNER_ROLLOUT_WHEELBASE_M)
    )
    flat_gear = output.candidate_gears.reshape(-1)
    probe = BidirectionalAckermannRolloutV3()
    raw = endpoint.new_zeros((batch * candidates, 15, 4))
    forward_bank = probe._bank(flat_state, flat_gear, raw, 1)[0]
    # Lowest speed row, left/straight/right: enough to certify that the
    # reverse action opened a usable forward manoeuvre without a 450-path map
    # query per sample.
    forward_bank = forward_bank[:, (0, 2, 4)]
    forward_bank = forward_bank.reshape(
        batch, candidates, 3, forward_bank.shape[-2], 6
    )
    composed = _compose_from_endpoints(trajectories, forward_bank)
    flat = composed.reshape(
        batch, candidates * 3, composed.shape[-2], composed.shape[-1]
    )
    clearance = swept_map_footprint_clearance(
        flat,
        map_distance_field,
        map_resolution,
        map_origin,
        chassis_to_map,
        footprint=footprint,
    ).amin(dim=(-1, -2))
    relation = corridor_candidate_relations(flat, route, route_mask)
    progress = relation[..., 3] * 3.0
    parent_progress = corridor_candidate_relations(
        trajectories, route, route_mask
    )[..., 3] * 3.0
    capable = (
        (clearance > float(clearance_margin_m))
        & (
            progress
            >= parent_progress.repeat_interleave(3, dim=1)
            + float(minimum_progress_m)
        )
    )
    return capable.reshape(batch, candidates, 3).any(dim=-1).to(trajectories)


class DEPCarObjectiveV3:
    objective_id = "dep_car_objective_v6_joint_gear_conditional_reverse_value"
    objective_revision = 6
    stages = (
        "bidirectional_candidate_capacity",
        "joint_gear_score_calibration",
        "sequence_recovery_finetune",
        "joint_smoke",
    )

    def __init__(
        self,
        config=DEPCarJointGearLossConfigV3(),
        footprint=FootprintConfig(),
    ):
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
        gear_history,
        sequence_gears=None,
        sequence_mask=None,
        stage="bidirectional_candidate_capacity",
    ):
        if stage not in self.stages:
            raise ValueError("unknown V3 stage: %s" % stage)
        cfg, base, weights = self.config, self.config.base, self.config.base.weights
        safe_per, minimum_clearance = swept_map_footprint_loss(
            output.trajectories,
            map_distance_field,
            map_resolution,
            map_origin,
            chassis_to_map,
            footprint=self.footprint,
            temperature_m=base.safety_temperature_m,
            cvar_fraction=base.safety_cvar_fraction,
            mean_weight=base.safety_mean_weight,
            cvar_weight=base.safety_cvar_weight,
            worst_weight=base.safety_worst_weight,
        )
        route_cfg = type(
            "RouteConfig",
            (),
            {
                "route_tube_radius_m": cfg.route_tube_radius_m,
                "route_outer_tube_radius_m": cfg.route_outer_tube_radius_m,
                "minimum_one_cycle_progress_m": cfg.minimum_one_cycle_progress_m,
            },
        )()
        guidance_per = route_tube_loss_v2(
            output.trajectories, route, route_mask, route_cfg
        )
        margin_per = F.relu(cfg.clearance_margin_m - minimum_clearance).square()
        future = constant_control_future(output.trajectories, cfg)
        future_clearance = swept_map_footprint_clearance(
            future,
            map_distance_field,
            map_resolution,
            map_origin,
            chassis_to_map,
            footprint=self.footprint,
        ).amin(dim=(-1, -2))
        future_per = F.relu(cfg.clearance_margin_m - future_clearance).square()
        kinematic_per = bidirectional_kinematic_loss(
            output.trajectories, output.motion_gears, cfg
        )
        components = bidirectional_kinematic_components(
            output.trajectories,
            output.motion_gears,
            lateral_acceleration_limit_mps2=base.maximum_lateral_acceleration_mps2,
        )
        violation = torch.stack(tuple(components.values()), dim=0).any(dim=0)
        hard_feasible = (minimum_clearance.detach() > 0.0) & ~violation
        relation = corridor_candidate_relations(
            output.trajectories, route, route_mask
        )
        route_progress = relation[..., 3] * 3.0
        recovery_target = forward_recovery_probe_target(
            output,
            map_distance_field=map_distance_field,
            map_resolution=map_resolution,
            map_origin=map_origin,
            chassis_to_map=chassis_to_map,
            route=route,
            route_mask=route_mask,
            footprint=self.footprint,
            clearance_margin_m=cfg.clearance_margin_m,
            minimum_progress_m=cfg.forward_probe_progress_m,
        )
        gear_cost, gear_diagnostics = conditional_gear_costs(
            candidate_gears=output.candidate_gears,
            shift_required=output.shift_required,
            hard_feasible=hard_feasible,
            route_progress_m=route_progress.detach(),
            path_distance_m=_path_distance(output.trajectories).detach(),
            forward_recovery_target=recovery_target.detach(),
            gear_history=gear_history,
            config=cfg,
        )
        comfort_per = comfort_loss(output.trajectories)
        diversity_per = bank_diversity_loss(output.trajectories, cfg)
        anchor_per = output.residuals.square().mean(dim=(-1, -2))
        candidate_cost = (
            weights.safety * safe_per
            + weights.guidance * guidance_per
            + weights.kinematic * kinematic_per
            + weights.comfort * comfort_per
            + cfg.clearance_margin_weight * margin_per
            + cfg.future_viability_weight * future_per
            + gear_cost
        )
        top_forward = torch.topk(
            candidate_cost[:, :15], base.capacity_top_k, dim=1, largest=False
        ).values.mean(dim=1)
        top_reverse = torch.topk(
            candidate_cost[:, 15:], base.capacity_top_k, dim=1, largest=False
        ).values.mean(dim=1)
        candidate_loss = (
            0.5 * (top_forward + top_reverse).mean()
            + weights.safety * safe_per.mean()
            + weights.kinematic_all * kinematic_per.mean()
            + weights.diversity * diversity_per.mean()
            + weights.anchor * anchor_per.mean()
            + cfg.clearance_margin_weight * margin_per.mean()
            + cfg.future_viability_weight * future_per.mean()
        )
        score_loss = score_ranking_loss(
            output.scores,
            candidate_cost,
            feasible=hard_feasible,
            temperature=base.score_temperature,
        ).mean()
        recovery_loss = F.binary_cross_entropy(
            output.forward_recovery_value,
            recovery_target.detach(),
        )

        sequence_loss = output.scores.new_zeros(())
        if sequence_gears is not None and sequence_mask is not None:
            if (
                sequence_gears.shape != sequence_mask.shape
                or sequence_gears.shape[0] != len(output.scores)
                or sequence_gears.shape[1] != cfg.sequence_actions
            ):
                raise ValueError("V3 sequence gear tensors have invalid shape")
            # Logged gear runs certify temporal manoeuvre coverage, but the
            # old supervisor is deliberately not treated as a gear oracle.
            # The current bank target comes from the V3 counterfactual cost:
            # evaluate forward and reverse at the same frame, prefer safe
            # forward progress, and value reverse when it unlocks a forward
            # continuation.  Multi-action rows receive the additional bank
            # calibration; single-action rows remain governed by ranking loss.
            sequence_valid = sequence_mask.bool().sum(dim=1) >= 2
            if bool(sequence_valid.any()):
                bank_energy = torch.stack(
                    (
                        -torch.logsumexp(-output.scores[:, :15], dim=1),
                        -torch.logsumexp(-output.scores[:, 15:], dim=1),
                    ),
                    dim=1,
                )
                feasible_cost = candidate_cost.detach().masked_fill(
                    ~hard_feasible, torch.inf
                )
                forward_oracle = feasible_cost[:, :15].amin(dim=1)
                reverse_oracle = feasible_cost[:, 15:].amin(dim=1)
                forward_oracle = torch.where(
                    torch.isfinite(forward_oracle),
                    forward_oracle,
                    candidate_cost[:, :15].detach().amin(dim=1),
                )
                reverse_oracle = torch.where(
                    torch.isfinite(reverse_oracle),
                    reverse_oracle,
                    candidate_cost[:, 15:].detach().amin(dim=1),
                )
                target = (reverse_oracle < forward_oracle).long()
                sequence_loss = F.cross_entropy(
                    -bank_energy[sequence_valid], target[sequence_valid]
                )

        if stage == "bidirectional_candidate_capacity":
            total = candidate_loss
        elif stage == "joint_gear_score_calibration":
            total = score_loss + cfg.recovery_head_weight * recovery_loss
        elif stage == "sequence_recovery_finetune":
            total = (
                score_loss
                + cfg.recovery_head_weight * recovery_loss
                + cfg.sequence_gear_weight * sequence_loss
            )
        else:
            total = (
                candidate_loss
                + score_loss
                + cfg.recovery_head_weight * recovery_loss
                + cfg.sequence_gear_weight * sequence_loss
            )
        return {
            "total": total,
            "candidate": candidate_loss,
            "score": score_loss,
            "recovery": recovery_loss,
            "sequence": sequence_loss,
            "candidate_cost": candidate_cost.detach(),
            "minimum_clearance": minimum_clearance.detach(),
            "future_minimum_clearance": future_clearance.detach(),
            "hard_feasible": hard_feasible.detach(),
            "route_progress_m": route_progress.detach(),
            "forward_recovery_target": recovery_target.detach(),
            "kinematic_violation": violation.detach(),
            "kinematic_violation_by_constraint": {
                key: value.detach() for key, value in components.items()
            },
            "forward_available": gear_diagnostics["forward_available"].detach(),
            "gear_cost": gear_cost.detach(),
            "recovery_credit": gear_diagnostics["recovery_credit"].detach(),
        }


__all__ = [
    "DEPCarJointGearLossConfigV3",
    "DEPCarObjectiveV3",
    "bank_diversity_loss",
    "bidirectional_kinematic_components",
    "bidirectional_kinematic_loss",
    "conditional_gear_costs",
    "forward_recovery_probe_target",
]
