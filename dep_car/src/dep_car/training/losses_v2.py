"""Route-tube and future-viability objective for DEPCarNetV2."""

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from dep_car.core.occupancy import FootprintConfig
from dep_car.core.state_contract import (
    ACCELERATION_LIMIT_MPS2,
    DECELERATION_LIMIT_MPS2,
    FORWARD_SPEED_LIMIT_MPS,
    REVERSE_SPEED_LIMIT_MPS,
)
from dep_car.core.vehicle import PLANNER_ROLLOUT_WHEELBASE_M, STEERING_OPERATING_LIMIT_RAD
from dep_car.model.dep_car_net_v2 import corridor_candidate_relations

from .losses import (
    DEPCarLossConfig,
    _valid_mean,
    candidate_diversity_loss,
    comfort_loss,
    kinematic_loss,
    kinematic_violation_components,
    score_ranking_loss,
    swept_map_footprint_clearance,
    swept_map_footprint_loss,
)


@dataclass(frozen=True)
class DEPCarRouteLossConfigV2:
    base: DEPCarLossConfig = DEPCarLossConfig()
    route_tube_radius_m: float = 0.30
    route_outer_tube_radius_m: float = 0.45
    minimum_one_cycle_progress_m: float = 0.05
    clearance_margin_m: float = 0.08
    future_horizon_s: float = 0.60
    future_steps: int = 7
    clearance_margin_weight: float = 2.0
    future_viability_weight: float = 1.5

    def validate(self):
        self.base.validate()
        if not 0.0 < self.route_tube_radius_m < self.route_outer_tube_radius_m:
            raise ValueError("route tube radii are invalid")
        if min(
            self.clearance_margin_m,
            self.future_horizon_s,
            self.clearance_margin_weight,
            self.future_viability_weight,
        ) <= 0.0 or self.future_steps < 2:
            raise ValueError("V2 route/safety parameters are invalid")


def route_tube_loss_v2(trajectories, route, route_mask, config):
    relation = corridor_candidate_relations(trajectories, route, route_mask)
    mean_cross = relation[..., 0] * 3.0
    maximum_cross = relation[..., 1] * 3.0
    progress = relation[..., 3] * 3.0
    heading = torch.atan2(relation[..., 4], relation[..., 5]).abs()
    return (
        F.relu(mean_cross - config.route_tube_radius_m).square()
        + 0.5 * F.relu(maximum_cross - config.route_outer_tube_radius_m).square()
        + 0.15 * heading.square()
        + 0.30 * F.relu(config.minimum_one_cycle_progress_m - progress).square()
    )


def constant_control_future(trajectories, config):
    """Differentiably expose collisions just beyond the one-second horizon."""

    endpoint = trajectories[..., -1, :]
    dt = float(config.future_horizon_s) / float(config.future_steps - 1)
    x, y, yaw = endpoint[..., 1], endpoint[..., 2], endpoint[..., 3]
    speed, steering = endpoint[..., 4], endpoint[..., 5]
    rows = [endpoint]
    for index in range(1, config.future_steps):
        yaw_rate = speed * torch.tan(steering) / PLANNER_ROLLOUT_WHEELBASE_M
        next_yaw = yaw + yaw_rate * dt
        mid_yaw = yaw + 0.5 * yaw_rate * dt
        x = x + speed * torch.cos(mid_yaw) * dt
        y = y + speed * torch.sin(mid_yaw) * dt
        yaw = next_yaw
        rows.append(
            torch.stack(
                (
                    endpoint[..., 0] + index * dt,
                    x,
                    y,
                    yaw,
                    speed,
                    steering,
                ),
                dim=-1,
            )
        )
    return torch.stack(rows, dim=-2)


class DEPCarObjectiveV2:
    objective_id = "dep_car_objective_v5_route_tube_future_viability"
    objective_revision = 5
    stages = ("candidate_capacity", "score_calibration", "joint_smoke")

    def __init__(self, config=DEPCarRouteLossConfigV2(), footprint=FootprintConfig()):
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
        requested_gear,
        geometry_valid=None,
        stage="candidate_capacity",
        **_unused,
    ):
        if stage not in self.stages:
            raise ValueError("unknown training stage: %s" % stage)
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
        guidance_per = route_tube_loss_v2(
            output.trajectories, route, route_mask, cfg
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

        operating_scale = 1.0 - base.kinematic_safety_margin_fraction
        kinematic_per = kinematic_loss(
            output.trajectories,
            requested_gear,
            steering_limit_rad=STEERING_OPERATING_LIMIT_RAD * operating_scale,
            steering_rate_limit_rad_s=0.75 * operating_scale,
            forward_speed_limit_mps=FORWARD_SPEED_LIMIT_MPS * operating_scale,
            reverse_speed_limit_mps=REVERSE_SPEED_LIMIT_MPS * operating_scale,
            acceleration_limit_mps2=ACCELERATION_LIMIT_MPS2 * operating_scale,
            deceleration_limit_mps2=DECELERATION_LIMIT_MPS2 * operating_scale,
            lateral_acceleration_limit_mps2=(
                base.maximum_lateral_acceleration_mps2 * operating_scale
            ),
        )
        violation_components = kinematic_violation_components(
            output.trajectories,
            requested_gear,
            lateral_acceleration_limit_mps2=base.maximum_lateral_acceleration_mps2,
        )
        violation = torch.stack(tuple(violation_components.values()), dim=0).any(dim=0)
        comfort_per = comfort_loss(output.trajectories)
        diversity_per = candidate_diversity_loss(
            output.trajectories,
            requested_gear,
            base.minimum_normalized_terminal_separation,
            base.forward_diversity_scales,
            base.reverse_diversity_scales,
        )
        anchor_per = output.residuals.square().mean(dim=(-1, -2))
        candidate_cost = (
            weights.safety * safe_per
            + weights.guidance * guidance_per
            + weights.kinematic * kinematic_per
            + weights.comfort * comfort_per
            + cfg.clearance_margin_weight * margin_per
            + cfg.future_viability_weight * future_per
        )
        top_k = torch.topk(
            candidate_cost, base.capacity_top_k, dim=1, largest=False
        ).values.mean(dim=1)
        all_kinematic = _valid_mean(kinematic_per.mean(dim=1), geometry_valid)
        candidate_loss = (
            _valid_mean(top_k, geometry_valid)
            + weights.safety * _valid_mean(safe_per.mean(dim=1), geometry_valid)
            + weights.kinematic_all * all_kinematic
            + weights.diversity * _valid_mean(diversity_per, geometry_valid)
            + weights.anchor * _valid_mean(anchor_per, geometry_valid)
            + cfg.clearance_margin_weight * _valid_mean(margin_per.mean(dim=1), geometry_valid)
            + cfg.future_viability_weight * _valid_mean(future_per.mean(dim=1), geometry_valid)
        )
        hard_feasible = (minimum_clearance.detach() > 0.0) & ~violation
        score_per = score_ranking_loss(
            output.scores,
            candidate_cost,
            feasible=hard_feasible,
            temperature=base.score_temperature,
        )
        score_loss = _valid_mean(score_per, geometry_valid)
        total = (
            candidate_loss if stage == "candidate_capacity"
            else weights.score * score_loss if stage == "score_calibration"
            else candidate_loss + weights.score * score_loss
        )
        return {
            "total": total,
            "candidate": candidate_loss,
            "score": score_loss,
            "safety": _valid_mean(safe_per.mean(dim=1), geometry_valid),
            "guidance": _valid_mean(guidance_per.mean(dim=1), geometry_valid),
            "kinematic": all_kinematic,
            "comfort": _valid_mean(comfort_per.mean(dim=1), geometry_valid),
            "diversity": _valid_mean(diversity_per, geometry_valid),
            "anchor": _valid_mean(anchor_per, geometry_valid),
            "clearance_margin": _valid_mean(margin_per.mean(dim=1), geometry_valid),
            "future_viability": _valid_mean(future_per.mean(dim=1), geometry_valid),
            "minimum_clearance": minimum_clearance.detach(),
            "future_minimum_clearance": future_clearance.detach(),
            "candidate_cost": candidate_cost.detach(),
            "safety_per_candidate": safe_per.detach(),
            "guidance_per_candidate": guidance_per.detach(),
            "kinematic_per_candidate": kinematic_per.detach(),
            "kinematic_violation": violation.detach(),
            "kinematic_violation_by_constraint": {
                name: value.detach() for name, value in violation_components.items()
            },
            "hard_feasible": hard_feasible.detach(),
            "comfort_per_candidate": comfort_per.detach(),
        }


__all__ = [
    "DEPCarObjectiveV2",
    "DEPCarRouteLossConfigV2",
    "constant_control_future",
    "route_tube_loss_v2",
]
