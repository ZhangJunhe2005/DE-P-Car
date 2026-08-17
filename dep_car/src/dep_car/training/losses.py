"""Differentiable P4 objectives for multimodal Ackermann candidates."""

import math
from dataclasses import dataclass

import torch
from torch.nn import functional as F

from dep_car.core.occupancy import (
    FootprintConfig,
    SWEPT_SUBSTEPS_PER_SEGMENT,
    TRAINING_ONE_DIAGONAL_MULTIPLIER,
)
from dep_car.core.state_contract import (
    ACCELERATION_LIMIT_MPS2,
    DECELERATION_LIMIT_MPS2,
)
from dep_car.core.vehicle import PLANNER_ROLLOUT_WHEELBASE_M, STEERING_OPERATING_LIMIT_RAD


@dataclass(frozen=True)
class DEPCarLossWeights:
    safety: float = 2.0
    guidance: float = 1.0
    kinematic: float = 0.5
    comfort: float = 0.05
    diversity: float = 0.20
    anchor: float = 0.01
    score: float = 1.0


@dataclass(frozen=True)
class DEPCarLossConfig:
    bev_extent_m: float = 8.0
    safety_temperature_m: float = 0.04
    safety_cvar_fraction: float = 0.10
    safety_mean_weight: float = 0.20
    safety_cvar_weight: float = 0.30
    safety_worst_weight: float = 0.50
    guidance_heading_weight: float = 0.25
    guidance_progress_weight: float = 0.50
    guidance_endpoint_weight: float = 0.20
    capacity_top_k: int = 3
    minimum_normalized_terminal_separation: float = 0.12
    forward_diversity_scales: tuple = (0.8, 0.3, 1.0)
    reverse_diversity_scales: tuple = (0.5, 0.1, 0.5)
    maximum_lateral_acceleration_mps2: float = 3.0
    score_temperature: float = 0.25
    weights: DEPCarLossWeights = DEPCarLossWeights()

    def validate(self):
        if self.bev_extent_m <= 0.0 or self.safety_temperature_m <= 0.0:
            raise ValueError("loss spatial scales must be positive")
        if not 0.0 < self.safety_cvar_fraction <= 1.0:
            raise ValueError("safety_cvar_fraction must be in (0,1]")
        safety_weights = (
            self.safety_mean_weight,
            self.safety_cvar_weight,
            self.safety_worst_weight,
        )
        if (
            any(not math.isfinite(value) or value < 0.0 for value in safety_weights)
            or not math.isclose(sum(safety_weights), 1.0, rel_tol=0.0, abs_tol=1.0e-9)
        ):
            raise ValueError("safety aggregation weights must be finite, non-negative and sum to one")
        if not 1 <= self.capacity_top_k <= 15:
            raise ValueError("capacity_top_k must be in [1,15]")
        for name, value in vars(self.weights).items():
            if value < 0.0:
                raise ValueError(f"loss weight {name} must be non-negative")


def _angle_difference(lhs, rhs):
    difference = lhs - rhs
    return torch.atan2(torch.sin(difference), torch.cos(difference))


def _densify_trajectories_se2(
    trajectories,
    substeps=SWEPT_SUBSTEPS_PER_SEGMENT,
):
    """Torch equivalent of the frozen NumPy continuous-sweep interpolation."""

    if trajectories.ndim != 4 or trajectories.shape[-1] < 4:
        raise ValueError("trajectories must have shape [B,C,T,>=4]")
    if trajectories.shape[-2] < 1:
        raise ValueError("trajectories must contain at least one time row")
    if isinstance(substeps, bool) or not isinstance(substeps, int) or substeps < 1:
        raise ValueError("swept interpolation substeps must be a positive integer")
    if trajectories.shape[-2] == 1:
        return trajectories

    alpha = torch.arange(
        substeps, device=trajectories.device, dtype=trajectories.dtype
    ) / float(substeps)
    start = trajectories[..., :-1, None, :]
    delta = trajectories[..., 1:, None, :] - start
    linear = start + alpha[None, None, None, :, None] * delta
    source_yaw_delta = trajectories[..., 1:, 3] - trajectories[..., :-1, 3]
    yaw_delta = torch.atan2(torch.sin(source_yaw_delta), torch.cos(source_yaw_delta))
    yaw = (
        trajectories[..., :-1, None, 3]
        + alpha[None, None, None, :] * yaw_delta[..., None]
    )
    dense = torch.cat((linear[..., :3], yaw[..., None], linear[..., 4:]), dim=-1)
    shape = (*trajectories.shape[:-2], -1, trajectories.shape[-1])
    dense = dense.reshape(shape)
    return torch.cat((dense, trajectories[..., -1:, :]), dim=-2)


def _swept_circle_centers(trajectories, footprint):
    trajectories = _densify_trajectories_se2(trajectories)
    offsets = trajectories.new_tensor(footprint.longitudinal_offsets)
    yaw = trajectories[..., 3]
    heading = torch.stack((torch.cos(yaw), torch.sin(yaw)), dim=-1)
    return trajectories[..., None, 1:3] + heading[..., None, :] * offsets[None, None, None, :, None]


def swept_footprint_clearance(
    trajectories,
    distance_field,
    *,
    extent_m=8.0,
    footprint=FootprintConfig(),
):
    """Sample obstacle distance under every circle in the swept footprint.

    ``distance_field`` is positive in known free space and negative in occupied
    or unknown BEV cells.  Outside the field is sampled as zero clearance and
    is therefore unsafe after footprint/grid inflation.
    """

    if trajectories.ndim != 4 or trajectories.shape[-1] < 4:
        raise ValueError("trajectories must have shape [B,15,T,>=4]")
    if distance_field.ndim != 4 or distance_field.shape[1] != 1:
        raise ValueError("distance_field must have shape [B,1,H,W]")
    if trajectories.shape[0] != distance_field.shape[0]:
        raise ValueError("trajectory and distance-field batches must match")
    centers = _swept_circle_centers(trajectories, footprint)
    grid = torch.stack((centers[..., 0], centers[..., 1]), dim=-1) / float(extent_m)
    batch, candidates, steps, circles = grid.shape[:4]
    grid = grid.reshape(batch, candidates * steps * circles, 1, 2)
    sampled = F.grid_sample(
        distance_field,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    ).reshape(batch, candidates, steps, circles)
    resolution = 2.0 * float(extent_m) / float(distance_field.shape[-1])
    # The OpenCV distance transform is defined at cell centres.  A continuous
    # bilinear query can over-estimate the underlying point distance by at most
    # half a cell diagonal, while the occupied cell itself extends another
    # half diagonal around its centre.  Reserving both terms makes the
    # differentiable training query conservative with respect to the runtime
    # floor-to-cell hard veto at grazing boundaries.
    cell_diagonal_allowance = (
        TRAINING_ONE_DIAGONAL_MULTIPLIER * (2.0 ** 0.5) * resolution
    )
    return sampled - float(footprint.circle_radius) - cell_diagonal_allowance


def swept_map_footprint_clearance(
    trajectories,
    map_distance_field,
    map_resolution,
    map_origin,
    chassis_to_map,
    *,
    footprint=FootprintConfig(),
):
    """Query a map-authoritative metric SDF for body-frame trajectories."""

    if map_distance_field.ndim != 4 or map_distance_field.shape[1] != 1:
        raise ValueError("map_distance_field must have shape [B,1,H,W]")
    batch = trajectories.shape[0]
    if map_resolution.shape not in ((batch,), (batch, 1)):
        raise ValueError("map_resolution must contain one value per sample")
    if map_origin.shape != (batch, 2) or chassis_to_map.shape != (batch, 4, 4):
        raise ValueError("map origin/transform batch shapes are invalid")
    centers = _swept_circle_centers(trajectories, footprint)
    rotation = chassis_to_map[:, :2, :2].to(centers)
    translation = chassis_to_map[:, :2, 3].to(centers)
    world = torch.einsum("bij,bntkj->bntki", rotation, centers)
    world = world + translation[:, None, None, None, :]
    height, width = map_distance_field.shape[-2:]
    resolution = map_resolution.reshape(batch, 1, 1, 1).to(centers)
    cell = (world - map_origin[:, None, None, None, :].to(centers)) / resolution[..., None]
    grid_x = 2.0 * cell[..., 0] / float(width) - 1.0
    grid_y = 2.0 * cell[..., 1] / float(height) - 1.0
    grid = torch.stack((grid_x, grid_y), dim=-1)
    candidates, steps, circles = grid.shape[1:4]
    grid = grid.reshape(batch, candidates * steps * circles, 1, 2)
    sampled = F.grid_sample(
        map_distance_field,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    ).reshape(batch, candidates, steps, circles)
    cell_diagonal_allowance = (
        TRAINING_ONE_DIAGONAL_MULTIPLIER
        * (2.0 ** 0.5)
        * map_resolution.reshape(batch, 1, 1, 1).to(sampled)
    )
    return sampled - float(footprint.circle_radius) - cell_diagonal_allowance


def _aggregate_swept_safety_penalty(
    clearance,
    *,
    temperature_m=0.04,
    cvar_fraction=0.10,
    mean_weight=0.20,
    cvar_weight=0.30,
    worst_weight=0.50,
):
    """Blend mean, worst-tail CVaR and the minimum-clearance barrier.

    A pure mean over ``time x footprint circles`` can dilute one collision by a
    factor of 55 for the default rollout.  The explicit worst term is the smooth
    barrier at the minimum clearance, while CVaR keeps gradients on the nearby
    hazardous portion of the sweep instead of only one arg-max location.
    """

    if clearance.ndim < 2 or clearance.shape[-1] < 1 or clearance.shape[-2] < 1:
        raise ValueError("swept clearance must have non-empty time and circle dimensions")
    if not math.isfinite(float(temperature_m)) or temperature_m <= 0.0:
        raise ValueError("safety temperature must be finite and positive")
    if not math.isfinite(float(cvar_fraction)) or not 0.0 < cvar_fraction <= 1.0:
        raise ValueError("safety CVaR fraction must be in (0,1]")
    weights = (float(mean_weight), float(cvar_weight), float(worst_weight))
    if (
        any(not math.isfinite(value) or value < 0.0 for value in weights)
        or not math.isclose(sum(weights), 1.0, rel_tol=0.0, abs_tol=1.0e-9)
    ):
        raise ValueError("safety aggregation weights must be finite, non-negative and sum to one")

    penalty = F.softplus(-clearance / temperature_m) * temperature_m
    flat = penalty.flatten(start_dim=-2)
    tail_count = max(1, int(math.ceil(flat.shape[-1] * float(cvar_fraction))))
    mean_penalty = flat.mean(dim=-1)
    cvar_penalty = torch.topk(flat, tail_count, dim=-1, largest=True).values.mean(dim=-1)
    worst_penalty = flat.amax(dim=-1)
    aggregate = (
        weights[0] * mean_penalty
        + weights[1] * cvar_penalty
        + weights[2] * worst_penalty
    )
    return aggregate, clearance.amin(dim=(-1, -2))


def swept_footprint_loss(
    trajectories,
    distance_field,
    *,
    extent_m=8.0,
    footprint=FootprintConfig(),
    temperature_m=0.04,
    cvar_fraction=0.10,
    mean_weight=0.20,
    cvar_weight=0.30,
    worst_weight=0.50,
):
    clearance = swept_footprint_clearance(
        trajectories, distance_field, extent_m=extent_m, footprint=footprint
    )
    return _aggregate_swept_safety_penalty(
        clearance,
        temperature_m=temperature_m,
        cvar_fraction=cvar_fraction,
        mean_weight=mean_weight,
        cvar_weight=cvar_weight,
        worst_weight=worst_weight,
    )


def swept_map_footprint_loss(
    trajectories,
    map_distance_field,
    map_resolution,
    map_origin,
    chassis_to_map,
    *,
    footprint=FootprintConfig(),
    temperature_m=0.04,
    cvar_fraction=0.10,
    mean_weight=0.20,
    cvar_weight=0.30,
    worst_weight=0.50,
):
    clearance = swept_map_footprint_clearance(
        trajectories,
        map_distance_field,
        map_resolution,
        map_origin,
        chassis_to_map,
        footprint=footprint,
    )
    return _aggregate_swept_safety_penalty(
        clearance,
        temperature_m=temperature_m,
        cvar_fraction=cvar_fraction,
        mean_weight=mean_weight,
        cvar_weight=cvar_weight,
        worst_weight=worst_weight,
    )


def route_guidance_loss(
    trajectories,
    route,
    route_mask,
    *,
    heading_weight=0.25,
    progress_weight=0.50,
    endpoint_weight=0.20,
):
    """Candidate route cost using route order, so reverse progress is valid."""

    if route.ndim != 3 or route.shape[-1] < 3:
        raise ValueError("route must have shape [B,R,>=3]")
    if route_mask.shape != route.shape[:2]:
        raise ValueError("route_mask shape must match route")
    if bool(torch.any(route_mask.sum(dim=1) < 2)):
        raise ValueError("each active route requires at least two poses")
    positions = trajectories[..., 1:3]
    route_xy = route[..., :2]
    pairwise = torch.linalg.vector_norm(
        positions[..., None, :] - route_xy[:, None, None, :, :], dim=-1
    )
    pairwise = pairwise.masked_fill(~route_mask[:, None, None, :], 1.0e4)
    cross_track, nearest = pairwise.min(dim=-1)
    route_yaw = route[..., 2]
    endpoint_nearest = nearest[..., -1]
    nearest_yaw = torch.gather(
        route_yaw[:, None, :].expand(-1, trajectories.shape[1], -1),
        2,
        endpoint_nearest.unsqueeze(-1),
    ).squeeze(-1)
    heading = _angle_difference(trajectories[..., -1, 3], nearest_yaw).abs()

    segment = torch.linalg.vector_norm(route_xy[:, 1:] - route_xy[:, :-1], dim=-1)
    valid_segment = route_mask[:, 1:] & route_mask[:, :-1]
    segment = segment * valid_segment.to(segment.dtype)
    arc = torch.cat((segment.new_zeros((len(route), 1)), torch.cumsum(segment, dim=1)), dim=1)
    total_arc = arc.amax(dim=1).clamp_min(1.0e-4)
    endpoint_progress = torch.gather(
        arc[:, None, :].expand(-1, trajectories.shape[1], -1),
        2,
        endpoint_nearest.unsqueeze(-1),
    ).squeeze(-1) / total_arc[:, None]
    last_index = route_mask.sum(dim=1).long() - 1
    route_endpoint = route_xy[torch.arange(len(route), device=route.device), last_index]
    endpoint_error = torch.linalg.vector_norm(
        trajectories[..., -1, 1:3] - route_endpoint[:, None, :], dim=-1
    )
    return (
        cross_track[..., 1:].mean(dim=-1)
        + heading_weight * heading
        + progress_weight * (1.0 - endpoint_progress)
        + endpoint_weight * endpoint_error
    )


def kinematic_loss(
    trajectories,
    requested_gear,
    *,
    wheelbase_m=PLANNER_ROLLOUT_WHEELBASE_M,
    steering_limit_rad=STEERING_OPERATING_LIMIT_RAD,
    steering_rate_limit_rad_s=0.75,
    forward_speed_limit_mps=2.5,
    reverse_speed_limit_mps=0.5,
    acceleration_limit_mps2=ACCELERATION_LIMIT_MPS2,
    deceleration_limit_mps2=DECELERATION_LIMIT_MPS2,
    lateral_acceleration_limit_mps2=3.0,
):
    time = trajectories[..., 0]
    speed = trajectories[..., 4]
    steering = trajectories[..., 5]
    dt = (time[..., 1:] - time[..., :-1]).clamp_min(1.0e-5)
    acceleration = (speed[..., 1:] - speed[..., :-1]) / dt
    steering_rate = (steering[..., 1:] - steering[..., :-1]) / dt
    gear = requested_gear.to(speed)[:, None, None]
    directed_acceleration = gear * acceleration
    speed_limit = torch.where(
        gear > 0,
        speed.new_tensor(forward_speed_limit_mps),
        speed.new_tensor(reverse_speed_limit_mps),
    )
    terms = (
        F.relu(-gear * speed).square().mean(dim=-1)
        + F.relu(speed.abs() - speed_limit).square().mean(dim=-1)
        + F.relu(steering.abs() - steering_limit_rad).square().mean(dim=-1)
        + F.relu(steering_rate.abs() - steering_rate_limit_rad_s).square().mean(dim=-1)
        + F.relu(directed_acceleration - acceleration_limit_mps2).square().mean(dim=-1)
        + F.relu(-directed_acceleration - deceleration_limit_mps2).square().mean(dim=-1)
    )
    lateral_acceleration = speed.square() * torch.tan(steering).abs() / wheelbase_m
    terms = terms + F.relu(
        lateral_acceleration - lateral_acceleration_limit_mps2
    ).square().mean(dim=-1)
    return terms


def kinematic_violation_mask(
    trajectories,
    requested_gear,
    *,
    wheelbase_m=PLANNER_ROLLOUT_WHEELBASE_M,
    steering_limit_rad=STEERING_OPERATING_LIMIT_RAD,
    steering_rate_limit_rad_s=0.75,
    forward_speed_limit_mps=2.5,
    reverse_speed_limit_mps=0.5,
    acceleration_limit_mps2=ACCELERATION_LIMIT_MPS2,
    deceleration_limit_mps2=DECELERATION_LIMIT_MPS2,
    lateral_acceleration_limit_mps2=3.0,
    tolerance=1.0e-6,
):
    """Return the explicit runtime-style kinematic veto for every candidate."""

    time = trajectories[..., 0]
    speed = trajectories[..., 4]
    steering = trajectories[..., 5]
    dt = (time[..., 1:] - time[..., :-1]).clamp_min(1.0e-5)
    acceleration = (speed[..., 1:] - speed[..., :-1]) / dt
    steering_rate = (steering[..., 1:] - steering[..., :-1]) / dt
    gear = requested_gear.to(speed)[:, None, None]
    directed_acceleration = gear * acceleration
    speed_limit = torch.where(
        gear > 0,
        speed.new_tensor(forward_speed_limit_mps),
        speed.new_tensor(reverse_speed_limit_mps),
    )
    lateral_acceleration = speed.square() * torch.tan(steering).abs() / wheelbase_m
    return (
        (gear * speed < -tolerance).any(dim=-1)
        | (speed.abs() > speed_limit + tolerance).any(dim=-1)
        | (steering.abs() > steering_limit_rad + tolerance).any(dim=-1)
        | (steering_rate.abs() > steering_rate_limit_rad_s + tolerance).any(dim=-1)
        | (directed_acceleration > acceleration_limit_mps2 + tolerance).any(dim=-1)
        | (directed_acceleration < -deceleration_limit_mps2 - tolerance).any(dim=-1)
        | (lateral_acceleration > lateral_acceleration_limit_mps2 + tolerance).any(dim=-1)
    )


def comfort_loss(trajectories):
    time = trajectories[..., 0]
    speed = trajectories[..., 4]
    steering = trajectories[..., 5]
    dt = (time[..., 1:] - time[..., :-1]).clamp_min(1.0e-5)
    acceleration = (speed[..., 1:] - speed[..., :-1]) / dt
    steering_rate = (steering[..., 1:] - steering[..., :-1]) / dt
    if trajectories.shape[-2] < 3:
        return acceleration.square().mean(dim=-1) + steering_rate.square().mean(dim=-1)
    mid_dt = 0.5 * (dt[..., 1:] + dt[..., :-1])
    jerk = (acceleration[..., 1:] - acceleration[..., :-1]) / mid_dt
    steering_acceleration = (steering_rate[..., 1:] - steering_rate[..., :-1]) / mid_dt
    return jerk.square().mean(dim=-1) + 0.25 * steering_acceleration.square().mean(dim=-1)


def candidate_diversity_loss(
    trajectories,
    requested_gear=None,
    minimum_normalized_separation=0.12,
    forward_scales=(0.8, 0.3, 1.0),
    reverse_scales=(0.5, 0.1, 0.5),
):
    """Penalize collapsed logical queries without biasing the short reverse bank.

    A fixed metric threshold over-penalizes reverse candidates because their
    calibrated 1 s envelope is intentionally much smaller than the forward
    envelope.  Normalize terminal ``x/y/yaw`` by frozen, gear-specific
    canonical envelopes before applying pairwise repulsion.
    """

    if requested_gear is None:
        requested_gear = torch.where(
            trajectories[..., -1, 4].mean(dim=1) >= 0,
            trajectories.new_ones(trajectories.shape[0]),
            -trajectories.new_ones(trajectories.shape[0]),
        )
    if requested_gear.shape != (trajectories.shape[0],):
        raise ValueError("requested_gear must have shape [B]")
    forward = trajectories.new_tensor(forward_scales)
    reverse = trajectories.new_tensor(reverse_scales)
    scales = torch.where((requested_gear > 0)[:, None], forward[None], reverse[None])
    terminal = trajectories[..., -1, 1:4] / scales[:, None, :]
    pairwise = torch.cdist(terminal, terminal)
    upper = torch.triu(
        torch.ones((15, 15), dtype=torch.bool, device=trajectories.device), diagonal=1
    )
    return F.relu(minimum_normalized_separation - pairwise[:, upper]).square().mean(dim=-1)


def score_ranking_loss(scores, target_cost, feasible=None, temperature=0.25):
    """Listwise lower-is-better calibration with all-infeasible fallback."""

    if scores.shape != target_cost.shape or scores.ndim != 2:
        raise ValueError("scores and target_cost must share shape [B,15]")
    teacher = target_cost.detach()
    if feasible is not None:
        feasible = feasible.bool()
        if feasible.shape != scores.shape:
            raise ValueError("feasible mask shape must match scores")
        any_feasible = feasible.any(dim=1, keepdim=True)
        effective = torch.where(any_feasible, feasible, torch.ones_like(feasible))
        teacher = teacher.masked_fill(~effective, 1.0e4)
    target_probability = torch.softmax(-teacher / temperature, dim=-1)
    return -(target_probability * F.log_softmax(-scores / temperature, dim=-1)).sum(dim=-1)


def _valid_mean(values, geometry_valid):
    if geometry_valid is None:
        return values.mean()
    weights = geometry_valid.to(values.dtype)
    while weights.ndim < values.ndim:
        weights = weights.unsqueeze(-1)
    return (values * weights).sum() / weights.expand_as(values).sum().clamp_min(1.0)


class DEPCarObjectiveV1:
    """Versioned P4 objective supporting the two P5 training stages."""

    objective_id = "dep_car_objective_v3_signed_sdf_cvar_continuous_swept_route_capacity_score"
    objective_revision = 3
    stages = ("candidate_capacity", "score_calibration", "joint_smoke")

    def __init__(self, config=DEPCarLossConfig(), footprint=FootprintConfig()):
        config.validate()
        self.config = config
        self.footprint = footprint

    def __call__(
        self,
        output,
        *,
        distance_field=None,
        map_distance_field=None,
        map_resolution=None,
        map_origin=None,
        chassis_to_map=None,
        route,
        route_mask,
        requested_gear,
        geometry_valid=None,
        stage="candidate_capacity",
    ):
        if stage not in self.stages:
            raise ValueError(f"unknown training stage: {stage}")
        cfg, weights = self.config, self.config.weights
        if map_distance_field is not None:
            if any(value is None for value in (map_resolution, map_origin, chassis_to_map)):
                raise ValueError("map-authoritative safety requires resolution, origin and transform")
            safe_per, minimum_clearance = swept_map_footprint_loss(
                output.trajectories,
                map_distance_field,
                map_resolution,
                map_origin,
                chassis_to_map,
                footprint=self.footprint,
                temperature_m=cfg.safety_temperature_m,
                cvar_fraction=cfg.safety_cvar_fraction,
                mean_weight=cfg.safety_mean_weight,
                cvar_weight=cfg.safety_cvar_weight,
                worst_weight=cfg.safety_worst_weight,
            )
        else:
            if distance_field is None:
                raise ValueError("a local or map-authoritative distance field is required")
            safe_per, minimum_clearance = swept_footprint_loss(
                output.trajectories,
                distance_field,
                extent_m=cfg.bev_extent_m,
                footprint=self.footprint,
                temperature_m=cfg.safety_temperature_m,
                cvar_fraction=cfg.safety_cvar_fraction,
                mean_weight=cfg.safety_mean_weight,
                cvar_weight=cfg.safety_cvar_weight,
                worst_weight=cfg.safety_worst_weight,
            )
        guide_per = route_guidance_loss(
            output.trajectories,
            route,
            route_mask,
            heading_weight=cfg.guidance_heading_weight,
            progress_weight=cfg.guidance_progress_weight,
            endpoint_weight=cfg.guidance_endpoint_weight,
        )
        kinematic_per = kinematic_loss(
            output.trajectories,
            requested_gear,
            lateral_acceleration_limit_mps2=cfg.maximum_lateral_acceleration_mps2,
        )
        kinematic_violation = kinematic_violation_mask(
            output.trajectories,
            requested_gear,
            lateral_acceleration_limit_mps2=cfg.maximum_lateral_acceleration_mps2,
        )
        comfort_per = comfort_loss(output.trajectories)
        diversity_per = candidate_diversity_loss(
            output.trajectories,
            requested_gear,
            cfg.minimum_normalized_terminal_separation,
            cfg.forward_diversity_scales,
            cfg.reverse_diversity_scales,
        )
        anchor_per = output.residuals.square().mean(dim=(-1, -2))
        capacity_per_candidate = (
            weights.safety * safe_per
            + weights.guidance * guide_per
            + weights.kinematic * kinematic_per
            + weights.comfort * comfort_per
        )
        top_k = torch.topk(
            capacity_per_candidate, cfg.capacity_top_k, dim=1, largest=False
        ).values.mean(dim=1)
        candidate_loss = (
            _valid_mean(top_k, geometry_valid)
            + weights.safety * _valid_mean(safe_per.mean(dim=1), geometry_valid)
            + weights.diversity * _valid_mean(diversity_per, geometry_valid)
            + weights.anchor * _valid_mean(anchor_per, geometry_valid)
        )
        score_per = score_ranking_loss(
            output.scores,
            capacity_per_candidate,
            # Score calibration must use the same hard-veto ordering as the
            # streaming/runtime metric: static swept-footprint clearance and
            # kinematic legality are checked before a learned cost is ranked.
            feasible=(minimum_clearance.detach() > 0.0) & ~kinematic_violation,
            temperature=cfg.score_temperature,
        )
        score_loss = _valid_mean(score_per, geometry_valid)
        if stage == "candidate_capacity":
            total = candidate_loss
        elif stage == "score_calibration":
            total = weights.score * score_loss
        else:
            total = candidate_loss + weights.score * score_loss
        return {
            "total": total,
            "candidate": candidate_loss,
            "score": score_loss,
            "safety": _valid_mean(safe_per.mean(dim=1), geometry_valid),
            "guidance": _valid_mean(guide_per.mean(dim=1), geometry_valid),
            "kinematic": _valid_mean(kinematic_per.mean(dim=1), geometry_valid),
            "comfort": _valid_mean(comfort_per.mean(dim=1), geometry_valid),
            "diversity": _valid_mean(diversity_per, geometry_valid),
            "anchor": _valid_mean(anchor_per, geometry_valid),
            "minimum_clearance": minimum_clearance.detach(),
            "candidate_cost": capacity_per_candidate.detach(),
            "safety_per_candidate": safe_per.detach(),
            "guidance_per_candidate": guide_per.detach(),
            "kinematic_per_candidate": kinematic_per.detach(),
            "kinematic_violation": kinematic_violation.detach(),
            "hard_feasible": ((minimum_clearance.detach() > 0.0) & ~kinematic_violation).detach(),
            "comfort_per_candidate": comfort_per.detach(),
        }
