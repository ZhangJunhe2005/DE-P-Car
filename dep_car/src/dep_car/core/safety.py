"""Deterministic hard safety and risk ranking for Ackermann candidates."""

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np

from .occupancy import FootprintConfig, OccupancyGrid2D
from .types import Candidate, DynamicTrack
from .vehicle import DYNAMIC_EGO_RADIUS_M


@dataclass(frozen=True)
class DynamicSafetyConfig:
    ego_radius: float = DYNAMIC_EGO_RADIUS_M
    hard_margin: float = 0.30
    covariance_sigma: float = 2.0


def evaluate_static(candidate: Candidate, grid: OccupancyGrid2D, footprint: FootprintConfig) -> Candidate:
    safe, clearance = grid.swept_footprint_clearance(candidate.trajectory, footprint)
    candidate.static_clearance = clearance
    if not safe:
        candidate.feasible = False
        candidate.veto_reason = "static_footprint_collision"
    return candidate


def evaluate_static_margin_egress(
    candidate: Candidate,
    grid: OccupancyGrid2D,
    footprint: FootprintConfig,
    *,
    maximum_overlap_m: float,
    minimum_improvement_m: float,
    worsening_tolerance_m: float,
) -> Candidate:
    """Certify motion that exits a small soft-margin overlap.

    This never relaxes the physical footprint and is intentionally opt-in.
    It is used only by an explicitly requested, bounded recovery primitive.
    Normal navigation continues to require the complete footprint, including
    its safety margin, to be positive at every trajectory row.
    """

    if min(maximum_overlap_m, minimum_improvement_m, worsening_tolerance_m) < 0.0:
        raise ValueError("margin-egress thresholds cannot be negative")
    signed_profile = getattr(
        grid, "swept_footprint_signed_clearance_profile", None
    )
    if not callable(signed_profile):
        # The frozen P4 OccupancyGrid2D intentionally has no margin-egress
        # API.  Only the P6 runtime subclass can grant this narrow authority.
        return candidate
    strict = signed_profile(candidate.trajectory, footprint)
    physical = FootprintConfig(
        length=footprint.length,
        width=footprint.width,
        safety_margin=0.0,
        circle_count=footprint.circle_count,
    )
    physical_profile = signed_profile(candidate.trajectory, physical)
    start = float(strict[0])
    certified = (
        np.all(np.isfinite(strict))
        and np.all(np.isfinite(physical_profile))
        and -float(maximum_overlap_m) <= start <= 0.0
        and float(np.min(physical_profile)) > 0.0
        and float(np.min(strict)) >= start - float(worsening_tolerance_m)
        and float(strict[-1]) >= start + float(minimum_improvement_m)
    )
    if certified:
        candidate.feasible = True
        candidate.static_clearance = float(np.min(physical_profile))
        candidate.veto_reason = ""
    return candidate


def _uncertainty_radius(track: DynamicTrack, horizon: float, sigma: float) -> float:
    if track.covariance is None:
        return 0.10 * horizon
    covariance = np.asarray(track.covariance)
    if covariance.shape[0] < 2:
        return 0.10 * horizon
    return sigma * float(np.sqrt(max(0.0, np.max(np.linalg.eigvalsh(covariance[:2, :2])))))


def evaluate_dynamic(
    candidate: Candidate,
    tracks: Iterable[DynamicTrack],
    config: DynamicSafetyConfig = DynamicSafetyConfig(),
) -> Candidate:
    minimum = float("inf")
    for row in candidate.trajectory:
        time_s, x, y = row[:3]
        ego = np.asarray([x, y])
        for track in tracks:
            actor = track.position_at(float(time_s))
            separation = float(np.linalg.norm(ego - actor))
            required = config.ego_radius + track.radius + config.hard_margin + _uncertainty_radius(
                track, float(time_s), config.covariance_sigma
            )
            clearance = separation - required
            minimum = min(minimum, clearance)
            if clearance <= 0.0:
                candidate.feasible = False
                candidate.veto_reason = "dynamic_reachability_collision"
                candidate.dynamic_clearance = clearance
                return candidate
    candidate.dynamic_clearance = minimum
    return candidate


def goal_cost(
    candidate: Candidate,
    subgoal_body: Tuple[float, float],
    target_heading: float = None,
    target_steering: float = None,
) -> float:
    endpoint = candidate.trajectory[-1, 1:3]
    distance = float(np.linalg.norm(endpoint - np.asarray(subgoal_body, dtype=float)))
    terminal_yaw = float(candidate.trajectory[-1, 3])
    target_yaw = (
        float(np.arctan2(subgoal_body[1], subgoal_body[0]))
        if target_heading is None
        else float(target_heading)
    )
    heading = abs(float(np.arctan2(np.sin(terminal_yaw - target_yaw), np.cos(terminal_yaw - target_yaw))))
    steering = (
        0.0
        if target_steering is None
        else abs(float(candidate.trajectory[-1, 5]) - float(target_steering))
    )
    candidate.guidance_cost = distance + 0.35 * heading + 0.50 * steering
    return candidate.guidance_cost
