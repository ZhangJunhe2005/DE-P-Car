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


def goal_cost(candidate: Candidate, subgoal_body: Tuple[float, float]) -> float:
    endpoint = candidate.trajectory[-1, 1:3]
    distance = float(np.linalg.norm(endpoint - np.asarray(subgoal_body, dtype=float)))
    terminal_yaw = float(candidate.trajectory[-1, 3])
    target_yaw = float(np.arctan2(subgoal_body[1], subgoal_body[0]))
    heading = abs(float(np.arctan2(np.sin(terminal_yaw - target_yaw), np.cos(terminal_yaw - target_yaw))))
    candidate.guidance_cost = distance + 0.35 * heading
    return candidate.guidance_cost
