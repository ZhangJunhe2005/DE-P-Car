"""Deterministic terminal-speed envelope and latched goal capture for P6."""

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class ArrivalState(str, Enum):
    TRACKING = "TRACKING"
    ACTIVE_BRAKING = "ACTIVE_BRAKING"
    HOLD = "HOLD"


@dataclass(frozen=True)
class ArrivalConfig:
    position_tolerance_m: float = 0.22
    heading_tolerance_rad: float = 0.35
    settled_speed_mps: float = 0.02
    approach_radius_m: float = 1.50
    stop_radius_m: float = 0.10
    comfortable_deceleration_mps2: float = 0.80
    maximum_approach_speed_mps: float = 0.80
    minimum_tracking_speed_mps: float = 0.10
    alignment_speed_mps: float = 0.25


@dataclass(frozen=True)
class ArrivalDecision:
    state: ArrivalState
    speed_limit_mps: Optional[float]

    @property
    def active_braking(self) -> bool:
        return self.state == ArrivalState.ACTIVE_BRAKING

    @property
    def hold(self) -> bool:
        return self.state == ArrivalState.HOLD


class GoalArrivalController:
    """Approach slowly, actively stop, then hold until a new goal arrives."""

    def __init__(self, config: ArrivalConfig = ArrivalConfig()):
        self.config = config
        self.state = ArrivalState.TRACKING

    def reset(self) -> None:
        self.state = ArrivalState.TRACKING

    def update(
        self,
        distance_m: float,
        heading_error_rad: float,
        signed_speed_mps: float,
        *,
        heading_required: bool = True,
    ) -> ArrivalDecision:
        cfg = self.config
        distance = max(0.0, float(distance_m))
        heading = abs(float(heading_error_rad))
        speed = abs(float(signed_speed_mps))
        if self.state == ArrivalState.HOLD:
            return ArrivalDecision(self.state, 0.0)
        if self.state == ArrivalState.ACTIVE_BRAKING:
            if speed <= cfg.settled_speed_mps:
                self.state = ArrivalState.HOLD
            return ArrivalDecision(self.state, 0.0)
        if distance <= cfg.position_tolerance_m and (
            not heading_required or heading <= cfg.heading_tolerance_rad
        ):
            self.state = (
                ArrivalState.HOLD
                if speed <= cfg.settled_speed_mps
                else ArrivalState.ACTIVE_BRAKING
            )
            return ArrivalDecision(self.state, 0.0)
        if distance >= cfg.approach_radius_m:
            return ArrivalDecision(self.state, None)
        remaining = max(0.0, distance - cfg.stop_radius_m)
        limit = math.sqrt(
            2.0 * cfg.comfortable_deceleration_mps2 * remaining
        )
        limit = min(cfg.maximum_approach_speed_mps, limit)
        if heading_required and heading > cfg.heading_tolerance_rad:
            limit = min(limit, cfg.alignment_speed_mps)
        limit = max(cfg.minimum_tracking_speed_mps, limit)
        return ArrivalDecision(self.state, limit)
