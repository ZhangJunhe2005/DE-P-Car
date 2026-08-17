"""Small immutable data contracts shared by planning and ROS adapters."""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional

import numpy as np


class Gear(IntEnum):
    """Discrete longitudinal direction used by planning and actuation."""

    REVERSE = -1
    NEUTRAL = 0
    FORWARD = 1

    @classmethod
    def require_drive(cls, value):
        gear = cls(int(value))
        if gear == cls.NEUTRAL:
            raise ValueError("a trajectory candidate requires forward or reverse gear")
        return gear


@dataclass(frozen=True)
class VehicleState:
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    speed: float = 0.0
    steering: float = 0.0
    acceleration: float = 0.0
    yaw_rate: float = 0.0
    stamp: float = 0.0


@dataclass
class Candidate:
    candidate_id: int
    speed_anchor: float
    steering_anchor: float
    duration: float
    trajectory: np.ndarray
    gear: Gear = Gear.FORWARD
    retime_factor: float = 1.0
    learned_score: float = 0.0
    guidance_cost: float = 0.0
    static_clearance: float = float("inf")
    dynamic_clearance: float = float("inf")
    feasible: bool = True
    veto_reason: str = ""

    @property
    def total_cost(self) -> float:
        risk = 0.0 if np.isinf(self.dynamic_clearance) else 1.0 / max(self.dynamic_clearance, 1e-3)
        return float(self.learned_score + self.guidance_cost + risk)


@dataclass(frozen=True)
class DynamicTrack:
    track_id: int
    x: float
    y: float
    vx: float
    vy: float
    radius: float = 0.35
    covariance: Optional[np.ndarray] = field(default=None, compare=False)
    confidence: float = 1.0
    stamp: float = 0.0

    def position_at(self, time_s: float) -> np.ndarray:
        return np.asarray([self.x + self.vx * time_s, self.y + self.vy * time_s])
