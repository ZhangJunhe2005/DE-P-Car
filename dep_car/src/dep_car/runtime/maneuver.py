"""Committed multi-leg Ackermann maneuver state for tight P6 geometry."""

import math
from dataclasses import dataclass
from enum import Enum
from typing import Tuple

import numpy as np

from dep_car.core.types import Gear


class ManeuverState(str, Enum):
    IDLE = "IDLE"
    DRIVE_LEG = "DRIVE_LEG"
    SETTLING = "SETTLING"


@dataclass(frozen=True)
class ManeuverConfig:
    base_leg_distance_m: float = 0.85
    maximum_leg_distance_m: float = 1.15
    minimum_useful_leg_m: float = 0.25
    maximum_legs: int = 8
    leg_timeout_s: float = 8.0
    lateral_offset_m: float = 0.35
    terminal_lateral_offset_m: float = 0.90
    settled_speed_mps: float = 0.04


class CommittedManeuver:
    """Keep one gear long enough to create turning radius before switching."""

    def __init__(self, config: ManeuverConfig = ManeuverConfig()):
        self.config = config
        self.reset()

    def reset(self) -> None:
        self.state = ManeuverState.IDLE
        self.gear = Gear.NEUTRAL
        self.last_completed_gear = Gear.NEUTRAL
        self.turn_sign = 0.0
        self.target_distance_m = 0.0
        self.travelled_m = 0.0
        self.leg_count = 0
        self.started_at = 0.0
        self.last_position = None
        self.finish_reason = ""
        self.purpose = ""

    @property
    def active(self) -> bool:
        return self.state != ManeuverState.IDLE

    @property
    def driving(self) -> bool:
        return self.state == ManeuverState.DRIVE_LEG

    @staticmethod
    def turn_direction(subgoal_body, heading_error_rad: float) -> float:
        lateral = float(subgoal_body[1])
        value = lateral if abs(lateral) >= 0.10 else float(heading_error_rad)
        return 1.0 if value >= 0.0 else -1.0

    def required_distance(self, subgoal_body, heading_error_rad: float) -> float:
        bearing = abs(math.atan2(float(subgoal_body[1]), max(0.10, abs(float(subgoal_body[0])))))
        severity = min(1.0, max(abs(float(heading_error_rad)), bearing) / (0.5 * math.pi))
        return min(
            self.config.maximum_leg_distance_m,
            self.config.base_leg_distance_m + 0.30 * severity,
        )

    def begin(
        self,
        gear: Gear,
        position,
        now: float,
        subgoal_body,
        heading_error_rad: float,
        purpose: str = "static_recovery",
    ) -> bool:
        if self.leg_count >= self.config.maximum_legs:
            return False
        self.gear = Gear.require_drive(gear)
        self.turn_sign = self.turn_direction(subgoal_body, heading_error_rad)
        self.target_distance_m = self.required_distance(
            subgoal_body, heading_error_rad
        )
        self.travelled_m = 0.0
        self.started_at = float(now)
        self.last_position = np.asarray(position, dtype=float).copy()
        self.finish_reason = ""
        self.purpose = str(purpose)
        self.leg_count += 1
        self.state = ManeuverState.DRIVE_LEG
        return True

    def observe(self, position, now: float) -> None:
        if not self.driving:
            return
        current = np.asarray(position, dtype=float)
        if self.last_position is not None:
            self.travelled_m += float(np.linalg.norm(current - self.last_position))
        self.last_position = current.copy()
        if self.travelled_m >= self.target_distance_m:
            self.settle("target_distance_reached")
        elif float(now) - self.started_at >= self.config.leg_timeout_s:
            self.settle("leg_timeout")

    def settle(self, reason: str) -> None:
        if self.active:
            self.finish_reason = str(reason)
            self.state = ManeuverState.SETTLING

    def finish_if_stopped(self, signed_speed_mps: float) -> bool:
        if (
            self.state == ManeuverState.SETTLING
            and abs(float(signed_speed_mps)) <= self.config.settled_speed_mps
        ):
            self.last_completed_gear = self.gear
            self.state = ManeuverState.IDLE
            self.gear = Gear.NEUTRAL
            self.last_position = None
            return True
        return False

    def recovery_gear_order(self, requested_gear: Gear):
        """Prefer alternating legs while a tight-space sequence is active."""

        requested = Gear.require_drive(requested_gear)
        opposite = Gear.REVERSE if requested == Gear.FORWARD else Gear.FORWARD
        if self.last_completed_gear == Gear.FORWARD:
            return Gear.REVERSE, Gear.FORWARD
        if self.last_completed_gear == Gear.REVERSE:
            return Gear.FORWARD, Gear.REVERSE
        return requested, opposite

    def body_subgoal(self) -> Tuple[float, float]:
        remaining = max(
            self.config.minimum_useful_leg_m,
            self.target_distance_m - self.travelled_m,
        )
        x = remaining if self.gear == Gear.FORWARD else -remaining
        # The steering sign must reverse with longitudinal direction to keep
        # rotating the body toward the same desired corner.
        lateral_offset = (
            self.config.terminal_lateral_offset_m
            if self.purpose == "terminal_alignment"
            else self.config.lateral_offset_m
        )
        y = self.turn_sign * lateral_offset
        if self.gear == Gear.REVERSE:
            y = -y
        return float(x), float(y)

    def proposed_subgoal(
        self,
        gear: Gear,
        subgoal_body,
        heading_error_rad: float,
        purpose: str = "static_recovery",
    ) -> Tuple[float, float]:
        gear = Gear.require_drive(gear)
        distance = self.required_distance(subgoal_body, heading_error_rad)
        turn_sign = self.turn_direction(subgoal_body, heading_error_rad)
        x = distance if gear == Gear.FORWARD else -distance
        lateral_offset = (
            self.config.terminal_lateral_offset_m
            if purpose == "terminal_alignment"
            else self.config.lateral_offset_m
        )
        y = turn_sign * lateral_offset
        if gear == Gear.REVERSE:
            y = -y
        return float(x), float(y)
