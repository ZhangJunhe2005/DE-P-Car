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
    # A forward-restoration turn must reserve room for its opposite-gear
    # continuation.  The longer adaptive leg remains available to ordinary
    # corner/dead-end recovery where backing far enough is the objective.
    forward_restoration_leg_distance_m: float = 0.85
    minimum_useful_leg_m: float = 0.25
    maximum_legs: int = 8
    leg_timeout_s: float = 8.0
    lateral_offset_m: float = 0.35
    terminal_lateral_offset_m: float = 0.90
    settled_speed_mps: float = 0.04
    settled_steering_rad: float = 0.08
    # Once the refreshed route is already near the front of the car, an
    # opposite-gear space-creation leg must preserve that alignment.  Full
    # counter-steer is reserved for a genuinely lateral/rear corridor.
    reverse_alignment_deadband_rad: float = math.radians(50.0)
    reverse_alignment_full_steer_rad: float = math.radians(100.0)


class MeasuredPoseReplanGate:
    """Request one global replan per materially different measured pose.

    Repeating an identical Hybrid-A* solution while the local hard-safety
    layer holds the car stationary is not progress.  After one measured-pose
    retry, local recovery gets authority until it has moved far enough to make
    another global solve meaningfully different.
    """

    def __init__(self, minimum_displacement_m: float = 0.25):
        minimum_displacement_m = float(minimum_displacement_m)
        if not math.isfinite(minimum_displacement_m) or minimum_displacement_m <= 0.0:
            raise ValueError("minimum_displacement_m must be finite and positive")
        self.minimum_displacement_m = minimum_displacement_m
        self.reset()

    def reset(self) -> None:
        self.last_requested_position = None

    def authorize(self, position) -> bool:
        current = np.asarray(position, dtype=float)
        if current.shape != (2,) or not np.all(np.isfinite(current)):
            raise ValueError("replan position must be a finite 2-vector")
        if self.last_requested_position is not None:
            displacement = float(np.linalg.norm(current - self.last_requested_position))
            if displacement < self.minimum_displacement_m:
                return False
        self.last_requested_position = current.copy()
        return True


class RouteRecoveryReplanGate:
    """Hold one failed local recovery until its route transaction changes.

    A committed recovery leg can be geometrically valid when it is selected
    and then become non-executable after the vehicle has stopped and changed
    gear.  Recreating that same zero-motion leg at 10 Hz only consumes the
    manoeuvre budget while the memory/FAR layer is told that a local
    transaction still owns motion.  This gate releases that ownership after
    one no-progress failure and permits another attempt only after a new route
    transaction arrives or the measured pose has materially changed.
    """

    def __init__(self, minimum_displacement_m: float = 0.25):
        minimum_displacement_m = float(minimum_displacement_m)
        if not math.isfinite(minimum_displacement_m) or minimum_displacement_m <= 0.0:
            raise ValueError("minimum_displacement_m must be finite and positive")
        self.minimum_displacement_m = minimum_displacement_m
        self.reset()

    def reset(self) -> None:
        self.failed_route_key = None
        self.failed_position = None

    def block(self, route_key, position) -> None:
        current = np.asarray(position, dtype=float)
        if current.shape != (2,) or not np.all(np.isfinite(current)):
            raise ValueError("failed recovery position must be a finite 2-vector")
        self.failed_route_key = tuple(route_key)
        self.failed_position = current.copy()

    def held(self, route_key, position) -> bool:
        if self.failed_route_key is None:
            return False
        current = np.asarray(position, dtype=float)
        if current.shape != (2,) or not np.all(np.isfinite(current)):
            raise ValueError("recovery position must be a finite 2-vector")
        if tuple(route_key) != self.failed_route_key:
            self.reset()
            return False
        if float(np.linalg.norm(current - self.failed_position)) >= self.minimum_displacement_m:
            self.reset()
            return False
        return True


class CommittedManeuver:
    """Keep one gear long enough to create turning radius before switching."""

    def __init__(self, config: ManeuverConfig = ManeuverConfig()):
        self.config = config
        self.reset()

    def reset(self) -> None:
        self.state = ManeuverState.IDLE
        self.gear = Gear.NEUTRAL
        self.last_completed_gear = Gear.NEUTRAL
        self.last_completed_reason = ""
        self.turn_sign = 0.0
        self.lateral_target_m = 0.0
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

    @property
    def exhausted(self) -> bool:
        """Whether this transaction has consumed its bounded leg budget."""

        return self.leg_count >= self.config.maximum_legs

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
        turn_sign_hint=None,
    ) -> bool:
        if self.leg_count >= self.config.maximum_legs:
            return False
        purpose = str(purpose)
        continuing_forward_restoration = (
            purpose == "forward_restoration"
            and self.purpose == purpose
            and self.leg_count > 0
            and self.turn_sign != 0.0
        )
        self.gear = Gear.require_drive(gear)
        if not continuing_forward_restoration:
            if turn_sign_hint is None:
                self.turn_sign = self.turn_direction(
                    subgoal_body, heading_error_rad
                )
            else:
                hint = float(turn_sign_hint)
                if not math.isfinite(hint) or abs(hint) < 1.0e-6:
                    raise ValueError("turn_sign_hint must be finite and nonzero")
                self.turn_sign = 1.0 if hint > 0.0 else -1.0
        self.target_distance_m = self.required_distance(subgoal_body, heading_error_rad)
        if purpose == "forward_restoration":
            self.target_distance_m = min(
                self.target_distance_m,
                self.config.forward_restoration_leg_distance_m,
            )
        self.lateral_target_m = (
            self._lateral_offset(
                self.gear,
                self.target_distance_m,
                heading_error_rad,
                purpose,
            )
        )
        self.travelled_m = 0.0
        self.started_at = float(now)
        self.last_position = np.asarray(position, dtype=float).copy()
        self.finish_reason = ""
        self.purpose = purpose
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

    def hold_for_drive_authorization(self, position, now: float) -> None:
        """Pause leg distance and timeout while stopping/changing gear.

        ``begin`` records the intended leg before GearSupervisor has
        necessarily completed the shift.  Motion in the old gear must not be
        counted toward the new leg, and the shift dwell must not consume the
        leg timeout.
        """

        if not self.driving:
            return
        current = np.asarray(position, dtype=float)
        if current.shape != (2,) or not np.all(np.isfinite(current)):
            raise ValueError("maneuver position must be a finite 2-vector")
        self.started_at = float(now)
        self.last_position = current.copy()

    def settle(self, reason: str) -> None:
        if self.active:
            self.finish_reason = str(reason)
            self.state = ManeuverState.SETTLING

    def finish_if_stopped(
        self, signed_speed_mps: float, steering_angle_rad: float = 0.0
    ) -> bool:
        if (
            self.state == ManeuverState.SETTLING
            and abs(float(signed_speed_mps)) <= self.config.settled_speed_mps
            and abs(float(steering_angle_rad)) <= self.config.settled_steering_rad
        ):
            self.last_completed_gear = self.gear
            self.last_completed_reason = self.finish_reason
            self.state = ManeuverState.IDLE
            self.gear = Gear.NEUTRAL
            self.last_position = None
            return True
        return False

    def renew_leg_budget(self) -> bool:
        """Grant one measured-pose continuation without changing turn intent.

        The first bounded set of legs is a scheduling budget, not proof that
        the manoeuvre is geometrically impossible.  A fresh route solve may
        therefore renew the counter while retaining the chosen turn side and
        the last completed gear.  Callers remain responsible for limiting the
        number of renewals.
        """

        if self.active or self.purpose != "forward_restoration":
            return False
        self.leg_count = 0
        self.finish_reason = ""
        self.travelled_m = 0.0
        self.target_distance_m = 0.0
        self.last_position = None
        return True

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
            self.lateral_target_m
            if self.purpose == "forward_restoration"
            else
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
        turn_sign_hint=None,
    ) -> Tuple[float, float]:
        gear = Gear.require_drive(gear)
        distance = self.required_distance(subgoal_body, heading_error_rad)
        purpose = str(purpose)
        if purpose == "forward_restoration":
            distance = min(
                distance,
                self.config.forward_restoration_leg_distance_m,
            )
        if (
            purpose == "forward_restoration"
            and self.purpose == purpose
            and self.leg_count > 0
            and self.turn_sign != 0.0
        ):
            turn_sign = self.turn_sign
        elif turn_sign_hint is None:
            turn_sign = self.turn_direction(subgoal_body, heading_error_rad)
        else:
            hint = float(turn_sign_hint)
            if not math.isfinite(hint) or abs(hint) < 1.0e-6:
                raise ValueError("turn_sign_hint must be finite and nonzero")
            turn_sign = 1.0 if hint > 0.0 else -1.0
        x = distance if gear == Gear.FORWARD else -distance
        lateral_offset = self._lateral_offset(
            gear, distance, heading_error_rad, purpose
        )
        y = turn_sign * lateral_offset
        if gear == Gear.REVERSE:
            y = -y
        return float(x), float(y)

    def _lateral_offset(
        self, gear: Gear, distance_m: float, heading_error_rad: float, purpose: str
    ) -> float:
        """Scale the forward exit arc to the remaining corridor angle."""

        if str(purpose) == "terminal_alignment":
            return float(self.config.terminal_lateral_offset_m)
        base = float(self.config.lateral_offset_m)
        heading = float(heading_error_rad)
        if str(purpose) == "forward_restoration":
            absolute_heading = abs(heading)
            if absolute_heading < 0.05:
                return 0.0
            # Keep the geometric estimate finite near a perpendicular route;
            # beyond that angle the configured full lateral target applies.
            bounded_heading = min(absolute_heading, math.radians(89.0))
            metric_lateral = abs(float(distance_m) * math.tan(bounded_heading))
            scaled = min(base, max(0.05, metric_lateral))
            if Gear.require_drive(gear) == Gear.FORWARD:
                return scaled if absolute_heading < 0.5 * math.pi else base
            # Reversing after the body is already aimed into the FAR corridor
            # is only a clearance operation.  Do not undo the achieved yaw
            # with the fixed maximum lateral offset used for a true U-turn.
            if absolute_heading <= self.config.reverse_alignment_deadband_rad:
                return 0.0
            span = max(
                1.0e-6,
                self.config.reverse_alignment_full_steer_rad
                - self.config.reverse_alignment_deadband_rad,
            )
            strength = min(
                1.0,
                max(
                    0.0,
                    (absolute_heading - self.config.reverse_alignment_deadband_rad)
                    / span,
                ),
            )
            return scaled * strength
        return base
