"""Human-like forward-restoration policy for topological route corridors.

The global planner supplies connectivity, not a persistent transmission
command.  This module turns the *local geometry* of that corridor into a
small, auditable state machine: drive forward when possible, perform a local
multi-leg turnaround when the corridor is genuinely behind the car, and use
reverse only as a bounded escape until turnaround primitives become safe.
"""

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np

from dep_car.core.types import Gear


class ForwardPreferenceState(str, Enum):
    FORWARD_CRUISE = "FORWARD_CRUISE"
    TURNAROUND_PENDING = "TURNAROUND_PENDING"
    REVERSE_ESCAPE = "REVERSE_ESCAPE"
    REVERSE_ESCAPE_EXHAUSTED = "REVERSE_ESCAPE_EXHAUSTED"


@dataclass(frozen=True)
class ForwardPreferenceConfig:
    direction_lookahead_m: float = 0.75
    behind_bearing_rad: float = math.radians(110.0)
    # Use hysteresis: entering a turnaround at 110 degrees but declaring it
    # complete only after the local corridor is comfortably in front avoids
    # repeated forward/reverse resets around a lateral route direction.
    forward_reacquired_bearing_rad: float = math.radians(75.0)
    behind_confirmation_cycles: int = 3
    minimum_reverse_escape_m: float = 0.30
    maximum_reverse_escape_m: float = 3.00


@dataclass(frozen=True)
class ForwardPreferenceDecision:
    state: ForwardPreferenceState
    requested_gear: Gear
    start_turnaround: bool
    corridor_bearing_rad: float
    reverse_escape_m: float
    reason: str


def corridor_direction_body(reference_path, lookahead_m=0.75):
    """Return a stable body-frame corridor direction at metric lookahead.

    A lateral 90-degree bend remains a normal forward corner.  Only a suffix
    whose bearing is materially beyond 90 degrees is considered behind.
    """

    points = np.asarray(reference_path, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
        return 0.0, 0.0
    travelled = 0.0
    previous = points[0]
    selected = points[-1]
    for point in points[1:]:
        travelled += float(np.linalg.norm(point - previous))
        selected = point
        previous = point
        if travelled >= float(lookahead_m):
            break
    return float(math.atan2(selected[1], selected[0])), travelled


class ForwardPreferenceSupervisor:
    """Bound reverse travel and restore forward motion at the first safe site."""

    def __init__(self, config=ForwardPreferenceConfig()):
        if config.direction_lookahead_m <= 0.0:
            raise ValueError("direction_lookahead_m must be positive")
        if not math.pi / 2.0 < config.behind_bearing_rad < math.pi:
            raise ValueError("behind_bearing_rad must be between 90 and 180 degrees")
        if not 0.0 < config.forward_reacquired_bearing_rad < math.pi / 2.0:
            raise ValueError(
                "forward_reacquired_bearing_rad must be between 0 and 90 degrees"
            )
        if config.behind_confirmation_cycles < 1:
            raise ValueError("behind_confirmation_cycles must be positive")
        if not 0.0 <= config.minimum_reverse_escape_m < config.maximum_reverse_escape_m:
            raise ValueError("reverse escape distance limits are invalid")
        self.config = config
        self.reset()

    def reset(self):
        self.state = ForwardPreferenceState.FORWARD_CRUISE
        self.behind_cycles = 0
        self.reverse_escape_m = 0.0

    def forward_corridor_reacquired(
        self, reference_path, *, route_requested_gear=None
    ):
        """Return true only when both near and look-ahead guidance face forward.

        The look-ahead bearing can enter the front quadrant while the first
        reachable corridor segment is still behind the rear axle.  Declaring
        a turnaround complete from the bearing alone makes the car shift to
        forward into the separating wall.  The global gear is used only as a
        near-corridor direction hint; it does not prescribe a trajectory.
        """

        bearing, route_length = corridor_direction_body(
            reference_path, self.config.direction_lookahead_m
        )
        near_corridor_forward = (
            route_requested_gear is None
            or Gear.require_drive(route_requested_gear) == Gear.FORWARD
        )
        return bool(
            route_length > 0.15
            and abs(bearing) < self.config.forward_reacquired_bearing_rad
            and near_corridor_forward
        )

    def update(
        self,
        reference_path,
        *,
        turnaround_feasible,
        progress_m=0.0,
        route_requested_gear=None,
    ):
        bearing, route_length = corridor_direction_body(
            reference_path, self.config.direction_lookahead_m
        )
        behind = (
            route_length > 0.15
            and abs(bearing) >= self.config.behind_bearing_rad
        )
        recovery_active = self.state != ForwardPreferenceState.FORWARD_CRUISE
        forward_reacquired = self.forward_corridor_reacquired(
            reference_path,
            route_requested_gear=route_requested_gear,
        )
        if not behind and (not recovery_active or forward_reacquired):
            self.reset()
            return ForwardPreferenceDecision(
                self.state,
                Gear.FORWARD,
                False,
                bearing,
                self.reverse_escape_m,
                (
                    "forward_corridor_reacquired"
                    if recovery_active
                    else "corridor_forward_or_lateral"
                ),
            )

        # Once a turnaround/reverse escape has begun, keep it latched through
        # the 75--110 degree hysteresis band.  Requiring three fresh samples
        # here would briefly re-authorize forward motion at exactly the point
        # where a replan can move the bearing across the 110-degree boundary.
        if recovery_active:
            self.behind_cycles = max(
                self.behind_cycles, self.config.behind_confirmation_cycles
            )
        else:
            self.behind_cycles += 1
        if self.behind_cycles < self.config.behind_confirmation_cycles:
            return ForwardPreferenceDecision(
                self.state,
                Gear.FORWARD,
                False,
                bearing,
                self.reverse_escape_m,
                "confirming_corridor_behind",
            )

        if self.state == ForwardPreferenceState.REVERSE_ESCAPE:
            self.reverse_escape_m += max(0.0, float(progress_m))
        if turnaround_feasible and (
            self.state != ForwardPreferenceState.REVERSE_ESCAPE
            or self.reverse_escape_m >= self.config.minimum_reverse_escape_m
        ):
            self.state = ForwardPreferenceState.TURNAROUND_PENDING
            return ForwardPreferenceDecision(
                self.state,
                Gear.FORWARD,
                True,
                bearing,
                self.reverse_escape_m,
                "safe_local_turnaround_available",
            )

        if self.reverse_escape_m >= self.config.maximum_reverse_escape_m:
            self.state = ForwardPreferenceState.REVERSE_ESCAPE_EXHAUSTED
            return ForwardPreferenceDecision(
                self.state,
                Gear.NEUTRAL,
                False,
                bearing,
                self.reverse_escape_m,
                "bounded_reverse_escape_exhausted",
            )

        self.state = ForwardPreferenceState.REVERSE_ESCAPE
        return ForwardPreferenceDecision(
            self.state,
            Gear.REVERSE,
            False,
            bearing,
            self.reverse_escape_m,
            "reverse_only_until_turnaround_space",
        )
