"""Stop-before-shift authority for signed Ackermann motion."""

from dataclasses import dataclass
from enum import Enum

from .types import Gear


class GearShiftState(str, Enum):
    DRIVE = "DRIVE"
    STOPPING = "STOPPING"
    SHIFT_HOLD = "SHIFT_HOLD"


class GearModeV1(str, Enum):
    """Discrete mode owned by the deterministic supervisor, never the network."""

    NEUTRAL = "NEUTRAL"
    FORWARD = "FORWARD"
    BRAKE_TO_REVERSE = "BRAKE_TO_REVERSE"
    REVERSE = "REVERSE"
    BRAKE_TO_FORWARD = "BRAKE_TO_FORWARD"


@dataclass(frozen=True)
class GearShiftConfig:
    stop_speed_tolerance: float = 0.03
    shift_hold_s: float = 0.20


@dataclass(frozen=True)
class GearDecision:
    requested: Gear
    engaged: Gear
    state: GearShiftState
    brake: bool

    @property
    def drive_enabled(self):
        return self.state == GearShiftState.DRIVE and self.engaged == self.requested

    @property
    def mode(self):
        if self.requested == Gear.NEUTRAL:
            return GearModeV1.NEUTRAL
        if self.state != GearShiftState.DRIVE:
            return GearModeV1.BRAKE_TO_REVERSE if self.requested == Gear.REVERSE else GearModeV1.BRAKE_TO_FORWARD
        return GearModeV1.REVERSE if self.engaged == Gear.REVERSE else GearModeV1.FORWARD


class GearSupervisor:
    """Never permits a forward/reverse transition while the car is moving."""

    def __init__(self, config: GearShiftConfig = GearShiftConfig()):
        self.config = config
        self.engaged = Gear.NEUTRAL
        self.pending = Gear.NEUTRAL
        self.state = GearShiftState.DRIVE
        self._stopped_since = None

    def update(self, requested, signed_speed: float, now: float) -> GearDecision:
        requested = Gear(int(requested))
        if requested == Gear.NEUTRAL:
            self.pending = Gear.NEUTRAL
            self.engaged = Gear.NEUTRAL
            self.state = GearShiftState.DRIVE
            self._stopped_since = None
            return GearDecision(requested, self.engaged, self.state, True)

        if requested != self.pending:
            self.pending = requested
            self._stopped_since = None
            self.state = GearShiftState.STOPPING if requested != self.engaged else GearShiftState.DRIVE

        if requested == self.engaged:
            self.state = GearShiftState.DRIVE
            self._stopped_since = None
            return GearDecision(requested, self.engaged, self.state, False)

        if abs(float(signed_speed)) > self.config.stop_speed_tolerance:
            self.state = GearShiftState.STOPPING
            self._stopped_since = None
            return GearDecision(requested, self.engaged, self.state, True)

        if self._stopped_since is None:
            self._stopped_since = float(now)
            self.state = GearShiftState.SHIFT_HOLD
            return GearDecision(requested, self.engaged, self.state, True)

        if float(now) - self._stopped_since < self.config.shift_hold_s:
            self.state = GearShiftState.SHIFT_HOLD
            return GearDecision(requested, self.engaged, self.state, True)

        self.engaged = requested
        self.state = GearShiftState.DRIVE
        self._stopped_since = None
        return GearDecision(requested, self.engaged, self.state, False)
