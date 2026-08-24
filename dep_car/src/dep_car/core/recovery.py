"""Bounded Ackermann recovery with explicit mission/subgoal ownership."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

from .vehicle import URBAN_CAR_LINEAR_SCALE


class RecoveryState(str, Enum):
    MISSION_TRACKING = "MISSION_TRACKING"
    DYNAMIC_YIELD = "DYNAMIC_YIELD"
    STATIC_DEADLOCK = "STATIC_DEADLOCK"
    RECOVERY_SEARCH = "RECOVERY_SEARCH"
    REVERSE_RECOVERY = "REVERSE_RECOVERY"
    FORWARD_ESCAPE = "FORWARD_ESCAPE"
    MISSION_REACQUIRE = "MISSION_REACQUIRE"


@dataclass(frozen=True)
class RecoveryConfig:
    # At 10 Hz, two seconds supplies 20 consecutive hard-static observations
    # before recovery.  The former 2.5 s consumed over one quarter of a short
    # pilot episode with known-infeasible forward banks.
    stagnation_time_s: float = 2.0
    minimum_progress_m: float = 0.15
    reverse_distance_m: float = 1.8 * URBAN_CAR_LINEAR_SCALE
    reverse_speed_mps: float = 0.35
    recovery_timeout_s: float = 7.0


class RecoveryManager:
    """State machine preserving the V4.9.1 mission/recovery lifecycle.

    Dynamic-only blocking pauses deadlock evidence.  A completed recovery must
    return to fresh mission-conditioned planning before recovery is re-armed.
    """

    def __init__(self, config: RecoveryConfig = RecoveryConfig()):
        self.config = config
        self.state = RecoveryState.MISSION_TRACKING
        self.mission_goal: Optional[Tuple[float, float]] = None
        self.recovery_subgoal: Optional[Tuple[float, float]] = None
        self._stagnant_since: Optional[float] = None
        self._recovery_started: Optional[float] = None
        self._armed = True

    @property
    def active_goal(self) -> Optional[Tuple[float, float]]:
        return self.recovery_subgoal if self.recovery_subgoal is not None else self.mission_goal

    def set_mission_goal(self, goal: Tuple[float, float]) -> None:
        self.mission_goal = tuple(goal)
        self.recovery_subgoal = None
        self.state = RecoveryState.MISSION_TRACKING
        self._stagnant_since = None
        self._recovery_started = None
        self._armed = True

    def start_authority_transaction(self) -> None:
        """Clear local deadlock evidence when mission authority changes.

        A FAR goal route, breadcrumb reverse, and recovery resume are mutually
        exclusive transactions.  Static evidence accumulated under the old
        route must not immediately launch an opposite-gear micro manoeuvre in
        the newly selected transaction.
        """

        self.recovery_subgoal = None
        self.state = RecoveryState.MISSION_TRACKING
        self._stagnant_since = None
        self._recovery_started = None
        self._armed = True

    def update_blockage(self, now: float, progress_m: float, static_blocked: bool, dynamic_only: bool) -> RecoveryState:
        if dynamic_only:
            self.state = RecoveryState.DYNAMIC_YIELD
            self._stagnant_since = None
            return self.state
        if not static_blocked or progress_m >= self.config.minimum_progress_m:
            if self.state in (
                RecoveryState.DYNAMIC_YIELD,
                RecoveryState.MISSION_TRACKING,
                RecoveryState.STATIC_DEADLOCK,
            ):
                self.state = RecoveryState.MISSION_TRACKING
            self._stagnant_since = None
            return self.state
        if not self._armed:
            return self.state
        if self._stagnant_since is None:
            self._stagnant_since = now
        elif now - self._stagnant_since >= self.config.stagnation_time_s:
            self.state = RecoveryState.STATIC_DEADLOCK
        return self.state

    def begin_recovery(self, now: float, certified_subgoal: Tuple[float, float]) -> None:
        if self.state != RecoveryState.STATIC_DEADLOCK or not self._armed:
            raise RuntimeError("recovery may start only from an armed static deadlock")
        self.state = RecoveryState.REVERSE_RECOVERY
        self.recovery_subgoal = tuple(certified_subgoal)
        self._recovery_started = now
        self._armed = False

    def finish_reverse(self) -> None:
        if self.state != RecoveryState.REVERSE_RECOVERY:
            raise RuntimeError("reverse completion is invalid in the current state")
        self.state = RecoveryState.FORWARD_ESCAPE

    def finish_escape(self) -> None:
        if self.state != RecoveryState.FORWARD_ESCAPE:
            raise RuntimeError("escape completion is invalid in the current state")
        self.state = RecoveryState.MISSION_REACQUIRE
        self.recovery_subgoal = None

    def mission_plan_reacquired(self) -> None:
        if self.state != RecoveryState.MISSION_REACQUIRE:
            raise RuntimeError("fresh mission planning is required to re-arm recovery")
        self.state = RecoveryState.MISSION_TRACKING
        self._stagnant_since = None
        self._recovery_started = None
        self._armed = True
