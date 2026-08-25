"""Small execution latch for a learned mixed-gear receding-horizon plan.

The V4 family predicts the complete manoeuvre sequence.  Runtime still asks
the network for a fresh, sensor-closed-loop candidate bank every cycle.  This
latch commits the selected model gear prefix while steering and speed continue
to come from the newest hard-safe candidate in the committed gear.  It does
not invent a turnaround or replay an open-loop trajectory.
"""

from dataclasses import dataclass

import numpy as np

from dep_car.core.planner import PlanningResult
from dep_car.core.types import Gear


def align_trajectory_between_chassis_frames(trajectory, anchor_pose, current_pose):
    """Rigidly express an anchored body-frame trajectory in today's frame.

    ``anchor_pose`` is the map-frame vehicle pose when the asynchronous policy
    query was published; ``current_pose`` is the map-frame pose immediately
    before the bank crosses the hard-safety boundary.  No learned geometry is
    warped and time, speed and steering are intentionally unchanged.
    """

    value = np.asarray(trajectory, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 6:
        raise ValueError("hybrid trajectory must have shape [N, 6]")
    if not np.all(np.isfinite(value)):
        raise ValueError("hybrid trajectory contains non-finite values")
    anchor = np.asarray(anchor_pose, dtype=np.float64)
    current = np.asarray(current_pose, dtype=np.float64)
    if anchor.shape != (3,) or current.shape != (3,):
        raise ValueError("trajectory poses must be (x, y, yaw)")
    if not np.all(np.isfinite(anchor)) or not np.all(np.isfinite(current)):
        raise ValueError("trajectory pose contains non-finite values")

    result = value.copy()
    ax, ay, ayaw = anchor
    cx, cy, cyaw = current
    ca, sa = np.cos(ayaw), np.sin(ayaw)
    cc, sc = np.cos(cyaw), np.sin(cyaw)
    x = value[:, 1]
    y = value[:, 2]
    world_x = ax + ca * x - sa * y
    world_y = ay + sa * x + ca * y
    dx = world_x - cx
    dy = world_y - cy
    result[:, 1] = cc * dx + sc * dy
    result[:, 2] = -sc * dx + cc * dy
    result[:, 3] = np.arctan2(
        np.sin(value[:, 3] + ayaw - cyaw),
        np.cos(value[:, 3] + ayaw - cyaw),
    )
    return result


@dataclass(frozen=True)
class HybridActionLatchConfig:
    minimum_action_s: float = 0.35
    maximum_action_s: float = 1.25
    maximum_unavailable_s: float = 0.80

    def __post_init__(self):
        if not 0.0 < self.minimum_action_s <= self.maximum_action_s:
            raise ValueError("invalid hybrid first-action latch limits")
        if self.maximum_unavailable_s <= 0.0:
            raise ValueError("invalid hybrid action availability timeout")


class HybridSequenceExecutionLatch:
    """Execute the model-proposed gear prefix without horizon procrastination.

    Receding-horizon inference can otherwise predict ``R,F,R`` every cycle but
    repeatedly execute only the first ``R``.  The latch stores the complete
    active prefix selected by the network.  At each action boundary it asks
    the newest bank for a hard-safe trajectory whose *first* action has the
    model-committed gear.  Steering and speed therefore stay sensor-closed-
    loop; the runtime does not invent a forward/reverse manoeuvre.

    If the committed next gear has no currently safe candidate, motion stops.
    A bounded wait then releases the obsolete sequence so a new complete
    model plan can take authority.  The timer for each action starts only after
    the zero-speed gear supervisor authorizes that gear.
    """

    def __init__(self, config=HybridActionLatchConfig()):
        self.config = config
        self.reset()

    def reset(self):
        self.action_gears = ()
        self.action_durations = ()
        self.action_index = 0
        self.drive_started_at = None
        self.unavailable_since = None
        self.candidate_id = -1

    @property
    def gear(self):
        if not self.armed:
            return Gear.NEUTRAL
        return Gear.require_drive(self.action_gears[self.action_index])

    @property
    def duration_s(self):
        if not self.armed:
            return 0.0
        return float(self.action_durations[self.action_index])

    @property
    def locked_sequence(self):
        return tuple(int(value) for value in self.action_gears)

    @property
    def armed(self):
        return self.action_index < len(self.action_gears)

    def expired(self, now):
        return bool(
            self.armed
            and self.drive_started_at is not None
            and float(now) >= self.drive_started_at + self.duration_s
        )

    def _arm(self, candidate):
        raw_gears = tuple(int(value) for value in getattr(candidate, "action_gears", ()))
        raw_mask = tuple(bool(value) for value in getattr(candidate, "action_mask", ()))
        raw_durations = tuple(
            float(value) for value in getattr(candidate, "action_durations", ())
        )
        if raw_gears and len(raw_gears) == len(raw_mask) == len(raw_durations):
            active = [
                (Gear.require_drive(gear), duration)
                for gear, mask, duration in zip(
                    raw_gears, raw_mask, raw_durations
                )
                if mask
            ]
        else:
            active = [
                (
                    Gear.require_drive(candidate.gear),
                    raw_durations[0]
                    if raw_durations
                    else float(candidate.duration),
                )
            ]
        if not active:
            raise ValueError("selected hybrid candidate has no active action")
        self.action_gears = tuple(item[0] for item in active)
        self.action_durations = tuple(
            min(
                self.config.maximum_action_s,
                max(self.config.minimum_action_s, item[1]),
            )
            for item in active
        )
        self.action_index = 0
        self.drive_started_at = None
        self.unavailable_since = None
        self.candidate_id = int(candidate.candidate_id)

    def _advance(self):
        self.action_index += 1
        self.drive_started_at = None
        self.unavailable_since = None
        if not self.armed:
            self.reset()

    def select(self, result, now):
        """Return a hard-safe selection respecting the learned gear prefix.

        ``result`` has already crossed the mandatory full-sequence hard veto.
        The returned PlanningResult shares its candidate diagnostics and only
        changes which feasible candidate is selected.
        """

        if result is None or not result.executable:
            return result, "no_executable_sequence"
        if self.expired(now):
            self._advance()
        feasible = [candidate for candidate in result.candidates if candidate.feasible]
        if self.armed:
            same_gear = [
                candidate
                for candidate in feasible
                if Gear.require_drive(candidate.gear) == self.gear
            ]
            if not same_gear:
                if self.unavailable_since is None:
                    self.unavailable_since = float(now)
                if (
                    float(now) - self.unavailable_since
                    < self.config.maximum_unavailable_s
                ):
                    return PlanningResult(
                        selected=None,
                        candidates=result.candidates,
                        retime_factor=None,
                        blocked_by_static=result.blocked_by_static,
                        blocked_by_dynamic=result.blocked_by_dynamic,
                        generation=result.generation,
                    ), "committed_sequence_action_unavailable"
                # The old model plan is no longer executable.  Release it and
                # let the newest complete model sequence acquire authority.
                self.reset()
            else:
                self.unavailable_since = None
                selected = min(
                    same_gear,
                    key=lambda item: (item.learned_score, item.candidate_id),
                )
                return PlanningResult(
                    selected=selected,
                    candidates=result.candidates,
                    retime_factor=result.retime_factor,
                    blocked_by_static=result.blocked_by_static,
                    blocked_by_dynamic=result.blocked_by_dynamic,
                    generation=result.generation,
                ), "committed_sequence_action_%d" % self.action_index
        self._arm(result.selected)
        return result, "new_model_sequence"

    def observe_drive_authorized(self, now, gear):
        gear = Gear.require_drive(gear)
        if not self.armed or gear != self.gear:
            return False
        if self.drive_started_at is None:
            self.drive_started_at = float(now)
        return True


# Compatibility alias for callers and older tests.  Its behavior now commits
# the complete learned prefix, not merely its first receding-horizon action.
HybridFirstActionLatch = HybridSequenceExecutionLatch


__all__ = [
    "HybridActionLatchConfig",
    "HybridFirstActionLatch",
    "HybridSequenceExecutionLatch",
    "align_trajectory_between_chassis_frames",
]
