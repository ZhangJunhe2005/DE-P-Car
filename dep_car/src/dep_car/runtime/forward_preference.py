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
    ROUTE_REVALIDATION = "ROUTE_REVALIDATION"
    TURNAROUND_CONFIRM = "TURNAROUND_CONFIRM"
    TURNAROUND_PENDING = "TURNAROUND_PENDING"
    TURNAROUND_VERIFY = "TURNAROUND_VERIFY"
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
    # A continuously safe forward arc is useful for a broad bend, but it is
    # not a licence to wander forward when the route is almost antiparallel
    # to the vehicle.  Beyond this angle the motion must be owned by one
    # committed multi-leg turnaround transaction.
    forward_capture_maximum_bearing_rad: float = math.radians(145.0)
    # If an ordinary forward course-capture starts to make the route bearing
    # materially worse, re-anchor the route at the measured pose instead of
    # interpreting that transient disagreement as reverse authority.
    forward_capture_divergence_rad: float = math.radians(18.0)
    behind_confirmation_cycles: int = 3
    forward_confirmation_cycles: int = 3
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


def navigation_authority_reference(
    reference_path,
    mission_goal_body,
    route_source,
    *,
    horizon_m,
):
    """Keep a local-exploration turnaround aimed at the mission, not a carrot.

    FAR and explored-topology routes carry durable direction information and
    are returned unchanged.  LOCAL_SAFE_EXPLORATION is only a short safe tube;
    its nearby endpoint may be passed during a multi-leg manoeuvre.  In that
    one mode the distant position goal supplies direction while the original
    route and hard veto continue to supply local collision safety.
    """

    reference = np.asarray(reference_path, dtype=float)
    if str(route_source) != "LOCAL_SAFE_EXPLORATION":
        return reference
    mission = np.asarray(mission_goal_body, dtype=float)
    if (
        mission.shape != (2,)
        or not np.all(np.isfinite(mission))
        or float(horizon_m) <= 0.0
    ):
        return reference
    distance = float(np.linalg.norm(mission))
    if distance <= 1.0e-6:
        return reference
    length = min(distance, float(horizon_m))
    return np.vstack((
        np.zeros((1, 2), dtype=float),
        mission.reshape(1, 2) * (length / distance),
    ))


def terminal_capture_route_authorized(route_source) -> bool:
    """Allow local final capture only under explicit global route authority."""

    source = str(route_source or "NONE")
    return bool(
        source.startswith("FAR_") or source == "KNOWN_TERMINAL_DIRECT"
    )


def route_requires_far_revalidation(route_source) -> bool:
    """Limit the measured-pose FAR handshake to actual FAR authority.

    Explored topology and local exploration are explicit fallback authorities,
    not rolling revisions of the previously active FAR route.  If either one
    points behind the car it must use its own bounded Ackermann turnaround,
    rather than stop and wait for an unrelated speculative FAR candidate.
    """

    return str(route_source or "NONE").startswith("FAR_")


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
        if not (
            config.behind_bearing_rad
            < config.forward_capture_maximum_bearing_rad
            < math.pi
        ):
            raise ValueError(
                "forward capture maximum bearing must be between the behind "
                "threshold and 180 degrees"
            )
        if not 0.0 < config.forward_capture_divergence_rad < math.pi / 2.0:
            raise ValueError(
                "forward capture divergence must be between 0 and 90 degrees"
            )
        if config.behind_confirmation_cycles < 1:
            raise ValueError("behind_confirmation_cycles must be positive")
        if config.forward_confirmation_cycles < 1:
            raise ValueError("forward_confirmation_cycles must be positive")
        if not 0.0 <= config.minimum_reverse_escape_m < config.maximum_reverse_escape_m:
            raise ValueError("reverse escape distance limits are invalid")
        self.config = config
        self.reset()

    def reset(self, *, preserve_forward_evidence=False):
        forward_seen = bool(
            preserve_forward_evidence
            and getattr(self, "forward_corridor_seen", False)
        )
        self.state = ForwardPreferenceState.FORWARD_CRUISE
        self.behind_cycles = 0
        self.forward_cycles = 0
        self.reverse_escape_m = 0.0
        self.course_capture_active = False
        self.course_capture_best_bearing_rad = math.inf
        self.forward_corridor_seen = forward_seen
        self.rear_route_revalidated = False

    def approve_revalidated_route(self):
        """Allow one measured-pose rear route to start a turnaround.

        A goal that starts behind the car needs no extra transaction.  This
        token is only for an abrupt front-to-rear change while already driving
        the same goal, so a single rolling FAR replacement cannot command a
        three-point turn before it has been replanned at the measured pose.
        """

        self.reset()
        self.rear_route_revalidated = True

    def approve_continuation_route(self):
        """Release an inter-leg route barrier without ending the manoeuvre.

        A measured-pose route refresh between two parking-style legs is not a
        new navigation transaction.  Resetting to ``FORWARD_CRUISE`` here
        makes a completed reverse leg look like a completed turnaround and can
        then deadlock the required forward/second reverse leg behind the
        new-turn rearm gate.  Keep recovery latched and let ``update`` either
        verify a genuinely completed forward exit or schedule the next safe
        opposite-gear leg.
        """

        self.state = ForwardPreferenceState.TURNAROUND_PENDING
        self.behind_cycles = max(
            self.behind_cycles, self.config.behind_confirmation_cycles
        )
        self.forward_cycles = 0
        self.course_capture_active = False
        self.course_capture_best_bearing_rad = math.inf
        self.rear_route_revalidated = True

    def request_route_revalidation(self):
        """Pause between finite legs until guidance is rebuilt at measured pose.

        A committed Ackermann leg can move the rear axle far enough that the
        route suffix which selected that leg is no longer a useful steering
        reference.  Keep the multi-leg manoeuvre transaction itself intact,
        but require the route owner to publish a fresh atomic route/command
        transaction before choosing the opposite-gear continuation.
        """

        self.state = ForwardPreferenceState.ROUTE_REVALIDATION
        self.behind_cycles = 0
        self.forward_cycles = 0

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
        forward_capture_feasible=False,
        forward_exit_verified=False,
        progress_m=0.0,
        route_requested_gear=None,
        turnaround_start_authorized=True,
    ):
        bearing, route_length = corridor_direction_body(
            reference_path, self.config.direction_lookahead_m
        )
        behind = (
            route_length > 0.15
            and abs(bearing) >= self.config.behind_bearing_rad
        )
        if self.state == ForwardPreferenceState.ROUTE_REVALIDATION:
            return ForwardPreferenceDecision(
                self.state,
                Gear.NEUTRAL,
                False,
                bearing,
                self.reverse_escape_m,
                "forward_course_capture_route_revalidation",
            )
        recovery_active = self.state not in (
            ForwardPreferenceState.FORWARD_CRUISE,
            ForwardPreferenceState.TURNAROUND_CONFIRM,
        )
        forward_reacquired = self.forward_corridor_reacquired(
            reference_path,
            route_requested_gear=route_requested_gear,
        )
        if not recovery_active and not behind:
            self.reset()
            self.forward_corridor_seen = True
            return ForwardPreferenceDecision(
                self.state,
                Gear.FORWARD,
                False,
                bearing,
                self.reverse_escape_m,
                (
                    "corridor_forward_or_lateral"
                ),
            )

        abrupt_rear_route = bool(
            not recovery_active
            and behind
            and abs(bearing) > self.config.forward_capture_maximum_bearing_rad
            and self.forward_corridor_seen
            and not self.rear_route_revalidated
            and not self.course_capture_active
        )
        if abrupt_rear_route:
            self.state = ForwardPreferenceState.ROUTE_REVALIDATION
            self.behind_cycles = 0
            self.forward_cycles = 0
            return ForwardPreferenceDecision(
                self.state,
                Gear.NEUTRAL,
                False,
                bearing,
                self.reverse_escape_m,
                "abrupt_rear_route_revalidation",
            )

        # A corridor becoming side/rearward does not by itself justify a gear
        # change.  If a continuously hard-safe forward Ackermann primitive
        # makes yaw progress toward it, use that primitive as ordinary course
        # capture.  Reverse becomes eligible only after forward capture is no
        # longer feasible, matching how a driver negotiates an open bend or
        # U-turn without parking-style oscillation.
        capture_diverged = bool(
            not recovery_active
            and behind
            and self.course_capture_active
            and (
                not bool(forward_capture_feasible)
                or abs(bearing) > self.config.forward_capture_maximum_bearing_rad
                or abs(bearing)
                > self.course_capture_best_bearing_rad
                + self.config.forward_capture_divergence_rad
            )
        )
        if capture_diverged:
            self.state = ForwardPreferenceState.ROUTE_REVALIDATION
            self.behind_cycles = 0
            self.forward_cycles = 0
            return ForwardPreferenceDecision(
                self.state,
                Gear.NEUTRAL,
                False,
                bearing,
                self.reverse_escape_m,
                "forward_course_capture_route_revalidation",
            )

        if (
            not recovery_active
            and behind
            and abs(bearing) <= self.config.forward_capture_maximum_bearing_rad
            and bool(forward_capture_feasible)
        ):
            self.state = ForwardPreferenceState.FORWARD_CRUISE
            self.behind_cycles = 0
            self.forward_cycles = 0
            self.reverse_escape_m = 0.0
            self.course_capture_active = True
            self.forward_corridor_seen = True
            self.course_capture_best_bearing_rad = min(
                self.course_capture_best_bearing_rad, abs(bearing)
            )
            return ForwardPreferenceDecision(
                self.state,
                Gear.FORWARD,
                False,
                bearing,
                self.reverse_escape_m,
                "safe_forward_course_capture",
            )

        # A single replanned corridor sample can briefly point forward while
        # the body is still midway through a multi-point Ackermann turn.  Hold
        # the vehicle and require a stable forward corridor before releasing
        # the turnaround transaction back to ordinary goal seeking.
        # Finishing a reverse leg can make a short look-ahead sample appear in
        # front even though the rear axle has not yet crossed the corner and a
        # forward shift would put the vehicle into the inside wall.  The caller
        # must therefore verify either a completed metric forward leg or a
        # sufficiently long, hard-safe forward primitive into a refreshed FAR
        # corridor.  A tiny emergency primitive is not exit evidence.
        verified_forward_exit = bool(
            forward_exit_verified and forward_capture_feasible
        )
        if recovery_active and forward_reacquired and verified_forward_exit:
            self.forward_cycles += 1
            if self.forward_cycles >= self.config.forward_confirmation_cycles:
                self.reset()
                self.forward_corridor_seen = True
                return ForwardPreferenceDecision(
                    self.state,
                    Gear.FORWARD,
                    False,
                    bearing,
                    self.reverse_escape_m,
                    "forward_corridor_reacquired",
                )
            self.state = ForwardPreferenceState.TURNAROUND_VERIFY
            return ForwardPreferenceDecision(
                self.state,
                Gear.NEUTRAL,
                False,
                bearing,
                self.reverse_escape_m,
                "confirming_forward_corridor",
            )
        self.forward_cycles = 0

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
            self.state = ForwardPreferenceState.TURNAROUND_CONFIRM
            return ForwardPreferenceDecision(
                self.state,
                Gear.NEUTRAL,
                False,
                bearing,
                self.reverse_escape_m,
                "confirming_corridor_behind",
            )

        if self.state == ForwardPreferenceState.REVERSE_ESCAPE:
            self.reverse_escape_m += max(0.0, float(progress_m))
        continuation_already_authorized = bool(
            self.state == ForwardPreferenceState.TURNAROUND_PENDING
        )
        if (
            not bool(turnaround_start_authorized)
            and not continuation_already_authorized
        ):
            # One position-goal transaction must not reinterpret every brief
            # FAR -> NO_ROUTE -> local-exploration handoff as a brand-new
            # request to turn around.  The caller rearms ordinary turnaround
            # authority only after meaningful forward travel (or a bounded
            # timeout); explicit memory/dead-end modes bypass this supervisor.
            # Keep the gearbox neutral while a fresh FAR route settles rather
            # than manufacture another reverse leg from a disposable local
            # carrot which happens to lie behind the rear axle.
            self.state = ForwardPreferenceState.TURNAROUND_CONFIRM
            self.behind_cycles = 0
            return ForwardPreferenceDecision(
                self.state,
                Gear.NEUTRAL,
                False,
                bearing,
                self.reverse_escape_m,
                "turnaround_rearm_forward_progress_pending",
            )
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

    def observe_committed_motion(self, progress_m: float, gear: Gear) -> None:
        """Credit motion executed while the local committed-leg path owns control.

        The ordinary supervisor update is intentionally bypassed during an
        active manoeuvre.  Without this hook a full safe reverse leg counted
        as only the single control-tick displacement observed after it ended,
        so the car repeatedly believed that it had reversed only 2--4 cm.
        """

        progress = max(0.0, float(progress_m))
        if (
            self.state == ForwardPreferenceState.REVERSE_ESCAPE
            and Gear.require_drive(gear) == Gear.REVERSE
        ):
            self.reverse_escape_m = min(
                self.config.maximum_reverse_escape_m,
                self.reverse_escape_m + progress,
            )
