"""Breadcrumb/topology memory and dead-end recovery primitives.

This module contains no ROS and no dense map-wide search.  The runtime records
the path actually driven in odometry coordinates, remembers failed branches,
and can later issue the reverse of certified history.  The ROS adapter may use
known walls from online SLAM as a local clearance authority, but never converts
the occupancy grid into an A* graph.
"""

from dataclasses import dataclass
from enum import Enum
import heapq
import math
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np


def wrap_angle(value: float) -> float:
    return math.atan2(math.sin(float(value)), math.cos(float(value)))


@dataclass(frozen=True)
class Breadcrumb:
    x: float
    y: float
    yaw: float
    stamp: float
    junction: bool = False
    turnaround: bool = False
    # Direction used to arrive at this sample: +1 forward, -1 reverse and
    # zero only for an initial/unknown anchor.  Time-reversing a trajectory
    # must invert this value; pose-only breadcrumbs cannot faithfully replay
    # an Ackermann multi-point turn containing both gears.
    motion_direction: int = 1
    # SLAM-corrected pose at observation time.  Odom remains the continuous
    # short-horizon motion frame; these optional map coordinates keep a long
    # dead-end connector from becoming an open-loop encoder replay.
    map_x: Optional[float] = None
    map_y: Optional[float] = None
    map_yaw: Optional[float] = None
    map_revision: int = -1


@dataclass(frozen=True)
class ReactiveHeadingDecision:
    """A map-agnostic local free-sector decision.

    A rear corridor is a request to perform an Ackermann turnaround, not an
    obstacle-avoidance shortcut.  It is therefore eligible only when the
    mission goal itself is behind the vehicle.  If every forward/side sector
    is blocked, ``blocked`` is set and the temporal dead-end supervisor gets
    authority instead of oscillating between an open rear ray and the goal.
    """

    angle: float
    clearance: float
    goal_bearing: float
    sector: str
    blocked: bool


@dataclass(frozen=True)
class RouteHandoffDecision:
    """Geometry-only decision for replacing one rolling route with another.

    ``accepted`` never means that the candidate path was translated or
    rotated.  It only says that the measured vehicle pose can attach to the
    candidate and that its immediate tangent is continuous with the active
    route.  The caller remains responsible for map/occupancy validation.
    """

    accepted: bool
    reason: str
    entry_distance_m: float
    direction_change_rad: Optional[float]
    candidate_progress_m: float
    candidate_carrot_m: Optional[float] = None


@dataclass(frozen=True)
class RouteProgressObservation:
    """Auditable state of a monotonic rolling-carrot route transaction."""

    route_id: str
    revision: int
    source: str
    progress_m: float
    total_m: float
    carrot_m: float
    carrot_xy: Tuple[float, float]
    tangent_bearing_rad: float
    deviation_m: float
    skipped_vertices: int
    lookahead_m: float
    carrot_distance_m: float
    carrot_advanced: bool
    carrot_hold_reason: str


class MonotonicRouteProgress:
    """Track physical progress without ever chasing a passed route point.

    Projection is restricted to a short interval ahead of the previous
    cursor.  This has two important properties for online navigation:

    * a self-intersection cannot teleport the cursor to a later lap; and
    * a pose correction or temporary reverse leg cannot move the cursor back
      to a route vertex that the vehicle already passed.

    A route replacement is handled as a *handoff*: both paths stay in their
    original coordinate frame and only the candidate attachment arclength is
    selected.  No route geometry is deformed by this class.
    """

    def __init__(
        self,
        *,
        minimum_lookahead_m: float = 1.0,
        maximum_lookahead_m: float = 2.5,
        curvature_window_m: float = 1.5,
        maximum_projection_advance_m: float = 1.25,
        projection_rollback_tolerance_m: float = 0.08,
        carrot_capture_radius_m: float = 0.40,
        maximum_carrot_advance_m: float = 0.70,
        maximum_carrot_distance_m: float = 1.50,
    ):
        if not 0.0 < minimum_lookahead_m <= maximum_lookahead_m:
            raise ValueError("route lookahead limits are invalid")
        if min(
            curvature_window_m,
            maximum_projection_advance_m,
            projection_rollback_tolerance_m,
            carrot_capture_radius_m,
            maximum_carrot_advance_m,
            maximum_carrot_distance_m,
        ) <= 0.0:
            raise ValueError("route progress distances must be positive")
        if maximum_carrot_distance_m < minimum_lookahead_m:
            raise ValueError(
                "maximum carrot distance must cover the minimum lookahead"
            )
        self.minimum_lookahead_m = float(minimum_lookahead_m)
        self.maximum_lookahead_m = float(maximum_lookahead_m)
        self.curvature_window_m = float(curvature_window_m)
        self.maximum_projection_advance_m = float(
            maximum_projection_advance_m
        )
        self.projection_rollback_tolerance_m = float(
            projection_rollback_tolerance_m
        )
        self.carrot_capture_radius_m = float(carrot_capture_radius_m)
        self.maximum_carrot_advance_m = float(maximum_carrot_advance_m)
        self.maximum_carrot_distance_m = float(maximum_carrot_distance_m)
        self.reset()

    def reset(self) -> None:
        self.route_id = ""
        self.revision = 0
        self.source = "NONE"
        self.path = np.empty((0, 2), dtype=float)
        self.cumulative = np.empty((0,), dtype=float)
        self.progress_m = 0.0
        self.carrot_m = None
        self.last_observation = None

    @property
    def active(self) -> bool:
        return len(self.path) >= 2 and bool(self.route_id)

    @staticmethod
    def _normalise(path) -> np.ndarray:
        values = np.asarray(path, dtype=float)
        if values.ndim != 2 or values.shape[1] != 2 or len(values) < 2:
            return np.empty((0, 2), dtype=float)
        if not np.all(np.isfinite(values)):
            return np.empty((0, 2), dtype=float)
        keep = np.concatenate((
            [True], np.linalg.norm(np.diff(values, axis=0), axis=1) > 1.0e-6
        ))
        values = values[keep]
        return values if len(values) >= 2 else np.empty((0, 2), dtype=float)

    @staticmethod
    def _cumulative(path: np.ndarray) -> np.ndarray:
        return np.concatenate((
            [0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))
        ))

    @staticmethod
    def _point_at(path: np.ndarray, cumulative: np.ndarray, arclength: float):
        if len(path) < 2:
            return None, 0
        query = min(max(0.0, float(arclength)), float(cumulative[-1]))
        segment = min(
            len(path) - 2,
            max(0, int(np.searchsorted(cumulative, query, side="right") - 1)),
        )
        length = float(cumulative[segment + 1] - cumulative[segment])
        ratio = 0.0 if length <= 1.0e-9 else (
            query - float(cumulative[segment])
        ) / length
        return path[segment] + ratio * (path[segment + 1] - path[segment]), segment

    @classmethod
    def _project(
        cls,
        point,
        path: np.ndarray,
        cumulative: np.ndarray,
        *,
        lower_m: float = 0.0,
        upper_m: float = math.inf,
    ):
        query = np.asarray(point, dtype=float)
        if query.shape != (2,) or not np.all(np.isfinite(query)) or len(path) < 2:
            return math.inf, 0.0, query, 0
        lower = max(0.0, float(lower_m))
        upper = min(float(cumulative[-1]), float(upper_m))
        if upper < lower:
            upper = lower
        best = (math.inf, lower, path[0], 0)
        for index, (first, second) in enumerate(zip(path[:-1], path[1:])):
            segment_start = float(cumulative[index])
            segment_end = float(cumulative[index + 1])
            if segment_end < lower - 1.0e-9 or segment_start > upper + 1.0e-9:
                continue
            delta = second - first
            length_squared = float(np.dot(delta, delta))
            if length_squared <= 1.0e-12:
                continue
            ratio = float(np.dot(query - first, delta)) / length_squared
            segment_length = segment_end - segment_start
            ratio_lower = max(0.0, (lower - segment_start) / segment_length)
            ratio_upper = min(1.0, (upper - segment_start) / segment_length)
            ratio = min(ratio_upper, max(ratio_lower, ratio))
            projection = first + ratio * delta
            distance = float(np.linalg.norm(query - projection))
            arclength = segment_start + ratio * segment_length
            if distance < best[0] - 1.0e-9 or (
                abs(distance - best[0]) <= 1.0e-9 and arclength < best[1]
            ):
                best = (distance, arclength, projection, index)
        return best

    @classmethod
    def _tangent(cls, path, cumulative, arclength, window_m=0.35) -> float:
        total = float(cumulative[-1])
        start = max(0.0, float(arclength) - 0.5 * float(window_m))
        end = min(total, float(arclength) + 0.5 * float(window_m))
        if end - start <= 1.0e-6:
            start = max(0.0, end - float(window_m))
        first, _ = cls._point_at(path, cumulative, start)
        second, _ = cls._point_at(path, cumulative, end)
        delta = second - first
        return float(math.atan2(delta[1], delta[0]))

    def preview_handoff(
        self,
        path,
        position,
        *,
        maximum_entry_deviation_m: float,
        maximum_direction_change_rad: float,
    ) -> RouteHandoffDecision:
        candidate = self._normalise(path)
        if len(candidate) < 2:
            return RouteHandoffDecision(
                False, "candidate_path_invalid", math.inf, None, 0.0
            )
        cumulative = self._cumulative(candidate)
        distance, progress, _, _ = self._project(
            position, candidate, cumulative
        )
        direction_change = None
        if self.active:
            old_tangent = self._tangent(
                self.path, self.cumulative, self.progress_m
            )
            new_tangent = self._tangent(candidate, cumulative, progress)
            direction_change = abs(wrap_angle(new_tangent - old_tangent))
        if distance > float(maximum_entry_deviation_m):
            return RouteHandoffDecision(
                False,
                "candidate_has_no_local_attachment",
                distance,
                direction_change,
                progress,
            )
        if (
            direction_change is not None
            and direction_change > float(maximum_direction_change_rad)
        ):
            return RouteHandoffDecision(
                False,
                "candidate_tangent_discontinuous",
                distance,
                direction_change,
                progress,
            )
        candidate_carrot = None
        if self.active and self.last_observation is not None:
            old_carrot = np.asarray(
                self.last_observation.carrot_xy, dtype=float
            )
            carrot_distance, carrot_progress, _, _ = self._project(
                old_carrot, candidate, cumulative
            )
            if (
                carrot_distance <= float(maximum_entry_deviation_m)
                and carrot_progress
                >= progress - self.projection_rollback_tolerance_m
            ):
                candidate_carrot = max(progress, carrot_progress)
        return RouteHandoffDecision(
            True,
            "candidate_progress_handoff",
            distance,
            direction_change,
            progress,
            candidate_carrot,
        )

    def bind(
        self,
        path,
        position,
        *,
        route_id: str,
        revision: int,
        source: str,
        initial_progress_m: Optional[float] = None,
        initial_carrot_m: Optional[float] = None,
    ) -> None:
        values = self._normalise(path)
        if len(values) < 2:
            raise ValueError("cannot bind an invalid rolling route")
        cumulative = self._cumulative(values)
        if initial_progress_m is None:
            _, progress, _, _ = self._project(position, values, cumulative)
        else:
            progress = min(
                float(cumulative[-1]), max(0.0, float(initial_progress_m))
            )
        self.route_id = str(route_id)
        self.revision = int(revision)
        self.source = str(source)
        self.path = values
        self.cumulative = cumulative
        self.progress_m = progress
        self.carrot_m = (
            None
            if initial_carrot_m is None
            else min(
                float(cumulative[-1]),
                max(progress, float(initial_carrot_m)),
            )
        )
        self.last_observation = None

    def _distance_bounded_carrot(self, position, desired_m: float) -> float:
        """Clamp a route target to a hard Euclidean radius around the car."""

        desired = min(
            float(self.cumulative[-1]),
            max(self.progress_m, float(desired_m)),
        )
        point, _ = self._point_at(self.path, self.cumulative, desired)
        query = np.asarray(position, dtype=float)
        if float(np.linalg.norm(point - query)) <= self.maximum_carrot_distance_m:
            return desired
        lower = self.progress_m
        upper = desired
        # Within the short rolling window the first distance crossing is the
        # useful one.  Bisection keeps the target on the original route; it
        # never translates or rotates FAR geometry to manufacture proximity.
        for _ in range(24):
            middle = 0.5 * (lower + upper)
            point, _ = self._point_at(self.path, self.cumulative, middle)
            if float(np.linalg.norm(point - query)) <= self.maximum_carrot_distance_m:
                lower = middle
            else:
                upper = middle
        return lower

    def observe(self, position, *, freeze: bool = False) -> RouteProgressObservation:
        if not self.active:
            raise RuntimeError("route cursor has no active path")
        if freeze:
            projection, _ = self._point_at(
                self.path, self.cumulative, self.progress_m
            )
            distance = float(
                np.linalg.norm(np.asarray(position, dtype=float) - projection)
            )
        else:
            lower = max(
                0.0, self.progress_m - self.projection_rollback_tolerance_m
            )
            upper = min(
                float(self.cumulative[-1]),
                self.progress_m + self.maximum_projection_advance_m,
            )
            distance, projected, _, _ = self._project(
                position,
                self.path,
                self.cumulative,
                lower_m=lower,
                upper_m=upper,
            )
            # The projection may move slightly backwards because of
            # cross-track noise, but the authoritative cursor never does.
            self.progress_m = max(self.progress_m, projected)
        tangent = self._tangent(
            self.path, self.cumulative, self.progress_m
        )
        future_s = min(
            float(self.cumulative[-1]),
            self.progress_m + self.curvature_window_m,
        )
        future_tangent = self._tangent(
            self.path, self.cumulative, future_s
        )
        turn = abs(wrap_angle(future_tangent - tangent))
        severity = min(1.0, turn / (0.5 * math.pi))
        lookahead = (
            self.maximum_lookahead_m
            - severity
            * (self.maximum_lookahead_m - self.minimum_lookahead_m)
        )
        desired_carrot = self._distance_bounded_carrot(
            position,
            min(float(self.cumulative[-1]), self.progress_m + lookahead),
        )
        previous_carrot = self.carrot_m
        advanced = False
        if previous_carrot is None:
            carrot_s = desired_carrot
            hold_reason = "initial_target"
            advanced = True
        else:
            previous_carrot = max(self.progress_m, float(previous_carrot))
            previous_point, _ = self._point_at(
                self.path, self.cumulative, previous_carrot
            )
            previous_distance = float(
                np.linalg.norm(
                    np.asarray(position, dtype=float) - previous_point
                )
            )
            captured = bool(
                previous_distance <= self.carrot_capture_radius_m
                or previous_carrot - self.progress_m
                <= self.carrot_capture_radius_m
            )
            if freeze:
                carrot_s = previous_carrot
                hold_reason = "maneuver_transaction_frozen"
            elif captured:
                carrot_s = min(
                    desired_carrot,
                    previous_carrot + self.maximum_carrot_advance_m,
                )
                advanced = carrot_s > previous_carrot + 1.0e-6
                hold_reason = (
                    "captured_target_advanced"
                    if advanced else "route_end_held"
                )
            else:
                carrot_s = previous_carrot
                hold_reason = "waiting_for_vehicle_capture"
            bounded = self._distance_bounded_carrot(position, carrot_s)
            if bounded < carrot_s - 1.0e-6:
                carrot_s = bounded
                advanced = False
                hold_reason = "maximum_vehicle_distance_clamp"
        self.carrot_m = float(carrot_s)
        carrot, _ = self._point_at(self.path, self.cumulative, carrot_s)
        carrot_distance = float(
            np.linalg.norm(np.asarray(position, dtype=float) - carrot)
        )
        skipped = max(
            0,
            int(np.searchsorted(
                self.cumulative, self.progress_m, side="right"
            ) - 1),
        )
        observation = RouteProgressObservation(
            route_id=self.route_id,
            revision=self.revision,
            source=self.source,
            progress_m=float(self.progress_m),
            total_m=float(self.cumulative[-1]),
            carrot_m=float(carrot_s),
            carrot_xy=(float(carrot[0]), float(carrot[1])),
            tangent_bearing_rad=float(tangent),
            deviation_m=float(distance),
            skipped_vertices=skipped,
            lookahead_m=float(lookahead),
            carrot_distance_m=carrot_distance,
            carrot_advanced=bool(advanced),
            carrot_hold_reason=hold_reason,
        )
        self.last_observation = observation
        return observation

    def remaining_path(self, position, *, maximum_deviation_m: float):
        observation = self.observe(position)
        if observation.deviation_m > float(maximum_deviation_m):
            return []
        projection, segment = self._point_at(
            self.path, self.cumulative, self.progress_m
        )
        output = [tuple(np.asarray(position, dtype=float)), tuple(projection)]
        output.extend(tuple(point) for point in self.path[segment + 1 :])
        deduplicated = [output[0]]
        for point in output[1:]:
            if np.linalg.norm(
                np.asarray(point) - np.asarray(deduplicated[-1])
            ) > 1.0e-4:
                deduplicated.append(point)
        return deduplicated


@dataclass(frozen=True)
class ProgressiveMapAcquisitionDecision:
    """One bounded move/scan decision while the global map is incomplete."""

    motion_authorized: bool
    remaining_m: float
    cycle: int
    accumulated_m: float
    known_cell_gain: int
    reason: str


class ProgressiveMapAcquisitionGate:
    """Allow short forward mapping pulses only while they reveal new space.

    A single goal-scoped distance allowance eventually deadlocks on a remote
    goal: the vehicle spends the allowance, FAR still sees an incomplete map,
    and no motion can reveal the next scan.  This gate turns that allowance
    into auditable move/settle/reobserve cycles.  A new cycle requires both a
    completed bounded movement and measurable map growth, while a cumulative
    cap prevents an unverified global hypothesis from becoming unlimited
    blind exploration.
    """

    def __init__(
        self,
        *,
        chunk_distance_m: float = 0.80,
        settle_time_s: float = 0.60,
        minimum_known_cell_gain: int = 20,
        maximum_accumulated_m: float = 8.0,
    ):
        if min(
            float(chunk_distance_m),
            float(settle_time_s),
            float(maximum_accumulated_m),
        ) <= 0.0:
            raise ValueError("map-acquisition distances and time must be positive")
        if int(minimum_known_cell_gain) < 1:
            raise ValueError("map acquisition needs positive known-cell gain")
        if float(maximum_accumulated_m) < float(chunk_distance_m):
            raise ValueError("map-acquisition total must cover at least one chunk")
        self.chunk_distance_m = float(chunk_distance_m)
        self.settle_time_s = float(settle_time_s)
        self.minimum_known_cell_gain = int(minimum_known_cell_gain)
        self.maximum_accumulated_m = float(maximum_accumulated_m)
        self.reset()

    def reset(self) -> None:
        self.anchor_xy = None
        self.anchor_known_cells = 0
        self.settle_started_at = None
        self.accumulated_m = 0.0
        self.cycle = 0

    def observe(self, position_xy, known_cells: int, stamp: float):
        position = np.asarray(position_xy, dtype=float)
        if position.shape != (2,) or not np.all(np.isfinite(position)):
            raise ValueError("map-acquisition position must be a finite 2-vector")
        known_cells = int(known_cells)
        stamp = float(stamp)
        if known_cells < 0 or not math.isfinite(stamp):
            raise ValueError("map-acquisition evidence is invalid")
        if self.anchor_xy is None:
            self.anchor_xy = position.copy()
            self.anchor_known_cells = known_cells
            return ProgressiveMapAcquisitionDecision(
                True,
                self.chunk_distance_m,
                self.cycle,
                self.accumulated_m,
                0,
                "mapping_pulse_active",
            )

        travelled = float(np.linalg.norm(position - self.anchor_xy))
        remaining = max(0.0, self.chunk_distance_m - travelled)
        known_gain = max(0, known_cells - self.anchor_known_cells)
        if remaining > 1.0e-6 and self.settle_started_at is None:
            return ProgressiveMapAcquisitionDecision(
                True,
                remaining,
                self.cycle,
                self.accumulated_m,
                known_gain,
                "mapping_pulse_active",
            )

        if self.settle_started_at is None:
            self.settle_started_at = stamp
            return ProgressiveMapAcquisitionDecision(
                False,
                0.0,
                self.cycle,
                self.accumulated_m,
                known_gain,
                "mapping_pulse_scan_settle",
            )
        if stamp - self.settle_started_at < self.settle_time_s:
            return ProgressiveMapAcquisitionDecision(
                False,
                0.0,
                self.cycle,
                self.accumulated_m,
                known_gain,
                "mapping_pulse_scan_settle",
            )

        completed = min(self.chunk_distance_m, travelled)
        prospective_total = self.accumulated_m + completed
        if known_gain < self.minimum_known_cell_gain:
            return ProgressiveMapAcquisitionDecision(
                False,
                0.0,
                self.cycle,
                prospective_total,
                known_gain,
                "mapping_pulse_no_new_observed_space",
            )
        if prospective_total + self.chunk_distance_m > self.maximum_accumulated_m + 1.0e-9:
            return ProgressiveMapAcquisitionDecision(
                False,
                0.0,
                self.cycle,
                prospective_total,
                known_gain,
                "mapping_pulse_goal_budget_exhausted",
            )

        self.accumulated_m = prospective_total
        self.cycle += 1
        self.anchor_xy = position.copy()
        self.anchor_known_cells = known_cells
        self.settle_started_at = None
        return ProgressiveMapAcquisitionDecision(
            True,
            self.chunk_distance_m,
            self.cycle,
            self.accumulated_m,
            0,
            "mapping_pulse_rearmed_after_map_growth",
        )


class BoundaryFollowMode(str, Enum):
    DIRECT = "DIRECT"
    FOLLOW_LEFT = "FOLLOW_LEFT"
    FOLLOW_RIGHT = "FOLLOW_RIGHT"


@dataclass(frozen=True)
class BoundaryFollowDecision:
    mode: BoundaryFollowMode
    active: bool
    side: int
    entered: bool
    left_boundary: bool
    loop_detected: bool
    direct_clearance_m: float
    hit_goal_distance_m: Optional[float]
    best_goal_distance_m: Optional[float]
    travelled_m: float
    reason: str


@dataclass(frozen=True)
class BoundarySideFailure:
    """A failed wall side near one obstacle-hit region for the current goal."""

    x: float
    y: float
    side: int


class BoundaryFollowSupervisor:
    """Stateful local Bug2/TangentBug-style obstacle transaction.

    This supervisor never searches an occupancy grid.  It latches one side
    when the direct swept corridor is blocked, keeps that side while local
    free-sector steering follows the boundary, and releases only after a
    demonstrably safer and goal-progressing direct corridor persists.  A full
    return to the hit region, or a bounded excessive traversal, is reported as
    a loop so breadcrumb/topology recovery can remember the failed branch.
    """

    def __init__(
        self,
        *,
        enter_clearance_m: float = 0.90,
        release_clearance_m: float = 1.35,
        leave_progress_m: float = 0.45,
        leave_confirmation_s: float = 0.80,
        loop_radius_m: float = 0.70,
        minimum_loop_travel_m: float = 4.0,
        maximum_boundary_travel_m: float = 14.0,
        failure_radius_m: float = 1.00,
        maximum_side_failures: int = 32,
    ):
        values = (
            enter_clearance_m,
            release_clearance_m,
            leave_progress_m,
            leave_confirmation_s,
            loop_radius_m,
            minimum_loop_travel_m,
            maximum_boundary_travel_m,
            failure_radius_m,
        )
        if min(float(value) for value in values) <= 0.0:
            raise ValueError("boundary-follow parameters must be positive")
        if float(release_clearance_m) < float(enter_clearance_m):
            raise ValueError("boundary release clearance must exceed entry clearance")
        if float(maximum_boundary_travel_m) <= float(minimum_loop_travel_m):
            raise ValueError("maximum boundary travel must exceed loop travel")
        if int(maximum_side_failures) <= 0:
            raise ValueError("maximum boundary side failures must be positive")
        self.enter_clearance_m = float(enter_clearance_m)
        self.release_clearance_m = float(release_clearance_m)
        self.leave_progress_m = float(leave_progress_m)
        self.leave_confirmation_s = float(leave_confirmation_s)
        self.loop_radius_m = float(loop_radius_m)
        self.minimum_loop_travel_m = float(minimum_loop_travel_m)
        self.maximum_boundary_travel_m = float(maximum_boundary_travel_m)
        self.failure_radius_m = float(failure_radius_m)
        self.maximum_side_failures = int(maximum_side_failures)
        self.failed_sides: List[BoundarySideFailure] = []
        self.reset(clear_failures=False)

    def reset(self, *, clear_failures: bool = False):
        self.mode = BoundaryFollowMode.DIRECT
        self.side = 0
        self.hit_xy = None
        self.last_xy = None
        self.hit_goal_distance_m = None
        self.best_goal_distance_m = None
        self.travelled_m = 0.0
        self.leave_clear_since = None
        if clear_failures:
            self.failed_sides.clear()

    @property
    def active(self) -> bool:
        return self.mode != BoundaryFollowMode.DIRECT

    def side_failed_near(self, current_xy: Sequence[float], side: int) -> bool:
        current = np.asarray(current_xy, dtype=float)
        return any(
            failure.side == int(side)
            and math.hypot(failure.x - current[0], failure.y - current[1])
            <= self.failure_radius_m
            for failure in self.failed_sides
        )

    def remember_current_failure(self) -> Optional[BoundarySideFailure]:
        """Forbid repeating this side near the current hit for this goal.

        Dense-map search is intentionally absent.  This small transaction
        memory is the Bug-family equivalent of remembering that a particular
        wall-follow direction returned to, or stalled near, the same hit.
        """

        if not self.active or self.hit_xy is None or not self.side:
            return None
        failure = BoundarySideFailure(
            x=float(self.hit_xy[0]),
            y=float(self.hit_xy[1]),
            side=int(self.side),
        )
        if not self.side_failed_near((failure.x, failure.y), failure.side):
            self.failed_sides.append(failure)
            if len(self.failed_sides) > self.maximum_side_failures:
                self.failed_sides = self.failed_sides[-self.maximum_side_failures :]
        return failure

    def heading_utility(self, angle: float) -> float:
        """Soft side commitment; hard safety remains outside this class."""

        if not self.active:
            return 0.0
        signed = float(self.side) * wrap_angle(angle)
        if signed < -0.40:
            return -3.0
        if signed < -0.10:
            return -0.9
        if signed <= 0.35:
            # Near-straight motion follows a tangent once the initial turn is
            # established; continuously demanding maximum steer would circle.
            return 0.55
        return 0.55 + 0.20 * min(1.0, signed)

    def update(
        self,
        stamp: float,
        current_xy: Sequence[float],
        goal_distance_m: float,
        direct_clearance_m: float,
        *,
        left_score: Optional[float] = None,
        right_score: Optional[float] = None,
    ) -> BoundaryFollowDecision:
        stamp = float(stamp)
        current = np.asarray(current_xy, dtype=float)
        if current.shape != (2,) or not np.all(np.isfinite(current)):
            raise ValueError("boundary current position must be a finite 2-vector")
        goal_distance = float(goal_distance_m)
        direct_clearance = max(0.0, float(direct_clearance_m))
        entered = False
        left_boundary = False
        loop_detected = False
        reason = "direct_corridor"

        if not self.active:
            scores = {
                1: -math.inf if left_score is None else float(left_score),
                -1: -math.inf if right_score is None else float(right_score),
            }
            for candidate_side in tuple(scores):
                if self.side_failed_near(current, candidate_side):
                    scores[candidate_side] = -math.inf
            side = max(scores, key=scores.get)
            if (
                direct_clearance < self.enter_clearance_m
                and math.isfinite(scores[side])
            ):
                self.side = int(side)
                self.mode = (
                    BoundaryFollowMode.FOLLOW_LEFT
                    if side > 0
                    else BoundaryFollowMode.FOLLOW_RIGHT
                )
                self.hit_xy = current.copy()
                self.last_xy = current.copy()
                self.hit_goal_distance_m = goal_distance
                self.best_goal_distance_m = goal_distance
                self.travelled_m = 0.0
                self.leave_clear_since = None
                entered = True
                reason = "direct_swept_corridor_blocked"
        else:
            if self.last_xy is not None:
                self.travelled_m += float(np.linalg.norm(current - self.last_xy))
            self.last_xy = current.copy()
            self.best_goal_distance_m = min(
                float(self.best_goal_distance_m), goal_distance
            )
            progressed = (
                goal_distance
                <= float(self.hit_goal_distance_m) - self.leave_progress_m
            )
            direct_released = direct_clearance >= self.release_clearance_m
            if progressed and direct_released:
                if self.leave_clear_since is None:
                    self.leave_clear_since = stamp
                elif stamp - self.leave_clear_since >= self.leave_confirmation_s:
                    mode, side = self.mode, self.side
                    hit_distance = self.hit_goal_distance_m
                    best_distance = self.best_goal_distance_m
                    travelled = self.travelled_m
                    self.reset(clear_failures=False)
                    return BoundaryFollowDecision(
                        mode=mode,
                        active=False,
                        side=side,
                        entered=False,
                        left_boundary=True,
                        loop_detected=False,
                        direct_clearance_m=direct_clearance,
                        hit_goal_distance_m=hit_distance,
                        best_goal_distance_m=best_distance,
                        travelled_m=travelled,
                        reason="progressing_direct_corridor_confirmed",
                    )
            else:
                self.leave_clear_since = None
            returned_to_hit = bool(
                self.hit_xy is not None
                and self.travelled_m >= self.minimum_loop_travel_m
                and np.linalg.norm(current - self.hit_xy) <= self.loop_radius_m
            )
            exceeded_bound = self.travelled_m >= self.maximum_boundary_travel_m
            if returned_to_hit or exceeded_bound:
                mode, side = self.mode, self.side
                hit_distance = self.hit_goal_distance_m
                best_distance = self.best_goal_distance_m
                travelled = self.travelled_m
                reason = (
                    "returned_to_boundary_hit_region"
                    if returned_to_hit
                    else "maximum_boundary_travel_exceeded"
                )
                self.remember_current_failure()
                self.reset(clear_failures=False)
                return BoundaryFollowDecision(
                    mode=mode,
                    active=False,
                    side=side,
                    entered=False,
                    left_boundary=False,
                    loop_detected=True,
                    direct_clearance_m=direct_clearance,
                    hit_goal_distance_m=hit_distance,
                    best_goal_distance_m=best_distance,
                    travelled_m=travelled,
                    reason=reason,
                )
            reason = "latched_boundary_side"

        return BoundaryFollowDecision(
            mode=self.mode,
            active=self.active,
            side=int(self.side),
            entered=entered,
            left_boundary=left_boundary,
            loop_detected=loop_detected,
            direct_clearance_m=direct_clearance,
            hit_goal_distance_m=self.hit_goal_distance_m,
            best_goal_distance_m=self.best_goal_distance_m,
            travelled_m=float(self.travelled_m),
            reason=reason,
        )


def select_reactive_heading(
    goal_bearing: float,
    angle_clearances: Iterable[Tuple[float, float]],
    *,
    previous_heading: Optional[float] = None,
    safe_clearance_m: float = 0.80,
    forward_sector_limit_rad: float = 1.65,
    turnaround_threshold_rad: float = 2.10,
    angle_utilities: Optional[Iterable[Tuple[float, float]]] = None,
) -> ReactiveHeadingDecision:
    """Select a persistent local direction without doing map-wide search.

    ``angle_clearances`` are body-frame rays from the current rolling map.
    For goals in the forward/side field of view, rear rays are deliberately
    excluded.  A fully blocked forward field then remains blocked long enough
    for breadcrumb recovery to engage.  For a genuinely rearward goal, only
    rearward directions are considered so the local planner receives one
    stable turnaround transaction instead of alternating gear requests.
    """

    if safe_clearance_m <= 0.0:
        raise ValueError("safe clearance must be positive")
    if not 0.0 < forward_sector_limit_rad < turnaround_threshold_rad < math.pi:
        raise ValueError("reactive heading sector thresholds are invalid")
    desired = wrap_angle(goal_bearing)
    utilities = {
        round(wrap_angle(angle), 6): float(value)
        for angle, value in (angle_utilities or ())
    }
    goal_is_behind = abs(desired) > float(turnaround_threshold_rad)
    rows = []
    seen = set()
    for raw_angle, raw_clearance in angle_clearances:
        angle = wrap_angle(raw_angle)
        key = round(angle, 6)
        if key in seen:
            continue
        seen.add(key)
        clearance = max(0.0, float(raw_clearance))
        in_sector = (
            abs(angle) > float(turnaround_threshold_rad)
            if goal_is_behind
            else abs(angle) <= float(forward_sector_limit_rad)
        )
        if in_sector:
            rows.append((angle, clearance))
    if not rows:
        raise ValueError("no sampled ray belongs to the required local sector")

    viable = [row for row in rows if row[1] >= float(safe_clearance_m)]
    pool = viable if viable else rows

    def score(row):
        angle, clearance = row
        goal_error = abs(wrap_angle(angle - desired))
        continuity = (
            0.0
            if previous_heading is None
            else abs(wrap_angle(angle - previous_heading))
        )
        # Continuity is intentionally stronger than the old implementation:
        # changing free sectors should require a real clearance advantage.
        return (
            clearance
            - 0.60 * goal_error
            - 0.45 * continuity
            + utilities.get(round(angle, 6), 0.0)
        )

    selected, clearance = max(pool, key=score)
    return ReactiveHeadingDecision(
        angle=float(selected),
        clearance=float(clearance),
        goal_bearing=float(desired),
        sector="TURNAROUND" if goal_is_behind else "FORWARD_EXPLORATION",
        blocked=not bool(viable),
    )


def ackermann_arc_trajectory(
    heading_change_rad: float,
    length_m: float,
    count: int = 40,
) -> np.ndarray:
    """Return a unit-speed forward arc as ``[t,x,y,yaw]`` rows."""

    if length_m <= 0.0 or count < 2:
        raise ValueError("Ackermann arc length/count are invalid")
    angle = float(heading_change_rad)
    progress = np.linspace(0.0, 1.0, int(count))
    if abs(angle) < 1.0e-6:
        x = float(length_m) * progress
        y = np.zeros_like(progress)
        yaw = np.zeros_like(progress)
    else:
        radius = float(length_m) / abs(angle)
        theta = abs(angle) * progress
        direction = math.copysign(1.0, angle)
        x = radius * np.sin(theta)
        y = direction * radius * (1.0 - np.cos(theta))
        yaw = direction * theta
    return np.column_stack((float(length_m) * progress, x, y, yaw))


class BreadcrumbTrail:
    """Distance/yaw-sampled driven-path memory with a bounded metric history."""

    def __init__(
        self,
        spacing_m: float = 0.20,
        heading_spacing_rad: float = math.radians(10.0),
        maximum_length_m: float = 25.0,
    ):
        if spacing_m <= 0.0 or heading_spacing_rad <= 0.0 or maximum_length_m <= 0.0:
            raise ValueError("breadcrumb sampling parameters must be positive")
        self.spacing_m = float(spacing_m)
        self.heading_spacing_rad = float(heading_spacing_rad)
        self.maximum_length_m = float(maximum_length_m)
        self.points: List[Breadcrumb] = []

    def clear(self):
        self.points.clear()

    def truncate_after(self, index: int):
        """Forget the abandoned suffix after backing out of a dead end."""

        if not self.points:
            return
        keep = min(max(0, int(index)), len(self.points) - 1)
        del self.points[keep + 1 :]

    @staticmethod
    def distance(a: Breadcrumb, b: Breadcrumb) -> float:
        return math.hypot(a.x - b.x, a.y - b.y)

    def record(
        self,
        x: float,
        y: float,
        yaw: float,
        stamp: float,
        *,
        junction: bool = False,
        turnaround: bool = False,
        motion_direction: int = 1,
        map_pose: Optional[Sequence[float]] = None,
        map_revision: int = -1,
        force: bool = False,
    ) -> bool:
        direction = int(motion_direction)
        if direction not in (-1, 0, 1):
            raise ValueError("breadcrumb motion direction must be -1, 0 or 1")
        map_values = (
            None
            if map_pose is None
            else tuple(float(value) for value in map_pose)
        )
        if map_values is not None and len(map_values) != 3:
            raise ValueError("breadcrumb map pose must contain x, y and yaw")
        item = Breadcrumb(
            float(x), float(y), wrap_angle(yaw), float(stamp),
            bool(junction), bool(turnaround), direction,
            None if map_values is None else map_values[0],
            None if map_values is None else map_values[1],
            None if map_values is None else wrap_angle(map_values[2]),
            int(map_revision),
        )
        if self.points and not force:
            previous = self.points[-1]
            displaced = self.distance(previous, item) >= self.spacing_m
            rotated = abs(wrap_angle(item.yaw - previous.yaw)) >= self.heading_spacing_rad
            promoted = (item.junction and not previous.junction) or (
                item.turnaround and not previous.turnaround
            )
            if not (displaced or rotated or promoted):
                return False
        self.points.append(item)
        self._trim()
        return True

    def _trim(self):
        length = 0.0
        keep_from = len(self.points) - 1
        for index in range(len(self.points) - 2, -1, -1):
            length += self.distance(self.points[index], self.points[index + 1])
            if length > self.maximum_length_m:
                break
            keep_from = index
        if keep_from > 0:
            del self.points[:keep_from]

    def nearest_index(self, x: float, y: float) -> Optional[int]:
        if not self.points:
            return None
        distances = [math.hypot(point.x - x, point.y - y) for point in self.points]
        return int(np.argmin(distances))

    def nearest_index_window(
        self,
        x: float,
        y: float,
        *,
        lower_index: int,
        upper_index: int,
    ) -> Optional[int]:
        """Nearest history index inside one monotonic replay window.

        A driven trail may cross itself.  A global nearest-point query at the
        crossing can jump to an unrelated lap and make reverse replay demand
        an impossible discontinuity.  Recovery therefore advances through a
        small, older-only index window.
        """

        if not self.points:
            return None
        lower = min(max(0, int(lower_index)), len(self.points) - 1)
        upper = min(max(lower, int(upper_index)), len(self.points) - 1)
        return min(
            range(lower, upper + 1),
            key=lambda index: math.hypot(
                self.points[index].x - float(x),
                self.points[index].y - float(y),
            ),
        )

    def reverse_replay_corridor(
        self,
        x: float,
        y: float,
        *,
        start_index: int,
        target_distance_m: float,
        minimum_points: int = 2,
    ) -> Tuple[List[Breadcrumb], Optional[int], int]:
        """Return one constant-gear time-reversed history transaction.

        The returned direction is the gear needed during replay.  A segment
        originally driven forward is replayed in reverse and vice versa.  The
        corridor stops at a historical gear boundary so the vehicle adapter
        can perform a real stop-and-shift before the following transaction.
        """

        if not self.points:
            return [], None, 0
        start = min(max(0, int(start_index)), len(self.points) - 1)
        original_direction = int(self.points[start].motion_direction)
        if original_direction == 0:
            for index in range(start - 1, -1, -1):
                original_direction = int(self.points[index + 1].motion_direction)
                if original_direction:
                    break
        if original_direction == 0:
            original_direction = 1
        anchor = self.points[start]
        output = [Breadcrumb(
            float(x), float(y), anchor.yaw, anchor.stamp,
            anchor.junction, anchor.turnaround, original_direction,
        )]
        travelled = 0.0
        previous_x, previous_y = float(x), float(y)
        target_index = start
        for index in range(start - 1, -1, -1):
            segment_direction = int(self.points[index + 1].motion_direction)
            if segment_direction not in (0, original_direction):
                break
            point = self.points[index]
            travelled += math.hypot(point.x - previous_x, point.y - previous_y)
            output.append(point)
            target_index = index
            previous_x, previous_y = point.x, point.y
            if travelled >= float(target_distance_m):
                break
        if len(output) < minimum_points:
            return [], start, -original_direction
        return output, target_index, -original_direction

    def older_corridor(
        self,
        x: float,
        y: float,
        *,
        target_distance_m: float,
        minimum_points: int = 2,
    ) -> Tuple[List[Breadcrumb], Optional[int]]:
        """Return current-to-older points in the direction a reversing car travels."""

        nearest = self.nearest_index(x, y)
        if nearest is None:
            return [], None
        output = [Breadcrumb(float(x), float(y), self.points[nearest].yaw, self.points[nearest].stamp)]
        travelled = 0.0
        previous_x, previous_y = float(x), float(y)
        target_index = nearest
        for index in range(nearest - 1, -1, -1):
            point = self.points[index]
            travelled += math.hypot(point.x - previous_x, point.y - previous_y)
            output.append(point)
            target_index = index
            previous_x, previous_y = point.x, point.y
            if travelled >= float(target_distance_m):
                break
        if len(output) < minimum_points:
            return [], nearest
        return output, target_index

    def path_distance(self, first_index: int, second_index: int) -> float:
        """Metric distance along recorded history between two indices."""

        if not self.points:
            return 0.0
        first = min(max(0, int(first_index)), len(self.points) - 1)
        second = min(max(0, int(second_index)), len(self.points) - 1)
        lower, upper = sorted((first, second))
        return float(
            sum(
                self.distance(self.points[index], self.points[index + 1])
                for index in range(lower, upper)
            )
        )

    def older_index_at_distance(self, start_index: int, distance_m: float) -> int:
        """Return the first older index at least ``distance_m`` down the trail."""

        if not self.points:
            raise ValueError("cannot query an empty breadcrumb trail")
        start = min(max(0, int(start_index)), len(self.points) - 1)
        target = start
        travelled = 0.0
        for index in range(start - 1, -1, -1):
            travelled += self.distance(self.points[index], self.points[index + 1])
            target = index
            if travelled + 1.0e-9 >= float(distance_m):
                break
        return target

    def most_recent_recovery_site(
        self,
        before_index: int,
        *,
        from_index: Optional[int] = None,
        minimum_path_distance_m: float = 0.0,
    ) -> Optional[int]:
        upper = min(max(0, int(before_index)), len(self.points) - 1)
        source = (
            len(self.points) - 1
            if from_index is None
            else min(max(0, int(from_index)), len(self.points) - 1)
        )
        for index in range(upper, -1, -1):
            point = self.points[index]
            sufficiently_old = (
                self.path_distance(index, source) + 1.0e-9
                >= float(minimum_path_distance_m)
            )
            if sufficiently_old and (point.junction or point.turnaround):
                return index
        if (
            self.points
            and self.path_distance(0, source) + 1.0e-9
            >= float(minimum_path_distance_m)
        ):
            return 0
        return None


def select_dead_end_egress_site(
    trail: BreadcrumbTrail,
    current_index: int,
    *,
    before_index: Optional[int] = None,
    minimum_distance_m: float = 1.5,
    maximum_distance_m: float = 8.0,
) -> Tuple[Optional[int], str, float]:
    """Select a real branch exit anchor instead of a fixed reverse pulse."""

    if not trail.points:
        return None, "EMPTY_TRAIL", 0.0
    current = min(max(0, int(current_index)), len(trail.points) - 1)
    before = max(0, current - 2) if before_index is None else min(
        max(0, int(before_index)), max(0, current - 1)
    )
    site = trail.most_recent_recovery_site(
        before,
        from_index=current,
        minimum_path_distance_m=float(minimum_distance_m),
    )
    kind = "RECOVERY_ANCHOR"
    if (
        site is None
        or trail.path_distance(site, current) > float(maximum_distance_m) + 1.0e-9
    ):
        site = trail.older_index_at_distance(
            current, float(maximum_distance_m)
        )
        kind = "CERTIFIED_TRAIL_LIMIT"
    distance = trail.path_distance(site, current)
    if site >= current or distance + 1.0e-9 < float(minimum_distance_m):
        return None, "INSUFFICIENT_CERTIFIED_INGRESS", distance
    return int(site), kind, float(distance)


class TopologyEdgeState(str, Enum):
    OPEN = "OPEN"
    DEAD_END = "DEAD_END"
    TEMP_BLOCKED = "TEMP_BLOCKED"


@dataclass
class TopologyNode:
    node_id: int
    x: float
    y: float
    yaw: float
    stamp: float
    junction: bool = False
    turnaround: bool = False
    visits: int = 1


@dataclass
class TopologyEdge:
    first: int
    second: int
    length: float
    state: TopologyEdgeState = TopologyEdgeState.OPEN
    traversals: int = 1
    failure_count: int = 0
    failure_terminal: Optional[Tuple[float, float]] = None
    blocked_until: Optional[float] = None


@dataclass
class FailedBranch:
    branch_id: int
    entry: Tuple[float, float]
    terminal: Tuple[float, float]
    points: Tuple[Tuple[float, float], ...]
    goal: Tuple[float, float]
    stamp: float
    static: bool = True
    failures: int = 1

    @property
    def length_m(self) -> float:
        values = np.asarray(self.points, dtype=float)
        if len(values) < 2:
            return 0.0
        return float(np.sum(np.linalg.norm(np.diff(values, axis=0), axis=1)))

    def egress_points(self) -> Tuple[Tuple[float, float], ...]:
        """Return the certified branch geometry from terminal to entry."""

        return tuple(reversed(self.points))


class TopologicalMemory:
    """Sparse, driven-path navigation memory without a dense-grid planner.

    Nodes and edges are created only from motion the vehicle has actually
    executed.  Dense SLAM occupancy is deliberately not converted into a
    global search graph.  Failed ingress suffixes remain as goal-aware taboo
    branches so escaping a dead end does not immediately erase the lesson.
    """

    def __init__(
        self,
        *,
        node_spacing_m: float = 0.75,
        merge_radius_m: float = 0.40,
        failure_buffer_m: float = 0.55,
        goal_branch_allowance_m: float = 0.90,
        maximum_nodes: int = 1200,
    ):
        if min(
            node_spacing_m,
            merge_radius_m,
            failure_buffer_m,
            goal_branch_allowance_m,
        ) <= 0.0:
            raise ValueError("topological memory distances must be positive")
        if maximum_nodes < 8:
            raise ValueError("topological memory requires at least eight nodes")
        self.node_spacing_m = float(node_spacing_m)
        self.merge_radius_m = float(merge_radius_m)
        self.failure_buffer_m = float(failure_buffer_m)
        self.goal_branch_allowance_m = float(goal_branch_allowance_m)
        self.maximum_nodes = int(maximum_nodes)
        self.nodes: Dict[int, TopologyNode] = {}
        self.edges: Dict[Tuple[int, int], TopologyEdge] = {}
        self.failed_branches: List[FailedBranch] = []
        self.last_node_id: Optional[int] = None
        self._next_node_id = 0
        self._next_branch_id = 0

    @staticmethod
    def _edge_key(first: int, second: int) -> Tuple[int, int]:
        return tuple(sorted((int(first), int(second))))

    @staticmethod
    def _distance_xy(first: Sequence[float], second: Sequence[float]) -> float:
        return math.hypot(float(first[0]) - float(second[0]), float(first[1]) - float(second[1]))

    @staticmethod
    def _point_segment_distance(point, first, second) -> float:
        point = np.asarray(point, dtype=float)
        first = np.asarray(first, dtype=float)
        second = np.asarray(second, dtype=float)
        delta = second - first
        denominator = float(np.dot(delta, delta))
        if denominator <= 1.0e-12:
            return float(np.linalg.norm(point - first))
        alpha = float(np.clip(np.dot(point - first, delta) / denominator, 0.0, 1.0))
        return float(np.linalg.norm(point - (first + alpha * delta)))

    def nearest_node(self, x: float, y: float, maximum_distance_m: Optional[float] = None):
        if not self.nodes:
            return None
        distance, node_id = min(
            (
                math.hypot(node.x - float(x), node.y - float(y)),
                node_id,
            )
            for node_id, node in self.nodes.items()
        )
        if maximum_distance_m is not None and distance > float(maximum_distance_m):
            return None
        return int(node_id)

    def reanchor(
        self,
        x: float,
        y: float,
        *,
        maximum_distance_m: float = 1.5,
    ) -> Optional[int]:
        """Anchor the next recorded edge at the vehicle's actual current node.

        Recovery motion is deliberately not learned as a new ingress trail.
        Consequently, a goal that pre-empts recovery can leave ``last_node_id``
        at the old dead-end terminal.  Re-anchoring prevents the next forward
        sample from creating an impossible shortcut across untraversed space.
        """

        self.last_node_id = self.nearest_node(
            float(x), float(y), maximum_distance_m=float(maximum_distance_m)
        )
        return self.last_node_id

    def _new_node(self, x, y, yaw, stamp, junction, turnaround):
        node_id = self._next_node_id
        self._next_node_id += 1
        self.nodes[node_id] = TopologyNode(
            node_id=node_id,
            x=float(x),
            y=float(y),
            yaw=wrap_angle(yaw),
            stamp=float(stamp),
            junction=bool(junction),
            turnaround=bool(turnaround),
        )
        return node_id

    def _connect(self, first: int, second: int):
        if first == second:
            return
        key = self._edge_key(first, second)
        length = self._distance_xy(
            (self.nodes[first].x, self.nodes[first].y),
            (self.nodes[second].x, self.nodes[second].y),
        )
        if length <= 1.0e-6:
            return
        edge = self.edges.get(key)
        if edge is None:
            self.edges[key] = TopologyEdge(first=key[0], second=key[1], length=length)
        else:
            edge.length = min(edge.length, length)
            edge.traversals += 1

    def record(
        self,
        x: float,
        y: float,
        yaw: float,
        stamp: float,
        *,
        junction: bool = False,
        turnaround: bool = False,
        force: bool = False,
    ) -> Optional[int]:
        """Record driven motion and return the current sparse node id."""

        nearest = self.nearest_node(x, y, self.merge_radius_m)
        special = bool(junction or turnaround)
        last = self.nodes.get(self.last_node_id) if self.last_node_id is not None else None
        displaced = (
            last is None
            or self._distance_xy((x, y), (last.x, last.y)) >= self.node_spacing_m
        )
        if nearest is None and not (force or special or displaced):
            return self.last_node_id
        if nearest is None:
            nearest = self._new_node(x, y, yaw, stamp, junction, turnaround)
        else:
            node = self.nodes[nearest]
            # ``record`` runs at control rate.  Remaining near one node is not
            # another visit; count only a real arrival from a different node.
            arrived = self.last_node_id is not None and nearest != self.last_node_id
            previous_weight = max(1, node.visits)
            sample_weight = previous_weight + 1
            node.x = (node.x * previous_weight + float(x)) / sample_weight
            node.y = (node.y * previous_weight + float(y)) / sample_weight
            node.yaw = wrap_angle(yaw)
            node.stamp = float(stamp)
            node.junction = node.junction or bool(junction)
            node.turnaround = node.turnaround or bool(turnaround)
            if arrived:
                node.visits += 1
        if self.last_node_id is not None:
            self._connect(self.last_node_id, nearest)
        self.last_node_id = nearest
        self._trim_nodes()
        return nearest

    def _trim_nodes(self):
        if len(self.nodes) <= self.maximum_nodes:
            return
        protected: Set[int] = set()
        for edge in self.edges.values():
            if edge.state != TopologyEdgeState.OPEN:
                protected.update((edge.first, edge.second))
        removable = sorted(
            (
                node for node in self.nodes.values()
                if node.node_id != self.last_node_id and node.node_id not in protected
            ),
            key=lambda node: node.stamp,
        )
        while len(self.nodes) > self.maximum_nodes and removable:
            node = removable.pop(0)
            self.nodes.pop(node.node_id, None)
            for key in [key for key in self.edges if node.node_id in key]:
                self.edges.pop(key, None)

    def mark_failed_branch(
        self,
        points: Iterable[Sequence[float]],
        *,
        goal_xy: Sequence[float],
        stamp: float,
        static: bool = True,
    ) -> Optional[FailedBranch]:
        values = tuple((float(point[0]), float(point[1])) for point in points)
        if len(values) < 2:
            return None
        entry, terminal = values[0], values[-1]
        branch = None
        for branch in self.failed_branches:
            terminal_close = self._distance_xy(
                branch.terminal, terminal
            ) <= 2.0 * self.failure_buffer_m
            entry_close = self._distance_xy(
                branch.entry, entry
            ) <= self.merge_radius_m
            existing = np.asarray(branch.points, dtype=float)
            candidate = np.asarray(values, dtype=float)
            overlaps = any(
                self._point_segment_distance(point, first, second)
                <= self.failure_buffer_m
                for point in candidate
                for first, second in zip(existing[:-1], existing[1:])
            )
            existing_direction = existing[-1] - existing[0]
            candidate_direction = candidate[-1] - candidate[0]
            denominator = float(
                np.linalg.norm(existing_direction)
                * np.linalg.norm(candidate_direction)
            )
            aligned = (
                denominator > 1.0e-9
                and float(np.dot(existing_direction, candidate_direction))
                / denominator
                > 0.35
            )
            if terminal_close and aligned and (entry_close or overlaps):
                branch.failures += 1
                old_length = float(
                    np.sum(np.linalg.norm(np.diff(existing, axis=0), axis=1))
                )
                new_length = float(
                    np.sum(np.linalg.norm(np.diff(candidate, axis=0), axis=1))
                )
                # A repeated failure often begins farther upstream.  Retain
                # the longer taboo suffix so the next attempt is diverted at
                # the junction rather than near the same terminal wall.
                if new_length > old_length:
                    branch.entry = entry
                    branch.terminal = terminal
                    branch.points = values
                branch.static = branch.static or bool(static)
                branch.stamp = float(stamp)
                break
        else:
            branch = FailedBranch(
                branch_id=self._next_branch_id,
                entry=entry,
                terminal=terminal,
                points=values,
                goal=(float(goal_xy[0]), float(goal_xy[1])),
                stamp=float(stamp),
                static=bool(static),
            )
            self._next_branch_id += 1
            self.failed_branches.append(branch)
        # Mark every edge observed in the newest failed traversal, including
        # an upstream extension of an already-known branch.  The old early
        # return incremented a counter but accidentally left these edges open.
        node_ids = []
        for point in values:
            node_id = self.nearest_node(*point, maximum_distance_m=1.5 * self.node_spacing_m)
            if node_id is not None and (not node_ids or node_ids[-1] != node_id):
                node_ids.append(node_id)
        for first, second in zip(node_ids[:-1], node_ids[1:]):
            edge = self.edges.get(self._edge_key(first, second))
            if edge is None:
                continue
            edge.state = TopologyEdgeState.DEAD_END if static else TopologyEdgeState.TEMP_BLOCKED
            edge.failure_count += 1
            edge.failure_terminal = terminal
        return branch

    def _branch_relevant(self, branch: FailedBranch, goal_xy: Sequence[float], stamp=None):
        if self._distance_xy(goal_xy, branch.terminal) <= self.goal_branch_allowance_m:
            return False
        if branch.static:
            return True
        return stamp is None or float(stamp) - branch.stamp <= 20.0

    def polyline_enters_failed_branch(
        self,
        points: Iterable[Sequence[float]],
        *,
        goal_xy: Sequence[float],
        stamp: Optional[float] = None,
    ) -> bool:
        """Return true when a candidate re-enters a remembered failed suffix."""

        values = np.asarray(list(points), dtype=float)
        if values.ndim != 2 or values.shape[1] != 2 or len(values) < 2:
            return False
        travelled = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(values, axis=0), axis=1))))
        probe = values[travelled >= min(0.35, 0.5 * travelled[-1])]
        if not len(probe):
            probe = values[-1:]
        candidate_direction = values[min(len(values) - 1, max(1, len(values) // 4))] - values[0]
        for branch in self.failed_branches:
            if not self._branch_relevant(branch, goal_xy, stamp):
                continue
            branch_values = np.asarray(branch.points, dtype=float)
            ingress = branch_values[min(len(branch_values) - 1, max(1, len(branch_values) // 4))] - branch_values[0]
            denominator = float(np.linalg.norm(candidate_direction) * np.linalg.norm(ingress))
            same_direction = denominator > 1.0e-9 and float(np.dot(candidate_direction, ingress)) / denominator > 0.45
            near_entry = self._distance_xy(values[0], branch.entry) <= 2.0 * self.failure_buffer_m
            intersects = any(
                self._point_segment_distance(point, first, second) <= self.failure_buffer_m
                for point in probe
                for first, second in zip(branch_values[:-1], branch_values[1:])
            )
            if intersects and (not near_entry or same_direction):
                return True
        return False

    def _transition_available(
        self,
        edge: TopologyEdge,
        source: int,
        target: int,
        goal_xy: Sequence[float],
        stamp=None,
    ):
        """Apply failed-branch state to one direction of a topology edge."""

        if edge.state == TopologyEdgeState.OPEN:
            return True
        if edge.failure_terminal is not None and self._distance_xy(
            goal_xy, edge.failure_terminal
        ) <= self.goal_branch_allowance_m:
            return True
        if edge.state == TopologyEdgeState.TEMP_BLOCKED:
            return edge.blocked_until is not None and stamp is not None and stamp >= edge.blocked_until
        if edge.failure_terminal is None:
            return False
        source_node = self.nodes.get(int(source))
        target_node = self.nodes.get(int(target))
        if source_node is None or target_node is None:
            return False
        source_distance = self._distance_xy(
            (source_node.x, source_node.y), edge.failure_terminal
        )
        target_distance = self._distance_xy(
            (target_node.x, target_node.y), edge.failure_terminal
        )
        # Moving farther from the failed terminal is egress and remains
        # legal.  The same geometric edge toward the terminal is taboo.
        return target_distance > source_distance + 1.0e-6

    def guidance_path(
        self,
        current_xy: Sequence[float],
        goal_xy: Sequence[float],
        *,
        stamp: Optional[float] = None,
        goal_proxy_radius_m: float = 2.0,
        minimum_goal_improvement_m: float = 0.45,
    ) -> List[Tuple[float, float]]:
        """Return a sparse known-path hint, never a dense occupancy-grid plan."""

        start = self.nearest_node(*current_xy, maximum_distance_m=1.5)
        if start is None:
            return []
        adjacency: Dict[int, List[Tuple[int, float]]] = {node_id: [] for node_id in self.nodes}
        for edge in self.edges.values():
            if self._transition_available(
                edge, edge.first, edge.second, goal_xy, stamp
            ):
                adjacency.setdefault(edge.first, []).append(
                    (edge.second, edge.length)
                )
            if self._transition_available(
                edge, edge.second, edge.first, goal_xy, stamp
            ):
                adjacency.setdefault(edge.second, []).append(
                    (edge.first, edge.length)
                )
        distances = {start: 0.0}
        previous: Dict[int, int] = {}
        queue = [(0.0, start)]
        while queue:
            cost, node_id = heapq.heappop(queue)
            if cost > distances.get(node_id, math.inf):
                continue
            for neighbour, length in adjacency.get(node_id, ()):
                candidate = cost + length + 0.03 * self.nodes[neighbour].visits
                if candidate + 1.0e-9 < distances.get(neighbour, math.inf):
                    distances[neighbour] = candidate
                    previous[neighbour] = node_id
                    heapq.heappush(queue, (candidate, neighbour))
        if len(distances) <= 1:
            return []
        current_goal_distance = self._distance_xy(current_xy, goal_xy)
        proxy_candidates = [
            node_id for node_id in distances
            if self._distance_xy((self.nodes[node_id].x, self.nodes[node_id].y), goal_xy)
            <= float(goal_proxy_radius_m)
        ]
        if proxy_candidates:
            target = min(
                proxy_candidates,
                key=lambda node_id: (
                    self._distance_xy(
                        (self.nodes[node_id].x, self.nodes[node_id].y), goal_xy
                    ),
                    distances[node_id],
                ),
            )
        else:
            target = min(
                distances,
                key=lambda node_id: (
                    self._distance_xy((self.nodes[node_id].x, self.nodes[node_id].y), goal_xy),
                    distances[node_id],
                ),
            )
            target_distance = self._distance_xy(
                (self.nodes[target].x, self.nodes[target].y), goal_xy
            )
            if target_distance > current_goal_distance - float(minimum_goal_improvement_m):
                return []
        if target == start:
            return []
        node_path = [target]
        while node_path[-1] != start:
            parent = previous.get(node_path[-1])
            if parent is None:
                return []
            node_path.append(parent)
        node_path.reverse()
        return [(self.nodes[node_id].x, self.nodes[node_id].y) for node_id in node_path]

    def summary(self):
        counts = {state.value: 0 for state in TopologyEdgeState}
        for edge in self.edges.values():
            counts[edge.state.value] += 1
        return {
            "nodes": len(self.nodes),
            "edges": len(self.edges),
            "edge_states": counts,
            "failed_branches": len(self.failed_branches),
        }


class MemoryNavigationState(Enum):
    IDLE = "IDLE"
    GOAL_SEEK = "GOAL_SEEK"
    SUSPECT_DEAD_END = "SUSPECT_DEAD_END"
    BACKTRACK_REVERSE = "BACKTRACK_REVERSE"
    FAR_DEAD_END_EGRESS = "FAR_DEAD_END_EGRESS"
    RESUME_FORWARD = "RESUME_FORWARD"
    DEAD_END_ESCAPED = "DEAD_END_ESCAPED"
    GOAL_REACHED = "GOAL_REACHED"
    SAFE_STOP = "SAFE_STOP"


class DeadEndRecoverySupervisor:
    """Temporal dead-end classifier; dynamic blockage never triggers backing.

    ``backtrack_enabled`` is deliberately an authority boundary rather than a
    tuning gain.  The legacy M4 backend can still replay a certified ingress
    stack, while the M6 visibility backend leaves all gear decisions to the
    Ackermann local planner and uses graph replanning for route recovery.
    """

    def __init__(
        self,
        confirmation_s: float = 1.8,
        minimum_trail_points: int = 4,
        *,
        backtrack_enabled: bool = True,
    ):
        if confirmation_s <= 0.0 or minimum_trail_points < 2:
            raise ValueError("dead-end supervisor parameters are invalid")
        self.confirmation_s = float(confirmation_s)
        self.minimum_trail_points = int(minimum_trail_points)
        self.backtrack_enabled = bool(backtrack_enabled)
        self.state = MemoryNavigationState.IDLE
        self.blocked_since: Optional[float] = None

    def reset(self, active: bool = False):
        self.state = MemoryNavigationState.GOAL_SEEK if active else MemoryNavigationState.IDLE
        self.blocked_since = None

    def update_goal_seek(
        self,
        stamp: float,
        *,
        progress_m: float,
        static_blocked: bool,
        dynamic_blocked: bool,
        rear_clear: bool,
        trail_points: int,
        maneuver_active: bool = False,
    ) -> MemoryNavigationState:
        if self.state not in (
            MemoryNavigationState.GOAL_SEEK,
            MemoryNavigationState.SUSPECT_DEAD_END,
        ):
            return self.state
        stamp = float(stamp)
        if maneuver_active:
            # A committed Ackermann turn can temporarily stop improving the
            # Euclidean goal distance.  It owns control until its completion
            # contract succeeds or hard safety explicitly exhausts a leg.
            self.blocked_since = None
            self.state = MemoryNavigationState.GOAL_SEEK
            return self.state
        blocked = bool(static_blocked) and not bool(dynamic_blocked) and progress_m < 0.03
        if not blocked:
            self.blocked_since = None
            self.state = MemoryNavigationState.GOAL_SEEK
            return self.state
        if self.blocked_since is None:
            self.blocked_since = stamp
        confirmed = stamp - self.blocked_since >= self.confirmation_s
        # Keep the externally visible authority stable during the debounce
        # window.  A one-frame hard veto is evidence, not a navigation-mode
        # transaction, and publishing SUSPECT immediately made RViz and the
        # local planner appear to oscillate between competing supervisors.
        if not confirmed:
            self.state = MemoryNavigationState.GOAL_SEEK
            return self.state
        self.state = MemoryNavigationState.SUSPECT_DEAD_END
        if (
            self.backtrack_enabled
            and confirmed
            and rear_clear
            and trail_points >= self.minimum_trail_points
        ):
            self.state = MemoryNavigationState.BACKTRACK_REVERSE
        return self.state

    def begin_resume(self):
        if self.state != MemoryNavigationState.BACKTRACK_REVERSE:
            raise RuntimeError("resume can start only after breadcrumb backtracking")
        self.state = MemoryNavigationState.RESUME_FORWARD

    def force_backtrack(self):
        """Commit a backtrack after an independently certified local loop."""

        if not self.backtrack_enabled:
            raise RuntimeError("breadcrumb motion authority is disabled")
        if self.state not in (
            MemoryNavigationState.GOAL_SEEK,
            MemoryNavigationState.SUSPECT_DEAD_END,
        ):
            raise RuntimeError("forced backtrack requires active goal seeking")
        self.state = MemoryNavigationState.BACKTRACK_REVERSE
        self.blocked_since = None

    def force_certified_egress(self):
        """Commit an exclusive, FAR-supervised dead-end exit transaction.

        This is deliberately separate from ordinary breadcrumb authority.  A
        caller must first establish repeated static hard blocks and supplies a
        bounded, physically driven trail.  Breadcrumb geometry supplies an
        exit connector, but current SLAM/LiDAR and the local hard veto remain
        the motion authority throughout the transaction.
        """

        if self.state not in (
            MemoryNavigationState.GOAL_SEEK,
            MemoryNavigationState.SUSPECT_DEAD_END,
        ):
            raise RuntimeError("certified egress requires active goal seeking")
        self.state = MemoryNavigationState.FAR_DEAD_END_EGRESS
        self.blocked_since = None

    def complete_certified_egress(self):
        if self.state != MemoryNavigationState.FAR_DEAD_END_EGRESS:
            raise RuntimeError("certified egress completion requires FAR dead-end egress")
        self.state = MemoryNavigationState.GOAL_SEEK
        self.blocked_since = None

    def escaped(self):
        if self.state != MemoryNavigationState.RESUME_FORWARD:
            raise RuntimeError("escape can finish only from forward resume")
        self.state = MemoryNavigationState.DEAD_END_ESCAPED

    def continue_goal_seek(self):
        if self.state != MemoryNavigationState.DEAD_END_ESCAPED:
            raise RuntimeError("goal seek can resume only after a certified escape")
        self.state = MemoryNavigationState.GOAL_SEEK
        self.blocked_since = None

    def resume_failed(self):
        """Return to certified backtracking when a recovery site cannot turn."""

        if self.state != MemoryNavigationState.RESUME_FORWARD:
            raise RuntimeError("resume failure is valid only during forward resume")
        self.state = MemoryNavigationState.BACKTRACK_REVERSE
        self.blocked_since = None

    def complete_goal(self):
        self.state = MemoryNavigationState.GOAL_REACHED
        self.blocked_since = None

    def fail_safe(self):
        self.state = MemoryNavigationState.SAFE_STOP


def ray_clearance(
    grid: np.ndarray,
    resolution: float,
    origin_xy: Sequence[float],
    angle: float,
    *,
    maximum_m: float = 3.0,
) -> float:
    """Known-free distance from body origin in a rolling occupancy grid."""

    values = np.asarray(grid)
    if values.ndim != 2 or resolution <= 0.0:
        raise ValueError("invalid occupancy grid")
    origin = np.asarray(origin_xy, dtype=float)
    step = max(0.5 * float(resolution), 0.025)
    distances = np.arange(0.0, float(maximum_m) + 0.5 * step, step)
    points = np.column_stack((np.cos(angle) * distances, np.sin(angle) * distances))
    cells = np.floor((points - origin) / float(resolution)).astype(int)
    for distance, cell in zip(distances, cells):
        x, y = int(cell[0]), int(cell[1])
        if y < 0 or x < 0 or y >= values.shape[0] or x >= values.shape[1]:
            return float(distance)
        if values[y, x] != 0:
            return float(distance)
    return float(maximum_m)


def local_space_features(grid, resolution, origin_xy):
    angles = {
        "front": 0.0,
        "left": math.pi / 2.0,
        "right": -math.pi / 2.0,
        "rear": math.pi,
    }
    clearances = {
        name: ray_clearance(grid, resolution, origin_xy, angle)
        for name, angle in angles.items()
    }
    openings = sum(value >= 1.0 for value in clearances.values())
    clearances["junction"] = openings >= 3
    clearances["turnaround"] = (
        clearances["front"] >= 0.75
        and clearances["rear"] >= 0.75
        and max(clearances["left"], clearances["right"]) >= 0.75
    )
    return clearances


def resample_polyline(points: Iterable[Sequence[float]], count: int = 40) -> np.ndarray:
    values = np.asarray(list(points), dtype=float)
    if values.ndim != 2 or values.shape[1] != 2 or len(values) < 2:
        raise ValueError("polyline needs at least two planar points")
    if count < 2:
        raise ValueError("resample count must be at least two")
    delta = np.linalg.norm(np.diff(values, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(delta)))
    keep = np.concatenate(([True], np.diff(cumulative) > 1.0e-6))
    values, cumulative = values[keep], cumulative[keep]
    if len(values) < 2 or cumulative[-1] <= 1.0e-6:
        raise ValueError("polyline length must be positive")
    query = np.linspace(0.0, cumulative[-1], int(count))
    return np.column_stack(
        (np.interp(query, cumulative, values[:, 0]), np.interp(query, cumulative, values[:, 1]))
    )
