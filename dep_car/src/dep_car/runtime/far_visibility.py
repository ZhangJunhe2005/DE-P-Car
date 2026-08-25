"""FAR-inspired dynamic 2-D visibility routing for online SLAM maps.

This is an independent ROS-free implementation of the architectural ideas in
FAR Planner (Yang et al., IROS 2022): obstacle contours are converted to a
sparse visibility graph, known-space routes are preferred, and an attemptable
route through unknown space is used only when no completely observed route is
available.  It intentionally does not search the occupancy grid and it does
not generate vehicle controls; Ackermann execution remains the responsibility
of DE-P and its hard safety layer.
"""

from dataclasses import dataclass
import heapq
import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class VisibilityNode:
    node_id: int
    x: float
    y: float
    kind: str


@dataclass(frozen=True)
class VisibilityEdge:
    first: int
    second: int
    length: float
    unknown_fraction: float

    @property
    def known(self) -> bool:
        return self.unknown_fraction <= 1.0e-6


@dataclass(frozen=True)
class VisibilityPlan:
    status: str
    mode: str
    path: Tuple[Tuple[float, float], ...]
    nodes: Tuple[VisibilityNode, ...]
    edges: Tuple[VisibilityEdge, ...]
    path_cost: Optional[float]
    known_edges: int
    attemptable_edges: int
    reason: str
    path_length: Optional[float] = None
    path_unknown_fraction: float = 1.0


@dataclass(frozen=True)
class VisibilityRouteAcquisitionDecision:
    # ``accepted`` means that the route has enough observed-space evidence to
    # become ordinary FAR route authority.  ``motion_authorized`` is the
    # separate FAR attemptable-navigation contract: a stable route may reveal
    # unknown space while the local planner and hard veto certify every short
    # Ackermann primitive.  Conflating the two contracts caused the vehicle to
    # wait for map growth which could only happen after it moved.
    accepted: bool
    confirmations: int
    bearing_rad: Optional[float]
    reason: str
    motion_authorized: bool = False


def route_initial_bearing(path, lookahead_m: float = 1.50):
    """Return the first metric-lookahead bearing of a 2-D route."""

    points = np.asarray(path, dtype=float)
    if points.ndim != 2 or points.shape[1:] != (2,) or len(points) < 2:
        return None
    start = points[0]
    target = points[-1]
    remaining = float(lookahead_m)
    for first, second in zip(points[:-1], points[1:]):
        delta = second - first
        length = float(np.linalg.norm(delta))
        if length <= 1.0e-9:
            continue
        if remaining <= length:
            target = first + (remaining / length) * delta
            break
        remaining -= length
    delta = target - start
    if float(np.linalg.norm(delta)) <= 1.0e-9:
        return None
    return math.atan2(float(delta[1]), float(delta[0]))


def polyline_prefix(path, distance_m: float):
    """Return an interpolated metric prefix without skipping route corners."""

    points = np.asarray(path, dtype=float)
    if (
        points.ndim != 2
        or points.shape[1:] != (2,)
        or not len(points)
        or not np.all(np.isfinite(points))
    ):
        return ()
    remaining = max(0.0, float(distance_m))
    output = [points[0].copy()]
    for first, second in zip(points[:-1], points[1:]):
        delta = second - first
        length = float(np.linalg.norm(delta))
        if length <= 1.0e-9:
            continue
        if remaining <= length:
            output.append(first + (remaining / length) * delta)
            break
        output.append(second.copy())
        remaining -= length
    if len(output) == 1 and len(points) >= 2:
        output.append(points[1].copy())
    return tuple((float(point[0]), float(point[1])) for point in output)


def measured_pose_revalidation_authorized(
    plan,
    *,
    observed_prefix_clear: bool,
    maximum_attemptable_unknown_fraction: float,
):
    """Authorize a stopped re-anchor only when its immediate prefix is known.

    A course-revalidation stop cannot create a new SLAM content revision, so
    requiring another motion-derived acquisition confirmation is a liveness
    deadlock.  This exception does not trust an arbitrary speculative route:
    the plan must start at the newly measured pose, its complete uncertainty
    must remain within the normal attemptable bound, and the short prefix that
    DE-P can execute next must be fully observed and footprint-clear.
    """

    limit = float(maximum_attemptable_unknown_fraction)
    if not 0.0 <= limit <= 1.0:
        raise ValueError("maximum attemptable unknown fraction is invalid")
    return bool(
        plan is not None
        and plan.status == "PASS"
        and len(plan.path) >= 2
        and observed_prefix_clear
        and (
            plan.mode == "KNOWN_VISIBILITY"
            or (
                plan.mode == "ATTEMPTABLE_VISIBILITY"
                and float(plan.path_unknown_fraction) <= limit
            )
        )
    )


def transient_route_lease_authorized(
    *,
    previous_route_motion_authorized: bool,
    same_goal: bool,
    local_prefix_clear: bool,
    local_prefix_length_m: float,
    minimum_prefix_length_m: float,
    dropout_age_s: float,
    grace_s: float,
):
    """Keep a certified route through a short rolling-planner dropout.

    A visibility-graph rebuild is a candidate replacement, not an atomic
    revocation of the route which is already controlling the vehicle.  The
    previous route may therefore remain authoritative for a bounded time when
    its *local execution prefix* is still footprint-clear on the newest map.
    This deliberately does not certify the distant suffix: DE-P and its hard
    veto continue to validate every short Ackermann primitive, while FAR keeps
    replanning the rest of the route in the background.
    """

    minimum = float(minimum_prefix_length_m)
    age = float(dropout_age_s)
    grace = float(grace_s)
    length = float(local_prefix_length_m)
    if minimum <= 0.0 or grace <= 0.0:
        raise ValueError("route lease distances and duration must be positive")
    if not all(math.isfinite(value) for value in (age, length)):
        return False
    return bool(
        previous_route_motion_authorized
        and same_goal
        and local_prefix_clear
        and length >= minimum
        and 0.0 <= age <= grace
    )


class VisibilityRouteAcquisitionGate:
    """Separate stable attemptable motion from observed-route acceptance.

    A newly published SLAM map can initially expose only one side of a wall.
    The first attemptable visibility path is therefore a hypothesis, not yet
    motion authority.  Known-space paths are accepted immediately.  Routes
    containing unknown space must retain a similar first direction and cost
    over distinct SLAM *content* revisions or sufficiently separated measured
    sensor poses before they may move the vehicle.  A repeated publication at
    the same pose is not new route evidence.  Known paths use the same short
    continuity transaction: an online graph can flip sides while SLAM is
    filling the first wall contour even when every edge of each individual
    hypothesis happens to lie in currently known cells.

    ``accepted`` and ``motion_authorized`` deliberately have different
    meanings.  A high-detour route may remain too unknown to be accepted as a
    complete route, while its stable, currently observed prefix is still the
    best way to reveal the next viewpoint.  The caller must independently
    certify that bounded prefix against the latest inflated occupancy map
    before using ``motion_authorized``.  This prevents the old deadlock where
    the route needed more observations but the vehicle was forbidden to move
    far enough to create them.
    """

    def __init__(
        self,
        *,
        minimum_confirmations: int = 2,
        minimum_stable_s: float = 0.60,
        maximum_bearing_change_rad: float = math.radians(30.0),
        maximum_relative_cost_change: float = 0.35,
        maximum_attemptable_detour_ratio: float = 2.25,
        maximum_high_detour_unknown_fraction: float = 0.20,
        high_detour_extra_confirmations: int = 2,
        high_detour_minimum_stable_s: float = 1.20,
        lookahead_m: float = 1.50,
        minimum_observer_displacement_m: float = 0.25,
    ):
        if int(minimum_confirmations) < 1:
            raise ValueError("route acquisition needs at least one confirmation")
        if min(
            float(minimum_stable_s),
            float(maximum_bearing_change_rad),
            float(maximum_relative_cost_change),
            float(maximum_high_detour_unknown_fraction),
            float(high_detour_minimum_stable_s),
            float(lookahead_m),
            float(minimum_observer_displacement_m),
        ) <= 0.0:
            raise ValueError("route acquisition thresholds must be positive")
        if float(maximum_attemptable_detour_ratio) <= 1.0:
            raise ValueError("attemptable detour ratio must exceed one")
        if not 0.0 < float(maximum_high_detour_unknown_fraction) < 1.0:
            raise ValueError("high-detour unknown fraction must be between zero and one")
        if int(high_detour_extra_confirmations) < 1:
            raise ValueError("high-detour routes need extra confirmations")
        self.minimum_confirmations = int(minimum_confirmations)
        self.minimum_stable_s = float(minimum_stable_s)
        self.maximum_bearing_change_rad = float(maximum_bearing_change_rad)
        self.maximum_relative_cost_change = float(maximum_relative_cost_change)
        self.maximum_attemptable_detour_ratio = float(
            maximum_attemptable_detour_ratio
        )
        self.maximum_high_detour_unknown_fraction = float(
            maximum_high_detour_unknown_fraction
        )
        self.high_detour_extra_confirmations = int(
            high_detour_extra_confirmations
        )
        self.high_detour_minimum_stable_s = float(
            high_detour_minimum_stable_s
        )
        self.lookahead_m = float(lookahead_m)
        self.minimum_observer_displacement_m = float(
            minimum_observer_displacement_m
        )
        self.reset()

    def reset(self):
        self.accepted = False
        self.motion_authorized = False
        self.confirmations = 0
        self.first_stable_stamp = None
        self.last_revision = None
        self.last_bearing = None
        self.last_cost = None
        self.last_observer_position = None
        self.reason = "waiting_for_visibility_route"

    @staticmethod
    def _wrap(angle):
        return math.atan2(math.sin(float(angle)), math.cos(float(angle)))

    def _first_bearing(self, path):
        return route_initial_bearing(path, self.lookahead_m)

    def update(
        self,
        plan,
        *,
        stamp: float,
        map_revision: int,
        observer_position=None,
    ):
        if plan is None or plan.status != "PASS" or len(plan.path) < 2:
            reason = (
                "visibility_graph_unavailable"
                if plan is None
                else str(plan.reason)
            )
            # Acceptance belongs to one continuous candidate route.  Keeping
            # ``accepted`` and its confirmations after FAR reports NO_ROUTE
            # lets a later, unrelated attemptable route inherit stale motion
            # authority.  It also makes diagnostics claim confirmations for
            # an empty route.  A disconnected/degenerate candidate therefore
            # ends the acquisition transaction completely.
            self.reset()
            self.reason = reason
            return VisibilityRouteAcquisitionDecision(
                False, 0, None, self.reason, False
            )
        bearing = self._first_bearing(plan.path)
        if bearing is None:
            self.reset()
            self.reason = "visibility_route_degenerate"
            return VisibilityRouteAcquisitionDecision(
                False, 0, None, self.reason, False
            )
        cost = float(plan.path_cost or 0.0)
        known_route = plan.mode == "KNOWN_VISIBILITY"
        observer = None
        if observer_position is not None:
            candidate = np.asarray(observer_position, dtype=float)
            if candidate.shape == (2,) and np.all(np.isfinite(candidate)):
                observer = candidate
        if self.accepted:
            consistent_with_accepted = bool(
                self.last_bearing is not None
                and abs(self._wrap(bearing - self.last_bearing))
                <= self.maximum_bearing_change_rad
                and self.last_cost is not None
                and abs(cost - self.last_cost)
                / max(1.0e-6, abs(self.last_cost))
                <= self.maximum_relative_cost_change
            )
            if consistent_with_accepted:
                self.last_revision = int(map_revision)
                self.last_bearing = float(bearing)
                self.last_cost = cost
                if observer is not None:
                    self.last_observer_position = observer.copy()
                return VisibilityRouteAcquisitionDecision(
                    True,
                    self.confirmations,
                    self.last_bearing,
                    self.reason,
                    True,
                )
            # A materially different attemptable path is a new candidate,
            # even if the previous one was accepted.  The caller can retain
            # the old safe suffix while this one earns fresh confirmations.
            self.reset()
        direct_distance = math.hypot(
            float(plan.path[-1][0]) - float(plan.path[0][0]),
            float(plan.path[-1][1]) - float(plan.path[0][1]),
        )
        path_length = plan.path_length
        if path_length is None:
            path_length = sum(
                math.hypot(
                    float(second[0]) - float(first[0]),
                    float(second[1]) - float(first[1]),
                )
                for first, second in zip(plan.path[:-1], plan.path[1:])
            )
        # ``path_cost`` also carries the amount of unknown-space exposure.
        # A geometrically short line through unmapped rooms can otherwise
        # look harmless even though its uncertainty-weighted cost is several
        # times the direct distance.  Use the stricter of geometric and
        # uncertainty-weighted ratios for the high-detour evidence gate.
        detour_ratio = max(float(path_length), cost) / max(
            1.0e-6, direct_distance
        )
        high_detour = detour_ratio > self.maximum_attemptable_detour_ratio
        unknown_fraction = float(plan.path_unknown_fraction)
        observer_advanced = bool(
            observer is not None
            and self.last_observer_position is not None
            and float(np.linalg.norm(observer - self.last_observer_position))
            >= self.minimum_observer_displacement_m
        )
        if self.last_revision == int(map_revision) and not observer_advanced:
            return VisibilityRouteAcquisitionDecision(
                self.accepted,
                self.confirmations,
                self.last_bearing,
                self.reason,
                self.motion_authorized,
            )
        consistent = bool(
            self.last_bearing is not None
            and abs(self._wrap(bearing - self.last_bearing))
            <= self.maximum_bearing_change_rad
            and self.last_cost is not None
            and abs(cost - self.last_cost) / max(1.0e-6, abs(self.last_cost))
            <= self.maximum_relative_cost_change
        )
        if consistent:
            self.confirmations += 1
        else:
            self.confirmations = 1
            self.first_stable_stamp = float(stamp)
        if self.first_stable_stamp is None:
            self.first_stable_stamp = float(stamp)
        self.last_revision = int(map_revision)
        self.last_bearing = float(bearing)
        self.last_cost = cost
        if observer is not None:
            self.last_observer_position = observer.copy()
        stable_s = float(stamp) - float(self.first_stable_stamp)
        required_confirmations = self.minimum_confirmations
        required_stable_s = self.minimum_stable_s
        if high_detour:
            required_confirmations += self.high_detour_extra_confirmations
            required_stable_s = max(
                required_stable_s, self.high_detour_minimum_stable_s
            )
        sufficiently_observed = bool(
            not high_detour
            or unknown_fraction <= self.maximum_high_detour_unknown_fraction
        )
        # A short, ordinary attemptable route may reveal its next viewpoint
        # after the direction and cost settle.  A highly unknown *detour* is a
        # qualitatively different hypothesis: it cannot become full route
        # authority until enough of the suffix is observed, but its measured
        # local prefix must remain usable to avoid a mapping-motion deadlock.
        # This is only *candidate* motion authority.  In particular, a stable
        # high-detour route may expose an unknown suffix.  The ROS owner must
        # still prove that the short execution prefix is observed and clear;
        # ``locally_certified_route_motion`` below is the common fail-closed
        # boundary used for that check.
        self.motion_authorized = bool(
            self.confirmations >= self.minimum_confirmations
            and stable_s >= self.minimum_stable_s
        )
        self.accepted = bool(
            sufficiently_observed
            and self.confirmations >= required_confirmations
            and stable_s >= required_stable_s
        )
        if self.accepted:
            self.reason = (
                "stable_observed_high_detour_route"
                if high_detour
                else "stable_known_visibility_route"
                if known_route
                else "stable_attemptable_route"
            )
        elif self.motion_authorized and high_detour:
            self.reason = "stable_attemptable_navigation_high_detour"
        elif self.motion_authorized:
            self.reason = "stable_attemptable_navigation"
        elif high_detour and not sufficiently_observed:
            self.reason = "high_detour_route_needs_more_observed_space"
        elif high_detour:
            self.reason = "settling_observed_high_detour_route"
        elif known_route:
            self.reason = "settling_known_visibility_route"
        else:
            self.reason = "settling_attemptable_route"
        return VisibilityRouteAcquisitionDecision(
            self.accepted,
            self.confirmations,
            self.last_bearing,
            self.reason,
            self.motion_authorized,
        )


def locally_certified_route_motion(plan, acquisition, observed_prefix_clear):
    """Authorize only a stable FAR route's measured, bounded local prefix.

    The whole visibility path may still cross unknown space.  This helper does
    not accept that suffix and does not weaken collision checking: it merely
    permits the rolling FAR target to advance through the already observed
    prefix so the sensor can reveal the next contour corner.
    """

    return bool(
        plan is not None
        and plan.status == "PASS"
        and len(plan.path) >= 2
        and acquisition is not None
        and acquisition.motion_authorized
        and observed_prefix_clear
    )


class DynamicVisibilityPlanner:
    """Build and search a sparse graph over occupied-map polygon corners."""

    def __init__(
        self,
        *,
        inflation_radius_m: float = 0.38,
        contour_simplification_m: float = 0.22,
        vertex_offset_m: float = 0.18,
        node_separation_m: float = 0.28,
        maximum_nodes: int = 220,
        maximum_edge_length_m: float = 12.0,
        unknown_cost_weight: float = 1.25,
        start_heading_weight_m: float = 1.20,
        reverse_start_penalty_m: float = 2.00,
        occupied_threshold: int = 50,
    ):
        positive = (
            inflation_radius_m,
            contour_simplification_m,
            vertex_offset_m,
            node_separation_m,
            maximum_edge_length_m,
            unknown_cost_weight,
        )
        if min(float(value) for value in positive) <= 0.0:
            raise ValueError("visibility planner distances and weights must be positive")
        if int(maximum_nodes) < 4:
            raise ValueError("visibility planner requires at least four nodes")
        if min(float(start_heading_weight_m), float(reverse_start_penalty_m)) < 0.0:
            raise ValueError("visibility start-heading penalties cannot be negative")
        self.inflation_radius_m = float(inflation_radius_m)
        self.contour_simplification_m = float(contour_simplification_m)
        self.vertex_offset_m = float(vertex_offset_m)
        self.node_separation_m = float(node_separation_m)
        self.maximum_nodes = int(maximum_nodes)
        self.maximum_edge_length_m = float(maximum_edge_length_m)
        self.unknown_cost_weight = float(unknown_cost_weight)
        self.start_heading_weight_m = float(start_heading_weight_m)
        self.reverse_start_penalty_m = float(reverse_start_penalty_m)
        self.occupied_threshold = int(occupied_threshold)

    @staticmethod
    def _world_to_grid(point, resolution, origin):
        return (
            int(math.floor((float(point[1]) - float(origin[1])) / resolution)),
            int(math.floor((float(point[0]) - float(origin[0])) / resolution)),
        )

    @staticmethod
    def _grid_to_world(row, column, resolution, origin):
        return (
            float(origin[0]) + (float(column) + 0.5) * resolution,
            float(origin[1]) + (float(row) + 0.5) * resolution,
        )

    @staticmethod
    def _inside(shape, row, column):
        return 0 <= row < shape[0] and 0 <= column < shape[1]

    def _inflated_occupancy(self, values, resolution):
        """Return the same footprint-inflated mask used by graph building.

        Route handoff uses this independently of graph reconstruction.  A
        previously selected visibility route therefore remains committed
        while it is still safe on the newest SLAM revision; newly observed
        occupied cells revoke it immediately.
        """

        occupied = (np.asarray(values) >= self.occupied_threshold).astype(np.uint8)
        radius = max(1, int(math.ceil(self.inflation_radius_m / float(resolution))))
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
        )
        return cv2.dilate(occupied, kernel) > 0

    def path_is_traversable(
        self,
        path,
        values,
        resolution,
        origin,
        *,
        maximum_unknown_fraction=1.0,
    ):
        """Check a route suffix against the newest map without replanning it.

        Unknown cells remain attemptable, matching :meth:`plan`; only a
        footprint-inflated occupied crossing invalidates the committed route.
        """

        values = np.asarray(values, dtype=np.int16)
        points = np.asarray(path, dtype=float)
        resolution = float(resolution)
        if values.ndim != 2 or not values.size or resolution <= 0.0:
            return False
        if points.ndim != 2 or points.shape[1:] != (2,) or len(points) < 2:
            return False
        maximum_unknown_fraction = float(maximum_unknown_fraction)
        if not 0.0 <= maximum_unknown_fraction <= 1.0:
            raise ValueError("maximum unknown fraction must be between zero and one")
        inflated = self._inflated_occupancy(values, resolution)
        profiles = [
            self._segment_profile(
                first,
                second,
                values,
                inflated,
                resolution,
                origin,
            )
            for first, second in zip(points[:-1], points[1:])
        ]
        if any(profile is None for profile in profiles):
            return False
        total_length = sum(profile[0] for profile in profiles)
        unknown_fraction = sum(
            profile[0] * profile[1] for profile in profiles
        ) / max(1.0e-9, total_length)
        return unknown_fraction <= maximum_unknown_fraction

    def longest_traversable_prefix(
        self,
        path,
        values,
        resolution,
        origin,
        *,
        maximum_unknown_fraction=1.0,
    ):
        """Return a safe rolling prefix without extrapolating stale history."""

        points = np.asarray(path, dtype=float)
        if points.ndim != 2 or points.shape[1:] != (2,) or len(points) < 2:
            return ()
        for stop in range(len(points), 1, -1):
            prefix = points[:stop]
            if self.path_is_traversable(
                prefix,
                values,
                resolution,
                origin,
                maximum_unknown_fraction=maximum_unknown_fraction,
            ):
                return tuple(
                    (float(point[0]), float(point[1])) for point in prefix
                )
        return ()

    def _margin_egress_path_is_traversable(
        self,
        path,
        values,
        resolution,
        origin,
        *,
        maximum_unknown_fraction,
        minimum_clearance_improvement_m,
        worsening_tolerance_m,
    ):
        """Validate motion out of inflation overlap without weakening walls.

        A vehicle that stopped close to a wall can begin a recovery with its
        centre inside the conservative visibility inflation band even though
        its physical footprint is collision-free.  Such a route is legal only
        while raw occupied cells are never crossed and obstacle clearance is
        monotonically recovered.  This mirrors the local swept-footprint
        margin-egress contract; it is not a general relaxation of FAR routes.
        """

        values = np.asarray(values, dtype=np.int16)
        points = np.asarray(path, dtype=float)
        resolution = float(resolution)
        if values.ndim != 2 or not values.size or resolution <= 0.0:
            return False
        if points.ndim != 2 or points.shape[1:] != (2,) or len(points) < 2:
            return False
        samples = []
        for first, second in zip(points[:-1], points[1:]):
            length = float(np.linalg.norm(second - first))
            count = max(
                2,
                int(math.ceil(length / max(0.5 * resolution, 0.025))) + 1,
            )
            segment = first[None, :] + np.linspace(0.0, 1.0, count)[:, None] * (
                second - first
            )[None, :]
            samples.extend(segment if not samples else segment[1:])
        samples = np.asarray(samples, dtype=float)
        columns = np.floor(
            (samples[:, 0] - float(origin[0])) / resolution
        ).astype(np.int32)
        rows = np.floor(
            (samples[:, 1] - float(origin[1])) / resolution
        ).astype(np.int32)
        inside = (
            (rows >= 0)
            & (rows < values.shape[0])
            & (columns >= 0)
            & (columns < values.shape[1])
        )
        if not bool(np.all(inside)):
            return False
        sampled_values = values[rows, columns]
        if bool(np.any(sampled_values >= self.occupied_threshold)):
            return False
        if float(np.mean(sampled_values < 0)) > float(maximum_unknown_fraction):
            return False
        raw_free = (values < self.occupied_threshold).astype(np.uint8)
        clearance = cv2.distanceTransform(raw_free, cv2.DIST_L2, 5) * resolution
        sampled_clearance = clearance[rows, columns]
        start = float(sampled_clearance[0])
        margin = float(self.inflation_radius_m)
        tolerance = float(worsening_tolerance_m)
        if start >= margin:
            return bool(np.min(sampled_clearance) >= margin - tolerance)
        # Starting overlap is allowed only for genuine egress.  No sampled
        # pose may move materially deeper into the margin and the endpoint
        # must create measurable new clearance.
        if float(np.min(sampled_clearance)) < start - tolerance:
            return False
        if float(sampled_clearance[-1]) < start + float(
            minimum_clearance_improvement_m
        ):
            return False
        clear_indices = np.flatnonzero(sampled_clearance >= margin)
        if len(clear_indices) and float(
            np.min(sampled_clearance[int(clear_indices[0]) :])
        ) < margin - tolerance:
            return False
        return True

    def longest_margin_egress_prefix(
        self,
        path,
        values,
        resolution,
        origin,
        *,
        maximum_unknown_fraction=0.08,
        minimum_clearance_improvement_m=0.02,
        worsening_tolerance_m=0.03,
    ):
        """Return the longest current-map prefix that safely exits a margin."""

        if not 0.0 <= float(maximum_unknown_fraction) <= 1.0:
            raise ValueError("maximum unknown fraction must be between zero and one")
        points = np.asarray(path, dtype=float)
        if points.ndim != 2 or points.shape[1:] != (2,) or len(points) < 2:
            return ()
        for stop in range(len(points), 1, -1):
            prefix = points[:stop]
            if self._margin_egress_path_is_traversable(
                prefix,
                values,
                resolution,
                origin,
                maximum_unknown_fraction=maximum_unknown_fraction,
                minimum_clearance_improvement_m=(
                    minimum_clearance_improvement_m
                ),
                worsening_tolerance_m=worsening_tolerance_m,
            ):
                return tuple(
                    (float(point[0]), float(point[1])) for point in prefix
                )
        return ()

    def _segment_profile(
        self, first, second, values, inflated, resolution, origin
    ) -> Optional[Tuple[float, float]]:
        first = np.asarray(first, dtype=float)
        second = np.asarray(second, dtype=float)
        length = float(np.linalg.norm(second - first))
        if length <= 1.0e-6:
            return None
        samples = max(2, int(math.ceil(length / max(0.5 * resolution, 0.025))) + 1)
        ratios = np.linspace(0.0, 1.0, samples)
        points = first[None, :] + ratios[:, None] * (second - first)[None, :]
        columns = np.floor((points[:, 0] - float(origin[0])) / resolution).astype(
            np.int32
        )
        rows = np.floor((points[:, 1] - float(origin[1])) / resolution).astype(
            np.int32
        )
        inside = (
            (rows >= 0)
            & (rows < inflated.shape[0])
            & (columns >= 0)
            & (columns < inflated.shape[1])
        )
        interior = inside.copy()
        interior[0] = False
        interior[-1] = False
        # Permit endpoint quantisation, but not a segment that remains in the
        # inflated body halo after leaving either endpoint.  Keeping this
        # operation vectorised is important: graph construction evaluates
        # thousands of line-of-sight edges for each SLAM revision.
        if np.any(inflated[rows[interior], columns[interior]]):
            return None
        unknown = int(np.count_nonzero(~inside))
        if np.any(inside):
            unknown += int(
                np.count_nonzero(values[rows[inside], columns[inside]] < 0)
            )
        return length, float(unknown) / samples

    def _candidate_vertices(self, inflated, values, resolution, origin, start, goal):
        contours, _ = cv2.findContours(
            inflated.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )
        clearance_pixels = cv2.distanceTransform(
            (~inflated).astype(np.uint8), cv2.DIST_L2, 5
        )
        candidates = []
        epsilon = max(1.0, self.contour_simplification_m / resolution)
        offset = max(self.vertex_offset_m, 1.5 * resolution)
        directions = [
            (math.cos(angle), math.sin(angle))
            for angle in np.linspace(-math.pi, math.pi, 16, endpoint=False)
        ]
        for contour in contours:
            if len(contour) < 3:
                continue
            approximate = cv2.approxPolyDP(contour, epsilon, True)
            for vertex in approximate[:, 0, :]:
                base = self._grid_to_world(
                    int(vertex[1]), int(vertex[0]), resolution, origin
                )
                around = []
                for dx, dy in directions:
                    point = (base[0] + offset * dx, base[1] + offset * dy)
                    row, column = self._world_to_grid(point, resolution, origin)
                    if not self._inside(inflated.shape, row, column):
                        score = 0.0
                    elif inflated[row, column]:
                        continue
                    else:
                        score = float(clearance_pixels[row, column]) * resolution
                        score += 0.05 * float(values[row, column] >= 0)
                    around.append((score, point))
                around.sort(reverse=True, key=lambda item: item[0])
                chosen = []
                for score, point in around:
                    if score < 0.5 * resolution:
                        continue
                    if all(
                        math.hypot(point[0] - old[0], point[1] - old[1])
                        >= self.node_separation_m
                        for old in chosen
                    ):
                        chosen.append(point)
                        candidates.append((score, point))
                    if len(chosen) >= 2:
                        break

        # A relevance score keeps the quadratic visibility test bounded while
        # retaining corners likely to connect the current pose and goal.
        candidates.sort(
            key=lambda item: (
                math.hypot(item[1][0] - start[0], item[1][1] - start[1])
                + math.hypot(item[1][0] - goal[0], item[1][1] - goal[1])
                - 0.2 * item[0]
            )
        )
        output = []
        for _, point in candidates:
            if all(
                math.hypot(point[0] - old[0], point[1] - old[1])
                >= self.node_separation_m
                for old in output
            ):
                output.append(point)
            if len(output) >= self.maximum_nodes - 2:
                break
        return output

    def _search(
        self,
        nodes,
        edges,
        *,
        known_only,
        unknown_cost_weight,
        start_yaw=None,
        directed_failed_branches=(),
        failure_buffer_m=0.45,
    ):
        adjacency: Dict[int, List[Tuple[int, float]]] = {
            node.node_id: [] for node in nodes
        }
        for edge in edges:
            if known_only and not edge.known:
                continue
            cost = edge.length * (
                1.0 + unknown_cost_weight * edge.unknown_fraction
            )
            if start_yaw is not None and edge.first == 0:
                start = nodes[0]
                target = nodes[edge.second]
                bearing = math.atan2(target.y - start.y, target.x - start.x)
                error = abs(
                    math.atan2(
                        math.sin(bearing - float(start_yaw)),
                        math.cos(bearing - float(start_yaw)),
                    )
                )
                cost += self.start_heading_weight_m * (error / math.pi) ** 2
                rear_fraction = max(
                    0.0, (error - 0.5 * math.pi) / (0.5 * math.pi)
                )
                cost += self.reverse_start_penalty_m * rear_fraction ** 2
            first = nodes[edge.first]
            second = nodes[edge.second]
            first_xy = (first.x, first.y)
            second_xy = (second.x, second.y)
            if not self._transition_enters_failed_branch(
                first_xy,
                second_xy,
                directed_failed_branches,
                failure_buffer_m,
            ):
                adjacency[edge.first].append((edge.second, cost))
            if not self._transition_enters_failed_branch(
                second_xy,
                first_xy,
                directed_failed_branches,
                failure_buffer_m,
            ):
                adjacency[edge.second].append((edge.first, cost))
        distances = {0: 0.0}
        previous = {}
        queue = [(0.0, 0)]
        while queue:
            cost, node_id = heapq.heappop(queue)
            if cost > distances.get(node_id, math.inf):
                continue
            if node_id == 1:
                break
            for neighbour, edge_cost in adjacency.get(node_id, ()):
                candidate = cost + edge_cost
                if candidate + 1.0e-9 < distances.get(neighbour, math.inf):
                    distances[neighbour] = candidate
                    previous[neighbour] = node_id
                    heapq.heappush(queue, (candidate, neighbour))
        if 1 not in distances:
            return None, None
        ids = [1]
        while ids[-1] != 0:
            parent = previous.get(ids[-1])
            if parent is None:
                return None, None
            ids.append(parent)
        ids.reverse()
        return ids, float(distances[1])

    @staticmethod
    def _project_to_polyline(point, polyline):
        """Return distance and arclength of ``point`` on a polyline."""

        values = np.asarray(polyline, dtype=float)
        query = np.asarray(point, dtype=float)
        if (
            values.ndim != 2
            or values.shape[1:] != (2,)
            or len(values) < 2
            or query.shape != (2,)
        ):
            return math.inf, 0.0, 0.0
        cumulative = np.concatenate((
            [0.0], np.cumsum(np.linalg.norm(np.diff(values, axis=0), axis=1))
        ))
        best_distance = math.inf
        best_arclength = 0.0
        for index, (first, second) in enumerate(zip(values[:-1], values[1:])):
            delta = second - first
            denominator = float(np.dot(delta, delta))
            if denominator <= 1.0e-12:
                continue
            ratio = float(np.clip(np.dot(query - first, delta) / denominator, 0.0, 1.0))
            projection = first + ratio * delta
            distance = float(np.linalg.norm(query - projection))
            arclength = float(cumulative[index]) + ratio * math.sqrt(denominator)
            if distance < best_distance:
                best_distance = distance
                best_arclength = arclength
        return best_distance, best_arclength, float(cumulative[-1])

    @classmethod
    def _transition_enters_failed_branch(
        cls,
        first,
        second,
        directed_failed_branches,
        failure_buffer_m,
    ):
        """Reject ingress toward a failed terminal while preserving egress.

        Failed branches are semantic, directed constraints rather than
        occupied geometry.  Painting them into the occupancy mask isolated a
        vehicle already inside a dead end from its own visibility graph.  A
        transition whose projection advances from branch entry to terminal is
        forbidden; the same edge in the terminal-to-entry direction remains
        available for a certified escape.
        """

        buffer_m = max(1.0e-3, float(failure_buffer_m))
        first_xy = np.asarray(first, dtype=float)
        second_xy = np.asarray(second, dtype=float)
        for polyline in directed_failed_branches:
            values = np.asarray(polyline, dtype=float)
            if values.ndim != 2 or values.shape[1:] != (2,) or len(values) < 2:
                continue
            edge_length = float(np.linalg.norm(second_xy - first_xy))
            sample_count = max(3, int(math.ceil(edge_length / max(0.10, buffer_m))) + 1)
            samples = first_xy[None, :] + np.linspace(0.0, 1.0, sample_count)[:, None] * (
                second_xy - first_xy
            )[None, :]
            near = []
            total = 0.0
            for sample in samples:
                distance, arclength, total = cls._project_to_polyline(sample, values)
                if distance <= buffer_m:
                    near.append(arclength)
            if not near:
                continue
            first_distance, first_s, _ = cls._project_to_polyline(first_xy, values)
            second_distance, second_s, _ = cls._project_to_polyline(second_xy, values)
            start_s = first_s if first_distance <= 1.5 * buffer_m else near[0]
            end_s = second_s if second_distance <= 1.5 * buffer_m else near[-1]
            minimum_advance = min(0.12, max(0.04, 0.05 * total))
            if end_s > start_s + minimum_advance:
                return True
        return False

    def plan(
        self,
        values,
        resolution,
        origin,
        start_xy,
        goal_xy,
        *,
        blocked_polylines: Iterable[Sequence[Sequence[float]]] = (),
        directed_failed_branches: Iterable[Sequence[Sequence[float]]] = (),
        failure_buffer_m: float = 0.45,
        start_yaw: Optional[float] = None,
    ) -> VisibilityPlan:
        values = np.asarray(values, dtype=np.int16)
        if values.ndim != 2 or not values.size:
            raise ValueError("visibility planning requires a non-empty 2-D grid")
        resolution = float(resolution)
        if resolution <= 0.0:
            raise ValueError("visibility map resolution must be positive")
        origin = (float(origin[0]), float(origin[1]))
        start = (float(start_xy[0]), float(start_xy[1]))
        goal = (float(goal_xy[0]), float(goal_xy[1]))
        if not all(math.isfinite(value) for value in start + goal):
            raise ValueError("visibility start and goal must be finite")
        if start_yaw is not None and not math.isfinite(float(start_yaw)):
            raise ValueError("visibility start yaw must be finite")

        occupied = (values >= self.occupied_threshold).astype(np.uint8)
        failure_thickness = max(
            1, int(math.ceil(2.0 * float(failure_buffer_m) / resolution))
        )
        for polyline in blocked_polylines:
            points = []
            for point in polyline:
                row, column = self._world_to_grid(point, resolution, origin)
                points.append((column, row))
            if len(points) >= 2:
                cv2.polylines(
                    occupied,
                    [np.asarray(points, dtype=np.int32)],
                    False,
                    1,
                    thickness=failure_thickness,
                )
        inflated = self._inflated_occupancy(occupied * 100, resolution)
        anchors = self._candidate_vertices(
            inflated, values, resolution, origin, start, goal
        )
        nodes = [
            VisibilityNode(0, start[0], start[1], "START"),
            VisibilityNode(1, goal[0], goal[1], "GOAL"),
        ]
        nodes.extend(
            VisibilityNode(index + 2, point[0], point[1], "POLYGON_VERTEX")
            for index, point in enumerate(anchors)
        )

        edges = []
        for first_index, first in enumerate(nodes):
            for second in nodes[first_index + 1 :]:
                length = math.hypot(second.x - first.x, second.y - first.y)
                if (
                    length > self.maximum_edge_length_m
                    and first.node_id not in (0, 1)
                    and second.node_id not in (0, 1)
                ):
                    continue
                profile = self._segment_profile(
                    (first.x, first.y),
                    (second.x, second.y),
                    values,
                    inflated,
                    resolution,
                    origin,
                )
                if profile is None:
                    continue
                edge_length, unknown_fraction = profile
                edges.append(
                    VisibilityEdge(
                        first.node_id,
                        second.node_id,
                        edge_length,
                        unknown_fraction,
                    )
                )

        known_ids, known_cost = self._search(
            nodes,
            edges,
            known_only=True,
            unknown_cost_weight=self.unknown_cost_weight,
            start_yaw=start_yaw,
            directed_failed_branches=directed_failed_branches,
            failure_buffer_m=failure_buffer_m,
        )
        if known_ids is not None:
            ids, cost, mode = known_ids, known_cost, "KNOWN_VISIBILITY"
        else:
            ids, cost = self._search(
                nodes,
                edges,
                known_only=False,
                unknown_cost_weight=self.unknown_cost_weight,
                start_yaw=start_yaw,
                directed_failed_branches=directed_failed_branches,
                failure_buffer_m=failure_buffer_m,
            )
            mode = "ATTEMPTABLE_VISIBILITY"
        known_edges = sum(edge.known for edge in edges)
        if ids is None:
            return VisibilityPlan(
                status="NO_ROUTE",
                mode="NONE",
                path=(),
                nodes=tuple(nodes),
                edges=tuple(edges),
                path_cost=None,
                known_edges=int(known_edges),
                attemptable_edges=int(len(edges) - known_edges),
                reason="visibility_graph_disconnected",
                path_length=None,
                path_unknown_fraction=1.0,
            )
        by_id = {node.node_id: node for node in nodes}
        path = tuple((by_id[node_id].x, by_id[node_id].y) for node_id in ids)
        edge_by_pair = {
            tuple(sorted((edge.first, edge.second))): edge for edge in edges
        }
        path_edges = [
            edge_by_pair[tuple(sorted((first, second)))]
            for first, second in zip(ids[:-1], ids[1:])
        ]
        path_length = float(sum(edge.length for edge in path_edges))
        path_unknown_fraction = float(
            sum(edge.length * edge.unknown_fraction for edge in path_edges)
            / max(1.0e-9, path_length)
        )
        return VisibilityPlan(
            status="PASS",
            mode=mode,
            path=path,
            nodes=tuple(nodes),
            edges=tuple(edges),
            path_cost=cost,
            known_edges=int(known_edges),
            attemptable_edges=int(len(edges) - known_edges),
            reason="known_route" if mode == "KNOWN_VISIBILITY" else "attempt_unknown_route",
            path_length=path_length,
            path_unknown_fraction=path_unknown_fraction,
        )
