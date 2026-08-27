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
import time
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
    # Graph-construction evidence.  These fields deliberately describe only
    # the current online occupancy snapshot; they are never persisted as a
    # map-specific route cache.
    candidate_vertices_total: int = 0
    candidate_vertices_selected: int = 0
    node_limit_hit: bool = False
    planning_node_limit: int = 0
    start_degree: int = 0
    goal_degree: int = 0
    connected_components: int = 0
    start_component_size: int = 0
    goal_component_size: int = 0
    start_clearance_m: Optional[float] = None
    goal_clearance_m: Optional[float] = None
    start_inside_inflation: bool = False
    goal_inside_inflation: bool = False
    disconnect_class: str = "NONE"
    progressive_stages: int = 1
    progressive_complete: bool = True
    partial_goal_progress_m: Optional[float] = None
    partial_frontier_unknown_fraction: Optional[float] = None
    # FAR upstream keeps driven trajectory/inter-navigation vertices in its
    # dynamic global graph.  These counters make the corresponding online,
    # map-agnostic bridge vertices auditable in the 2-D port.
    trajectory_vertices_total: int = 0
    trajectory_vertices_selected: int = 0


@dataclass
class ProgressiveVisibilitySession:
    """Transient graph-construction state for one live pose/map request.

    The owner must discard this object when the goal, measured start pose,
    failed-branch set or online occupancy revision changes.  It deliberately
    carries no map UUID and is never serialized, so resuming a dense solve
    cannot become a cross-map route cache.
    """

    values: np.ndarray
    resolution: float
    origin: Tuple[float, float]
    start: Tuple[float, float]
    goal: Tuple[float, float]
    start_yaw: Optional[float]
    blocked_polylines: Tuple[Tuple[Tuple[float, float], ...], ...]
    directed_failed_branches: Tuple[Tuple[Tuple[float, float], ...], ...]
    trajectory_points: Tuple[Tuple[float, float], ...]
    failure_buffer_m: float
    occupied: np.ndarray
    inflated: np.ndarray
    all_nodes: Tuple[VisibilityNode, ...]
    candidate_vertices_total: int
    limits: Tuple[int, ...]
    start_clearance_m: Optional[float]
    goal_clearance_m: Optional[float]
    start_inside_inflation: bool
    goal_inside_inflation: bool
    edges: List[VisibilityEdge]
    trajectory_vertices_total: int = 0
    trajectory_vertices_selected: int = 0
    previous_node_count: int = 1
    next_stage_index: int = 0
    complete: bool = False
    last_plan: Optional[VisibilityPlan] = None


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
                plan.mode in (
                    "ATTEMPTABLE_VISIBILITY",
                    "PARTIAL_ATTEMPTABLE",
                )
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


def goal_connected_incumbent_retention_reason(
    *,
    previous_route_motion_authorized: bool,
    same_goal: bool,
    previous_mode: str,
    previous_route_globally_traversable: bool,
    candidate_mode: str,
    candidate_motion_authorized: bool,
    handoff_accepted: bool,
):
    """Keep a valid goal route ahead of weaker rolling rebuild results.

    Upstream FAR plans over a persistent graph and carries path momentum while
    that path remains valid.  A snapshot rebuild in this adapter is only a
    *candidate* replacement.  In particular, a bounded frontier prefix must
    never erase a still-traversable route which already reaches the goal.

    The complete incumbent geometry is checked on the newest inflated map.
    A newly observed wall therefore revokes the hold and permits a genuine
    reroute or turnaround; this cannot preserve a known-collision route.
    """

    if not (
        previous_route_motion_authorized
        and same_goal
        and str(previous_mode)
        in ("KNOWN_VISIBILITY", "ATTEMPTABLE_VISIBILITY")
        and previous_route_globally_traversable
    ):
        return None
    if str(candidate_mode) == "PARTIAL_ATTEMPTABLE":
        return "partial_candidate_cannot_preempt_goal_route"
    if not candidate_motion_authorized:
        return "unsettled_candidate_cannot_preempt_goal_route"
    if not handoff_accepted:
        return "discontinuous_candidate_cannot_preempt_goal_route"
    return None


def visibility_plan_is_goal_connected(plan):
    """Return whether a FAR result reaches the current mission goal.

    ``PARTIAL_ATTEMPTABLE`` deliberately does not qualify: its last vertex is
    only a frontier inside START's connected component.  Keeping this
    distinction in one helper prevents a short frontier from being promoted to
    the same authority class as a complete maze detour.
    """

    return bool(
        plan is not None
        and plan.status == "PASS"
        and plan.mode in ("KNOWN_VISIBILITY", "ATTEMPTABLE_VISIBILITY")
        and len(plan.path) >= 2
    )


def goal_route_direction_continuity_hold(
    *, previous_plan, previous_route_safe, lease_prefix_clear, handoff_accepted
):
    """Protect direction continuity only for an incumbent goal route.

    A short PARTIAL frontier is expendable.  Letting its initial tangent veto a
    newly discovered complete detour recreates the exact local-minimum failure
    that dense FAR expansion is intended to resolve.
    """

    return bool(
        visibility_plan_is_goal_connected(previous_plan)
        and (bool(previous_route_safe) or bool(lease_prefix_clear))
        and not bool(handoff_accepted)
    )


def partial_frontier_authority_reason(
    plan,
    *,
    path_clear,
    connected_candidate_available=False,
    explicit_egress=False,
    minimum_goal_progress_m=0.05,
    minimum_information_gain=0.05,
    maximum_information_detour_m=0.50,
):
    """Classify whether a bounded partial route may move the vehicle.

    A partial visibility route is useful while a connected route is not yet
    available, but it must not become a local goal-seeking loop in a known
    cul-de-sac.  Normal partial motion therefore needs either measurable goal
    progress or information gain with tightly bounded regression.  Moving
    away from the goal remains possible only inside an explicit dead-end
    egress transaction.

    The return value is a stable diagnostic reason; ``None`` means no motion
    authority.  Physical hard-veto remains an independent final requirement.
    """

    if not (
        plan is not None
        and plan.status == "PASS"
        and plan.mode == "PARTIAL_ATTEMPTABLE"
        and len(plan.path) >= 2
        and bool(path_clear)
        and float(plan.path_unknown_fraction) <= 0.02
    ):
        return None
    progress = plan.partial_goal_progress_m
    information = plan.partial_frontier_unknown_fraction
    progress = -math.inf if progress is None else float(progress)
    information = 0.0 if information is None else float(information)
    if explicit_egress:
        return "explicit_egress_partial_frontier"
    if connected_candidate_available:
        return None
    if progress >= float(minimum_goal_progress_m):
        return "positive_progress_partial_frontier"
    if (
        information >= float(minimum_information_gain)
        and progress >= -float(maximum_information_detour_m)
    ):
        return "information_gain_partial_frontier"
    return None


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
        self.last_candidate_signature = None
        self.reason = "waiting_for_visibility_route"

    @staticmethod
    def _wrap(angle):
        return math.atan2(math.sin(float(angle)), math.cos(float(angle)))

    def _first_bearing(self, path):
        return route_initial_bearing(path, self.lookahead_m)

    @staticmethod
    def _candidate_signature(plan):
        """Identify an exact candidate without any map/scenario identity."""

        return (
            str(plan.mode),
            tuple(
                (round(float(point[0]), 3), round(float(point[1]), 3))
                for point in plan.path
            ),
            round(float(plan.path_cost or 0.0), 3),
            round(float(plan.path_unknown_fraction), 4),
        )

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
        partial_route = plan.mode == "PARTIAL_ATTEMPTABLE"
        candidate_signature = self._candidate_signature(plan)
        observer = None
        if observer_position is not None:
            candidate = np.asarray(observer_position, dtype=float)
            if candidate.shape == (2,) and np.all(np.isfinite(candidate)):
                observer = candidate
        if partial_route and self.accepted:
            # A bounded frontier prefix cannot inherit full-goal acceptance
            # from an older connected route.
            self.reset()
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
                self.last_candidate_signature = candidate_signature
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
        # Only an exact duplicate at the same measured pose is not new route
        # evidence.  A dense solve can upgrade a PARTIAL frontier to a complete
        # KNOWN route on the same immutable map revision; treating those two
        # candidates as identical was the P6 V4.3 long-detour handoff bug.
        if (
            self.last_revision == int(map_revision)
            and not observer_advanced
            and self.last_candidate_signature == candidate_signature
        ):
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
        self.last_candidate_signature = candidate_signature
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
            not partial_route
            and sufficiently_observed
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
        elif partial_route:
            self.reason = "partial_reachable_frontier"
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

    def _candidate_vertices(
        self,
        inflated,
        values,
        resolution,
        origin,
        start,
        goal,
        *,
        maximum_nodes=None,
        coverage_order=False,
    ):
        """Return an ordered vertex pool and its uncropped unique size.

        The ordinary graph retains the historical start/goal relevance order.
        A dense recovery solve may additionally interleave spatial and contour
        representatives.  This prevents a long but valid detour from losing all
        of its remote corner vertices merely because a well explored map has
        many locally relevant contours.
        """

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
        for contour_index, contour in enumerate(contours):
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
                        candidates.append((score, point, contour_index))
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
        unique = []
        for score, point, contour_index in candidates:
            if all(
                math.hypot(point[0] - old[1][0], point[1] - old[1][1])
                >= self.node_separation_m
                for old in unique
            ):
                unique.append((score, point, contour_index))

        ordered = unique
        if coverage_order and unique:
            # Interleave four relevance-ranked vertices with one spatial-bin
            # representative and one contour representative.  All queues are
            # deterministic and every selected item still passed the same
            # inflated-map clearance test above.
            bin_size = max(1.0, 4.0 * self.node_separation_m)
            spatial = []
            seen_bins = set()
            contour = []
            seen_contours = set()
            for item in unique:
                point = item[1]
                key = (
                    int(math.floor(point[0] / bin_size)),
                    int(math.floor(point[1] / bin_size)),
                )
                if key not in seen_bins:
                    seen_bins.add(key)
                    spatial.append(item)
                if item[2] not in seen_contours:
                    seen_contours.add(item[2])
                    contour.append(item)
            queues = (unique, spatial, contour)
            pattern = (0, 0, 0, 0, 1, 2)
            indices = [0, 0, 0]
            used = set()
            ordered = []
            while len(ordered) < len(unique):
                advanced = False
                for queue_index in pattern:
                    queue = queues[queue_index]
                    while indices[queue_index] < len(queue):
                        item = queue[indices[queue_index]]
                        indices[queue_index] += 1
                        key = (round(item[1][0], 6), round(item[1][1], 6))
                        if key in used:
                            continue
                        used.add(key)
                        ordered.append(item)
                        advanced = True
                        break
                if not advanced:
                    break

        node_limit = self.maximum_nodes if maximum_nodes is None else int(maximum_nodes)
        selected = max(0, node_limit - 2)
        return [item[1] for item in ordered[:selected]], len(unique)

    def _trajectory_vertices(
        self,
        trajectory_points,
        inflated,
        values,
        resolution,
        origin,
        start,
        goal,
        *,
        maximum_vertices,
    ):
        """Return safe persistent inter-navigation anchors for this snapshot.

        Upstream FAR does not throw away the complete graph at every sensor
        update.  In particular, positions actually traversed by the robot are
        retained as trajectory/inter-nav vertices and can bridge two contour
        subgraphs after a newly observed wall invalidates a speculative edge.

        The 2-D port stores those positions in odometry and the ROS owner
        transforms them with the *current* online ``map<-odom`` transform.
        This method never retrieves a map-specific route: it merely admits
        current-map-free points as visibility vertices and rechecks every edge
        against the latest inflated occupancy mask.
        """

        maximum = max(0, int(maximum_vertices))
        if maximum == 0:
            return [], 0
        valid = []
        separation = max(1.0e-6, self.node_separation_m)
        spatial_bins = {}
        for point in trajectory_points:
            try:
                candidate = (float(point[0]), float(point[1]))
            except (TypeError, ValueError, IndexError):
                continue
            if not all(math.isfinite(value) for value in candidate):
                continue
            row, column = self._world_to_grid(candidate, resolution, origin)
            if not self._inside(values.shape, row, column):
                continue
            # A historical traversal is connectivity evidence, never an
            # override for newly observed occupied space or footprint margin.
            if values[row, column] < 0 or inflated[row, column]:
                continue
            if min(
                math.hypot(candidate[0] - endpoint[0], candidate[1] - endpoint[1])
                for endpoint in (start, goal)
            ) < 0.5 * self.node_separation_m:
                continue
            bin_key = (
                int(math.floor(candidate[0] / separation)),
                int(math.floor(candidate[1] / separation)),
            )
            neighbours = [
                old
                for dx in (-1, 0, 1)
                for dy in (-1, 0, 1)
                for old in spatial_bins.get(
                    (bin_key[0] + dx, bin_key[1] + dy), ()
                )
            ]
            if any(
                math.hypot(candidate[0] - old[0], candidate[1] - old[1])
                < separation
                for old in neighbours
            ):
                continue
            valid.append(candidate)
            spatial_bins.setdefault(bin_key, []).append(candidate)

        total = len(valid)
        if total <= maximum:
            return valid, total

        # Preserve spatial coverage across the entire online trajectory rather
        # than keeping only its newest or goal-nearest end.  This mirrors the
        # role of FAR's inter-nav chain without forcing any stale edge to stay
        # connected; visibility and hard safety are still recomputed below.
        indices = np.linspace(0, total - 1, maximum, dtype=np.int32)
        selected = []
        used = set()
        for index in indices:
            index = int(index)
            if index in used:
                continue
            used.add(index)
            selected.append(valid[index])
        return selected, total

    def _build_snapshot_nodes(
        self,
        inflated,
        values,
        resolution,
        origin,
        start,
        goal,
        *,
        node_limit,
        coverage_order,
        trajectory_points=(),
        maximum_trajectory_vertices=64,
    ):
        """Build one graph snapshot while reserving room for driven anchors."""

        bridge_budget = min(
            max(0, int(maximum_trajectory_vertices)),
            max(0, int(node_limit) - 4),
        )
        trajectory, trajectory_total = self._trajectory_vertices(
            trajectory_points,
            inflated,
            values,
            resolution,
            origin,
            start,
            goal,
            maximum_vertices=bridge_budget,
        )
        polygon_limit = max(4, int(node_limit) - len(trajectory))
        anchors, polygon_total = self._candidate_vertices(
            inflated,
            values,
            resolution,
            origin,
            start,
            goal,
            maximum_nodes=polygon_limit,
            coverage_order=bool(coverage_order),
        )
        # A driven anchor takes precedence over a nearby contour vertex.  Both
        # would generate nearly identical quadratic edge work, while the
        # trajectory vertex has stronger real-traversal evidence.
        filtered_anchors = [
            point
            for point in anchors
            if all(
                math.hypot(point[0] - bridge[0], point[1] - bridge[1])
                >= self.node_separation_m
                for bridge in trajectory
            )
        ]
        nodes = [
            VisibilityNode(0, start[0], start[1], "START"),
            VisibilityNode(1, goal[0], goal[1], "GOAL"),
        ]
        nodes.extend(
            VisibilityNode(index + 2, point[0], point[1], "TRAJECTORY")
            for index, point in enumerate(trajectory)
        )
        offset = len(nodes)
        nodes.extend(
            VisibilityNode(offset + index, point[0], point[1], "POLYGON_VERTEX")
            for index, point in enumerate(filtered_anchors)
        )
        return (
            nodes,
            int(polygon_total + trajectory_total),
            int(trajectory_total),
            int(len(trajectory)),
        )

    @staticmethod
    def _component_diagnostics(nodes, edges):
        adjacency = {node.node_id: set() for node in nodes}
        for edge in edges:
            adjacency[edge.first].add(edge.second)
            adjacency[edge.second].add(edge.first)
        components = []
        component_by_node = {}
        for node in nodes:
            if node.node_id in component_by_node:
                continue
            pending = [node.node_id]
            component = set()
            while pending:
                current = pending.pop()
                if current in component:
                    continue
                component.add(current)
                pending.extend(adjacency[current] - component)
            index = len(components)
            for node_id in component:
                component_by_node[node_id] = index
            components.append(component)
        start_component = components[component_by_node[0]]
        goal_component = components[component_by_node[1]]
        return {
            "start_degree": len(adjacency[0]),
            "goal_degree": len(adjacency[1]),
            "connected_components": len(components),
            "start_component_size": len(start_component),
            "goal_component_size": len(goal_component),
        }

    def _endpoint_clearance(self, point, occupied, inflated, resolution, origin):
        row, column = self._world_to_grid(point, resolution, origin)
        if not self._inside(occupied.shape, row, column):
            return None, False
        raw_free = (~occupied.astype(bool)).astype(np.uint8)
        clearance = cv2.distanceTransform(raw_free, cv2.DIST_L2, 5)
        return float(clearance[row, column]) * resolution, bool(inflated[row, column])

    def _shortest_path_tree(
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
            for neighbour, edge_cost in adjacency.get(node_id, ()):
                candidate = cost + edge_cost
                if candidate + 1.0e-9 < distances.get(neighbour, math.inf):
                    distances[neighbour] = candidate
                    previous[neighbour] = node_id
                    heapq.heappush(queue, (candidate, neighbour))
        return distances, previous

    @staticmethod
    def _reconstruct_path(previous, target_id):
        ids = [int(target_id)]
        while ids[-1] != 0:
            parent = previous.get(ids[-1])
            if parent is None:
                return None
            ids.append(parent)
        ids.reverse()
        return ids

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
        distances, previous = self._shortest_path_tree(
            nodes,
            edges,
            known_only=known_only,
            unknown_cost_weight=unknown_cost_weight,
            start_yaw=start_yaw,
            directed_failed_branches=directed_failed_branches,
            failure_buffer_m=failure_buffer_m,
        )
        if 1 not in distances:
            return None, None
        ids = self._reconstruct_path(previous, 1)
        if ids is None:
            return None, None
        return ids, float(distances[1])

    def _partial_attemptable_path(
        self,
        nodes,
        edges,
        values,
        inflated,
        resolution,
        origin,
        goal,
        *,
        start_yaw=None,
        directed_failed_branches=(),
        failure_buffer_m=0.45,
        maximum_path_m=3.0,
    ):
        """Choose a safe, bounded route inside START's reachable component.

        This is not an arbitrary Euclidean fallback.  Every returned segment
        belongs to the current visibility graph, avoids directed failed
        branches, and its complete bounded prefix is observed and clear on the
        inflated occupancy grid.  Goal progress is preferred, while a modest
        information-gain term permits the first leg of a necessary detour.
        """

        distances, previous = self._shortest_path_tree(
            nodes,
            edges,
            known_only=False,
            unknown_cost_weight=self.unknown_cost_weight,
            start_yaw=start_yaw,
            directed_failed_branches=directed_failed_branches,
            failure_buffer_m=failure_buffer_m,
        )
        if len(distances) <= 1:
            return None
        by_id = {node.node_id: node for node in nodes}
        start = np.asarray((nodes[0].x, nodes[0].y), dtype=float)
        goal_xy = np.asarray(goal, dtype=float)
        initial_goal_distance = float(np.linalg.norm(goal_xy - start))
        raw_free = (np.asarray(values) < self.occupied_threshold).astype(np.uint8)
        clearance = cv2.distanceTransform(raw_free, cv2.DIST_L2, 5)
        information_radius = max(2, int(math.ceil(0.75 / float(resolution))))
        best = None
        for node_id, route_cost in distances.items():
            if node_id in (0, 1):
                continue
            ids = self._reconstruct_path(previous, node_id)
            if ids is None or len(ids) < 2:
                continue
            path = tuple((by_id[item].x, by_id[item].y) for item in ids)
            full_length = sum(
                math.hypot(second[0] - first[0], second[1] - first[1])
                for first, second in zip(path[:-1], path[1:])
            )
            if full_length < 0.45:
                continue
            bounded_path = polyline_prefix(path, min(maximum_path_m, full_length))
            if len(bounded_path) < 2 or not self.path_is_traversable(
                bounded_path,
                values,
                resolution,
                origin,
                maximum_unknown_fraction=0.02,
            ):
                continue
            endpoint = np.asarray(bounded_path[-1], dtype=float)
            endpoint_goal_distance = float(np.linalg.norm(goal_xy - endpoint))
            progress = initial_goal_distance - endpoint_goal_distance
            row, column = self._world_to_grid(endpoint, resolution, origin)
            if not self._inside(values.shape, row, column):
                continue
            row0 = max(0, row - information_radius)
            row1 = min(values.shape[0], row + information_radius + 1)
            column0 = max(0, column - information_radius)
            column1 = min(values.shape[1], column + information_radius + 1)
            neighbourhood = values[row0:row1, column0:column1]
            unknown_fraction = float(np.mean(neighbourhood < 0))
            clearance_m = float(clearance[row, column]) * float(resolution)
            # The graph may need to move briefly away from the goal to reach
            # the visible side of an occluding wall.  Bound that regression,
            # then use frontier gain and clearance to break ties rather than
            # falling back to a raw goal-bearing vector.
            if progress < -2.0 and unknown_fraction < 0.05:
                continue
            score = (
                2.0 * progress
                + 1.25 * unknown_fraction
                + 0.30 * min(clearance_m, 1.0)
                + 0.08 * min(full_length, maximum_path_m)
                - 0.04 * float(route_cost)
            )
            candidate = (
                score,
                progress,
                unknown_fraction,
                bounded_path,
                float(route_cost),
            )
            if best is None or candidate[0] > best[0]:
                best = candidate
        return best

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
        trajectory_points: Iterable[Sequence[float]] = (),
        maximum_trajectory_vertices: int = 64,
        failure_buffer_m: float = 0.45,
        start_yaw: Optional[float] = None,
        _maximum_nodes: Optional[int] = None,
        _coverage_order: bool = False,
        _progressive_stages: int = 1,
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
        node_limit = (
            self.maximum_nodes
            if _maximum_nodes is None
            else max(4, int(_maximum_nodes))
        )
        (
            nodes,
            candidate_vertices_total,
            trajectory_vertices_total,
            trajectory_vertices_selected,
        ) = self._build_snapshot_nodes(
            inflated,
            values,
            resolution,
            origin,
            start,
            goal,
            node_limit=node_limit,
            coverage_order=bool(_coverage_order),
            trajectory_points=trajectory_points,
            maximum_trajectory_vertices=maximum_trajectory_vertices,
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
        graph_diagnostics = self._component_diagnostics(nodes, edges)
        start_clearance, start_inside_inflation = self._endpoint_clearance(
            start, occupied, inflated, resolution, origin
        )
        goal_clearance, goal_inside_inflation = self._endpoint_clearance(
            goal, occupied, inflated, resolution, origin
        )
        selected_vertices = max(0, len(nodes) - 2)
        node_limit_hit = candidate_vertices_total > selected_vertices
        common_diagnostics = dict(
            candidate_vertices_total=int(candidate_vertices_total),
            candidate_vertices_selected=int(selected_vertices),
            node_limit_hit=bool(node_limit_hit),
            planning_node_limit=int(node_limit),
            start_degree=int(graph_diagnostics["start_degree"]),
            goal_degree=int(graph_diagnostics["goal_degree"]),
            connected_components=int(graph_diagnostics["connected_components"]),
            start_component_size=int(graph_diagnostics["start_component_size"]),
            goal_component_size=int(graph_diagnostics["goal_component_size"]),
            start_clearance_m=start_clearance,
            goal_clearance_m=goal_clearance,
            start_inside_inflation=bool(start_inside_inflation),
            goal_inside_inflation=bool(goal_inside_inflation),
            progressive_stages=max(1, int(_progressive_stages)),
            trajectory_vertices_total=int(trajectory_vertices_total),
            trajectory_vertices_selected=int(trajectory_vertices_selected),
        )
        if ids is None:
            if start_inside_inflation:
                disconnect_class = "START_IN_INFLATION"
            elif goal_inside_inflation:
                disconnect_class = "GOAL_IN_INFLATION"
            elif node_limit_hit:
                disconnect_class = "NODE_LIMIT_COMPONENT_DISCONNECTED"
            elif graph_diagnostics["start_degree"] == 0:
                disconnect_class = "START_ISOLATED"
            elif graph_diagnostics["goal_degree"] == 0:
                disconnect_class = "GOAL_ISOLATED"
            else:
                disconnect_class = "COMPONENT_DISCONNECTED"
            partial = None
            if not start_inside_inflation and not goal_inside_inflation:
                partial = self._partial_attemptable_path(
                    nodes,
                    edges,
                    values,
                    inflated,
                    resolution,
                    origin,
                    goal,
                    start_yaw=start_yaw,
                    directed_failed_branches=directed_failed_branches,
                    failure_buffer_m=failure_buffer_m,
                )
                # An uncapped graph over a fully observed closed obstacle has
                # no attemptable frontier.  Do not turn ordinary goal progress
                # toward that wall into an endless pseudo route.  A node-capped
                # graph may still expose a known prefix while its background
                # expansion searches the remaining current-map vertices.
                if (
                    partial is not None
                    and not node_limit_hit
                    and float(partial[2]) < 0.01
                ):
                    partial = None
            if partial is not None:
                _, progress, frontier_unknown, path, partial_cost = partial
                path_length = float(sum(
                    math.hypot(
                        second[0] - first[0], second[1] - first[1]
                    )
                    for first, second in zip(path[:-1], path[1:])
                ))
                path_profiles = [
                    self._segment_profile(
                        first,
                        second,
                        values,
                        inflated,
                        resolution,
                        origin,
                    )
                    for first, second in zip(path[:-1], path[1:])
                ]
                path_unknown_fraction = float(
                    sum(
                        profile[0] * profile[1]
                        for profile in path_profiles
                        if profile is not None
                    )
                    / max(1.0e-9, path_length)
                )
                return VisibilityPlan(
                    status="PASS",
                    mode="PARTIAL_ATTEMPTABLE",
                    path=tuple(path),
                    nodes=tuple(nodes),
                    edges=tuple(edges),
                    path_cost=float(partial_cost),
                    known_edges=int(known_edges),
                    attemptable_edges=int(len(edges) - known_edges),
                    reason="partial_reachable_frontier",
                    path_length=path_length,
                    path_unknown_fraction=path_unknown_fraction,
                    disconnect_class=disconnect_class,
                    partial_goal_progress_m=float(progress),
                    partial_frontier_unknown_fraction=float(frontier_unknown),
                    **common_diagnostics,
                )
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
                disconnect_class=disconnect_class,
                **common_diagnostics,
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
            disconnect_class="NONE",
            **common_diagnostics,
        )

    def _plan_progressive_legacy(
        self,
        values,
        resolution,
        origin,
        start_xy,
        goal_xy,
        *,
        initial_maximum_nodes=192,
        maximum_nodes=320,
        node_step=32,
        time_budget_s=2.5,
        blocked_polylines: Iterable[Sequence[Sequence[float]]] = (),
        directed_failed_branches: Iterable[Sequence[Sequence[float]]] = (),
        failure_buffer_m: float = 0.45,
        start_yaw: Optional[float] = None,
    ) -> VisibilityPlan:
        """Run a bounded coverage-preserving dense retry.

        This method is intended for a background worker after the ordinary
        relevance-ranked graph reports both ``NO_ROUTE`` and ``node_limit_hit``.
        It never stores a map identifier or a route for a future run.  Stages
        expand only the current occupancy snapshot and stop as soon as one
        connected route is found or the wall-clock budget is exhausted.
        """

        values = np.asarray(values, dtype=np.int16)
        resolution = float(resolution)
        origin = (float(origin[0]), float(origin[1]))
        start = (float(start_xy[0]), float(start_xy[1]))
        goal = (float(goal_xy[0]), float(goal_xy[1]))
        if values.ndim != 2 or not values.size or resolution <= 0.0:
            raise ValueError("progressive visibility planning needs a valid grid")
        if not all(math.isfinite(value) for value in start + goal):
            raise ValueError("progressive visibility endpoints must be finite")
        initial = max(4, int(initial_maximum_nodes))
        maximum = max(initial, int(maximum_nodes))
        step = max(1, int(node_step))
        budget = max(0.05, float(time_budget_s))
        started = time.perf_counter()

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
        anchors, candidate_vertices_total = self._candidate_vertices(
            inflated,
            values,
            resolution,
            origin,
            start,
            goal,
            maximum_nodes=maximum,
            coverage_order=True,
        )
        all_nodes = [
            VisibilityNode(0, start[0], start[1], "START"),
            VisibilityNode(1, goal[0], goal[1], "GOAL"),
        ]
        all_nodes.extend(
            VisibilityNode(index + 2, point[0], point[1], "POLYGON_VERTEX")
            for index, point in enumerate(anchors)
        )
        available_nodes = len(all_nodes)
        limits = list(range(initial, maximum + 1, step))
        if not limits or limits[-1] != maximum:
            limits.append(maximum)
        limits = sorted({min(available_nodes, limit) for limit in limits})
        if available_nodes not in limits:
            limits.append(available_nodes)
            limits.sort()

        start_clearance, start_inside_inflation = self._endpoint_clearance(
            start, occupied, inflated, resolution, origin
        )
        goal_clearance, goal_inside_inflation = self._endpoint_clearance(
            goal, occupied, inflated, resolution, origin
        )
        edges = []
        previous_node_count = 1
        result = None
        for stages, node_count in enumerate(limits, start=1):
            # Edges whose second endpoint was already in the previous stage are
            # retained.  Only pairs touching a newly admitted vertex are
            # profiled, avoiding three complete O(N^2) graph rebuilds.
            for second_index in range(max(1, previous_node_count), node_count):
                second = all_nodes[second_index]
                for first_index in range(second_index):
                    first = all_nodes[first_index]
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
            previous_node_count = node_count
            nodes = all_nodes[:node_count]
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
            graph = self._component_diagnostics(nodes, edges)
            selected = max(0, node_count - 2)
            node_limit_hit = candidate_vertices_total > selected
            common = dict(
                candidate_vertices_total=int(candidate_vertices_total),
                candidate_vertices_selected=int(selected),
                node_limit_hit=bool(node_limit_hit),
                planning_node_limit=int(node_count),
                start_degree=int(graph["start_degree"]),
                goal_degree=int(graph["goal_degree"]),
                connected_components=int(graph["connected_components"]),
                start_component_size=int(graph["start_component_size"]),
                goal_component_size=int(graph["goal_component_size"]),
                start_clearance_m=start_clearance,
                goal_clearance_m=goal_clearance,
                start_inside_inflation=bool(start_inside_inflation),
                goal_inside_inflation=bool(goal_inside_inflation),
                progressive_stages=int(stages),
            )
            if ids is None:
                if start_inside_inflation:
                    disconnect_class = "START_IN_INFLATION"
                elif goal_inside_inflation:
                    disconnect_class = "GOAL_IN_INFLATION"
                elif node_limit_hit:
                    disconnect_class = "NODE_LIMIT_COMPONENT_DISCONNECTED"
                elif graph["start_degree"] == 0:
                    disconnect_class = "START_ISOLATED"
                elif graph["goal_degree"] == 0:
                    disconnect_class = "GOAL_ISOLATED"
                else:
                    disconnect_class = "COMPONENT_DISCONNECTED"
                result = VisibilityPlan(
                    status="NO_ROUTE",
                    mode="NONE",
                    path=(),
                    nodes=tuple(nodes),
                    edges=tuple(edges),
                    path_cost=None,
                    known_edges=int(known_edges),
                    attemptable_edges=int(len(edges) - known_edges),
                    reason="visibility_graph_disconnected",
                    disconnect_class=disconnect_class,
                    **common,
                )
            else:
                by_id = {node.node_id: node for node in nodes}
                path = tuple(
                    (by_id[node_id].x, by_id[node_id].y) for node_id in ids
                )
                edge_by_pair = {
                    tuple(sorted((edge.first, edge.second))): edge
                    for edge in edges
                }
                path_edges = [
                    edge_by_pair[tuple(sorted((first, second)))]
                    for first, second in zip(ids[:-1], ids[1:])
                ]
                path_length = float(sum(edge.length for edge in path_edges))
                unknown_fraction = float(
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
                    reason=(
                        "known_route"
                        if mode == "KNOWN_VISIBILITY"
                        else "attempt_unknown_route"
                    ),
                    path_length=path_length,
                    path_unknown_fraction=unknown_fraction,
                    disconnect_class="NONE",
                    **common,
                )
            if not node_limit_hit or time.perf_counter() - started >= budget:
                return result
        return result

    def plan_progressive(
        self,
        values,
        resolution,
        origin,
        start_xy,
        goal_xy,
        *,
        initial_maximum_nodes=192,
        maximum_nodes=320,
        node_step=32,
        time_budget_s=2.5,
        blocked_polylines: Iterable[Sequence[Sequence[float]]] = (),
        directed_failed_branches: Iterable[Sequence[Sequence[float]]] = (),
        trajectory_points: Iterable[Sequence[float]] = (),
        maximum_trajectory_vertices: int = 64,
        failure_buffer_m: float = 0.45,
        start_yaw: Optional[float] = None,
        session: Optional[ProgressiveVisibilitySession] = None,
        return_session: bool = False,
    ):
        """Run or resume a dense solve for one transient online request.

        A completed node stage, its vertices and its visibility edges survive
        the wall-clock yield.  The next invocation therefore advances to the
        next node limit instead of repeating the expensive first stage.
        """

        values = np.asarray(values, dtype=np.int16)
        resolution = float(resolution)
        origin = (float(origin[0]), float(origin[1]))
        start = (float(start_xy[0]), float(start_xy[1]))
        goal = (float(goal_xy[0]), float(goal_xy[1]))
        if values.ndim != 2 or not values.size or resolution <= 0.0:
            raise ValueError("progressive visibility planning needs a valid grid")
        if not all(math.isfinite(value) for value in start + goal):
            raise ValueError("progressive visibility endpoints must be finite")
        initial = max(4, int(initial_maximum_nodes))
        maximum = max(initial, int(maximum_nodes))
        step = max(1, int(node_step))
        budget = max(0.05, float(time_budget_s))
        started = time.perf_counter()
        branch_signature = tuple(
            tuple((float(point[0]), float(point[1])) for point in branch)
            for branch in directed_failed_branches
        )
        blocked_signature = tuple(
            tuple((float(point[0]), float(point[1])) for point in polyline)
            for polyline in blocked_polylines
        )
        trajectory_signature = tuple(
            (float(point[0]), float(point[1])) for point in trajectory_points
        )
        yaw_matches = bool(
            session is None
            or (session.start_yaw is None and start_yaw is None)
            or (
                session.start_yaw is not None
                and start_yaw is not None
                and abs(
                    math.atan2(
                        math.sin(float(session.start_yaw) - float(start_yaw)),
                        math.cos(float(session.start_yaw) - float(start_yaw)),
                    )
                ) <= math.radians(3.0)
            )
        )
        session_compatible = bool(
            session is not None
            and session.values.shape == values.shape
            and session.resolution == resolution
            and session.origin == origin
            and math.hypot(
                session.start[0] - start[0], session.start[1] - start[1]
            ) <= 0.05
            and session.goal == goal
            and yaw_matches
            and session.blocked_polylines == blocked_signature
            and session.directed_failed_branches == branch_signature
            and session.trajectory_points == trajectory_signature
            and session.failure_buffer_m == float(failure_buffer_m)
            and np.array_equal(session.values, values)
        )
        if not session_compatible:
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
            (
                all_nodes,
                candidate_vertices_total,
                trajectory_vertices_total,
                trajectory_vertices_selected,
            ) = self._build_snapshot_nodes(
                inflated,
                values,
                resolution,
                origin,
                start,
                goal,
                node_limit=maximum,
                coverage_order=True,
                trajectory_points=trajectory_signature,
                maximum_trajectory_vertices=maximum_trajectory_vertices,
            )
            available_nodes = len(all_nodes)
            limits = list(range(initial, maximum + 1, step))
            if not limits or limits[-1] != maximum:
                limits.append(maximum)
            limits = sorted({min(available_nodes, limit) for limit in limits})
            if available_nodes not in limits:
                limits.append(available_nodes)
                limits.sort()
            start_clearance, start_inside_inflation = self._endpoint_clearance(
                start, occupied, inflated, resolution, origin
            )
            goal_clearance, goal_inside_inflation = self._endpoint_clearance(
                goal, occupied, inflated, resolution, origin
            )
            session = ProgressiveVisibilitySession(
                values=values.copy(),
                resolution=resolution,
                origin=origin,
                start=start,
                goal=goal,
                start_yaw=None if start_yaw is None else float(start_yaw),
                blocked_polylines=blocked_signature,
                directed_failed_branches=branch_signature,
                trajectory_points=trajectory_signature,
                failure_buffer_m=float(failure_buffer_m),
                occupied=occupied,
                inflated=inflated,
                all_nodes=tuple(all_nodes),
                candidate_vertices_total=int(candidate_vertices_total),
                limits=tuple(limits),
                start_clearance_m=start_clearance,
                goal_clearance_m=goal_clearance,
                start_inside_inflation=bool(start_inside_inflation),
                goal_inside_inflation=bool(goal_inside_inflation),
                edges=[],
                trajectory_vertices_total=int(trajectory_vertices_total),
                trajectory_vertices_selected=int(
                    trajectory_vertices_selected
                ),
            )
        if session.complete and session.last_plan is not None:
            output = session.last_plan
            return (output, session) if return_session else output

        result = session.last_plan
        while session.next_stage_index < len(session.limits):
            node_count = session.limits[session.next_stage_index]
            for second_index in range(
                max(1, session.previous_node_count), node_count
            ):
                second = session.all_nodes[second_index]
                for first_index in range(second_index):
                    first = session.all_nodes[first_index]
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
                        session.values,
                        session.inflated,
                        session.resolution,
                        session.origin,
                    )
                    if profile is None:
                        continue
                    edge_length, unknown_fraction = profile
                    session.edges.append(
                        VisibilityEdge(
                            first.node_id,
                            second.node_id,
                            edge_length,
                            unknown_fraction,
                        )
                    )
            session.previous_node_count = node_count
            session.next_stage_index += 1
            nodes = session.all_nodes[:node_count]
            known_ids, known_cost = self._search(
                nodes,
                session.edges,
                known_only=True,
                unknown_cost_weight=self.unknown_cost_weight,
                start_yaw=session.start_yaw,
                directed_failed_branches=session.directed_failed_branches,
                failure_buffer_m=session.failure_buffer_m,
            )
            if known_ids is not None:
                ids, cost, mode = known_ids, known_cost, "KNOWN_VISIBILITY"
            else:
                ids, cost = self._search(
                    nodes,
                    session.edges,
                    known_only=False,
                    unknown_cost_weight=self.unknown_cost_weight,
                    start_yaw=session.start_yaw,
                    directed_failed_branches=session.directed_failed_branches,
                    failure_buffer_m=session.failure_buffer_m,
                )
                mode = "ATTEMPTABLE_VISIBILITY"
            known_edges = sum(edge.known for edge in session.edges)
            graph = self._component_diagnostics(nodes, session.edges)
            selected = max(0, node_count - 2)
            node_limit_hit = session.candidate_vertices_total > selected
            session.complete = bool(
                session.next_stage_index >= len(session.limits)
                or not node_limit_hit
            )
            common = dict(
                candidate_vertices_total=int(session.candidate_vertices_total),
                candidate_vertices_selected=int(selected),
                node_limit_hit=bool(node_limit_hit),
                planning_node_limit=int(node_count),
                start_degree=int(graph["start_degree"]),
                goal_degree=int(graph["goal_degree"]),
                connected_components=int(graph["connected_components"]),
                start_component_size=int(graph["start_component_size"]),
                goal_component_size=int(graph["goal_component_size"]),
                start_clearance_m=session.start_clearance_m,
                goal_clearance_m=session.goal_clearance_m,
                start_inside_inflation=bool(session.start_inside_inflation),
                goal_inside_inflation=bool(session.goal_inside_inflation),
                progressive_stages=int(session.next_stage_index),
                progressive_complete=bool(session.complete),
                trajectory_vertices_total=int(
                    getattr(session, "trajectory_vertices_total", 0)
                ),
                trajectory_vertices_selected=int(
                    getattr(session, "trajectory_vertices_selected", 0)
                ),
            )
            if ids is None:
                if session.start_inside_inflation:
                    disconnect_class = "START_IN_INFLATION"
                elif session.goal_inside_inflation:
                    disconnect_class = "GOAL_IN_INFLATION"
                elif node_limit_hit:
                    disconnect_class = "NODE_LIMIT_COMPONENT_DISCONNECTED"
                elif graph["start_degree"] == 0:
                    disconnect_class = "START_ISOLATED"
                elif graph["goal_degree"] == 0:
                    disconnect_class = "GOAL_ISOLATED"
                else:
                    disconnect_class = "COMPONENT_DISCONNECTED"
                partial = self._partial_attemptable_path(
                    nodes,
                    session.edges,
                    session.values,
                    session.inflated,
                    session.resolution,
                    session.origin,
                    session.goal,
                    start_yaw=session.start_yaw,
                    directed_failed_branches=session.directed_failed_branches,
                    failure_buffer_m=session.failure_buffer_m,
                )
                if partial is None:
                    result = VisibilityPlan(
                        status="NO_ROUTE",
                        mode="NONE",
                        path=(),
                        nodes=tuple(nodes),
                        edges=tuple(session.edges),
                        path_cost=None,
                        known_edges=int(known_edges),
                        attemptable_edges=int(len(session.edges) - known_edges),
                        reason="visibility_graph_disconnected",
                        disconnect_class=disconnect_class,
                        **common,
                    )
                else:
                    _, progress, frontier_unknown, path, partial_cost = partial
                    path_length = float(sum(
                        math.hypot(
                            second[0] - first[0], second[1] - first[1]
                        )
                        for first, second in zip(path[:-1], path[1:])
                    ))
                    path_profiles = [
                        self._segment_profile(
                            first,
                            second,
                            session.values,
                            session.inflated,
                            session.resolution,
                            session.origin,
                        )
                        for first, second in zip(path[:-1], path[1:])
                    ]
                    path_unknown_fraction = float(
                        sum(
                            profile[0] * profile[1]
                            for profile in path_profiles
                            if profile is not None
                        )
                        / max(1.0e-9, path_length)
                    )
                    result = VisibilityPlan(
                        status="PASS",
                        mode="PARTIAL_ATTEMPTABLE",
                        path=tuple(path),
                        nodes=tuple(nodes),
                        edges=tuple(session.edges),
                        path_cost=float(partial_cost),
                        known_edges=int(known_edges),
                        attemptable_edges=int(len(session.edges) - known_edges),
                        reason="partial_reachable_frontier",
                        path_length=path_length,
                        path_unknown_fraction=path_unknown_fraction,
                        disconnect_class=disconnect_class,
                        partial_goal_progress_m=float(progress),
                        partial_frontier_unknown_fraction=float(frontier_unknown),
                        **common,
                    )
            else:
                by_id = {node.node_id: node for node in nodes}
                path = tuple(
                    (by_id[node_id].x, by_id[node_id].y) for node_id in ids
                )
                edge_by_pair = {
                    tuple(sorted((edge.first, edge.second))): edge
                    for edge in session.edges
                }
                path_edges = [
                    edge_by_pair[tuple(sorted((first, second)))]
                    for first, second in zip(ids[:-1], ids[1:])
                ]
                path_length = float(sum(edge.length for edge in path_edges))
                unknown_fraction = float(
                    sum(edge.length * edge.unknown_fraction for edge in path_edges)
                    / max(1.0e-9, path_length)
                )
                session.complete = True
                common["progressive_complete"] = True
                result = VisibilityPlan(
                    status="PASS",
                    mode=mode,
                    path=path,
                    nodes=tuple(nodes),
                    edges=tuple(session.edges),
                    path_cost=cost,
                    known_edges=int(known_edges),
                    attemptable_edges=int(len(session.edges) - known_edges),
                    reason=(
                        "known_route"
                        if mode == "KNOWN_VISIBILITY"
                        else "attempt_unknown_route"
                    ),
                    path_length=path_length,
                    path_unknown_fraction=unknown_fraction,
                    disconnect_class="NONE",
                    **common,
                )
            session.last_plan = result
            if session.complete or time.perf_counter() - started >= budget:
                break
        if result is None:
            raise RuntimeError("progressive visibility session produced no stage")
        return (result, session) if return_session else result
