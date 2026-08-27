"""P6 route-tracking helpers between global topology and local motion.

The topological planner owns connectivity while the local planner owns
Ackermann feasibility.  These helpers keep that boundary explicit: a global
route is used only as a soft reference and to prevent a lookahead target from
jumping through an obstacle.  Static/dynamic hard vetoes remain authoritative.
"""

import math
from typing import Sequence, Tuple

import numpy as np

from dep_car.core.occupancy import FootprintConfig


def wrap_angle(value: float) -> float:
    return float(math.atan2(math.sin(value), math.cos(value)))


def route_xy(poses: Sequence) -> np.ndarray:
    """Return route positions without depending on a concrete pose class."""

    return np.asarray([[float(pose[0]), float(pose[1])] for pose in poses], dtype=float)


def required_center_clearance(grid) -> float:
    """Circular centre-line envelope used only for route visibility."""

    return float(FootprintConfig().circle_radius + grid.fixed_grid_allowance())


def segment_minimum_clearance(grid, first, second, spacing_m=None) -> float:
    """Sample the minimum map clearance along a straight centre segment."""

    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    distance = float(np.linalg.norm(second - first))
    spacing = (
        max(0.01, 0.5 * float(grid.resolution))
        if spacing_m is None
        else max(0.01, float(spacing_m))
    )
    count = max(2, int(math.ceil(distance / spacing)) + 1)
    samples = first + np.linspace(0.0, 1.0, count)[:, None] * (second - first)
    if hasattr(grid, "sample_distance_field"):
        clearances = grid.sample_distance_field(samples)
    else:
        clearances = np.asarray([grid.point_clearance(point) for point in samples])
    return float(np.min(clearances))


def segment_is_visible(grid, first, second, minimum_clearance_m=None) -> bool:
    required = (
        required_center_clearance(grid)
        if minimum_clearance_m is None
        else float(minimum_clearance_m)
    )
    return segment_minimum_clearance(grid, first, second) > required


def monotonic_route_index(
    poses: Sequence,
    begin: int,
    position,
    *,
    grid=None,
    maximum_search: int = 20,
) -> Tuple[int, float]:
    """Advance to a nearby visible route point without jumping across walls."""

    if not poses:
        raise ValueError("poses cannot be empty")
    begin = min(max(0, int(begin)), len(poses) - 1)
    end = min(len(poses), begin + max(1, int(maximum_search)))
    position = np.asarray(position, dtype=float)
    points = route_xy(poses)
    visible = []
    for index in range(begin, end):
        if grid is None or segment_is_visible(grid, position, points[index]):
            visible.append(index)
    candidates = visible or [begin]
    distances = [float(np.linalg.norm(points[index] - position)) for index in candidates]
    selected = candidates[int(np.argmin(distances))]
    return selected, float(min(distances))


def visible_corridor_subgoal(
    poses: Sequence,
    begin: int,
    start_xy,
    grid,
    lookahead_m: float,
    *,
    requested_gear=None,
    visibility_buffer_m: float = 0.08,
    allowed_clearance_degradation_m: float = 0.01,
) -> Tuple[int, float, float, str]:
    """Select the furthest directly visible route point inside the lookahead.

    The selected point remains on the A* route.  Visibility does not certify a
    vehicle command; it only prevents the local objective from landing beyond
    a wall or around a blind corner.
    """

    if not poses:
        raise ValueError("poses cannot be empty")
    if lookahead_m <= 0.0:
        raise ValueError("lookahead_m must be positive")
    begin = min(max(0, int(begin)), len(poses) - 1)
    if begin == len(poses) - 1:
        clearance = segment_minimum_clearance(grid, start_xy, poses[begin][:2])
        return begin, 0.0, clearance, "path_end"

    start_xy = np.asarray(start_xy, dtype=float)
    hard_required = required_center_clearance(grid)
    start_clearance = float(grid.point_clearance(start_xy))
    # In a narrow passage the current pose may not have the preferred extra
    # margin, so demanding it absolutely would deadlock.  Instead require the
    # chord to preserve current clearance (within a small sampling tolerance),
    # capped at the desired buffer above the hard circular envelope.
    visibility_required = max(
        hard_required,
        min(
            hard_required + max(0.0, float(visibility_buffer_m)),
            start_clearance - max(0.0, float(allowed_clearance_degradation_m)),
        ),
    )
    previous = np.asarray(poses[begin][:2], dtype=float)
    selected = begin
    travelled = 0.0
    selected_clearance = segment_minimum_clearance(grid, start_xy, previous)
    reason = "path_end"
    for index in range(begin + 1, len(poses)):
        pose = poses[index]
        if requested_gear is not None and getattr(pose, "gear", requested_gear) != requested_gear:
            reason = "gear_boundary"
            break
        current = np.asarray(pose[:2], dtype=float)
        step = float(np.linalg.norm(current - previous))
        proposed_travel = travelled + step
        clearance = segment_minimum_clearance(grid, start_xy, current)
        if clearance <= visibility_required:
            reason = "visibility_boundary"
            break
        selected = index
        travelled = proposed_travel
        selected_clearance = clearance
        previous = current
        if travelled >= lookahead_m:
            reason = "lookahead"
            break

    # A valid A* route should always expose its immediate next cell.  Retain a
    # forward-progress target if grid quantisation makes that one cell land on
    # the exact conservative visibility threshold; local hard safety still
    # decides whether any motion may execute.
    if selected == begin:
        selected = begin + 1
        selected_clearance = segment_minimum_clearance(
            grid, start_xy, poses[selected][:2]
        )
        travelled = float(
            np.linalg.norm(np.asarray(poses[selected][:2], dtype=float) - previous)
        )
        reason = "minimum_progress"
    return selected, travelled, selected_clearance, reason


def route_reference_body(
    world_points,
    vehicle_pose,
    begin: int,
    *,
    horizon_m: float = 2.5,
) -> np.ndarray:
    """Transform a bounded, monotonic global-route suffix into body axes."""

    points = np.asarray(world_points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or not len(points):
        return np.empty((0, 2), dtype=float)
    begin = min(max(0, int(begin)), len(points) - 1)
    x, y, heading = (float(value) for value in vehicle_pose)
    origin = np.asarray([x, y], dtype=float)
    c, s = math.cos(heading), math.sin(heading)
    output = [np.zeros(2, dtype=float)]
    travelled = 0.0
    previous = origin
    # ``begin`` is the closest already-reached route sample published by the
    # global tracker.  Starting the reference at that slightly-behind point
    # can manufacture an almost-pi "corner" as the vehicle passes it.  The
    # body origin already represents current progress, so track from the next
    # route point onward.
    suffix = points[begin + 1 :] if begin + 1 < len(points) else points[begin:]
    for point in suffix:
        step = float(np.linalg.norm(point - previous))
        travelled += step
        previous = point
        delta = point - origin
        body = np.asarray([c * delta[0] + s * delta[1], -s * delta[0] + c * delta[1]])
        if np.linalg.norm(body - output[-1]) > 1.0e-6:
            output.append(body)
        if travelled >= horizon_m:
            break
    return np.asarray(output, dtype=float)


def monotonic_route_reference_body(
    world_points,
    vehicle_pose,
    begin: int,
    *,
    grid=None,
    horizon_m: float = 2.5,
    maximum_search: int = 30,
):
    """Attach a rolling corridor to the nearest not-yet-passed sample.

    FAR republishes a route from the current pose during ordinary driving, but
    intentionally freezes it while a multi-leg Ackermann turn owns control.
    Continuing to read that frozen route from index zero makes the already
    traversed dense prefix dominate metric lookahead and can point the local
    planner back into the failed branch.  Advance a monotonic cursor in body
    coordinates, where the live local occupancy grid can also reject a
    geometrically close point across a wall.
    """

    points = np.asarray(world_points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or not len(points):
        return np.empty((0, 2), dtype=float), 0
    begin = min(max(0, int(begin)), len(points) - 1)
    x, y, heading = (float(value) for value in vehicle_pose)
    delta = points - np.asarray((x, y), dtype=float)
    c, s = math.cos(heading), math.sin(heading)
    body = np.column_stack((
        c * delta[:, 0] + s * delta[:, 1],
        -s * delta[:, 0] + c * delta[:, 1],
    ))
    selected, _ = monotonic_route_index(
        body.tolist(),
        begin,
        (0.0, 0.0),
        grid=grid,
        maximum_search=maximum_search,
    )
    return (
        route_reference_body(
            points,
            vehicle_pose,
            selected,
            horizon_m=horizon_m,
        ),
        int(selected),
    )


def _point_polyline_distances(points: np.ndarray, polyline: np.ndarray) -> np.ndarray:
    if len(polyline) == 1:
        return np.linalg.norm(points - polyline[0], axis=1)
    starts = polyline[:-1]
    segments = polyline[1:] - starts
    denominator = np.sum(segments * segments, axis=1)
    delta = points[:, None, :] - starts[None, :, :]
    projection = np.divide(
        np.sum(delta * segments[None, :, :], axis=2),
        denominator[None, :],
        out=np.zeros((len(points), len(segments)), dtype=float),
        where=denominator[None, :] > 1.0e-12,
    )
    projection = np.clip(projection, 0.0, 1.0)
    closest = starts[None, :, :] + projection[:, :, None] * segments[None, :, :]
    return np.min(np.linalg.norm(points[:, None, :] - closest, axis=2), axis=1)


def apply_runtime_route_preference(
    result,
    reference_path,
    occupancy,
    *,
    corridor_weight: float = 2.0,
    desired_future_clearance_m: float = 0.15,
    clearance_weight: float = 1.5,
    corner_corridor_minimum_scale: float = 0.25,
):
    """Re-rank a hard-vetoed bank without forcing an Ackermann car onto an L.

    A topology path owns connectivity, not the exact centre-line geometry of
    a turn.  Its weight is therefore relaxed continuously near a strong bend,
    allowing the local planner to take a wider feasible arc around the inner
    wall while retaining full route guidance on straight segments.
    """

    if result is None or not result.executable:
        return result
    reference = np.asarray(reference_path, dtype=float)
    if reference.ndim != 2 or reference.shape[1] != 2 or len(reference) < 2:
        return result
    minimum_scale = float(corner_corridor_minimum_scale)
    if not 0.0 <= minimum_scale <= 1.0:
        raise ValueError("corner_corridor_minimum_scale must be in [0,1]")
    severity = corner_severity(reference)
    effective_corridor_weight = float(corridor_weight) * (
        1.0 - severity * (1.0 - minimum_scale)
    )
    for candidate in result.candidates:
        if not candidate.feasible:
            continue
        trajectory = np.asarray(candidate.trajectory, dtype=float)
        distances = _point_polyline_distances(trajectory[1:, 1:3], reference)
        route_penalty = float(np.mean(distances) + 0.5 * np.max(distances))
        tail = trajectory[min(2, len(trajectory) - 1) :]
        _, future_clearance = occupancy.swept_footprint_clearance(tail)
        clearance_deficit = max(0.0, desired_future_clearance_m - future_clearance)
        candidate.guidance_cost += (
            effective_corridor_weight * route_penalty
            + float(clearance_weight) * clearance_deficit
        )
    feasible = [candidate for candidate in result.candidates if candidate.feasible]
    if feasible:
        result.selected = min(feasible, key=lambda candidate: candidate.total_cost)
    return result


def corner_severity(
    reference_path,
    *,
    trigger_rad: float = 0.35,
    full_strength_rad: float = 1.20,
) -> float:
    """Return a smooth 0--1 activation for an upcoming route bend."""

    return _angle_severity(
        route_turn_angle(reference_path),
        trigger_rad=trigger_rad,
        full_strength_rad=full_strength_rad,
    )


def _angle_severity(
    angle_rad: float,
    *,
    trigger_rad: float,
    full_strength_rad: float,
) -> float:
    """Return a smooth activation for one absolute angular demand."""

    trigger = float(trigger_rad)
    full = float(full_strength_rad)
    if not 0.0 <= trigger < full <= math.pi:
        raise ValueError("corner severity thresholds are invalid")
    angle = abs(float(angle_rad))
    linear = min(1.0, max(0.0, (angle - trigger) / (full - trigger)))
    # Smoothstep avoids candidate-ranking chatter as the sampled route angle
    # moves by one grid cell around the activation threshold.
    return float(linear * linear * (3.0 - 2.0 * linear))


def trajectory_turn_angle(trajectory) -> float:
    """Return the largest body-yaw excursion in a candidate rollout.

    A short rolling exploration reference can look almost straight while the
    Ackermann bank is already beginning one long 90-degree turn.  Inspecting
    the candidate bank keeps the elastic corner halo active across a temporary
    FAR/local-authority handoff.
    """

    values = np.asarray(trajectory, dtype=float)
    if values.ndim != 2 or values.shape[1] < 4 or len(values) < 2:
        return 0.0
    yaw = np.unwrap(values[:, 3])
    if not np.all(np.isfinite(yaw)):
        return 0.0
    return float(np.max(np.abs(yaw - yaw[0])))


def apply_corner_clearance_preference(
    result,
    reference_path,
    occupancy,
    *,
    soft_clearance_m: float = 0.30,
    weight: float = 3.0,
    trigger_rad: float = 0.35,
    full_strength_rad: float = 1.20,
    candidate_trigger_rad: float = 0.10,
    candidate_full_strength_rad: float = 0.60,
    learned_score_base: bool = False,
):
    """Push feasible turning candidates away from walls without blocking them.

    This is a soft swept-footprint halo, not occupancy inflation.  Candidates
    inside the halo remain feasible and hard-veto semantics are untouched.
    The normalized quadratic barrier strongly distinguishes two nearly
    colliding inner arcs while becoming exactly zero once useful margin is
    available.  It is applied to deterministic and learned banks alike.
    """

    if result is None or not result.executable:
        return result
    target = float(soft_clearance_m)
    strength = float(weight)
    if target < 0.0 or strength < 0.0:
        raise ValueError("corner soft-clearance parameters cannot be negative")
    if target == 0.0 or strength == 0.0:
        return result
    route_severity = corner_severity(
        reference_path,
        trigger_rad=trigger_rad,
        full_strength_rad=full_strength_rad,
    )
    feasible = [candidate for candidate in result.candidates if candidate.feasible]
    candidate_turn = max(
        (trajectory_turn_angle(candidate.trajectory) for candidate in feasible),
        default=0.0,
    )
    candidate_severity = _angle_severity(
        candidate_turn,
        trigger_rad=candidate_trigger_rad,
        full_strength_rad=candidate_full_strength_rad,
    )
    # Use one bank-wide activation.  Per-candidate activation would let a
    # straight-but-wrong candidate evade the halo exactly when all of the
    # useful candidates are turning around a convex wall corner.
    severity = max(route_severity, candidate_severity)
    result.corner_soft_route_severity = float(route_severity)
    result.corner_soft_candidate_severity = float(candidate_severity)
    result.corner_soft_candidate_turn_angle = float(candidate_turn)
    result.corner_soft_effective_severity = float(severity)
    result.corner_soft_applied = False
    result.corner_soft_selected_clearance = None
    result.corner_soft_selected_cost = 0.0
    if severity <= 0.0:
        return result

    soft_costs = {}
    clearances = {}
    for candidate in result.candidates:
        if not candidate.feasible:
            continue
        trajectory = np.asarray(candidate.trajectory, dtype=float)
        tail = trajectory[min(2, len(trajectory) - 1) :]
        _, future_clearance = occupancy.swept_footprint_clearance(tail)
        normalized_deficit = max(0.0, target - future_clearance) / target
        soft_cost = strength * severity * normalized_deficit ** 2
        candidate.guidance_cost += soft_cost
        soft_costs[id(candidate)] = soft_cost
        clearances[id(candidate)] = float(future_clearance)

    if feasible:
        if learned_score_base:
            result.selected = min(
                feasible,
                key=lambda candidate: (
                    candidate.learned_score + soft_costs.get(id(candidate), 0.0),
                    candidate.candidate_id,
                ),
            )
        else:
            result.selected = min(
                feasible,
                key=lambda candidate: (candidate.total_cost, candidate.candidate_id),
            )
        result.corner_soft_applied = any(value > 0.0 for value in soft_costs.values())
        result.corner_soft_selected_clearance = clearances.get(id(result.selected))
        result.corner_soft_selected_cost = float(
            soft_costs.get(id(result.selected), 0.0)
        )
    return result


def route_turn_angle(reference_path, near_m: float = 0.40, far_m: float = 1.20) -> float:
    """Return a smoothed upcoming turn angle from a body-frame route suffix."""

    points = np.asarray(reference_path, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3:
        return 0.0
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    arc = np.concatenate(([0.0], np.cumsum(steps)))
    near_index = int(np.searchsorted(arc, near_m, side="left"))
    far_index = int(np.searchsorted(arc, far_m, side="left"))
    near_index = min(max(1, near_index), len(points) - 2)
    far_index = min(max(near_index + 1, far_index), len(points) - 1)
    first = points[near_index] - points[0]
    second = points[far_index] - points[near_index]
    if np.linalg.norm(first) < 1.0e-6 or np.linalg.norm(second) < 1.0e-6:
        return 0.0
    return abs(
        wrap_angle(math.atan2(second[1], second[0]) - math.atan2(first[1], first[0]))
    )


def corner_speed_limit(
    turn_angle_rad: float,
    *,
    trigger_rad: float = 0.20,
    straight_speed_mps: float = 2.00,
    ninety_degree_speed_mps: float = 0.26,
    turn_window_m: float = 1.20,
    lateral_acceleration_limit_mps2: float = 0.75,
):
    """Return a curvature-aware cap while preserving sharp-corner caution.

    The old linear envelope started at 0.55 m/s immediately above the turn
    trigger, so even a 13-degree bend over the 1.2 m lookahead disabled safe
    cruise.  The new cap is the lower of the existing sharp-corner taper and
    ``sqrt(a_lat / curvature)``.  It raises no physical limit and continues to
    approach ``ninety_degree_speed_mps`` at a right angle.
    """

    angle = abs(float(turn_angle_rad))
    if angle <= trigger_rad:
        return None
    if min(
        straight_speed_mps,
        ninety_degree_speed_mps,
        turn_window_m,
        lateral_acceleration_limit_mps2,
    ) <= 0.0:
        raise ValueError("corner speed-envelope parameters must be positive")
    severity = min(1.0, (angle - trigger_rad) / (0.5 * math.pi - trigger_rad))
    tapered = float(
        straight_speed_mps
        + severity * (ninety_degree_speed_mps - straight_speed_mps)
    )
    curvature = angle / float(turn_window_m)
    lateral = math.sqrt(
        float(lateral_acceleration_limit_mps2) / max(curvature, 1.0e-6)
    )
    return float(min(tapered, lateral))


__all__ = [
    "apply_corner_clearance_preference",
    "apply_runtime_route_preference",
    "corner_severity",
    "corner_speed_limit",
    "monotonic_route_index",
    "required_center_clearance",
    "route_reference_body",
    "monotonic_route_reference_body",
    "trajectory_turn_angle",
    "route_turn_angle",
    "segment_is_visible",
    "segment_minimum_clearance",
    "visible_corridor_subgoal",
]
