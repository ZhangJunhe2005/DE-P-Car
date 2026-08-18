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
):
    """Re-rank an already hard-vetoed bank using route and clearance costs."""

    if result is None or not result.executable:
        return result
    reference = np.asarray(reference_path, dtype=float)
    if reference.ndim != 2 or reference.shape[1] != 2 or len(reference) < 2:
        return result
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
            float(corridor_weight) * route_penalty
            + float(clearance_weight) * clearance_deficit
        )
    feasible = [candidate for candidate in result.candidates if candidate.feasible]
    if feasible:
        result.selected = min(feasible, key=lambda candidate: candidate.total_cost)
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
    straight_speed_mps: float = 0.55,
    ninety_degree_speed_mps: float = 0.26,
):
    """Return no cap on straight routes and a smooth cap near corners."""

    angle = abs(float(turn_angle_rad))
    if angle <= trigger_rad:
        return None
    severity = min(1.0, (angle - trigger_rad) / (0.5 * math.pi - trigger_rad))
    return float(straight_speed_mps + severity * (ninety_degree_speed_mps - straight_speed_mps))


__all__ = [
    "apply_runtime_route_preference",
    "corner_speed_limit",
    "monotonic_route_index",
    "required_center_clearance",
    "route_reference_body",
    "route_turn_angle",
    "segment_is_visible",
    "segment_minimum_clearance",
    "visible_corridor_subgoal",
]
