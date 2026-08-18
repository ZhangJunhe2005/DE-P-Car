"""P6 hard-veto boundary for learned physical candidate trajectories."""

from typing import Iterable, Sequence, Tuple

import numpy as np

from dep_car.core.occupancy import FootprintConfig, OccupancyGrid2D
from dep_car.core.planner import PlanningResult
from dep_car.core.safety import DynamicSafetyConfig, evaluate_dynamic, evaluate_static, goal_cost
from dep_car.core.state_contract import (
    ACCELERATION_LIMIT_MPS2,
    DECELERATION_LIMIT_MPS2,
    FORWARD_SPEED_LIMIT_MPS,
    REVERSE_SPEED_LIMIT_MPS,
)
from dep_car.core.types import Candidate, DynamicTrack, Gear
from dep_car.core.vehicle import PLANNER_ROLLOUT_WHEELBASE_M, STEERING_OPERATING_LIMIT_RAD


KINEMATIC_NUMERICAL_TOLERANCE = 1.0e-5
STEERING_RATE_LIMIT_RAD_S = 0.75
LATERAL_ACCELERATION_LIMIT_MPS2 = 3.0


def kinematic_veto_reason(candidate: Candidate) -> str:
    """Return the first deterministic physical violation, or an empty string."""

    trajectory = np.asarray(candidate.trajectory, dtype=np.float64)
    if trajectory.ndim != 2 or trajectory.shape[1] != 6 or len(trajectory) < 2:
        return "kinematic_shape"
    if not np.all(np.isfinite(trajectory)):
        return "kinematic_non_finite"
    dt = np.diff(trajectory[:, 0])
    if np.any(dt <= 0.0):
        return "kinematic_time"
    gear = int(Gear.require_drive(candidate.gear))
    speed = trajectory[:, 4]
    steering = trajectory[:, 5]
    tolerance = KINEMATIC_NUMERICAL_TOLERANCE
    speed_limit = FORWARD_SPEED_LIMIT_MPS if gear > 0 else REVERSE_SPEED_LIMIT_MPS
    checks = (
        (np.any(gear * speed < -tolerance), "kinematic_opposite_motion"),
        (np.any(np.abs(speed) > speed_limit + tolerance), "kinematic_speed_limit"),
        (
            np.any(np.abs(steering) > STEERING_OPERATING_LIMIT_RAD + tolerance),
            "kinematic_steering_limit",
        ),
    )
    for failed, reason in checks:
        if failed:
            return reason
    acceleration = np.diff(speed) / dt
    steering_rate = np.diff(steering) / dt
    directed_acceleration = gear * acceleration
    if np.any(steering_rate > STEERING_RATE_LIMIT_RAD_S + tolerance) or np.any(
        steering_rate < -STEERING_RATE_LIMIT_RAD_S - tolerance
    ):
        return "kinematic_steering_rate"
    if np.any(directed_acceleration > ACCELERATION_LIMIT_MPS2 + tolerance):
        return "kinematic_acceleration"
    if np.any(directed_acceleration < -DECELERATION_LIMIT_MPS2 - tolerance):
        return "kinematic_deceleration"
    lateral = speed * speed * np.abs(np.tan(steering)) / PLANNER_ROLLOUT_WHEELBASE_M
    if np.any(lateral > LATERAL_ACCELERATION_LIMIT_MPS2 + tolerance):
        return "kinematic_lateral_acceleration"
    return ""


def evaluate_learned_candidate_bank(
    candidates: Sequence[Candidate],
    subgoal_body: Tuple[float, float],
    occupancy: OccupancyGrid2D,
    tracks: Iterable[DynamicTrack] = (),
    *,
    generation=0,
    footprint: FootprintConfig = FootprintConfig(),
    dynamic: DynamicSafetyConfig = DynamicSafetyConfig(),
) -> PlanningResult:
    """Apply kinematic/static/dynamic vetoes before learned-score ranking.

    The Score Head was trained with lower-is-better listwise ranking.  Runtime
    guidance cost is retained for diagnostics but is not added a second time
    to the learned score.
    """

    candidates = list(candidates)
    if len(candidates) != 15:
        raise ValueError("DEPCarNetV1 must provide exactly 15 candidates")
    ids = [candidate.candidate_id for candidate in candidates]
    if ids != list(range(15)):
        raise ValueError("learned candidates must preserve canonical IDs 0..14")
    tracks = tuple(tracks)
    any_static_safe = False
    any_dynamic_rejected = False
    for candidate in candidates:
        candidate.feasible = True
        candidate.veto_reason = ""
        goal_cost(candidate, subgoal_body)
        reason = kinematic_veto_reason(candidate)
        if reason:
            candidate.feasible = False
            candidate.veto_reason = reason
            continue
        evaluate_static(candidate, occupancy, footprint)
        if not candidate.feasible:
            continue
        any_static_safe = True
        evaluate_dynamic(candidate, tracks, dynamic)
        if not candidate.feasible:
            any_dynamic_rejected = True
    feasible = [candidate for candidate in candidates if candidate.feasible]
    selected = min(feasible, key=lambda item: (item.learned_score, item.candidate_id)) if feasible else None
    return PlanningResult(
        selected=selected,
        candidates=candidates,
        retime_factor=1.0 if selected is not None else None,
        blocked_by_static=selected is None and not any_static_safe,
        blocked_by_dynamic=selected is None and any_static_safe and any_dynamic_rejected,
        generation=int(generation),
    )


__all__ = ["evaluate_learned_candidate_bank", "kinematic_veto_reason"]
