from types import SimpleNamespace

import numpy as np

from dep_car.core.types import Candidate, Gear
from dep_car.runtime.safe_cruise import (
    ROUTE_CONFIDENCE_CONNECTED,
    ROUTE_CONFIDENCE_PARTIAL,
    SafeCruiseConfig,
    SafeCruiseContext,
    SafeCruiseSupervisor,
    prefer_progress_candidate,
    stopping_distance_m,
)


def context(**overrides):
    values = {
        "route_confidence": ROUTE_CONFIDENCE_CONNECTED,
        "goal_connected": True,
        "route_stable": True,
        "verified_prefix_m": 2.50,
        "forward_cruise": True,
    }
    values.update(overrides)
    return SafeCruiseContext(**values)


def candidate(
    candidate_id,
    speed,
    progress,
    cost,
    *,
    feasible=True,
    steering=0.0,
    static_clearance=float("inf"),
    dynamic_clearance=float("inf"),
):
    trajectory = np.zeros((3, 6), dtype=float)
    trajectory[-1, 1] = float(progress)
    value = Candidate(
        candidate_id=candidate_id,
        speed_anchor=float(speed),
        steering_anchor=float(steering),
        duration=1.0,
        trajectory=trajectory,
        gear=Gear.FORWARD,
        feasible=feasible,
        static_clearance=float(static_clearance),
        dynamic_clearance=float(dynamic_clearance),
    )
    value.guidance_cost = float(cost)
    return value


def test_stopping_distance_contract_fits_two_mps_inside_route_horizon():
    config = SafeCruiseConfig()
    assert stopping_distance_m(0.60, config) == 0.61
    assert stopping_distance_m(1.20, config) == 1.15
    assert stopping_distance_m(2.00, config) == 2.15


def test_connected_route_promotes_slowly_and_partial_revokes_immediately():
    supervisor = SafeCruiseSupervisor()
    first = supervisor.update(context())
    assert first.speed_limit_mps == 0.60
    for _ in range(4):
        promoted = supervisor.update(context())
    assert promoted.speed_limit_mps == 1.20
    for _ in range(5):
        promoted = supervisor.update(context())
    assert promoted.speed_limit_mps == 2.00

    revoked = supervisor.update(
        context(
            route_confidence=ROUTE_CONFIDENCE_PARTIAL,
            goal_connected=False,
            route_stable=False,
        )
    )
    assert revoked.speed_limit_mps == 0.60
    assert revoked.stable_cycles == 0


def test_prefix_never_authorizes_speed_which_cannot_stop_inside_it():
    supervisor = SafeCruiseSupervisor()
    for _ in range(20):
        decision = supervisor.update(context(verified_prefix_m=1.50))
    assert decision.target_tier_mps == 1.20
    assert decision.speed_limit_mps == 1.20
    assert decision.required_stopping_distance_m <= 1.50


def test_dynamic_headway_and_narrow_clearance_only_reduce_authority():
    supervisor = SafeCruiseSupervisor()
    for _ in range(10):
        decision = supervisor.update(context())
    assert decision.speed_limit_mps == 2.00

    dynamic = supervisor.update(context(dynamic_clearance_m=0.45))
    assert dynamic.speed_limit_mps == 0.45
    assert "dynamic_headway" in dynamic.reason

    narrow = supervisor.update(context(static_clearance_m=0.05))
    assert narrow.speed_limit_mps == 0.60
    assert "narrow_clearance" in narrow.reason


def test_progress_preference_cannot_restore_vetoed_or_over_limit_candidate():
    slow = candidate(0, 0.60, 0.50, 1.00)
    medium = candidate(1, 1.20, 0.80, 1.10)
    fast = candidate(2, 2.00, 0.95, 1.05)
    vetoed = candidate(3, 1.20, 1.20, 0.90, feasible=False)
    result = SimpleNamespace(
        selected=slow,
        candidates=[slow, medium, fast, vetoed],
        executable=True,
    )

    selected = prefer_progress_candidate(result, 1.20).selected
    assert selected is medium
    assert selected is not vetoed
    assert selected is not fast


def test_progress_preference_preserves_steering_lane_and_clearance_limits():
    slow = candidate(0, 0.60, 0.50, 0.1, steering=0.26)
    medium = candidate(1, 1.20, 0.85, 2.0, steering=0.26)
    wrong_lane = candidate(2, 2.00, 1.20, 0.0, steering=-0.26)
    dynamic_close = candidate(
        3, 2.00, 1.10, 0.0, steering=0.26, dynamic_clearance=0.8
    )
    result = SimpleNamespace(
        selected=slow,
        candidates=[slow, medium, wrong_lane, dynamic_close],
        executable=True,
    )

    selected = prefer_progress_candidate(result, 2.00).selected
    assert selected is medium
    assert selected is not wrong_lane
    assert selected is not dynamic_close


def test_nonconnected_context_never_accumulates_promotion_cycles():
    supervisor = SafeCruiseSupervisor()
    for _ in range(20):
        decision = supervisor.update(
            context(
                route_confidence=ROUTE_CONFIDENCE_PARTIAL,
                goal_connected=False,
                route_stable=False,
            )
        )
    assert decision.speed_limit_mps == 0.60
    assert decision.stable_cycles == 0
