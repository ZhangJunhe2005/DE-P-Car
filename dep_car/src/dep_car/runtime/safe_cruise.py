"""Route-confidence speed envelope for safe P6 Ackermann cruising.

The global route is direction authority, not collision authority.  This
module therefore never makes an unsafe candidate executable.  It only grants
higher forward-speed tiers after a goal-connected route has a live,
footprint-validated prefix long enough to stop inside.
"""

from dataclasses import dataclass
import math
from typing import Optional, Sequence


ROUTE_CONFIDENCE_LOCAL = 0
ROUTE_CONFIDENCE_PARTIAL = 1
ROUTE_CONFIDENCE_CONNECTED = 2


@dataclass(frozen=True)
class SafeCruiseConfig:
    speed_tiers_mps: Sequence[float] = (0.60, 1.20, 2.00)
    exploration_speed_mps: float = 0.60
    reaction_time_s: float = 0.45
    braking_deceleration_mps2: float = 2.00
    stopping_margin_m: float = 0.25
    promotion_confirmation_cycles: int = 5
    dynamic_time_headway_s: float = 1.00
    narrow_clearance_m: float = 0.10
    narrow_clearance_speed_mps: float = 0.60

    def __post_init__(self):
        tiers = tuple(float(value) for value in self.speed_tiers_mps)
        if not tiers or any(value <= 0.0 for value in tiers):
            raise ValueError("safe-cruise speed tiers must be positive")
        if any(second <= first for first, second in zip(tiers[:-1], tiers[1:])):
            raise ValueError("safe-cruise speed tiers must be strictly increasing")
        if self.exploration_speed_mps <= 0.0:
            raise ValueError("safe-cruise exploration speed must be positive")
        if self.reaction_time_s < 0.0:
            raise ValueError("safe-cruise reaction time cannot be negative")
        if self.braking_deceleration_mps2 <= 0.0:
            raise ValueError("safe-cruise braking deceleration must be positive")
        if self.stopping_margin_m < 0.0:
            raise ValueError("safe-cruise stopping margin cannot be negative")
        if self.promotion_confirmation_cycles <= 0:
            raise ValueError("safe-cruise promotion cycles must be positive")
        if self.dynamic_time_headway_s <= 0.0:
            raise ValueError("safe-cruise dynamic time headway must be positive")
        if self.narrow_clearance_m < 0.0 or self.narrow_clearance_speed_mps <= 0.0:
            raise ValueError("safe-cruise narrow-clearance contract is invalid")


@dataclass(frozen=True)
class SafeCruiseContext:
    route_confidence: int
    goal_connected: bool
    route_stable: bool
    verified_prefix_m: float
    forward_cruise: bool
    dynamic_clearance_m: float = float("inf")
    static_clearance_m: float = float("inf")


@dataclass(frozen=True)
class SafeCruiseDecision:
    speed_limit_mps: float
    target_tier_mps: float
    stable_cycles: int
    required_stopping_distance_m: float
    reason: str


def stopping_distance_m(speed_mps: float, config: SafeCruiseConfig) -> float:
    """Conservative distance needed before an unobserved/blocked suffix."""

    speed = max(0.0, abs(float(speed_mps)))
    return (
        speed * float(config.reaction_time_s)
        + speed * speed / (2.0 * float(config.braking_deceleration_mps2))
        + float(config.stopping_margin_m)
    )


def prefix_speed_tier_mps(prefix_m: float, config: SafeCruiseConfig) -> float:
    """Return the fastest configured tier which can stop in ``prefix_m``."""

    prefix = max(0.0, float(prefix_m))
    tiers = tuple(float(value) for value in config.speed_tiers_mps)
    permitted = [
        speed for speed in tiers if stopping_distance_m(speed, config) <= prefix
    ]
    return permitted[-1] if permitted else min(
        float(config.exploration_speed_mps), tiers[0]
    )


class SafeCruiseSupervisor:
    """Promote speed slowly and revoke it immediately when evidence weakens."""

    def __init__(self, config: SafeCruiseConfig = SafeCruiseConfig()):
        self.config = config
        self.stable_cycles = 0
        self.authorized_tier_index = 0

    def reset(self):
        self.stable_cycles = 0
        self.authorized_tier_index = 0

    def update(self, context: SafeCruiseContext) -> SafeCruiseDecision:
        cfg = self.config
        tiers = tuple(float(value) for value in cfg.speed_tiers_mps)
        connected = bool(
            context.forward_cruise
            and context.goal_connected
            and context.route_stable
            and int(context.route_confidence) == ROUTE_CONFIDENCE_CONNECTED
        )
        if not connected:
            self.reset()
            limit = min(float(cfg.exploration_speed_mps), tiers[0])
            return SafeCruiseDecision(
                speed_limit_mps=limit,
                target_tier_mps=limit,
                stable_cycles=0,
                required_stopping_distance_m=stopping_distance_m(limit, cfg),
                reason="conservative_nonconnected_route",
            )

        self.stable_cycles += 1
        prefix_target = prefix_speed_tier_mps(context.verified_prefix_m, cfg)
        target_index = max(
            0,
            max(
                index
                for index, value in enumerate(tiers)
                if value <= prefix_target + 1.0e-9
            ),
        )
        confirmed_index = min(
            len(tiers) - 1,
            self.stable_cycles // int(cfg.promotion_confirmation_cycles),
        )
        # Loss of prefix/clearance authority revokes a tier immediately.  A
        # gain is deliberately rate-limited by the confirmation counter.
        self.authorized_tier_index = min(target_index, confirmed_index)
        limit = tiers[self.authorized_tier_index]
        reasons = ["connected_prefix_authorized"]

        dynamic_clearance = float(context.dynamic_clearance_m)
        if math.isfinite(dynamic_clearance):
            dynamic_limit = max(
                0.0, dynamic_clearance / float(cfg.dynamic_time_headway_s)
            )
            if dynamic_limit < limit:
                limit = dynamic_limit
                reasons.append("dynamic_headway")

        static_clearance = float(context.static_clearance_m)
        if (
            math.isfinite(static_clearance)
            and static_clearance < float(cfg.narrow_clearance_m)
            and limit > float(cfg.narrow_clearance_speed_mps)
        ):
            limit = float(cfg.narrow_clearance_speed_mps)
            reasons.append("narrow_clearance")

        limit = max(0.0, limit)
        return SafeCruiseDecision(
            speed_limit_mps=limit,
            target_tier_mps=tiers[target_index],
            stable_cycles=self.stable_cycles,
            required_stopping_distance_m=stopping_distance_m(limit, cfg),
            reason="+".join(reasons),
        )


def prefer_progress_candidate(
    result,
    speed_limit_mps: Optional[float],
    config: SafeCruiseConfig = SafeCruiseConfig(),
    steering_lane_tolerance_rad: float = 0.04,
):
    """Prefer efficient progress only inside an already hard-safe bank.

    Candidate feasibility and route/corner soft costs have already been
    evaluated by the caller.  The selected low-speed candidate defines the
    steering lane; promotion may only choose a faster primitive in that same
    lane.  This avoids trading route geometry for speed while preventing the
    local subgoal distance term from permanently preferring the 0.6 m/s bank.
    This function cannot restore a vetoed candidate.
    """

    if result is None or not result.executable or speed_limit_mps is None:
        return result
    cap = max(0.0, float(speed_limit_mps))
    baseline = result.selected
    tolerance = max(0.0, float(steering_lane_tolerance_rad))

    def candidate_is_speed_safe(candidate):
        speed = abs(float(candidate.speed_anchor))
        if speed > cap + 1.0e-9:
            return False
        dynamic_clearance = float(candidate.dynamic_clearance)
        if (
            math.isfinite(dynamic_clearance)
            and dynamic_clearance
            < speed * float(config.dynamic_time_headway_s)
        ):
            return False
        static_clearance = float(candidate.static_clearance)
        if (
            math.isfinite(static_clearance)
            and static_clearance < float(config.narrow_clearance_m)
            and speed > float(config.narrow_clearance_speed_mps)
        ):
            return False
        return True

    feasible = [
        candidate
        for candidate in result.candidates
        if candidate.feasible
        and int(candidate.gear) == int(baseline.gear)
        and abs(
            float(candidate.steering_anchor)
            - float(baseline.steering_anchor)
        ) <= tolerance
        and candidate_is_speed_safe(candidate)
    ]
    if not feasible:
        return result
    result.selected = min(
        feasible,
        key=lambda candidate: (
            -float(candidate.trajectory[-1, 1]),
            -abs(float(candidate.speed_anchor)),
            float(candidate.total_cost),
            int(candidate.candidate_id),
        ),
    )
    return result


__all__ = [
    "ROUTE_CONFIDENCE_CONNECTED",
    "ROUTE_CONFIDENCE_LOCAL",
    "ROUTE_CONFIDENCE_PARTIAL",
    "SafeCruiseConfig",
    "SafeCruiseContext",
    "SafeCruiseDecision",
    "SafeCruiseSupervisor",
    "prefix_speed_tier_mps",
    "prefer_progress_candidate",
    "stopping_distance_m",
]
