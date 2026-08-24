"""Candidate generation, hard filtering, risk ranking and ordered retiming."""

import math
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

from .lattice import AckermannLattice, LatticeConfig
from .occupancy import FootprintConfig, OccupancyGrid2D
from .safety import (
    DynamicSafetyConfig,
    evaluate_dynamic,
    evaluate_static,
    evaluate_static_margin_egress,
    goal_cost,
)
from .types import Candidate, DynamicTrack, Gear, VehicleState


@dataclass(frozen=True)
class PlannerConfig:
    lattice: LatticeConfig = LatticeConfig()
    footprint: FootprintConfig = FootprintConfig()
    dynamic: DynamicSafetyConfig = DynamicSafetyConfig()
    retime_factors: Sequence[float] = (1.0, 1.2, 1.4)
    command_freshness_s: float = 0.35


@dataclass
class PlanningResult:
    selected: Optional[Candidate]
    candidates: List[Candidate]
    retime_factor: Optional[float]
    blocked_by_static: bool
    blocked_by_dynamic: bool
    generation: int
    # Geometric horizon retained by the selected safety fallback.  Retime
    # factor alone cannot encode this because a 0.75 spatial primitive slowed
    # by 1.4 has the same duration scale as a full primitive.  Turnaround exit
    # verification needs to distinguish a useful forward corridor from a tiny
    # emergency twitch.
    spatial_scale: Optional[float] = None

    @property
    def executable(self) -> bool:
        return self.selected is not None


class DeterministicPlanner:
    """Safe fallback planner and runtime authority around learned offsets.

    Learned outputs may perturb and rank candidates, but cannot restore a
    candidate rejected by the static or dynamic hard safety checks.
    """

    def __init__(self, config: PlannerConfig = PlannerConfig()):
        self.config = config
        self.lattice = AckermannLattice(config.lattice)
        self._generation = 0

    def plan(
        self,
        state: VehicleState,
        subgoal_body: Tuple[float, float],
        occupancy: OccupancyGrid2D,
        tracks: Iterable[DynamicTrack] = (),
        requested_gear: Gear = Gear.FORWARD,
        target_heading: float = None,
        target_steering: float = None,
        spatial_scales: Sequence[float] = (1.0,),
        required_yaw_direction: float = None,
        minimum_yaw_progress_rad: float = 0.0,
        allow_static_margin_egress: bool = False,
        maximum_margin_overlap_m: float = 0.06,
        minimum_margin_improvement_m: float = 0.02,
        margin_worsening_tolerance_m: float = 0.01,
        learned_offsets: Optional[Tuple[Iterable[float], Iterable[float], Iterable[float]]] = None,
    ) -> PlanningResult:
        self._generation += 1
        tracks = tuple(tracks)
        offsets = learned_offsets or (None, None, None)
        last_candidates = []
        any_static_safe = False
        any_dynamic_rejected = False

        scales = tuple(float(value) for value in spatial_scales)
        if not scales or any(value <= 0.0 or value > 1.0 for value in scales):
            raise ValueError("spatial_scales must contain values in (0,1]")
        yaw_direction = None
        if required_yaw_direction is not None:
            value = float(required_yaw_direction)
            if not math.isfinite(value) or abs(value) < 1.0e-6:
                raise ValueError("required_yaw_direction must be finite and nonzero")
            yaw_direction = 1.0 if value > 0.0 else -1.0
        minimum_yaw_progress_rad = float(minimum_yaw_progress_rad)
        if not math.isfinite(minimum_yaw_progress_rad) or minimum_yaw_progress_rad < 0.0:
            raise ValueError("minimum_yaw_progress_rad must be finite and nonnegative")
        for spatial_scale in scales:
            for factor in self.config.retime_factors:
                duration_scale = factor * spatial_scale
                candidates = self.lattice.generate(
                    state,
                    speed_offsets=offsets[0],
                    steering_offsets=offsets[1],
                    learned_scores=offsets[2],
                    gear=requested_gear,
                    speed_scale=1.0 / factor,
                    duration_scale=duration_scale,
                )
                for candidate in candidates:
                    goal_cost(
                        candidate,
                        subgoal_body,
                        target_heading=target_heading,
                        target_steering=target_steering,
                    )
                    evaluate_static(candidate, occupancy, self.config.footprint)
                    if not candidate.feasible and allow_static_margin_egress:
                        evaluate_static_margin_egress(
                            candidate,
                            occupancy,
                            self.config.footprint,
                            maximum_overlap_m=maximum_margin_overlap_m,
                            minimum_improvement_m=minimum_margin_improvement_m,
                            worsening_tolerance_m=margin_worsening_tolerance_m,
                        )
                    if candidate.feasible:
                        any_static_safe = True
                        evaluate_dynamic(candidate, tracks, self.config.dynamic)
                        if not candidate.feasible:
                            any_dynamic_rejected = True
                    if candidate.feasible and yaw_direction is not None:
                        yaw_progress = float(candidate.trajectory[-1, 3])
                        if yaw_direction * yaw_progress < minimum_yaw_progress_rad:
                            candidate.feasible = False
                            candidate.veto_reason = "required_yaw_progress_not_met"
                feasible = [candidate for candidate in candidates if candidate.feasible]
                last_candidates = candidates
                if feasible:
                    selected = min(feasible, key=lambda candidate: candidate.total_cost)
                    return PlanningResult(
                        selected=selected,
                        candidates=candidates,
                        retime_factor=duration_scale,
                        blocked_by_static=False,
                        blocked_by_dynamic=False,
                        generation=self._generation,
                        spatial_scale=spatial_scale,
                    )

        return PlanningResult(
            selected=None,
            candidates=last_candidates,
            retime_factor=None,
            blocked_by_static=not any_static_safe,
            blocked_by_dynamic=any_static_safe and any_dynamic_rejected,
            generation=self._generation,
            spatial_scale=None,
        )
