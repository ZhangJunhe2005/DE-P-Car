import numpy as np

from dep_car.core.occupancy import FootprintConfig, OccupancyGrid2D
from dep_car.core.planner import DeterministicPlanner
from dep_car.core.safety import evaluate_dynamic, evaluate_static
from dep_car.core.types import Candidate, DynamicTrack, VehicleState
from dep_car.core.types import Gear


def open_grid():
    return OccupancyGrid2D(np.zeros((240, 240), dtype=np.int8), 0.1, (-12.0, -12.0))


def test_swept_footprint_rejects_side_collision():
    grid_data = np.zeros((240, 240), dtype=np.int8)
    point = np.floor((np.asarray([1.0, 0.20]) - np.asarray([-12.0, -12.0])) / 0.1).astype(int)
    grid_data[point[1], point[0]] = 100
    grid = OccupancyGrid2D(grid_data, 0.1, (-12.0, -12.0))
    trajectory = np.asarray([[0.0, 0.0, 0.0, 0.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0, 1.0, 0.0]])
    candidate = Candidate(0, 1.0, 0.0, 1.0, trajectory)
    evaluate_static(candidate, grid, FootprintConfig())
    assert not candidate.feasible
    assert candidate.veto_reason == "static_footprint_collision"


def test_dynamic_hard_veto_cannot_be_rescued_by_score():
    trajectory = np.asarray([[0.0, 0.0, 0.0, 0.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0, 1.0, 0.0]])
    candidate = Candidate(0, 1.0, 0.0, 1.0, trajectory, learned_score=-1000.0)
    evaluate_dynamic(candidate, [DynamicTrack(1, 1.0, 0.0, 0.0, 0.0)])
    assert not candidate.feasible
    assert candidate.veto_reason == "dynamic_reachability_collision"


def test_open_world_planner_selects_executable_candidate():
    result = DeterministicPlanner().plan(VehicleState(), (4.0, 0.0), open_grid())
    assert result.executable
    assert result.selected is not None
    assert result.retime_factor == 1.0


def test_open_world_reverse_recovery_exposes_full_candidate_bank():
    result = DeterministicPlanner().plan(
        VehicleState(), (-0.6, 0.0), open_grid(), requested_gear=Gear.REVERSE,
    )
    assert result.executable
    assert len(result.candidates) == 15
    assert all(candidate.gear == Gear.REVERSE for candidate in result.candidates)


def test_static_wall_blocks_all_candidates():
    data = np.zeros((240, 240), dtype=np.int8)
    x_cell = int(np.floor((0.7 + 12.0) / 0.1))
    data[:, x_cell : x_cell + 2] = 100
    grid = OccupancyGrid2D(data, 0.1, (-12.0, -12.0))
    result = DeterministicPlanner().plan(VehicleState(), (4.0, 0.0), grid)
    assert not result.executable
    assert result.blocked_by_static
