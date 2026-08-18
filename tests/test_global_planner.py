import numpy as np

from dep_car.core.occupancy import OccupancyGrid2D
from dep_car.global_planner.hybrid_astar import HybridAStar
from dep_car.global_planner.topological_astar import TopologicalAStar, corridor_gear_hint
from dep_car.core.types import Gear
from dep_car.runtime.occupancy import RuntimeOccupancyGrid2D


def test_hybrid_astar_finds_open_map_path():
    grid = OccupancyGrid2D(np.zeros((120, 120), dtype=np.int8), 0.1, (-6.0, -6.0))
    path = HybridAStar().plan(grid, (-3.0, 0.0, 0.0), (3.0, 0.0, 0.0))
    assert path is not None
    assert len(path) > 5
    assert abs(path[-1][0] - 3.0) < 0.8


def test_hybrid_astar_preserves_reverse_gear_in_path_contract():
    grid = OccupancyGrid2D(np.zeros((120, 120), dtype=np.int8), 0.1, (-6.0, -6.0))
    path = HybridAStar().plan(grid, (0.0, 0.0, 0.0), (-1.0, 0.0, 0.0))
    assert path is not None
    assert path[0].gear == Gear.NEUTRAL
    assert any(pose.gear == Gear.REVERSE for pose in path[1:])


def test_narrow_heading_reversal_produces_three_point_gear_sequence():
    resolution = 0.1
    data = np.full((80, 80), 100, dtype=np.int8)
    coordinates = (np.arange(80) + 0.5) * resolution - 4.0
    corridor = (np.abs(coordinates[:, None]) < 0.65) & (np.abs(coordinates[None, :]) < 2.0)
    data[corridor] = 0
    grid = OccupancyGrid2D(data, resolution, (-4.0, -4.0))
    path = HybridAStar().plan(grid, (-0.6, 0.0, 0.0), (0.6, 0.0, np.pi))
    assert path is not None
    sequence = []
    for pose in path[1:]:
        if not sequence or sequence[-1] != pose.gear:
            sequence.append(pose.gear)
    assert sequence[:3] == [Gear.FORWARD, Gear.REVERSE, Gear.FORWARD]


def test_dead_end_escape_reverses_through_the_only_opening():
    resolution, origin = 0.1, -6.0
    data = np.zeros((120, 120), dtype=np.int8)
    coordinates = (np.arange(120) + 0.5) * resolution + origin
    x, y = np.meshgrid(coordinates, coordinates)
    side_walls = (x > 0.0) & (x < 2.4) & (np.abs(y) > 0.55) & (np.abs(y) < 0.80)
    end_wall = (x > 2.15) & (x < 2.4) & (np.abs(y) < 0.80)
    data[side_walls | end_wall] = 100
    grid = OccupancyGrid2D(data, resolution, (origin, origin))
    path = HybridAStar().plan(grid, (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0))
    assert path is not None
    assert {pose.gear for pose in path[1:]} == {Gear.REVERSE}


def test_goal_precheck_only_rejects_unplaceable_static_endpoints():
    open_grid = OccupancyGrid2D(
        np.zeros((80, 80), dtype=np.int8), 0.1, (-4.0, -4.0)
    )
    planner = HybridAStar()
    valid, reason, clearance = planner.validate_goal_pose(
        open_grid, (0.0, 0.0, 0.0)
    )
    assert valid and reason == "GOAL_VALID" and clearance > 0.0
    valid, reason, _ = planner.validate_goal_pose(open_grid, (20.0, 0.0, 0.0))
    assert not valid and reason == "GOAL_OUTSIDE_MAP"

    tight = np.full((21, 21), 100, dtype=np.int8)
    tight[10, 10] = 0
    tight_grid = OccupancyGrid2D(tight, 0.1, (-1.05, -1.05))
    valid, reason, _ = planner.validate_goal_pose(tight_grid, (0.0, 0.0, 0.0))
    assert not valid and reason == "GOAL_FOOTPRINT_COLLISION"


def test_hybrid_astar_cancels_stale_search_without_changing_path_contract():
    grid = OccupancyGrid2D(
        np.zeros((120, 120), dtype=np.int8), 0.1, (-6.0, -6.0)
    )
    planner = HybridAStar()
    calls = 0

    def canceled():
        nonlocal calls
        calls += 1
        return calls >= 3

    path = planner.plan(
        grid, (-3.0, 0.0, 0.0), (3.0, 0.0, 0.0), cancel_requested=canceled
    )
    assert path is None
    assert planner.last_status == "CANCELED"
    assert planner.last_expansions < planner.config.maximum_expansions


def test_start_validation_distinguishes_invalid_footprint_from_no_motion_space():
    open_grid = OccupancyGrid2D(
        np.zeros((120, 120), dtype=np.int8), 0.1, (-6.0, -6.0)
    )
    planner = HybridAStar()
    valid, reason, clearance, primitives = planner.validate_start_pose(
        open_grid, (0.0, 0.0, 0.0)
    )
    assert valid and reason == "START_VALID"
    assert clearance > 0.0 and primitives == 10

    occupied = np.zeros((120, 120), dtype=np.int8)
    occupied[60, 60] = 100
    invalid_grid = OccupancyGrid2D(occupied, 0.1, (-6.0, -6.0))
    valid, reason, _, primitives = planner.validate_start_pose(
        invalid_grid, (0.05, 0.05, 0.0)
    )
    assert not valid and reason == "START_CELL_OCCUPIED" and primitives == 0

    resolution, origin, size = 0.05, -2.0, 80
    coordinates = (np.arange(size) + 0.5) * resolution + origin
    x, y = np.meshgrid(coordinates, coordinates)
    pocket = np.full((size, size), 100, dtype=np.int8)
    pocket[(np.abs(x) < 0.50) & (np.abs(y) < 0.35)] = 0
    blocked_grid = RuntimeOccupancyGrid2D(
        pocket, resolution, (origin, origin)
    )
    valid, reason, clearance, primitives = planner.validate_start_pose(
        blocked_grid, (0.0, 0.0, 0.0)
    )
    assert not valid and reason == "START_NO_SAFE_PRIMITIVE"
    assert clearance > 0.0 and primitives == 0


def test_topological_corridor_reaches_a_far_goal_behind_without_prescribing_gear():
    grid = OccupancyGrid2D(
        np.zeros((240, 240), dtype=np.int8), 0.1, (-12.0, -12.0)
    )
    planner = TopologicalAStar()
    path = planner.plan(grid, (6.0, 0.0, 0.0), (-8.0, 7.0, np.pi / 2.0))

    assert path is not None
    assert planner.last_status == "SUCCESS"
    assert planner.last_expansions < 240 * 240
    assert np.hypot(path[-1].x + 8.0, path[-1].y - 7.0) < 1.0e-9
    assert {pose.gear for pose in path} == {Gear.NEUTRAL}


def test_corridor_direction_is_only_a_nearby_forward_reverse_hint():
    assert corridor_gear_hint((0.0, 0.0, 0.0), (0.5, 0.1)) == Gear.FORWARD
    assert corridor_gear_hint((0.0, 0.0, 0.0), (-0.5, 0.1)) == Gear.REVERSE
