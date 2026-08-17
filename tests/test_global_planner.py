import numpy as np

from dep_car.core.occupancy import OccupancyGrid2D
from dep_car.global_planner.hybrid_astar import HybridAStar
from dep_car.core.types import Gear


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
