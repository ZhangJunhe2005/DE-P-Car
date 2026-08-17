import numpy as np
import pytest

from dep_car.core.occupancy import (
    FOOTPRINT_ALLOWANCE_SCHEMA,
    FootprintConfig,
    FootprintAllowancePolicy,
    OccupancyGrid2D,
    RUNTIME_FOOTPRINT_ALLOWANCE,
    RUNTIME_HALF_DIAGONAL_MULTIPLIER,
    SWEPT_INTERPOLATION_SCHEMA,
    SWEPT_SUBSTEPS_PER_SEGMENT,
    TRAINING_FOOTPRINT_ALLOWANCE,
    TRAINING_ONE_DIAGONAL_MULTIPLIER,
    densify_trajectory_se2,
)


def obstacle_grid(*points, resolution=0.01, extent=2.0):
    size = int(round(2.0 * extent / resolution))
    data = np.zeros((size, size), dtype=np.int8)
    origin = np.asarray((-extent, -extent), dtype=np.float64)
    for point in points:
        cell = np.floor((np.asarray(point, dtype=np.float64) - origin) / resolution).astype(int)
        data[cell[1], cell[0]] = 100
    return OccupancyGrid2D(data, resolution, tuple(origin))


def stationary_trajectory():
    return np.asarray([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float64)


def test_scaled_urban_car_uses_a_non_degenerate_conservative_five_circle_cover():
    footprint = FootprintConfig()
    offsets = footprint.longitudinal_offsets

    assert offsets.shape == (5,)
    assert len(np.unique(offsets)) >= 3
    assert np.all(np.diff(offsets) > 0.0)
    assert offsets == pytest.approx(-offsets[::-1])
    assert offsets[0] < 0.0 < offsets[-1]

    # Every corner of every longitudinal strip must lie inside its circle
    # before the independent safety margin is added.
    half_strip = 0.5 * footprint.longitudinal_strip_length
    physical_cover_radius = footprint.circle_radius - footprint.safety_margin
    assert physical_cover_radius == pytest.approx(np.hypot(half_strip, 0.5 * footprint.width))
    for offset in offsets:
        for dx in (-half_strip, half_strip):
            for y in (-0.5 * footprint.width, 0.5 * footprint.width):
                assert np.hypot(dx, y) <= physical_cover_radius + 1e-12


@pytest.mark.parametrize(
    "obstacle",
    (
        "front",
        "rear",
        "front_left",
        "front_right",
        "rear_left",
        "rear_right",
    ),
)
def test_stationary_footprint_rejects_front_rear_and_corner_obstacles(obstacle):
    footprint = FootprintConfig()
    half_length = 0.5 * footprint.length
    half_width = 0.5 * footprint.width
    margin_probe = 0.5 * footprint.safety_margin
    points = {
        "front": (half_length + margin_probe, 0.0),
        "rear": (-half_length - margin_probe, 0.0),
        "front_left": (half_length, half_width + margin_probe),
        "front_right": (half_length, -half_width - margin_probe),
        "rear_left": (-half_length, half_width + margin_probe),
        "rear_right": (-half_length, -half_width - margin_probe),
    }

    safe, clearance = obstacle_grid(points[obstacle]).swept_footprint_clearance(
        stationary_trajectory(), footprint
    )

    assert not safe
    assert clearance == 0.0


def test_swept_footprint_checks_a_collision_at_a_middle_trajectory_pose():
    footprint = FootprintConfig()
    obstacle = (
        0.5 * footprint.length,
        0.5 * footprint.width + 0.5 * footprint.safety_margin,
    )
    trajectory = np.asarray(
        [
            [0.0, -1.0, 0.0, 0.0, 1.0, 0.0],
            [0.5, 0.0, 0.0, 0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )

    safe, clearance = obstacle_grid(obstacle).swept_footprint_clearance(trajectory, footprint)

    assert not safe
    assert clearance == 0.0


def test_frozen_continuous_sweep_rejects_obstacle_between_source_rows():
    assert SWEPT_INTERPOLATION_SCHEMA == "PiecewiseLinearSE2SubstepsV1"
    assert SWEPT_SUBSTEPS_PER_SEGMENT == 16
    footprint = FootprintConfig(
        length=0.01, width=0.01, safety_margin=0.0, circle_count=1
    )
    trajectory = np.asarray(
        (
            (0.0, -0.50, 0.0, 0.0, 1.0, 0.0),
            (1.0, 0.50, 0.0, 0.0, 1.0, 0.0),
        ),
        dtype=np.float64,
    )
    dense = densify_trajectory_se2(trajectory)
    assert dense.shape == (SWEPT_SUBSTEPS_PER_SEGMENT + 1, 6)
    np.testing.assert_allclose(dense[SWEPT_SUBSTEPS_PER_SEGMENT // 2, 1:3], 0.0)

    safe, clearance = obstacle_grid((0.0, 0.0)).swept_footprint_clearance(
        trajectory, footprint
    )
    assert not safe
    assert clearance == 0.0


def test_numpy_continuous_sweep_interpolates_yaw_on_the_short_arc():
    trajectory = np.asarray(
        (
            (0.0, 0.0, 0.0, np.deg2rad(179.0)),
            (1.0, 1.0, 0.0, np.deg2rad(-179.0)),
        )
    )
    dense = densify_trajectory_se2(trajectory)
    midpoint = dense[SWEPT_SUBSTEPS_PER_SEGMENT // 2, 3]
    assert abs(abs(midpoint) - np.pi) < 1.0e-12


def test_grid_allowance_is_a_closed_two_policy_contract_with_runtime_default():
    resolution = 0.05
    grid = obstacle_grid(resolution=resolution)
    expected_diagonal = np.sqrt(2.0) * resolution

    assert FOOTPRINT_ALLOWANCE_SCHEMA == "FixedGridCellDiagonalAllowanceV1"
    assert RUNTIME_HALF_DIAGONAL_MULTIPLIER == 0.5
    assert TRAINING_ONE_DIAGONAL_MULTIPLIER == 1.0
    assert RUNTIME_FOOTPRINT_ALLOWANCE is FootprintAllowancePolicy.RUNTIME_HALF_DIAGONAL
    assert TRAINING_FOOTPRINT_ALLOWANCE is FootprintAllowancePolicy.TRAINING_ONE_DIAGONAL
    assert grid.fixed_grid_allowance() == pytest.approx(0.5 * expected_diagonal)
    assert grid.fixed_grid_allowance(TRAINING_FOOTPRINT_ALLOWANCE) == pytest.approx(
        expected_diagonal
    )

    with pytest.raises(TypeError, match="FootprintAllowancePolicy"):
        grid.fixed_grid_allowance("training_one_diagonal")
    with pytest.raises(TypeError, match="FootprintAllowancePolicy"):
        grid.swept_footprint_clearance(
            stationary_trajectory(), allowance_policy=0.75
        )
