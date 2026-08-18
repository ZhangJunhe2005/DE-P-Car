"""P6-only continuous occupancy sampling without changing P4 model authority."""

from typing import Sequence, Tuple

import numpy as np

from dep_car.core.occupancy import (
    FootprintAllowancePolicy,
    FootprintConfig,
    OccupancyGrid2D,
    RUNTIME_FOOTPRINT_ALLOWANCE,
    RUNTIME_HALF_DIAGONAL_MULTIPLIER,
    densify_trajectory_se2,
)


RUNTIME_DISTANCE_FIELD_SAMPLING_SCHEMA = "BilinearCellCentreDistanceV1"
EGO_UNKNOWN_CLEARANCE_SCHEMA = "FiveCircleObservedObstaclePreservingV1"
EGO_UNKNOWN_CLEARANCE_CELL_GUARD = 0.25


def ego_unknown_clearance_mask(
    x_coordinates: np.ndarray,
    y_coordinates: np.ndarray,
    *,
    resolution: float,
    blind_radius: float,
    footprint: FootprintConfig = FootprintConfig(),
) -> np.ndarray:
    """Return unknown cells that may be cleared beneath the current car.

    A rolling lidar grid cannot observe through the vehicle.  Its self-clear
    mask therefore has to contain the *same* five-circle cover used by the P6
    hard veto.  A smaller circular mask can make the stationary car collide
    with unknown cells at its own nose or tail, rejecting every forward and
    reverse primitive before either one moves.

    This function only builds a mask.  The caller must apply it to unknown
    cells exclusively, so an observed obstacle inside the cover is preserved.
    One quarter cell is added beyond the runtime half-diagonal allowance to keep
    the stationary footprint strictly positive under cell-centre distance
    sampling instead of merely tangent to unknown space.  The guard is smaller
    than the shortest micro-primitive displacement, so a car cannot creep
    through entirely unobserved space by repeatedly recentering the grid.
    """

    x = np.asarray(x_coordinates, dtype=np.float64)
    y = np.asarray(y_coordinates, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("ego-clearance coordinates must be one-dimensional")
    if resolution <= 0.0:
        raise ValueError("ego-clearance resolution must be positive")
    if blind_radius < 0.0:
        raise ValueError("ego-clearance blind radius cannot be negative")

    xx, yy = np.meshgrid(x, y)
    mask = xx * xx + yy * yy <= float(blind_radius) ** 2
    grid_allowance = (
        RUNTIME_HALF_DIAGONAL_MULTIPLIER * np.sqrt(2.0) * float(resolution)
    )
    cover_radius = (
        footprint.circle_radius
        + grid_allowance
        + EGO_UNKNOWN_CLEARANCE_CELL_GUARD * float(resolution)
    )
    for offset in footprint.longitudinal_offsets:
        mask |= (xx - float(offset)) ** 2 + yy * yy <= cover_radius ** 2
    return mask


class RuntimeOccupancyGrid2D(OccupancyGrid2D):
    """Online grid with continuous metric queries at actual body positions.

    The trained P4/P5 artifacts remain bound to the frozen core occupancy
    implementation.  Only P6 ROS planning, hard vetoes and scenario-start
    audits instantiate this subclass.
    """

    def sample_distance_field(self, points: np.ndarray) -> np.ndarray:
        values = np.asarray(points, dtype=np.float64)
        if values.ndim < 1 or values.shape[-1] != 2:
            raise ValueError("distance-field points must have shape [...,2]")
        flat = values.reshape((-1, 2))
        cells = (flat - self.origin) / self.resolution
        inside = (
            (cells[:, 0] >= 0.0)
            & (cells[:, 1] >= 0.0)
            & (cells[:, 0] < self.data.shape[1])
            & (cells[:, 1] < self.data.shape[0])
        )
        output = np.zeros(len(flat), dtype=np.float64)
        if not np.any(inside):
            return output.reshape(values.shape[:-1])
        if not np.all(np.isfinite(self._distance_field)):
            output[inside] = np.inf
            return output.reshape(values.shape[:-1])
        sample = cells[inside] - 0.5
        sx = np.clip(sample[:, 0], 0.0, self.data.shape[1] - 1.0)
        sy = np.clip(sample[:, 1], 0.0, self.data.shape[0] - 1.0)
        x0 = np.floor(sx).astype(np.int64)
        y0 = np.floor(sy).astype(np.int64)
        x1 = np.minimum(x0 + 1, self.data.shape[1] - 1)
        y1 = np.minimum(y0 + 1, self.data.shape[0] - 1)
        wx, wy = sx - x0, sy - y0
        field = self._distance_field
        output[inside] = (
            (1.0 - wx) * (1.0 - wy) * field[y0, x0]
            + wx * (1.0 - wy) * field[y0, x1]
            + (1.0 - wx) * wy * field[y1, x0]
            + wx * wy * field[y1, x1]
        )
        return output.reshape(values.shape[:-1])

    def point_clearance(self, point: Sequence[float], max_range: float = 8.0) -> float:
        value = self.sample_distance_field(np.asarray([point], dtype=np.float64))[0]
        return float(min(max_range, value))

    def swept_footprint_clearance(
        self,
        trajectory: np.ndarray,
        footprint: FootprintConfig = FootprintConfig(),
        allowance_policy: FootprintAllowancePolicy = RUNTIME_FOOTPRINT_ALLOWANCE,
    ) -> Tuple[bool, float]:
        # Preserve the frozen training/audit behavior if a formal caller
        # explicitly asks for its one-diagonal policy.
        if allowance_policy is not FootprintAllowancePolicy.RUNTIME_HALF_DIAGONAL:
            return super().swept_footprint_clearance(
                trajectory, footprint, allowance_policy
            )
        trajectory = densify_trajectory_se2(trajectory)
        yaw = trajectory[:, 3]
        headings = np.column_stack((np.cos(yaw), np.sin(yaw)))
        centers = (
            trajectory[:, None, 1:3]
            + headings[:, None, :] * footprint.longitudinal_offsets[None, :, None]
        )
        flat = centers.reshape((-1, 2))
        cells = self.world_to_cell(flat)
        x, y = cells[:, 0], cells[:, 1]
        inside = (
            (x >= 0) & (y >= 0)
            & (x < self.data.shape[1]) & (y < self.data.shape[0])
        )
        if not np.all(inside):
            return False, 0.0
        clearances = (
            self.sample_distance_field(flat)
            - footprint.circle_radius
            - self.fixed_grid_allowance(allowance_policy)
        )
        minimum = float(np.min(clearances))
        return (minimum > 0.0), max(0.0, minimum)
