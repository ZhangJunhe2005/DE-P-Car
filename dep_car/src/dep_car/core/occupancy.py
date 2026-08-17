"""2-D occupancy authority and swept Urban Car footprint checks."""

from dataclasses import dataclass
from enum import Enum
from typing import Sequence, Tuple

import cv2
import numpy as np

from .vehicle import FOOTPRINT_SAFETY_MARGIN_M, LENGTH_M, WIDTH_M


# Frozen continuous-sweep contract.  The formal learned rollout has at most
# 0.125 s between its eleven output rows (1.25 s maximum duration), 2.5 m/s
# longitudinal speed and 1.8182 1/m calibrated curvature.  Sixteen substeps
# bound one interpolated interval to roughly 0.0196 m translation and 0.0356 rad
# yaw.  Including rotation of the outer footprint circle, its centre moves less
# than 0.027 m per substep, below half a diagonal of the finest frozen 0.05 m
# runtime grid.  Network output remains eleven rows; densification is a safety
# query operation only.
SWEPT_INTERPOLATION_SCHEMA = "PiecewiseLinearSE2SubstepsV1"
SWEPT_SUBSTEPS_PER_SEGMENT = 16
FIVE_CIRCLE_FOOTPRINT_SCHEMA = "FiveCircleContinuousSweptFootprintV3"

# Fixed grid-cell allowances.  Runtime uses the geometrically sufficient half
# diagonal, while training and offline authority audits deliberately use one
# full diagonal as an additional conservative guard.  Callers select only one
# of these named policies; accepting an arbitrary numeric multiplier here
# would make dataset authority dependent on an untracked CLI value.
FOOTPRINT_ALLOWANCE_SCHEMA = "FixedGridCellDiagonalAllowanceV1"
RUNTIME_HALF_DIAGONAL_MULTIPLIER = 0.5
TRAINING_ONE_DIAGONAL_MULTIPLIER = 1.0


class FootprintAllowancePolicy(str, Enum):
    """Closed set of clearance allowances in the frozen footprint contract."""

    RUNTIME_HALF_DIAGONAL = "runtime_half_diagonal"
    TRAINING_ONE_DIAGONAL = "training_one_diagonal"


RUNTIME_FOOTPRINT_ALLOWANCE = FootprintAllowancePolicy.RUNTIME_HALF_DIAGONAL
TRAINING_FOOTPRINT_ALLOWANCE = FootprintAllowancePolicy.TRAINING_ONE_DIAGONAL


def densify_trajectory_se2(
    trajectory: np.ndarray,
    substeps: int = SWEPT_SUBSTEPS_PER_SEGMENT,
) -> np.ndarray:
    """Densify ``[t,x,y,yaw,...]`` rows with shortest-angle SE(2) interpolation.

    Each source interval is divided into the same frozen number of substeps.
    The final source row is appended once, so an ``N`` row trajectory produces
    ``(N - 1) * substeps + 1`` rows.  All non-yaw fields are linearly
    interpolated; yaw follows its shortest wrapped difference.
    """

    values = np.asarray(trajectory, dtype=float)
    if values.ndim != 2 or values.shape[1] < 4 or not len(values):
        raise ValueError("trajectory must have shape [N,>=4]")
    if not np.all(np.isfinite(values)):
        raise ValueError("trajectory must contain only finite values")
    if isinstance(substeps, bool) or not isinstance(substeps, (int, np.integer)) or substeps < 1:
        raise ValueError("swept interpolation substeps must be a positive integer")
    if len(values) == 1:
        return values.copy()

    alpha = np.arange(int(substeps), dtype=values.dtype) / float(substeps)
    start = values[:-1, None, :]
    delta = values[1:, None, :] - start
    dense = start + alpha[None, :, None] * delta
    yaw_delta = np.arctan2(
        np.sin(values[1:, 3] - values[:-1, 3]),
        np.cos(values[1:, 3] - values[:-1, 3]),
    )
    dense[..., 3] = values[:-1, None, 3] + alpha[None, :] * yaw_delta[:, None]
    return np.concatenate((dense.reshape((-1, values.shape[1])), values[-1:]), axis=0)


@dataclass(frozen=True)
class FootprintConfig:
    length: float = LENGTH_M
    width: float = WIDTH_M
    safety_margin: float = FOOTPRINT_SAFETY_MARGIN_M
    circle_count: int = 5

    def __post_init__(self):
        if self.length <= 0.0 or self.width <= 0.0:
            raise ValueError("footprint length and width must be positive")
        if self.safety_margin < 0.0:
            raise ValueError("footprint safety margin cannot be negative")
        if isinstance(self.circle_count, bool) or not isinstance(self.circle_count, (int, np.integer)) or self.circle_count < 1:
            raise ValueError("footprint circle_count must be a positive integer")

    @property
    def longitudinal_strip_length(self) -> float:
        """Length represented by each circle in the conservative cover."""

        return self.length / self.circle_count

    @property
    def circle_radius(self) -> float:
        """Radius of a circle covering one longitudinal body strip.

        The rectangular body is partitioned into ``circle_count`` equal
        longitudinal strips and a circle is placed at every strip centre.  A
        strip corner is at most ``hypot(strip_length / 2, width / 2)`` from
        its centre.  Adding the safety margin therefore covers the complete
        Minkowski expansion of the physical rectangle, rather than merely its
        centre line.
        """

        half_strip = 0.5 * self.longitudinal_strip_length
        return float(np.hypot(half_strip, 0.5 * self.width) + self.safety_margin)

    @property
    def longitudinal_offsets(self) -> np.ndarray:
        """Return non-degenerate strip-centre offsets in the body x-axis."""

        strip = self.longitudinal_strip_length
        first = -0.5 * self.length + 0.5 * strip
        return first + strip * np.arange(self.circle_count, dtype=np.float64)


class OccupancyGrid2D:
    """ROS-compatible occupancy grid with unknown space treated as occupied."""

    def __init__(
        self,
        data: np.ndarray,
        resolution: float,
        origin: Tuple[float, float] = (0.0, 0.0),
        occupied_threshold: int = 50,
        unknown_is_occupied: bool = True,
    ):
        array = np.asarray(data)
        if array.ndim != 2:
            raise ValueError("occupancy data must be a 2-D array")
        if resolution <= 0:
            raise ValueError("resolution must be positive")
        self.data = array
        self.resolution = float(resolution)
        self.origin = np.asarray(origin, dtype=float)
        self.occupied_threshold = occupied_threshold
        self.unknown_is_occupied = unknown_is_occupied
        occupied = np.argwhere((self.data >= self.occupied_threshold) | ((self.data < 0) & self.unknown_is_occupied))
        occupied_mask = (self.data >= self.occupied_threshold) | ((self.data < 0) & self.unknown_is_occupied)
        self._distance_field = (
            cv2.distanceTransform((~occupied_mask).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
            * self.resolution
            if np.any(occupied_mask)
            else np.full(self.data.shape, np.inf, dtype=np.float32)
        )
        self._occupied_centers = (
            np.column_stack((occupied[:, 1], occupied[:, 0])) * self.resolution + self.origin
            if occupied.size else np.empty((0, 2), dtype=float)
        )

    def world_to_cell(self, points: np.ndarray) -> np.ndarray:
        return np.floor((np.asarray(points) - self.origin) / self.resolution).astype(np.int64)

    def is_occupied(self, points: np.ndarray) -> np.ndarray:
        cells = self.world_to_cell(points)
        x = cells[..., 0]
        y = cells[..., 1]
        inside = (x >= 0) & (y >= 0) & (x < self.data.shape[1]) & (y < self.data.shape[0])
        result = np.ones(inside.shape, dtype=bool)
        values = np.full(inside.shape, -1, dtype=self.data.dtype)
        values[inside] = self.data[y[inside], x[inside]]
        result[inside] = values[inside] >= self.occupied_threshold
        if not self.unknown_is_occupied:
            result[inside & (values < 0)] = False
        return result

    def point_clearance(self, point: Sequence[float], max_range: float = 8.0) -> float:
        cell = self.world_to_cell(np.asarray([point]))[0]
        x, y = int(cell[0]), int(cell[1])
        if x < 0 or y < 0 or x >= self.data.shape[1] or y >= self.data.shape[0]:
            return 0.0
        return float(min(max_range, self._distance_field[y, x]))

    def fixed_grid_allowance(
        self,
        policy: FootprintAllowancePolicy = RUNTIME_FOOTPRINT_ALLOWANCE,
    ) -> float:
        """Return the metric allowance for one of the two frozen policies.

        Requiring the enum instance (rather than coercing strings or accepting
        a numeric multiplier) keeps runtime, training and offline audits on a
        small, versioned set of semantics.
        """

        if not isinstance(policy, FootprintAllowancePolicy):
            raise TypeError("policy must be a FootprintAllowancePolicy")
        multiplier = {
            FootprintAllowancePolicy.RUNTIME_HALF_DIAGONAL: RUNTIME_HALF_DIAGONAL_MULTIPLIER,
            FootprintAllowancePolicy.TRAINING_ONE_DIAGONAL: TRAINING_ONE_DIAGONAL_MULTIPLIER,
        }[policy]
        return float(multiplier * np.sqrt(2.0) * self.resolution)

    def swept_footprint_clearance(
        self,
        trajectory: np.ndarray,
        footprint: FootprintConfig = FootprintConfig(),
        allowance_policy: FootprintAllowancePolicy = RUNTIME_FOOTPRINT_ALLOWANCE,
    ) -> Tuple[bool, float]:
        trajectory = densify_trajectory_se2(trajectory)
        yaw = trajectory[:, 3]
        headings = np.column_stack((np.cos(yaw), np.sin(yaw)))
        centers = trajectory[:, None, 1:3] + headings[:, None, :] * footprint.longitudinal_offsets[None, :, None]
        flat = centers.reshape((-1, 2))
        cells = self.world_to_cell(flat)
        x, y = cells[:, 0], cells[:, 1]
        inside = (x >= 0) & (y >= 0) & (x < self.data.shape[1]) & (y < self.data.shape[0])
        if not np.all(inside):
            return False, 0.0
        # The default runtime contract inflates by half a cell diagonal.  P5
        # training and offline dataset audits explicitly select the frozen
        # one-diagonal policy when they require the stronger authority gate.
        grid_allowance = self.fixed_grid_allowance(allowance_policy)
        clearances = self._distance_field[y, x] - footprint.circle_radius - grid_allowance
        minimum = float(np.min(clearances))
        return (minimum > 0.0), max(0.0, minimum)
