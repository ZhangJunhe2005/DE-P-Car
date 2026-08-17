"""Point-cloud frame and self-return contracts shared by runtime and datasets."""

from dataclasses import dataclass

import numpy as np

from dep_car.core.vehicle import LENGTH_M, WIDTH_M


@dataclass(frozen=True)
class SelfFilterConfig:
    """Axis-aligned chassis exclusion box for lidar returns on the ego body."""

    length: float = LENGTH_M
    width: float = WIDTH_M
    padding: float = 0.03


@dataclass(frozen=True)
class ObstacleFilterConfig:
    minimum_height: float = 0.05
    maximum_height: float = 1.30
    self_filter: SelfFilterConfig = SelfFilterConfig()


def lidar_environment_mask(points_chassis, config=SelfFilterConfig()):
    """Return points that cannot originate inside the physical ego footprint.

    The box is deliberately the physical body plus a small mesh/timing tolerance,
    not the larger planning safety margin.  A real obstacle inside this box is
    already intersecting the vehicle and cannot be distinguished from self hits.
    """

    points = np.asarray(points_chassis)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError("points_chassis must have shape [N,>=3]")
    half_length = 0.5 * config.length + config.padding
    half_width = 0.5 * config.width + config.padding
    self_return = (np.abs(points[:, 0]) <= half_length) & (np.abs(points[:, 1]) <= half_width)
    return ~self_return


def filter_lidar_self_returns(points_chassis, config=SelfFilterConfig()):
    points = np.asarray(points_chassis)
    return points[lidar_environment_mask(points, config)]


def filter_lidar_obstacles(points_chassis, config=ObstacleFilterConfig()):
    """Select non-self points in the chassis-frame obstacle height band."""

    points = np.asarray(points_chassis)
    mask = lidar_environment_mask(points, config.self_filter)
    mask &= (points[:, 2] >= config.minimum_height) & (points[:, 2] <= config.maximum_height)
    return points[mask]
