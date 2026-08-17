"""Deterministic 360-degree LiDAR bird's-eye-view representation."""

import hashlib
import json
from dataclasses import dataclass
from dataclasses import asdict

import numpy as np


BEV_CHANNELS = (
    "occupancy",
    "log_density",
    "minimum_height",
    "maximum_height",
    "nearest_range",
    "observed",
)


@dataclass(frozen=True)
class LidarBEVConfig:
    extent: float = 8.0
    resolution: float = 0.10
    minimum_height: float = -0.50
    maximum_height: float = 1.50
    density_clip: float = 32.0

    @property
    def size(self):
        return int(round(2.0 * self.extent / self.resolution))


def lidar_bev_preprocessing_contract(config: LidarBEVConfig = LidarBEVConfig(), obstacle_filter=None):
    """Return the canonical, hashed point-cloud-to-BEV preprocessing contract."""

    if obstacle_filter is None:
        from dep_car.perception.pointcloud import ObstacleFilterConfig

        obstacle_filter = ObstacleFilterConfig()
    payload = {
        "schema": "LidarBEVPreprocessingV1",
        "input_frame": "chassis",
        "channels": list(BEV_CHANNELS),
        "bev": asdict(config),
        "obstacle_filter": asdict(obstacle_filter),
        "ego_self_filter_before_bev": True,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload["sha256"] = hashlib.sha256(encoded).hexdigest()
    return payload


def _bresenham(x0, y0, x1, y1):
    points = []
    dx, sx = abs(x1 - x0), 1 if x0 < x1 else -1
    dy, sy = -abs(y1 - y0), 1 if y0 < y1 else -1
    error = dx + dy
    while True:
        points.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        twice = 2 * error
        if twice >= dy:
            error += dy
            x0 += sx
        if twice <= dx:
            error += dx
            y0 += sy
    return points


def build_lidar_bev(points, config: LidarBEVConfig = LidarBEVConfig()):
    """Return six deterministic channels from points expressed in chassis.

    Input columns must begin with x, y, z. Additional intensity/ring columns
    are preserved in the raw dataset but are not required for this V1 BEV.
    """

    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError("LiDAR points must have shape [N,>=3]")
    size = config.size
    if size <= 0 or config.resolution <= 0.0:
        raise ValueError("BEV extent and resolution must be positive")
    output = np.zeros((len(BEV_CHANNELS), size, size), dtype=np.float32)
    if not len(points):
        return output

    finite = np.all(np.isfinite(points[:, :3]), axis=1)
    inside = (
        finite
        & (np.abs(points[:, 0]) < config.extent)
        & (np.abs(points[:, 1]) < config.extent)
        & (points[:, 2] >= config.minimum_height)
        & (points[:, 2] <= config.maximum_height)
    )
    points = points[inside]
    if not len(points):
        return output

    cells = np.floor((points[:, :2] + config.extent) / config.resolution).astype(np.int32)
    x, y = cells[:, 0], cells[:, 1]
    counts = np.zeros((size, size), dtype=np.float32)
    minimum = np.full((size, size), np.inf, dtype=np.float32)
    maximum = np.full((size, size), -np.inf, dtype=np.float32)
    nearest = np.full((size, size), np.inf, dtype=np.float32)
    np.add.at(counts, (y, x), 1.0)
    np.minimum.at(minimum, (y, x), points[:, 2])
    np.maximum.at(maximum, (y, x), points[:, 2])
    np.minimum.at(nearest, (y, x), np.linalg.norm(points[:, :2], axis=1))

    occupied = counts > 0.0
    output[0, occupied] = 1.0
    output[1] = np.log1p(np.minimum(counts, config.density_clip)) / np.log1p(config.density_clip)
    output[2, occupied] = np.clip(minimum[occupied], config.minimum_height, config.maximum_height)
    output[3, occupied] = np.clip(maximum[occupied], config.minimum_height, config.maximum_height)
    output[4, occupied] = np.minimum(nearest[occupied] / config.extent, 1.0)

    center = size // 2
    observed = output[5]
    for cell_x, cell_y in np.unique(cells, axis=0):
        ray = _bresenham(center, center, int(cell_x), int(cell_y))
        ray_x, ray_y = zip(*ray)
        observed[np.asarray(ray_y), np.asarray(ray_x)] = 1.0
    return output
