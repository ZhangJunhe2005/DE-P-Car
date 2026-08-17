"""Convert a VLP-16 point cloud to a deterministic 16 x N range image."""

import numpy as np


def build_range_image(points_xyz, azimuth_bins=440, channels=16, min_range=0.9, max_range=40.0):
    points = np.asarray(points_xyz, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_xyz must have shape [N,3]")
    finite = np.all(np.isfinite(points), axis=1)
    points = points[finite]
    ranges = np.linalg.norm(points, axis=1)
    valid = (ranges >= min_range) & (ranges <= max_range)
    points, ranges = points[valid], ranges[valid]
    image = np.full((channels, azimuth_bins), max_range, dtype=np.float32)
    mask = np.zeros((channels, azimuth_bins), dtype=np.float32)
    if points.size == 0:
        return image / max_range, mask
    azimuth = np.arctan2(points[:, 1], points[:, 0])
    elevation = np.arcsin(np.clip(points[:, 2] / ranges, -1.0, 1.0))
    columns = np.floor((azimuth + np.pi) / (2.0 * np.pi) * azimuth_bins).astype(int) % azimuth_bins
    rows = np.rint((elevation - np.deg2rad(-15.0)) / np.deg2rad(30.0) * (channels - 1)).astype(int)
    within = (rows >= 0) & (rows < channels)
    for row, column, value in zip(rows[within], columns[within], ranges[within]):
        if value < image[row, column]:
            image[row, column] = value
            mask[row, column] = 1.0
    return image / max_range, mask

