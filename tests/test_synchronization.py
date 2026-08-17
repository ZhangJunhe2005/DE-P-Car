import math

import numpy as np

from dep_car.training.synchronization import bracket, interpolate_angle, interpolate_matrix


def test_bracket_rejects_nearest_only_and_enforces_both_sides():
    entries = [(0.0, "a"), (1.0, "b")]
    assert bracket(entries, 0.5, 0.6) == (entries[0], entries[1])
    assert bracket(entries, 0.5, 0.4) is None
    assert bracket(entries, -0.1, 1.0) is None


def test_angle_and_transform_interpolation_cross_wrap_safely():
    midpoint = interpolate_angle(math.radians(179), math.radians(-179), 0.5)
    assert abs(abs(midpoint) - math.pi) < 1e-6
    first, second = np.eye(4), np.eye(4)
    second[0, 3] = 2.0
    assert np.allclose(interpolate_matrix(first, second, 0.25)[:3, 3], [0.5, 0.0, 0.0])
