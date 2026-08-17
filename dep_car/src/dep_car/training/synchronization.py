"""Deterministic timestamp interpolation helpers used by V2 bag extraction."""

import bisect
import math

import numpy as np


def bracket(entries, stamp, maximum_source_distance):
    """Return the samples bracketing ``stamp`` or ``None`` outside the gate.

    Entries begin with ``(timestamp, ...)``.  An exact sample is returned twice,
    making provenance and interpolation logic uniform.
    """

    if not entries:
        return None
    times = [entry[0] for entry in entries]
    index = bisect.bisect_left(times, stamp)
    if index < len(entries) and abs(entries[index][0] - stamp) <= 1e-9:
        return entries[index], entries[index]
    if index == 0 or index == len(entries):
        return None
    before, after = entries[index - 1], entries[index]
    if max(stamp - before[0], after[0] - stamp) > maximum_source_distance:
        return None
    return before, after


def interpolation_alpha(before_stamp, after_stamp, stamp):
    if after_stamp <= before_stamp:
        return 0.0
    return float(np.clip((stamp - before_stamp) / (after_stamp - before_stamp), 0.0, 1.0))


def interpolate_linear(first, second, alpha):
    return (1.0 - alpha) * np.asarray(first, dtype=np.float64) + alpha * np.asarray(second, dtype=np.float64)


def interpolate_angle(first, second, alpha):
    delta = math.atan2(math.sin(second - first), math.cos(second - first))
    value = first + alpha * delta
    return math.atan2(math.sin(value), math.cos(value))


def quaternion_slerp(first, second, alpha):
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    first /= max(np.linalg.norm(first), 1e-12)
    second /= max(np.linalg.norm(second), 1e-12)
    dot = float(np.dot(first, second))
    if dot < 0.0:
        second, dot = -second, -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        result = interpolate_linear(first, second, alpha)
        return result / max(np.linalg.norm(result), 1e-12)
    theta = math.acos(dot)
    return (math.sin((1.0 - alpha) * theta) * first + math.sin(alpha * theta) * second) / math.sin(theta)


def matrix_to_quaternion(matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    trace = float(np.trace(matrix[:3, :3]))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        return np.asarray([
            (matrix[2, 1] - matrix[1, 2]) / scale,
            (matrix[0, 2] - matrix[2, 0]) / scale,
            (matrix[1, 0] - matrix[0, 1]) / scale,
            0.25 * scale,
        ])
    diagonal = np.diag(matrix[:3, :3])
    index = int(np.argmax(diagonal))
    if index == 0:
        scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
        return np.asarray([0.25 * scale, (matrix[0, 1] + matrix[1, 0]) / scale, (matrix[0, 2] + matrix[2, 0]) / scale, (matrix[2, 1] - matrix[1, 2]) / scale])
    if index == 1:
        scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
        return np.asarray([(matrix[0, 1] + matrix[1, 0]) / scale, 0.25 * scale, (matrix[1, 2] + matrix[2, 1]) / scale, (matrix[0, 2] - matrix[2, 0]) / scale])
    scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
    return np.asarray([(matrix[0, 2] + matrix[2, 0]) / scale, (matrix[1, 2] + matrix[2, 1]) / scale, 0.25 * scale, (matrix[1, 0] - matrix[0, 1]) / scale])


def quaternion_to_matrix(quaternion):
    x, y, z, w = np.asarray(quaternion, dtype=np.float64)
    output = np.eye(4, dtype=np.float64)
    output[:3, :3] = (
        (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
        (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
        (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
    )
    return output


def interpolate_matrix(first, second, alpha):
    output = quaternion_to_matrix(quaternion_slerp(matrix_to_quaternion(first), matrix_to_quaternion(second), alpha))
    output[:3, 3] = interpolate_linear(np.asarray(first)[:3, 3], np.asarray(second)[:3, 3], alpha)
    return output.astype(np.float32)
