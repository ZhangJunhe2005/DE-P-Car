"""Small timestamp buffers for P6 online sensor alignment.

P3 samples are anchored to a LiDAR frame and use nearest depth plus bracketed
vehicle-state sources.  P6 must reproduce that temporal contract instead of
comparing whichever messages happened to be latest at the policy timer tick.
"""

import bisect
from collections import deque

import numpy as np


class StampedHistory:
    """Bounded monotonic history that recovers cleanly from a clock reset."""

    def __init__(self, maximum_length=32):
        if int(maximum_length) < 2:
            raise ValueError("history maximum_length must be at least two")
        self._entries = deque(maxlen=int(maximum_length))

    def append(self, stamp, value):
        stamp = float(stamp)
        if not np.isfinite(stamp):
            raise ValueError("sensor timestamp must be finite")
        if self._entries and stamp < self._entries[-1][0] - 1.0e-9:
            self._entries.clear()
        if self._entries and abs(stamp - self._entries[-1][0]) <= 1.0e-9:
            self._entries[-1] = (stamp, value)
        else:
            self._entries.append((stamp, value))

    def snapshot(self):
        return tuple(self._entries)


def nearest(entries, anchor_stamp, maximum_distance):
    """Return ``(value, source_distance)`` for the nearest accepted sample."""

    entries = tuple(entries)
    if not entries:
        return None
    anchor_stamp = float(anchor_stamp)
    times = [entry[0] for entry in entries]
    index = bisect.bisect_left(times, anchor_stamp)
    candidates = []
    if index < len(entries):
        candidates.append(entries[index])
    if index > 0:
        candidates.append(entries[index - 1])
    stamp, value = min(candidates, key=lambda entry: (abs(entry[0] - anchor_stamp), entry[0]))
    distance = abs(float(stamp) - anchor_stamp)
    if distance > float(maximum_distance) + 1.0e-9:
        return None
    return value, distance


def interpolated(entries, anchor_stamp, maximum_source_distance):
    """Linearly interpolate numeric values bracketing the anchor timestamp."""

    entries = tuple(entries)
    if not entries:
        return None
    anchor_stamp = float(anchor_stamp)
    times = [entry[0] for entry in entries]
    index = bisect.bisect_left(times, anchor_stamp)
    if index < len(entries) and abs(times[index] - anchor_stamp) <= 1.0e-9:
        return np.asarray(entries[index][1], dtype=np.float64), 0.0
    if index == 0 or index == len(entries):
        return None
    before, after = entries[index - 1], entries[index]
    source_distance = max(anchor_stamp - before[0], after[0] - anchor_stamp)
    if source_distance > float(maximum_source_distance) + 1.0e-9:
        return None
    interval = after[0] - before[0]
    alpha = 0.0 if interval <= 0.0 else (anchor_stamp - before[0]) / interval
    first = np.asarray(before[1], dtype=np.float64)
    second = np.asarray(after[1], dtype=np.float64)
    if first.shape != second.shape or not np.all(np.isfinite(first)) or not np.all(np.isfinite(second)):
        raise ValueError("interpolation sources must have matching finite shapes")
    return (1.0 - alpha) * first + alpha * second, float(source_distance)


def newest_synchronized_anchor(
    anchors,
    interpolated_sources,
    nearest_sources=None,
):
    """Return the newest anchor for which every source can be synchronized.

    Online callbacks commonly receive the newest anchor just before the
    bracketing state sample.  Rejecting that one frame is correct, but it must
    not hide an immediately preceding anchor that already has the complete
    P3/P5 temporal evidence.  Source mappings contain
    ``name: (entries, tolerance)`` pairs.
    """

    anchors = tuple(anchors)
    nearest_sources = nearest_sources or {}
    for anchor_stamp, anchor_value in reversed(anchors):
        matches = {
            name: interpolated(entries, anchor_stamp, tolerance)
            for name, (entries, tolerance) in interpolated_sources.items()
        }
        matches.update({
            name: nearest(entries, anchor_stamp, tolerance)
            for name, (entries, tolerance) in nearest_sources.items()
        })
        if all(value is not None for value in matches.values()):
            return float(anchor_stamp), anchor_value, matches
    return None


__all__ = [
    "StampedHistory",
    "interpolated",
    "nearest",
    "newest_synchronized_anchor",
]
