"""Online temporal state for the unified V4 hybrid-sequence policy."""

from collections import deque

import numpy as np


class JointGearHistoryTracker:
    """Reproduce ``DEPCarJointGearHistoryV1`` from measured motion.

    Training grouped history by episode and used observed signed motion.  The
    online view therefore follows odometry, not a requested gear from the
    legacy high-level state machine.  While stopped, the last measured drive
    direction is retained as temporal context and current gear remains neutral.
    """

    schema = "DEPCarJointGearHistoryV1"

    def __init__(self, *, stop_tolerance_mps=0.03, initial_gear=1):
        if float(stop_tolerance_mps) <= 0.0:
            raise ValueError("stop tolerance must be positive")
        if int(initial_gear) not in (-1, 1):
            raise ValueError("initial gear must be forward or reverse")
        self.stop_tolerance_mps = float(stop_tolerance_mps)
        self.previous_gear = int(initial_gear)
        self.shift_age_s = 0.0
        self.forward_progress_m = 0.0
        self.reverse_progress_m = 0.0
        self.last_stamp = None
        self.switches = deque()

    def reset(self, *, stamp=None, initial_gear=1):
        if int(initial_gear) not in (-1, 1):
            raise ValueError("initial gear must be forward or reverse")
        self.previous_gear = int(initial_gear)
        self.shift_age_s = 0.0
        self.forward_progress_m = 0.0
        self.reverse_progress_m = 0.0
        self.last_stamp = None if stamp is None else float(stamp)
        self.switches.clear()

    def observe(self, stamp, signed_speed, *, recovery_mode=False):
        stamp = float(stamp)
        speed = float(signed_speed)
        if not np.isfinite(stamp) or not np.isfinite(speed):
            raise ValueError("gear history observation must be finite")
        dt = (
            0.0
            if self.last_stamp is None or stamp < self.last_stamp
            else min(0.5, max(0.0, stamp - self.last_stamp))
        )
        self.forward_progress_m += max(0.0, speed) * dt
        self.reverse_progress_m += max(0.0, -speed) * dt
        current_gear = (
            1
            if speed > self.stop_tolerance_mps
            else -1
            if speed < -self.stop_tolerance_mps
            else 0
        )
        if current_gear == 0:
            self.shift_age_s += dt
        elif current_gear == self.previous_gear:
            self.shift_age_s += dt
        else:
            self.previous_gear = current_gear
            self.shift_age_s = 0.0
            self.switches.append(stamp)
        while self.switches and stamp - self.switches[0] > 6.0:
            self.switches.popleft()
        self.last_stamp = stamp
        history = np.asarray(
            [
                float(self.previous_gear),
                min(1.0, self.shift_age_s / 5.0),
                min(2.0, self.forward_progress_m / 2.0),
                min(2.0, self.reverse_progress_m / 2.0),
                float(min(4, len(self.switches))),
                1.0 if recovery_mode else 0.0,
            ],
            dtype=np.float32,
        )
        return current_gear, history


__all__ = ["JointGearHistoryTracker"]
