"""Non-holonomic 3x5 primitive lattice and bicycle-model rollout."""

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np

from .state_contract import ACCELERATION_LIMIT_MPS2, DECELERATION_LIMIT_MPS2
from .types import Candidate, Gear, VehicleState
from .vehicle import PLANNER_ROLLOUT_WHEELBASE_M, STEERING_OPERATING_LIMIT_RAD


@dataclass(frozen=True)
class LatticeConfig:
    wheelbase: float = PLANNER_ROLLOUT_WHEELBASE_M
    speed_anchors: Sequence[float] = (0.6, 1.2, 2.0)
    reverse_speed_anchors: Sequence[float] = (0.20, 0.35, 0.50)
    steering_anchors: Sequence[float] = (-0.52, -0.26, 0.0, 0.26, 0.52)
    # One second is the spatial safety horizon at the 10 Hz replanning rate.
    # The former 2 s primitives were longer than several scaled-Urban-Car
    # corridor manoeuvres and rejected otherwise valid compound turns.
    horizon: float = 1.0
    dt: float = 0.1
    max_steering: float = STEERING_OPERATING_LIMIT_RAD
    max_steering_rate: float = 0.75
    max_acceleration: float = ACCELERATION_LIMIT_MPS2
    max_deceleration: float = DECELERATION_LIMIT_MPS2


class AckermannLattice:
    """Generate speed x steering candidates with a kinematic bicycle model.

    Steering follows REP-103: positive is a left turn.  Every trajectory row is
    ``[t, x, y, yaw, speed, steering]`` in the vehicle's starting frame.
    """

    def __init__(self, config: LatticeConfig = LatticeConfig()):
        self.config = config
        if len(config.speed_anchors) != 3 or len(config.reverse_speed_anchors) != 3 or len(config.steering_anchors) != 5:
            raise ValueError("DE-P-Car V1 requires exactly a 3 speed x 5 steering lattice")

    def rollout(
        self,
        state: VehicleState,
        target_speed: float,
        target_steering: float,
        duration: float = None,
    ) -> np.ndarray:
        cfg = self.config
        duration = cfg.horizon if duration is None else float(duration)
        count = max(2, int(np.ceil(duration / cfg.dt)) + 1)
        times = np.linspace(0.0, duration, count, dtype=np.float64)
        out = np.zeros((count, 6), dtype=np.float64)
        out[0] = [0.0, 0.0, 0.0, 0.0, state.speed, state.steering]
        steer_target = float(np.clip(target_steering, -cfg.max_steering, cfg.max_steering))
        gear_sign = 1.0 if target_speed >= 0.0 else -1.0

        for index in range(1, count):
            step = times[index] - times[index - 1]
            _, x, y, yaw, speed, steering = out[index - 1]
            # Acceleration/deceleration are magnitudes relative to the selected
            # drive direction.  Applying their asymmetric limits directly to
            # signed world-frame dv/dt would incorrectly let reverse accelerate
            # at the braking limit and brake only at the acceleration limit.
            directed_acceleration = np.clip(
                gear_sign * (target_speed - speed) / max(step, 1e-6),
                -cfg.max_deceleration,
                cfg.max_acceleration,
            )
            acceleration = gear_sign * directed_acceleration
            steering_rate = np.clip(
                (steer_target - steering) / max(step, 1e-6),
                -cfg.max_steering_rate,
                cfg.max_steering_rate,
            )
            next_speed = speed + acceleration * step
            next_steering = float(np.clip(steering + steering_rate * step, -cfg.max_steering, cfg.max_steering))
            mean_speed = 0.5 * (speed + next_speed)
            mean_steering = 0.5 * (steering + next_steering)
            yaw_rate = mean_speed * np.tan(mean_steering) / cfg.wheelbase
            next_yaw = yaw + yaw_rate * step
            mid_yaw = yaw + 0.5 * yaw_rate * step
            out[index] = [
                times[index],
                x + mean_speed * np.cos(mid_yaw) * step,
                y + mean_speed * np.sin(mid_yaw) * step,
                next_yaw,
                next_speed,
                next_steering,
            ]
        return out

    def generate(
        self,
        state: VehicleState,
        speed_offsets: Iterable[float] = None,
        steering_offsets: Iterable[float] = None,
        learned_scores: Iterable[float] = None,
        gear: Gear = Gear.FORWARD,
        speed_scale: float = 1.0,
        duration_scale: float = 1.0,
    ) -> List[Candidate]:
        speed_offsets = np.zeros(15) if speed_offsets is None else np.asarray(list(speed_offsets), dtype=float)
        steering_offsets = np.zeros(15) if steering_offsets is None else np.asarray(list(steering_offsets), dtype=float)
        learned_scores = np.zeros(15) if learned_scores is None else np.asarray(list(learned_scores), dtype=float)
        if any(values.shape != (15,) for values in (speed_offsets, steering_offsets, learned_scores)):
            raise ValueError("learned lattice outputs must contain 15 values")

        gear = Gear.require_drive(gear)
        anchors = self.config.speed_anchors if gear == Gear.FORWARD else self.config.reverse_speed_anchors
        candidates = []
        candidate_id = 0
        for speed in anchors:
            for steering in self.config.steering_anchors:
                speed_magnitude = max(0.0, (float(speed) + speed_offsets[candidate_id]) * speed_scale)
                target_speed = float(gear) * speed_magnitude
                target_steering = float(steering) + steering_offsets[candidate_id]
                duration = self.config.horizon * duration_scale
                candidates.append(
                    Candidate(
                        candidate_id=candidate_id,
                        speed_anchor=target_speed,
                        steering_anchor=target_steering,
                        duration=duration,
                        trajectory=self.rollout(state, target_speed, target_steering, duration),
                        gear=gear,
                        retime_factor=duration_scale,
                        learned_score=float(learned_scores[candidate_id]),
                    )
                )
                candidate_id += 1
        return candidates
