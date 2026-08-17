"""Differentiable, gear-conditioned Ackermann candidate rollout.

The supervisor owns the discrete gear.  This module only refines the fifteen
speed-major/steering-minor candidates for that already selected gear and never
predicts a gear transition.
"""

from dataclasses import dataclass
from typing import NamedTuple

import torch
from torch import nn

from dep_car.core.state_contract import (
    ACCELERATION_LIMIT_MPS2,
    DECELERATION_LIMIT_MPS2,
    FORWARD_SPEED_LIMIT_MPS,
    REVERSE_SPEED_LIMIT_MPS,
)
from dep_car.core.vehicle import (
    PLANNER_ROLLOUT_WHEELBASE_M,
    STEERING_OPERATING_LIMIT_RAD,
)


# Artifact contracts use a semantic schema independent of the stable Python
# class name.  Revision 2 makes the asymmetric acceleration/braking limits
# explicit in the requested-gear frame (``gear * dv/dt``).
ACKERMANN_ROLLOUT_SCHEMA = "AckermannRolloutV2GearAlignedLongitudinalLimits"
LONGITUDINAL_LIMITS_FRAME = "requested_gear"


class AckermannRolloutOutput(NamedTuple):
    trajectory: torch.Tensor
    controls: torch.Tensor
    residuals: torch.Tensor


@dataclass(frozen=True)
class AckermannRolloutConfig:
    wheelbase_m: float = PLANNER_ROLLOUT_WHEELBASE_M
    horizon_s: float = 1.0
    steps: int = 11
    forward_speed_anchors_mps: tuple = (0.6, 1.2, 2.0)
    reverse_speed_anchors_mps: tuple = (0.20, 0.35, 0.50)
    steering_anchors_rad: tuple = (-0.52, -0.26, 0.0, 0.26, 0.52)
    steering_residual_rad: float = 0.08
    duration_residual_s: float = 0.25
    steering_limit_rad: float = STEERING_OPERATING_LIMIT_RAD
    steering_rate_limit_rad_s: float = 0.75
    forward_speed_limit_mps: float = FORWARD_SPEED_LIMIT_MPS
    reverse_speed_limit_mps: float = REVERSE_SPEED_LIMIT_MPS
    acceleration_limit_mps2: float = ACCELERATION_LIMIT_MPS2
    deceleration_limit_mps2: float = DECELERATION_LIMIT_MPS2
    opposite_motion_reject_mps: float = 0.03


class AckermannRolloutV1(nn.Module):
    """Turn four normalized residuals into fifteen signed trajectories.

    Residual order is ``[steering_mid, steering_end, speed_end, duration]``.
    Output trajectory rows follow the dataset authority order
    ``[t, x, y, yaw, signed_speed, steering]``.
    """

    residual_names = (
        "steering_mid",
        "steering_end",
        "speed_end",
        "duration",
    )

    def __init__(self, config: AckermannRolloutConfig = AckermannRolloutConfig()):
        super().__init__()
        if config.steps < 2:
            raise ValueError("rollout requires at least two time samples")
        if len(config.forward_speed_anchors_mps) != 3 or len(config.reverse_speed_anchors_mps) != 3:
            raise ValueError("DE-P-Car requires three speed anchors per gear")
        if len(config.steering_anchors_rad) != 5:
            raise ValueError("DE-P-Car requires five steering anchors")
        self.config = config
        steering = torch.tensor(config.steering_anchors_rad, dtype=torch.float32)
        self.register_buffer("steering_anchors", steering.repeat(3), persistent=True)
        self.register_buffer(
            "forward_speed_anchors",
            torch.tensor(config.forward_speed_anchors_mps, dtype=torch.float32).repeat_interleave(5),
            persistent=True,
        )
        self.register_buffer(
            "reverse_speed_anchors",
            torch.tensor(config.reverse_speed_anchors_mps, dtype=torch.float32).repeat_interleave(5),
            persistent=True,
        )

    @staticmethod
    def _validate_inputs(vehicle_state, requested_gear, raw_residuals):
        if vehicle_state.ndim != 2 or vehicle_state.shape[1] != 9:
            raise ValueError("vehicle_state must have shape [B,9]")
        if requested_gear.ndim != 1 or requested_gear.shape[0] != vehicle_state.shape[0]:
            raise ValueError("requested_gear must have shape [B]")
        if raw_residuals.ndim != 3 or raw_residuals.shape[1:] != (15, 4):
            raise ValueError("candidate residuals must have shape [B,15,4]")
        if not bool(torch.all((requested_gear == -1) | (requested_gear == 1))):
            raise ValueError("requested_gear must contain only -1 or +1")

    def bound_residuals(self, raw_residuals, requested_gear):
        """Return physical residuals while retaining smooth interior gradients."""

        cfg = self.config
        unit = torch.tanh(raw_residuals)
        gear = requested_gear.to(device=unit.device, dtype=unit.dtype)[:, None]
        anchors = torch.where(
            gear > 0,
            self.forward_speed_anchors.to(unit),
            self.reverse_speed_anchors.to(unit),
        )
        speed_limit = torch.where(
            gear > 0,
            unit.new_tensor(cfg.forward_speed_limit_mps),
            unit.new_tensor(cfg.reverse_speed_limit_mps),
        )
        # The lower span and upper span differ (the 0.50 m/s reverse anchor is
        # already at its calibrated limit).  This piecewise map is continuous,
        # equals the canonical anchor at zero, and cannot reverse the gear.
        lower_span = anchors - 0.05
        upper_span = speed_limit - anchors
        speed_delta = torch.where(unit[..., 2] >= 0.0, unit[..., 2] * upper_span, unit[..., 2] * lower_span)
        return torch.stack(
            (
                unit[..., 0] * cfg.steering_residual_rad,
                unit[..., 1] * cfg.steering_residual_rad,
                speed_delta,
                unit[..., 3] * cfg.duration_residual_s,
            ),
            dim=-1,
        )

    def forward(self, vehicle_state, requested_gear, raw_residuals):
        self._validate_inputs(vehicle_state, requested_gear, raw_residuals)
        cfg = self.config
        dtype, device = raw_residuals.dtype, raw_residuals.device
        state = vehicle_state.to(device=device, dtype=dtype)
        gear = requested_gear.to(device=device, dtype=dtype)
        opposite = gear * state[:, 0] < -cfg.opposite_motion_reject_mps
        if bool(torch.any(opposite)):
            raise ValueError("GearSupervisor must stop the car before changing requested gear")

        physical = self.bound_residuals(raw_residuals, requested_gear)
        steering_anchor = self.steering_anchors.to(raw_residuals)[None, :]
        speed_anchor = torch.where(
            gear[:, None] > 0,
            self.forward_speed_anchors.to(raw_residuals)[None, :],
            self.reverse_speed_anchors.to(raw_residuals)[None, :],
        )
        steering_mid = steering_anchor + physical[..., 0]
        steering_end = steering_anchor + physical[..., 1]
        speed_end_magnitude = speed_anchor + physical[..., 2]
        duration = cfg.horizon_s + physical[..., 3]

        steering_mid = steering_mid.clamp(-cfg.steering_limit_rad, cfg.steering_limit_rad)
        steering_end = steering_end.clamp(-cfg.steering_limit_rad, cfg.steering_limit_rad)
        speed_limit = torch.where(
            gear[:, None] > 0,
            raw_residuals.new_tensor(cfg.forward_speed_limit_mps),
            raw_residuals.new_tensor(cfg.reverse_speed_limit_mps),
        )
        speed_end_magnitude = speed_end_magnitude.clamp(min=0.0)
        speed_end_magnitude = torch.minimum(speed_end_magnitude, speed_limit)

        batch, candidates = raw_residuals.shape[:2]
        signed_speed_target = gear[:, None] * speed_end_magnitude
        initial_speed = gear[:, None] * torch.relu(gear[:, None] * state[:, 0:1]).expand(-1, candidates)
        initial_steering = state[:, 2:3].clamp(
            -cfg.steering_limit_rad, cfg.steering_limit_rad
        ).expand(-1, candidates)

        dt = duration / float(cfg.steps - 1)
        x = raw_residuals.new_zeros((batch, candidates))
        y = raw_residuals.new_zeros((batch, candidates))
        yaw = raw_residuals.new_zeros((batch, candidates))
        speed = initial_speed
        steering = initial_steering
        rows = [torch.stack((x, x, y, yaw, speed, steering), dim=-1)]

        for step_index in range(1, cfg.steps):
            u = raw_residuals.new_tensor(step_index / float(cfg.steps - 1))
            if step_index * 2 <= cfg.steps - 1:
                blend = 2.0 * u
                steering_target = steering_anchor + blend * (steering_mid - steering_anchor)
            else:
                blend = 2.0 * u - 1.0
                steering_target = steering_mid + blend * (steering_end - steering_mid)

            steering_delta = (steering_target - steering).clamp(
                min=-cfg.steering_rate_limit_rad_s * dt,
                max=cfg.steering_rate_limit_rad_s * dt,
            )
            next_steering = (steering + steering_delta).clamp(
                -cfg.steering_limit_rad, cfg.steering_limit_rad
            )
            # Apply the asymmetric longitudinal limits in the selected drive
            # direction.  For reverse, a more negative signed velocity is
            # acceleration, while a positive dv/dt is braking.
            directed_delta = gear[:, None] * (signed_speed_target - speed)
            directed_delta = directed_delta.clamp(
                min=-cfg.deceleration_limit_mps2 * dt,
                max=cfg.acceleration_limit_mps2 * dt,
            )
            next_speed = speed + gear[:, None] * directed_delta
            # Numerical clipping cannot create an opposite-gear velocity.
            next_speed = gear[:, None] * torch.relu(gear[:, None] * next_speed)

            mean_speed = 0.5 * (speed + next_speed)
            mean_steering = 0.5 * (steering + next_steering)
            yaw_rate = mean_speed * torch.tan(mean_steering) / cfg.wheelbase_m
            next_yaw = yaw + yaw_rate * dt
            mid_yaw = yaw + 0.5 * yaw_rate * dt
            next_x = x + mean_speed * torch.cos(mid_yaw) * dt
            next_y = y + mean_speed * torch.sin(mid_yaw) * dt
            time = duration * u
            rows.append(torch.stack((time, next_x, next_y, next_yaw, next_speed, next_steering), dim=-1))
            x, y, yaw, speed, steering = next_x, next_y, next_yaw, next_speed, next_steering

        trajectory = torch.stack(rows, dim=2)
        controls = torch.stack((steering_mid, steering_end, signed_speed_target, duration), dim=-1)
        return AckermannRolloutOutput(trajectory=trajectory, controls=controls, residuals=physical)
