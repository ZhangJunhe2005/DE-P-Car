"""Differentiable multi-action Ackermann rollout for DEPCarNetV4."""

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


HYBRID_SEQUENCE_ROLLOUT_SCHEMA = "DEPCarHybridSequenceRolloutV4SignedAckermann6"


class HybridSequenceRolloutOutput(NamedTuple):
    trajectory: torch.Tensor
    controls: torch.Tensor
    gear_tokens: torch.Tensor
    action_gears: torch.Tensor
    action_mask: torch.Tensor
    motion_gears: torch.Tensor
    shift_required: torch.Tensor
    transition_duration: torch.Tensor


@dataclass(frozen=True)
class HybridSequenceRolloutConfigV4:
    candidates: int = 15
    actions: int = 6
    steps_per_action: int = 5
    minimum_action_duration_s: float = 0.35
    maximum_action_duration_s: float = 1.25
    stop_duration_s: float = 0.45
    shift_dwell_s: float = 0.25
    opposite_motion_threshold_mps: float = 0.03
    acceleration_limit_mps2: float = ACCELERATION_LIMIT_MPS2
    deceleration_limit_mps2: float = DECELERATION_LIMIT_MPS2
    steering_rate_limit_rad_s: float = 0.75
    steering_limit_rad: float = STEERING_OPERATING_LIMIT_RAD
    forward_speed_limit_mps: float = FORWARD_SPEED_LIMIT_MPS
    reverse_speed_limit_mps: float = REVERSE_SPEED_LIMIT_MPS
    wheelbase_m: float = PLANNER_ROLLOUT_WHEELBASE_M

    def validate(self):
        if self.candidates < 2 or self.actions < 3 or self.steps_per_action < 2:
            raise ValueError("V4 hybrid sequence dimensions are too small")
        if not 0.0 < self.minimum_action_duration_s < self.maximum_action_duration_s:
            raise ValueError("V4 action-duration bounds are invalid")
        positive = (
            self.stop_duration_s,
            self.shift_dwell_s,
            self.opposite_motion_threshold_mps,
            self.acceleration_limit_mps2,
            self.deceleration_limit_mps2,
            self.steering_rate_limit_rad_s,
            self.steering_limit_rad,
            self.forward_speed_limit_mps,
            self.reverse_speed_limit_mps,
            self.wheelbase_m,
        )
        if any(float(value) <= 0.0 for value in positive):
            raise ValueError("V4 rollout parameters must be positive")


class HybridSequenceAckermannRolloutV4(nn.Module):
    """Roll out 15 candidates whose six macro actions jointly include gear.

    Gear-token order is STOP, FORWARD, REVERSE.  A STOP token terminates the
    remaining macro actions.  A change of drive direction inserts a physical
    brake-to-zero and shift dwell before the new signed-speed command.
    """

    gear_token_order = ("STOP", "FORWARD", "REVERSE")

    def __init__(self, config=HybridSequenceRolloutConfigV4()):
        super().__init__()
        config.validate()
        self.config = config

    def _validate(self, vehicle_state, current_gear, raw_controls, gear_logits):
        cfg = self.config
        if vehicle_state.ndim != 2 or vehicle_state.shape[1] != 9:
            raise ValueError("V4 vehicle_state must have shape [B,9]")
        if current_gear.shape != (len(vehicle_state),):
            raise ValueError("V4 current_gear must have shape [B]")
        if not bool(torch.all((current_gear == -1) | (current_gear == 0) | (current_gear == 1))):
            raise ValueError("V4 current_gear must contain -1, 0 or +1")
        expected = (len(vehicle_state), cfg.candidates, cfg.actions)
        if raw_controls.shape != expected + (4,):
            raise ValueError("V4 raw_controls must have shape [B,15,6,4]")
        if gear_logits.shape != expected + (3,):
            raise ValueError("V4 gear_logits must have shape [B,15,6,3]")

    def bound_controls(self, raw_controls, action_gears):
        cfg = self.config
        unit = torch.tanh(raw_controls)
        steering_mid = unit[..., 0] * float(cfg.steering_limit_rad)
        steering_end = unit[..., 1] * float(cfg.steering_limit_rad)
        speed_unit = torch.sigmoid(raw_controls[..., 2])
        speed_limit = torch.where(
            action_gears > 0,
            raw_controls.new_tensor(cfg.forward_speed_limit_mps),
            raw_controls.new_tensor(cfg.reverse_speed_limit_mps),
        )
        speed_magnitude = (0.08 + 0.92 * speed_unit) * speed_limit
        signed_speed = action_gears.to(raw_controls) * speed_magnitude
        duration = float(cfg.minimum_action_duration_s) + torch.sigmoid(
            raw_controls[..., 3]
        ) * float(cfg.maximum_action_duration_s - cfg.minimum_action_duration_s)
        return torch.stack((steering_mid, steering_end, signed_speed, duration), dim=-1)

    @staticmethod
    def _integrate_pose(x, y, yaw, speed, steering, dt, wheelbase):
        yaw_rate = speed * torch.tan(steering) / float(wheelbase)
        next_yaw = yaw + yaw_rate * dt
        mid_yaw = yaw + 0.5 * yaw_rate * dt
        return (
            x + speed * torch.cos(mid_yaw) * dt,
            y + speed * torch.sin(mid_yaw) * dt,
            next_yaw,
        )

    def forward(self, vehicle_state, current_gear, raw_controls, gear_logits):
        self._validate(vehicle_state, current_gear, raw_controls, gear_logits)
        cfg = self.config
        batch, candidates = raw_controls.shape[:2]
        token = gear_logits.argmax(dim=-1)
        token_gear = torch.where(
            token == 1,
            current_gear.new_ones(token.shape),
            torch.where(token == 2, current_gear.new_full(token.shape, -1), current_gear.new_zeros(token.shape)),
        )
        # STOP terminates this and every later macro action.
        non_stop = token != 0
        alive_before = torch.cat(
            (
                torch.ones_like(non_stop[..., :1]),
                torch.cumprod(non_stop[..., :-1].to(torch.int64), dim=-1).bool(),
            ),
            dim=-1,
        )
        action_mask = alive_before & non_stop
        action_gears = torch.where(action_mask, token_gear, torch.zeros_like(token_gear))
        controls = self.bound_controls(raw_controls, action_gears)

        shape = (batch, candidates)
        x = raw_controls.new_zeros(shape)
        y = raw_controls.new_zeros(shape)
        yaw = raw_controls.new_zeros(shape)
        speed = vehicle_state[:, 0:1].to(raw_controls).expand(-1, candidates).clone()
        steering = vehicle_state[:, 2:3].to(raw_controls).expand(-1, candidates).clone()
        engaged = current_gear[:, None].expand(-1, candidates).clone()
        elapsed = raw_controls.new_zeros(shape)
        effective_initial_gear = torch.where(
            speed > cfg.opposite_motion_threshold_mps,
            engaged.new_ones(shape),
            torch.where(
                speed < -cfg.opposite_motion_threshold_mps,
                engaged.new_full(shape, -1),
                torch.where(engaged == 0, engaged.new_ones(shape), engaged),
            ),
        )
        rows = [torch.stack((elapsed, x, y, yaw, speed, steering), dim=-1)]
        gear_rows = [effective_initial_gear]
        shifts = []
        transition_durations = []

        for action_index in range(cfg.actions):
            active = action_mask[..., action_index]
            desired = action_gears[..., action_index]
            desired_effective = torch.where(active, desired, torch.where(engaged == 0, engaged.new_ones(shape), engaged))
            shift = active & (
                (engaged != desired)
                | (desired.to(speed) * speed < -float(cfg.opposite_motion_threshold_mps))
            )
            brake_time = torch.where(
                shift,
                speed.abs() / float(cfg.deceleration_limit_mps2),
                speed.new_zeros(shape),
            )
            transition = torch.where(
                shift,
                brake_time + float(cfg.shift_dwell_s),
                speed.new_zeros(shape),
            )
            # Account for the swept displacement while braking.  The first
            # sampled row of the next action connects continuously to it.
            brake_speed = 0.5 * speed
            x_brake, y_brake, yaw_brake = self._integrate_pose(
                x, y, yaw, brake_speed, steering, brake_time, cfg.wheelbase_m
            )
            x = torch.where(shift, x_brake, x)
            y = torch.where(shift, y_brake, y)
            yaw = torch.where(shift, yaw_brake, yaw)
            centered = torch.sign(steering) * torch.relu(
                steering.abs() - float(cfg.steering_rate_limit_rad_s) * brake_time
            )
            steering = torch.where(shift, centered, steering)
            speed = torch.where(shift, speed.new_zeros(shape), speed)
            engaged = torch.where(shift, desired, engaged)
            engaged = torch.where(active & (engaged == 0), desired, engaged)
            elapsed = elapsed + transition
            shifts.append(shift)
            transition_durations.append(transition)

            control = controls[..., action_index, :]
            target_mid, target_end = control[..., 0], control[..., 1]
            target_speed = torch.where(active, control[..., 2], speed.new_zeros(shape))
            duration = torch.where(
                active,
                control[..., 3],
                speed.new_full(shape, float(cfg.stop_duration_s)),
            )
            dt = duration / float(cfg.steps_per_action)
            start_steering = steering
            for step_index in range(1, cfg.steps_per_action + 1):
                fraction = raw_controls.new_tensor(step_index / float(cfg.steps_per_action))
                target_steering = torch.where(
                    fraction <= 0.5,
                    start_steering + (fraction * 2.0) * (target_mid - start_steering),
                    target_mid + ((fraction - 0.5) * 2.0) * (target_end - target_mid),
                )
                target_steering = torch.where(active, target_steering, steering.new_zeros(shape))
                steering_delta = (target_steering - steering).clamp(
                    min=-float(cfg.steering_rate_limit_rad_s) * dt,
                    max=float(cfg.steering_rate_limit_rad_s) * dt,
                )
                next_steering = (steering + steering_delta).clamp(
                    -float(cfg.steering_limit_rad), float(cfg.steering_limit_rad)
                )
                directed_delta = desired_effective.to(speed) * (target_speed - speed)
                directed_delta = directed_delta.clamp(
                    min=-float(cfg.deceleration_limit_mps2) * dt,
                    max=float(cfg.acceleration_limit_mps2) * dt,
                )
                next_speed = speed + desired_effective.to(speed) * directed_delta
                next_speed = torch.where(
                    active,
                    desired_effective.to(speed) * torch.relu(desired_effective.to(speed) * next_speed),
                    torch.sign(speed) * torch.relu(speed.abs() - float(cfg.deceleration_limit_mps2) * dt),
                )
                mean_speed = 0.5 * (speed + next_speed)
                mean_steering = 0.5 * (steering + next_steering)
                next_x, next_y, next_yaw = self._integrate_pose(
                    x, y, yaw, mean_speed, mean_steering, dt, cfg.wheelbase_m
                )
                elapsed = elapsed + dt
                x, y, yaw = next_x, next_y, next_yaw
                speed, steering = next_speed, next_steering
                row_gear = torch.where(
                    speed > cfg.opposite_motion_threshold_mps,
                    engaged.new_ones(shape),
                    torch.where(
                        speed < -cfg.opposite_motion_threshold_mps,
                        engaged.new_full(shape, -1),
                        torch.where(engaged == 0, engaged.new_ones(shape), engaged),
                    ),
                )
                rows.append(torch.stack((elapsed, x, y, yaw, speed, steering), dim=-1))
                gear_rows.append(row_gear)

        trajectory = torch.stack(rows, dim=2)
        motion_gears = torch.stack(gear_rows, dim=2)
        return HybridSequenceRolloutOutput(
            trajectory=trajectory,
            controls=controls,
            gear_tokens=token,
            action_gears=action_gears,
            action_mask=action_mask,
            motion_gears=motion_gears,
            shift_required=torch.stack(shifts, dim=-1),
            transition_duration=torch.stack(transition_durations, dim=-1),
        )


__all__ = [
    "HYBRID_SEQUENCE_ROLLOUT_SCHEMA",
    "HybridSequenceAckermannRolloutV4",
    "HybridSequenceRolloutConfigV4",
    "HybridSequenceRolloutOutput",
]
