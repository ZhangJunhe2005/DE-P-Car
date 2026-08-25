"""Joint forward/reverse Ackermann rollout with an explicit shift prefix.

V2 receives one supervisor-selected gear and can only refine that bank.  V3
evaluates both banks in one forward pass.  A bank opposite to the currently
engaged gear is prefixed by a deterministic brake-to-zero and shift dwell, so
the learned score cannot pretend that a transmission reversal is instantaneous
or spatially free.
"""

from dataclasses import dataclass
from typing import NamedTuple

import torch
from torch import nn

from dep_car.core.state_contract import DECELERATION_LIMIT_MPS2
from dep_car.core.vehicle import PLANNER_ROLLOUT_WHEELBASE_M

from .ackermann_rollout import AckermannRolloutConfig, AckermannRolloutV1


BIDIRECTIONAL_ROLLOUT_SCHEMA = (
    "BidirectionalAckermannRolloutV3BrakeShiftForwardReverse30"
)


class BidirectionalAckermannRolloutOutput(NamedTuple):
    trajectory: torch.Tensor
    controls: torch.Tensor
    residuals: torch.Tensor
    candidate_gears: torch.Tensor
    motion_gears: torch.Tensor
    shift_required: torch.Tensor
    transition_duration: torch.Tensor


@dataclass(frozen=True)
class BidirectionalAckermannRolloutConfig:
    transition_steps: int = 5
    shift_dwell_s: float = 0.25
    braking_deceleration_mps2: float = DECELERATION_LIMIT_MPS2
    steering_center_rate_rad_s: float = 0.75
    opposite_motion_threshold_mps: float = 0.03

    def validate(self):
        if self.transition_steps < 2:
            raise ValueError("transition_steps must be at least two")
        if min(
            self.shift_dwell_s,
            self.braking_deceleration_mps2,
            self.steering_center_rate_rad_s,
            self.opposite_motion_threshold_mps,
        ) <= 0.0:
            raise ValueError("bidirectional transition parameters must be positive")


class BidirectionalAckermannRolloutV3(nn.Module):
    """Roll out 15 forward and 15 reverse candidates from one measured state."""

    candidate_count = 30
    candidates_per_gear = 15
    gear_order = (1, -1)

    def __init__(
        self,
        rollout_config: AckermannRolloutConfig = AckermannRolloutConfig(),
        config: BidirectionalAckermannRolloutConfig = (
            BidirectionalAckermannRolloutConfig()
        ),
    ):
        super().__init__()
        config.validate()
        self.config = config
        self.single = AckermannRolloutV1(rollout_config)

    @staticmethod
    def _validate(vehicle_state, current_gear, raw_residuals):
        if vehicle_state.ndim != 2 or vehicle_state.shape[1] != 9:
            raise ValueError("vehicle_state must have shape [B,9]")
        if current_gear.shape != (len(vehicle_state),):
            raise ValueError("current_gear must have shape [B]")
        if not bool(
            torch.all(
                (current_gear == -1)
                | (current_gear == 0)
                | (current_gear == 1)
            )
        ):
            raise ValueError("current_gear must contain only -1, 0 or +1")
        if raw_residuals.shape != (len(vehicle_state), 30, 4):
            raise ValueError("raw_residuals must have shape [B,30,4]")

    def _needs_shift(self, state, current_gear, desired_gear):
        desired = current_gear.new_full(current_gear.shape, int(desired_gear))
        engaged_opposite = (current_gear != 0) & (current_gear != desired)
        moving_opposite = (
            desired.to(state) * state[:, 0]
            < -float(self.config.opposite_motion_threshold_mps)
        )
        return engaged_opposite | moving_opposite

    def _transition_prefix(self, state, current_gear, desired_gear, needs_shift):
        """Return [B,P,6] prefix, post-shift state and row-wise gear schedule."""

        cfg = self.config
        batch = len(state)
        dtype, device = state.dtype, state.device
        speed0 = state[:, 0]
        steering0 = state[:, 2]
        motion_duration = torch.where(
            needs_shift,
            speed0.abs() / float(cfg.braking_deceleration_mps2),
            speed0.new_zeros(batch),
        )
        shift_duration = torch.where(
            needs_shift,
            speed0.new_full((batch,), float(cfg.shift_dwell_s)),
            speed0.new_zeros(batch),
        )
        dt = motion_duration / float(cfg.transition_steps - 1)
        x = state.new_zeros(batch)
        y = state.new_zeros(batch)
        yaw = state.new_zeros(batch)
        rows = []
        for index in range(cfg.transition_steps):
            fraction = state.new_tensor(index / float(cfg.transition_steps - 1))
            time = fraction * motion_duration
            speed = torch.where(needs_shift, speed0 * (1.0 - fraction), speed0)
            centered = torch.sign(steering0) * torch.relu(
                steering0.abs()
                - float(cfg.steering_center_rate_rad_s) * time
            )
            steering = torch.where(needs_shift, centered, steering0)
            if index:
                previous = rows[-1]
                mean_speed = 0.5 * (previous[:, 4] + speed)
                mean_steering = 0.5 * (previous[:, 5] + steering)
                yaw_rate = (
                    mean_speed
                    * torch.tan(mean_steering)
                    / float(PLANNER_ROLLOUT_WHEELBASE_M)
                )
                next_yaw = yaw + yaw_rate * dt
                mid_yaw = yaw + 0.5 * yaw_rate * dt
                x = x + mean_speed * torch.cos(mid_yaw) * dt
                y = y + mean_speed * torch.sin(mid_yaw) * dt
                yaw = next_yaw
            rows.append(torch.stack((time, x, y, yaw, speed, steering), dim=-1))
        prefix = torch.stack(rows, dim=1)
        # The dwell happens at zero speed and therefore changes time only.
        prefix = prefix.clone()
        prefix[:, -1, 0] = prefix[:, -1, 0] + shift_duration

        post = state.clone()
        post[:, 0] = torch.where(needs_shift, post[:, 0].new_zeros(batch), post[:, 0])
        post[:, 1] = torch.where(needs_shift, post[:, 1].new_zeros(batch), post[:, 1])
        post[:, 2] = prefix[:, -1, 5]

        desired = current_gear.new_full((batch,), int(desired_gear))
        # During the braking prefix, row-wise direction follows measured
        # motion rather than blindly trusting a possibly stale gearbox report.
        # This keeps a car that is still rolling backward while engaging
        # forward physically valid until it reaches zero; the final dwell row
        # adopts the desired drive gear.
        prefix_speed = prefix[..., 4]
        schedule = torch.where(
            prefix_speed > cfg.opposite_motion_threshold_mps,
            current_gear.new_ones(prefix_speed.shape),
            torch.where(
                prefix_speed < -cfg.opposite_motion_threshold_mps,
                current_gear.new_full(prefix_speed.shape, -1),
                desired[:, None].expand_as(prefix_speed),
            ),
        )
        schedule[:, -1] = desired
        return prefix, post, schedule, motion_duration + shift_duration

    @staticmethod
    def _compose(prefix, bank):
        """Compose a body-frame bank after the final transition-prefix pose."""

        endpoint = prefix[:, -1]
        px, py, pyaw = endpoint[:, 1], endpoint[:, 2], endpoint[:, 3]
        x, y = bank[..., 1], bank[..., 2]
        cosine = torch.cos(pyaw)[:, None, None]
        sine = torch.sin(pyaw)[:, None, None]
        composed = bank.clone()
        composed[..., 0] = bank[..., 0] + endpoint[:, None, None, 0]
        composed[..., 1] = px[:, None, None] + cosine * x - sine * y
        composed[..., 2] = py[:, None, None] + sine * x + cosine * y
        composed[..., 3] = pyaw[:, None, None] + bank[..., 3]
        expanded = prefix[:, None].expand(-1, bank.shape[1], -1, -1)
        return torch.cat((expanded, composed[..., 1:, :]), dim=2)

    def _bank(self, state, current_gear, raw, desired_gear):
        needs_shift = self._needs_shift(state, current_gear, desired_gear)
        prefix, post, prefix_gears, transition_duration = self._transition_prefix(
            state, current_gear, desired_gear, needs_shift
        )
        desired = current_gear.new_full((len(state),), int(desired_gear))
        bank = self.single(post, desired, raw)
        trajectory = self._compose(prefix, bank.trajectory)
        post_gears = desired[:, None].expand(
            -1, self.single.config.steps - 1
        )
        motion_gears = torch.cat((prefix_gears, post_gears), dim=1)
        motion_gears = motion_gears[:, None].expand(-1, 15, -1)
        return (
            trajectory,
            bank.controls,
            bank.residuals,
            desired[:, None].expand(-1, 15),
            motion_gears,
            needs_shift[:, None].expand(-1, 15),
            transition_duration[:, None].expand(-1, 15),
        )

    def forward(self, vehicle_state, current_gear, raw_residuals):
        self._validate(vehicle_state, current_gear, raw_residuals)
        state = vehicle_state.to(raw_residuals)
        banks = []
        for bank_index, gear in enumerate(self.gear_order):
            start = bank_index * self.candidates_per_gear
            stop = start + self.candidates_per_gear
            banks.append(
                self._bank(state, current_gear, raw_residuals[:, start:stop], gear)
            )
        columns = tuple(torch.cat((banks[0][i], banks[1][i]), dim=1) for i in range(7))
        return BidirectionalAckermannRolloutOutput(*columns)


__all__ = [
    "BIDIRECTIONAL_ROLLOUT_SCHEMA",
    "BidirectionalAckermannRolloutConfig",
    "BidirectionalAckermannRolloutOutput",
    "BidirectionalAckermannRolloutV3",
]
