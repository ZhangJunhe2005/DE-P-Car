import numpy as np
import pytest
import torch

from dep_car.core.lattice import AckermannLattice
from dep_car.core.types import Gear, VehicleState
from dep_car.model.ackermann_rollout import AckermannRolloutV1
from dep_car.model.symmetry import mirror_candidate_values, mirror_trajectory, mirror_vehicle_state


@pytest.mark.parametrize("gear", (Gear.FORWARD, Gear.REVERSE))
def test_differentiable_rollout_matches_canonical_numpy_lattice(gear):
    torch_rollout = AckermannRolloutV1()
    state = torch.zeros(1, 9)
    output = torch_rollout(state, torch.tensor([int(gear)]), torch.zeros(1, 15, 4))
    authority = AckermannLattice().generate(VehicleState(), gear=gear)
    for candidate_id, candidate in enumerate(authority):
        np.testing.assert_allclose(
            output.trajectory[0, candidate_id].detach().numpy(),
            candidate.trajectory,
            atol=1.0e-6,
        )


@pytest.mark.parametrize("gear", (-1, 1))
def test_rollout_is_signed_bounded_and_has_constant_external_gear(gear):
    rollout = AckermannRolloutV1()
    raw = torch.full((1, 15, 4), 1.0e6)
    output = rollout(torch.zeros(1, 9), torch.tensor([gear]), raw)
    trajectory = output.trajectory
    assert torch.all(gear * trajectory[..., 4] >= -1.0e-7)
    speed_limit = 2.5 if gear > 0 else 0.5
    assert float(trajectory[..., 4].abs().max()) <= speed_limit + 1.0e-6
    assert float(trajectory[..., 5].abs().max()) <= rollout.config.steering_limit_rad + 1.0e-6
    dt = trajectory[..., 1:, 0] - trajectory[..., :-1, 0]
    steering_rate = (trajectory[..., 1:, 5] - trajectory[..., :-1, 5]) / dt
    acceleration = (trajectory[..., 1:, 4] - trajectory[..., :-1, 4]) / dt
    directed_acceleration = gear * acceleration
    assert float(steering_rate.abs().max()) <= 0.75 + 1.0e-5
    assert float(directed_acceleration.max()) <= 1.5 + 1.0e-5
    assert float((-directed_acceleration).max()) <= 2.0 + 1.0e-5
    assert trajectory.shape == (1, 15, 11, 6)


def test_torch_rollout_applies_reverse_acceleration_and_braking_in_drive_frame():
    rollout = AckermannRolloutV1()
    state = torch.zeros(4, 9)
    state[:, 0] = torch.tensor([0.0, -0.5, 0.0, 2.5])
    gear = torch.tensor([-1, -1, 1, 1])
    output = rollout(state, gear, torch.zeros(4, 15, 4)).trajectory
    dt = output[:, 0, 1, 0] - output[:, 0, 0, 0]
    signed_acceleration = (
        output[:, 0, 1, 4] - output[:, 0, 0, 4]
    ) / dt

    torch.testing.assert_close(
        signed_acceleration,
        torch.tensor([-1.5, 2.0, 1.5, -2.0]),
        atol=1.0e-6,
        rtol=0.0,
    )
    directed = gear.to(signed_acceleration) * signed_acceleration
    assert torch.all(directed <= 1.5 + 1.0e-6)
    assert torch.all(directed >= -2.0 - 1.0e-6)


def test_rollout_rejects_illegal_shift_before_stop_and_neutral_request():
    rollout = AckermannRolloutV1()
    state = torch.zeros(1, 9)
    state[0, 0] = 0.2
    with pytest.raises(ValueError, match="stop"):
        rollout(state, torch.tensor([-1]), torch.zeros(1, 15, 4))
    with pytest.raises(ValueError, match="-1 or \\+1"):
        rollout(torch.zeros(1, 9), torch.tensor([0]), torch.zeros(1, 15, 4))


@pytest.mark.parametrize("gear", (-1, 1))
def test_rollout_left_right_reflection_equivariance(gear):
    torch.manual_seed(3)
    rollout = AckermannRolloutV1()
    state = torch.zeros(2, 9)
    state[:, 2] = torch.tensor([0.1, -0.2])
    raw = torch.randn(2, 15, 4) * 0.2
    requested = torch.full((2,), gear, dtype=torch.long)
    original = rollout(state, requested, raw).trajectory
    mirrored_input = mirror_candidate_values(raw, steering_channels=(0, 1))
    mirrored = rollout(mirror_vehicle_state(state), requested, mirrored_input).trajectory
    torch.testing.assert_close(original, mirror_trajectory(mirrored), atol=2.0e-6, rtol=1.0e-5)


def test_rollout_autograd_gradcheck():
    rollout = AckermannRolloutV1().double()
    state = torch.zeros(1, 9, dtype=torch.double)
    gear = torch.ones(1, dtype=torch.long)
    selector = torch.zeros(1, 15, 1, dtype=torch.double)
    selector[:, 7] = 1.0

    def selected_trajectory(parameters):
        raw = selector * parameters[:, None, :]
        return rollout(state, gear, raw).trajectory[:, 7, :, 1:4]

    parameters = torch.tensor([[0.05, -0.04, 0.08, 0.03]], dtype=torch.double, requires_grad=True)
    assert torch.autograd.gradcheck(selected_trajectory, (parameters,), eps=1.0e-6, atol=1.0e-5, rtol=1.0e-3)
