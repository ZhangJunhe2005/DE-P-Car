import numpy as np
import pytest

from dep_car.core.lattice import AckermannLattice
from dep_car.core.types import Gear, VehicleState


def test_lattice_has_three_by_five_candidates():
    candidates = AckermannLattice().generate(VehicleState())
    assert len(candidates) == 15
    assert {item.candidate_id for item in candidates} == set(range(15))
    assert all(item.trajectory.shape == (11, 6) for item in candidates)


def test_positive_rep103_steering_turns_left():
    trajectory = AckermannLattice().rollout(VehicleState(), 1.0, 0.3)
    assert trajectory[-1, 2] > 0.0
    assert trajectory[-1, 3] > 0.0
    assert np.all(np.diff(trajectory[:, 0]) > 0.0)


def test_retiming_preserves_candidate_count_and_slows_speed():
    lattice = AckermannLattice()
    nominal = lattice.generate(VehicleState(), duration_scale=1.0, speed_scale=1.0)
    slowed = lattice.generate(VehicleState(), duration_scale=1.4, speed_scale=1 / 1.4)
    assert len(slowed) == len(nominal) == 15
    assert slowed[12].trajectory[-1, 4] < nominal[12].trajectory[-1, 4]


def test_reverse_bank_has_fifteen_signed_nonholonomic_candidates():
    candidates = AckermannLattice().generate(VehicleState(), gear=Gear.REVERSE)
    assert len(candidates) == 15
    assert all(item.gear == Gear.REVERSE for item in candidates)
    assert all(item.trajectory[-1, 4] < 0.0 for item in candidates)
    left_in_reverse = candidates[4].trajectory
    assert left_in_reverse[-1, 3] < 0.0


def test_longitudinal_acceleration_and_braking_limits_are_gear_relative():
    lattice = AckermannLattice()
    cases = (
        (VehicleState(speed=0.0), -0.5, -1.5),
        (VehicleState(speed=-0.5), -0.2, 2.0),
        (VehicleState(speed=0.0), 0.6, 1.5),
        (VehicleState(speed=2.5), 0.6, -2.0),
    )
    for state, target_speed, expected_signed_acceleration in cases:
        trajectory = lattice.rollout(state, target_speed, 0.0)
        dt = trajectory[1, 0] - trajectory[0, 0]
        signed_acceleration = (trajectory[1, 4] - trajectory[0, 4]) / dt
        assert signed_acceleration == pytest.approx(expected_signed_acceleration)

        gear = -1.0 if target_speed < 0.0 else 1.0
        directed_acceleration = gear * signed_acceleration
        assert directed_acceleration <= lattice.config.max_acceleration + 1.0e-12
        assert directed_acceleration >= -lattice.config.max_deceleration - 1.0e-12


def test_candidate_bank_has_one_supervisor_conditioned_gear_and_left_right_equivariance():
    lattice = AckermannLattice()
    for gear in (Gear.FORWARD, Gear.REVERSE):
        candidates = lattice.generate(VehicleState(), gear=gear)
        assert {candidate.gear for candidate in candidates} == {gear}
        left, right = candidates[4].trajectory, candidates[0].trajectory
        assert np.allclose(left[:, 1], right[:, 1], atol=1e-9)
        assert np.allclose(left[:, 2], -right[:, 2], atol=1e-9)
        assert np.allclose(left[:, 3], -right[:, 3], atol=1e-9)
