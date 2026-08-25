import math
import importlib.util
from pathlib import Path

import torch

from dep_car.model.bidirectional_rollout import BidirectionalAckermannRolloutV3
from dep_car.model.dep_car_net import DEPCarNetConfig
from dep_car.model.dep_car_net_v3 import DEPCarNetV3
from dep_car.training.losses_v3 import (
    DEPCarJointGearLossConfigV3,
    bidirectional_kinematic_components,
    conditional_gear_costs,
)


def test_bidirectional_rollout_exposes_both_gears_and_brake_shift_prefix():
    rollout = BidirectionalAckermannRolloutV3()
    state = torch.zeros(1, 9)
    state[:, 0] = 0.8
    state[:, 2] = 0.30
    raw = torch.zeros(1, 30, 4)
    output = rollout(state, torch.tensor([1]), raw)

    assert output.trajectory.shape == (1, 30, 15, 6)
    assert output.candidate_gears.shape == (1, 30)
    assert torch.all(output.candidate_gears[:, :15] == 1)
    assert torch.all(output.candidate_gears[:, 15:] == -1)
    assert not bool(output.shift_required[:, :15].any())
    assert bool(output.shift_required[:, 15:].all())
    reverse = output.trajectory[0, 15]
    # The opposite bank first brakes in the actually engaged forward gear,
    # reaches zero, dwells, and only then develops negative velocity.
    assert torch.all(reverse[:5, 4] >= -1.0e-7)
    assert math.isclose(float(reverse[4, 4]), 0.0, abs_tol=1.0e-7)
    assert torch.all(reverse[5:, 4] <= 1.0e-7)
    assert float(output.transition_duration[0, 15]) > 0.25
    assert torch.all(output.motion_gears[0, 15, :4] == 1)
    assert torch.all(output.motion_gears[0, 15, 4:] == -1)
    components = bidirectional_kinematic_components(
        output.trajectory, output.motion_gears
    )
    # High-speed/full-lock canonical forward anchors may intentionally be
    # vetoed by lateral acceleration.  The new brake/shift boundary itself
    # must not create an opposite-motion, acceleration or deceleration veto.
    for name in ("opposite_motion", "acceleration", "deceleration"):
        assert not bool(components[name][0, 15])


def test_bidirectional_rollout_allows_multileg_reverse_from_settled_neutral():
    rollout = BidirectionalAckermannRolloutV3()
    state = torch.zeros(2, 9)
    raw = torch.zeros(2, 30, 4)
    output = rollout(state, torch.tensor([0, 0]), raw)

    assert not bool(output.shift_required.any())
    assert torch.all(output.transition_duration == 0.0)
    assert torch.all(output.trajectory[:, :15, -1, 4] >= 0.0)
    assert torch.all(output.trajectory[:, 15:, -1, 4] <= 0.0)


def test_bidirectional_transition_uses_measured_rolling_direction_until_stop():
    rollout = BidirectionalAckermannRolloutV3()
    state = torch.zeros(1, 9)
    state[:, 0] = -0.2
    output = rollout(state, torch.tensor([1]), torch.zeros(1, 30, 4))
    forward = output.trajectory[:, :15]
    forward_gears = output.motion_gears[:, :15]
    moving = forward[..., 4].abs() > rollout.config.opposite_motion_threshold_mps
    assert torch.all(
        forward_gears[moving].to(forward) * forward[..., 4][moving] > 0.0
    )
    assert torch.all(forward_gears[..., :4] == -1)
    assert torch.all(forward_gears[..., -1] == 1)
    components = bidirectional_kinematic_components(forward, forward_gears)
    assert not bool(components["opposite_motion"].any())


def test_v3_joint_energy_selects_from_one_thirty_candidate_contract():
    model = DEPCarNetV3(
        config=DEPCarNetConfig(enforce_reflection_equivariance=False)
    )
    batch = 2
    depth = torch.zeros(batch, 2, 96, 160)
    depth[:, 1] = 1.0
    lidar = torch.zeros(batch, 6, 160, 160)
    state = torch.zeros(batch, 9)
    current_gear = torch.tensor([1, -1])
    history = torch.zeros(batch, 6)
    history[:, 0] = current_gear
    route = torch.zeros(batch, 8, 3)
    route[:, :, 0] = torch.linspace(0.2, 2.0, 8)
    route_mask = torch.ones(batch, 8, dtype=torch.bool)

    output = model(
        depth,
        lidar,
        state,
        current_gear,
        history,
        route,
        route_mask,
    )
    assert model.predicts_gear is True
    assert output.raw_residuals.shape == (batch, 30, 4)
    assert output.trajectories.shape == (batch, 30, 15, 6)
    assert output.scores.shape == (batch, 30)
    assert output.forward_recovery_value.shape == (batch, 30)
    assert torch.all(torch.isfinite(output.scores))
    assert torch.all((output.forward_recovery_value >= 0.0))
    assert torch.all((output.forward_recovery_value <= 1.0))


def test_v3_can_transfer_all_shared_v2_weights():
    source = DEPCarNetV3(
        config=DEPCarNetConfig(enforce_reflection_equivariance=False)
    )
    v2_state = {
        key: value
        for key, value in source.state_dict().items()
        if not key.startswith(
            (
                "joint_rollout.",
                "history_encoder.",
                "transition_encoder.",
                "joint_energy_head.",
                "forward_recovery_head.",
            )
        )
    }
    target = DEPCarNetV3(
        config=DEPCarNetConfig(enforce_reflection_equivariance=False)
    )
    transferred = target.initialize_from_v2(v2_state)
    assert "route_encoder.point.0.weight" in transferred
    assert "candidate_head.weight" in transferred


def test_conditional_reverse_burden_preserves_useful_multileg_turns():
    config = DEPCarJointGearLossConfigV3()
    gears = torch.tensor([[1] * 15 + [-1] * 15])
    feasible = torch.ones(1, 30, dtype=torch.bool)
    shift = torch.zeros(1, 30, dtype=torch.bool)
    shift[:, 15:] = True
    progress = torch.zeros(1, 30)
    distance = torch.ones(1, 30)
    recovery = torch.zeros(1, 30)
    recovery[:, 15] = 1.0
    history = torch.zeros(1, 6)

    # No forward candidate makes progress.  A reverse leg that unlocks the
    # next forward leg receives recovery credit and is cheaper than an
    # unproductive reverse, even though both travel the same distance.
    blocked_cost, blocked = conditional_gear_costs(
        candidate_gears=gears,
        shift_required=shift,
        hard_feasible=feasible,
        route_progress_m=progress,
        path_distance_m=distance,
        forward_recovery_target=recovery,
        gear_history=history,
        config=config,
    )
    assert not bool(blocked["forward_available"][0])
    assert blocked_cost[0, 15] < blocked_cost[0, 16]
    assert blocked_cost[0, 15] < 0.0

    # On open road, safe forward progress activates the full reverse/switch
    # burden.  This is the case where ordinary people would keep driving
    # forward rather than start a pointless parking manoeuvre.
    progress[:, 0] = config.forward_available_progress_m + 0.2
    open_cost, opened = conditional_gear_costs(
        candidate_gears=gears,
        shift_required=shift,
        hard_feasible=feasible,
        route_progress_m=progress,
        path_distance_m=distance,
        forward_recovery_target=recovery,
        gear_history=history,
        config=config,
    )
    assert bool(opened["forward_available"][0])
    assert open_cost[0, 15] > blocked_cost[0, 15]

    # Several already executed legs do not terminate the manoeuvre.  While
    # forward remains blocked, even a third shift retains recovery value; the
    # history is context, not a hard maximum-number-of-shifts gate.
    history[:, 4] = 3.0
    continued_cost, continued = conditional_gear_costs(
        candidate_gears=gears,
        shift_required=shift,
        hard_feasible=feasible,
        route_progress_m=torch.zeros_like(progress),
        path_distance_m=distance,
        forward_recovery_target=recovery,
        gear_history=history,
        config=config,
    )
    assert not bool(continued["forward_available"][0])
    assert continued_cost[0, 15] < continued_cost[0, 16]


def test_sequence_index_keeps_full_multileg_episode_without_old_gear_oracle():
    path = Path(__file__).resolve().parents[1] / "tools/build_p3_v5_joint_gear_index.py"
    spec = importlib.util.spec_from_file_location("joint_index", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = []
    for index, gear in enumerate((1, -1, 1, -1, 1)):
        rows.append({
            "sample_id": "sample_%d" % index,
            "task_id": "one_turnaround",
            "split": "train",
            "map_uuid": "map-a",
            "source_content_sha256": "0" * 64,
            "stamp": float(index),
            "speed_mps": 0.2 * gear,
            "current_gear": gear,
            "observed_requested_gear": gear,
            "candidate_context": "RECOVERY",
            "reference_gear_runs": [gear],
        })
    attached = module.attach_history(rows)
    assert attached[0]["sequence_gears"][:5] == [1, -1, 1, -1, 1]
    assert attached[0]["sequence_mask"][:5] == [True] * 5
    assert len(attached[0]["history"]) == 6
