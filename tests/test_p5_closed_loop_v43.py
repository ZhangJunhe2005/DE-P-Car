from types import SimpleNamespace

import torch

from dep_car.core.types import Gear
from dep_car.global_planner.hybrid_astar import HybridPathPose
from dep_car.model.dep_car_net_v4 import DEPCarNetV4
from dep_car.model.dep_car_net_v43 import DEPCarNetV43
from dep_car.training.losses_v43 import (
    action_plan_geometry_loss,
    mandatory_execution_mask,
)


def test_v43_exact_plan_run_endpoints_preserve_complete_turnaround():
    import build_p3_v7_v43_closed_loop_index as indexer

    start = (2.0, -1.0, 0.0)
    path = [
        HybridPathPose(2.0, -1.0, 0.0, Gear.NEUTRAL, 0.0),
        HybridPathPose(2.3, -1.0, 0.1, Gear.FORWARD, 0.2),
        HybridPathPose(2.1, -1.1, 0.4, Gear.REVERSE, -0.2),
        HybridPathPose(2.4, -0.9, 0.8, Gear.FORWARD, 0.2),
    ]
    runs = indexer._gear_runs(path, start)
    assert [row["gear"] for row in runs] == [1, -1, 1]
    assert len(runs) == 3
    assert all(len(row["endpoint"]) == 3 for row in runs)


def test_v43_action_plan_geometry_prefers_matching_complete_sequence():
    trajectories = torch.zeros((1, 2, 31, 6), dtype=torch.float32)
    target = torch.tensor(
        [[[0.4, 0.1, 0.2], [0.2, -0.1, 0.4], [0.6, 0.0, 0.8],
          [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]
    )
    trajectories[0, 0, 5, 1:4] = target[0, 0]
    trajectories[0, 0, 10, 1:4] = target[0, 1]
    trajectories[0, 0, 15, 1:4] = target[0, 2]
    trajectories[0, 1, :, 1] = 2.0
    gears = torch.tensor([[1, -1, 1, 0, 0, 0]])
    mask = gears != 0
    output = SimpleNamespace(
        trajectories=trajectories,
        action_gears=torch.tensor(
            [[[1, -1, 1, 1, 1, 1], [-1, 1, -1, 1, -1, 1]]]
        ),
        action_mask=torch.tensor(
            [[[True, True, True, False, False, False], [True] * 6]]
        ),
    )
    loss, per_candidate = action_plan_geometry_loss(
        output, gears, mask, target
    )
    assert float(loss) < 1.0e-8
    assert float(per_candidate[0, 0]) < float(per_candidate[0, 1])


def test_v43_authority_names_are_explicitly_offline_only():
    import build_p3_v7_v43_closed_loop_index as indexer
    import train_dep_car_closed_loop_v43 as trainer

    assert indexer.SEQUENCE_AUTHORITY == trainer.SEQUENCE_AUTHORITY
    assert "HYBRID_ASTAR" in trainer.SEQUENCE_AUTHORITY


def test_v43_initializes_exactly_from_unified_v4_without_new_gear_selector():
    source = DEPCarNetV4()
    model = DEPCarNetV43()
    transferred = model.initialize_from_v4(source.state_dict())
    assert transferred
    assert all(name.startswith("closed_loop_score_adapter.") for name in transferred)
    assert set(source.state_dict()) < set(model.state_dict())
    assert model.requires_mandatory_hard_veto is True
    assert model.high_level_gear_state_machine is False
    assert not hasattr(model, "gear_selector")
    assert model.closed_loop_score_adapter[0].in_features == 113
    assert model.closed_loop_score_adapter[0].out_features == 64


def test_v43_mandatory_guard_filters_candidates_without_selecting_a_gear():
    initial_safe = torch.tensor([True, False, True])
    hard = torch.tensor([
        [True, False, True],
        [False, False, False],
        [False, False, False],
    ])
    egress = torch.tensor([
        [False, True, False],
        [False, True, True],
        [True, True, True],
    ])
    allowed = mandatory_execution_mask(initial_safe, hard, egress)
    assert allowed.tolist() == [
        [True, False, True],
        [False, True, True],
        [False, False, False],
    ]


def test_v43_zero_residual_preserves_source_scores_with_contextual_inputs():
    source = DEPCarNetV4().eval()
    model = DEPCarNetV43().eval()
    model.initialize_from_v4(source.state_dict())
    depth = torch.zeros(1, 2, 96, 160)
    depth[:, 1] = 1.0
    lidar = torch.zeros(1, 6, 160, 160)
    state = torch.tensor([[0.4, 0.1, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0]])
    current = torch.tensor([-1])
    history = torch.tensor([[-1.0, 0.2, 1.0, 0.0, 2.0, 0.1]])
    route = torch.zeros(1, 8, 3)
    route[:, :, 0] = torch.linspace(0.2, 2.0, 8)
    route_mask = torch.ones(1, 8, dtype=torch.bool)
    with torch.inference_mode():
        source_output = source(
            depth, lidar, state, current, history, route, route_mask
        )
        output = model(
            depth, lidar, state, current, history, route, route_mask
        )
    assert torch.equal(output.gear_tokens, source_output.gear_tokens)
    assert torch.allclose(output.trajectories, source_output.trajectories)
    assert torch.allclose(output.scores, source_output.scores, atol=1.0e-7)
