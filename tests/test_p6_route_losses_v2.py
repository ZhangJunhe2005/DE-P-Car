import torch

from dep_car.model.dep_car_net import DEPCarNetworkOutput
from dep_car.training.losses_v2 import (
    DEPCarObjectiveV2,
    constant_control_future,
    route_tube_loss_v2,
)


def trajectories(y=0.0):
    time = torch.linspace(0.0, 1.0, 11)
    x = torch.linspace(0.0, 0.8, 11)
    row = torch.stack((time, x, torch.full_like(x, y), torch.zeros_like(x), torch.full_like(x, 0.4), torch.zeros_like(x)), dim=-1)
    return row[None, None].expand(1, 15, -1, -1).clone()


def test_route_tube_allows_smooth_offset_but_penalizes_large_corner_cut():
    route = torch.zeros(1, 8, 3)
    route[0, :, 0] = torch.linspace(0.1, 1.6, 8)
    mask = torch.ones(1, 8, dtype=torch.bool)
    near = route_tube_loss_v2(trajectories(0.18), route, mask, DEPCarObjectiveV2().config)
    far = route_tube_loss_v2(trajectories(0.80), route, mask, DEPCarObjectiveV2().config)
    assert float(near.mean()) < float(far.mean())


def test_constant_control_future_extends_from_candidate_endpoint():
    value = trajectories()
    future = constant_control_future(value, DEPCarObjectiveV2().config)
    assert future.shape == (1, 15, 7, 6)
    torch.testing.assert_close(future[..., 0, 1:4], value[..., -1, 1:4])
    assert bool(torch.all(future[..., -1, 1] > future[..., 0, 1]))
