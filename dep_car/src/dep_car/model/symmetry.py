"""Frozen left/right reflection rules for DE-P-Car tensors."""

import torch


STATE_ODD_INDICES = (2, 3, 5, 6, 8)


def candidate_reflection_permutation(device=None):
    """Speed-major permutation: each steering row is reversed."""

    return torch.arange(15, device=device).reshape(3, 5).flip(1).reshape(-1)


def mirror_depth(depth):
    return depth.flip(-1)


def mirror_lidar_bev(lidar_bev):
    # BEV layout is [C, body-y, body-x], so left/right is the H/y axis.
    return lidar_bev.flip(-2)


def mirror_vehicle_state(vehicle_state):
    mirrored = vehicle_state.clone()
    mirrored[..., list(STATE_ODD_INDICES)] = -mirrored[..., list(STATE_ODD_INDICES)]
    return mirrored


def mirror_route(route):
    mirrored = route.clone()
    mirrored[..., 1] = -mirrored[..., 1]
    mirrored[..., 2] = -mirrored[..., 2]
    return mirrored


def mirror_candidate_values(values, steering_channels=(), candidate_dim=-2):
    """Reflect candidate order and selected channels of a candidate tensor."""

    permutation = candidate_reflection_permutation(values.device)
    mirrored = values.index_select(candidate_dim, permutation).clone()
    if steering_channels:
        mirrored[..., list(steering_channels)] = -mirrored[..., list(steering_channels)]
    return mirrored


def mirror_scores(scores):
    return scores.index_select(-1, candidate_reflection_permutation(scores.device))


def mirror_trajectory(trajectory):
    """Reflect [t,x,y,yaw,v,steering] and candidate ordering."""

    mirrored = mirror_candidate_values(trajectory, candidate_dim=-3)
    mirrored[..., 2] = -mirrored[..., 2]
    mirrored[..., 3] = -mirrored[..., 3]
    mirrored[..., 5] = -mirrored[..., 5]
    return mirrored
