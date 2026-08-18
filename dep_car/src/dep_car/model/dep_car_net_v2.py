"""Route-conditioned DE-P-Car V2 for learned Ackermann corner negotiation."""

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from .dep_car_net import DEPCarNetConfig, DEPCarNetV1, DEPCarNetworkOutput
from .symmetry import (
    mirror_candidate_values,
    mirror_depth,
    mirror_lidar_bev,
    mirror_scores,
    mirror_vehicle_state,
)


@dataclass(frozen=True)
class RouteCorridorConfigV2:
    coordinate_scale_m: float = 3.0
    global_feature_dim: int = 64
    relation_feature_dim: int = 32


def mirror_route_corridor(route_pose):
    mirrored = route_pose.clone()
    mirrored[..., 1] = -mirrored[..., 1]
    mirrored[..., 2] = -mirrored[..., 2]
    return mirrored


class RouteCorridorEncoderV2(nn.Module):
    """Masked point encoder for a 2.5--3 m body-frame route corridor."""

    def __init__(self, config=RouteCorridorConfigV2()):
        super().__init__()
        self.config = config
        self.point = nn.Sequential(
            nn.Linear(5, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU()
        )
        self.pool = nn.Sequential(
            nn.Linear(128, config.global_feature_dim), nn.SiLU()
        )

    def forward(self, route_pose, route_mask):
        if route_pose.ndim != 3 or route_pose.shape[-1] != 3:
            raise ValueError("route_pose must have shape [B,R,3]")
        if route_mask.shape != route_pose.shape[:2]:
            raise ValueError("route_mask shape must match route_pose")
        if bool(torch.any(route_mask.sum(dim=1) < 2)):
            raise ValueError("each route corridor requires at least two points")
        values = torch.stack(
            (
                route_pose[..., 0] / self.config.coordinate_scale_m,
                route_pose[..., 1] / self.config.coordinate_scale_m,
                torch.sin(route_pose[..., 2]),
                torch.cos(route_pose[..., 2]),
                torch.linalg.vector_norm(route_pose[..., :2], dim=-1)
                / self.config.coordinate_scale_m,
            ),
            dim=-1,
        )
        encoded = self.point(values)
        weights = route_mask.to(encoded)[..., None]
        mean = (encoded * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        maximum = encoded.masked_fill(~route_mask[..., None], -1.0e4).amax(dim=1)
        return self.pool(torch.cat((mean, maximum), dim=-1))


def corridor_candidate_relations(trajectories, route_pose, route_mask):
    """Candidate-specific smooth route-tube geometry [B,C,6]."""

    if trajectories.ndim != 4 or trajectories.shape[-1] != 6:
        raise ValueError("trajectories must have shape [B,C,T,6]")
    origin = route_pose.new_zeros((len(route_pose), 1, 3))
    route = torch.cat((origin, route_pose), dim=1)
    mask = torch.cat(
        (
            torch.ones((len(route_mask), 1), dtype=torch.bool, device=route_mask.device),
            route_mask,
        ),
        dim=1,
    )
    starts = route[:, :-1, :2]
    segments = route[:, 1:, :2] - starts
    lengths = torch.linalg.vector_norm(segments, dim=-1)
    valid = (mask[:, 1:] & mask[:, :-1]) & (lengths > 1.0e-5)
    positions = trajectories[..., 1:3]
    delta = positions[..., None, :] - starts[:, None, None, :, :]
    projection = (
        (delta * segments[:, None, None, :, :]).sum(dim=-1)
        / lengths.square().clamp_min(1.0e-8)[:, None, None, :]
    ).clamp(0.0, 1.0)
    closest = starts[:, None, None, :, :] + projection[..., None] * segments[
        :, None, None, :, :
    ]
    distance = torch.linalg.vector_norm(
        positions[..., None, :] - closest, dim=-1
    ).masked_fill(~valid[:, None, None, :], 1.0e4)
    cross_track, nearest = distance.min(dim=-1)
    lengths = lengths * valid.to(lengths)
    arc_start = torch.cat(
        (lengths.new_zeros((len(route), 1)), torch.cumsum(lengths, dim=1)[:, :-1]),
        dim=1,
    )
    progress_all = arc_start[:, None, None, :] + projection * lengths[
        :, None, None, :
    ]
    progress = torch.gather(progress_all, -1, nearest[..., None]).squeeze(-1)
    nearest_segment = torch.gather(
        segments[:, None, :, :].expand(-1, trajectories.shape[1], -1, -1),
        2,
        nearest[..., -1, None, None].expand(-1, -1, 1, 2),
    ).squeeze(2)
    route_heading = torch.atan2(nearest_segment[..., 1], nearest_segment[..., 0])
    heading_error = torch.atan2(
        torch.sin(trajectories[..., -1, 3] - route_heading),
        torch.cos(trajectories[..., -1, 3] - route_heading),
    )
    scale = 3.0
    return torch.stack(
        (
            cross_track[..., 1:].mean(dim=-1) / scale,
            cross_track[..., 1:].amax(dim=-1) / scale,
            cross_track[..., -1] / scale,
            progress[..., -1] / scale,
            torch.sin(heading_error),
            torch.cos(heading_error),
        ),
        dim=-1,
    )


class DEPCarNetV2(DEPCarNetV1):
    """Condition both candidate generation and ranking on the route corridor."""

    architecture_id = "dep_car_multimodal_v2_route_ackermann_3x5"

    def __init__(
        self,
        config: DEPCarNetConfig = DEPCarNetConfig(),
        route_config: RouteCorridorConfigV2 = RouteCorridorConfigV2(),
        **kwargs,
    ):
        super().__init__(config=config, **kwargs)
        base_fused_dim = self.candidate_tower[0].in_features
        route_dim = route_config.global_feature_dim
        relation_dim = route_config.relation_feature_dim
        self.route_encoder = RouteCorridorEncoderV2(route_config)
        self.candidate_route_relation_encoder = nn.Sequential(
            nn.Linear(6, relation_dim), nn.SiLU()
        )
        self.score_route_relation_encoder = nn.Sequential(
            nn.Linear(6, relation_dim), nn.SiLU()
        )
        candidate_fused_dim = base_fused_dim + route_dim + relation_dim
        self.candidate_tower = nn.Sequential(
            nn.Linear(candidate_fused_dim, 256), nn.SiLU(),
            nn.Linear(256, 256), nn.SiLU(),
        )
        self.score_tower = nn.Sequential(
            nn.Linear(candidate_fused_dim + 32 + relation_dim, 256), nn.SiLU(),
            nn.Linear(256, 256), nn.SiLU(),
        )
        self._reset_v2_parameters()

    def _reset_v2_parameters(self):
        for module in (
            self.route_encoder,
            self.candidate_route_relation_encoder,
            self.score_route_relation_encoder,
            self.candidate_tower,
            self.score_tower,
        ):
            for child in module.modules():
                if isinstance(child, nn.Linear):
                    nn.init.kaiming_uniform_(child.weight, a=5 ** 0.5)
                    if child.bias is not None:
                        nn.init.zeros_(child.bias)

    def initialize_from_v1(self, state_dict):
        """Transfer V1 exactly, with new route channels initially neutral."""

        current = self.state_dict()
        direct = {
            key: value
            for key, value in state_dict.items()
            if key in current
            and current[key].shape == value.shape
            and not key.startswith(("candidate_tower.", "score_tower."))
        }
        current.update(direct)
        base_dim = state_dict["candidate_tower.0.weight"].shape[1]
        for tower in ("candidate_tower", "score_tower"):
            old_weight = state_dict[tower + ".0.weight"]
            old_bias = state_dict[tower + ".0.bias"]
            new_weight = current[tower + ".0.weight"]
            new_weight.zero_()
            if tower == "candidate_tower":
                new_weight[:, :base_dim].copy_(old_weight)
            else:
                # V1 order is [base, geometry]; V2 is
                # [base, route-global, canonical-route, geometry, actual-route].
                route_width = current["candidate_tower.0.weight"].shape[1] - base_dim
                new_weight[:, :base_dim].copy_(old_weight[:, :base_dim])
                new_weight[:, base_dim + route_width : base_dim + route_width + 32].copy_(
                    old_weight[:, base_dim:]
                )
            current[tower + ".0.bias"].copy_(old_bias)
            for suffix in ("2.weight", "2.bias"):
                current[tower + "." + suffix].copy_(state_dict[tower + "." + suffix])
        self.load_state_dict(current, strict=True)

    def _forward_route_raw(
        self, depth, lidar_bev, vehicle_state, requested_gear, modality_mask,
        route_pose, route_mask,
    ):
        batch = vehicle_state.shape[0]
        depth_present = modality_mask[:, 0] > 0.5
        lidar_present = modality_mask[:, 1] > 0.5
        depth_candidate = self.depth_missing_token.expand(batch, 15, -1)
        depth_indices = torch.nonzero(depth_present, as_tuple=False).flatten()
        if depth_indices.numel():
            depth_map = self.depth_encoder(depth.index_select(0, depth_indices))
            directional = depth_map.mean(dim=-2).flip(-1).transpose(1, 2)
            present = directional[:, None].expand(-1, 3, -1, -1).reshape(
                depth_indices.numel(), 15, -1
            )
            depth_candidate = depth_candidate.index_copy(0, depth_indices, present)

        canonical = self._rollout_fp32(
            vehicle_state, requested_gear, vehicle_state.new_zeros((batch, 15, 4))
        ).trajectory
        lidar_candidate = self.lidar_missing_token.expand(batch, 15, -1)
        lidar_indices = torch.nonzero(lidar_present, as_tuple=False).flatten()
        if lidar_indices.numel():
            bev_map = self.lidar_encoder(
                self.normalize_bev(lidar_bev.index_select(0, lidar_indices))
            )
            sampled = self.bev_sampler(bev_map, canonical.index_select(0, lidar_indices))
            global_bev = F.adaptive_avg_pool2d(bev_map, 1).flatten(1)[:, None].expand(
                -1, 15, -1
            )
            lidar_candidate = lidar_candidate.index_copy(
                0, lidar_indices, torch.cat((sampled, global_bev), dim=-1)
            )
        state_feature = self.state_encoder(self.normalize_state(vehicle_state))[:, None].expand(-1, 15, -1)
        gear_feature = self.gear_embedding((requested_gear > 0).long())[:, None].expand(-1, 15, -1)
        base = torch.cat(
            (depth_candidate, lidar_candidate, state_feature, gear_feature,
             self._logical_queries(batch, vehicle_state.device)), dim=-1
        )
        route_global = self.route_encoder(route_pose, route_mask)[:, None].expand(-1, 15, -1)
        canonical_relation = self.candidate_route_relation_encoder(
            corridor_candidate_relations(canonical, route_pose, route_mask)
        )
        candidate_fused = torch.cat((base, route_global, canonical_relation), dim=-1)
        raw_residuals = self.candidate_head(self.candidate_tower(candidate_fused))
        rollout = self._rollout_fp32(vehicle_state, requested_gear, raw_residuals)
        with torch.autocast(device_type=vehicle_state.device.type, enabled=False):
            bounded = self.rollout.bound_residuals(raw_residuals.float(), requested_gear).detach()
        geometry = self.score_geometry_encoder(bounded)
        actual_relation = self.score_route_relation_encoder(
            corridor_candidate_relations(rollout.trajectory, route_pose, route_mask)
        )
        score = self.score_head(
            self.score_tower(torch.cat((candidate_fused, geometry, actual_relation), dim=-1))
        ).squeeze(-1)
        return raw_residuals, score

    def forward(
        self,
        depth,
        lidar_bev,
        vehicle_state,
        requested_gear,
        route_pose,
        route_mask,
        modality_mask: Optional[torch.Tensor] = None,
    ):
        batch = vehicle_state.shape[0]
        if modality_mask is None:
            modality_mask = vehicle_state.new_ones((batch, 2))
        self._validate_modality_mask(modality_mask, batch)
        if route_pose.shape[0] != batch or route_mask.shape != route_pose.shape[:2]:
            raise ValueError("route corridor batch shape mismatch")
        raw, score = self._forward_route_raw(
            depth, lidar_bev, vehicle_state, requested_gear, modality_mask,
            route_pose, route_mask,
        )
        if self.config.enforce_reflection_equivariance:
            mirrored_raw, mirrored_score = self._forward_route_raw(
                mirror_depth(depth), mirror_lidar_bev(lidar_bev),
                mirror_vehicle_state(vehicle_state), requested_gear, modality_mask,
                mirror_route_corridor(route_pose), route_mask,
            )
            raw = 0.5 * (
                raw + mirror_candidate_values(mirrored_raw, steering_channels=(0, 1))
            )
            score = 0.5 * (score + mirror_scores(mirrored_score))
        rollout = self._rollout_fp32(vehicle_state, requested_gear, raw)
        with torch.autocast(device_type=vehicle_state.device.type, enabled=False):
            return DEPCarNetworkOutput(
                raw.float(), rollout.residuals, rollout.controls,
                rollout.trajectory, F.softplus(score.float())
            )

    def candidate_parameters(self):
        yield from super().candidate_parameters()
        yield from self.route_encoder.parameters()
        yield from self.candidate_route_relation_encoder.parameters()

    def score_parameters(self):
        yield from super().score_parameters()
        yield from self.score_route_relation_encoder.parameters()


__all__ = [
    "DEPCarNetV2",
    "RouteCorridorConfigV2",
    "RouteCorridorEncoderV2",
    "corridor_candidate_relations",
    "mirror_route_corridor",
]
