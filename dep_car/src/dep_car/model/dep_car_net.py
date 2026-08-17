"""Formal P4 multimodal DE-P-Car network.

The network preserves DE-P's multi-candidate planning idea while replacing the
UAV primitive frame with logical speed/steering queries and a signed Ackermann
rollout.  Discrete gear remains external deterministic state.
"""

from dataclasses import dataclass
from typing import NamedTuple, Optional

import torch
from torch import nn
from torch.nn import functional as F

from dep_car.core.state_contract import STATE_FIELDS, STATE_NORMALIZATION_SCALE

from .ackermann_rollout import AckermannRolloutConfig, AckermannRolloutV1
from .lidar_dep import load_frozen_backbone
from .symmetry import (
    mirror_candidate_values,
    mirror_depth,
    mirror_lidar_bev,
    mirror_scores,
    mirror_vehicle_state,
)


class DEPCarNetworkOutput(NamedTuple):
    raw_residuals: torch.Tensor
    residuals: torch.Tensor
    controls: torch.Tensor
    trajectories: torch.Tensor
    scores: torch.Tensor


@dataclass(frozen=True)
class DEPCarNetConfig:
    depth_size: tuple = (96, 160)
    depth_max_m: float = 10.0
    lidar_channels: int = 6
    lidar_size: tuple = (160, 160)
    lidar_extent_m: float = 8.0
    state_dim: int = 9
    feature_dim: int = 64
    state_feature_dim: int = 64
    gear_feature_dim: int = 16
    query_feature_dim: int = 64
    enforce_reflection_equivariance: bool = True


class DepthEncoderV1(nn.Module):
    """Mask-aware normalized-depth encoder initialized from the DE-P backbone."""

    def __init__(self, output_dim=64, input_size=(96, 160), maximum_depth_m=10.0, source_tree=None):
        super().__init__()
        self.input_size = tuple(input_size)
        self.maximum_depth_m = float(maximum_depth_m)
        self.image_backbone = load_frozen_backbone(
            output_dim, source_tree=source_tree, input_size=self.input_size
        )
        self.validity_encoder = nn.Sequential(
            ConvNormAct(1, 16, stride=2),
            ConvNormAct(16, 32, stride=2),
            ConvNormAct(32, output_dim, stride=2),
            nn.AdaptiveAvgPool2d((3, 5)),
        )

    def forward(self, depth_metric_and_validity):
        if depth_metric_and_validity.ndim != 4 or depth_metric_and_validity.shape[1] != 2:
            raise ValueError("depth input must have shape [B,2,H,W] (depth/10m, validity)")
        normalized = depth_metric_and_validity[:, :1]
        validity = depth_metric_and_validity[:, 1:2]
        normalized = F.interpolate(normalized, size=self.input_size, mode="bilinear", align_corners=False)
        validity = F.interpolate(validity, size=self.input_size, mode="nearest") > 0.5
        normalized = normalized.clamp(0.0, 1.0)
        # Invalid pixels have a distinct validity contract before being mapped
        # to the safe far-depth fill used by the one-channel legacy stem.
        normalized = torch.where(validity, normalized, torch.ones_like(normalized))
        return self.image_backbone(normalized) + self.validity_encoder(validity.to(normalized.dtype))


class ConvNormAct(nn.Sequential):
    def __init__(self, input_channels, output_channels, stride=1):
        super().__init__(
            nn.Conv2d(input_channels, output_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.SiLU(inplace=True),
        )


class LidarBEVEncoderV1(nn.Module):
    """Small 360-degree BEV CNN; no image-grid/candidate-grid conflation."""

    def __init__(self, input_channels=6, output_dim=64):
        super().__init__()
        self.network = nn.Sequential(
            ConvNormAct(input_channels, 24, stride=2),
            ConvNormAct(24, 32, stride=2),
            ConvNormAct(32, 48, stride=2),
            ConvNormAct(48, output_dim, stride=1),
        )

    def forward(self, lidar_bev):
        if lidar_bev.ndim != 4 or lidar_bev.shape[1] != 6:
            raise ValueError("LiDAR BEV must have shape [B,6,H,W]")
        return self.network(lidar_bev)


class TrajectoryConditionedBEVSamplerV1(nn.Module):
    """Pool BEV features along each canonical Ackermann candidate."""

    def __init__(self, extent_m=8.0):
        super().__init__()
        self.extent_m = float(extent_m)
        if self.extent_m <= 0.0:
            raise ValueError("BEV extent must be positive")

    def forward(self, feature_map, canonical_trajectory):
        if canonical_trajectory.ndim != 4 or canonical_trajectory.shape[1] != 15:
            raise ValueError("canonical trajectory must have shape [B,15,T,6]")
        xy = canonical_trajectory[..., 1:3]
        # BEV storage is [body-y, body-x]. grid_sample expects [W, H].
        grid = torch.stack((xy[..., 0], xy[..., 1]), dim=-1) / self.extent_m
        sampled = F.grid_sample(
            feature_map, grid, mode="bilinear", padding_mode="zeros", align_corners=False
        )
        return sampled.mean(dim=-1).transpose(1, 2)


class DEPCarNetV1(nn.Module):
    architecture_id = "dep_car_multimodal_v1_ackermann_3x5"
    state_fields = STATE_FIELDS
    candidate_order = "speed_major_3x5_steering_minor"
    predicts_gear = False

    def __init__(
        self,
        config: DEPCarNetConfig = DEPCarNetConfig(),
        rollout_config: AckermannRolloutConfig = AckermannRolloutConfig(),
        source_tree=None,
    ):
        super().__init__()
        if config.state_dim != len(self.state_fields):
            raise ValueError("DE-P-Car V1 state contract is exactly 9D")
        self.config = config
        self.rollout = AckermannRolloutV1(rollout_config)
        self.depth_encoder = DepthEncoderV1(
            config.feature_dim,
            config.depth_size,
            config.depth_max_m,
            source_tree=source_tree,
        )
        self.lidar_encoder = LidarBEVEncoderV1(config.lidar_channels, config.feature_dim)
        self.bev_sampler = TrajectoryConditionedBEVSamplerV1(config.lidar_extent_m)
        self.state_encoder = nn.Sequential(
            nn.Linear(config.state_dim, config.state_feature_dim),
            nn.LayerNorm(config.state_feature_dim),
            nn.SiLU(),
            nn.Linear(config.state_feature_dim, config.state_feature_dim),
            nn.SiLU(),
        )
        self.gear_embedding = nn.Embedding(2, config.gear_feature_dim)
        self.speed_embedding = nn.Embedding(3, config.query_feature_dim // 2)
        self.steering_embedding = nn.Embedding(5, config.query_feature_dim // 2)
        query_dim = 2 * (config.query_feature_dim // 2)
        fused_dim = (
            config.feature_dim
            + 2 * config.feature_dim
            + config.state_feature_dim
            + config.gear_feature_dim
            + query_dim
        )
        self.depth_missing_token = nn.Parameter(torch.zeros(1, 1, config.feature_dim))
        self.lidar_missing_token = nn.Parameter(torch.zeros(1, 1, 2 * config.feature_dim))
        self.candidate_tower = nn.Sequential(
            nn.Linear(fused_dim, 256), nn.SiLU(),
            nn.Linear(256, 256), nn.SiLU(),
        )
        self.candidate_head = nn.Linear(256, 4)
        self.score_geometry_encoder = nn.Sequential(nn.Linear(4, 32), nn.SiLU())
        self.score_tower = nn.Sequential(
            nn.Linear(fused_dim + 32, 256), nn.SiLU(),
            nn.Linear(256, 256), nn.SiLU(),
        )
        self.score_head = nn.Linear(256, 1)
        speed_ids = torch.arange(3).repeat_interleave(5)
        steering_ids = torch.arange(5).repeat(3)
        self.register_buffer("speed_query_ids", speed_ids, persistent=True)
        self.register_buffer("steering_query_ids", steering_ids, persistent=True)
        self.reset_new_parameters()

    def reset_new_parameters(self):
        """Deterministically-friendly initialization; caller controls RNG seed."""

        for module in (
            self.depth_encoder.validity_encoder,
            self.lidar_encoder,
            self.state_encoder,
            self.candidate_tower,
            self.candidate_head,
            self.score_geometry_encoder,
            self.score_tower,
            self.score_head,
        ):
            for child in module.modules():
                if isinstance(child, (nn.Linear, nn.Conv2d)):
                    nn.init.kaiming_uniform_(child.weight, a=5 ** 0.5)
                    if child.bias is not None:
                        nn.init.zeros_(child.bias)
        nn.init.normal_(self.gear_embedding.weight, std=0.02)
        nn.init.normal_(self.speed_embedding.weight, std=0.02)
        nn.init.normal_(self.steering_embedding.weight, std=0.02)
        nn.init.normal_(self.candidate_head.weight, std=1.0e-3)
        nn.init.zeros_(self.candidate_head.bias)
        nn.init.zeros_(self.score_head.bias)

    @staticmethod
    def _validate_modality_mask(modality_mask, batch):
        if modality_mask is None:
            return None
        if modality_mask.ndim != 2 or modality_mask.shape != (batch, 2):
            raise ValueError("modality_mask must have shape [B,2] for depth and LiDAR")
        if not bool(torch.all((modality_mask == 0) | (modality_mask == 1))):
            raise ValueError("modality_mask values must be zero or one")
        if bool(torch.any(modality_mask.sum(dim=1) < 1)):
            raise ValueError("at least one sensor modality must be available per sample")
        return modality_mask

    @staticmethod
    def normalize_state(vehicle_state):
        scale = vehicle_state.new_tensor(STATE_NORMALIZATION_SCALE)
        normalized = vehicle_state / scale
        return normalized.clamp(-2.0, 2.0)

    @staticmethod
    def normalize_bev(lidar_bev):
        # P3TrainingDatasetV1 freezes all six channels to [0,1], including
        # occupancy-masked height normalization.  The model only enforces that
        # boundary and must not normalize an already normalized BEV twice.
        return lidar_bev.clamp(0.0, 1.0)

    def _logical_queries(self, batch, device):
        speed = self.speed_embedding(self.speed_query_ids.to(device))
        steering = self.steering_embedding(self.steering_query_ids.to(device))
        return torch.cat((speed, steering), dim=-1)[None, :, :].expand(batch, -1, -1)

    def _forward_raw(self, depth, lidar_bev, vehicle_state, requested_gear, modality_mask):
        batch = vehicle_state.shape[0]
        depth_present = modality_mask[:, 0] > 0.5
        lidar_present = modality_mask[:, 1] > 0.5

        # Only present rows enter an encoder.  Apart from saving ablation
        # compute, this prevents zero-filled missing rows from changing
        # BatchNorm running statistics during fusion sensor dropout.
        depth_candidate = self.depth_missing_token.expand(batch, 15, -1)
        depth_indices = torch.nonzero(depth_present, as_tuple=False).flatten()
        if depth_indices.numel():
            depth_map = self.depth_encoder(depth.index_select(0, depth_indices))
            # Candidate steering -0.52 (right) reads image right; +0.52 reads left.
            depth_directional = depth_map.mean(dim=-2).flip(-1).transpose(1, 2)
            present_depth = depth_directional[:, None, :, :].expand(
                -1, 3, -1, -1
            ).reshape(depth_indices.numel(), 15, -1)
            depth_candidate = depth_candidate.index_copy(
                0, depth_indices, present_depth
            )

        canonical = self.rollout(
            vehicle_state,
            requested_gear,
            vehicle_state.new_zeros((batch, 15, 4)),
        ).trajectory
        lidar_candidate = self.lidar_missing_token.expand(batch, 15, -1)
        lidar_indices = torch.nonzero(lidar_present, as_tuple=False).flatten()
        if lidar_indices.numel():
            bev_map = self.lidar_encoder(
                self.normalize_bev(lidar_bev.index_select(0, lidar_indices))
            )
            present_canonical = canonical.index_select(0, lidar_indices)
            lidar_sampled = self.bev_sampler(bev_map, present_canonical)
            lidar_global = F.adaptive_avg_pool2d(bev_map, 1).flatten(1)[:, None, :].expand(
                -1, 15, -1
            )
            present_lidar = torch.cat((lidar_sampled, lidar_global), dim=-1)
            lidar_candidate = lidar_candidate.index_copy(
                0, lidar_indices, present_lidar
            )

        state_feature = self.state_encoder(self.normalize_state(vehicle_state))[:, None, :].expand(-1, 15, -1)
        gear_index = (requested_gear > 0).long()
        gear_feature = self.gear_embedding(gear_index)[:, None, :].expand(-1, 15, -1)
        query = self._logical_queries(batch, vehicle_state.device)
        fused = torch.cat((depth_candidate, lidar_candidate, state_feature, gear_feature, query), dim=-1)
        raw_residuals = self.candidate_head(self.candidate_tower(fused))
        # Score calibration consumes the bounded physical residual contract,
        # not unbounded logits whose magnitude can grow while tanh leaves the
        # executable trajectory unchanged.
        score_geometry_input = self.rollout.bound_residuals(
            raw_residuals, requested_gear
        ).detach()
        score_geometry = self.score_geometry_encoder(score_geometry_input)
        raw_score = self.score_head(self.score_tower(torch.cat((fused, score_geometry), dim=-1))).squeeze(-1)
        return raw_residuals, raw_score

    def forward(
        self,
        depth,
        lidar_bev,
        vehicle_state,
        requested_gear,
        modality_mask: Optional[torch.Tensor] = None,
    ):
        if vehicle_state.ndim != 2 or vehicle_state.shape[1] != self.config.state_dim:
            raise ValueError("vehicle_state must have shape [B,9]")
        batch = vehicle_state.shape[0]
        if depth.ndim != 4 or depth.shape[0] != batch or depth.shape[1] != 2:
            raise ValueError("depth must have shape [B,2,H,W]")
        if lidar_bev.ndim != 4 or lidar_bev.shape[0] != batch or lidar_bev.shape[1] != 6:
            raise ValueError("lidar_bev must have shape [B,6,H,W]")
        if requested_gear.ndim != 1 or requested_gear.shape[0] != batch:
            raise ValueError("requested_gear must have shape [B]")
        if not bool(torch.all((requested_gear == -1) | (requested_gear == 1))):
            raise ValueError("requested_gear must contain only -1 or +1")
        if modality_mask is None:
            modality_mask = vehicle_state.new_ones((batch, 2))
        self._validate_modality_mask(modality_mask, batch)
        modality_mask = modality_mask.to(device=vehicle_state.device)

        raw_residuals, raw_score = self._forward_raw(
            depth, lidar_bev, vehicle_state, requested_gear, modality_mask
        )
        if self.config.enforce_reflection_equivariance:
            mirrored_residuals, mirrored_score = self._forward_raw(
                mirror_depth(depth),
                mirror_lidar_bev(lidar_bev),
                mirror_vehicle_state(vehicle_state),
                requested_gear,
                modality_mask,
            )
            mirrored_residuals = mirror_candidate_values(
                mirrored_residuals, steering_channels=(0, 1)
            )
            mirrored_score = mirror_scores(mirrored_score)
            raw_residuals = 0.5 * (raw_residuals + mirrored_residuals)
            raw_score = 0.5 * (raw_score + mirrored_score)

        rollout = self.rollout(vehicle_state, requested_gear, raw_residuals)
        return DEPCarNetworkOutput(
            raw_residuals=raw_residuals,
            residuals=rollout.residuals,
            controls=rollout.controls,
            trajectories=rollout.trajectory,
            scores=F.softplus(raw_score),
        )

    def candidate_parameters(self):
        modules = (
            self.depth_encoder,
            self.lidar_encoder,
            self.state_encoder,
            self.gear_embedding,
            self.speed_embedding,
            self.steering_embedding,
            self.candidate_tower,
            self.candidate_head,
        )
        for module in modules:
            yield from module.parameters()
        yield self.depth_missing_token
        yield self.lidar_missing_token

    def score_parameters(self):
        for module in (self.score_geometry_encoder, self.score_tower, self.score_head):
            yield from module.parameters()
