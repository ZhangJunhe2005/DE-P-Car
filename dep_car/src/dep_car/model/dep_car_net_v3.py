"""Joint-gear DE-P-Car V3 with one energy over both Ackermann directions."""

from dataclasses import dataclass
from typing import NamedTuple, Optional

import torch
from torch import nn
from torch.nn import functional as F

from .bidirectional_rollout import (
    BidirectionalAckermannRolloutConfig,
    BidirectionalAckermannRolloutV3,
)
from .dep_car_net import DEPCarNetConfig
from .dep_car_net_v2 import (
    DEPCarNetV2,
    RouteCorridorConfigV2,
    corridor_candidate_relations,
    mirror_route_corridor,
)
from .symmetry import (
    mirror_candidate_values,
    mirror_depth,
    mirror_lidar_bev,
    mirror_scores,
    mirror_vehicle_state,
)


class DEPCarNetworkOutputV3(NamedTuple):
    raw_residuals: torch.Tensor
    residuals: torch.Tensor
    controls: torch.Tensor
    trajectories: torch.Tensor
    scores: torch.Tensor
    candidate_gears: torch.Tensor
    motion_gears: torch.Tensor
    shift_required: torch.Tensor
    transition_duration: torch.Tensor
    forward_recovery_value: torch.Tensor


@dataclass(frozen=True)
class JointGearConfigV3:
    history_dim: int = 6
    history_feature_dim: int = 32
    transition_feature_dim: int = 32


class DEPCarNetV3(DEPCarNetV2):
    """Generate and rank a joint bank of 15 forward + 15 reverse paths.

    Gear selection is implicit in the lowest-energy candidate.  There is no
    separate gear classifier whose decision could disagree with trajectory
    ranking.  ``current_gear`` is retained only as measured actuator state;
    ``gear_history`` provides temporal context but never hard-codes a desired
    next gear.
    """

    architecture_id = "dep_car_multimodal_v3_joint_gear_route_ackermann_2x3x5"
    candidate_order = "gear_forward_then_reverse__speed_major_3x5"
    predicts_gear = True

    def __init__(
        self,
        config: DEPCarNetConfig = DEPCarNetConfig(),
        route_config: RouteCorridorConfigV2 = RouteCorridorConfigV2(),
        joint_config: JointGearConfigV3 = JointGearConfigV3(),
        transition_config: BidirectionalAckermannRolloutConfig = (
            BidirectionalAckermannRolloutConfig()
        ),
        **kwargs,
    ):
        super().__init__(config=config, route_config=route_config, **kwargs)
        if joint_config.history_dim < 1:
            raise ValueError("V3 history_dim must be positive")
        self.joint_config = joint_config
        single_config = self.rollout.config
        self.joint_rollout = BidirectionalAckermannRolloutV3(
            single_config, transition_config
        )
        self.history_encoder = nn.Sequential(
            nn.Linear(joint_config.history_dim, joint_config.history_feature_dim),
            nn.LayerNorm(joint_config.history_feature_dim),
            nn.SiLU(),
            nn.Linear(
                joint_config.history_feature_dim,
                joint_config.history_feature_dim,
            ),
            nn.SiLU(),
        )
        # [gear, shift, transition time, reverse, same previous gear,
        #  route progress] is embedded before being combined with route
        # geometry and temporal state.
        self.transition_encoder = nn.Sequential(
            nn.Linear(6, joint_config.transition_feature_dim),
            nn.SiLU(),
        )
        joint_width = (
            joint_config.history_feature_dim
            + joint_config.transition_feature_dim
            + 6
        )
        self.joint_energy_head = nn.Sequential(
            nn.Linear(joint_width, 96),
            nn.SiLU(),
            nn.Linear(96, 1),
        )
        self.forward_recovery_head = nn.Sequential(
            nn.Linear(joint_width, 96),
            nn.SiLU(),
            nn.Linear(96, 1),
        )
        self._reset_v3_parameters()

    def _reset_v3_parameters(self):
        for module in (
            self.history_encoder,
            self.transition_encoder,
            self.joint_energy_head,
            self.forward_recovery_head,
        ):
            for child in module.modules():
                if isinstance(child, nn.Linear):
                    nn.init.kaiming_uniform_(child.weight, a=5 ** 0.5)
                    if child.bias is not None:
                        nn.init.zeros_(child.bias)

    def initialize_from_v2(self, state_dict):
        """Transfer every shape-compatible V2 tensor; V3 heads stay new."""

        current = self.state_dict()
        transferred = {
            key: value
            for key, value in state_dict.items()
            if key in current
            and current[key].shape == value.shape
            and not key.startswith("joint_rollout.")
        }
        current.update(transferred)
        # The single-bank rollout is parameter-free except for persistent
        # anchor buffers.  Copy those anchors into the V3 joint rollout too.
        for key, value in state_dict.items():
            target = "joint_rollout.single." + key[len("rollout.") :]
            if key.startswith("rollout.") and target in current:
                if current[target].shape == value.shape:
                    current[target] = value
        self.load_state_dict(current, strict=True)
        return tuple(sorted(transferred))

    @staticmethod
    def _mirror_bank_values(values, *, steering_channels=()):
        if values.ndim < 2 or values.shape[1] != 30:
            raise ValueError("V3 candidate tensor must contain 30 candidates")
        forward = mirror_candidate_values(
            values[:, :15], steering_channels=steering_channels
        )
        reverse = mirror_candidate_values(
            values[:, 15:], steering_channels=steering_channels
        )
        return torch.cat((forward, reverse), dim=1)

    @staticmethod
    def _mirror_bank_scores(values):
        if values.shape[1] != 30:
            raise ValueError("V3 score tensor must contain 30 candidates")
        return torch.cat(
            (mirror_scores(values[:, :15]), mirror_scores(values[:, 15:])),
            dim=1,
        )

    def _post_shift_state(self, state, current_gear, desired_gear):
        needs = self.joint_rollout._needs_shift(
            state, current_gear, desired_gear
        )
        _, post, _, _ = self.joint_rollout._transition_prefix(
            state, current_gear, desired_gear, needs
        )
        return post

    def _forward_joint_raw(
        self,
        depth,
        lidar_bev,
        vehicle_state,
        current_gear,
        gear_history,
        modality_mask,
        route_pose,
        route_mask,
    ):
        batch = len(vehicle_state)
        forward_gear = current_gear.new_ones(batch)
        reverse_gear = current_gear.new_full((batch,), -1)
        forward_state = self._post_shift_state(
            vehicle_state, current_gear, 1
        )
        reverse_state = self._post_shift_state(
            vehicle_state, current_gear, -1
        )
        zero = vehicle_state.new_zeros((batch, 15, 4))
        canonical_forward = self._rollout_fp32(
            forward_state, forward_gear, zero
        ).trajectory
        canonical_reverse = self._rollout_fp32(
            reverse_state, reverse_gear, zero
        ).trajectory
        canonical = torch.cat((canonical_forward, canonical_reverse), dim=1)

        depth_present = modality_mask[:, 0] > 0.5
        lidar_present = modality_mask[:, 1] > 0.5
        depth_candidate = self.depth_missing_token.expand(batch, 30, -1)
        depth_indices = torch.nonzero(depth_present, as_tuple=False).flatten()
        if depth_indices.numel():
            depth_map = self.depth_encoder(depth.index_select(0, depth_indices))
            directional = depth_map.mean(dim=-2).flip(-1).transpose(1, 2)
            present = directional[:, None].expand(-1, 3, -1, -1).reshape(
                depth_indices.numel(), 15, -1
            )
            present = torch.cat((present, present), dim=1)
            depth_candidate = depth_candidate.to(present)
            depth_candidate = depth_candidate.index_copy(
                0, depth_indices, present
            )

        lidar_candidate = self.lidar_missing_token.expand(batch, 30, -1)
        lidar_indices = torch.nonzero(lidar_present, as_tuple=False).flatten()
        if lidar_indices.numel():
            bev_map = self.lidar_encoder(
                self.normalize_bev(lidar_bev.index_select(0, lidar_indices))
            )
            selected_canonical = canonical.index_select(0, lidar_indices)
            # Keep the accepted V1/V2 sampler contract byte-for-byte frozen.
            # V3 samples each canonical 15-path bank independently and then
            # concatenates the features into its 30-path joint bank.
            sampled = torch.cat(
                (
                    self.bev_sampler(bev_map, selected_canonical[:, :15]),
                    self.bev_sampler(bev_map, selected_canonical[:, 15:]),
                ),
                dim=1,
            )
            global_bev = F.adaptive_avg_pool2d(bev_map, 1).flatten(1)
            global_bev = global_bev[:, None].expand(-1, 30, -1)
            present_lidar = torch.cat((sampled, global_bev), dim=-1)
            lidar_candidate = lidar_candidate.to(present_lidar)
            lidar_candidate = lidar_candidate.index_copy(
                0, lidar_indices, present_lidar
            )

        state_feature = self.state_encoder(
            self.normalize_state(vehicle_state)
        )[:, None].expand(-1, 30, -1)
        candidate_gears = torch.cat(
            (forward_gear[:, None].expand(-1, 15), reverse_gear[:, None].expand(-1, 15)),
            dim=1,
        )
        gear_feature = self.gear_embedding(
            (candidate_gears > 0).long()
        )
        logical_queries = torch.cat(
            (
                self._logical_queries(batch, vehicle_state.device),
                self._logical_queries(batch, vehicle_state.device),
            ),
            dim=1,
        )
        base = torch.cat(
            (
                depth_candidate,
                lidar_candidate,
                state_feature,
                gear_feature,
                logical_queries,
            ),
            dim=-1,
        )
        route_global = self.route_encoder(route_pose, route_mask)
        route_global = route_global[:, None].expand(-1, 30, -1)
        canonical_relation = self.candidate_route_relation_encoder(
            corridor_candidate_relations(canonical, route_pose, route_mask)
        )
        candidate_fused = torch.cat(
            (base, route_global, canonical_relation), dim=-1
        )
        raw = self.candidate_head(self.candidate_tower(candidate_fused))
        rollout = self.joint_rollout(vehicle_state, current_gear, raw)
        bounded = rollout.residuals.detach()
        geometry = self.score_geometry_encoder(bounded)
        actual_relation_raw = corridor_candidate_relations(
            rollout.trajectory, route_pose, route_mask
        )
        actual_relation = self.score_route_relation_encoder(
            actual_relation_raw
        )
        base_score = self.score_head(
            self.score_tower(
                torch.cat(
                    (candidate_fused, geometry, actual_relation), dim=-1
                )
            )
        ).squeeze(-1)

        history = self.history_encoder(gear_history)
        history = history[:, None].expand(-1, 30, -1)
        previous_gear = gear_history[:, 0:1]
        transition_raw = torch.stack(
            (
                candidate_gears.to(vehicle_state),
                rollout.shift_required.to(vehicle_state),
                (rollout.transition_duration / 2.0).clamp(0.0, 2.0),
                (candidate_gears < 0).to(vehicle_state),
                (candidate_gears.to(vehicle_state) == previous_gear).to(
                    vehicle_state
                ),
                actual_relation_raw[..., 3],
            ),
            dim=-1,
        )
        transition = self.transition_encoder(transition_raw)
        joint = torch.cat((history, transition, actual_relation_raw), dim=-1)
        energy = base_score + self.joint_energy_head(joint).squeeze(-1)
        recovery = torch.sigmoid(
            self.forward_recovery_head(joint).squeeze(-1)
        )
        return raw, energy, recovery, rollout

    def forward(
        self,
        depth,
        lidar_bev,
        vehicle_state,
        current_gear,
        gear_history,
        route_pose,
        route_mask,
        modality_mask: Optional[torch.Tensor] = None,
    ):
        batch = len(vehicle_state)
        if current_gear.shape != (batch,):
            raise ValueError("current_gear must have shape [B]")
        if gear_history.shape != (batch, self.joint_config.history_dim):
            raise ValueError(
                "gear_history must have shape [B,%d]"
                % self.joint_config.history_dim
            )
        if not bool(torch.isfinite(gear_history).all()):
            raise ValueError("gear_history must be finite")
        if modality_mask is None:
            modality_mask = vehicle_state.new_ones((batch, 2))
        self._validate_modality_mask(modality_mask, batch)
        if route_pose.shape[0] != batch or route_mask.shape != route_pose.shape[:2]:
            raise ValueError("route corridor batch shape mismatch")

        raw, energy, recovery, _ = self._forward_joint_raw(
            depth,
            lidar_bev,
            vehicle_state,
            current_gear,
            gear_history,
            modality_mask,
            route_pose,
            route_mask,
        )
        if self.config.enforce_reflection_equivariance:
            mirrored_raw, mirrored_energy, mirrored_recovery, _ = (
                self._forward_joint_raw(
                    mirror_depth(depth),
                    mirror_lidar_bev(lidar_bev),
                    mirror_vehicle_state(vehicle_state),
                    current_gear,
                    gear_history,
                    modality_mask,
                    mirror_route_corridor(route_pose),
                    route_mask,
                )
            )
            raw = 0.5 * (
                raw
                + self._mirror_bank_values(
                    mirrored_raw, steering_channels=(0, 1)
                )
            )
            energy = 0.5 * (
                energy + self._mirror_bank_scores(mirrored_energy)
            )
            recovery = 0.5 * (
                recovery + self._mirror_bank_scores(mirrored_recovery)
            )
        with torch.autocast(device_type=vehicle_state.device.type, enabled=False):
            rollout = self.joint_rollout(
                vehicle_state.float(), current_gear, raw.float()
            )
            return DEPCarNetworkOutputV3(
                raw.float(),
                rollout.residuals,
                rollout.controls,
                rollout.trajectory,
                F.softplus(energy.float()),
                rollout.candidate_gears,
                rollout.motion_gears,
                rollout.shift_required,
                rollout.transition_duration,
                recovery.float(),
            )

    def score_parameters(self):
        yield from super().score_parameters()
        yield from self.history_encoder.parameters()
        yield from self.transition_encoder.parameters()
        yield from self.joint_energy_head.parameters()
        yield from self.forward_recovery_head.parameters()


__all__ = [
    "DEPCarNetV3",
    "DEPCarNetworkOutputV3",
    "JointGearConfigV3",
]
