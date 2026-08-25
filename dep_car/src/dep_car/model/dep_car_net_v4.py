"""DEPCarNetV4: unified hybrid gear/control sequence candidates."""

from dataclasses import dataclass
from typing import NamedTuple, Optional

import torch
from torch import nn
from torch.nn import functional as F

from dep_car.core.state_contract import STATE_NORMALIZATION_SCALE

from .dep_car_net_v2 import corridor_candidate_relations
from .dep_car_net_v3 import DEPCarNetV3
from .hybrid_sequence_rollout import (
    HybridSequenceAckermannRolloutV4,
    HybridSequenceRolloutConfigV4,
)


class DEPCarNetworkOutputV4(NamedTuple):
    raw_controls: torch.Tensor
    gear_logits: torch.Tensor
    controls: torch.Tensor
    trajectories: torch.Tensor
    scores: torch.Tensor
    safety_logits: torch.Tensor
    viability_logits: torch.Tensor
    gear_tokens: torch.Tensor
    action_gears: torch.Tensor
    action_mask: torch.Tensor
    motion_gears: torch.Tensor
    shift_required: torch.Tensor
    transition_duration: torch.Tensor
    primitive_scores: torch.Tensor


@dataclass(frozen=True)
class HybridSequenceConfigV4:
    candidates: int = 15
    actions: int = 6
    hidden_dim: int = 128
    primitive_dim: int = 64
    stage_dim: int = 32
    template_logit_bias: float = 1.25

    def validate(self):
        if self.candidates != 15 or self.actions != 6:
            raise ValueError("DEPCarNetV4 contract requires 15x6 hybrid actions")
        if min(self.hidden_dim, self.primitive_dim, self.stage_dim) < 16:
            raise ValueError("DEPCarNetV4 feature dimensions are too small")
        if self.template_logit_bias <= 0.0:
            raise ValueError("DEPCarNetV4 template bias must be positive")


def _gear_templates(candidates, actions):
    patterns = (
        (1,), (-1,), (-1, 1), (1, -1, 1), (-1, 1, -1),
        (1, -1, 1, -1), (-1, 1, -1, 1),
        (1, -1, 1, -1, 1), (-1, 1, -1, 1, -1),
        (1, -1, 1, -1, 1, -1), (-1, 1, -1, 1, -1, 1),
        (1, -1), (-1, 1, -1, 1, -1, 1),
        (1, -1, 1, -1, 1), (-1, 1, -1, 1),
    )
    if len(patterns) != candidates:
        raise RuntimeError("V4 template count differs from candidate contract")
    tokens = torch.zeros((candidates, actions), dtype=torch.long)
    for candidate, pattern in enumerate(patterns):
        for action, gear in enumerate(pattern[:actions]):
            tokens[candidate, action] = 1 if gear > 0 else 2
    return tokens


class DEPCarNetV4(nn.Module):
    """Generate and rank complete forward/reverse manoeuvre sequences.

    The accepted V3 primitive policy is used only as a frozen perceptual and
    local-geometry feature extractor.  It never chooses the executed bank.
    V4 emits one joint gear/control sequence per candidate and scores the
    complete signed-Ackermann rollout.
    """

    architecture_id = "dep_car_multimodal_v4_hybrid_sequence_route_ackermann_15x6"
    objective_scope = "unified_hybrid_gear_control_sequences"
    candidate_count = 15
    macro_actions = 6
    predicts_gear = True
    predicts_explicit_gear_sequence = True
    gear_token_order = ("STOP", "FORWARD", "REVERSE")

    def __init__(
        self,
        base_model: Optional[DEPCarNetV3] = None,
        sequence_config=HybridSequenceConfigV4(),
        rollout_config=HybridSequenceRolloutConfigV4(),
    ):
        super().__init__()
        sequence_config.validate()
        if (
            rollout_config.candidates != sequence_config.candidates
            or rollout_config.actions != sequence_config.actions
        ):
            raise ValueError("V4 decoder and rollout dimensions differ")
        self.sequence_config = sequence_config
        self.base_model = base_model if base_model is not None else DEPCarNetV3()
        self.rollout = HybridSequenceAckermannRolloutV4(rollout_config)
        # score, recovery, primitive gear/shift/dwell, endpoint(6), route(6)
        primitive_input = 17
        primitive_dim = int(sequence_config.primitive_dim)
        hidden = int(sequence_config.hidden_dim)
        self.primitive_encoder = nn.Sequential(
            nn.Linear(primitive_input, primitive_dim), nn.SiLU(),
            nn.Linear(primitive_dim, primitive_dim), nn.SiLU(),
        )
        self.primitive_attention = nn.Linear(primitive_dim, 1)
        # weighted+maximum primitive pools, frozen route feature(64), state(9),
        # history(6), current gear/neutral(2), modality mask(2).
        context_input = 2 * primitive_dim + 64 + 9 + 6 + 2 + 2
        self.context_encoder = nn.Sequential(
            nn.LayerNorm(context_input), nn.Linear(context_input, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        self.candidate_queries = nn.Parameter(torch.empty(sequence_config.candidates, hidden))
        self.stage_embedding = nn.Embedding(sequence_config.actions, sequence_config.stage_dim)
        self.sequence_cell = nn.GRUCell(sequence_config.stage_dim + 3, hidden)
        self.gear_head = nn.Linear(hidden, 3)
        self.control_head = nn.Linear(hidden, 4)
        score_input = hidden + 6 + 3
        self.safety_head = nn.Sequential(nn.Linear(score_input, 64), nn.SiLU(), nn.Linear(64, 1))
        self.viability_head = nn.Sequential(nn.Linear(score_input, 64), nn.SiLU(), nn.Linear(64, 1))
        self.sequence_score_head = nn.Sequential(
            nn.Linear(score_input + 2, 96), nn.SiLU(), nn.Linear(96, 1)
        )
        template = _gear_templates(sequence_config.candidates, sequence_config.actions)
        bias = torch.full((sequence_config.candidates, sequence_config.actions, 3), -0.5)
        bias.scatter_(2, template[..., None], float(sequence_config.template_logit_bias))
        self.register_buffer("gear_template_tokens", template, persistent=True)
        self.register_buffer("gear_template_bias", bias, persistent=True)
        self._reset_v4_parameters()
        self.freeze_base()

    def _reset_v4_parameters(self):
        nn.init.normal_(self.candidate_queries, mean=0.0, std=0.08)
        for module in (
            self.primitive_encoder, self.primitive_attention, self.context_encoder,
            self.sequence_cell, self.gear_head, self.control_head,
            self.safety_head, self.viability_head, self.sequence_score_head,
        ):
            for child in module.modules():
                if isinstance(child, nn.Linear):
                    nn.init.kaiming_uniform_(child.weight, a=5 ** 0.5)
                    if child.bias is not None:
                        nn.init.zeros_(child.bias)

    def freeze_base(self):
        self.base_model.eval()
        for parameter in self.base_model.parameters():
            parameter.requires_grad_(False)

    def initialize_base(self, state_dict):
        current = self.base_model.state_dict()
        transferred = {
            key: value for key, value in state_dict.items()
            if key in current and current[key].shape == value.shape
        }
        if len(transferred) != len(current):
            missing = sorted(set(current) - set(transferred))
            raise RuntimeError("V4 base initialization is incomplete: %s" % missing[:5])
        self.base_model.load_state_dict(transferred, strict=True)
        self.freeze_base()
        return tuple(sorted(transferred))

    @staticmethod
    def _normalize_state(vehicle_state):
        return vehicle_state.float() / vehicle_state.new_tensor(STATE_NORMALIZATION_SCALE)

    def _primitive_context(
        self, base, vehicle_state, current_gear, gear_history,
        route_pose, route_mask, modality_mask,
    ):
        score = base.scores.detach().float()
        score = (score - score.mean(dim=1, keepdim=True)) / score.std(
            dim=1, keepdim=True, unbiased=False
        ).clamp_min(1.0e-4)
        endpoint = base.trajectories.detach().float()[..., -1, :]
        endpoint_feature = torch.stack(
            (
                endpoint[..., 1] / 3.0, endpoint[..., 2] / 3.0,
                torch.sin(endpoint[..., 3]), torch.cos(endpoint[..., 3]),
                endpoint[..., 4] / 2.5, endpoint[..., 5] / 0.6,
            ), dim=-1,
        )
        relation = corridor_candidate_relations(
            base.trajectories.detach().float(), route_pose.float(), route_mask
        )
        primitive = torch.cat(
            (
                score[..., None], base.forward_recovery_value.detach().float()[..., None],
                base.candidate_gears.detach().float()[..., None],
                base.shift_required.detach().float()[..., None],
                (base.transition_duration.detach().float() / 2.0)[..., None],
                endpoint_feature, relation,
            ), dim=-1,
        )
        encoded = self.primitive_encoder(primitive)
        attention = torch.softmax(self.primitive_attention(encoded).squeeze(-1), dim=1)
        weighted = (encoded * attention[..., None]).sum(dim=1)
        maximum = encoded.amax(dim=1)
        route_global = self.base_model.route_encoder(route_pose, route_mask).detach().float()
        history_scale = gear_history.new_tensor((1.0, 1.0, 2.0, 2.0, 4.0, 1.0))
        current = torch.stack((current_gear.float(), (current_gear == 0).float()), dim=1)
        value = torch.cat(
            (
                weighted, maximum, route_global, self._normalize_state(vehicle_state),
                gear_history.float() / history_scale, current, modality_mask.float(),
            ), dim=1,
        )
        return self.context_encoder(value)

    def forward(
        self, depth, lidar_bev, vehicle_state, current_gear, gear_history,
        route_pose, route_mask, modality_mask: Optional[torch.Tensor] = None,
    ):
        batch = len(vehicle_state)
        if current_gear.shape != (batch,) or gear_history.shape != (batch, 6):
            raise ValueError("V4 current gear/history shapes are invalid")
        if modality_mask is None:
            modality_mask = vehicle_state.new_ones((batch, 2))
        if modality_mask.shape != (batch, 2):
            raise ValueError("V4 modality_mask must have shape [B,2]")
        self.base_model.eval()
        with torch.no_grad():
            base = self.base_model(
                depth, lidar_bev, vehicle_state, current_gear, gear_history,
                route_pose, route_mask, modality_mask,
            )
        context = self._primitive_context(
            base, vehicle_state, current_gear, gear_history,
            route_pose, route_mask, modality_mask,
        )
        cfg = self.sequence_config
        hidden = context[:, None, :] + self.candidate_queries[None, :, :]
        hidden = hidden.reshape(batch * cfg.candidates, cfg.hidden_dim)
        initial_token = torch.where(
            current_gear > 0,
            current_gear.new_ones((batch,)),
            torch.where(current_gear < 0, current_gear.new_full((batch,), 2), current_gear.new_zeros((batch,))),
        )
        previous = F.one_hot(initial_token, num_classes=3).to(context)
        previous = previous[:, None].expand(-1, cfg.candidates, -1).reshape(-1, 3)
        gear_logits, raw_controls = [], []
        for action in range(cfg.actions):
            stage = self.stage_embedding.weight[action][None].expand(len(hidden), -1)
            hidden = self.sequence_cell(torch.cat((stage, previous), dim=1), hidden)
            logits = self.gear_head(hidden).reshape(batch, cfg.candidates, 3)
            logits = logits + self.gear_template_bias[:, action, :][None].to(logits)
            raw = self.control_head(hidden).reshape(batch, cfg.candidates, 4)
            gear_logits.append(logits)
            raw_controls.append(raw)
            previous = torch.softmax(logits, dim=-1).reshape(-1, 3)
        gear_logits = torch.stack(gear_logits, dim=2)
        raw_controls = torch.stack(raw_controls, dim=2)
        rollout = self.rollout(vehicle_state.float(), current_gear, raw_controls.float(), gear_logits.float())
        relation = corridor_candidate_relations(
            rollout.trajectory, route_pose.float(), route_mask
        )
        reverse_fraction = (
            (rollout.action_gears < 0) & rollout.action_mask
        ).float().sum(dim=-1) / rollout.action_mask.float().sum(dim=-1).clamp_min(1.0)
        action_fraction = rollout.action_mask.float().mean(dim=-1)
        shift_fraction = rollout.shift_required.float().mean(dim=-1)
        final_hidden = hidden.reshape(batch, cfg.candidates, cfg.hidden_dim)
        score_feature = torch.cat(
            (final_hidden, relation, torch.stack((reverse_fraction, action_fraction, shift_fraction), dim=-1)),
            dim=-1,
        )
        safety_logits = self.safety_head(score_feature).squeeze(-1)
        viability_logits = self.viability_head(score_feature).squeeze(-1)
        score = self.sequence_score_head(
            torch.cat((score_feature, safety_logits[..., None], viability_logits[..., None]), dim=-1)
        ).squeeze(-1)
        return DEPCarNetworkOutputV4(
            raw_controls=raw_controls.float(), gear_logits=gear_logits.float(),
            controls=rollout.controls, trajectories=rollout.trajectory,
            scores=F.softplus(score.float()), safety_logits=safety_logits.float(),
            viability_logits=viability_logits.float(), gear_tokens=rollout.gear_tokens,
            action_gears=rollout.action_gears, action_mask=rollout.action_mask,
            motion_gears=rollout.motion_gears, shift_required=rollout.shift_required,
            transition_duration=rollout.transition_duration,
            primitive_scores=base.scores.detach().float(),
        )

    def train(self, mode=True):
        super().train(mode)
        self.base_model.eval()
        return self

    def candidate_parameters(self):
        modules = (
            self.primitive_encoder, self.primitive_attention, self.context_encoder,
            self.stage_embedding, self.sequence_cell, self.gear_head, self.control_head,
            self.safety_head, self.viability_head,
        )
        yield self.candidate_queries
        for module in modules:
            yield from module.parameters()

    def score_parameters(self):
        yield from self.sequence_score_head.parameters()

    def all_v4_parameters(self):
        for parameter in self.parameters():
            if not any(parameter is base for base in self.base_model.parameters()):
                yield parameter


__all__ = ["DEPCarNetV4", "DEPCarNetworkOutputV4", "HybridSequenceConfigV4"]
