"""Explicit learned gear selection on top of the frozen DE-P-Car V3 policy.

V3 ranked forward and reverse trajectories with one shared energy.  V3.3
keeps those accepted trajectory banks immutable and learns a separate binary
decision: which bank should be used now.  Hard feasibility remains an
independent execution-time safety boundary and is not encoded in this module.
"""

from dataclasses import dataclass
from typing import NamedTuple, Optional

import torch
from torch import nn

from dep_car.core.state_contract import STATE_NORMALIZATION_SCALE

from .dep_car_net_v2 import corridor_candidate_relations
from .dep_car_net_v3 import DEPCarNetV3


class DEPCarNetworkOutputV33(NamedTuple):
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
    gear_logits: torch.Tensor


@dataclass(frozen=True)
class GearSelectorConfigV33:
    score_top_k: int = 3
    hidden_dim: int = 64
    history_dim: int = 6

    def validate(self):
        if not 1 <= int(self.score_top_k) <= 15:
            raise ValueError("V3.3 score_top_k must be in [1,15]")
        if int(self.hidden_dim) < 16:
            raise ValueError("V3.3 hidden_dim must be at least 16")
        if int(self.history_dim) != 6:
            raise ValueError("V3.3 gear-history contract requires six fields")


class DEPCarGearSelectorV33(nn.Module):
    """Frozen V3 trajectory policy plus explicit FORWARD/REVERSE logits.

    Logit index 0 means FORWARD and index 1 means REVERSE.  The selector sees
    the relative quality of both banks, route relations, vehicle state and
    recent gear history.  All base-policy features are detached so training
    this head cannot silently alter candidate generation or ranking.
    """

    architecture_id = "dep_car_multimodal_v33_explicit_gear_selector_route_ackermann"
    objective_scope = "explicit_gear_over_frozen_v31_trajectory_policy"
    predicts_gear = True
    predicts_explicit_gear = True
    gear_logit_order = ("FORWARD", "REVERSE")

    def __init__(
        self,
        base_model: Optional[DEPCarNetV3] = None,
        selector_config: GearSelectorConfigV33 = GearSelectorConfigV33(),
    ):
        super().__init__()
        selector_config.validate()
        self.base_model = base_model if base_model is not None else DEPCarNetV3()
        self.selector_config = selector_config
        # Per-bank features:
        # score top-k + score min/mean/std + recovery top-k/mean + one selected
        # route relation (6).  Global features are state(9), history(6),
        # current-gear sign/neutral encoding(2), and modality mask(2).
        per_bank = int(selector_config.score_top_k) * 2 + 3 + 1 + 6
        input_dim = 2 * per_bank + 9 + 6 + 2 + 2
        self.feature_dim = input_dim
        hidden = int(selector_config.hidden_dim)
        self.gear_selector = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden // 2),
            nn.SiLU(),
            nn.Linear(hidden // 2, 2),
        )
        self._reset_selector()

    def _reset_selector(self):
        for module in self.gear_selector.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, a=5 ** 0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def freeze_base(self):
        self.base_model.eval()
        for parameter in self.base_model.parameters():
            parameter.requires_grad_(False)

    def selector_parameters(self):
        yield from self.gear_selector.parameters()

    @staticmethod
    def _mirror_invariant_state(vehicle_state):
        scale = vehicle_state.new_tensor(STATE_NORMALIZATION_SCALE)
        state = vehicle_state.float() / scale
        # Steering, yaw rate, lateral target, heading sine and reference
        # curvature change sign under a left/right reflection.  Gear choice
        # should not, so expose their magnitude to this binary head.
        state = state.clone()
        state[:, (2, 3, 5, 6, 8)] = state[:, (2, 3, 5, 6, 8)].abs()
        return state

    def selector_features(
        self,
        output,
        vehicle_state,
        current_gear,
        gear_history,
        modality_mask,
        route_pose,
        route_mask,
    ):
        if output.scores.ndim != 2 or output.scores.shape[1] != 30:
            raise ValueError("V3.3 expects a 30-candidate V3 bank")
        batch = len(output.scores)
        if current_gear.shape != (batch,):
            raise ValueError("current_gear must have shape [B]")
        if gear_history.shape != (batch, self.selector_config.history_dim):
            raise ValueError("gear_history must have shape [B,6]")
        if modality_mask.shape != (batch, 2):
            raise ValueError("modality_mask must have shape [B,2]")

        score = output.scores.detach().float()
        center = score.mean(dim=1, keepdim=True)
        spread = score.std(dim=1, keepdim=True, unbiased=False).clamp_min(1.0e-4)
        score = (score - center) / spread
        recovery = output.forward_recovery_value.detach().float()
        relation = corridor_candidate_relations(
            output.trajectories.detach().float(),
            route_pose.detach().float(),
            route_mask,
        )
        relation = relation.clone()
        relation[..., 4] = relation[..., 4].abs()

        features = []
        top_k = int(self.selector_config.score_top_k)
        for start in (0, 15):
            bank_score = score[:, start : start + 15]
            bank_recovery = recovery[:, start : start + 15]
            best_index = bank_score.argmin(dim=1)
            best_relation = relation[:, start : start + 15].gather(
                1, best_index[:, None, None].expand(-1, 1, 6)
            ).squeeze(1)
            features.extend(
                (
                    torch.topk(bank_score, top_k, dim=1, largest=False).values,
                    torch.stack(
                        (
                            bank_score.amin(dim=1),
                            bank_score.mean(dim=1),
                            bank_score.std(dim=1, unbiased=False),
                        ),
                        dim=1,
                    ),
                    torch.topk(bank_recovery, top_k, dim=1, largest=True).values,
                    bank_recovery.mean(dim=1, keepdim=True),
                    best_relation,
                )
            )

        history_scale = gear_history.new_tensor((1.0, 1.0, 2.0, 2.0, 4.0, 1.0))
        # A two-value sign/neutral representation distinguishes all three
        # actuator states {-1, 0, +1}; treating neutral as forward would make
        # the selector ambiguous during the mandatory stop-before-shift dwell.
        current = torch.stack(
            (current_gear.float(), (current_gear == 0).float()), dim=1
        )
        features.extend(
            (
                self._mirror_invariant_state(vehicle_state),
                gear_history.float() / history_scale,
                current,
                modality_mask.float(),
            )
        )
        value = torch.cat(features, dim=1)
        if value.shape != (batch, self.feature_dim):
            raise RuntimeError(
                "V3.3 selector feature contract changed: got %s expected (%d,%d)"
                % (tuple(value.shape), batch, self.feature_dim)
            )
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError("V3.3 selector features are non-finite")
        return value.detach()

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
        if modality_mask is None:
            modality_mask = vehicle_state.new_ones((len(vehicle_state), 2))
        # The base is deliberately evaluated without a graph.  Calling
        # ``train()`` on this wrapper cannot unfreeze or update it.
        self.base_model.eval()
        with torch.no_grad():
            base = self.base_model(
                depth,
                lidar_bev,
                vehicle_state,
                current_gear,
                gear_history,
                route_pose,
                route_mask,
                modality_mask,
            )
        features = self.selector_features(
            base,
            vehicle_state,
            current_gear,
            gear_history,
            modality_mask,
            route_pose,
            route_mask,
        )
        logits = self.gear_selector(features.float())
        return DEPCarNetworkOutputV33(*base, logits.float())

    def train(self, mode=True):
        super().train(mode)
        self.base_model.eval()
        return self


__all__ = [
    "DEPCarGearSelectorV33",
    "DEPCarNetworkOutputV33",
    "GearSelectorConfigV33",
]
