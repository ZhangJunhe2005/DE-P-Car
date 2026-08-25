"""DEPCarNetV4.3 guarded closed-loop residual sequence selector."""

import math

import torch
from torch import nn

from .dep_car_net_v2 import corridor_candidate_relations
from .dep_car_net_v4 import DEPCarNetV4


class DEPCarNetV43(DEPCarNetV4):
    """Refine the accepted V4.1 score without changing hybrid candidates.

    The accepted source score is preserved as a frozen baseline. A compact,
    zero-initialised residual sees candidate-relative route geometry, the
    complete six-action gear probabilities and motion statistics, plus the
    frozen route context, vehicle state and gear history that explain *when*
    an otherwise similar manoeuvre is a recovery manoeuvre. Its output is
    bounded, so DAgger supervision cannot catastrophically overwrite the
    already accepted perception, candidate or score capacity.
    """

    architecture_id = (
        "dep_car_multimodal_v43_guarded_contextual_residual_closed_loop_hybrid_"
        "sequence_ackermann_15x6"
    )
    requires_mandatory_hard_veto = True
    predicts_explicit_gear_sequence = True
    high_level_gear_state_machine = False

    def __init__(self, *args, residual_score_span=3.0, **kwargs):
        if (
            not math.isfinite(float(residual_score_span))
            or float(residual_score_span) <= 0.0
        ):
            raise ValueError("V4.3 residual score span must be positive")
        super().__init__(*args, **kwargs)
        self.residual_score_span = float(residual_score_span)
        # Candidate-relative: source score/safety/viability + route relation(6)
        # + gear p(18) + reverse/action/shift fractions(3) = 30.
        # State context: route global(64) + vehicle state(9) + gear history(6)
        # + current gear/neutral(2) + modality mask(2) = 83.
        self.closed_loop_score_adapter = nn.Sequential(
            nn.Linear(113, 64), nn.SiLU(), nn.Linear(64, 1)
        )
        nn.init.kaiming_uniform_(
            self.closed_loop_score_adapter[0].weight, a=5 ** 0.5
        )
        nn.init.zeros_(self.closed_loop_score_adapter[0].bias)
        nn.init.zeros_(self.closed_loop_score_adapter[2].weight)
        nn.init.zeros_(self.closed_loop_score_adapter[2].bias)

    def initialize_from_v4(self, state_dict):
        incompatible = self.load_state_dict(state_dict, strict=False)
        expected = {
            name for name in self.state_dict()
            if name.startswith("closed_loop_score_adapter.")
        }
        if set(incompatible.missing_keys) != expected or incompatible.unexpected_keys:
            raise RuntimeError("V4.3 source initialization is incomplete")
        return tuple(sorted(expected))

    @staticmethod
    def _candidate_standardize(value):
        mean = value.mean(dim=1, keepdim=True)
        scale = value.std(dim=1, keepdim=True, unbiased=False).clamp_min(0.10)
        return (value - mean) / scale

    def forward(
        self, depth, lidar_bev, vehicle_state, current_gear, gear_history,
        route_pose, route_mask, modality_mask=None,
    ):
        batch = len(vehicle_state)
        if modality_mask is None:
            modality_mask = vehicle_state.new_ones((batch, 2))
        output = super().forward(
            depth, lidar_bev, vehicle_state, current_gear, gear_history,
            route_pose, route_mask, modality_mask,
        )
        relation = corridor_candidate_relations(
            output.trajectories.float(), route_pose.float(), route_mask
        )
        gear_probability = torch.softmax(output.gear_logits.float(), dim=-1)
        reverse_fraction = (
            (output.action_gears < 0) & output.action_mask
        ).float().sum(dim=-1) / output.action_mask.float().sum(dim=-1).clamp_min(1.0)
        action_fraction = output.action_mask.float().mean(dim=-1)
        shift_fraction = output.shift_required.float().mean(dim=-1)
        candidate_feature = torch.cat(
            (
                torch.stack(
                    (output.scores, output.safety_logits, output.viability_logits),
                    dim=-1,
                ),
                relation,
                gear_probability.flatten(start_dim=-2),
                torch.stack(
                    (reverse_fraction, action_fraction, shift_fraction), dim=-1
                ),
            ),
            dim=-1,
        )
        candidate_feature = self._candidate_standardize(candidate_feature).detach()
        # Do not standardize broadcast state features across candidates: doing
        # so would turn every one of them into zero and erase the very context
        # needed to distinguish a normal forward state from an R->F recovery.
        route_global = self.base_model.route_encoder(
            route_pose, route_mask
        ).detach().float()
        history_scale = gear_history.new_tensor((1.0, 1.0, 2.0, 2.0, 4.0, 1.0))
        current = torch.stack(
            (current_gear.float(), (current_gear == 0).float()), dim=1
        )
        global_context = torch.cat(
            (
                route_global,
                self._normalize_state(vehicle_state),
                gear_history.float() / history_scale,
                current,
                modality_mask.float(),
            ),
            dim=1,
        ).detach()
        feature = torch.cat(
            (
                candidate_feature,
                global_context[:, None].expand(-1, candidate_feature.shape[1], -1),
            ),
            dim=-1,
        )
        residual = self.residual_score_span * torch.tanh(
            self.closed_loop_score_adapter(feature).squeeze(-1)
        )
        return output._replace(scores=(output.scores + residual).float())


__all__ = ["DEPCarNetV43"]
