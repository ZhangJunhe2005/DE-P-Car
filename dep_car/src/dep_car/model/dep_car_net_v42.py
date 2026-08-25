"""V4.2 calibrated execution view with mandatory external hard veto."""

from torch.nn import functional as F

from .dep_car_net_v4 import DEPCarNetV4


class DEPCarNetV42(DEPCarNetV4):
    """Apply the sealed V4.1 viability calibration to complete sequences.

    Static/dynamic hard veto remains an external, mandatory execution layer.
    This class changes only the differentiable ranking score and has exactly
    the same trainable state dictionary as V4/V4.1.
    """

    architecture_id = (
        "dep_car_multimodal_v42_calibrated_hybrid_sequence_execution_15x6"
    )
    source_architecture_id = DEPCarNetV4.architecture_id
    viability_risk_weight = 8.0
    safety_risk_weight = 0.0
    requires_mandatory_hard_veto = True

    def forward(self, *args, **kwargs):
        output = super().forward(*args, **kwargs)
        calibrated = (
            output.scores
            + float(self.viability_risk_weight)
            * F.softplus(-output.viability_logits.float())
        )
        return output._replace(scores=calibrated)


__all__ = ["DEPCarNetV42"]
