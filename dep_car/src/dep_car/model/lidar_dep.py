"""LiDAR-native DE-P-Car network with the frozen V4.8.3 architecture lineage.

The MobileNetV3 backbone and reusable independent-head towers can be imported
from the pinned DE-P source.  Car-specific input/state/output layers remain
new because their physical meaning changed.
"""

import os
import sys
from pathlib import Path

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover
    torch = None
    nn = object
    _IMPORT_ERROR = exc


def default_dep_source_tree():
    configured = os.environ.get("DEP_SOURCE_TREE")
    if configured:
        return Path(configured)
    return Path(__file__).resolve().parents[4] / "third_party" / "DE-P" / "DE-P"


def load_frozen_backbone(width, source_tree=None, input_size=(16, 440)):
    source_tree = Path(source_tree or default_dep_source_tree()).resolve()
    if not (source_tree / "policy" / "models" / "backbone.py").is_file():
        raise FileNotFoundError(f"pinned DE-P source tree is unavailable: {source_tree}")
    source_text = str(source_tree)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    from policy.models.backbone import DepBackbone
    return DepBackbone(width, input_size=tuple(input_size), backbone_variant="legacy")


if torch is not None:
    class LidarDEPCarV1(nn.Module):
        architecture_id = "dep_car_lidar_v1_3x5_mobilenetv3_v483"

        def __init__(self, state_dim=8, width=64, source_tree=None):
            super().__init__()
            self.state_dim = state_dim
            self.lidar_adapter = nn.Conv2d(2, 1, kernel_size=1, bias=False)
            with torch.no_grad():
                self.lidar_adapter.weight.zero_()
                self.lidar_adapter.weight[:, 0] = 1.0
            self.image_backbone = load_frozen_backbone(width, source_tree)
            self.spatial_projection = nn.AdaptiveAvgPool2d((3, 5))
            self.candidate_tower = nn.Sequential(
                nn.Conv2d(width + state_dim, 256, 1), nn.ReLU(),
                nn.Conv2d(256, 256, 1), nn.ReLU(),
            )
            self.score_tower = nn.Sequential(
                nn.Conv2d(width + state_dim, 256, 1), nn.ReLU(),
                nn.Conv2d(256, 256, 1), nn.ReLU(),
            )
            self.speed_head = nn.Conv2d(256, 1, 1)
            self.steering_head = nn.Conv2d(256, 1, 1)
            self.score_head = nn.Conv2d(256, 1, 1)

        def forward(self, lidar_range_and_mask, vehicle_state):
            if lidar_range_and_mask.ndim != 4 or lidar_range_and_mask.shape[1:] != (2, 16, 440):
                raise ValueError("LiDAR input must have shape [B,2,16,440]")
            if vehicle_state.ndim != 2 or vehicle_state.shape[1] != self.state_dim:
                raise ValueError("vehicle state shape does not match state_dim")
            features = self.spatial_projection(self.image_backbone(self.lidar_adapter(lidar_range_and_mask)))
            state = vehicle_state[:, :, None, None].expand(-1, -1, 3, 5)
            fused = torch.cat((features, state), dim=1)
            candidate_features = self.candidate_tower(fused)
            score_features = self.score_tower(fused)
            speed_offset = 0.35 * torch.tanh(self.speed_head(candidate_features)).flatten(1)
            steering_offset = 0.18 * torch.tanh(self.steering_head(candidate_features)).flatten(1)
            score = torch.nn.functional.softplus(self.score_head(score_features)).flatten(1)
            return speed_offset, steering_offset, score
else:
    class LidarDEPCarV1:  # pragma: no cover
        architecture_id = "dep_car_lidar_v1_3x5_mobilenetv3_v483"
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for LidarDEPCarV1") from _IMPORT_ERROR
