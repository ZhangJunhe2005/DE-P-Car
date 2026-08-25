"""V4 hybrid-sequence view over the sealed P3/V3 development data."""

import numpy as np
import torch

from .p4_dataset import P3TrainingDataError
from .score_dataset import P3JointGearDatasetV3


class P3HybridSequenceDatasetV4(P3JointGearDatasetV3):
    """Add the immutable first-action geometry teacher required by V4.

    The six-action sidecar remains weak gear supervision.  Only the already
    sealed P3 candidate trajectories are exposed as first-action geometry;
    no later continuous action is invented.
    """

    schema = "P3HybridSequenceDatasetV4"

    def __getitem__(self, index):
        item = super().__getitem__(index)
        path = item["metadata"]["path"]
        try:
            with np.load(path, allow_pickle=False) as data:
                trajectories = np.asarray(data["trajectories"], dtype=np.float32)
                feasible = np.asarray(data["feasible"], dtype=np.bool_)
                guidance = np.asarray(data["guidance_cost"], dtype=np.float32)
        except Exception as exc:
            raise P3TrainingDataError(
                "unable to read V4 first-action teacher %s: %s" % (path, exc)
            ) from exc
        if (
            trajectories.ndim != 3
            or trajectories.shape[0] != 15
            or trajectories.shape[1] < 2
            or trajectories.shape[2] != 6
            or feasible.shape != (15,)
            or guidance.shape != (15,)
            or not np.all(np.isfinite(trajectories))
            or not np.all(np.isfinite(guidance))
        ):
            raise P3TrainingDataError("V4 first-action teacher is invalid: %s" % path)
        # P3 episodes legitimately contain both 1.0 s (11 rows) and 1.4 s
        # (15 rows) candidate banks.  V4's first macro action is supervised at
        # a common one-second horizon so batching never changes the semantic
        # target or requires padding variable-length trajectories.
        time_axis = trajectories[0, :, 0]
        teacher_horizon_s = min(1.0, float(time_axis[-1]))
        teacher_index = int(np.abs(time_axis - teacher_horizon_s).argmin())
        item["target_first_action_pose"] = torch.from_numpy(
            trajectories[:, teacher_index, 1:4].copy()
        )
        item["target_first_action_valid"] = torch.from_numpy(feasible)
        item["target_guidance_cost"] = torch.from_numpy(guidance)
        item["metadata"].update({
            "schema": self.schema,
            "continuous_sequence_authority": "FIRST_ACTION_ONLY",
            "later_action_supervision": "DIFFERENTIABLE_ROLLOUT_ROUTE_MAP",
            "first_action_teacher_horizon_s": teacher_horizon_s,
            "first_action_teacher_available": bool(feasible.any()),
        })
        return item


__all__ = ["P3HybridSequenceDatasetV4"]
