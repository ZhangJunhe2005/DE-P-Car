"""V4.3 DAgger states with exact offline signed-plan supervision."""

import json
from pathlib import Path

import numpy as np
import torch

from .p4_dataset import P3TrainingDataError, _sha256_file
from .score_dataset import P3ScoreTrainingDatasetV1, _canonical_sha256


class P3ClosedLoopSequenceDatasetV43(P3ScoreTrainingDatasetV1):
    schema = "P3ClosedLoopSequenceDatasetV43"
    sequence_schema = "DEPCarV43ClosedLoopSequenceIndexV2"
    sequence_authority = "REOBSERVED_STATE_EXACT_SIGNED_HYBRID_ASTAR_PLAN"
    sequence_actions = 6

    def __init__(self, *args, sequence_index_path, expected_sequence_index_sha256=None, **kwargs):
        self.joint_gear_view = True
        super().__init__(*args, **kwargs)
        self.sequence_index_path = Path(sequence_index_path).resolve()
        if (
            expected_sequence_index_sha256 is not None
            and _sha256_file(self.sequence_index_path) != expected_sequence_index_sha256
        ):
            raise P3TrainingDataError("V4.3 sequence index hash differs")
        try:
            payload = json.loads(self.sequence_index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise P3TrainingDataError("unable to read V4.3 sequence index") from exc
        claimed = payload.get("content_sha256")
        content = dict(payload); content.pop("content_sha256", None)
        if (
            payload.get("schema") != self.sequence_schema
            or payload.get("sequence_actions") != self.sequence_actions
            or payload.get("sequence_authority") != self.sequence_authority
            or payload.get("test_split_opened") is not False
            or claimed != _canonical_sha256(content)
            or Path(payload.get("source_index", "")).resolve() != self.index_path
            or payload.get("source_index_sha256") != _sha256_file(self.index_path)
        ):
            raise P3TrainingDataError("V4.3 sequence authority is invalid")
        rows = {
            row["sample_id"]: row for row in payload.get("rows", ())
            if row.get("split") == self.split
        }
        expected = {entry["sample_id"] for entry in self.entries}
        if set(rows) != expected:
            raise P3TrainingDataError("V4.3 sequence index does not cover split exactly")
        self.sequence_rows = rows
        self.sequence_index_sha256 = _sha256_file(self.sequence_index_path)

    def __getitem__(self, index):
        item = super().__getitem__(index)
        row = self.sequence_rows[item["metadata"]["sample_id"]]
        history = np.asarray(row.get("history", ()), dtype=np.float32)
        gears = np.asarray(row.get("sequence_gears", ()), dtype=np.int64)
        mask = np.asarray(row.get("sequence_mask", ()), dtype=np.bool_)
        endpoints = np.asarray(
            row.get("action_plan_endpoints_body", ()), dtype=np.float32
        )
        if (
            history.shape != (6,) or not np.all(np.isfinite(history))
            or gears.shape != (6,) or mask.shape != (6,)
            or endpoints.shape != (6, 3)
            or not np.all(np.isfinite(endpoints))
            or not np.all(np.isin(gears, (-1, 0, 1)))
            or np.any((gears == 0) & mask) or np.any((gears != 0) & ~mask)
            or row.get("sequence_authority") != self.sequence_authority
        ):
            raise P3TrainingDataError("invalid V4.3 exact signed-plan row")
        path = item["metadata"]["path"]
        try:
            with np.load(path, allow_pickle=False) as data:
                first_gear = int(gears[0]) if bool(mask[0]) else 0
                prefix = "forward" if first_gear > 0 else "reverse"
                if first_gear:
                    trajectories = np.asarray(
                        data["dagger_%s_trajectories" % prefix], dtype=np.float32
                    )
                    feasible = np.asarray(
                        data["dagger_%s_feasible" % prefix], dtype=np.bool_
                    )
                    guidance = np.asarray(
                        data["dagger_%s_guidance_cost" % prefix], dtype=np.float32
                    )
                else:
                    trajectories = np.asarray(data["trajectories"], dtype=np.float32)
                    feasible = np.zeros(15, dtype=np.bool_)
                    guidance = np.asarray(data["guidance_cost"], dtype=np.float32)
        except Exception as exc:
            raise P3TrainingDataError("unable to read V4.3 teacher bank") from exc
        if (
            trajectories.ndim != 3 or trajectories.shape[0] != 15
            or trajectories.shape[2] != 6 or trajectories.shape[1] < 2
            or feasible.shape != (15,) or guidance.shape != (15,)
            or not np.all(np.isfinite(trajectories)) or not np.all(np.isfinite(guidance))
        ):
            raise P3TrainingDataError("V4.3 teacher candidate bank is invalid")
        time_axis = trajectories[0, :, 0]
        teacher_index = int(np.abs(time_axis - min(1.0, float(time_axis[-1]))).argmin())
        item.update({
            "gear_history": torch.from_numpy(history),
            "sequence_gears": torch.from_numpy(gears),
            "sequence_mask": torch.from_numpy(mask),
            "target_first_action_pose": torch.from_numpy(
                trajectories[:, teacher_index, 1:4].copy()
            ),
            "target_first_action_valid": torch.from_numpy(feasible),
            "target_guidance_cost": torch.from_numpy(guidance),
            "target_action_plan_pose": torch.from_numpy(endpoints),
            "target_action_plan_mask": torch.from_numpy(mask.copy()),
        })
        if bool(mask[0]):
            # The base score view carries the extraction-time one-step label.
            # V4.3 must instead bind geometry and sequence loss to the exact
            # signed-plan first action chosen at this re-observed state.
            item["requested_gear"] = torch.tensor(
                int(gears[0]), dtype=torch.int64
            )
        item["metadata"].update({
            "schema": self.schema,
            "continuous_sequence_authority": self.sequence_authority,
            "sequence_index_sha256": self.sequence_index_sha256,
            "teacher_plan_status": row.get("teacher_plan_status", "UNKNOWN"),
            "teacher_target_source": row.get("teacher_target_source", "UNKNOWN"),
            "model_raw_sequence": row.get("model_raw_sequence", ()),
        })
        return item


__all__ = ["P3ClosedLoopSequenceDatasetV43"]
