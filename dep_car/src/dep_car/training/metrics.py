"""Streaming P5 metrics with hard-veto-before-ranking semantics."""

from collections import defaultdict

import torch


def candidate_batch_metrics(output, objective_result):
    clearance = objective_result["minimum_clearance"]
    cost = objective_result["candidate_cost"]
    if clearance.shape != output.scores.shape or cost.shape != output.scores.shape:
        raise ValueError("candidate metric tensors must all have shape [B,15]")
    kinematic_violation = objective_result.get("kinematic_violation")
    if kinematic_violation is None:
        kinematic = objective_result.get("kinematic_per_candidate")
        if kinematic is None:
            kinematic = torch.zeros_like(clearance)
        if kinematic.shape != clearance.shape:
            raise ValueError("kinematic metric tensor must have shape [B,15]")
        kinematic_violation = kinematic > 1.0e-8
    if kinematic_violation.shape != clearance.shape:
        raise ValueError("kinematic violation tensor must have shape [B,15]")
    kinematic_violation = kinematic_violation.bool()
    feasible = objective_result.get("hard_feasible")
    if feasible is None:
        feasible = (clearance > 0.0) & ~kinematic_violation
    if feasible.shape != clearance.shape:
        raise ValueError("hard feasible tensor must have shape [B,15]")
    feasible = feasible.bool()
    feasible_count = feasible.sum(dim=1)
    zero_feasible = feasible_count == 0
    oracle_cost = cost.masked_fill(~feasible, float("inf")).amin(dim=1)
    oracle_fallback = cost.amin(dim=1)
    oracle_cost = torch.where(zero_feasible, oracle_fallback, oracle_cost)
    shielded_score = output.scores.masked_fill(~feasible, float("inf"))
    selected_index = shielded_score.argmin(dim=1)
    selected_index = torch.where(zero_feasible, selected_index.new_full((), -1), selected_index)
    safe_index = selected_index.clamp_min(0)
    selected_cost = torch.gather(cost, 1, safe_index[:, None]).squeeze(1)
    selected_cost = torch.where(zero_feasible, selected_cost.new_full((), float("nan")), selected_cost)
    regret = selected_cost - oracle_cost
    best_clearance = clearance.amax(dim=1)
    return {
        "feasible_count": feasible_count,
        "zero_feasible": zero_feasible,
        "best_clearance": best_clearance,
        "oracle_cost": oracle_cost,
        "selected_index": selected_index,
        "selected_cost": selected_cost,
        "oracle_regret": regret,
        "kinematic_violation_count": kinematic_violation.sum(dim=1),
    }


class CandidateMetricAccumulator:
    """Accumulate aggregate and maneuver-specific metrics without frame leakage."""

    def __init__(self):
        self.rows = defaultdict(list)
        self.mode_rows = defaultdict(lambda: defaultdict(list))

    def update(self, metrics, maneuver_modes=None):
        batch = len(metrics["feasible_count"])
        if maneuver_modes is not None and len(maneuver_modes) != batch:
            raise ValueError("maneuver mode count must match metric batch")
        for name, values in metrics.items():
            detached = values.detach().cpu()
            self.rows[name].append(detached)
            if maneuver_modes is not None:
                for index, mode in enumerate(maneuver_modes):
                    self.mode_rows[str(mode)][name].append(detached[index:index + 1])

    @staticmethod
    def _summarize(rows):
        values = {name: torch.cat(chunks) for name, chunks in rows.items()}
        valid_regret = values["oracle_regret"][torch.isfinite(values["oracle_regret"])]
        return {
            "frames": int(len(values["zero_feasible"])),
            "zero_feasible_rate": float(values["zero_feasible"].float().mean()),
            "mean_feasible_candidates": float(values["feasible_count"].float().mean()),
            "median_feasible_candidates": float(values["feasible_count"].float().median()),
            "mean_best_clearance_m": float(values["best_clearance"].mean()),
            "selected_feasible_rate": float((values["selected_index"] >= 0).float().mean()),
            "kinematic_violation_rate": float(
                values["kinematic_violation_count"].float().sum()
                / float(15 * len(values["kinematic_violation_count"]))
            ),
            "mean_oracle_regret": float(valid_regret.mean()) if len(valid_regret) else None,
        }

    def compute(self):
        if not self.rows:
            raise RuntimeError("no candidate metrics were accumulated")
        return {
            "overall": self._summarize(self.rows),
            "by_maneuver": {
                mode: self._summarize(rows)
                for mode, rows in sorted(self.mode_rows.items())
            },
        }
