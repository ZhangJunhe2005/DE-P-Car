import math

import numpy as np

from dep_car.core.occupancy import OccupancyGrid2D
from dep_car.core.types import Gear
from dep_car.global_planner.hybrid_astar import HybridPathPose
from dep_car.training.pilot import PILOT_MANEUVER_MODES, candidate_expressiveness, classify_maneuver, make_pilot_manifest


def pose(x, y, yaw, gear, steering=0.0):
    return HybridPathPose(x, y, yaw, gear, steering)


def test_pilot_modes_cover_reviewed_static_buckets():
    assert set(PILOT_MANEUVER_MODES) == {
        "NORMAL", "SHARP_TURN", "NARROW_CORRIDOR", "U_TURN",
        "DEAD_END_ESCAPE", "REVERSE_EXIT", "THREE_POINT_TURN",
    }


def test_classifier_gives_three_point_gear_sequence_priority():
    grid = OccupancyGrid2D(np.zeros((100, 100), dtype=np.int8), 0.1, (-5.0, -5.0))
    path = [
        pose(0, 0, 0, Gear.NEUTRAL), pose(0.3, 0, 0.2, Gear.FORWARD, 0.5),
        pose(0.1, -0.1, 0.8, Gear.REVERSE, -0.5), pose(0.4, 0, math.pi, Gear.FORWARD, 0.5),
    ]
    mode, evidence = classify_maneuver(grid, (0, 0, 0), (0.4, 0, math.pi), path)
    assert mode == "THREE_POINT_TURN"
    assert evidence["gear_runs"] == [1, -1, 1]


def test_manifest_hash_is_deterministic_and_content_bound():
    first = make_pilot_manifest([], seed=1, map_selection={}, quotas={}, generator_contract={})
    second = make_pilot_manifest([], seed=1, map_selection={}, quotas={}, generator_contract={})
    changed = make_pilot_manifest([], seed=2, map_selection={}, quotas={}, generator_contract={})
    assert first["task_manifest_sha256"] == second["task_manifest_sha256"]
    assert first["task_manifest_sha256"] != changed["task_manifest_sha256"]


def test_candidate_expressiveness_uses_best_feasible_route_match():
    trajectories = np.zeros((3, 2, 6), dtype=np.float32)
    trajectories[0, -1, 1:4] = (1.0, 0.8, 0.7)
    trajectories[1, -1, 1:4] = (1.0, 0.0, 0.0)
    trajectories[2, -1, 1:4] = (2.0, 0.0, 0.0)
    metric = candidate_expressiveness(
        trajectories, [1, 1, 0], np.asarray([[0, 0, 0], [1, 0, 0], [2, 0, 0]]),
        [2.0, 1.0, 0.0], [0.2, 0.4, 0.8],
    )
    assert metric["feasible_count"] == 2
    assert metric["oracle_route_error_m"] == 0.0
    assert metric["oracle_route_progress_m"] == 1.0
    assert metric["oracle_static_clearance_m"] == 0.4
