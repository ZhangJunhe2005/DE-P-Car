"""Small 2-D constant-velocity Kalman tracker adapted from DE-P semantics."""

from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np

from dep_car.core.types import DynamicTrack


@dataclass(frozen=True)
class TrackerConfig:
    association_distance: float = 1.2
    process_noise: float = 0.8
    measurement_noise: float = 0.15
    confirm_hits: int = 3
    maximum_misses: int = 5


@dataclass
class _TrackState:
    track_id: int
    state: np.ndarray
    covariance: np.ndarray
    hits: int = 1
    misses: int = 0
    stamp: float = 0.0


class ConstantVelocityTracker:
    """Causal one-to-one nearest-neighbour association with Kalman updates."""

    def __init__(self, config: TrackerConfig = TrackerConfig()):
        self.config = config
        self._tracks: Dict[int, _TrackState] = {}
        self._next_id = 1

    def _predict(self, track: _TrackState, stamp: float) -> None:
        dt = max(0.0, min(1.0, stamp - track.stamp))
        transition = np.asarray(
            [[1.0, 0.0, dt, 0.0], [0.0, 1.0, 0.0, dt], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
        )
        process = np.eye(4) * self.config.process_noise * max(dt, 1e-3)
        track.state = transition @ track.state
        track.covariance = transition @ track.covariance @ transition.T + process
        track.stamp = stamp

    def update(self, observations_xy: Iterable[Iterable[float]], stamp: float) -> List[DynamicTrack]:
        observations = np.asarray(list(observations_xy), dtype=float)
        if observations.size == 0:
            observations = np.empty((0, 2), dtype=float)
        observations = observations.reshape((-1, 2))
        for track in self._tracks.values():
            self._predict(track, stamp)

        unmatched_tracks = set(self._tracks)
        unmatched_observations = set(range(len(observations)))
        proposals = []
        for track_id, track in self._tracks.items():
            for obs_id, observation in enumerate(observations):
                proposals.append((float(np.linalg.norm(track.state[:2] - observation)), track_id, obs_id))
        for distance, track_id, obs_id in sorted(proposals):
            if distance > self.config.association_distance:
                break
            if track_id not in unmatched_tracks or obs_id not in unmatched_observations:
                continue
            track = self._tracks[track_id]
            measurement = np.zeros((2, 4))
            measurement[0, 0] = measurement[1, 1] = 1.0
            noise = np.eye(2) * self.config.measurement_noise
            innovation_covariance = measurement @ track.covariance @ measurement.T + noise
            gain = track.covariance @ measurement.T @ np.linalg.inv(innovation_covariance)
            track.state += gain @ (observations[obs_id] - measurement @ track.state)
            track.covariance = (np.eye(4) - gain @ measurement) @ track.covariance
            track.hits += 1
            track.misses = 0
            unmatched_tracks.remove(track_id)
            unmatched_observations.remove(obs_id)

        for track_id in unmatched_tracks:
            self._tracks[track_id].misses += 1
        for obs_id in unmatched_observations:
            self._tracks[self._next_id] = _TrackState(
                self._next_id,
                np.asarray([observations[obs_id, 0], observations[obs_id, 1], 0.0, 0.0]),
                np.diag([0.25, 0.25, 1.0, 1.0]),
                stamp=stamp,
            )
            self._next_id += 1
        self._tracks = {
            track_id: track for track_id, track in self._tracks.items() if track.misses <= self.config.maximum_misses
        }
        return [
            DynamicTrack(
                track_id=track.track_id,
                x=float(track.state[0]),
                y=float(track.state[1]),
                vx=float(track.state[2]),
                vy=float(track.state[3]),
                covariance=track.covariance.copy(),
                confidence=min(1.0, track.hits / self.config.confirm_hits),
                stamp=stamp,
            )
            for track in self._tracks.values()
            if track.hits >= self.config.confirm_hits
        ]

