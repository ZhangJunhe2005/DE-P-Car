from dep_car.dynamic.tracker import ConstantVelocityTracker


def test_tracker_confirms_and_estimates_motion():
    tracker = ConstantVelocityTracker()
    assert tracker.update([(0.0, 0.0)], 0.0) == []
    assert tracker.update([(0.2, 0.0)], 0.2) == []
    tracks = tracker.update([(0.4, 0.0)], 0.4)
    assert len(tracks) == 1
    assert tracks[0].confidence == 1.0
    assert tracks[0].vx > 0.0

