import pytest

from dep_car.core.recovery import RecoveryManager, RecoveryState


def test_dynamic_yield_does_not_accumulate_static_deadlock():
    manager = RecoveryManager()
    for now in (0.0, 2.0, 10.0):
        assert manager.update_blockage(now, 0.0, False, True) == RecoveryState.DYNAMIC_YIELD
    assert manager.update_blockage(11.0, 0.0, True, False) == RecoveryState.DYNAMIC_YIELD
    assert manager.update_blockage(14.0, 0.0, True, False) == RecoveryState.STATIC_DEADLOCK


def test_recovery_requires_mission_reacquisition_before_rearm():
    manager = RecoveryManager()
    manager.set_mission_goal((10.0, 0.0))
    manager.update_blockage(0.0, 0.0, True, False)
    manager.update_blockage(3.0, 0.0, True, False)
    manager.begin_recovery(3.0, (-1.8, 0.0))
    assert manager.active_goal == (-1.8, 0.0)
    manager.finish_reverse(); manager.finish_escape()
    assert manager.active_goal == (10.0, 0.0)
    with pytest.raises(RuntimeError): manager.begin_recovery(4.0, (-1.8, 0.0))
    manager.mission_plan_reacquired()
    assert manager.state == RecoveryState.MISSION_TRACKING

