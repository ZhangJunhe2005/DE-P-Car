from dep_car.core.gear import GearModeV1, GearShiftState, GearSupervisor
from dep_car.core.types import Gear


def test_supervisor_requires_stop_and_hold_before_direction_change():
    supervisor = GearSupervisor()
    first = supervisor.update(Gear.FORWARD, 0.0, 0.0)
    assert first.state == GearShiftState.SHIFT_HOLD and first.brake
    assert supervisor.update(Gear.FORWARD, 0.0, 0.25).drive_enabled

    stopping = supervisor.update(Gear.REVERSE, 0.4, 1.0)
    assert stopping.state == GearShiftState.STOPPING and stopping.brake
    assert stopping.mode == GearModeV1.BRAKE_TO_REVERSE
    holding = supervisor.update(Gear.REVERSE, 0.0, 1.1)
    assert holding.state == GearShiftState.SHIFT_HOLD and holding.brake
    assert supervisor.update(Gear.REVERSE, 0.0, 1.35).drive_enabled

    stopping_forward = supervisor.update(Gear.FORWARD, -0.3, 2.0)
    assert stopping_forward.brake and stopping_forward.mode == GearModeV1.BRAKE_TO_FORWARD
    assert supervisor.update(Gear.FORWARD, 0.0, 2.1).mode == GearModeV1.BRAKE_TO_FORWARD
    forward = supervisor.update(Gear.FORWARD, 0.0, 2.35)
    assert forward.drive_enabled and forward.mode == GearModeV1.FORWARD


def test_neutral_always_brakes():
    decision = GearSupervisor().update(Gear.NEUTRAL, 0.0, 0.0)
    assert decision.brake and decision.engaged == Gear.NEUTRAL
