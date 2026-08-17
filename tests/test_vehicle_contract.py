import math

from dep_car.core.vehicle import (
    MAXIMUM_KINEMATIC_CURVATURE_PER_M,
    MINIMUM_KINEMATIC_TURN_RADIUS_M,
    STEERING_OPERATING_LIMIT_RAD,
    WHEELBASE_M,
    FRONT_TRACK_M,
    center_steering_from_wheel_angles,
    CALIBRATED_GUARANTEED_CURVATURE_PER_M,
    CALIBRATED_MINIMUM_TURN_RADIUS_M,
    PLANNER_ROLLOUT_WHEELBASE_M,
    world_velocity_to_body_longitudinal,
)


def test_p0_turning_contract_is_internally_consistent():
    expected = math.tan(STEERING_OPERATING_LIMIT_RAD) / WHEELBASE_M
    assert math.isclose(MAXIMUM_KINEMATIC_CURVATURE_PER_M, expected)
    assert math.isclose(MINIMUM_KINEMATIC_TURN_RADIUS_M, 1.0 / expected)
    assert math.isclose(CALIBRATED_GUARANTEED_CURVATURE_PER_M, 1.0 / CALIBRATED_MINIMUM_TURN_RADIUS_M)
    assert math.isclose(
        math.tan(STEERING_OPERATING_LIMIT_RAD) / PLANNER_ROLLOUT_WHEELBASE_M,
        CALIBRATED_GUARANTEED_CURVATURE_PER_M,
    )


def test_ackermann_wheel_angles_reconstruct_center_steering():
    center = STEERING_OPERATING_LIMIT_RAD
    simulator_center = -center
    radius = WHEELBASE_M / math.tan(simulator_center)
    left = math.atan(WHEELBASE_M / (radius + 0.5 * FRONT_TRACK_M))
    right = math.atan(WHEELBASE_M / (radius - 0.5 * FRONT_TRACK_M))
    assert math.isclose(center_steering_from_wheel_angles(left, right), center, abs_tol=1e-12)


def test_gazebo_world_velocity_is_projected_onto_vehicle_heading():
    assert math.isclose(world_velocity_to_body_longitudinal(0.0, 2.0, math.pi / 2.0), 2.0)
    assert math.isclose(world_velocity_to_body_longitudinal(0.0, -1.0, math.pi / 2.0), -1.0)
