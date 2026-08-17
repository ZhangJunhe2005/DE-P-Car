"""Authoritative geometry for the scaled Urban Car used by DE-P-Car."""

import math

# The upstream hifzhil/car-simulator Urban model is too large for the Arena
# indoor maps.  All physical linear dimensions are scaled uniformly.
URBAN_CAR_LINEAR_SCALE = 1.0 / 3.0

UPSTREAM_LENGTH_M = 1.49
UPSTREAM_WIDTH_M = 0.83
UPSTREAM_HEIGHT_M = 1.55
UPSTREAM_WHEELBASE_M = 0.98
UPSTREAM_FRONT_TRACK_M = 0.83
UPSTREAM_REAR_TRACK_M = 0.75
UPSTREAM_WHEEL_RADIUS_M = 0.19

LENGTH_M = UPSTREAM_LENGTH_M * URBAN_CAR_LINEAR_SCALE
WIDTH_M = UPSTREAM_WIDTH_M * URBAN_CAR_LINEAR_SCALE
HEIGHT_M = UPSTREAM_HEIGHT_M * URBAN_CAR_LINEAR_SCALE
WHEELBASE_M = UPSTREAM_WHEELBASE_M * URBAN_CAR_LINEAR_SCALE
FRONT_TRACK_M = UPSTREAM_FRONT_TRACK_M * URBAN_CAR_LINEAR_SCALE
REAR_TRACK_M = UPSTREAM_REAR_TRACK_M * URBAN_CAR_LINEAR_SCALE
WHEEL_RADIUS_M = UPSTREAM_WHEEL_RADIUS_M * URBAN_CAR_LINEAR_SCALE
MASS_SCALE = URBAN_CAR_LINEAR_SCALE ** 3
INERTIA_SCALE = URBAN_CAR_LINEAR_SCALE ** 5
EFFORT_SCALE = URBAN_CAR_LINEAR_SCALE ** 4

# The URDF limit applies to each wheel joint.  The inner Ackermann wheel reaches
# it before the equivalent bicycle-center angle does, so derive rather than
# copy the joint limit into planning.
STEERING_WHEEL_JOINT_HARD_LIMIT_RAD = 0.785398
MINIMUM_ACKERMANN_CENTER_RADIUS_M = (
    WHEELBASE_M / math.tan(STEERING_WHEEL_JOINT_HARD_LIMIT_RAD) + 0.5 * FRONT_TRACK_M
)
STEERING_OPERATING_LIMIT_RAD = math.atan(WHEELBASE_M / MINIMUM_ACKERMANN_CENTER_RADIUS_M)
MAXIMUM_KINEMATIC_CURVATURE_PER_M = math.tan(STEERING_OPERATING_LIMIT_RAD) / WHEELBASE_M
MINIMUM_KINEMATIC_TURN_RADIUS_M = 1.0 / MAXIMUM_KINEMATIC_CURVATURE_PER_M

# Conservative P0 Gazebo envelope: the largest radius observed at the center
# steering limit across low/medium speed and both directions was 0.5488 m.
# Round outward so rollout never assumes the ideal no-slip curvature.
CALIBRATED_MINIMUM_TURN_RADIUS_M = 0.55
CALIBRATED_GUARANTEED_CURVATURE_PER_M = 1.0 / CALIBRATED_MINIMUM_TURN_RADIUS_M
PLANNER_ROLLOUT_WHEELBASE_M = math.tan(STEERING_OPERATING_LIMIT_RAD) / CALIBRATED_GUARANTEED_CURVATURE_PER_M

# Safety clearances remain real-world distances; they are not visual/model
# dimensions and therefore are intentionally not divided by three.
FOOTPRINT_SAFETY_MARGIN_M = 0.12
DYNAMIC_EGO_RADIUS_M = 0.5 * WIDTH_M + FOOTPRINT_SAFETY_MARGIN_M


def center_steering_from_wheel_angles(left, right, simulator_positive_right=True):
    """Recover bicycle-center steering from the two Ackermann wheel joints."""

    left, right = float(left), float(right)
    if abs(left) < 1e-8 and abs(right) < 1e-8:
        return 0.0
    radii = []
    if abs(math.tan(left)) > 1e-8:
        radii.append(WHEELBASE_M / math.tan(left) - 0.5 * FRONT_TRACK_M)
    if abs(math.tan(right)) > 1e-8:
        radii.append(WHEELBASE_M / math.tan(right) + 0.5 * FRONT_TRACK_M)
    simulator_center = math.atan(WHEELBASE_M / (sum(radii) / len(radii)))
    return -simulator_center if simulator_positive_right else simulator_center


def world_velocity_to_body_longitudinal(vx, vy, yaw):
    """Project Gazebo P3D world-axis planar velocity onto the car heading."""

    return math.cos(float(yaw)) * float(vx) + math.sin(float(yaw)) * float(vy)
