"""ROS-independent rear-wheel Ackermann odometry.

The online-navigation backend deliberately keeps this estimator separate from
Gazebo's pose plugin.  It accepts only joint-derived wheel speed and steering
and integrates the bicycle model in an ``odom`` frame.  An EKF may fuse its
output with an IMU heading and angular velocity, but neither stage has access
to map truth.
"""

from dataclasses import dataclass
import math


def wrap_angle(value: float) -> float:
    return math.atan2(math.sin(float(value)), math.cos(float(value)))


def compose_planar_pose(parent_from_child, child_from_body):
    """Compose ``parent<-child`` and ``child<-body`` planar poses."""

    tx, ty, tyaw = (float(value) for value in parent_from_child)
    x, y, yaw = (float(value) for value in child_from_body)
    cosine, sine = math.cos(tyaw), math.sin(tyaw)
    return (
        tx + cosine * x - sine * y,
        ty + sine * x + cosine * y,
        wrap_angle(tyaw + yaw),
    )


@dataclass(frozen=True)
class PlanarTransformCorrection:
    """One newly stamped rigid-transform correction.

    A TF listener can return the same latest transform for many high-rate
    odometry callbacks.  Treating every callback as a new correction makes
    downstream route-revalidation logic run at the EKF rate even though SLAM
    has not changed its estimate.
    """

    translation_delta: float
    yaw_delta: float


class PlanarTransformRevisionTracker:
    """Accept a planar transform at most once for each source TF stamp."""

    def __init__(self):
        self.stamp = None
        self.pose = None

    def observe(self, stamp, pose):
        if stamp == self.stamp:
            return None
        current = tuple(float(value) for value in pose)
        if len(current) != 3 or not all(math.isfinite(value) for value in current):
            raise ValueError("planar transform pose must contain three finite values")
        if self.pose is None:
            translation_delta = 0.0
            yaw_delta = 0.0
        else:
            translation_delta = math.hypot(
                current[0] - self.pose[0], current[1] - self.pose[1]
            )
            yaw_delta = wrap_angle(current[2] - self.pose[2])
        self.stamp = stamp
        self.pose = current
        return PlanarTransformCorrection(
            translation_delta=translation_delta,
            yaw_delta=yaw_delta,
        )


@dataclass(frozen=True)
class WheelOdometryState:
    stamp: float
    x: float
    y: float
    yaw: float
    speed: float
    yaw_rate: float


class AckermannWheelOdometry:
    """Integrate signed rear-wheel speed using measured center steering."""

    def __init__(
        self,
        wheel_radius: float,
        wheelbase: float,
        direction: float = 1.0,
        speed_deadband_mps: float = 0.0,
    ):
        if wheel_radius <= 0.0 or wheelbase <= 0.0:
            raise ValueError("wheel radius and wheelbase must be positive")
        if direction == 0.0:
            raise ValueError("wheel direction cannot be zero")
        if not math.isfinite(float(speed_deadband_mps)) or speed_deadband_mps < 0.0:
            raise ValueError("speed deadband must be finite and nonnegative")
        self.wheel_radius = float(wheel_radius)
        self.wheelbase = float(wheelbase)
        self.direction = math.copysign(1.0, float(direction))
        self.speed_deadband_mps = float(speed_deadband_mps)
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.last_stamp = None

    def reset(self, *, x: float = 0.0, y: float = 0.0, yaw: float = 0.0, stamp=None):
        self.x = float(x)
        self.y = float(y)
        self.yaw = wrap_angle(yaw)
        self.last_stamp = None if stamp is None else float(stamp)

    def update(
        self,
        stamp: float,
        left_wheel_velocity: float,
        right_wheel_velocity: float,
        steering: float,
    ) -> WheelOdometryState:
        stamp = float(stamp)
        if not math.isfinite(stamp):
            raise ValueError("odometry stamp must be finite")
        wheel_speed = 0.5 * (
            float(left_wheel_velocity) + float(right_wheel_velocity)
        )
        speed = self.direction * self.wheel_radius * wheel_speed
        # ros_control wheel joints retain small velocity noise while the
        # steering servo is moving against an active brake.  Feeding that
        # noise through tan(steering) fabricates both translation and yaw at
        # exactly the stop/shift phase where the user observed trail jumps.
        if abs(speed) < self.speed_deadband_mps:
            speed = 0.0
        yaw_rate = speed * math.tan(float(steering)) / self.wheelbase
        if self.last_stamp is not None:
            dt = stamp - self.last_stamp
            # Joint states can be repeated during simulator startup.  Ignore
            # non-monotonic samples and cap long gaps rather than fabricating
            # a large displacement from a stale wheel velocity.
            if 0.0 < dt <= 0.25:
                middle_yaw = self.yaw + 0.5 * yaw_rate * dt
                self.x += speed * math.cos(middle_yaw) * dt
                self.y += speed * math.sin(middle_yaw) * dt
                self.yaw = wrap_angle(self.yaw + yaw_rate * dt)
        self.last_stamp = stamp
        return WheelOdometryState(
            stamp=stamp,
            x=self.x,
            y=self.y,
            yaw=self.yaw,
            speed=speed,
            yaw_rate=yaw_rate,
        )
