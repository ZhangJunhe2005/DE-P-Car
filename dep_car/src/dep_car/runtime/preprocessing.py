"""Online preprocessing matching the frozen P3/P5 tensor contract.

This module is deliberately outside the checkpoint's P4 implementation
allowlist.  It adapts ROS measurements to that immutable contract without
changing the trained network or its signed artifact identity.
"""

import math

import cv2
import numpy as np

from dep_car.core.state_contract import (
    ACCELERATION_LIMIT_MPS2,
    DECELERATION_LIMIT_MPS2,
    FORWARD_SPEED_LIMIT_MPS,
    REVERSE_SPEED_LIMIT_MPS,
    YAW_RATE_SCALE_RADPS,
)
from dep_car.core.vehicle import STEERING_OPERATING_LIMIT_RAD


DEPTH_MINIMUM_M = 0.20
DEPTH_MAXIMUM_M = 10.0
DEPTH_NETWORK_SIZE = (160, 96)  # OpenCV width, height.
LIDAR_BEV_SHAPE = (6, 160, 160)
LIDAR_HEIGHT_MINIMUM_M = 0.05
LIDAR_HEIGHT_MAXIMUM_M = 1.30


def normalize_depth_metric(
    depth_metric,
    *,
    minimum_m=DEPTH_MINIMUM_M,
    maximum_m=DEPTH_MAXIMUM_M,
):
    """Return ``[normalized depth, validity]`` exactly as P5 saw it."""

    depth = np.asarray(depth_metric, dtype=np.float32)
    if depth.ndim != 2 or not depth.size:
        raise ValueError("metric depth must be a non-empty 2-D array")
    if not 0.0 <= float(minimum_m) < float(maximum_m):
        raise ValueError("depth limits are invalid")
    validity = np.isfinite(depth) & (depth >= minimum_m) & (depth <= maximum_m)
    normalized = np.where(
        validity,
        np.clip(depth, 0.0, maximum_m) / maximum_m,
        1.0,
    ).astype(np.float32)
    normalized = cv2.resize(
        normalized, DEPTH_NETWORK_SIZE, interpolation=cv2.INTER_NEAREST
    )
    validity = cv2.resize(
        validity.astype(np.float32),
        DEPTH_NETWORK_SIZE,
        interpolation=cv2.INTER_NEAREST,
    )
    output = np.stack((normalized, validity)).astype(np.float32)
    if output.shape != (2, 96, 160) or not np.all(np.isfinite(output)):
        raise ValueError("normalized depth contract is invalid")
    return output


def normalize_lidar_bev(
    lidar_bev,
    *,
    minimum_height_m=LIDAR_HEIGHT_MINIMUM_M,
    maximum_height_m=LIDAR_HEIGHT_MAXIMUM_M,
):
    """Normalize the six raw BEV channels using the P3 obstacle interval."""

    output = np.asarray(lidar_bev, dtype=np.float32).copy()
    if output.shape != LIDAR_BEV_SHAPE or not np.all(np.isfinite(output)):
        raise ValueError("raw LiDAR BEV must be finite [6,160,160]")
    if maximum_height_m <= minimum_height_m:
        raise ValueError("LiDAR height normalization interval is invalid")
    output[0] = np.clip(output[0], 0.0, 1.0)
    output[1] = np.clip(output[1], 0.0, 1.0)
    occupied = output[0] >= 0.5
    for channel in (2, 3):
        normalized = np.clip(
            (output[channel] - minimum_height_m)
            / (maximum_height_m - minimum_height_m),
            0.0,
            1.0,
        )
        output[channel] = np.where(occupied, normalized, 0.0)
    output[4] = np.clip(output[4], 0.0, 1.0)
    output[5] = np.clip(output[5], 0.0, 1.0)
    return output.astype(np.float32, copy=False)


def build_policy_state(
    *,
    signed_speed,
    longitudinal_acceleration,
    steering,
    yaw_rate,
    subgoal_body,
    heading_error,
    reference_curvature,
    requested_gear,
    current_gear,
):
    """Build the physical 9-D state with P5's shift-boundary sanitation."""

    requested_gear = int(requested_gear)
    current_gear = int(current_gear)
    if requested_gear not in (-1, 1):
        raise ValueError("requested gear must be forward or reverse")
    if current_gear not in (-1, 0, 1):
        raise ValueError("current gear is invalid")
    subgoal = np.asarray(subgoal_body, dtype=np.float32)
    if subgoal.shape != (2,) or not np.all(np.isfinite(subgoal)):
        raise ValueError("subgoal_body must be finite [2]")
    values = np.asarray(
        [
            signed_speed,
            longitudinal_acceleration,
            steering,
            yaw_rate,
            subgoal[0],
            subgoal[1],
            math.sin(float(heading_error)),
            math.cos(float(heading_error)),
            reference_curvature,
        ],
        dtype=np.float32,
    )
    if not np.all(np.isfinite(values)):
        raise ValueError("policy state contains a non-finite value")
    values[0] = np.clip(
        values[0], -REVERSE_SPEED_LIMIT_MPS, FORWARD_SPEED_LIMIT_MPS
    )
    directed_acceleration = np.clip(
        requested_gear * values[1],
        -DECELERATION_LIMIT_MPS2,
        ACCELERATION_LIMIT_MPS2,
    )
    values[1] = requested_gear * directed_acceleration
    values[2] = np.clip(
        values[2], -STEERING_OPERATING_LIMIT_RAD, STEERING_OPERATING_LIMIT_RAD
    )
    values[3] = np.clip(values[3], -YAW_RATE_SCALE_RADPS, YAW_RATE_SCALE_RADPS)
    values[6:8] = np.clip(values[6:8], -1.0, 1.0)
    # The network never owns a moving direction change.  Training represented
    # an opposite-gear measurement at the executable, stopped boundary while
    # GearSupervisor retained the raw speed for stop-before-shift authority.
    if current_gear == -requested_gear:
        values[0] = 0.0
        values[1] = 0.0
    return values


def build_joint_policy_state(
    *,
    signed_speed,
    longitudinal_acceleration,
    steering,
    yaw_rate,
    subgoal_body,
    heading_error,
    reference_curvature,
):
    """Build V3/V4 measured state without prescribing a drive direction."""

    subgoal = np.asarray(subgoal_body, dtype=np.float32)
    if subgoal.shape != (2,) or not np.all(np.isfinite(subgoal)):
        raise ValueError("subgoal_body must be finite [2]")
    values = np.asarray(
        [
            signed_speed,
            longitudinal_acceleration,
            steering,
            yaw_rate,
            subgoal[0],
            subgoal[1],
            math.sin(float(heading_error)),
            math.cos(float(heading_error)),
            reference_curvature,
        ],
        dtype=np.float32,
    )
    if not np.all(np.isfinite(values)):
        raise ValueError("joint policy state contains a non-finite value")
    values[0] = np.clip(
        values[0], -REVERSE_SPEED_LIMIT_MPS, FORWARD_SPEED_LIMIT_MPS
    )
    values[1] = np.clip(
        values[1], -DECELERATION_LIMIT_MPS2, ACCELERATION_LIMIT_MPS2
    )
    values[2] = np.clip(
        values[2], -STEERING_OPERATING_LIMIT_RAD, STEERING_OPERATING_LIMIT_RAD
    )
    values[3] = np.clip(values[3], -YAW_RATE_SCALE_RADPS, YAW_RATE_SCALE_RADPS)
    values[6:8] = np.clip(values[6:8], -1.0, 1.0)
    return values


def current_gear_from_speed(signed_speed, stop_tolerance=0.03):
    speed = float(signed_speed)
    if abs(speed) <= float(stop_tolerance):
        return 0
    return 1 if speed > 0.0 else -1


__all__ = [
    "DEPTH_MAXIMUM_M",
    "DEPTH_MINIMUM_M",
    "LIDAR_BEV_SHAPE",
    "build_joint_policy_state",
    "build_policy_state",
    "current_gear_from_speed",
    "normalize_depth_metric",
    "normalize_lidar_bev",
]
