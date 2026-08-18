#!/usr/bin/env python3
"""Reset a spawned Urban Car to the task pose after contact settling."""

import argparse
import json
import math
import time

import rospy
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import GetModelState, SetModelState
from std_srvs.srv import Empty
from tf.transformations import euler_from_quaternion, quaternion_from_euler


def yaw_from_quaternion(quaternion):
    return math.atan2(
        2.0 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y),
        1.0 - 2.0 * (quaternion.y * quaternion.y + quaternion.z * quaternion.z),
    )


def wrap_angle(value):
    return math.atan2(math.sin(value), math.cos(value))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="urban_model")
    parser.add_argument("--x", type=float, required=True)
    parser.add_argument("--y", type=float, required=True)
    parser.add_argument("--yaw", type=float, required=True)
    parser.add_argument("--pre-settle-s", type=float, default=1.0)
    parser.add_argument("--post-settle-s", type=float, default=0.25)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--position-tolerance", type=float, default=0.02)
    parser.add_argument("--yaw-tolerance", type=float, default=0.02)
    args, _ = parser.parse_known_args()
    rospy.init_node("dep_car_pilot_pose_reset", anonymous=True)
    service_names = (
        "/gazebo/get_model_state", "/gazebo/set_model_state",
        "/gazebo/pause_physics", "/gazebo/unpause_physics",
    )
    for service in service_names:
        rospy.wait_for_service(service, timeout=30.0)
    get_state = rospy.ServiceProxy("/gazebo/get_model_state", GetModelState)
    set_state = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)
    pause = rospy.ServiceProxy("/gazebo/pause_physics", Empty)
    unpause = rospy.ServiceProxy("/gazebo/unpause_physics", Empty)
    # P6 deliberately starts Gazebo paused so camera/LiDAR rendering plugins
    # can finish loading before their first frame.  This wait must therefore
    # use wall time; rospy.sleep would block forever while /clock is paused.
    time.sleep(args.pre_settle_s)
    history = []
    success = False
    for attempt in range(1, args.attempts + 1):
        before = get_state(args.model, "world")
        if not before.success:
            raise RuntimeError("get_model_state failed: " + before.status_message)
        quaternion = before.pose.orientation
        roll, pitch, _ = euler_from_quaternion((quaternion.x, quaternion.y, quaternion.z, quaternion.w))
        target_quaternion = quaternion_from_euler(roll, pitch, args.yaw)
        state = ModelState()
        state.model_name = args.model
        state.reference_frame = "world"
        state.pose = before.pose
        state.pose.position.x = args.x
        state.pose.position.y = args.y
        state.pose.orientation.x, state.pose.orientation.y = target_quaternion[:2]
        state.pose.orientation.z, state.pose.orientation.w = target_quaternion[2:]
        pause()
        try:
            response = set_state(state)
            if not response.success:
                raise RuntimeError("set_model_state failed: " + response.status_message)
        finally:
            unpause()
        time.sleep(args.post_settle_s)
        after = get_state(args.model, "world")
        distance = math.hypot(after.pose.position.x - args.x, after.pose.position.y - args.y)
        yaw_error = abs(wrap_angle(yaw_from_quaternion(after.pose.orientation) - args.yaw))
        history.append({
            "attempt": attempt, "stable_z": before.pose.position.z,
            "position_error_m": distance, "yaw_error_rad": yaw_error,
        })
        if distance <= args.position_tolerance and yaw_error <= args.yaw_tolerance:
            success = True
            break
    payload = {
        "schema": "DEPCarPilotPoseResetV1", "status": "PASS" if success else "FAIL",
        "model": args.model, "target": [args.x, args.y, args.yaw], "attempts": history,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    raise SystemExit(0 if success else 2)


if __name__ == "__main__":
    main()
