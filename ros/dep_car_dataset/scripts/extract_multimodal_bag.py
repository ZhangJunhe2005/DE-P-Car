#!/usr/bin/env python3
"""Deterministically materialize synchronized V2 samples from a raw rosbag."""

import argparse
import bisect
import copy
import hashlib
import json
import math
import os
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path

import numpy as np
import rosbag
from sensor_msgs import point_cloud2


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "dep_car/src"))
from dep_car.core.types import Candidate as CoreCandidate
from dep_car.core.types import Gear
from dep_car.core.recovery import RecoveryConfig
from dep_car.core.vehicle import center_steering_from_wheel_angles, world_velocity_to_body_longitudinal
from dep_car.perception.bev import LidarBEVConfig, build_lidar_bev, lidar_bev_preprocessing_contract
from dep_car.perception.pointcloud import filter_lidar_obstacles
from dep_car.training.dataset import MANEUVER_MODES, save_multimodal_sample
from dep_car.training.synchronization import bracket, interpolate_linear, interpolate_matrix, interpolation_alpha, quaternion_slerp


TOPICS = (
    "/camera/depth/image_raw",
    "/velodyne_points",
    "/base_pose_ground_truth",
    "/imu/data",
    "/urban_model/joint_states",
    "/dep_car/candidates",
    "/dep_car/global_route",
    "/dep_car/local_route_command",
    "/tf",
    "/tf_static",
    "/dep_car/lidar/bev",
    "/odometry/filtered",
    "/dep_car/policy_query",
    "/dep_car/policy_candidates_raw",
    "/dep_car/dagger_teacher_forward",
    "/dep_car/dagger_teacher_reverse",
    "/dep_car/cmd_ackermann",
)


def yaw_from_quaternion(quaternion):
    return math.atan2(
        2.0 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y),
        1.0 - 2.0 * (quaternion.y * quaternion.y + quaternion.z * quaternion.z),
    )


def matrix_from_transform(transform):
    q = transform.rotation
    x, y, z, w = q.x, q.y, q.z, q.w
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, :3] = (
        (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
        (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
        (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
    )
    matrix[:3, 3] = (transform.translation.x, transform.translation.y, transform.translation.z)
    return matrix


def normalize_frame(frame):
    return frame.lstrip("/")


def build_tf_graph(tf_messages):
    static, dynamic = {}, defaultdict(list)
    for topic, message, bag_time in tf_messages:
        for stamped in message.transforms:
            parent, child = normalize_frame(stamped.header.frame_id), normalize_frame(stamped.child_frame_id)
            matrix = matrix_from_transform(stamped.transform)
            if topic == "/tf_static":
                static[(child, parent)] = matrix
            else:
                stamp = stamped.header.stamp.to_sec() or bag_time.to_sec()
                dynamic[(child, parent)].append((stamp, matrix))
    for history in dynamic.values():
        history.sort(key=lambda item: item[0])
    return static, dynamic


def resolve_transform(graph, target, source, stamp, maximum_source_distance=0.20):
    target, source = normalize_frame(target), normalize_frame(source)
    if target == source:
        return np.eye(4, dtype=np.float32)
    static, dynamic = graph
    adjacency = defaultdict(list)
    for (child, parent), matrix in static.items():
        adjacency[child].append((parent, matrix))
        adjacency[parent].append((child, np.linalg.inv(matrix).astype(np.float32)))
    for (child, parent), history in dynamic.items():
        pair = bracket(history, stamp, maximum_source_distance)
        if pair is None:
            continue
        first, second = pair
        alpha = interpolation_alpha(first[0], second[0], stamp)
        matrix = first[1] if first is second else interpolate_matrix(first[1], second[1], alpha)
        adjacency[child].append((parent, matrix))
        adjacency[parent].append((child, np.linalg.inv(matrix).astype(np.float32)))
    queue = deque(((source, np.eye(4, dtype=np.float32)),))
    visited = {source}
    while queue:
        frame, accumulated = queue.popleft()
        for neighbour, edge in adjacency.get(frame, ()):
            if neighbour in visited:
                continue
            composed = edge @ accumulated
            if neighbour == target:
                return composed.astype(np.float32)
            visited.add(neighbour)
            queue.append((neighbour, composed))
    raise ValueError("no recorded TF chain from %s to %s" % (source, target))


def message_time(message, bag_time):
    if hasattr(message, "header") and message.header.stamp.to_sec() > 0.0:
        return message.header.stamp.to_sec()
    return bag_time.to_sec()


def nearest(entries, stamp, maximum_skew):
    if not entries:
        return None
    times = [item[0] for item in entries]
    index = bisect.bisect_left(times, stamp)
    choices = entries[max(0, index - 1):min(len(entries), index + 1)]
    selected = min(choices, key=lambda item: abs(item[0] - stamp))
    return selected if abs(selected[0] - stamp) <= maximum_skew else None


def latest(entries, stamp):
    if not entries:
        return None
    times = [item[0] for item in entries]
    index = bisect.bisect_right(times, stamp) - 1
    return entries[index] if index >= 0 else None


def interpolation_detail(pair, stamp, method):
    first, second = pair
    return {
        "method": "exact" if first[0] == second[0] else method,
        "target_stamp": stamp,
        "source_stamps": [first[0], second[0]],
        "source_indices": [first[1], second[1]],
        "alpha": interpolation_alpha(first[0], second[0], stamp),
    }


def interpolate_odometry(entries, stamp, maximum_source_distance):
    pair = bracket(entries, stamp, maximum_source_distance)
    if pair is None:
        return None
    first, second = pair
    alpha = interpolation_alpha(first[0], second[0], stamp)
    output = copy.deepcopy(first[2])
    p1, p2 = first[2].pose.pose.position, second[2].pose.pose.position
    position = interpolate_linear((p1.x, p1.y, p1.z), (p2.x, p2.y, p2.z), alpha)
    output.pose.pose.position.x, output.pose.pose.position.y, output.pose.pose.position.z = position
    q1, q2 = first[2].pose.pose.orientation, second[2].pose.pose.orientation
    quaternion = quaternion_slerp((q1.x, q1.y, q1.z, q1.w), (q2.x, q2.y, q2.z, q2.w), alpha)
    output.pose.pose.orientation.x, output.pose.pose.orientation.y = quaternion[:2]
    output.pose.pose.orientation.z, output.pose.pose.orientation.w = quaternion[2:]
    for group in ("linear", "angular"):
        a = getattr(first[2].twist.twist, group)
        b = getattr(second[2].twist.twist, group)
        value = interpolate_linear((a.x, a.y, a.z), (b.x, b.y, b.z), alpha)
        target = getattr(output.twist.twist, group)
        target.x, target.y, target.z = value
    return output, interpolation_detail(pair, stamp, "linear+slerp")


def interpolate_joint_state(entries, stamp, maximum_source_distance):
    pair = bracket(entries, stamp, maximum_source_distance)
    if pair is None:
        return None
    first, second = pair
    if first[2].name != second[2].name:
        raise ValueError("joint names changed across interpolation bracket")
    alpha = interpolation_alpha(first[0], second[0], stamp)
    output = copy.deepcopy(first[2])
    output.position = interpolate_linear(first[2].position, second[2].position, alpha).tolist()
    if first[2].velocity and len(first[2].velocity) == len(second[2].velocity):
        output.velocity = interpolate_linear(first[2].velocity, second[2].velocity, alpha).tolist()
    return output, interpolation_detail(pair, stamp, "linear")


def interpolate_imu(entries, stamp, maximum_source_distance):
    pair = bracket(entries, stamp, maximum_source_distance)
    if pair is None:
        return None
    first, second = pair
    alpha = interpolation_alpha(first[0], second[0], stamp)
    output = copy.deepcopy(first[2])
    q1, q2 = first[2].orientation, second[2].orientation
    quaternion = quaternion_slerp((q1.x, q1.y, q1.z, q1.w), (q2.x, q2.y, q2.z, q2.w), alpha)
    output.orientation.x, output.orientation.y, output.orientation.z, output.orientation.w = quaternion
    for group in ("angular_velocity", "linear_acceleration"):
        a, b = getattr(first[2], group), getattr(second[2], group)
        value = interpolate_linear((a.x, a.y, a.z), (b.x, b.y, b.z), alpha)
        target = getattr(output, group)
        target.x, target.y, target.z = value
    return output, interpolation_detail(pair, stamp, "linear+slerp")


def imu_vector(message):
    return np.asarray([
        message.orientation.x, message.orientation.y, message.orientation.z, message.orientation.w,
        message.angular_velocity.x, message.angular_velocity.y, message.angular_velocity.z,
        message.linear_acceleration.x, message.linear_acceleration.y, message.linear_acceleration.z,
    ], dtype=np.float32)


def raw_message_reference(topic, entry):
    return {"topic": topic, "message_index": int(entry[1]), "timestamp": float(entry[0])}


def infer_maneuver_mode(current_gear, requested_gear, path_gears):
    sequence = [int(gear) for gear in path_gears if int(gear) != 0]
    changes = sum(first != second for first, second in zip(sequence, sequence[1:]))
    if changes >= 2:
        return "THREE_POINT_TURN"
    if int(current_gear) != int(requested_gear):
        return "GEAR_SHIFT"
    if int(requested_gear) == -1:
        return "REVERSE_EXIT"
    return "NORMAL"


def decode_depth(message, minimum=0.2, maximum=10.0):
    if message.encoding != "32FC1":
        raise ValueError("depth input must use 32FC1")
    depth = np.frombuffer(message.data, dtype=np.float32).reshape(message.height, message.width).copy()
    validity = np.isfinite(depth) & (depth >= minimum) & (depth <= maximum)
    depth[~validity] = maximum
    return depth, validity.astype(np.uint8)


def decode_cloud(message, matrix):
    fields = {field.name for field in message.fields}
    selected = [name for name in ("x", "y", "z", "intensity", "ring") if name in fields]
    if selected[:3] != ["x", "y", "z"]:
        raise ValueError("point cloud lacks x/y/z")
    values = np.asarray(list(point_cloud2.read_points(message, field_names=selected, skip_nans=True)), dtype=np.float32)
    if values.size == 0:
        return np.empty((0, 5), dtype=np.float32)
    values = values.reshape((-1, len(selected)))
    output = np.zeros((len(values), 5), dtype=np.float32)
    homogeneous = np.column_stack((values[:, :3], np.ones(len(values), dtype=np.float32)))
    output[:, :3] = (matrix @ homogeneous.T).T[:, :3]
    for source, target in (("intensity", 3), ("ring", 4)):
        if source in selected:
            output[:, target] = values[:, selected.index(source)]
    return output


def decode_bev(message):
    if message.encoding != "32FC6" or message.height != 160 or message.width != 160:
        raise ValueError("recorded LiDAR BEV must be 32FC6 160x160")
    values = np.frombuffer(message.data, dtype=np.float32)
    if values.size != 6 * 160 * 160:
        raise ValueError("recorded LiDAR BEV byte count is invalid")
    return values.reshape(160, 160, 6).transpose(2, 0, 1).copy()


def ground_truth_chassis_to_map(odom):
    """Training-only static-map pose; never exposed to the deployed policy."""

    yaw = yaw_from_quaternion(odom.pose.pose.orientation)
    cosine, sine = math.cos(yaw), math.sin(yaw)
    matrix = np.eye(4, dtype=np.float32)
    matrix[0, 0], matrix[0, 1] = cosine, -sine
    matrix[1, 0], matrix[1, 1] = sine, cosine
    matrix[0, 3] = float(odom.pose.pose.position.x)
    matrix[1, 3] = float(odom.pose.pose.position.y)
    matrix[2, 3] = float(odom.pose.pose.position.z)
    return matrix


def route_from_policy_query(message, maximum_points=80):
    rows = np.asarray(
        [(pose.x, pose.y, pose.theta) for pose in message.route_corridor_body],
        dtype=np.float32,
    )
    if rows.ndim != 2 or rows.shape[1:] != (3,) or len(rows) < 2:
        raise ValueError("V4.3 policy query has no usable route corridor")
    return rows[:maximum_points].copy()


def selected_candidate(message):
    feasible = [candidate for candidate in message.candidates if candidate.feasible]
    if not feasible:
        return None
    selected_id = int(message.selected_candidate_id)
    for candidate in feasible:
        if int(candidate.candidate_id) == selected_id:
            return candidate
    return min(feasible, key=lambda item: (float(item.guidance_cost), int(item.candidate_id)))


def teacher_candidate_cost(candidate, subgoal, gear, current_gear):
    if candidate is None or not candidate.path.poses:
        return float("inf"), -float("inf"), float("inf")
    pose = candidate.path.poses[-1].pose
    endpoint = np.asarray([pose.position.x, pose.position.y], dtype=np.float64)
    target = np.asarray(subgoal, dtype=np.float64)
    start_distance = float(np.linalg.norm(target))
    end_distance = float(np.linalg.norm(target - endpoint))
    progress = start_distance - end_distance
    target_bearing = math.atan2(target[1] - endpoint[1], target[0] - endpoint[0])
    terminal_yaw = yaw_from_quaternion(pose.orientation)
    heading_error = abs(math.atan2(
        math.sin(terminal_yaw - target_bearing),
        math.cos(terminal_yaw - target_bearing),
    ))
    shift_penalty = 0.10 if int(current_gear) in (-1, 1) and int(current_gear) != int(gear) else 0.0
    # Reverse is not forbidden: it pays a small efficiency cost only when a
    # safe forward primitive produces comparable route progress.
    reverse_penalty = 0.22 if int(gear) < 0 else 0.0
    cost = end_distance + 0.28 * heading_error + shift_penalty + reverse_penalty
    return cost, progress, heading_error


def choose_dagger_teacher(forward_message, reverse_message, query, current_gear):
    forward = selected_candidate(forward_message)
    reverse = selected_candidate(reverse_message)
    subgoal = (float(query.subgoal_body.x), float(query.subgoal_body.y))
    f_cost, f_progress, f_heading = teacher_candidate_cost(
        forward, subgoal, 1, current_gear
    )
    r_cost, r_progress, r_heading = teacher_candidate_cost(
        reverse, subgoal, -1, current_gear
    )
    if forward is None and reverse is None:
        # Do not discard precisely the closed-loop states DAgger is intended
        # to correct.  This bank choice is storage-only; the exact signed
        # Hybrid A* oracle at index time decides STOP/FORWARD/REVERSE and the
        # complete multi-action sequence from both preserved banks.
        preferred_gear = int(getattr(query, "requested_gear", 0))
        if preferred_gear not in (-1, 1):
            preferred_gear = -1 if subgoal[0] < -0.20 else 1
        chosen = reverse_message if preferred_gear < 0 else forward_message
        reason = "NO_HARD_SAFE_BANK_PRESERVED_FOR_OFFLINE_SEQUENCE_ORACLE"
    elif forward is None:
        chosen, reason = reverse_message, "REVERSE_ONLY_HARD_SAFE"
    elif reverse is None:
        chosen, reason = forward_message, "FORWARD_ONLY_HARD_SAFE"
    else:
        # A forward candidate that is making non-negative progress retains
        # authority unless reverse offers a material geometric advantage.
        reverse_advantage = f_cost - r_cost
        route_is_rearward = subgoal[0] < -0.20
        if (
            (route_is_rearward and r_progress > f_progress + 0.08)
            or reverse_advantage > 0.30
            or (f_progress < -0.08 and r_progress > f_progress + 0.15)
        ):
            chosen, reason = reverse_message, "REVERSE_REQUIRED_FOR_ROUTE_ALIGNMENT"
        else:
            chosen, reason = forward_message, "FORWARD_PREFERRED_WHEN_COMPARABLY_SAFE"
    diagnostics = {
        "forward_hard_safe": forward is not None,
        "reverse_hard_safe": reverse is not None,
        "forward_cost": f_cost,
        "forward_progress_m": f_progress,
        "forward_heading_error_rad": f_heading,
        "reverse_cost": r_cost,
        "reverse_progress_m": r_progress,
        "reverse_heading_error_rad": r_heading,
    }
    return chosen, int(chosen.requested_gear), reason, diagnostics


def actual_steering(message, simulator_positive_right=True):
    positions = dict(zip(message.name, message.position))
    left, right = positions["front_left_steer_joint"], positions["front_right_steer_joint"]
    return center_steering_from_wheel_angles(left, right, simulator_positive_right)


def body_xy(world_x, world_y, odom):
    heading = yaw_from_quaternion(odom.pose.pose.orientation)
    dx, dy = world_x - odom.pose.pose.position.x, world_y - odom.pose.pose.position.y
    return math.cos(heading) * dx + math.sin(heading) * dy, -math.sin(heading) * dx + math.cos(heading) * dy


def signed_body_speed(odom):
    heading = yaw_from_quaternion(odom.pose.pose.orientation)
    velocity = odom.twist.twist.linear
    return world_velocity_to_body_longitudinal(velocity.x, velocity.y, heading)


def local_route(message, odom, maximum_points=80):
    if not message.points:
        raise ValueError("route contains no points")
    position = np.asarray([odom.pose.pose.position.x, odom.pose.pose.position.y])
    world = np.asarray([[point.pose.position.x, point.pose.position.y] for point in message.points])
    start = int(np.argmin(np.linalg.norm(world - position, axis=1)))
    heading = yaw_from_quaternion(odom.pose.pose.orientation)
    rows, gears = [], []
    for point in message.points[start:start + maximum_points]:
        x, y = body_xy(point.pose.position.x, point.pose.position.y, odom)
        route_yaw = yaw_from_quaternion(point.pose.orientation)
        rows.append((x, y, math.atan2(math.sin(route_yaw - heading), math.cos(route_yaw - heading))))
        gears.append(point.gear)
    return np.asarray(rows, dtype=np.float32), np.asarray(gears, dtype=np.int8)


def recovery_route(candidate_message, fallback_distance=RecoveryConfig().reverse_distance_m):
    """Build a body-frame reference for a deterministic reverse recovery bank."""

    if hasattr(candidate_message, "subgoal_body"):
        gx = float(candidate_message.subgoal_body.x)
        gy = float(candidate_message.subgoal_body.y)
    else:
        # CandidateArrayV1 bags predate explicit context but already carry the
        # supervisor-requested gear, making a mission/recovery mismatch clear.
        gx, gy = -float(fallback_distance), 0.0
    distance = max(1e-6, math.hypot(gx, gy))
    count = max(2, int(math.ceil(distance / 0.25)) + 1)
    path = np.zeros((count, 3), dtype=np.float32)
    path[:, 0] = np.linspace(0.0, gx, count, dtype=np.float32)
    path[:, 1] = np.linspace(0.0, gy, count, dtype=np.float32)
    gears = np.full(count, int(candidate_message.requested_gear), dtype=np.int8)
    return path, gears, gx, gy


def reference_curvature(path):
    if len(path) < 3:
        return 0.0
    distance = float(np.linalg.norm(path[2, :2] - path[0, :2]))
    return 0.0 if distance < 1e-4 else float(path[2, 2] - path[0, 2]) / distance


def core_candidates(message):
    output = []
    for item in message.candidates:
        count = len(item.path.poses)
        trajectory = np.zeros((count, 6), dtype=np.float64)
        trajectory[:, 0] = np.linspace(0.0, item.duration, count)
        for index, pose in enumerate(item.path.poses):
            trajectory[index, 1:3] = (pose.pose.position.x, pose.pose.position.y)
            trajectory[index, 3] = yaw_from_quaternion(pose.pose.orientation)
            trajectory[index, 4:6] = (item.speed_anchor, item.steering_anchor)
        output.append(CoreCandidate(
            item.candidate_id, item.speed_anchor, item.steering_anchor, item.duration, trajectory,
            gear=Gear(int(item.gear)), retime_factor=item.retime_factor,
            learned_score=item.learned_score, guidance_cost=item.guidance_cost,
            static_clearance=item.static_clearance, dynamic_clearance=item.dynamic_clearance,
            feasible=item.feasible, veto_reason=item.veto_reason,
        ))
    return output


def dagger_bank_arrays(message, prefix):
    """Preserve both hard-vetoed banks for the offline sequence oracle.

    The extraction-time one-step selector remains diagnostic metadata only.
    V4.3 training chooses the first bank from an exact signed plan produced at
    index time, so both alternatives must remain available in the sample.
    """

    values = core_candidates(message)
    if len(values) != 15:
        raise ValueError("V4.3 teacher bank must contain exactly 15 candidates")
    return {
        "dagger_%s_trajectories" % prefix: np.stack(
            [np.asarray(value.trajectory, dtype=np.float32) for value in values]
        ),
        "dagger_%s_feasible" % prefix: np.asarray(
            [value.feasible for value in values], dtype=np.uint8
        ),
        "dagger_%s_static_clearance" % prefix: np.asarray(
            [value.static_clearance for value in values], dtype=np.float32
        ),
        "dagger_%s_guidance_cost" % prefix: np.asarray(
            [value.guidance_cost for value in values], dtype=np.float32
        ),
    }


def append_dagger_arrays(path, arrays):
    """Atomically extend a V4.3 sample without changing the frozen V2 writer."""

    path = Path(path)
    extras = {}
    for name, value in arrays.items():
        if not str(name).startswith("dagger_"):
            raise ValueError("V4.3 arrays must use the dagger_ namespace")
        array = np.asarray(value)
        if array.dtype.kind in "fc" and not np.all(np.isfinite(array)):
            raise ValueError("V4.3 arrays must be finite")
        extras[str(name)] = array
    with np.load(path, allow_pickle=False) as archive:
        collision = set(archive.files) & set(extras)
        base = {name: np.asarray(archive[name]) for name in archive.files}
    # Re-extraction intentionally replaces earlier V4.3 arrays with the same
    # names, but a base-contract collision remains impossible.
    for name in collision:
        if not name.startswith("dagger_"):
            raise ValueError("V4.3 array collides with the base sample contract")
        base.pop(name)
    temporary = path.with_name(path.name + ".v43.tmp.npz")
    np.savez_compressed(str(temporary), **base, **extras)
    os.replace(temporary, path)


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bag", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "data/static_multimodal_v2")
    parser.add_argument("--map-uuid", required=True)
    parser.add_argument("--map-hash", default="")
    parser.add_argument("--simulator-seed", type=int, default=-1)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--maximum-samples", type=int, default=0)
    parser.add_argument(
        "--maximum-duration-s", type=float, default=0.0,
        help="limit accepted synchronized frames by their recorded timestamps",
    )
    parser.add_argument("--embed-raw-lidar", action="store_true", help="duplicate raw points into NPZ for diagnostics")
    parser.add_argument("--task-id", default="")
    parser.add_argument("--task-manifest-sha256", default="")
    parser.add_argument("--maneuver-mode", choices=sorted(MANEUVER_MODES))
    parser.add_argument("--episode-result", type=Path)
    parser.add_argument(
        "--dagger-v43", action="store_true",
        help="label re-observed guarded-policy states with two-bank route/safety teacher",
    )
    args = parser.parse_args()
    episode_result = {}
    if args.episode_result:
        episode_result = json.loads(args.episode_result.read_text(encoding="utf-8"))
    streams, tf_messages = defaultdict(list), []
    topic_indices = Counter()
    with rosbag.Bag(str(args.bag), "r") as bag:
        for topic, message, bag_time in bag.read_messages(topics=TOPICS):
            if topic in ("/tf", "/tf_static"):
                tf_messages.append((topic, message, bag_time))
            else:
                streams[topic].append((message_time(message, bag_time), topic_indices[topic], message))
                topic_indices[topic] += 1
    for values in streams.values():
        values.sort(key=lambda item: item[0])
    graph = build_tf_graph(tf_messages)
    accepted, rejected = 0, Counter()
    accepted_start_stamp = None
    bag_sha256 = file_hash(args.bag)
    anchors = streams["/velodyne_points"][::max(1, args.stride)]
    bev_contract = lidar_bev_preprocessing_contract(LidarBEVConfig())
    for lidar_stamp, lidar_index, cloud in anchors:
        if args.maximum_samples and accepted >= args.maximum_samples:
            break
        matches = {
            "depth": nearest(streams["/camera/depth/image_raw"], lidar_stamp, 0.05),
            "odom": interpolate_odometry(streams["/base_pose_ground_truth"], lidar_stamp, 0.02),
            "joint_state": interpolate_joint_state(streams["/urban_model/joint_states"], lidar_stamp, 0.05),
            "imu": interpolate_imu(streams["/imu/data"], lidar_stamp, 0.02),
            "route": latest(streams["/dep_car/global_route"], lidar_stamp),
            "route_command": latest(streams["/dep_car/local_route_command"], lidar_stamp),
        }
        if args.dagger_v43:
            matches.update({
                "bev": nearest(streams["/dep_car/lidar/bev"], lidar_stamp, 0.05),
                "policy_query": nearest(streams["/dep_car/policy_query"], lidar_stamp, 0.15),
                "policy_raw": nearest(streams["/dep_car/policy_candidates_raw"], lidar_stamp, 0.30),
                "teacher_forward": nearest(streams["/dep_car/dagger_teacher_forward"], lidar_stamp, 0.15),
                "teacher_reverse": nearest(streams["/dep_car/dagger_teacher_reverse"], lidar_stamp, 0.15),
                "executed_command": latest(streams["/dep_car/cmd_ackermann"], lidar_stamp),
            })
        else:
            matches["candidates"] = nearest(
                streams["/dep_car/candidates"], lidar_stamp, 0.15
            )
        missing = [name for name, value in matches.items() if value is None]
        if missing:
            rejected["missing_" + "+".join(missing)] += 1
            continue
        try:
            depth_stamp, depth_index, depth_message = matches["depth"]
            odom, odom_interpolation = matches["odom"]
            joints, joint_interpolation = matches["joint_state"]
            imu, imu_interpolation = matches["imu"]
            route_stamp, route_index, route = matches["route"]
            command_stamp, command_index, route_command = matches["route_command"]
            matrix = resolve_transform(graph, "chassis", cloud.header.frame_id, lidar_stamp)
            camera_matrix = resolve_transform(graph, "chassis", depth_message.header.frame_id, lidar_stamp)
            map_matrix = resolve_transform(graph, "map", "chassis", lidar_stamp)
            depth, depth_validity = decode_depth(depth_message)
            points = decode_cloud(cloud, matrix)
            environment_points = filter_lidar_obstacles(points)
            vehicle_yaw = yaw_from_quaternion(odom.pose.pose.orientation)
            speed = signed_body_speed(odom)
            current_gear = Gear.NEUTRAL if abs(speed) < 0.03 else (Gear.FORWARD if speed > 0 else Gear.REVERSE)
            dagger_metadata = {}
            dagger_arrays = {}
            if args.dagger_v43:
                _bev_stamp, _bev_index, bev_message = matches["bev"]
                _query_stamp, _query_index, policy_query = matches["policy_query"]
                _raw_stamp, _raw_index, policy_raw = matches["policy_raw"]
                _forward_stamp, _forward_index, teacher_forward = matches["teacher_forward"]
                _reverse_stamp, _reverse_index, teacher_reverse = matches["teacher_reverse"]
                _command_stamp, _command_index, executed_command = matches["executed_command"]
                selected_bank, teacher_gear, teacher_reason, teacher_diagnostics = choose_dagger_teacher(
                    teacher_forward, teacher_reverse, policy_query, int(policy_raw.current_gear)
                )
                candidate_message = selected_bank
                candidate_requested_gear = Gear(int(teacher_gear))
                if int(candidate_requested_gear) > 0:
                    candidate_stamp, candidate_index = _forward_stamp, _forward_index
                    candidate_topic = "/dep_car/dagger_teacher_forward"
                else:
                    candidate_stamp, candidate_index = _reverse_stamp, _reverse_index
                    candidate_topic = "/dep_car/dagger_teacher_reverse"
                current_gear = Gear(int(policy_raw.current_gear))
                recovery_mode = bool(policy_query.recovery_mode)
                path = route_from_policy_query(policy_query)
                path_gears = np.full(len(path), int(candidate_requested_gear), dtype=np.int8)
                gx, gy = float(policy_query.subgoal_body.x), float(policy_query.subgoal_body.y)
                heading_error = float(policy_query.heading_error)
                map_matrix = ground_truth_chassis_to_map(odom)
                processed_bev = decode_bev(bev_message)
                raw_candidates = list(policy_raw.candidates)
                raw_selected = min(
                    raw_candidates,
                    key=lambda item: (float(item.learned_score), int(item.candidate_id)),
                ) if raw_candidates else None
                raw_sequence = (
                    [int(value) for value, keep in zip(raw_selected.action_gears, raw_selected.action_mask) if keep]
                    if raw_selected is not None else []
                )
                dagger_metadata = {
                    "dagger_schema": "DEPCarV43ClosedLoopObservationV1",
                    "dagger_reobserved_state": True,
                    "dagger_teacher_gear": int(candidate_requested_gear),
                    "dagger_teacher_reason": teacher_reason,
                    "dagger_teacher_diagnostics": teacher_diagnostics,
                    "dagger_model_raw_sequence": raw_sequence,
                    "dagger_executed_gear": int(executed_command.gear),
                    "dagger_gear_history": [float(value) for value in policy_raw.gear_history],
                    "dagger_policy_generation": int(policy_raw.generation),
                    "dagger_ground_truth_used_for_offline_map_label_only": True,
                }
                dagger_arrays.update(dagger_bank_arrays(teacher_forward, "forward"))
                dagger_arrays.update(dagger_bank_arrays(teacher_reverse, "reverse"))
            else:
                candidate_stamp, candidate_index, candidate_message = matches["candidates"]
                candidate_requested_gear = Gear(int(candidate_message.requested_gear))
                recovery_mode = bool(getattr(candidate_message, "recovery_mode", False)) or (
                    int(candidate_requested_gear) != int(route_command.requested_gear)
                )
                if recovery_mode:
                    path, path_gears, gx, gy = recovery_route(candidate_message)
                    heading_error = 0.0
                else:
                    path, path_gears = local_route(route, odom)
                    gx, gy = body_xy(
                        route_command.target.position.x,
                        route_command.target.position.y,
                        odom,
                    )
                    target_yaw = yaw_from_quaternion(route_command.target.orientation)
                    heading_error = math.atan2(
                        math.sin(target_yaw - vehicle_yaw),
                        math.cos(target_yaw - vehicle_yaw),
                    )
                processed_bev = build_lidar_bev(environment_points, LidarBEVConfig())
                candidate_topic = "/dep_car/candidates"
            state = np.asarray([
                speed, imu.linear_acceleration.x, actual_steering(joints),
                imu.angular_velocity.z, gx, gy, math.sin(heading_error),
                math.cos(heading_error), reference_curvature(path),
            ], dtype=np.float32)
            if accepted_start_stamp is None:
                accepted_start_stamp = float(lidar_stamp)
            if (
                args.maximum_duration_s > 0.0
                and float(lidar_stamp) - accepted_start_stamp
                > args.maximum_duration_s + 1.0e-6
            ):
                break
            sample_id = "%s-%06d" % (args.bag.stem, accepted)
            sample_path = args.output / args.map_uuid / (sample_id + ".npz")
            save_multimodal_sample(
                sample_path,
                map_uuid=args.map_uuid,
                depth_metric=depth,
                depth_validity=depth_validity,
                lidar_points=points if args.embed_raw_lidar else None,
                lidar_bev=processed_bev,
                imu_measurement=imu_vector(imu),
                vehicle_state=state,
                current_gear=current_gear,
                requested_gear=candidate_requested_gear,
                local_path=path,
                local_path_gears=path_gears,
                subgoal_body=(gx, gy),
                candidates=core_candidates(candidate_message),
                timestamps={
                    "lidar": lidar_stamp, "depth": depth_stamp, "odom": lidar_stamp,
                    "joint_state": lidar_stamp, "imu": lidar_stamp, "candidates": candidate_stamp,
                },
                transforms={
                    "lidar_to_chassis": {"matrix": matrix.tolist(), "measurement_stamp": lidar_stamp, "source_frame": normalize_frame(cloud.header.frame_id), "target_frame": "chassis"},
                    "camera_to_chassis": {"matrix": camera_matrix.tolist(), "measurement_stamp": lidar_stamp, "source_frame": normalize_frame(depth_message.header.frame_id), "target_frame": "chassis"},
                    "chassis_to_map": {"matrix": map_matrix.tolist(), "measurement_stamp": lidar_stamp, "source_frame": "chassis", "target_frame": "map"},
                },
                raw_authority={
                    "kind": "rosbag_reference", "bag_path": str(args.bag), "bag_sha256": bag_sha256,
                    "embedded_lidar": bool(args.embed_raw_lidar),
                    "messages": {
                        "lidar": raw_message_reference("/velodyne_points", (lidar_stamp, lidar_index, cloud)),
                        "depth": raw_message_reference("/camera/depth/image_raw", matches["depth"]),
                        "imu": raw_message_reference("/imu/data", bracket(streams["/imu/data"], lidar_stamp, 0.02)[0]),
                        "imu_before": raw_message_reference("/imu/data", bracket(streams["/imu/data"], lidar_stamp, 0.02)[0]),
                        "imu_after": raw_message_reference("/imu/data", bracket(streams["/imu/data"], lidar_stamp, 0.02)[1]),
                        "odom_before": raw_message_reference("/base_pose_ground_truth", bracket(streams["/base_pose_ground_truth"], lidar_stamp, 0.02)[0]),
                        "odom_after": raw_message_reference("/base_pose_ground_truth", bracket(streams["/base_pose_ground_truth"], lidar_stamp, 0.02)[1]),
                        "joint_before": raw_message_reference("/urban_model/joint_states", bracket(streams["/urban_model/joint_states"], lidar_stamp, 0.05)[0]),
                        "joint_after": raw_message_reference("/urban_model/joint_states", bracket(streams["/urban_model/joint_states"], lidar_stamp, 0.05)[1]),
                        "candidates": raw_message_reference(
                            candidate_topic,
                            (candidate_stamp, candidate_index, candidate_message),
                        ),
                        "route": raw_message_reference("/dep_car/global_route", (route_stamp, route_index, route)),
                        "route_command": raw_message_reference("/dep_car/local_route_command", (command_stamp, command_index, route_command)),
                    },
                },
                preprocessing={"lidar_bev": bev_contract},
                interpolation={"odom": odom_interpolation, "joint_state": joint_interpolation, "imu": imu_interpolation},
                lidar_timing={
                    "model": "gazebo_ray_instantaneous_snapshot", "scan_period_s": 0.10,
                    "scan_start_stamp": lidar_stamp, "scan_end_stamp": lidar_stamp,
                    "per_point_time_field_present": True, "per_point_time_available": False,
                    "per_point_time_observed_range_s": [0.0, 0.0], "deskew_applied": False,
                    "deskew_reason": "Gazebo cloud time field is present but every observed value is zero",
                },
                maneuver_mode=args.maneuver_mode or infer_maneuver_mode(current_gear, candidate_requested_gear, path_gears),
                metadata={
                    "sample_id": sample_id, "source_bag": args.bag.name,
                    "source_bag_sha256": bag_sha256, "map_occupancy_sha256": args.map_hash,
                    "simulator_seed": args.simulator_seed, "bev_contract": bev_contract["schema"],
                    "lidar_self_filter": "physical_chassis_box_plus_0.03m",
                    "lidar_points_include_self_returns": bool(args.embed_raw_lidar),
                    "formal_training_authority": True,
                    "pilot_task_id": args.task_id,
                    "pilot_task_manifest_sha256": args.task_manifest_sha256,
                    "pilot_episode_status": episode_result.get("status", ""),
                    "pilot_episode_result_schema": episode_result.get("schema", ""),
                    "candidate_context": "RECOVERY" if recovery_mode else "MISSION",
                    **dagger_metadata,
                },
            )
            if dagger_arrays:
                append_dagger_arrays(sample_path, dagger_arrays)
            accepted += 1
        except Exception as exc:
            rejected[type(exc).__name__ + ":" + str(exc)[:80]] += 1
    payload = {"status": "PASS" if accepted else "FAIL", "accepted": accepted, "rejected": dict(rejected)}
    print(json.dumps(payload, indent=2, sort_keys=True))
    raise SystemExit(0 if accepted else 1)


if __name__ == "__main__":
    main()
