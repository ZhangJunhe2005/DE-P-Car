#!/usr/bin/env python3
"""Deterministically materialize synchronized V2 samples from a raw rosbag."""

import argparse
import bisect
import copy
import hashlib
import json
import math
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
    parser.add_argument("--embed-raw-lidar", action="store_true", help="duplicate raw points into NPZ for diagnostics")
    parser.add_argument("--task-id", default="")
    parser.add_argument("--task-manifest-sha256", default="")
    parser.add_argument("--maneuver-mode", choices=sorted(MANEUVER_MODES))
    parser.add_argument("--episode-result", type=Path)
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
            "candidates": nearest(streams["/dep_car/candidates"], lidar_stamp, 0.15),
            "route": latest(streams["/dep_car/global_route"], lidar_stamp),
            "route_command": latest(streams["/dep_car/local_route_command"], lidar_stamp),
        }
        missing = [name for name, value in matches.items() if value is None]
        if missing:
            rejected["missing_" + "+".join(missing)] += 1
            continue
        try:
            depth_stamp, depth_index, depth_message = matches["depth"]
            odom, odom_interpolation = matches["odom"]
            joints, joint_interpolation = matches["joint_state"]
            imu, imu_interpolation = matches["imu"]
            candidate_stamp, candidate_index, candidate_message = matches["candidates"]
            route_stamp, route_index, route = matches["route"]
            command_stamp, command_index, route_command = matches["route_command"]
            matrix = resolve_transform(graph, "chassis", cloud.header.frame_id, lidar_stamp)
            camera_matrix = resolve_transform(graph, "chassis", depth_message.header.frame_id, lidar_stamp)
            map_matrix = resolve_transform(graph, "map", "chassis", lidar_stamp)
            depth, depth_validity = decode_depth(depth_message)
            points = decode_cloud(cloud, matrix)
            environment_points = filter_lidar_obstacles(points)
            candidate_requested_gear = Gear(int(candidate_message.requested_gear))
            recovery_mode = bool(getattr(candidate_message, "recovery_mode", False)) or (
                int(candidate_requested_gear) != int(route_command.requested_gear)
            )
            if recovery_mode:
                path, path_gears, gx, gy = recovery_route(candidate_message)
            else:
                path, path_gears = local_route(route, odom)
                gx, gy = body_xy(route_command.target.position.x, route_command.target.position.y, odom)
            vehicle_yaw = yaw_from_quaternion(odom.pose.pose.orientation)
            if recovery_mode:
                heading_error = 0.0
            else:
                target_yaw = yaw_from_quaternion(route_command.target.orientation)
                heading_error = math.atan2(math.sin(target_yaw - vehicle_yaw), math.cos(target_yaw - vehicle_yaw))
            speed = signed_body_speed(odom)
            state = np.asarray([
                speed, imu.linear_acceleration.x, actual_steering(joints), imu.angular_velocity.z,
                gx, gy, math.sin(heading_error), math.cos(heading_error), reference_curvature(path),
            ], dtype=np.float32)
            current_gear = Gear.NEUTRAL if abs(speed) < 0.03 else (Gear.FORWARD if speed > 0 else Gear.REVERSE)
            sample_id = "%s-%06d" % (args.bag.stem, accepted)
            save_multimodal_sample(
                args.output / args.map_uuid / (sample_id + ".npz"),
                map_uuid=args.map_uuid,
                depth_metric=depth,
                depth_validity=depth_validity,
                lidar_points=points if args.embed_raw_lidar else None,
                lidar_bev=build_lidar_bev(environment_points, LidarBEVConfig()),
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
                        "candidates": raw_message_reference("/dep_car/candidates", matches["candidates"]),
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
                },
            )
            accepted += 1
        except Exception as exc:
            rejected[type(exc).__name__ + ":" + str(exc)[:80]] += 1
    payload = {"status": "PASS" if accepted else "FAIL", "accepted": accepted, "rejected": dict(rejected)}
    print(json.dumps(payload, indent=2, sort_keys=True))
    raise SystemExit(0 if accepted else 1)


if __name__ == "__main__":
    main()
