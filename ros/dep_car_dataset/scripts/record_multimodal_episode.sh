#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 OUTPUT_BAG [bz2|lz4|none]" >&2
  exit 2
fi

output_bag="$1"
compression="${2:-bz2}"
case "$compression" in
  bz2) compression_args=(--bz2) ;;
  lz4) compression_args=(--lz4) ;;
  none) compression_args=() ;;
  *)
    echo "unsupported rosbag compression: $compression" >&2
    exit 2
    ;;
esac
mkdir -p "$(dirname -- "$output_bag")"
exec rosbag record "${compression_args[@]}" -O "$output_bag" \
  /camera/depth/image_raw \
  /camera/depth/camera_info \
  /velodyne_points \
  /imu/data \
  /base_pose_ground_truth \
  /urban_model/joint_states \
  /dep_car/candidates \
  /dep_car/local_costmap \
  /dep_car/global_route \
  /dep_car/local_route_command \
  /dep_car/cmd_ackermann \
  /dep_car/planner_state \
  /move_base_simple/goal \
  /tf /tf_static
