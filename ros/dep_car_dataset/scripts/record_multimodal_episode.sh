#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 OUTPUT_BAG" >&2
  exit 2
fi

output_bag="$1"
mkdir -p "$(dirname -- "$output_bag")"
exec rosbag record --lz4 -O "$output_bag" \
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
