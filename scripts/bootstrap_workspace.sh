#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE="$ROOT/catkin_ws/src"
mkdir -p "$SOURCE/third_party"

for package in "$ROOT"/ros/dep_car_*; do
  name="$(basename "$package")"
  [[ -e "$SOURCE/$name" ]] || ln -s "$package" "$SOURCE/$name"
done
[[ -e "$SOURCE/dep_car_core" ]] || ln -s "$ROOT/dep_car" "$SOURCE/dep_car_core"
[[ -e "$SOURCE/third_party/car_msgs" ]] || ln -s "$ROOT/third_party/car-simulator/msgs" "$SOURCE/third_party/car_msgs"
[[ -e "$SOURCE/third_party/spawn_car" ]] || ln -s "$ROOT/third_party/car-simulator/spawn_car" "$SOURCE/third_party/spawn_car"
[[ -e "$SOURCE/third_party/car_simulator" ]] || ln -s "$ROOT/third_party/car-simulator/car_simulator" "$SOURCE/third_party/car_simulator"

if [[ ! -e "$SOURCE/CMakeLists.txt" ]]; then
  (cd "$SOURCE" && catkin_init_workspace)
fi

missing=()
for package in effort_controllers joint_trajectory_controller velodyne_description velodyne_gazebo_plugins map_server; do
  rospack find "$package" >/dev/null 2>&1 || missing+=("$package")
done
if (( ${#missing[@]} )); then
  printf 'Missing ROS runtime packages: %s\n' "${missing[*]}" >&2
  printf '%s\n' 'Install: sudo apt-get install ros-noetic-effort-controllers ros-noetic-joint-trajectory-controller ros-noetic-velodyne-description ros-noetic-velodyne-gazebo-plugins ros-noetic-map-server' >&2
fi

cd "$ROOT/catkin_ws"
catkin_make -DCMAKE_BUILD_TYPE=Release -DPYTHON_EXECUTABLE=/usr/bin/python3

