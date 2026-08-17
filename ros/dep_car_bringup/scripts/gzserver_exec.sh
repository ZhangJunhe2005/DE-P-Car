#!/usr/bin/env bash
# Run the real Gazebo process as the roslaunch child so Ctrl+C reaches it.
set -eo pipefail

gazebo_master_uri="${GAZEBO_MASTER_URI:-}"
gazebo_database_uri="${GAZEBO_MODEL_DATABASE_URI:-}"
gazebo_prefix="$(pkg-config --variable=prefix gazebo)"
source "$gazebo_prefix/share/gazebo/setup.sh"
if [[ -n "$gazebo_master_uri" ]]; then export GAZEBO_MASTER_URI="$gazebo_master_uri"; fi
if [[ -n "$gazebo_database_uri" ]]; then export GAZEBO_MODEL_DATABASE_URI="$gazebo_database_uri"; fi

gazebo_args=()
ros_remaps=()
for argument in "$@"; do
  if [[ "$argument" == *":="* ]]; then ros_remaps+=("$argument"); else gazebo_args+=("$argument"); fi
done

paths_plugin="$(catkin_find --first-only libgazebo_ros_paths_plugin.so)"
api_plugin="$(catkin_find --first-only libgazebo_ros_api_plugin.so)"
exec /usr/bin/gzserver "${gazebo_args[@]}" -s "$paths_plugin" -s "$api_plugin" "${ros_remaps[@]}"

