#!/usr/bin/env bash
# Start Gazebo GUI and translate roslaunch's SIGINT into the SIGTERM that Qt
# reliably handles.  This avoids roslaunch waiting 15 seconds and reporting an
# "escalating to SIGTERM" shutdown error.
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
child_pid=""

shutdown_child() {
  trap - INT TERM
  if [[ -n "$child_pid" ]] && kill -0 "$child_pid" 2>/dev/null; then
    kill -TERM "$child_pid" 2>/dev/null || true
  fi
  wait "$child_pid" 2>/dev/null || true
  exit 0
}

trap shutdown_child INT TERM
/usr/bin/gzclient "${gazebo_args[@]}" -g "$paths_plugin" "${ros_remaps[@]}" &
child_pid=$!
set +e
wait "$child_pid"
status=$?
set -e
exit "$status"
