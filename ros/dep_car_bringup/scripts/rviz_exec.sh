#!/usr/bin/env bash
# RViz/Qt may ignore roslaunch's SIGINT. Translate it into a prompt, clean
# SIGTERM so roslaunch does not have to escalate after its shutdown timeout.
set -eo pipefail

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
/opt/ros/noetic/lib/rviz/rviz "$@" &
child_pid=$!
set +e
wait "$child_pid"
status=$?
set -e
exit "$status"
