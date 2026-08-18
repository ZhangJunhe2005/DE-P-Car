#!/usr/bin/env bash
# RViz/Qt may ignore roslaunch's SIGINT. Translate it into a prompt, clean
# SIGTERM so roslaunch does not have to escalate after its shutdown timeout.
set -eo pipefail

child_pid=""
startup_delay="0"

if [[ "${1:-}" == "--delay" ]]; then
  if [[ $# -lt 2 ]]; then
    echo "rviz_exec.sh: --delay requires seconds" >&2
    exit 2
  fi
  startup_delay="$2"
  shift 2
fi

shutdown_child() {
  trap - INT TERM
  if [[ -n "$child_pid" ]] && kill -0 "$child_pid" 2>/dev/null; then
    kill -TERM "$child_pid" 2>/dev/null || true
  fi
  wait "$child_pid" 2>/dev/null || true
  exit 0
}

trap shutdown_child INT TERM
if [[ "$startup_delay" != "0" && "$startup_delay" != "0.0" ]]; then
  sleep "$startup_delay"
fi
/opt/ros/noetic/lib/rviz/rviz "$@" &
child_pid=$!
set +e
wait "$child_pid"
status=$?
set -e
exit "$status"
