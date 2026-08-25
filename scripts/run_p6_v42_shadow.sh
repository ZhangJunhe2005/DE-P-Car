#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="$PROJECT_ROOT/dep_car/config/p6_memory_navigation_v42_shadow.yaml"

if [[ "${1:-}" == "--stage" && "${2:-}" == "audit" ]]; then
  shift 2
  if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook 2>/dev/null)"
    while [[ "${CONDA_SHLVL:-0}" -gt 0 ]]; do conda deactivate; done
  fi
  source /opt/ros/noetic/setup.bash
  source "$PROJECT_ROOT/catkin_ws/devel/setup.bash"
  export PYTHONPATH="$PROJECT_ROOT/dep_car/src${PYTHONPATH:+:$PYTHONPATH}"
  exec /usr/bin/python3 "$PROJECT_ROOT/tools/audit_p6_v42_shadow.py" "$@"
fi

# V4.2 has only P6 shadow authority.  Reject an accidental active request at
# the host entry before ROS is started.
for argument in "$@"; do
  if [[ "$argument" == "active" || "$argument" == "--policy-mode=active" ]]; then
    echo "V4.2 active control is not authorized; use --policy-mode shadow" >&2
    exit 2
  fi
done

exec "$PROJECT_ROOT/scripts/run_memory_navigation.sh" \
  --config "$CONFIG" --policy-mode shadow "$@"
