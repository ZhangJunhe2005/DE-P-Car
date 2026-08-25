#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="$PROJECT_ROOT/dep_car/config/p6_memory_navigation_v42_guarded.yaml"

if [[ "${1:-}" == "--stage" && "${2:-}" == "audit" ]]; then
  shift 2
  if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook 2>/dev/null)"
    while [[ "${CONDA_SHLVL:-0}" -gt 0 ]]; do conda deactivate; done
  fi
  source /opt/ros/noetic/setup.bash
  source "$PROJECT_ROOT/catkin_ws/devel/setup.bash"
  export PYTHONPATH="$PROJECT_ROOT/dep_car/src${PYTHONPATH:+:$PYTHONPATH}"
  exec /usr/bin/python3 "$PROJECT_ROOT/tools/audit_p6_v42_guarded.py" "$@"
fi

# This entry exists only for Gazebo development.  The generic `active` mode
# and deterministic motion fallback remain unavailable to V4.2.
for argument in "$@"; do
  if [[ "$argument" == "active" || "$argument" == "shadow" || "$argument" == --policy-mode* ]]; then
    echo "V4.2 guarded entry fixes policy mode to guarded Gazebo simulation" >&2
    exit 2
  fi
done

exec "$PROJECT_ROOT/scripts/run_memory_navigation.sh" \
  --config "$CONFIG" --policy-mode guarded "$@"
