#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="$PROJECT_ROOT/dep_car/config/p6_memory_navigation_v43_shadow.yaml"

if [[ "${1:-}" == "--stage" && "${2:-}" == "audit" ]]; then
  shift 2
  if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook 2>/dev/null)"
    while [[ "${CONDA_SHLVL:-0}" -gt 0 ]]; do conda deactivate; done
  fi
  source /opt/ros/noetic/setup.bash
  source "$PROJECT_ROOT/catkin_ws/devel/setup.bash"
  export PYTHONPATH="$PROJECT_ROOT/dep_car/src${PYTHONPATH:+:$PYTHONPATH}"
  exec /usr/bin/python3 "$PROJECT_ROOT/tools/audit_p6_v43_shadow.py" "$@"
fi

# The P5 acceptance scope is P6_SHADOW_ONLY.  A CLI spelling must not promote
# that artifact to guarded/active control or bypass the deterministic driver.
for argument in "$@"; do
  case "$argument" in
    active|guarded|shadow|--policy-mode*)
      echo "V4.3 entry fixes policy mode to shadow; learned control is not authorized" >&2
      exit 2
      ;;
  esac
done

exec "$PROJECT_ROOT/scripts/run_memory_navigation.sh" \
  --config "$CONFIG" --policy-mode shadow "$@"
