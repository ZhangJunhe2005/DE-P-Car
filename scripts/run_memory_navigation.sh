#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
POLICY_PYTHON="${DEP_CAR_POLICY_PYTHON:-}"

if [[ -z "$POLICY_PYTHON" ]] && command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base 2>/dev/null || true)"
  if [[ -n "$CONDA_BASE" && -x "$CONDA_BASE/envs/yopo/bin/python" ]]; then
    POLICY_PYTHON="$CONDA_BASE/envs/yopo/bin/python"
  fi
fi
if [[ -z "$POLICY_PYTHON" || ! -x "$POLICY_PYTHON" ]]; then
  echo "PyTorch policy interpreter not found." >&2
  echo "Create Conda env 'yopo' or export DEP_CAR_POLICY_PYTHON=/path/to/python." >&2
  exit 2
fi

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook 2>/dev/null)"
  while [[ "${CONDA_SHLVL:-0}" -gt 0 ]]; do
    conda deactivate
  done
fi

source /opt/ros/noetic/setup.bash
source "$PROJECT_ROOT/catkin_ws/devel/setup.bash"
export DEP_CAR_POLICY_PYTHON="$POLICY_PYTHON"
export PYTHONPATH="$PROJECT_ROOT/dep_car/src${PYTHONPATH:+:$PYTHONPATH}"

exec /usr/bin/python3 "$PROJECT_ROOT/tools/run_memory_navigation.py" "$@"
