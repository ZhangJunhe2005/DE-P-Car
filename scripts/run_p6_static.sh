#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
POLICY_PYTHON="${DEP_CAR_POLICY_PYTHON:-/home/zjh/miniconda3/envs/yopo/bin/python}"

if [[ ! -x "$POLICY_PYTHON" ]]; then
  echo "P6 policy Python is unavailable: $POLICY_PYTHON" >&2
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

exec "$POLICY_PYTHON" "$PROJECT_ROOT/tools/run_p6_static.py" "$@"
