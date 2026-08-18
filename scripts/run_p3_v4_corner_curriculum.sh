#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="$PROJECT_ROOT/dep_car/config/p3_v4_corner_curriculum.yaml"

if [[ $# -eq 0 ]]; then
  echo "Usage: bash scripts/run_p3_v4_corner_curriculum.sh --stage validate|prepare|collect|recover-extraction|qualify-invalid-goals|bundle|audit|status [options]" >&2
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

exec /home/zjh/miniconda3/envs/yopo/bin/python \
  "$PROJECT_ROOT/tools/run_p3_v3_pipeline.py" \
  --config "$CONFIG" "$@"
