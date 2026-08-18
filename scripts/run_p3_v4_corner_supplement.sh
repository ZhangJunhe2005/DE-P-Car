#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
pipeline_args=("$@")

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
  --config "$PROJECT_ROOT/dep_car/config/p3_v4_corner_supplement.yaml" \
  "${pipeline_args[@]}"
