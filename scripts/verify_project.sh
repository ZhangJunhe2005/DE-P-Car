#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT/dep_car/src${PYTHONPATH:+:$PYTHONPATH}"
/usr/bin/python3 -m compileall -q "$ROOT/dep_car/src" "$ROOT/ros" "$ROOT/tools"
/home/zjh/miniconda3/envs/yopo/bin/python -m pytest -q "$ROOT/tests"
bash "$ROOT/third_party/DE-P/DE-P/scripts/verify_route_a_v4_9_1_checkout.sh"
