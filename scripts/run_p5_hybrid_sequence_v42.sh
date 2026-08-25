#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="/home/zjh/miniconda3/envs/yopo/bin/python"
AUDITOR="$PROJECT_ROOT/tools/audit_p5_hybrid_sequence_v42_execution.py"
LOG_ROOT="$PROJECT_ROOT/logs/p5_hybrid_sequence_v42"
REPORT_ROOT="$PROJECT_ROOT/reports"

stage=""; workers=8
usage() {
  echo "Usage: bash scripts/run_p5_hybrid_sequence_v42.sh --stage dry-run|acceptance [--workers N]"
}
while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage) stage="${2:?missing stage}"; shift 2 ;;
    --workers) workers="${2:?missing workers}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done
case "$stage" in dry-run|acceptance) ;; *) usage >&2; exit 2 ;; esac
[[ "$workers" =~ ^[1-9][0-9]*$ ]] || {
  echo "--workers must be positive" >&2; exit 2;
}
mkdir -p "$LOG_ROOT" "$REPORT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/dep_car/src:$PROJECT_ROOT/tools${PYTHONPATH:+:$PYTHONPATH}"

if [[ "$stage" == "dry-run" ]]; then
  "$PYTHON" "$AUDITOR" --workers "$workers" --device cuda --dry-run \
    | tee "$REPORT_ROOT/p5_hybrid_sequence_v42_execution_dry_run.json"
else
  "$PYTHON" "$AUDITOR" --workers "$workers" --device cuda \
    2>&1 | tee "$LOG_ROOT/execution_acceptance.log"
fi
