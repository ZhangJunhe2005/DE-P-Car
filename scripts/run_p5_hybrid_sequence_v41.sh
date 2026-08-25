#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="/home/zjh/miniconda3/envs/yopo/bin/python"
TRAINER="$PROJECT_ROOT/tools/train_dep_car_hybrid_sequence_v41.py"
AUDITOR="$PROJECT_ROOT/tools/audit_p5_hybrid_sequence_v41.py"
MODEL_ROOT="$PROJECT_ROOT/models/dep_car/p5_hybrid_sequence_v41"
LOG_ROOT="$PROJECT_ROOT/logs/p5_hybrid_sequence_v41"
REPORT_ROOT="$PROJECT_ROOT/reports"
SOURCE="$PROJECT_ROOT/models/dep_car/p5_hybrid_sequence_v4/fusion_hybrid_sequence_capacity.best.pth"
OUTPUT="$MODEL_ROOT/fusion_hierarchical_score.pth"
PILOT="$MODEL_ROOT/pilot/fusion_hierarchical_score.pth"

stage=""; workers=8; maximum_samples=512; maximum_steps=8; resume=""
usage() {
  echo "Usage: bash scripts/run_p5_hybrid_sequence_v41.sh --stage STAGE [options]"
  echo "Stages: dry-run | pilot | pilot-acceptance | train | acceptance"
  echo "Options: --workers N --maximum-samples N --maximum-steps N --resume"
}
while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage) stage="${2:?missing stage}"; shift 2 ;;
    --workers) workers="${2:?missing workers}"; shift 2 ;;
    --maximum-samples) maximum_samples="${2:?missing samples}"; shift 2 ;;
    --maximum-steps) maximum_steps="${2:?missing steps}"; shift 2 ;;
    --resume) resume="true"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done
case "$stage" in
  dry-run|pilot|pilot-acceptance|train|acceptance) ;;
  *) usage >&2; exit 2 ;;
esac
[[ "$workers" =~ ^[1-9][0-9]*$ ]] || {
  echo "--workers must be positive" >&2; exit 2;
}
mkdir -p "$MODEL_ROOT/pilot" "$LOG_ROOT" "$REPORT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/dep_car/src:$PROJECT_ROOT/tools${PYTHONPATH:+:$PYTHONPATH}"

run_train() {
  local output="$1" log="$2"
  shift 2
  local extra=()
  [[ -z "$resume" ]] || extra+=(--resume "$output")
  "$PYTHON" "$TRAINER" --source "$SOURCE" --output "$output" \
    --workers "$workers" "${extra[@]}" "$@" 2>&1 | tee "$LOG_ROOT/$log"
  return "${PIPESTATUS[0]}"
}

case "$stage" in
  dry-run)
    "$PYTHON" "$TRAINER" --source "$SOURCE" --output "$OUTPUT" \
      --workers "$workers" --dry-run \
      | tee "$REPORT_ROOT/p5_hybrid_sequence_v41_dry_run.json"
    ;;
  pilot)
    run_train "$PILOT" pilot.log --epochs 3 \
      --max-samples "$maximum_samples" --max-steps "$maximum_steps"
    ;;
  pilot-acceptance)
    "$PYTHON" "$AUDITOR" --checkpoint "${PILOT%.pth}.best.pth" \
      --maximum-samples "$maximum_samples" --workers "$workers" --device cuda \
      2>&1 | tee "$LOG_ROOT/pilot_acceptance.log"
    ;;
  train)
    run_train "$OUTPUT" training.log
    ;;
  acceptance)
    "$PYTHON" "$AUDITOR" --checkpoint "${OUTPUT%.pth}.best.pth" \
      --workers "$workers" --device cuda \
      2>&1 | tee "$LOG_ROOT/acceptance.log"
    ;;
esac
