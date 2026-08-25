#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="/home/zjh/miniconda3/envs/yopo/bin/python"
TRAINER="$PROJECT_ROOT/tools/train_dep_car_hybrid_sequence_v4.py"
AUDITOR="$PROJECT_ROOT/tools/audit_p5_hybrid_sequence_v4.py"
MODEL_ROOT="$PROJECT_ROOT/models/dep_car/p5_hybrid_sequence_v4"
LOG_ROOT="$PROJECT_ROOT/logs/p5_hybrid_sequence_v4"
REPORT_ROOT="$PROJECT_ROOT/reports"
V3_SOURCE="$PROJECT_ROOT/models/dep_car/p5_joint_gear_v31/fusion_sequence_correction.best.pth"
CAPACITY="$MODEL_ROOT/fusion_hybrid_sequence_capacity.pth"
SCORE="$MODEL_ROOT/fusion_hybrid_sequence_score.pth"
CLOSED="$MODEL_ROOT/fusion_closed_loop_sequence.pth"
PILOT="$MODEL_ROOT/pilot/fusion_hybrid_sequence_capacity.pth"
SCORE_PILOT="$MODEL_ROOT/pilot/fusion_hybrid_sequence_score.pth"

stage=""; workers=8; maximum_samples=512; maximum_steps=8; resume=""
usage() {
  echo "Usage: bash scripts/run_p5_hybrid_sequence_v4.sh --stage STAGE [options]"
  echo "Stages: data-audit | capacity-dry-run | capacity-pilot | capacity-pilot-acceptance"
  echo "        capacity-train | capacity-acceptance | score-dry-run | score-pilot"
  echo "        score-pilot-acceptance | score-train | score-acceptance"
  echo "        closed-loop-dry-run | closed-loop-train | closed-loop-acceptance"
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
  data-audit|capacity-dry-run|capacity-pilot|capacity-pilot-acceptance|capacity-train|capacity-acceptance|score-dry-run|score-pilot|score-pilot-acceptance|score-train|score-acceptance|closed-loop-dry-run|closed-loop-train|closed-loop-acceptance) ;;
  *) usage >&2; exit 2 ;;
esac
[[ "$workers" =~ ^[1-9][0-9]*$ ]] || { echo "--workers must be positive" >&2; exit 2; }
mkdir -p "$MODEL_ROOT/pilot" "$LOG_ROOT" "$REPORT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/dep_car/src:$PROJECT_ROOT/tools${PYTHONPATH:+:$PYTHONPATH}"

if [[ "$stage" == "data-audit" ]]; then
  "$PYTHON" "$PROJECT_ROOT/tools/audit_p3_v6_hybrid_sequence.py"
  exit $?
fi

run_train() {
  local train_stage="$1" source="$2" output="$3" log="$4"
  shift 4
  local extra=()
  [[ -z "$resume" ]] || extra+=(--resume "$output")
  "$PYTHON" "$TRAINER" --stage "$train_stage" --source "$source" \
    --output "$output" --workers "$workers" "${extra[@]}" "$@" \
    2>&1 | tee "$LOG_ROOT/$log"
  return "${PIPESTATUS[0]}"
}

case "$stage" in
  capacity-dry-run)
    "$PYTHON" "$TRAINER" --stage hybrid_sequence_capacity --source "$V3_SOURCE" \
      --output "$CAPACITY" --workers "$workers" --dry-run \
      | tee "$REPORT_ROOT/p5_hybrid_sequence_v4_capacity_dry_run.json"
    ;;
  capacity-pilot)
    run_train hybrid_sequence_capacity "$V3_SOURCE" "$PILOT" \
      capacity_pilot.log --epochs 3 --max-samples "$maximum_samples" \
      --max-steps "$maximum_steps"
    ;;
  capacity-pilot-acceptance)
    "$PYTHON" "$AUDITOR" --stage hybrid_sequence_capacity \
      --checkpoint "${PILOT%.pth}.best.pth" --maximum-samples "$maximum_samples" \
      --workers "$workers" --device cuda \
      2>&1 | tee "$LOG_ROOT/capacity_pilot_acceptance.log"
    ;;
  capacity-train)
    run_train hybrid_sequence_capacity "$V3_SOURCE" "$CAPACITY" capacity_training.log
    ;;
  capacity-acceptance)
    "$PYTHON" "$AUDITOR" --stage hybrid_sequence_capacity \
      --checkpoint "${CAPACITY%.pth}.best.pth" --workers "$workers" --device cuda \
      2>&1 | tee "$LOG_ROOT/capacity_acceptance.log"
    ;;
  score-dry-run)
    "$PYTHON" "$TRAINER" --stage hybrid_sequence_score \
      --source "${CAPACITY%.pth}.best.pth" --output "$SCORE" \
      --workers "$workers" --dry-run \
      | tee "$REPORT_ROOT/p5_hybrid_sequence_v4_score_dry_run.json"
    ;;
  score-pilot)
    run_train hybrid_sequence_score "${CAPACITY%.pth}.best.pth" "$SCORE_PILOT" \
      score_pilot.log --epochs 3 --max-samples "$maximum_samples" \
      --max-steps "$maximum_steps"
    ;;
  score-pilot-acceptance)
    "$PYTHON" "$AUDITOR" --stage hybrid_sequence_score \
      --checkpoint "${SCORE_PILOT%.pth}.best.pth" \
      --maximum-samples "$maximum_samples" --workers "$workers" --device cuda \
      2>&1 | tee "$LOG_ROOT/score_pilot_acceptance.log"
    ;;
  score-train)
    run_train hybrid_sequence_score "${CAPACITY%.pth}.best.pth" "$SCORE" score_training.log
    ;;
  score-acceptance)
    "$PYTHON" "$AUDITOR" --stage hybrid_sequence_score \
      --checkpoint "${SCORE%.pth}.best.pth" --workers "$workers" --device cuda \
      2>&1 | tee "$LOG_ROOT/score_acceptance.log"
    ;;
  closed-loop-dry-run)
    "$PYTHON" "$TRAINER" --stage closed_loop_sequence_finetune \
      --source "${SCORE%.pth}.best.pth" --output "$CLOSED" \
      --workers "$workers" --dry-run \
      | tee "$REPORT_ROOT/p5_hybrid_sequence_v4_closed_loop_dry_run.json"
    ;;
  closed-loop-train)
    run_train closed_loop_sequence_finetune "${SCORE%.pth}.best.pth" "$CLOSED" closed_loop_training.log
    ;;
  closed-loop-acceptance)
    "$PYTHON" "$AUDITOR" --stage closed_loop_sequence_finetune \
      --checkpoint "${CLOSED%.pth}.best.pth" --workers "$workers" --device cuda \
      2>&1 | tee "$LOG_ROOT/closed_loop_acceptance.log"
    ;;
esac
