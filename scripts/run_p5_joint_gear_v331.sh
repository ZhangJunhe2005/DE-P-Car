#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="/home/zjh/miniconda3/envs/yopo/bin/python"
TRAINER="$PROJECT_ROOT/tools/train_dep_car_gear_selector_v331.py"
AUDITOR="$PROJECT_ROOT/tools/audit_p5_joint_gear_v331.py"
MODEL_ROOT="$PROJECT_ROOT/models/dep_car/p5_joint_gear_v331"
LOG_ROOT="$PROJECT_ROOT/logs/p5_joint_gear_v331"
REPORT_ROOT="$PROJECT_ROOT/reports"
FORMAL_OUTPUT="$MODEL_ROOT/fusion_unilateral_safe_bank_correction.pth"
PILOT_OUTPUT="$MODEL_ROOT/pilot/fusion_unilateral_safe_bank_correction.pth"

stage=""; workers=8; maximum_samples=1024; maximum_steps=16; resume=""
usage() {
  echo "Usage: bash scripts/run_p5_joint_gear_v331.sh --stage STAGE [options]"
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
case "$stage" in dry-run|pilot|pilot-acceptance|train|acceptance) ;; *) usage >&2; exit 2 ;; esac
[[ "$workers" =~ ^[1-9][0-9]*$ ]] || { echo "--workers must be positive" >&2; exit 2; }
mkdir -p "$MODEL_ROOT/pilot" "$LOG_ROOT" "$REPORT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/dep_car/src${PYTHONPATH:+:$PYTHONPATH}"

if [[ "$stage" == "dry-run" ]]; then
  "$PYTHON" "$TRAINER" --output "$FORMAL_OUTPUT" --workers "$workers" --dry-run \
    | tee "$REPORT_ROOT/p5_joint_gear_v331_dry_run.json"
  exit "${PIPESTATUS[0]}"
fi
if [[ "$stage" == "pilot" ]]; then
  "$PYTHON" "$TRAINER" --output "$PILOT_OUTPUT" --epochs 4 --workers "$workers" \
    --max-samples "$maximum_samples" --max-steps "$maximum_steps" \
    2>&1 | tee "$LOG_ROOT/unilateral_safe_bank_pilot.log"
  exit "${PIPESTATUS[0]}"
fi
if [[ "$stage" == "pilot-acceptance" ]]; then
  "$PYTHON" "$AUDITOR" --checkpoint "${PILOT_OUTPUT%.pth}.best.pth" \
    --maximum-samples "$maximum_samples" --batch-size 64 --workers "$workers" --device cuda \
    2>&1 | tee "$LOG_ROOT/unilateral_safe_bank_pilot_acceptance.log"
  exit "${PIPESTATUS[0]}"
fi
if [[ "$stage" == "acceptance" ]]; then
  "$PYTHON" "$AUDITOR" --checkpoint "${FORMAL_OUTPUT%.pth}.best.pth" \
    --batch-size 64 --workers "$workers" --device cuda \
    2>&1 | tee "$LOG_ROOT/unilateral_safe_bank_acceptance.log"
  exit "${PIPESTATUS[0]}"
fi
extra=(); [[ -z "$resume" ]] || extra+=(--resume "$FORMAL_OUTPUT")
"$PYTHON" "$TRAINER" --output "$FORMAL_OUTPUT" --workers "$workers" "${extra[@]}" \
  2>&1 | tee "$LOG_ROOT/unilateral_safe_bank_training.log"
exit "${PIPESTATUS[0]}"
