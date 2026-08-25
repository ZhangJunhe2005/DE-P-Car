#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="/home/zjh/miniconda3/envs/yopo/bin/python"
INDEX_BUILDER="$PROJECT_ROOT/tools/build_p3_v5_joint_gear_index.py"
INDEX_AUDITOR="$PROJECT_ROOT/tools/audit_p3_v5_joint_gear_index.py"
TRAINER="$PROJECT_ROOT/tools/train_dep_car_joint_gear_v3.py"
ACCEPTANCE="$PROJECT_ROOT/tools/audit_p5_joint_gear_v3.py"
MODEL_ROOT="$PROJECT_ROOT/models/dep_car/p5_joint_gear_v3"
LOG_ROOT="$PROJECT_ROOT/logs/p5_joint_gear_v3"
REPORT_ROOT="$PROJECT_ROOT/reports"
SEQUENCE_INDEX="$PROJECT_ROOT/data/p3_v5/joint_gear_sequence_index.json"

stage=""
workers=8
maximum_samples=512
maximum_steps=16
resume=""

usage() {
  echo "Usage: bash scripts/run_p5_joint_gear_v3.sh --stage STAGE [options]"
  echo "Stages:"
  echo "  index-dry-run | index | index-audit | gpu-smoke"
  echo "  candidate-dry-run | candidate-pilot | candidate | candidate-acceptance"
  echo "  score-dry-run | score-pilot | score | score-acceptance"
  echo "  sequence-dry-run | sequence-pilot | sequence | sequence-acceptance"
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
  index-dry-run|index|index-audit|gpu-smoke|candidate-dry-run|candidate-pilot|candidate|candidate-acceptance|score-dry-run|score-pilot|score|score-acceptance|sequence-dry-run|sequence-pilot|sequence|sequence-acceptance) ;;
  *) usage >&2; exit 2 ;;
esac
if [[ ! "$workers" =~ ^[1-9][0-9]*$ ]]; then
  echo "--workers must be positive" >&2; exit 2
fi

mkdir -p "$MODEL_ROOT" "$MODEL_ROOT/pilot" "$LOG_ROOT" "$REPORT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/dep_car/src${PYTHONPATH:+:$PYTHONPATH}"

if [[ "$stage" == "gpu-smoke" ]]; then
  "$PYTHON" "$PROJECT_ROOT/tools/smoke_p5_joint_gear_v3.py" \
    --batch-size 32 --steps 4 \
    | tee "$REPORT_ROOT/p5_joint_gear_v3_cuda_smoke.json"
  exit "${PIPESTATUS[0]}"
fi

if [[ "$stage" == "index-dry-run" ]]; then
  "$PYTHON" "$INDEX_BUILDER" \
    --index "$PROJECT_ROOT/data/p3_v4/bundle_v1/training_index.json" \
    --output "$SEQUENCE_INDEX" --workers "$workers" \
    --maximum-samples 32 --dry-run
  exit 0
fi
if [[ "$stage" == "index" ]]; then
  "$PYTHON" "$INDEX_BUILDER" \
    --index "$PROJECT_ROOT/data/p3_v4/bundle_v1/training_index.json" \
    --output "$SEQUENCE_INDEX" --workers "$workers" \
    2>&1 | tee "$LOG_ROOT/joint_gear_index.log"
  exit "${PIPESTATUS[0]}"
fi
if [[ "$stage" == "index-audit" ]]; then
  "$PYTHON" "$INDEX_AUDITOR" \
    --index "$SEQUENCE_INDEX" \
    --bundle "$PROJECT_ROOT/data/p3_v4/bundle_v1/bundle_authority.json" \
    --output "$PROJECT_ROOT/data/p3_v5/joint_gear_sequence_authority.json" \
    | tee "$REPORT_ROOT/p3_v5_joint_gear_sequence_audit.json"
  exit "${PIPESTATUS[0]}"
fi

case "$stage" in
  candidate-*) train_stage="bidirectional_candidate_capacity"; artifact="candidate" ;;
  candidate) train_stage="bidirectional_candidate_capacity"; artifact="candidate" ;;
  score-*) train_stage="joint_gear_score_calibration"; artifact="score" ;;
  score) train_stage="joint_gear_score_calibration"; artifact="score" ;;
  sequence-*) train_stage="sequence_recovery_finetune"; artifact="sequence" ;;
  sequence) train_stage="sequence_recovery_finetune"; artifact="sequence" ;;
esac

case "$artifact" in
  candidate)
    source="$PROJECT_ROOT/models/dep_car/p5_route_v2/fusion_score_calibration.best.pth"
    output="$MODEL_ROOT/fusion_bidirectional_candidate_capacity.pth"
    ;;
  score)
    source="$MODEL_ROOT/fusion_bidirectional_candidate_capacity.best.pth"
    output="$MODEL_ROOT/fusion_joint_gear_score_calibration.pth"
    ;;
  sequence)
    source="$MODEL_ROOT/fusion_joint_gear_score_calibration.best.pth"
    output="$MODEL_ROOT/fusion_sequence_recovery_finetune.pth"
    ;;
esac

run_train() {
  "$PYTHON" "$TRAINER" \
    --stage "$train_stage" --source "$source" --output "$output" \
    --workers "$workers" "${@:1}"
}

if [[ "$stage" == *-dry-run ]]; then
  run_train --dry-run | tee "$REPORT_ROOT/p5_joint_gear_v3_${artifact}_dry_run.json"
  exit "${PIPESTATUS[0]}"
fi
if [[ "$stage" == *-pilot ]]; then
  pilot_output="$MODEL_ROOT/pilot/fusion_${artifact}_pilot.pth"
  output="$pilot_output"
  run_train --epochs 1 --max-samples "$maximum_samples" \
    --max-steps "$maximum_steps" 2>&1 | tee "$LOG_ROOT/${artifact}_pilot.log"
  exit "${PIPESTATUS[0]}"
fi
if [[ "$stage" == *-acceptance ]]; then
  "$PYTHON" "$ACCEPTANCE" \
    --stage "$train_stage" --checkpoint "${output%.pth}.best.pth" \
    --batch-size 64 --workers "$workers" --device cuda \
    2>&1 | tee "$LOG_ROOT/${artifact}_acceptance.log"
  exit "${PIPESTATUS[0]}"
fi

extra=()
if [[ -n "$resume" ]]; then
  extra+=(--resume "$output")
fi
run_train "${extra[@]}" 2>&1 | tee "$LOG_ROOT/${artifact}_training.log"
exit "${PIPESTATUS[0]}"
