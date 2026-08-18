#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="/home/zjh/miniconda3/envs/yopo/bin/python"
TRAINER="$PROJECT_ROOT/tools/train_dep_car_route_v2.py"
CAPACITY_AUDITOR="$PROJECT_ROOT/tools/audit_p6_corner_capacity.py"
MODEL_ROOT="$PROJECT_ROOT/models/dep_car/p5_route_v2"
LOG_ROOT="$PROJECT_ROOT/logs/p5_route_v2"
FORMAL_AUTHORITY="$PROJECT_ROOT/data/p3_v4/bundle_v1/bundle_authority.json"

stage=""
modality="fusion"
authority=""
maximum_samples=512
maximum_steps=16
resume=""

usage() {
  echo "Usage:"
  echo "  bash scripts/run_p5_route_v2.sh --stage dry-run"
  echo "  bash scripts/run_p5_route_v2.sh --stage pilot [--maximum-samples 512 --maximum-steps 16]"
  echo "  bash scripts/run_p5_route_v2.sh --stage candidate_capacity"
  echo "  bash scripts/run_p5_route_v2.sh --stage candidate_capacity --resume"
  echo "  bash scripts/run_p5_route_v2.sh --stage candidate_acceptance"
  echo "  bash scripts/run_p5_route_v2.sh --stage score-dry-run"
  echo "  bash scripts/run_p5_route_v2.sh --stage score-pilot [--maximum-samples 512 --maximum-steps 16]"
  echo "  bash scripts/run_p5_route_v2.sh --stage score_calibration"
  echo "  bash scripts/run_p5_route_v2.sh --stage score_calibration --resume"
  echo "Depth/LiDAR-only are retained only for bounded diagnostic pilot runs."
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage) stage="${2:?missing stage}"; shift 2 ;;
    --modality) modality="${2:?missing modality}"; shift 2 ;;
    --authority) authority="${2:?missing authority}"; shift 2 ;;
    --maximum-samples) maximum_samples="${2:?missing samples}"; shift 2 ;;
    --maximum-steps) maximum_steps="${2:?missing steps}"; shift 2 ;;
    --resume) resume="true"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ "$stage" != "dry-run" && "$stage" != "pilot" && "$stage" != "candidate_capacity" && "$stage" != "candidate_acceptance" && "$stage" != "score-dry-run" && "$stage" != "score-pilot" && "$stage" != "score_calibration" ]]; then
  usage >&2; exit 2
fi
if [[ "$modality" != "depth_only" && "$modality" != "lidar_only" && "$modality" != "fusion" ]]; then
  usage >&2; exit 2
fi
if [[ "$modality" != "fusion" && "$stage" != "pilot" ]]; then
  echo "Only fusion is authorized for dry-run and formal V2 training" >&2; exit 2
fi

mkdir -p "$MODEL_ROOT" "$LOG_ROOT"
export PYTHONPATH="$PROJECT_ROOT/dep_car/src${PYTHONPATH:+:$PYTHONPATH}"

candidate_v1() { echo "$PROJECT_ROOT/models/dep_car/p5_v2/$1_candidate_capacity.best.pth"; }
candidate_v2() { echo "$MODEL_ROOT/$1_candidate_capacity.best.pth"; }
output_for() { echo "$MODEL_ROOT/$1_$2.pth"; }

run_one() {
  local selected="$1"
  local selected_stage="$2"
  local selected_authority="$3"
  local source
  local selected_output="${OUTPUT_OVERRIDE:-$(output_for "$selected" "$selected_stage")}"
  if [[ "$selected_stage" == "candidate_capacity" ]]; then
    source="$(candidate_v1 "$selected")"
  else
    source="$(candidate_v2 "$selected")"
  fi
  "$PYTHON" "$TRAINER" \
    --stage "$selected_stage" --modality "$selected" \
    --source "$source" --authority "$selected_authority" \
    --output "$selected_output" "${@:4}"
}

if [[ "$stage" == "dry-run" ]]; then
  selected_authority="${authority:-$FORMAL_AUTHORITY}"
  report="$PROJECT_ROOT/reports/p5_route_v2_fusion_candidate_dry_run.json"
  run_one fusion candidate_capacity "$selected_authority" --dry-run | tee "$report"
  "$PYTHON" -c 'import json,sys; p=json.load(open(sys.argv[1])); assert p["status"]=="DRY_RUN_READY" and p["formal_training_authorized"] is True and p["formal_modalities"]==["fusion"] and p["formal_training_authority_gate"]["passed"] is True; print("fusion V2 formal dry-run PASS")' "$report"
  exit 0
fi

if [[ "$stage" == "pilot" ]]; then
  selected_authority="${authority:-$FORMAL_AUTHORITY}"
  mkdir -p "$MODEL_ROOT/pilot"
  OUTPUT_OVERRIDE="$MODEL_ROOT/pilot/${modality}_candidate_capacity_p3v4.pth" \
  run_one "$modality" candidate_capacity "$selected_authority" \
    --epochs 1 --max-samples "$maximum_samples" --max-steps "$maximum_steps" \
    2>&1 | tee "$LOG_ROOT/${modality}_candidate_p3v4_pilot.log"
  exit 0
fi

if [[ "$stage" == "candidate_acceptance" ]]; then
  report="$PROJECT_ROOT/reports/p5_route_v2_fusion_candidate_acceptance.json"
  "$PYTHON" "$CAPACITY_AUDITOR" \
    --checkpoint "$(candidate_v2 fusion)" \
    --modality fusion --output "$report" \
    --maximum-samples 0 --batch-size 16 --workers 8 --device cuda
  "$PYTHON" -c 'import json,sys; p=json.load(open(sys.argv[1])); assert p["status"]=="PASS" and p["gate_passed"] is True and p["smoke_limited"] is False and p["test_split_accessed"] is False and p["population_frames_selected"]==p["population_frames_available"]; print("fusion V2 Candidate formal acceptance PASS")' "$report"
  exit 0
fi

if [[ "$stage" == "score-dry-run" ]]; then
  selected_authority="${authority:-$FORMAL_AUTHORITY}"
  report="$PROJECT_ROOT/reports/p5_route_v2_fusion_score_dry_run.json"
  run_one fusion score_calibration "$selected_authority" --dry-run | tee "$report"
  "$PYTHON" -c 'import json,sys; p=json.load(open(sys.argv[1])); g=p["source_acceptance_gate"]; assert p["status"]=="DRY_RUN_READY" and p["formal_training_authorized"] is True and p["stage"]=="score_calibration" and g["passed"] is True and g["test_split_accessed"] is False; print("fusion V2 Score formal dry-run PASS")' "$report"
  exit 0
fi

if [[ "$stage" == "score-pilot" ]]; then
  selected_authority="${authority:-$FORMAL_AUTHORITY}"
  mkdir -p "$MODEL_ROOT/pilot"
  OUTPUT_OVERRIDE="$MODEL_ROOT/pilot/fusion_score_calibration_p3v4.pth" \
  run_one fusion score_calibration "$selected_authority" \
    --epochs 1 --max-samples "$maximum_samples" --max-steps "$maximum_steps" \
    2>&1 | tee "$LOG_ROOT/fusion_score_p3v4_pilot.log"
  exit "${PIPESTATUS[0]}"
fi

selected_authority="${authority:-$FORMAL_AUTHORITY}"
extra=()
if [[ -n "$resume" ]]; then
  extra+=(--resume "$(output_for "$modality" "$stage")")
fi
run_one "$modality" "$stage" "$selected_authority" "${extra[@]}" \
  2>&1 | tee "$LOG_ROOT/${modality}_${stage}.log"
exit "${PIPESTATUS[0]}"
