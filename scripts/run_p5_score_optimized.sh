#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
YOPO_PYTHON="/home/zjh/miniconda3/envs/yopo/bin/python"
TRAINER="$PROJECT_ROOT/tools/train_dep_car_score_optimized.py"
CANDIDATE_ROOT="$PROJECT_ROOT/models/dep_car/p5_v2"
ARTIFACT_ROOT="$PROJECT_ROOT/models/dep_car/p5_score_v1"
LOG_ROOT="$PROJECT_ROOT/logs/p5_score_v1"
REPORT_ROOT="$PROJECT_ROOT/reports"

stage=""
modality=""
source_checkpoint=""
resume_checkpoint=""
output_checkpoint=""
maximum_samples=512
maximum_steps=16

usage() {
  echo "Usage:"
  echo "  bash scripts/run_p5_score_optimized.sh --stage dry-run --modality all"
  echo "  bash scripts/run_p5_score_optimized.sh --stage pilot --modality all [--maximum-samples 512 --maximum-steps 16]"
  echo "  bash scripts/run_p5_score_optimized.sh --stage score_calibration --modality MODALITY [--source PATH | --resume PATH --output PATH]"
  echo "  MODALITY: depth_only | lidar_only | fusion"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage)
      stage="${2:?missing --stage value}"
      shift 2
      ;;
    --modality)
      modality="${2:?missing --modality value}"
      shift 2
      ;;
    --source)
      source_checkpoint="${2:?missing --source path}"
      shift 2
      ;;
    --resume)
      resume_checkpoint="${2:?missing --resume path}"
      shift 2
      ;;
    --output)
      output_checkpoint="${2:?missing --output path}"
      shift 2
      ;;
    --maximum-samples)
      maximum_samples="${2:?missing --maximum-samples value}"
      shift 2
      ;;
    --maximum-steps)
      maximum_steps="${2:?missing --maximum-steps value}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "$stage" != "dry-run" && "$stage" != "pilot" && "$stage" != "score_calibration" ]]; then
  echo "Invalid or missing --stage" >&2
  usage >&2
  exit 2
fi
if [[ "$modality" != "depth_only" && "$modality" != "lidar_only" && "$modality" != "fusion" && !( ( "$stage" == "dry-run" || "$stage" == "pilot" ) && "$modality" == "all" ) ]]; then
  echo "Invalid or missing --modality" >&2
  usage >&2
  exit 2
fi
if [[ ! "$maximum_samples" =~ ^[1-9][0-9]*$ || "$maximum_samples" -gt 512 ]]; then
  echo "--maximum-samples must be an integer in [1,512]" >&2
  exit 2
fi
if [[ ! "$maximum_steps" =~ ^[1-9][0-9]*$ || "$maximum_steps" -gt 64 ]]; then
  echo "--maximum-steps must be an integer in [1,64]" >&2
  exit 2
fi
if [[ -n "$source_checkpoint" && -n "$resume_checkpoint" ]]; then
  echo "--source and --resume are mutually exclusive" >&2
  exit 2
fi
if [[ "$stage" != "score_calibration" && ( -n "$source_checkpoint" || -n "$resume_checkpoint" || -n "$output_checkpoint" ) ]]; then
  echo "$stage uses canonical isolated paths; do not pass --source, --resume or --output" >&2
  exit 2
fi

mkdir -p "$ARTIFACT_ROOT" "$LOG_ROOT" "$REPORT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/dep_car/src${PYTHONPATH:+:$PYTHONPATH}"

require_cuda() {
  "$YOPO_PYTHON" -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"; print("CUDA", torch.cuda.get_device_name(0))'
}

candidate_path() {
  echo "$CANDIDATE_ROOT/$1_candidate_capacity.best.pth"
}

score_path() {
  echo "$ARTIFACT_ROOT/$1_score_calibration.pth"
}

dry_run_one() {
  local selected="$1"
  local report="$REPORT_ROOT/p5_score_optimized_${selected}_dry_run.json"
  local temporary="${report}.tmp"
  "$YOPO_PYTHON" "$TRAINER" \
    --modality "$selected" \
    --init "$(candidate_path "$selected")" \
    --output "$(score_path "$selected")" \
    --dry-run >"$temporary"
  mv "$temporary" "$report"
  "$YOPO_PYTHON" -c 'import json,sys; p=json.load(open(sys.argv[1])); assert p["status"]=="DRY_RUN_READY" and p["formal_training_authorized"] is True and p["optimized_score_training"] is True; print(json.dumps({"modality":p["modality"],"status":p["status"],"formal_training_authorized":p["formal_training_authorized"],"batch_size":p["batch_size"],"device":p["device"]},indent=2))' "$report"
}

pilot_one() {
  local selected="$1"
  local pilot_root="$ARTIFACT_ROOT/pilot"
  local output="$pilot_root/${selected}_score_calibration.pth"
  mkdir -p "$pilot_root"
  "$YOPO_PYTHON" "$TRAINER" \
    --modality "$selected" \
    --init "$(candidate_path "$selected")" \
    --output "$output" \
    --epochs 1 \
    --max-samples "$maximum_samples" \
    --max-steps "$maximum_steps" 2>&1 | tee "$LOG_ROOT/${selected}_score_pilot.log"
  return "${PIPESTATUS[0]}"
}

require_cuda
if [[ "$stage" == "dry-run" ]]; then
  if [[ "$modality" == "all" ]]; then
    for selected in depth_only lidar_only fusion; do
      dry_run_one "$selected"
    done
  else
    dry_run_one "$modality"
  fi
  exit 0
fi

if [[ "$stage" == "pilot" ]]; then
  if [[ "$modality" == "all" ]]; then
    for selected in depth_only lidar_only fusion; do
      pilot_one "$selected"
    done
  else
    pilot_one "$modality"
  fi
  exit 0
fi

selected_output="${output_checkpoint:-$(score_path "$modality")}"
command=(
  "$YOPO_PYTHON" "$TRAINER"
  --modality "$modality"
  --output "$selected_output"
)
if [[ -n "$resume_checkpoint" ]]; then
  if [[ "$resume_checkpoint" == "$selected_output" ]]; then
    echo "Resume source and output must differ" >&2
    exit 2
  fi
  command+=(--resume "$resume_checkpoint")
else
  selected_source="${source_checkpoint:-$(candidate_path "$modality")}"
  if [[ "$selected_source" == "$selected_output" ]]; then
    echo "Score source and output must differ" >&2
    exit 2
  fi
  command+=(--init "$selected_source")
fi
"${command[@]}" 2>&1 | tee "$LOG_ROOT/${modality}_score_calibration.log"
exit "${PIPESTATUS[0]}"
