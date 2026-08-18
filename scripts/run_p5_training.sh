#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
YOPO_PYTHON="/home/zjh/miniconda3/envs/yopo/bin/python"
TRAINER="$PROJECT_ROOT/tools/train_dep_car.py"
ACCEPTOR="$PROJECT_ROOT/tools/accept_p5_candidate.py"
INITIALIZATION="$PROJECT_ROOT/models/dep_car/dep_car_net_v1_depth_v483_init.pth"
ARTIFACT_ROOT="$PROJECT_ROOT/models/dep_car/p5_v2"
# P4 acceptance consumes these canonical dry-run paths.  Reissuing them after
# an implementation change deliberately replaces the stale authorization
# reports; learned v2 artifacts remain isolated under models/logs p5_v2.
REPORT_ROOT="$PROJECT_ROOT/reports"
LOG_ROOT="$PROJECT_ROOT/logs/p5_v2"

stage=""
modality=""
source_checkpoint=""
resume_checkpoint=""
output_checkpoint=""
workers=8
pilot_samples=512
pilot_steps=64

usage() {
  echo "Usage:"
  echo "  bash scripts/run_p5_training.sh --stage dry-run --modality all"
  echo "  bash scripts/run_p5_training.sh --stage candidate_pilot --modality all [--maximum-samples 512 --maximum-steps 64]"
  echo "  bash scripts/run_p5_training.sh --stage candidate_capacity --modality MODALITY [--resume PATH --output NEW_PATH]"
  echo "  bash scripts/run_p5_training.sh --stage accept_candidate --modality MODALITY [--source PATH]"
  echo "  bash scripts/run_p5_training.sh --stage score_calibration --modality MODALITY [--source PATH | --resume PATH --output NEW_PATH]"
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
      source_checkpoint="${2:?missing checkpoint path}"
      shift 2
      ;;
    --resume)
      resume_checkpoint="${2:?missing checkpoint path}"
      shift 2
      ;;
    --output)
      output_checkpoint="${2:?missing output path}"
      shift 2
      ;;
    --workers)
      workers="${2:?missing --workers value}"
      shift 2
      ;;
    --maximum-samples)
      pilot_samples="${2:?missing --maximum-samples value}"
      shift 2
      ;;
    --maximum-steps)
      pilot_steps="${2:?missing --maximum-steps value}"
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

if [[ ! "$workers" =~ ^[1-9][0-9]*$ ]]; then
  echo "--workers must be a positive integer" >&2
  exit 2
fi
if [[ "$stage" != "dry-run" && "$stage" != "candidate_pilot" && "$stage" != "candidate_capacity" && "$stage" != "accept_candidate" && "$stage" != "score_calibration" ]]; then
  echo "Invalid or missing --stage" >&2
  usage >&2
  exit 2
fi
if [[ "$modality" != "depth_only" && "$modality" != "lidar_only" && "$modality" != "fusion" && !( ( "$stage" == "dry-run" || "$stage" == "candidate_pilot" ) && "$modality" == "all" ) ]]; then
  echo "Invalid or missing --modality" >&2
  usage >&2
  exit 2
fi
if [[ ! "$pilot_samples" =~ ^[1-9][0-9]*$ || "$pilot_samples" -gt 512 ]]; then
  echo "--maximum-samples must be an integer in [1,512]" >&2
  exit 2
fi
if [[ ! "$pilot_steps" =~ ^[1-9][0-9]*$ || "$pilot_steps" -gt 64 ]]; then
  echo "--maximum-steps must be an integer in [1,64]" >&2
  exit 2
fi
if [[ -n "$source_checkpoint" && -n "$resume_checkpoint" ]]; then
  echo "--source and --resume are mutually exclusive" >&2
  exit 2
fi

mkdir -p "$ARTIFACT_ROOT" "$REPORT_ROOT" "$LOG_ROOT"
export PYTHONPATH="$PROJECT_ROOT/dep_car/src${PYTHONPATH:+:$PYTHONPATH}"

require_cuda() {
  "$YOPO_PYTHON" -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"; print("CUDA", torch.cuda.get_device_name(0))'
}

candidate_path() {
  echo "$ARTIFACT_ROOT/$1_candidate_capacity.pth"
}

candidate_best_path() {
  echo "$ARTIFACT_ROOT/$1_candidate_capacity.best.pth"
}

score_path() {
  echo "$ARTIFACT_ROOT/$1_score_calibration.pth"
}

run_dry_one() {
  local selected="$1"
  local report="$REPORT_ROOT/p5_${selected}_candidate_dry_run.json"
  local temporary="${report}.tmp"
  "$YOPO_PYTHON" "$TRAINER" \
    --stage candidate_capacity \
    --modality "$selected" \
    --init "$INITIALIZATION" \
    --output "$(candidate_path "$selected")" \
    --workers "$workers" \
    --torch-threads 8 \
    --device cuda \
    --dry-run >"$temporary"
  mv "$temporary" "$report"
  "$YOPO_PYTHON" -c 'import json,sys; p=json.load(open(sys.argv[1])); assert p["status"]=="DRY_RUN_READY" and p["formal_training_authorized"] is True; print(json.dumps({"modality":p["modality"],"status":p["status"],"formal_training_authorized":p["formal_training_authorized"],"samples":p["samples"],"device":p["device"]},indent=2))' "$report"
}

if [[ "$stage" == "dry-run" ]]; then
  if [[ -n "$source_checkpoint" || -n "$resume_checkpoint" || -n "$output_checkpoint" ]]; then
    echo "dry-run uses the frozen initialization and canonical outputs; do not pass --source, --resume, or --output" >&2
    exit 2
  fi
  require_cuda
  if [[ "$modality" == "all" ]]; then
    for selected in depth_only lidar_only fusion; do
      run_dry_one "$selected"
    done
  else
    run_dry_one "$modality"
  fi
  exit 0
fi

run_pilot_one() {
  local selected="$1"
  local pilot_root="$ARTIFACT_ROOT/pilot"
  local pilot_output="$pilot_root/${selected}_candidate_capacity.pth"
  mkdir -p "$pilot_root"
  "$YOPO_PYTHON" "$TRAINER" \
    --stage candidate_capacity \
    --modality "$selected" \
    --init "$INITIALIZATION" \
    --output "$pilot_output" \
    --epochs 1 \
    --max-samples "$pilot_samples" \
    --max-steps "$pilot_steps" \
    --workers "$workers" \
    --torch-threads 8 \
    --device cuda \
    --amp 2>&1 | tee "$LOG_ROOT/${selected}_candidate_pilot.log"
  return "${PIPESTATUS[0]}"
}

if [[ "$stage" == "candidate_pilot" ]]; then
  if [[ -n "$source_checkpoint" || -n "$resume_checkpoint" || -n "$output_checkpoint" ]]; then
    echo "candidate_pilot uses the frozen initialization and isolated pilot outputs" >&2
    exit 2
  fi
  require_cuda
  if [[ "$modality" == "all" ]]; then
    for selected in depth_only lidar_only fusion; do
      run_pilot_one "$selected"
    done
  else
    run_pilot_one "$modality"
  fi
  exit 0
fi

if [[ "$stage" == "accept_candidate" ]]; then
  if [[ -n "$resume_checkpoint" || -n "$output_checkpoint" ]]; then
    echo "accept_candidate accepts only the optional --source checkpoint" >&2
    exit 2
  fi
  selected_source="${source_checkpoint:-$(candidate_best_path "$modality")}"
  "$YOPO_PYTHON" "$ACCEPTOR" "$selected_source" | tee "$LOG_ROOT/${modality}_candidate_acceptance.log"
  exit "${PIPESTATUS[0]}"
fi

if [[ "$stage" == "candidate_capacity" ]]; then
  if [[ -n "$source_checkpoint" ]]; then
    echo "candidate_capacity uses --resume for a prior candidate checkpoint, not --source" >&2
    exit 2
  fi
  preflight_output="${output_checkpoint:-$(candidate_path "$modality")}"
  if [[ -n "$resume_checkpoint" && "$resume_checkpoint" == "$preflight_output" ]]; then
    echo "Resume source and output must differ; pass --output with a new path" >&2
    exit 2
  fi
else
  preflight_output="${output_checkpoint:-$(score_path "$modality")}"
  if [[ -n "$resume_checkpoint" ]]; then
    if [[ "$resume_checkpoint" == "$preflight_output" ]]; then
      echo "Resume source and output must differ; pass --output with a new path" >&2
      exit 2
    fi
  else
    preflight_source="${source_checkpoint:-$(candidate_best_path "$modality")}"
    if [[ "$preflight_source" == "$preflight_output" ]]; then
      echo "Score source and output must differ" >&2
      exit 2
    fi
  fi
fi

require_cuda
if [[ "$stage" == "candidate_capacity" ]]; then
  selected_output="${output_checkpoint:-$(candidate_path "$modality")}"
  command=(
    "$YOPO_PYTHON" "$TRAINER"
    --stage candidate_capacity
    --modality "$modality"
    --output "$selected_output"
    --workers "$workers"
    --torch-threads 8
    --device cuda
    --amp
  )
  if [[ -n "$resume_checkpoint" ]]; then
    command+=(--resume "$resume_checkpoint")
  else
    command+=(--init "$INITIALIZATION")
  fi
  "${command[@]}" 2>&1 | tee "$LOG_ROOT/${modality}_candidate_capacity.log"
  exit "${PIPESTATUS[0]}"
fi

selected_output="${output_checkpoint:-$(score_path "$modality")}"
command=(
  "$YOPO_PYTHON" "$TRAINER"
  --stage score_calibration
  --modality "$modality"
  --output "$selected_output"
  --workers "$workers"
  --torch-threads 8
  --device cuda
  --amp
)
if [[ -n "$resume_checkpoint" ]]; then
  command+=(--resume "$resume_checkpoint")
else
  selected_source="${source_checkpoint:-$(candidate_best_path "$modality")}"
  command+=(--init "$selected_source")
fi
"${command[@]}" 2>&1 | tee "$LOG_ROOT/${modality}_score_calibration.log"
exit "${PIPESTATUS[0]}"
