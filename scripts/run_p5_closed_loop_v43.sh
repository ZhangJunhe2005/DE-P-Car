#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
YOPO_PYTHON="/home/zjh/miniconda3/envs/yopo/bin/python"
SYSTEM_PYTHON="/usr/bin/python3"
COLLECTION_CONFIG="$PROJECT_ROOT/dep_car/config/p5_closed_loop_v43_collection.yaml"
MANIFEST="$PROJECT_ROOT/data/p3_v7_v43/task_manifest.json"
RUN_ROOT="$PROJECT_ROOT/data/p3_v7_v43/run"
INDEX_ROOT="$PROJECT_ROOT/data/p3_v7_v43/index"
MODEL_ROOT="$PROJECT_ROOT/models/dep_car/p5_closed_loop_v43"
LOG_ROOT="$PROJECT_ROOT/logs/p5_closed_loop_v43"
REPORT_ROOT="$PROJECT_ROOT/reports"
SOURCE="$PROJECT_ROOT/models/dep_car/p5_hybrid_sequence_v41/fusion_hierarchical_score.best.pth"
OUTPUT="$MODEL_ROOT/fusion_closed_loop_sequence.pth"
PILOT="$MODEL_ROOT/pilot_contextual_exact/fusion_closed_loop_sequence.pth"

stage=""; workers=8; gazebo_workers=2; maximum_tasks=0
maximum_samples=512; maximum_steps=8; task_id=""; retry_failed=""
rerun_complete=""
fail_fast=""
rerun_recovered=""
usage() {
  echo "Usage: bash scripts/run_p5_closed_loop_v43.sh --stage STAGE [options]"
  echo "Stages: prepare | collect-dry-run | collect | recover-dry-run | recover | index | integrity-audit | authorize | dry-run | pilot | pilot-acceptance | train | acceptance"
  echo "Options: --workers N --gazebo-workers N --maximum-tasks N --task-id ID --retry-failed --rerun-complete --rerun-recovered --fail-fast --maximum-samples N --maximum-steps N"
}
while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage) stage="${2:?missing stage}"; shift 2 ;;
    --workers) workers="${2:?missing workers}"; shift 2 ;;
    --gazebo-workers) gazebo_workers="${2:?missing workers}"; shift 2 ;;
    --maximum-tasks) maximum_tasks="${2:?missing tasks}"; shift 2 ;;
    --task-id) task_id="${2:?missing task id}"; shift 2 ;;
    --retry-failed) retry_failed="true"; shift ;;
    --rerun-complete) rerun_complete="true"; shift ;;
    --rerun-recovered) rerun_recovered="true"; shift ;;
    --fail-fast) fail_fast="true"; shift ;;
    --maximum-samples) maximum_samples="${2:?missing samples}"; shift 2 ;;
    --maximum-steps) maximum_steps="${2:?missing steps}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done
case "$stage" in
  prepare|collect-dry-run|collect|recover-dry-run|recover|index|integrity-audit|authorize|dry-run|pilot|pilot-acceptance|train|acceptance) ;;
  *) usage >&2; exit 2 ;;
esac
for value in "$workers" "$gazebo_workers"; do
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || { echo "worker counts must be positive" >&2; exit 2; }
done
mkdir -p "$MODEL_ROOT/pilot_contextual_exact" "$LOG_ROOT" "$REPORT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/dep_car/src:$PROJECT_ROOT/tools${PYTHONPATH:+:$PYTHONPATH}"

setup_ros() {
  set +u
  source /opt/ros/noetic/setup.bash
  source "$PROJECT_ROOT/catkin_ws/devel/setup.bash"
  set -u
}

collection_args=()
[[ "$maximum_tasks" == "0" ]] || collection_args+=(--maximum-tasks "$maximum_tasks")
[[ -z "$task_id" ]] || collection_args+=(--task-id "$task_id")
[[ -z "$retry_failed" ]] || collection_args+=(--retry-failed)
[[ -z "$rerun_complete" ]] || collection_args+=(--rerun-complete)
[[ -z "$fail_fast" ]] || collection_args+=(--fail-fast)
recovery_args=()
[[ -z "$rerun_recovered" ]] || recovery_args+=(--rerun-recovered)

case "$stage" in
  prepare)
    "$SYSTEM_PYTHON" "$PROJECT_ROOT/tools/prepare_p5_closed_loop_v43_manifest.py" \
      --output "$MANIFEST" | tee "$REPORT_ROOT/p5_closed_loop_v43_prepare.json"
    ;;
  collect-dry-run)
    setup_ros
    "$SYSTEM_PYTHON" "$PROJECT_ROOT/ros/dep_car_dataset/scripts/run_pilot_collection.py" \
      --config "$COLLECTION_CONFIG" --manifest "$MANIFEST" --work-root "$RUN_ROOT" \
      --workers "$gazebo_workers" --startup-stagger 3.0 --dry-run "${collection_args[@]}"
    ;;
  collect)
    setup_ros
    "$SYSTEM_PYTHON" "$PROJECT_ROOT/ros/dep_car_dataset/scripts/run_pilot_collection.py" \
      --config "$COLLECTION_CONFIG" --manifest "$MANIFEST" --work-root "$RUN_ROOT" \
      --workers "$gazebo_workers" --startup-stagger 3.0 "${collection_args[@]}"
    ;;
  recover-dry-run)
    setup_ros
    "$SYSTEM_PYTHON" "$PROJECT_ROOT/tools/recover_p5_closed_loop_v43_collection.py" \
      --config "$COLLECTION_CONFIG" --manifest "$MANIFEST" --work-root "$RUN_ROOT" \
      --workers "$workers" --dry-run "${recovery_args[@]}"
    ;;
  recover)
    setup_ros
    "$SYSTEM_PYTHON" "$PROJECT_ROOT/tools/recover_p5_closed_loop_v43_collection.py" \
      --config "$COLLECTION_CONFIG" --manifest "$MANIFEST" --work-root "$RUN_ROOT" \
      --workers "$workers" "${recovery_args[@]}" \
      | tee "$REPORT_ROOT/p5_closed_loop_v43_collection_recovery.json"
    ;;
  index)
    "$YOPO_PYTHON" "$PROJECT_ROOT/tools/build_p3_v7_v43_closed_loop_index.py" \
      --samples "$RUN_ROOT/samples" --manifest "$MANIFEST" \
      --collection-state "$RUN_ROOT/collection_state.json" --output "$INDEX_ROOT" \
      --workers "$workers" | tee "$REPORT_ROOT/p3_v7_v43_closed_loop_data_audit.json"
    ;;
  integrity-audit)
    "$YOPO_PYTHON" "$PROJECT_ROOT/tools/audit_p3_v7_v43_integrity.py" \
      --workers "$workers" \
      | tee "$REPORT_ROOT/p3_v7_v43_independent_integrity_audit.console.json"
    ;;
  authorize)
    "$YOPO_PYTHON" "$PROJECT_ROOT/tools/authorize_p5_closed_loop_v43.py" \
      --authority "$INDEX_ROOT/closed_loop_data_authority.json" \
      | tee "$REPORT_ROOT/p5_closed_loop_v43_training_authorization.json"
    ;;
  dry-run)
    "$YOPO_PYTHON" "$PROJECT_ROOT/tools/train_dep_car_closed_loop_v43.py" \
      --source "$SOURCE" --output "$OUTPUT" --workers "$workers" --dry-run \
      | tee "$REPORT_ROOT/p5_closed_loop_v43_dry_run.json"
    ;;
  pilot)
    "$YOPO_PYTHON" "$PROJECT_ROOT/tools/train_dep_car_closed_loop_v43.py" \
      --source "$SOURCE" --output "$PILOT" \
      --epochs 8 --workers "$workers" \
      --max-samples "$maximum_samples" --max-steps "$maximum_steps" \
      2>&1 | tee "$LOG_ROOT/pilot.log"
    ;;
  pilot-acceptance)
    "$YOPO_PYTHON" "$PROJECT_ROOT/tools/audit_p5_closed_loop_v43.py" \
      --checkpoint "${PILOT%.pth}.best.pth" --workers "$workers" \
      --maximum-samples "$maximum_samples" 2>&1 | tee "$LOG_ROOT/pilot_acceptance.log"
    ;;
  train)
    "$YOPO_PYTHON" "$PROJECT_ROOT/tools/train_dep_car_closed_loop_v43.py" \
      --source "$SOURCE" --output "$OUTPUT" --workers "$workers" \
      2>&1 | tee "$LOG_ROOT/training.log"
    ;;
  acceptance)
    "$YOPO_PYTHON" "$PROJECT_ROOT/tools/audit_p5_closed_loop_v43.py" \
      --checkpoint "${OUTPUT%.pth}.best.pth" --workers "$workers" \
      2>&1 | tee "$LOG_ROOT/acceptance.log"
    ;;
esac
