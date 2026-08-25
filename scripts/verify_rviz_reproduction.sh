#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
EXPECTED_CHECKPOINT_SHA="c89d5401774477caf11159495ee3d5e8eb3fbe6c95fe742ee0c8d528f0f535ac"

require_file() {
  if [[ ! -f "$PROJECT_ROOT/$1" ]]; then
    echo "Missing reproduction artifact: $1" >&2
    exit 2
  fi
}

require_revision() {
  local relative="$1"
  local expected="$2"
  if [[ ! -d "$PROJECT_ROOT/$relative/.git" ]]; then
    echo "Missing locked checkout: $relative" >&2
    exit 2
  fi
  local actual
  actual="$(git -C "$PROJECT_ROOT/$relative" rev-parse HEAD)"
  if [[ "$actual" != "$expected" ]]; then
    echo "$relative revision $actual differs from $expected" >&2
    exit 2
  fi
}

require_file models/dep_car/p5_closed_loop_v43/fusion_closed_loop_sequence.best.pth
require_file models/dep_car/p5_closed_loop_v43/fusion_closed_loop_sequence.best.contract.json
require_file models/dep_car/p5_closed_loop_v43/fusion_closed_loop_sequence.best.acceptance.json
require_file models/dep_car/p5_closed_loop_v43/fusion_p6_shadow.authority.json
require_file data/p6_static/reproduction_manifest.json
require_file data/p6_static/maps/dep_car_map_0000_2a91f458/map.world
require_file data/p6_static/maps/dep_car_map_0000_2a91f458/model.sdf
require_revision third_party/car-simulator b113e3b0cd942585f54f444a8cde25154fcee360
require_revision third_party/far_planner 2799b6964c141cacd1c32a14b19bc7abffbe0e52

ACTUAL_CHECKPOINT_SHA="$(sha256sum "$PROJECT_ROOT/models/dep_car/p5_closed_loop_v43/fusion_closed_loop_sequence.best.pth" | awk '{print $1}')"
if [[ "$ACTUAL_CHECKPOINT_SHA" != "$EXPECTED_CHECKPOINT_SHA" ]]; then
  echo "V4.3 checkpoint SHA-256 mismatch: $ACTUAL_CHECKPOINT_SHA" >&2
  exit 2
fi

if [[ ! -f "$PROJECT_ROOT/catkin_ws/devel/setup.bash" ]]; then
  echo "catkin workspace is not built; run bash scripts/bootstrap_workspace.sh" >&2
  exit 2
fi

bash "$PROJECT_ROOT/scripts/run_p6_v43_shadow.sh" --stage audit
bash "$PROJECT_ROOT/scripts/run_p6_v43_shadow.sh" --stage interactive --dry-run

echo "RViz reproduction preflight PASS"
