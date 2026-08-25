#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:---runtime}"

case "$MODE" in
  --runtime|--all) ;;
  *)
    echo "Usage: bash scripts/fetch_locked_repositories.sh [--runtime|--all]" >&2
    exit 2
    ;;
esac

fetch_locked() {
  local name="$1"
  local repository="$2"
  local commit="$3"
  local target="$PROJECT_ROOT/third_party/$name"

  if [[ -d "$target/.git" ]]; then
    local current
    current="$(git -C "$target" rev-parse HEAD)"
    if [[ "$current" != "$commit" ]]; then
      echo "$name is at $current; expected $commit" >&2
      echo "Move the existing checkout aside explicitly before retrying." >&2
      return 2
    fi
    if [[ -n "$(git -C "$target" status --porcelain)" ]]; then
      echo "$name has local changes; refusing to treat it as locked." >&2
      return 2
    fi
    echo "$name already locked at $commit"
    return 0
  fi
  if [[ -e "$target" ]]; then
    echo "$target exists but is not a Git checkout." >&2
    return 2
  fi
  git clone --no-checkout --filter=blob:none "$repository" "$target"
  git -C "$target" checkout --detach "$commit"
  [[ "$(git -C "$target" rev-parse HEAD)" == "$commit" ]]
  echo "$name fetched at $commit"
}

mkdir -p "$PROJECT_ROOT/third_party"

# Minimal P6 RViz runtime and provenance set.
fetch_locked \
  car-simulator \
  https://github.com/hifzhil/car-simulator.git \
  b113e3b0cd942585f54f444a8cde25154fcee360
fetch_locked \
  far_planner \
  https://github.com/MichaelFYang/far_planner.git \
  2799b6964c141cacd1c32a14b19bc7abffbe0e52

if [[ "$MODE" == "--all" ]]; then
  fetch_locked \
    DE-P \
    https://github.com/ZhangJunhe2005/DE-P.git \
    cbcc61d466b803dd5f32d7fa893e18793f2aa5ec
  fetch_locked \
    arena-tools \
    https://github.com/ignc-research/arena-tools.git \
    664b950d88c91b34e3cdce62512c8864b574b9d4
  fetch_locked \
    arena-rosnav-3D \
    https://github.com/ignc-research/arena-rosnav-3D.git \
    634bcb091a90b362087cdba5a9cd3856466d493c
  fetch_locked \
    ActorCollisionsPlugin \
    https://github.com/ignc-research/ActorCollisionsPlugin.git \
    eb2dc9de71f43669becc217f536951cf51d50a69
fi

echo "Locked repositories are ready for mode $MODE."
