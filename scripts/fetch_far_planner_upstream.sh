#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="$PROJECT_ROOT/third_party/far_planner"
REPOSITORY="https://github.com/MichaelFYang/far_planner.git"
COMMIT="2799b6964c141cacd1c32a14b19bc7abffbe0e52"

if [[ -d "$TARGET/.git" ]]; then
  CURRENT="$(git -C "$TARGET" rev-parse HEAD)"
  if [[ "$CURRENT" == "$COMMIT" ]]; then
    echo "FAR Planner already locked at $COMMIT"
    exit 0
  fi
  echo "Existing FAR checkout is at $CURRENT; expected $COMMIT" >&2
  echo "Move it aside explicitly before fetching the locked version." >&2
  exit 2
fi
if [[ -e "$TARGET" ]]; then
  echo "Target exists but is not a Git checkout: $TARGET" >&2
  exit 2
fi

git clone --no-checkout --filter=blob:none "$REPOSITORY" "$TARGET"
git -C "$TARGET" checkout --detach "$COMMIT"
ACTUAL="$(git -C "$TARGET" rev-parse HEAD)"
[[ "$ACTUAL" == "$COMMIT" ]]
echo "FAR Planner fetched at locked commit $ACTUAL"
echo "The DE-P-Car runtime uses its native adapter; this checkout is retained for provenance and comparison."
