#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

# This is the explicit M6 entry.  Keep run_memory_navigation.sh as a backwards-
# compatible alias so existing replay/matrix commands and reports still work.
exec "$PROJECT_ROOT/scripts/run_memory_navigation.sh" "$@"
