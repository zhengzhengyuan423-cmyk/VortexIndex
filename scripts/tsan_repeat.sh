#!/usr/bin/env bash

set -uo pipefail
BUILD="${1:-build-tsan}"
ROUNDS="${2:-30}"
BIN="$BUILD/tests/vortex_tests"
[ -x "$BIN" ] || { echo "no tsan binary at $BIN"; exit 2; }
export TSAN_OPTIONS="halt_on_error=1"
FILTER='ParallelBuildTest.*:RwStressTest.*'
i=1
while [ "$i" -le "$ROUNDS" ]; do
    if ! setarch "$(uname -m)" -R "$BIN" --gtest_filter="$FILTER" >/dev/null 2>&1; then
        echo "FAIL round $i"
        setarch "$(uname -m)" -R "$BIN" --gtest_filter="$FILTER" 2>&1 | tail -20
        exit 1
    fi
    i=$((i + 1))
done
echo "TSAN OK: $ROUNDS rounds clean"
