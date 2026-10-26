#!/usr/bin/env bash
# 全流程：只建库本体 → --install 到 .smoke_tmp/prefix → 干净目录里
# find_package 构建 examples/ 消费者 → 跑 → 断言首个命中是 id=11。

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="$ROOT/.smoke_tmp"
PREFIX="$WORK/prefix"
LIB_BUILD="$WORK/lib_build"
APP_BUILD="$WORK/app_build"
rm -rf "$WORK"

# 1. 库本体：tests/bench/cli 全关， install/export 路径不依赖它们
cmake -S "$ROOT" -B "$LIB_BUILD" -DCMAKE_BUILD_TYPE=Release \
      -DVORTEX_BUILD_TESTS=OFF -DVORTEX_BUILD_BENCH=OFF -DVORTEX_BUILD_CLI=OFF >/dev/null
cmake --build "$LIB_BUILD" --target vortex -j >/dev/null
cmake --install "$LIB_BUILD" --prefix "$PREFIX" >/dev/null

# 2. 消费者：只给 CMAKE_PREFIX_PATH
cmake -S "$ROOT/examples" -B "$APP_BUILD" \
      -DCMAKE_PREFIX_PATH="$PREFIX" >/dev/null
cmake --build "$APP_BUILD" -j >/dev/null

# 3. 实跑断言：两行结果，第一行 id=11（距 query 最近的点）
OUT="$("$APP_BUILD"/consumer)"
echo "$OUT"
LINES="$(echo "$OUT" | wc -l)"
[ "$LINES" -eq 2 ] || { echo "FAIL: expect 2 result lines, got $LINES"; exit 1; }
head -n1 <<< "$OUT" | grep -q '^id=11 ' || { echo "FAIL: first hit not id=11"; exit 1; }
echo "SMOKE OK"
