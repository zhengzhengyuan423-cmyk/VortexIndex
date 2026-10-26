#!/usr/bin/env bash
# TSan 并发用例复读：跑 N 轮 ParallelBuildTest+RwStressTest，任何一轮红就停并报轮号。
# 用法：bash scripts/tsan_repeat.sh [build-tsan 路径] [轮数]
# setarch -R 关 ASLR：本机 gcc 11.4 的 TSan 影子内存区和高位随机 mmap 撞车，
# 不开会死在 "unexpected memory mapping"——这是编译器运行时与内核布局的事，
# 不是被测代码的数据竞争（环境坑记 QA 当日条目）
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
