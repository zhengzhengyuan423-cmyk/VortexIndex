#!/usr/bin/env bash
# hnswlib 对标：拿 hnswlib 当尺子量自己。一条命令跑完，两行结果并排打印。
# 前置：data/siftsmall 已就位；build-rel 已配置（随便 cmake --build 过一次就行）。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# 1. 拿包
TAR=third_party/hnswlib-0.8.0.tar.gz
[ -f "$TAR" ] || curl -L -o "$TAR" \
    https://codeload.github.com/nmslib/hnswlib/tar.gz/refs/tags/v0.8.0
[ -d third_party/hnswlib-0.8.0 ] || tar xzf "$TAR" -C third_party/

# 2. 编译对照程序
g++ -O3 -std=c++17 -Wall -pthread -o compare/hnsw_probe compare/run_hnswlib.cpp \
    src/io/dataset.cpp -Iinclude -Isrc -isystem third_party/hnswlib-0.8.0

# 3. 两臂各跑一遍：同数据同参（k=10 M=16 efC=200 ef=100 串行）
cmake --build build-rel --target run_bench >/dev/null
./build-rel/bench/run_bench data/siftsmall 10 5 api 16 200 100 heuristic 1 0
./compare/hnsw_probe data/siftsmall 10 5 16 200 100

# 4. 多线程建图：只看建图时间
./build-rel/bench/run_bench data/siftsmall 10 1 api 16 200 100 heuristic 8 0
./compare/hnsw_probe data/siftsmall 10 1 16 200 100 8

# 5. 并行查询：1→8 线程加速比
g++ -O3 -std=c++17 -Wall -pthread -o compare/mt_query compare/mt_query.cpp \
    src/io/dataset.cpp build-rel/src/libvortex.a -Iinclude -Isrc \
    -isystem third_party/hnswlib-0.8.0
./compare/mt_query data/siftsmall 16 200 100 8
