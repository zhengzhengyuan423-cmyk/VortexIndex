# VortexIndex

C++17 实现的 HNSW 近似最近邻检索库。技术栈：C++17 / CMake / pthread。

核心指标（siftsmall 10k×128 维，M=16，efConstruction=200，k=10，ef=100，AMD Ryzen 7 8845H）：recall@10 = 0.9990；串行查询约 4000 QPS，8 线程并发查询 21k–26k QPS；51 个测试全部通过；串行建图同 seed 产物字节级一致。

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build -j
```

## 关键设计决策

| 决策点 | 选择 | 放弃 | 理由 |
|---|---|---|---|
| 图存储 | SoA：向量区/层号区/邻接区各自连续大数组 | 每节点一个对象（AoS） | 搜索跳边时连续内存缓存友好；序列化可整区连续落盘 |
| 分层抽签 | mt19937 + 用户传入 seed | random_device | 串行建图可字节复现，验收测试可用 `cmp` 直接比对产物 |
| 并行建图 | 窗口化批提交（乱序搜索、定序落笔） | 每元素细锁 | 细锁方案未能论证"半更新的邻居表对并发读者安全"，先行搁置；后续对照实验表明落笔段未并行，8 线程仅 1.2× 加速，细锁+即时提交已列入重构计划 |
| 邻居剪枝 | 默认论文启发式（heuristic），保留 naive 开关 | 只留一种 | 启发式与 hnswlib 默认配置一致；naive 开关留作对照 |
| 持久化 | 手写单文件二进制，68B 头 + 五个连续区 | protobuf / cereal | 不引入依赖；加载时做四项校验（magic/版本/参数/邻接边界） |
| 删除 | 墓碑标记 + compact 整体重建 | 就地删边 | HNSW 入边无法枚举索引，直接删边不保证连通性；compact 重新抽取层号、重建连续存储 |
| 读写并发 | 一根 shared_mutex：查询读锁、提交写锁 | 全局互斥 / 无锁结构 | 8 线程并发查询实测加速 7.0–8.2×，未见读锁显著开销 |


## 性能数据

环境：AMD Ryzen 7 8845H（8 核）/ Ubuntu 22.04 / g++ 11.4 `-O3`。数据集：siftsmall（128 维，10k base / 1k query / 官方 groundtruth），k=10。下表为同场次多次运行的中位区间。

| 指标 | VortexIndex | hnswlib v0.8.0 |
|---|---|---|
| recall@10（ef=100） | 0.9990 | 0.9980 |
| 串行查询 QPS | ~4000 | ~9000（0.42×） |
| 串行建图（efC=200） | ~12s | ~2.1s |
| 8 线程建图 | 9.9s（1.2×） | 0.3s（~7×） |
| 8 线程查询 QPS | 21k–26k（加速 7.0–8.2×） | 23k–41k（加速 3.2–6.3×） |

差距归因（复现命令见 [compare/RESULTS.md](compare/RESULTS.md)）：

- **串行查询 QPS 0.42×**：距离函数未手写 SIMD。编译器不允许对浮点求和重结合，反汇编中 `evalDistance` 的 mulps 仅 9 条、addss 35 条；hnswlib 手写 SSE 全程 SIMD。这是串行差距主因。
- **并行建图 1.2×**：提交粒度是瓶颈，写锁整批独占，搜索段并行而落笔段串行。见 compare/RESULTS.md 多线程建图一节。
- **读锁无瓶颈**：8 线程并发查询加速 7.0–8.2×，锁开销未测出。

## 快速开始

依赖：CMake ≥ 3.20、支持 C++17 的 g++、pthread。库本体构建不需要联网（gtest 仅测试目标需联网拉取）；siftsmall 已随仓库提交，可用 `data/fetch_datasets.sh` 重新下载。

两种使用形式都从 clone 构建开始：

```bash
git clone https://github.com/zhengzhengyuan423-cmyk/VortexIndex && cd VortexIndex
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

### 1. 命令行模式

直接用 `./build/tools/vortex-cli`，三个子命令：

```
build <数据集目录> <输出.bin> [线程数] [M] [efConstruction]   建图并存盘
query <索引.bin> <数据集目录> <第几行> [k] [ef]               加载 .bin，查询指定的一条 query
bench <数据集目录> [线程数] [k] [ef]                          现场建图 + 全量查询，输出一行 JSON（recall_at_k / qps / avg_latency_ms / build_seconds）
```

数据集目录按后缀找文件：`*_base.fvecs`（库向量）、`*_query.fvecs`（查询向量）、`*_groundtruth.ivecs`（官方真值，bench 需要）。

```bash
./build/tools/vortex-cli build data/siftsmall index.bin 8   # 8 线程建图
./build/tools/vortex-cli query index.bin data/siftsmall 0   # 用 index.bin 查第 0 条 query
./build/tools/vortex-cli bench data/siftsmall 1 10 64       # 单线程，k=10，ef=64
```

### 2. 接口调用模式

先安装，供其他工程 `find_package`：

```bash
cmake --install build --prefix ./install
```

自己的 CMake 工程里把安装目录加进 `CMAKE_PREFIX_PATH`（配置期 `-DCMAKE_PREFIX_PATH=/绝对路径/install`），然后：

```cmake
find_package(vortex 0.1 CONFIG REQUIRED)
target_link_libraries(my_app PRIVATE vortex::vortex)
```

头文件 `#include <vortex/index.hpp>`，全部在 `vortex::` 命名空间。接口一览：

| 接口 | 签名 | 用法 |
|---|---|---|
| IndexParams | `dims` / `capacity` / `M=16` / `efConstruction=200` / `seed` / `metric=L2` | 构造参数。dims、capacity 必填；capacity 建库时固定，不自动扩容；metric 可选 L2 / InnerProduct / Cosine |
| `Index(params)` | 构造函数 | 建空索引；参数非法抛 `ValueError` |
| `addPoints` | `void addPoints(const float* data, const std::int64_t* ids, std::size_t rows, std::size_t threads = 1)`；文件重载 `void addPoints(const std::string& fvecsPath, std::size_t threads = 1)` | 批量插入，rows 行 × dims 维，外部 id 由调用方给；文件重载整读一个 fvecs（仅 fvecs 格式），外部 id = 文件行号，threads>1 为并行建图 |
| `searchKnn` | `std::vector<SearchResult> searchKnn(const float* query, int k, int ef, const IdFilter& filter = {}) const` | 查询 k 近邻，返回按距离升序的 `SearchResult`；要求 k ≥ 1、ef ≥ k；filter 是 id 谓词，对结果做搜后筛，缺省不筛 |
| `SearchResult` | `struct SearchResult { std::int64_t id; float distance; };` | searchKnn 的结果元素：`id` 为插入时传入的外部 id；`distance` 按索引度量计算——L2 为平方欧氏距离（不开方），InnerProduct 为 1 − a·b，Cosine 为 1 − cos θ |
| `removePoint` | `std::size_t removePoint(std::int64_t id)`，批量重载 `removePoint(const std::int64_t*, std::size_t)` | 墓碑删除，返回实际删除的槽位数；删除不影响图结构 |
| `compact` | `void compact()` | 按槽号升序把活点重建为新图，外部 id 不变；墓碑清零 |
| `size` / `deletedCount` | `std::size_t size() const` | 活跃向量数（不含墓碑）/ 墓碑槽位数 |
| `save` / `load` | `void save(const std::string&) const`；`static Index load(const std::string&)` | 单文件二进制存取；加载时做 magic/版本/参数/邻接边界四项校验 |

典型调用序列：建索引 → addPoints → searchKnn → save，之后 `Index::load(path)` 直接查，无需重建。

与 hnswlib 的对比运行 `bash compare/run.sh`，依次输出：串行两臂、8 线程建图、并行查询加速比（需联网下载 hnswlib 源码包）。

## 项目结构

```
include/vortex/   对外头文件（Index、参数、错误类型）
src/core/         HNSW 内核：图存储、搜索、建图流水线、剪枝、分层抽签
src/io/           fvecs/ivecs 数据集读写 + 单文件持久化格式
src/api/          Index 门面：参数校验、shared_mutex、墓碑表
tools/            vortex-cli（build/query/bench 三个子命令）
tests/            51 个 GoogleTest 用例（单元 + 并发 + 持久化 + 确定性验收）
bench/            基准程序（bruteforce / 内核直连 / api 三条路径）
compare/          与 hnswlib v0.8.0 的对比（基准程序 + 运行脚本 + 结果文档）
```

## 代码来源与基准选型

本库按 HNSW 论文（[原文](https://arxiv.org/pdf/1603.09320)）独立实现——分层插入、单层候选扩展、启发式邻居选择均按论文描述编写，性能对照选 hnswlib v0.8.0，数据与差距归因见 [compare/RESULTS.md](compare/RESULTS.md)；hnswlib 只出现在 `compare/` 目录，库本体与发布包不链接它。

## 已知局限（v0.1）

- 仅验证 siftsmall（10k 条）；10 万级以上未测试，更大规模下批提交窗口的扩展开销与并行搜索的缓存行为需重测。
- 距离函数为标量循环，依赖编译器自动向量化；手写 SIMD 列入重构计划首位。
- 并行建图仅 1.2× 加速：写图要攒一批再独占锁落笔，这期间算图和查询都排队；改成即时提交之前不会更快。
- 持久化无校验和：向量区内容损坏不会被检测到，文件仍可加载、召回率下降无提示；四项校验只覆盖文件误用、截断、版本不符、邻接越界。
- `capacity` 建库时固定，不自动扩容；达到容量后继续添加会抛出异常。
- 过滤仅支持"搜索后再筛"（post-filter），无图内过滤，无属性索引。
- 单机单进程，无分布式；无 CI，验证通过本地脚本完成：smoke 测试、sanitizer 检查、TSan 多轮并发测试，跑分记录在 [compare/](compare/) 与 [bench/results/](bench/results/)。

## v0.2 构想

定位：从距离搜索库走向可被他人使用的 RAG（检索增强生成）检索层。

- 逐条消化上述已知局限：手写 SIMD 距离内核、建图细粒度锁 + 即时提交、sift1M 规模验证、属性过滤下推进图遍历等。
- SQ8 量化 + fp32 重排 + 分区 mmap 加载：内存降约 1/4，向量区可留磁盘。
- 补半径查询与批量查询。
- Python 绑定，接现成 embedding 模型跑通"灌文档 → 查询 → 返回上下文段落"演示。
