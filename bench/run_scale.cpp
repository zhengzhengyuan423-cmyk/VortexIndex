// 并发查询扩展性测量：串行建 10000 个 16 维随机向量（进 build_seconds字段但不计查询时间）
// 再 threads 个线程各把 1000 条查询（k=10、ef=100）
// 用法：run_scale [threads] [save]；save=1 

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <random>
#include <thread>
#include <vector>
#include "bench_common.hpp"
#include "vortex/config.hpp"
#include "vortex/index.hpp"

// 固定种子随机向量批次
static std::vector<float> makeData(std::size_t rows, std::size_t dims, std::uint64_t seed)
{
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> unit(0.f, 1.f);
    std::vector<float> v(rows * dims);
    for (auto& x : v) x = unit(rng);
    return v;
}

int main(int argc, char** argv)
{
    const std::size_t threads = argc > 1 ? static_cast<std::size_t>(std::stoul(argv[1])) : 1;
    const bool save = argc > 2 && std::stoul(argv[2]) != 0;

    const std::size_t dims = 16;
    const std::size_t rows = 10000;
    const std::size_t queries = 1000;
    const int k = 10;
    const int ef = 100;

    vortex::IndexParams p;
    p.dims = dims;
    p.capacity = rows;
    p.M = 16;
    p.efConstruction = 200;
    p.seed = 20261014u;

    const auto data = makeData(rows, dims, 20261021u);
    const auto qdata = makeData(queries, dims, 20261022u);
    std::vector<std::int64_t> ids(rows);
    for (std::size_t i = 0; i < rows; i++)
        ids[i] = static_cast<std::int64_t>(i);

    vortex::Index idx(p);
    const auto bb = std::chrono::steady_clock::now();
    idx.addPoints(data.data(), ids.data(), rows, 1);  // 串行建，不进查询计时
    const double buildSecs = std::chrono::duration<double>(
                                 std::chrono::steady_clock::now() - bb).count();

    // 每线程各跑全套 queries 条；总查询量 = threads * queries
    auto reader = [&]
    {
        for (std::size_t r = 0; r < queries; r++)
            idx.searchKnn(&qdata[r * dims], k, ef);
    };
    const auto begin = std::chrono::steady_clock::now();
    std::vector<std::thread> pool;
    for (std::size_t t = 0; t < threads; t++) pool.emplace_back(reader);  // reader 无参数、不捕获循环变量（QA30）
    for (auto& th : pool) th.join();
    const double secs = std::chrono::duration<double>(
                            std::chrono::steady_clock::now() - begin).count();
    const double totalQ = static_cast<double>(queries) * static_cast<double>(threads);


    double agree = 0.0;
    const std::size_t checks = 100;
    for (std::size_t q = 0; q < checks; q++)
    {
        const auto fast = idx.searchKnn(&qdata[q * dims], k, ef);
        const auto exact = idx.searchKnn(&qdata[q * dims], k,
                                         static_cast<int>(rows));
        std::size_t inter = 0;
        for (const auto& h : fast)
            for (const auto& e2 : exact)
                if (e2.id == h.id) inter++;
        agree += static_cast<double>(inter) / static_cast<double>(k);
    }
    const double recall = agree / static_cast<double>(checks);

    const auto [human, fileTag] = bench::stamp();
    char line[512];
    std::snprintf(line, sizeof line,
                  "{\"time\":\"%s\",\"method\":\"query\",\"threads\":%zu,"
                  "\"recall_at_k\":%.4f,\"qps\":%.1f,"
                  "\"avg_latency_ms\":%.3f,\"build_seconds\":%.1f}\n",
                  human.c_str(), threads, recall, totalQ / secs,
                  secs * 1000.0 / totalQ, buildSecs);
    bench::emit(line, save, fileTag);
    return 0;
}
