// 用法：run_bench [dir] [k] [reps] [method] [M] [efC] [ef] [strategy] [threads] [save]
// method：bruteforce | hnsw（内部件直连）| api
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <optional>
#include <string>
#include <vector>
#include "bench_common.hpp"
#include "core/bruteforce.hpp"
#include "core/graph_store.hpp"
#include "core/hnsw_index.hpp"
#include "core/level_generator.hpp"
#include "io/dataset.hpp"
#include "vortex/errors.hpp"
#include "vortex/index.hpp"
#include "core/knn_hit.hpp"

// argv[i] 可选数字参数，缺省用 def
static std::size_t arg(int argc, char** argv, int i, std::size_t def)
{
    return argc > i ? static_cast<std::size_t>(std::stoul(argv[i])) : def;
}

static std::vector<std::int64_t> sortedIds(const std::vector<vortex::KnnHit>& hits)
{
    std::vector<std::int64_t> v;
    for (const auto& h : hits) v.push_back(h.id);
    std::sort(v.begin(), v.end());
    return v;
}

// 真值行按距离序排列，前 n 个就是 top-n；排序只为和 our ids 比较
static std::vector<std::int64_t> sortedTruth(const std::int32_t* row, std::size_t n)
{
    std::vector<std::int64_t> v(row, row + n);
    std::sort(v.begin(), v.end());
    return v;
}

// recall@k = |命中 ∩ 真值前 k| / k
static double recallAtK(const std::vector<vortex::KnnHit>& hits,
                        const std::int32_t* truthRow, std::size_t k)
{
    const auto ours = sortedIds(hits);
    const auto truth = sortedTruth(truthRow, k);
    std::vector<std::int64_t> inter;
    std::set_intersection(ours.begin(), ours.end(), truth.begin(),
                          truth.end(), std::back_inserter(inter));
    return static_cast<double>(inter.size()) / static_cast<double>(k);
}

int main(int argc, char** argv)
{
    const std::string dir = argc > 1 ? argv[1] : "data/siftsmall";
    const std::size_t k = arg(argc, argv, 2, 10);
    const std::size_t reps = arg(argc, argv, 3, 20);
    const std::string method = argc > 4 ? argv[4] : "bruteforce";
    const std::size_t M = arg(argc, argv, 5, 16);
    const std::size_t efC = arg(argc, argv, 6, 40);
    const std::size_t ef = arg(argc, argv, 7, 100);
    const std::string strategy = argc > 8 ? argv[8] : "naive";
    const bool viaIndex = (method == "api");
    const std::size_t threads =
        viaIndex ? arg(argc, argv, 9, 4) : 1;  // 其余路没起线程
    const bool save = argc > 10 && std::stoul(argv[10]) != 0;
    int exitCode = 0;

    try
    {
        const vortex::Dataset base =
            vortex::readFvecs(dir + "/siftsmall_base.fvecs");
        const vortex::Dataset query =
            vortex::readFvecs(dir + "/siftsmall_query.fvecs");
        const vortex::IntDataset gt =
            vortex::readIvecs(dir + "/siftsmall_groundtruth.ivecs");
        if (query.rows != gt.rows)
            throw vortex::IOError("query rows != groundtruth rows");
        if (gt.dims < k)
            throw vortex::IOError("groundtruth top-N smaller than requested k");

        
        if (method == "bruteforce")
        {
            std::size_t mismatches = 0;
            for (std::size_t q = 0; q < query.rows; q++)
            {
                const auto hits = vortex::bruteForceKnn(
                    vortex::DistanceMetric::L2, base.data.data(), base.rows,
                    base.dims, query.row(q), gt.dims);
                if (sortedIds(hits) != sortedTruth(gt.row(q), gt.dims))
                    mismatches++;
            }
            if (mismatches > 0)
            {
                std::fprintf(stderr, "groundtruth self-check: %zu/%zu mismatch\n",
                             mismatches, query.rows);
                exitCode = 2;
            }
        }


        std::optional<vortex::GraphStore> store;
        std::optional<vortex::LevelGenerator> levels;
        std::optional<vortex::Index> idx;
        double buildSecs = 0.0;
        if (method == "hnsw")
        {
            store.emplace(base.dims, M, base.rows);
            levels.emplace(M, 20261014u);
            const auto bb = std::chrono::steady_clock::now();
            vortex::buildIndex(*store, vortex::DistanceMetric::L2,
                               base.data.data(), base.rows, *levels, efC,
                               strategy == "heuristic"
                                   ? vortex::SelectStrategy::Heuristic
                                   : vortex::SelectStrategy::Naive);
            buildSecs = std::chrono::duration<double>(
                            std::chrono::steady_clock::now() - bb).count();
        }
        else if (viaIndex)  // api 路写入固定启发式剪枝
        {
            vortex::IndexParams p;
            p.dims = base.dims;
            p.capacity = base.rows;
            p.M = static_cast<int>(M);
            p.efConstruction = static_cast<int>(efC);
            p.seed = 20261014u;
            idx.emplace(p);
            std::vector<std::int64_t> ids(base.rows);
            for (std::size_t i = 0; i < base.rows; i++)
                ids[i] = static_cast<std::int64_t>(i);
            const auto bb = std::chrono::steady_clock::now();
            idx->addPoints(base.data.data(), ids.data(), base.rows, threads);
            buildSecs = std::chrono::duration<double>(
                            std::chrono::steady_clock::now() - bb).count();
        }

        // 查询：recall 只在第 1 遍算（答案确定，多遍纯计时）
        double recall = 0.0;
        const auto begin = std::chrono::steady_clock::now();
        for (std::size_t r = 0; r < reps; r++)
            for (std::size_t q = 0; q < query.rows; q++)
            {
                std::vector<vortex::KnnHit> hits;
                if (method == "hnsw")
                {
                    hits = vortex::searchKnn(*store, vortex::DistanceMetric::L2,
                                             query.row(q), ef);
                    if (hits.size() > k) hits.resize(k);
                }
                else if (viaIndex)
                {
                    const auto res = idx->searchKnn(
                        query.row(q), static_cast<int>(k), static_cast<int>(ef));
                    hits.reserve(res.size());
                    for (const auto& h : res) hits.push_back({h.id, h.distance});
                }
                else
                {
                    hits = vortex::bruteForceKnn(
                        vortex::DistanceMetric::L2, base.data.data(),
                        base.rows, base.dims, query.row(q), k);
                }
                if (r == 0) recall += recallAtK(hits, gt.row(q), k);
            }
        const double secs = std::chrono::duration<double>(
                                std::chrono::steady_clock::now() - begin).count();

        const auto [human, fileTag] = bench::stamp();
        char line[512];
        std::snprintf(line, sizeof line,
                      "{\"time\":\"%s\",\"method\":\"%s\",\"threads\":%zu,"
                      "\"recall_at_k\":%.4f,\"qps\":%.1f,\"avg_latency_ms\":%.3f,"
                      "\"build_seconds\":%.1f}\n",
                      human.c_str(), method.c_str(), threads,
                      recall / static_cast<double>(query.rows),
                      static_cast<double>(query.rows) * reps / secs,
                      secs * 1000.0 / static_cast<double>(query.rows * reps),
                      buildSecs);
        bench::emit(line, save, fileTag);
        return exitCode;
    }
    catch (const vortex::VortexError& e)
    {
        std::cerr << e.what() << "\n";
        return 1;
    }
}
