// 评测 harness
// 用法：run_bench [dir] [k] [reps] [method] [M] [efC] [ef]
#include <chrono>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include "core/bruteforce.hpp"
#include "core/graph_store.hpp"
#include "core/hnsw_index.hpp"
#include "core/level_generator.hpp"
#include "io/dataset.hpp"
#include "vortex/errors.hpp"

// 按id排，查的快
static std::vector<std::int64_t> sortedIds(const std::vector<vortex::KnnHit>& hits)
{
    std::vector<std::int64_t> v;
    for (const auto& h : hits) v.push_back(h.id);
    std::sort(v.begin(), v.end());
    return v;
}

static std::vector<std::int64_t> sortedTruth(const std::int32_t* row, std::size_t n)
{
    std::vector<std::int64_t> v(row, row + n);
    std::sort(v.begin(), v.end());
    return v;
}

// 一条查询的 recall@k = 命中集合与真值前 k 的交集大小 / k
static double recallAtK(const std::vector<vortex::KnnHit>& hits,
                        const std::int32_t* truthRow, std::size_t k)
{
    const auto ours = sortedIds(hits);
    const auto truth = sortedTruth(truthRow, k);
    std::vector<std::int64_t> inter;
    std::set_intersection(ours.begin(), ours.end(), truth.begin(), truth.end(), std::back_inserter(inter));
    return static_cast<double>(inter.size()) / static_cast<double>(k);
}

// argv[i] 可选数字参数，缺省用 def
static std::size_t arg(int argc, char** argv, int i, std::size_t def)
{
    return argc > i ? static_cast<std::size_t>(std::stoul(argv[i])) : def;
}

class JsonRow
{
  public:
    void addStr(const std::string& key, const std::string& v)
    {
        emit(key);
        oss_ << "\"" << v << "\"";
    }
    void addInt(const std::string& key, std::size_t v)
    {
        emit(key);
        oss_ << v;
    }
    void addDouble(const std::string& key, double v, int prec)
    {
        emit(key);
        oss_ << std::fixed << std::setprecision(prec) << v;
    }
    std::string str() const { return "{" + oss_.str() + "}\n"; }

  private:
    void emit(const std::string& key)
    {
        if (!first_) oss_ << ", ";
        first_ = false;
        oss_ << "\"" << key << "\": ";
    }

    std::ostringstream oss_;
    bool first_ = true;
};

int main(int argc, char** argv)
{
    // 读参数
    const std::string dir = argc > 1 ? argv[1] : "data/siftsmall";
    const std::size_t k = arg(argc, argv, 2, 10);
    const std::size_t reps = arg(argc, argv, 3, 20);
    const std::string method = argc > 4 ? argv[4] : "bruteforce";
    const std::size_t M = arg(argc, argv, 5, 16);
    const std::size_t efC = arg(argc, argv, 6, 40);
    const std::size_t ef = arg(argc, argv, 7, 100);
    const std::string strategy = argc > 8 ? argv[8] : "naive";

    try
    {
        const vortex::Dataset base = vortex::readFvecs(dir + "/siftsmall_base.fvecs");         //基底数据
        const vortex::Dataset query = vortex::readFvecs(dir + "/siftsmall_query.fvecs");       //目标向量
        const vortex::IntDataset gt = vortex::readIvecs(dir + "/siftsmall_groundtruth.ivecs"); //真值

        std::printf(
            "base: %zu x %zu | query: %zu x %zu | groundtruth: %zu x %zu\n",
            base.rows, base.dims, query.rows, query.dims, gt.rows, gt.dims);
        if (query.rows != gt.rows)   //query 和 groundtruth 的行数必须一致
            throw vortex::IOError("query rows != groundtruth rows");
        if (gt.dims < k)
            throw vortex::IOError("groundtruth top-N smaller than requested k");

        // 自检只给 bruteforce 跑
        std::size_t mismatches = 0;
        if (method == "bruteforce")
        {
            for (std::size_t q = 0; q < query.rows; q++)
            {
                const auto hits = vortex::bruteForceKnn
                (vortex::DistanceMetric::L2, base.data.data(), base.rows, base.dims, query.row(q), gt.dims);
                if (sortedIds(hits) != sortedTruth(gt.row(q), gt.dims))
                {
                    if (mismatches < 5)
                        std::fprintf(stderr,"groundtruth mismatch at query %zu\n", q);
                    mismatches++;
                }
            }
            std::printf(
                "self-check: %zu/%zu queries disagree with official "
                "groundtruth\n",
                mismatches, query.rows);
        }

        // hnsw：先建图，单独计时，不混进查询 QPS
        vortex::GraphStore store(base.dims, M, base.rows);
        vortex::LevelGenerator levels(M, 20261014u);
        double buildSecs = 0.0;
        if (method == "hnsw")
        {
            const auto bb = std::chrono::steady_clock::now();
            vortex::buildIndex(store, vortex::DistanceMetric::L2,
                               base.data.data(), base.rows, levels, efC,
                               strategy == "heuristic" ? vortex::SelectStrategy::Heuristic : vortex::SelectStrategy::Naive);
            buildSecs = std::chrono::duration<double>(std::chrono::steady_clock::now() - bb).count();
            std::printf("build done in %.1f s | entry=%lld maxLevel=%zu\n",
                        buildSecs, static_cast<long long>(store.entryPoint()),
                        store.maxLevel());
        }

        double recall = 0.0;
        const auto begin = std::chrono::steady_clock::now();
        for (std::size_t r = 0; r < reps; r++)
            for (std::size_t q = 0; q < query.rows; q++)
            {
                std::vector<vortex::KnnHit> hits;
                if (method == "hnsw")
                {
                    hits = vortex::searchKnn(store, vortex::DistanceMetric::L2,
                                             query.row(q), ef);
                    // hits 已距离升序定序，截断即 top-k
                    if (hits.size() > k) hits.resize(k);
                }
                else
                {
                    hits = vortex::bruteForceKnn(vortex::DistanceMetric::L2,
                                                 base.data.data(), base.rows,
                                                 base.dims, query.row(q), k);
                }
                if (r == 0) recall += recallAtK(hits, gt.row(q), k);
            }
        const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();

        recall /= static_cast<double>(query.rows);
        JsonRow j;
        j.addStr("dataset", dir);
        j.addStr("method", method);
        j.addStr("metric", "L2");
        j.addInt("k", k);
        j.addInt("queries", query.rows);
        j.addInt("reps", reps);
        j.addDouble("recall_at_k", recall, 4);
        j.addDouble("qps", static_cast<double>(query.rows) * reps / secs, 1);
        j.addDouble("avg_latency_ms", secs * 1000.0 / (query.rows * reps), 3);
        j.addInt("groundtruth_mismatches", mismatches);
        if (method == "hnsw")
        {
            j.addInt("M", M);
            j.addInt("ef_construction", efC);
            j.addInt("ef", ef);
            j.addDouble("build_seconds", buildSecs, 1);
            j.addStr("prune", strategy);
        }
        std::cout << j.str();
        return mismatches == 0 ? 0 : 2;
    }
    catch (const vortex::VortexError& e)
    {
        std::cerr << e.what() << "\n";
        return 1;
    }
}
