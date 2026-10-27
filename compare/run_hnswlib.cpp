// hnswlib 对照组：同数据同参，跑完直接打印三个数。

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <thread>
#include <vector>
#include "io/dataset.hpp"
#include "vortex/errors.hpp"
#include "hnswlib/hnswlib.h"

static std::size_t arg(int argc, char** argv, int i, std::size_t def)
{
    return argc > i ? static_cast<std::size_t>(std::stoul(argv[i])) : def;
}

int main(int argc, char** argv)
{
    const std::string dir = argc > 1 ? argv[1] : "data/siftsmall";
    const std::size_t k = arg(argc, argv, 2, 10);
    const std::size_t reps = arg(argc, argv, 3, 5);
    const std::size_t M = arg(argc, argv, 4, 16);
    const std::size_t efC = arg(argc, argv, 5, 200);
    const std::size_t ef = arg(argc, argv, 6, 100);
    const std::size_t nth = arg(argc, argv, 7, 1);

    try
    {
        const vortex::Dataset base = vortex::readFvecs(dir + "/siftsmall_base.fvecs");
        const vortex::Dataset query = vortex::readFvecs(dir + "/siftsmall_query.fvecs");
        const vortex::IntDataset gt = vortex::readIvecs(dir + "/siftsmall_groundtruth.ivecs");

        hnswlib::L2Space space(base.dims);  // L2Space 返回平方距离，和我们口径一致
        hnswlib::HierarchicalNSW<float> idx(&space, base.rows, M, efC);

        const auto bb = std::chrono::steady_clock::now();
        std::vector<std::thread> pool;
        for (std::size_t t = 0; t < nth; t++)
            pool.emplace_back([&, t] {  // 跳跃分片：线程 t 插 t, t+nth, t+2*nth…
                for (std::size_t r = t; r < base.rows; r += nth)
                    idx.addPoint(base.row(r), static_cast<hnswlib::labeltype>(r));
            });
        for (auto& th : pool) th.join();
        const double buildSecs = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - bb).count();
        idx.setEf(ef);

        double recall = 0.0;
        const auto begin = std::chrono::steady_clock::now();
        for (std::size_t rp = 0; rp < reps; rp++)
            for (std::size_t q = 0; q < query.rows; q++)
            {
                // searchKnn 的堆顶是远端，pop 出来的是 id 集合
                auto pq = idx.searchKnn(query.row(q), k);
                std::vector<std::int64_t> ours;
                while (!pq.empty())
                {
                    ours.push_back(static_cast<std::int64_t>(pq.top().second));
                    pq.pop();
                }
                if (rp == 0)  // recall 第一遍算，后面几遍纯计时
                {
                    std::sort(ours.begin(), ours.end());
                    std::vector<std::int64_t> truth(gt.row(q), gt.row(q) + k);
                    std::sort(truth.begin(), truth.end());
                    std::vector<std::int64_t> inter;
                    std::set_intersection(ours.begin(), ours.end(), truth.begin(),
                                          truth.end(), std::back_inserter(inter));
                    recall += static_cast<double>(inter.size()) /
                              static_cast<double>(k);
                }
            }
        const double secs = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - begin).count();

        std::printf("hnswlib  M=%zu efC=%zu ef=%zu threads=%zu | 建图 %.1fs | recall %.4f | QPS %.0f\n",
                    M, efC, ef, nth, buildSecs,
                    recall / static_cast<double>(query.rows),
                    static_cast<double>(query.rows) * reps / secs);
        return 0;
    }
    catch (const vortex::VortexError& e)
    {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
