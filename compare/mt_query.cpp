
// 用法：mt_query [dir] [M] [efC] [ef] [threads]
#include <atomic>
#include <chrono>
#include <cstdio>
#include <thread>
#include <vector>
#include "io/dataset.hpp"
#include "vortex/index.hpp"
#include "hnswlib/hnswlib.h"

static std::size_t arg(int argc, char** argv, int i, std::size_t def)
{
    return argc > i ? static_cast<std::size_t>(std::stoul(argv[i])) : def;
}

using Clock = std::chrono::steady_clock;

// 每线程认领 [lo,hi) 段查询，跑 reps 遍，数完成条数
template <typename Fn>
static double runQps(const Fn& search, std::size_t rows, std::size_t nth,
                     std::size_t reps)
{
    std::atomic<std::size_t> done{0};
    std::vector<std::thread> pool;
    const auto t0 = Clock::now();
    for (std::size_t t = 0; t < nth; t++)
        pool.emplace_back([&, t] {
            const std::size_t lo = rows * t / nth, hi = rows * (t + 1) / nth;
            for (std::size_t r = 0; r < reps; r++)
                for (std::size_t q = lo; q < hi; q++) { search(q); done.fetch_add(1); }
        });
    for (auto& th : pool) th.join();
    const double secs = std::chrono::duration<double>(Clock::now() - t0).count();
    return static_cast<double>(done.load()) / secs;
}

int main(int argc, char** argv)
{
    const std::string dir = argc > 1 ? argv[1] : "data/siftsmall";
    const std::size_t M = arg(argc, argv, 2, 16);
    const std::size_t efC = arg(argc, argv, 3, 200);
    const std::size_t ef = arg(argc, argv, 4, 100);
    const std::size_t nth = arg(argc, argv, 5, 8);
    const std::size_t k = 10, reps = 5;

    const vortex::Dataset base = vortex::readFvecs(dir + "/siftsmall_base.fvecs");
    const vortex::Dataset query = vortex::readFvecs(dir + "/siftsmall_query.fvecs");

    hnswlib::L2Space space(base.dims);
    hnswlib::HierarchicalNSW<float> hs(&space, base.rows, M, efC);
    for (std::size_t r = 0; r < base.rows; r++)
        hs.addPoint(base.row(r), static_cast<hnswlib::labeltype>(r));
    hs.setEf(ef);

    vortex::IndexParams p;
    p.dims = base.dims;
    p.capacity = base.rows;
    p.M = static_cast<int>(M);
    p.efConstruction = static_cast<int>(efC);
    p.seed = 20261014u;
    vortex::Index vx(p);
    std::vector<std::int64_t> ids(base.rows);
    for (std::size_t i = 0; i < base.rows; i++) ids[i] = static_cast<std::int64_t>(i);
    vx.addPoints(base.data.data(), ids.data(), base.rows, 1);  // 串行建图

    const auto hsSearch = [&](std::size_t q) { hs.searchKnn(query.row(q), k); };
    const auto vxSearch = [&](std::size_t q) {
        vx.searchKnn(query.row(q), static_cast<int>(k), static_cast<int>(ef));
    };

    const double h1 = runQps(hsSearch, query.rows, 1, reps);
    const double v1 = runQps(vxSearch, query.rows, 1, reps);
    const double hn = runQps(hsSearch, query.rows, nth, reps);
    const double vn = runQps(vxSearch, query.rows, nth, reps);

    std::printf("查询加速比 ef=%zu %zu线程 | hnswlib: %.0f→%.0f (%.1fx) | vortex: %.0f→%.0f (%.1fx)\n",
                ef, nth, h1, hn, hn / h1, v1, vn, vn / v1);
    return 0;
}
