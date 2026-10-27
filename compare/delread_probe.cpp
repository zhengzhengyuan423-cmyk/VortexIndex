// 边删边读探针：hnswlib 的无锁墓碑位翻写 vs 查询裸读，看竞争长什么样。
// 用法：g++ -O1 -g -fsanitize=thread 编译后直接跑，观察 ThreadSanitizer 输出。
#include <atomic>
#include <cstdio>
#include <random>
#include <thread>
#include <vector>

#include "hnswlib/hnswlib.h"

int main()
{
    const std::size_t n = 2000, dim = 64, readers = 4;
    hnswlib::L2Space space(dim);
    auto* idx = new hnswlib::HierarchicalNSW<float>(&space, n, 16, 200);

    std::vector<float> data(n * dim);
    std::mt19937 rng(7);
    for (auto& v : data) v = static_cast<float>(rng() % 1000) / 1000.f;
    for (std::size_t i = 0; i < n; i++) idx->addPoint(&data[i * dim], i);

    std::atomic<bool> stop{false};
    std::atomic<std::size_t> hits{0};
    std::vector<std::thread> pool;
    for (std::size_t r = 0; r < readers; r++) {
        pool.emplace_back([&, r] {
            std::vector<float> q(dim);
            std::mt19937 g(100 + r);
            while (!stop.load(std::memory_order_relaxed)) {
                for (auto& v : q) v = static_cast<float>(g() % 1000) / 1000.f;
                auto res = idx->searchKnn(q.data(), 10);
                hits += res.size();
            }
        });
    }
    for (std::size_t i = 0; i < n; i++) idx->markDelete(i);
    stop.store(true);
    for (auto& t : pool) t.join();
    std::printf("done deleted=%zu results=%zu\n", idx->getDeletedCount(), hits.load());
    delete idx;
}
