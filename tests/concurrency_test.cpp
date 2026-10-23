#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>
#include <thread>
#include <unordered_set>
#include <vector>
#include "vortex/errors.hpp"
#include "vortex/index.hpp"

namespace vortex
{
    static IndexParams concParams(std::size_t cap)
    {
        IndexParams p;
        p.dims = 2;
        p.capacity = cap;
        p.M = 8;
        p.efConstruction = 200;  // 拉高让串行下图接近连通；并发下仍不保证（QA18）
        return p;
    }

    // 数轴 0..n-1 上的 2 维点 {x,x}，第 i 行就是 id i
    static std::vector<float> lineData(std::size_t n)
    {
        std::vector<float> v(2 * n);
        for (std::size_t i = 0; i < n; i++)
        {
            v[2 * i] = static_cast<float>(i);
            v[2 * i + 1] = static_cast<float>(i);
        }
        return v;
    }

    static std::vector<std::int64_t> seqIds(std::size_t n)
    {
        std::vector<std::int64_t> ids(n);
        for (std::size_t i = 0; i < n; i++) ids[i] = static_cast<std::int64_t>(i);
        return ids;
    }

    // 单点插入姿势：唯一写入口 addPoints 且 rows=1
    static void addOne(Index& idx, const float* v, std::int64_t id)
    {
        idx.addPoints(v, &id, 1);
    }

    // 读线程节拍：必须 sleep，无休自旋的读者会饿死写者；
    // 跨线程一律 EXPECT（ASSERT 失败只退得出本线程）
    static void readerLoop(const Index& idx, std::atomic<bool>& stop,
                           long& okSearches, float x)
    {
        const float q[] = {x, x};
        okSearches = 0;
        while (!stop.load(std::memory_order_acquire))
        {
            const auto r = idx.searchKnn(q, 10, 50);
            EXPECT_LE(r.size(), 10u);
            okSearches++;
            std::this_thread::sleep_for(std::chrono::microseconds(200));
        }
    }

    // 1. 并发写（D11）：4 线程各逐条灌 250 个不相交 id，顺序不定、总量定
    TEST(RwStressTest, MultiWriterDisjointIds)
    {
        Index idx(concParams(1000));
        std::vector<std::thread> writers;
        for (int t = 0; t < 4; t++)
            writers.emplace_back([&idx, t]
            {
                // y=x 一条线、各线程领一段连续编号；刻意避开"4 堆各一小段线"：
                // 病态到达次序下偶发死角，断连就分不清是谁的锅（QA18）
                for (int i = 0; i < 250; i++)
                {
                    const float x = static_cast<float>(t * 250 + i);
                    const float v[] = {x, x};
                    addOne(idx, v, static_cast<std::int64_t>(t) * 1000 + i);
                }
            });
        for (auto& th : writers) th.join();
        EXPECT_EQ(idx.size(), 1000u);
        // 只钉快照不变量：非空、id 合法、无重复；"必找齐"强断言留给单线程
        const float q[] = {0.f, 0.f};
        const auto r = idx.searchKnn(q, 1000, 1000);
        ASSERT_FALSE(r.empty());
        std::unordered_set<std::int64_t> seen;
        for (const auto& h : r)
        {
            EXPECT_GE(h.id, 0);
            EXPECT_LT(h.id, 4000);  // t*1000+i，最大 3249
            EXPECT_TRUE(seen.insert(h.id).second);
        }
    }

    // 2. threads=1 必须与逐条串行全等：两索引查询逐条相等
    TEST(ParallelBuildTest, ThreadsOneEqualsSerialLoop)
    {
        const auto data = lineData(300);
        const auto ids = seqIds(300);
        Index serial(concParams(1000));
        for (int i = 0; i < 300; i++) addOne(serial, &data[2 * i], i);
        Index batch(concParams(1000));
        batch.addPoints(data.data(), ids.data(), 300, 1);

        EXPECT_EQ(batch.size(), 300u);
        for (const float x : {5.f, 123.f, 299.f})
        {
            const float q[] = {x, x};
            EXPECT_EQ(serial.searchKnn(q, 10, 50), batch.searchKnn(q, 10, 50));
            EXPECT_FALSE(serial.searchKnn(q, 10, 50).empty());
        }
    }

    // 3. 4 线程批量灌：数量、id 合法无重复、召回@10 下限 0.75
    TEST(ParallelBuildTest, FourThreadInvariantsAndRecallFloor)
    {
        const std::size_t n = 1200;
        const auto data = lineData(n);
        const auto ids = seqIds(n);

        Index idx(concParams(n));
        idx.addPoints(data.data(), ids.data(), n, 4);
        ASSERT_EQ(idx.size(), n);

        const float q0[] = {0.f, 0.f};
        const auto all = idx.searchKnn(q0, static_cast<int>(n), static_cast<int>(n));
        ASSERT_FALSE(all.empty());
        std::unordered_set<std::int64_t> seen;
        for (const auto& h : all)
        {
            EXPECT_GE(h.id, 0);
            EXPECT_LT(h.id, static_cast<std::int64_t>(n));
            EXPECT_TRUE(seen.insert(h.id).second);
        }

        // 真值=数轴上 |id-x| 最小的 10 个（并列取小编号）；串行实测近 1.0，
        // 并行到达次序不可控，只允许差一小截
        double hit = 0.0;
        for (int qi = 0; qi < 50; qi++)
        {
            const float x = static_cast<float>((qi + 1) * n / 52);
            const float q[] = {x, x};
            const auto r = idx.searchKnn(q, 10, 100);
            std::vector<std::int64_t> truth = seqIds(n);
            std::partial_sort(truth.begin(), truth.begin() + 10, truth.end(),
                              [&](std::int64_t a, std::int64_t b)
                              {
                                  const double da = std::fabs(static_cast<double>(a) - x);
                                  const double db = std::fabs(static_cast<double>(b) - x);
                                  return da < db || (da == db && a < b);
                              });
            for (const auto& h : r)
                if (std::find(truth.begin(), truth.begin() + 10, h.id) != truth.begin() + 10)
                    hit += 1.0;
        }
        EXPECT_GE(hit / 500.0, 0.75);
    }

    // 4. 预检不过 = 分毫未动：容量/空指针/threads=0/Cosine 零行；
    //    threads>rows 自动砍、rows=0 合法空批
    TEST(ParallelBuildTest, PrecheckFailuresLeaveIndexUntouched)
    {
        const auto data = lineData(100);
        const auto ids = seqIds(100);

        Index tooSmall(concParams(50));
        EXPECT_THROW(tooSmall.addPoints(data.data(), ids.data(), 100, 4),
                     CapacityError);
        EXPECT_EQ(tooSmall.size(), 0u);
        ASSERT_NO_THROW(addOne(tooSmall, data.data(), 7));  // 失败后还能正常插
        EXPECT_EQ(tooSmall.size(), 1u);

        Index idx(concParams(1000));
        const std::int64_t* nullIds = nullptr;
        EXPECT_THROW(idx.addPoints(nullptr, ids.data(), 10, 4), ValueError);
        EXPECT_THROW(idx.addPoints(data.data(), nullIds, 10, 4), ValueError);
        EXPECT_THROW(idx.addPoints(data.data(), ids.data(), 10, 0), ValueError);
        EXPECT_EQ(idx.size(), 0u);
        EXPECT_NO_THROW(idx.addPoints(data.data(), ids.data(), 1, 8));  // 砍到 1 线程
        EXPECT_EQ(idx.size(), 1u);
        EXPECT_NO_THROW(idx.addPoints(data.data(), ids.data(), 0, 8));  // 空批合法
        EXPECT_EQ(idx.size(), 1u);

        IndexParams cp = concParams(1000);
        cp.metric = DistanceMetric::Cosine;
        Index cosine(cp);
        const auto ids8 = seqIds(8);
        // 8 行 {i+1,i+1} 里第 3 行改成 {0,0}：批前必须逮住，整批未进
        std::vector<float> dirty;
        for (int i = 0; i < 8; i++)
        {
            dirty.push_back(static_cast<float>(i + 1));
            dirty.push_back(static_cast<float>(i + 1));
        }
        dirty[6] = 0.f;
        dirty[7] = 0.f;
        EXPECT_THROW(cosine.addPoints(dirty.data(), ids8.data(), 8, 4), ValueError);
        EXPECT_EQ(cosine.size(), 0u);
    }

    // 5. 并行建图与查询同场：4 线程建 1000 点，3 条读线程全程狂查
    //    （管线搜索发共享锁、落笔发独占锁，这条就是 TSan 的门禁搭档）
    TEST(ParallelBuildTest, ParallelBuildConcurrentWithReaders)
    {
        const std::size_t n = 1000;
        const auto data = lineData(n);
        const auto ids = seqIds(n);

        Index idx(concParams(n));
        std::atomic<bool> stop{false};
        std::vector<std::thread> readers;
        std::vector<long> counts(3, 0);
        for (int t = 0; t < 3; t++)
            readers.emplace_back([&, t] { readerLoop(idx, stop, counts[t], 3.5f); });

        idx.addPoints(data.data(), ids.data(), n, 4);
        stop.store(true, std::memory_order_release);
        for (auto& th : readers) th.join();

        EXPECT_EQ(idx.size(), n);
        EXPECT_GT(counts[0] + counts[1] + counts[2], 0);
    }

}
