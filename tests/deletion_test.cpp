// 墓碑删除、批量删除与压实的测试
#include <gtest/gtest.h>
#include <vector>
#include "vortex/errors.hpp"
#include "vortex/index.hpp"

namespace vortex
{
    static IndexParams delParams(std::size_t dims, std::size_t cap, int M = 4)
    {
        IndexParams p;
        p.dims = dims;
        p.capacity = cap;
        p.M = M;
        p.efConstruction = 100;  // >= M，过 validate
        return p;
    }

    static void add1(Index& idx, const float* v, std::int64_t id)
    {
        idx.addPoints(v, &id, 1);
    }

    // 1. 同名多向量一次全埋：一篇文档切三段共用一个 id，删 id 就是删整篇
    TEST(DeletionTest, RemoveAllSlotsOfSameId)
    {
        Index idx(delParams(2, 10));
        const float a[] = {0.f, 0.f}, b[] = {1.f, 0.f}, c[] = {0.f, 1.f}, d[] = {9.f, 9.f};
        add1(idx, a, 7);
        add1(idx, b, 7);
        add1(idx, c, 7);
        add1(idx, d, 8);
        EXPECT_EQ(idx.removePoint(7), 3u);       // 三槽同灭
        EXPECT_EQ(idx.size(), 1u);
        EXPECT_EQ(idx.deletedCount(), 3u);
        const float q[] = {0.f, 0.f};
        const auto r = idx.searchKnn(q, 10, 10);  // ef=10 >= 4 槽，全图精确
        ASSERT_EQ(r.size(), 1u);                  // 墓碑一条不漏出来
        EXPECT_EQ(r[0].id, 8);
        EXPECT_EQ(idx.removePoint(7), 0u);        // 幂等
        EXPECT_EQ(idx.removePoint(999), 0u);      // 查无此人
    }

    // 2. 批删：返回实删数，查无跳过，重复 id 只算一次；null 防御
    TEST(DeletionTest, BatchRemoveCounts)
    {
        Index idx(delParams(1, 100));
        for (int i = 0; i < 50; i++)
        {
            const float x = static_cast<float>(i);
            add1(idx, &x, i);
        }
        const std::vector<std::int64_t> ids = {0, 1, 2, 3, 4, 200, 0};
        EXPECT_EQ(idx.removePoint(ids.data(), ids.size()), 5u);  // 200 无，0 重复
        EXPECT_EQ(idx.deletedCount(), 5u);
        EXPECT_EQ(idx.size(), 45u);
        EXPECT_THROW(idx.removePoint(nullptr, 1), ValueError);
        EXPECT_EQ(idx.removePoint(ids.data(), 0), 0u);           // 空批合法
    }

    // 3. 删后重插同 id：查到的是新向量；旧墓碑不重复计数
    TEST(DeletionTest, ReinsertAfterRemove)
    {
        Index idx(delParams(1, 10));
        const float old0[] = {0.f}, fresh[] = {50.f};
        add1(idx, old0, 1);
        EXPECT_EQ(idx.removePoint(1), 1u);
        add1(idx, fresh, 1);
        EXPECT_EQ(idx.size(), 1u);
        EXPECT_EQ(idx.deletedCount(), 1u);
        const auto r = idx.searchKnn(fresh, 1, 10);
        ASSERT_EQ(r.size(), 1u);
        EXPECT_EQ(r[0].id, 1);
        EXPECT_NEAR(r[0].distance, 0.f, 1e-6f);   // 命中的是新向量不是旧的
        EXPECT_EQ(idx.removePoint(1), 1u);        // 只埋新槽，旧的已是墓碑
        EXPECT_EQ(idx.deletedCount(), 2u);
        EXPECT_EQ(idx.size(), 0u);
    }

    // 4. 墓碑占槽：容量不还不算数，compact 才腾地方
    TEST(DeletionTest, TombstonesOccupyCapacityUntilCompact)
    {
        Index idx(delParams(1, 3));
        const float x0[] = {0.f}, x1[] = {1.f}, x2[] = {2.f}, x3[] = {3.f};
        add1(idx, x0, 0);
        add1(idx, x1, 1);
        add1(idx, x2, 2);
        EXPECT_EQ(idx.removePoint(0), 1u);
        EXPECT_THROW(add1(idx, x3, 3), CapacityError);  // 槽还占着
        idx.compact();
        EXPECT_NO_THROW(add1(idx, x3, 3));              // 压实后腾出来了
        EXPECT_EQ(idx.size(), 3u);
        EXPECT_EQ(idx.deletedCount(), 0u);
    }

    // 5. 压实保结果：数轴 0..39 埋掉全部奇数，compact 前后全搜 top-10 逐位一致
    TEST(DeletionTest, CompactPreservesExactResults)
    {
        Index idx(delParams(1, 40));
        for (int i = 0; i < 40; i++)
        {
            const float x = static_cast<float>(i);
            add1(idx, &x, i);
        }
        std::vector<std::int64_t> odds;
        for (int i = 1; i < 40; i += 2) odds.push_back(i);
        EXPECT_EQ(idx.removePoint(odds.data(), odds.size()), 20u);

        const float q = 20.f;
        const auto before = idx.searchKnn(&q, 10, 40);  // ef=槽数，全图精确
        idx.compact();
        const auto after = idx.searchKnn(&q, 10, 40);
        ASSERT_EQ(before.size(), 10u);
        EXPECT_EQ(after, before);                        // 值序全同
        EXPECT_EQ(idx.size(), 20u);
        EXPECT_EQ(idx.deletedCount(), 0u);
        // 压实后容量口径也复位：墓碑不占位了，还能再灌 20 个
        const float x = 100.f;
        EXPECT_NO_THROW(add1(idx, &x, 100));
    }

    // 6. 压实确定性：同库同活点集，删除怎么分批不影响压实产物逐位一致
    TEST(DeletionTest, CompactDeterministicAcrossRemoveBatches)
    {
        Index a(delParams(1, 40)), b(delParams(1, 40));
        for (int i = 0; i < 40; i++)
        {
            const float x = static_cast<float>(i);
            add1(a, &x, i);
            add1(b, &x, i);
        }
        std::vector<std::int64_t> first = {1, 3, 5, 7, 9}, second = {33, 35, 37, 39};
        second.insert(second.end(), {11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31});
        ASSERT_EQ(a.removePoint(first.data(), first.size()), 5u);
        ASSERT_EQ(a.removePoint(second.data(), second.size()), 15u);
        std::vector<std::int64_t> all = second;          // b：乱序合成一批删
        all.insert(all.end(), first.begin(), first.end());
        ASSERT_EQ(b.removePoint(all.data(), all.size()), 20u);
        a.compact();
        b.compact();
        const float q = 12.f;
        EXPECT_EQ(a.searchKnn(&q, 10, 40), b.searchKnn(&q, 10, 40));
    }

    // 7. 无墓碑时 compact 是空操作：图一个字节没动
    TEST(DeletionTest, CompactNoopWithoutTombstones)
    {
        Index idx(delParams(1, 20));
        for (int i = 0; i < 20; i++)
        {
            const float x = static_cast<float>(i) * 0.5f;
            add1(idx, &x, i);
        }
        const float q = 3.f;
        const auto before = idx.searchKnn(&q, 5, 20);
        idx.compact();
        EXPECT_EQ(idx.searchKnn(&q, 5, 20), before);
        EXPECT_EQ(idx.size(), 20u);
    }

    // 8. 全删光再压实：空图合法，查询回空不炸
    TEST(DeletionTest, CompactToEmpty)
    {
        Index idx(delParams(1, 5));
        for (int i = 0; i < 5; i++)
        {
            const float x = static_cast<float>(i);
            add1(idx, &x, i);
        }
        EXPECT_EQ(idx.removePoint(0), 1u);
        EXPECT_EQ(idx.removePoint(1), 1u);
        EXPECT_EQ(idx.removePoint(2), 1u);
        EXPECT_EQ(idx.removePoint(3), 1u);
        EXPECT_EQ(idx.removePoint(4), 1u);
        idx.compact();
        EXPECT_EQ(idx.size(), 0u);
        EXPECT_EQ(idx.deletedCount(), 0u);
        const float q = 2.f;
        EXPECT_TRUE(idx.searchKnn(&q, 3, 10).empty());
        const float x = 42.f;                            // 空图后照常可插
        EXPECT_NO_THROW(add1(idx, &x, 9));
    }

    // 9. 墓碑 + 业务过滤器叠加：两样各扣各的，凑不满 k 就少给，不报错
    TEST(DeletionTest, TombstoneAndFilterStackUp)
    {
        Index idx(delParams(1, 4));
        for (int i = 1; i <= 4; i++)
        {
            const float x = static_cast<float>(i);
            add1(idx, &x, i);
        }
        EXPECT_EQ(idx.removePoint(2), 1u);               // 墓碑埋掉 2
        const float q = 1.f;
        const auto r = idx.searchKnn(&q, 10, 10,
                                     [](std::int64_t id) { return id != 3; });  // 过滤 3
        ASSERT_EQ(r.size(), 2u);                         // 剩 1、4
        EXPECT_EQ(r[0].id, 1);
        EXPECT_EQ(r[1].id, 4);
    }

}
