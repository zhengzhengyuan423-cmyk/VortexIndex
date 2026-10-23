// 对外 API（Index 类）的边界测试
#include <gtest/gtest.h>
#include <vector>
#include "vortex/errors.hpp"
#include "vortex/index.hpp"

namespace vortex
{
    static IndexParams makeParams(std::size_t dims, std::size_t cap, int M = 8)
    {
        IndexParams p;
        p.dims = dims;
        p.capacity = cap;
        p.M = M;
        p.efConstruction = 100;  // >= M，过 validate
        return p;
    }

    // 单点插入姿势：写入只有一道门，addPoints 且 rows=1
    static void addOne(Index& idx, const float* v, std::int64_t id)
    {
        idx.addPoints(v, &id, 1);
    }

    // 1. 构造期参数校验：非法值当场 ValueError，不渗透进内核
    TEST(IndexTest, ParamValidation)
    {
        EXPECT_THROW(Index(makeParams(0, 10)), ValueError);       // dims=0
        EXPECT_THROW(Index(makeParams(3, 0)), ValueError);        // capacity=0
        EXPECT_THROW(Index(makeParams(3, 10, 1)), ValueError);    // M<2
        EXPECT_THROW(Index(makeParams(3, 10, 200)), ValueError);  // M>128
        IndexParams p = makeParams(3, 10);
        p.efConstruction = 1;  // efC<M
        EXPECT_THROW(Index{p}, ValueError);  // 花括号躲 most-vexing-parse
        EXPECT_NO_THROW(Index(makeParams(3, 10)));
    }

    // 2. 空索引：查询返回空，不抛——API 层替内核兜住 entry=-1
    TEST(IndexTest, EmptyIndexSearchReturnsEmpty)
    {
        Index idx(makeParams(3, 10));
        EXPECT_EQ(idx.size(), 0u);
        EXPECT_EQ(idx.deletedCount(), 0u);
        const float q[] = {1.f, 2.f, 3.f};
        EXPECT_TRUE(idx.searchKnn(q, 5, 10).empty());  // k>n 且空图
    }

    // 3. 入参防御：null / k<1 / ef<k 各自 ValueError
    TEST(IndexTest, SearchArgGuards)
    {
        Index idx(makeParams(2, 10, 4));
        const float v[] = {0.f, 0.f};
        addOne(idx, v, 0);
        const float* nullp = nullptr;
        EXPECT_THROW(addOne(idx, nullp, 1), ValueError);
        EXPECT_THROW(idx.searchKnn(nullp, 1, 1), ValueError);
        EXPECT_THROW(idx.searchKnn(v, 0, 5), ValueError);  // k<1
        EXPECT_THROW(idx.searchKnn(v, 5, 2), ValueError);  // ef<k
    }

    // 4. dims=1 端到端：数轴 0..99，查询 50 用 ef=n 全搜，必精确命中自身
    TEST(IndexTest, DimsOneEndToEnd)
    {
        Index idx(makeParams(1, 100, 4));
        for (int i = 0; i < 100; i++)
        {
            const float x = static_cast<float>(i);
            addOne(idx, &x, i);
        }
        EXPECT_EQ(idx.size(), 100u);
        const float q = 50.f;
        const auto r = idx.searchKnn(&q, 1, 100);  // ef=n => 精确
        ASSERT_EQ(r.size(), 1u);
        EXPECT_EQ(r[0].id, 50);
        EXPECT_NEAR(r[0].distance, 0.f, 1e-6f);
    }

    // 5. k > 已插入数：返回全部 n 条，不越界、不补空
    TEST(IndexTest, KGreaterThanNReturnsAll)
    {
        Index idx(makeParams(2, 10, 4));
        const float a[] = {0.f, 0.f}, b[] = {1.f, 1.f}, c[] = {5.f, 5.f};
        addOne(idx, a, 10);
        addOne(idx, b, 11);
        addOne(idx, c, 12);
        const auto r = idx.searchKnn(a, 10, 10);  // k=10 > n=3
        EXPECT_EQ(r.size(), 3u);
        EXPECT_EQ(r[0].id, 10);  // a 距 query 0，排第一
    }

    // 6. 重复向量全收：id 不唯一是设计，三条重合点都要出现在结果里
    TEST(IndexTest, DuplicateVectorsAllAccepted)
    {
        Index idx(makeParams(2, 10, 4));
        const float v[] = {1.f, 2.f};
        addOne(idx, v, 100);
        addOne(idx, v, 200);
        addOne(idx, v, 300);
        EXPECT_EQ(idx.size(), 3u);
        const auto r = idx.searchKnn(v, 3, 3);
        EXPECT_EQ(r.size(), 3u);  // 三条距离并列 0，全返回
    }

    // 7. 满容量：插入第 capacity+1 个抛 CapacityError
    TEST(IndexTest, CapacityFullThrows)
    {
        Index idx(makeParams(2, 2, 4));
        const float v[] = {0.f, 0.f};
        addOne(idx, v, 1);
        addOne(idx, v, 2);
        EXPECT_THROW(addOne(idx, v, 3), CapacityError);
    }

    // 8. 过滤：post-filter 掉某 id，结果里不得出现它
    TEST(IndexTest, FilterApplied)
    {
        Index idx(makeParams(2, 10, 4));
        const float a[] = {0.f, 0.f}, b[] = {1.f, 0.f}, c[] = {2.f, 0.f};
        addOne(idx, a, 1);
        addOne(idx, b, 2);
        addOne(idx, c, 3);
        const float q[] = {0.f, 0.f};
        const auto r = idx.searchKnn(q, 5, 10,
                                     [](std::int64_t id) { return id != 1; });
        ASSERT_FALSE(r.empty());
        for (const auto& h : r) EXPECT_NE(h.id, 1);
    }

    // 9. Cosine 零向量：语义无定义，addPoints 预检当场 ValueError；
    // 先验后占保证抛出时状态原样——一行没占、count 没动，之后一切照常
    TEST(IndexTest, CosineZeroVectorThrowsStateIntact)
    {
        IndexParams p = makeParams(2, 10, 4);
        p.metric = DistanceMetric::Cosine;
        Index idx(p);
        const float v[] = {1.f, 0.f}, z[] = {0.f, 0.f}, w[] = {0.f, 1.f};
        ASSERT_NO_THROW(addOne(idx, v, 1));
        EXPECT_THROW(addOne(idx, z, 2), ValueError);
        EXPECT_EQ(idx.size(), 1u);  // 抛的那次没留下孤儿槽
        ASSERT_NO_THROW(addOne(idx, w, 3));  // 图还能继续用
        EXPECT_EQ(idx.size(), 2u);
        const auto r = idx.searchKnn(v, 2, 10);  // 正常查询不受那次抛影响
        EXPECT_EQ(r.size(), 2u);
        // 已知限制：searchKnn 没做零向量预检，内核 evalDistance 照抛
        EXPECT_THROW(idx.searchKnn(z, 1, 10), ValueError);
    }

}
