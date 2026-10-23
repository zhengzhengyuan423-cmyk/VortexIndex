// 图结构层测试：level_generator / graph_store / neighbor_select。
#include <gtest/gtest.h>
#include <vector>
#include "core/graph_store.hpp"
#include "core/level_generator.hpp"
#include "core/neighbor_select.hpp"
#include "vortex/errors.hpp"

namespace vortex
{
    static std::vector<std::int64_t> idsOf(const std::vector<KnnHit>& hits)
    {
        std::vector<std::int64_t> v;
        for (const auto& h : hits) v.push_back(h.id);
        return v;
    }

    // level_generator

    static std::vector<std::size_t> draw(LevelGenerator& gen, std::size_t n)
    {
        std::vector<std::size_t> v;
        v.reserve(n);
        for (std::size_t i = 0; i < n; i++) v.push_back(gen.nextLevel());
        return v;
    }

    TEST(LevelGeneratorTest, SameSeedSameSequence)
    {
        LevelGenerator a(16, 0x9E3779B9u), b(16, 0x9E3779B9u);
        EXPECT_EQ(draw(a, 1000), draw(b, 1000));
        LevelGenerator c(16, 12345u), d(16, 0x9E3779B9u);
        EXPECT_NE(draw(c, 100), draw(d, 100));
        EXPECT_THROW(LevelGenerator(1, 7u), ValueError);  // M 必须 >= 2
    }

    // M=16 时 P(level >= l) = 16^(-l)。tolerance 按理论分布取 5σ
    TEST(LevelGeneratorTest, GeometricDecay)
    {
        // N=100000：ge1 期望 6250（5σ≈383），ge2 期望 391（5σ≈99）
        LevelGenerator gen(16, 42u);
        std::size_t ge1 = 0, ge2 = 0;
        const std::size_t n = 100000;
        for (std::size_t i = 0; i < n; i++)
        {
            const std::size_t lvl = gen.nextLevel();
            if (lvl >= 1) ge1++;
            if (lvl >= 2) ge2++;
        }
        EXPECT_NEAR(static_cast<double>(ge1), static_cast<double>(n) / 16.0, 400.0);
        EXPECT_NEAR(static_cast<double>(ge2), static_cast<double>(n) / 256.0, 100.0);
    }

    // graph_store

    TEST(GraphStoreTest, AddVectorAndLinks)
    {
        GraphStore store(3, 4, 16);
        const float v[] = {1.f, 2.f, 3.f};
        EXPECT_EQ(store.addNode(v, 2), 0);
        EXPECT_EQ(store.size(), 1u);
        EXPECT_EQ(store.nodeLevel(0), 2u);
        EXPECT_FLOAT_EQ(store.vector(0)[0], 1.f);
        EXPECT_FLOAT_EQ(store.vector(0)[2], 3.f);
        EXPECT_TRUE(store.neighbors(0, 0).empty());
        EXPECT_THROW(store.neighbors(0, 3), ValueError);  // 超出节点层级
        // 第 0 层容量 2M，其余层 M
        EXPECT_EQ(store.neighborCapacity(0), 8u);
        EXPECT_EQ(store.neighborCapacity(1), 4u);
        // 定长块 + addLink：连边写回读得到
        store.addNode(v, 2);
        store.addLink(0, 0, 1);
        store.addLink(1, 0, 0);
        EXPECT_EQ(store.neighbors(0, 0).size(), 1u);
        EXPECT_EQ(store.neighbors(0, 0)[0], 1);
        // 第 0 层容量 2M=8：补满后下一条当场 CapacityError，块不扩容
        for (std::int64_t nb = 2; nb <= 8; nb++) store.addLink(0, 0, nb);
        EXPECT_EQ(store.neighbors(0, 0).size(), 8u);
        EXPECT_THROW(store.addLink(0, 0, 9), CapacityError);
        EXPECT_THROW(store.setLinks(0, 0, {1, 2, 3, 4, 5, 6, 7, 8, 9}),
                     CapacityError);  // 9 条 > 8
    }

    TEST(GraphStoreTest, EntryPointTracksHighestFirstWinsTie)
    {
        GraphStore store(1, 4, 16);
        const float v[] = {0.f};
        store.addNode(v, 0);  // id0 lvl0 => 入口
        store.addNode(v, 3);  // id1 lvl3 => 新高，换入口
        store.addNode(v, 3);  // id2 lvl3 并列 => 不换（先插入的赢）
        EXPECT_EQ(store.entryPoint(), 1);
        EXPECT_EQ(store.maxLevel(), 3u);
    }

    // neighbor_select

    TEST(NeighborSelectTest, NaiveSortedTopM)
    {
        // KnnHit 字段顺序是 {id, distance}：按距离升序取前 2
        std::vector<KnnHit> cand{{0, 5.f}, {1, 1.f}, {2, 3.f}, {3, 2.f}};
        EXPECT_EQ(idsOf(selectNeighbors(cand, 2)),
                  (std::vector<std::int64_t>{1, 3}));
        // M 超过候选数：全留仍定序
        std::vector<KnnHit> two{{0, 2.f}, {1, 1.f}};
        EXPECT_EQ(idsOf(selectNeighbors(two, 10)),
                  (std::vector<std::int64_t>{1, 0}));
        // 并列按 id 升序（确定性）
        std::vector<KnnHit> ties{{7, 1.f}, {2, 1.f}, {5, 1.f}};
        EXPECT_EQ(idsOf(selectNeighbors(ties, 3)),
                  (std::vector<std::int64_t>{2, 5, 7}));
        EXPECT_TRUE(selectNeighbors(cand, 0).empty());
    }


    static PairDistance coordDist(std::vector<float> x)
    {
        return [x](std::int64_t a, std::int64_t b)
        {
            const float d = x[a] - x[b];
            return d * d;
        };
    }

    TEST(NeighborSelectTest, HeuristicPrunesSameDirection)
    {
        // 缺 pairDist 的启发式模式当场抛
        std::vector<KnnHit> one{{0, 1.f}};
        EXPECT_THROW(selectNeighbors(one, 1, SelectStrategy::Heuristic),
                     ValueError);
        // q=0，候选全在正半轴：1(id0,d1)、2(id1,d4)、3(id2,d9)
        // id1 被挡：d(1,2)=1 < 4；id2 被挡：d(1,3)=4 < 9 → M=1 只留 id0
        std::vector<KnnHit> cand{{0, 1.f}, {1, 4.f}, {2, 9.f}};
        const auto out = selectNeighbors(cand, 1, SelectStrategy::Heuristic,
                                         coordDist({1.f, 2.f, 3.f}));
        ASSERT_EQ(out.size(), 1u);
        EXPECT_EQ(out[0].id, 0);
    }

    TEST(NeighborSelectTest, HeuristicBackfillsToM)
    {
        // 三点全挤同侧：启发式只留 1 个，M=2 必须按距离回填最近被剪者
        std::vector<KnnHit> cand{{0, 1.f}, {1, 4.f}, {2, 9.f}};
        const auto out = selectNeighbors(cand, 2, SelectStrategy::Heuristic,
                                         coordDist({1.f, 2.f, 3.f}));
        ASSERT_EQ(out.size(), 2u);
        EXPECT_EQ(idsOf(out), (std::vector<std::int64_t>{0, 1}));
    }

}
