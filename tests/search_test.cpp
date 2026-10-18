// 搜索与建图层测试：bruteforce / searchLayer / insertNode / searchKnn / buildIndex
#include <gtest/gtest.h>
#include <algorithm>
#include <iostream>
#include <random>
#include <set>
#include <vector>
#include "core/bruteforce.hpp"
#include "core/graph_store.hpp"
#include "core/hnsw_index.hpp"
#include "core/level_generator.hpp"
#include "io/dataset.hpp"
#include "vortex/errors.hpp"

namespace vortex
{
    static std::vector<std::int64_t> idsOf(const std::vector<KnnHit>& hits)
    {
        std::vector<std::int64_t> v;
        for (const auto& h : hits) v.push_back(h.id);
        return v;
    }

    // bruteforce：5 个 2D 点
    const float kPts[] = {0, 0, 1, 0, 0, 2, 3, 4, -1, -1};

    TEST(BruteforceTest, TopKL2)
    {
        // query (0,0) 到各点平方距离: id0=0, id1=1, id2=4, id3=25, id4=2
        const float q[] = {0.f, 0.f};
        const auto hits = bruteForceKnn(DistanceMetric::L2, kPts, 5, 2, q, 3);
        ASSERT_EQ(hits.size(), 3u);
        EXPECT_EQ(idsOf(hits), (std::vector<std::int64_t>{0, 1, 4}));
        EXPECT_NEAR(hits[2].distance, 2.f, 1e-6f);
        // k=10 > rows=5：全量升序 0, 1, 2, 4, 25
        const auto all = bruteForceKnn(DistanceMetric::L2, kPts, 5, 2, q, 10);
        ASSERT_EQ(all.size(), 5u);
        EXPECT_EQ(idsOf(all), (std::vector<std::int64_t>{0, 1, 4, 2, 3}));
        // 重合点 (5,5)x2 距离并列，k=1 取先出现的 id0
        const float dup[] = {5, 5, 5, 5};
        const auto tie = bruteForceKnn(DistanceMetric::L2, dup, 2, 2, q, 1);
        EXPECT_EQ(tie[0].id, 0);
    }

    TEST(BruteforceTest, GuardsAndInnerProduct)
    {
        const float q[] = {0.f, 0.f};
        EXPECT_TRUE(bruteForceKnn(DistanceMetric::L2, kPts, 5, 2, q, 0).empty());
        EXPECT_TRUE(bruteForceKnn(DistanceMetric::L2, nullptr, 0, 2, q, 3).empty());
        EXPECT_THROW(bruteForceKnn(DistanceMetric::L2, kPts, 5, 2, nullptr, 3), ValueError);
        // IP 距离 = 1 - dot：id0 (2,0) -> -1；id1 (0,3) -> -2。
        // 负距离不是 bug，排序仍升序：id1 在前
        const float pts[] = {2, 0, 0, 3};
        const float qip[] = {1.f, 1.f};
        const auto hits = bruteForceKnn(DistanceMetric::InnerProduct, pts, 2, 2, qip, 2);
        EXPECT_EQ(idsOf(hits), (std::vector<std::int64_t>{1, 0}));
        EXPECT_NEAR(hits[0].distance, -2.f, 1e-6f);
    }

    // searchLayer 
    // 手工构造：6 个点在数轴 0..5（dims=1），双向链 0-1-2-3-4-5，入口 0。
    // 链图上的搜索行为可以逐拍推演，写死期望值
    static GraphStore buildChain()
    {
        GraphStore store(1, 4, 6);
        for (int i = 0; i < 6; i++)
        {
            const float x = static_cast<float>(i);
            store.addNode(&x, 0);
        }
        for (int i = 0; i + 1 < 6; i++)
        {
            store.neighbors(i, 0).push_back(i + 1);
            store.neighbors(i + 1, 0).push_back(i);
        }
        return store;
    }

    TEST(SearchLayerTest, Chain)
    {
        const GraphStore store = buildChain();
        const float q[] = {2.6f};
        // 推演：入口0(6.76)→1(2.56)→2(0.36)→3(0.16)，之后候选都比 0.36 远，
        // 提前终止。ef=2 恰好收住 {3,2}
        const auto top2 = searchLayer(store, DistanceMetric::L2, q, 0, 2, 0);
        ASSERT_EQ(top2.size(), 2u);
        EXPECT_EQ(idsOf(top2), (std::vector<std::int64_t>{3, 2}));
        EXPECT_NEAR(top2[0].distance, 0.16f, 1e-5f);
        // ef=6：found 未满前一路接受，链全走完才停
        const auto all = searchLayer(store, DistanceMetric::L2, q, 0, 6, 0);
        ASSERT_EQ(all.size(), 6u);
        EXPECT_EQ(idsOf(all), (std::vector<std::int64_t>{3, 2, 4, 1, 5, 0}));
    }

    TEST(SearchLayerTest, GuardsAndTies)
    {
        const GraphStore store = buildChain();
        const float q[] = {2.6f};
        const float* nullQ = nullptr;
        EXPECT_THROW(searchLayer(store, DistanceMetric::L2, nullQ, 0, 2, 0), ValueError);
        EXPECT_THROW(searchLayer(store, DistanceMetric::L2, q, -1, 2, 0), ValueError);
        EXPECT_THROW(searchLayer(store, DistanceMetric::L2, q, 6, 2, 0), ValueError);
        EXPECT_TRUE(searchLayer(store, DistanceMetric::L2, q, 0, 0, 0).empty());
        GraphStore empty(1, 4, 2);
        EXPECT_TRUE(searchLayer(empty, DistanceMetric::L2, q, 0, 2, 0).empty());  // 空图先于 entry

        // 悬空邻居 id 必须当场抛，不能静默越界
        GraphStore bad = buildChain();
        bad.neighbors(0, 0).push_back(99);
        EXPECT_THROW(searchLayer(bad, DistanceMetric::L2, q, 0, 3, 0), ValueError);

        // 重合点 (1,0)(2,0) 都连入口 0=(0,0)：距离全并列，输出按 id 升序
        GraphStore dup(1, 4, 3);
        const float a = 0.f, b = 1.f;
        dup.addNode(&a, 0);
        dup.addNode(&b, 0);
        dup.addNode(&b, 0);
        dup.neighbors(0, 0) = {1, 2};
        const float q0[] = {0.f};
        const auto hits = searchLayer(dup, DistanceMetric::L2, q0, 0, 3, 0);
        EXPECT_EQ(idsOf(hits), (std::vector<std::int64_t>{0, 1, 2}));
    }

    // 50 点 2D 合成图（每点连 4 近邻、对称化）：不同 ef 的 recall@10 打印成表。
    TEST(SearchLayerTest, Random50RecallTable)
    {
        std::mt19937 rng(20261014u);
        std::uniform_real_distribution<float> uni(0.f, 1.f);

        const std::size_t n = 50;
        Dataset pts;
        pts.rows = n;
        pts.dims = 2;
        for (std::size_t i = 0; i < n; i++)
        {
            pts.data.push_back(uni(rng));
            pts.data.push_back(uni(rng));
        }

        GraphStore store(2, 4, n);
        for (std::size_t i = 0; i < n; i++) store.addNode(pts.row(i), 0);
        // k=5 含自己，取其余 4 个做近邻，双向连边
        for (std::size_t i = 0; i < n; i++)
        {
            const auto nn = bruteForceKnn(DistanceMetric::L2, pts.data.data(),
                                          n, 2, pts.row(i), 5);
            for (std::size_t j = 1; j < nn.size(); j++)
            {
                auto& mine = store.neighbors(static_cast<std::int64_t>(i), 0);
                if (std::find(mine.begin(), mine.end(), nn[j].id) == mine.end())
                    mine.push_back(nn[j].id);
                auto& theirs = store.neighbors(nn[j].id, 0);
                if (std::find(theirs.begin(), theirs.end(),
                              static_cast<std::int64_t>(i)) == theirs.end())
                    theirs.push_back(static_cast<std::int64_t>(i));
            }
        }

        // 连通性前置检查：不连通的话下面的对比没有意义，直接判失败
        std::set<std::size_t> seen;
        std::vector<std::size_t> stack = {0};
        seen.insert(0);
        while (!stack.empty())
        {
            const std::size_t cur = stack.back();
            stack.pop_back();
            for (const std::int64_t nb :
                 store.neighbors(static_cast<std::int64_t>(cur), 0))
                if (seen.insert(static_cast<std::size_t>(nb)).second)
                    stack.push_back(static_cast<std::size_t>(nb));
        }
        ASSERT_EQ(seen.size(), n) << "synthetic graph is not connected";

        const float q[] = {uni(rng), uni(rng)};
        const auto truth = bruteForceKnn(DistanceMetric::L2, pts.data.data(), n, 2, q, 10);
        const std::vector<std::int64_t> truthVec = idsOf(truth);
        const std::set<std::int64_t> truthIds(truthVec.begin(), truthVec.end());

        std::cout << "ef -> recall@10 (query on 50-node kNN4 graph):\n";
        double recallAt25 = 0.0;
        for (const std::size_t ef : {5u, 10u, 25u, 50u})
        {
            const auto hits =
                searchLayer(store, DistanceMetric::L2, q, 0, ef, 0);
            const std::vector<std::int64_t> hitVec = idsOf(hits);
            const std::set<std::int64_t> hitIds(hitVec.begin(), hitVec.end());
            std::size_t inter = 0;
            for (const auto id : hitIds)
                if (truthIds.count(id)) inter++;
            const double r = static_cast<double>(inter) / 10.0;
            std::cout << "  ef=" << ef << ": recall@10 = " << r << "\n";
            if (ef == 25) recallAt25 = r;
        }
        EXPECT_GE(recallAt25, 0.5);
    }

    // insertNode / searchKnn / buildIndex

    // 图不变量：所有邻居 id 在 [0, size) 且不超过该层容量、无自环。
    // 悬空 id 和超容是连边代码最原始的两种错法。
    static void expectGraphSane(const GraphStore& store, std::size_t M)
    {
        const std::size_t n = store.size();
        for (std::size_t i = 0; i < n; i++)
        {
            const std::size_t lvl =
                store.nodeLevel(static_cast<std::int64_t>(i));
            for (std::size_t lc = 0; lc <= lvl; lc++)
            {
                const auto& links =
                    store.neighbors(static_cast<std::int64_t>(i), lc);
                EXPECT_LE(links.size(), store.neighborCapacity(lc))
                    << "node " << i << " layer " << lc;
                for (const std::int64_t nb : links)
                {
                    ASSERT_GE(nb, 0);
                    ASSERT_LT(static_cast<std::size_t>(nb), n)
                        << "dangling from node " << i;
                    EXPECT_NE(nb, static_cast<std::int64_t>(i))
                        << "self-loop on node " << i;
                }
            }
        }
    }

    TEST(InsertNodeTest, FirstAndSecondNodes)
    {
        GraphStore store(1, 2, 4);
        const float a = 0.f, b = 1.f;
        // 空图：第一个点直接成入口，没得连
        EXPECT_EQ(insertNode(store, DistanceMetric::L2, &a, 0, 8), 0);
        EXPECT_EQ(store.entryPoint(), 0);
        EXPECT_TRUE(store.neighbors(0, 0).empty());
        // 第二个点：双向各连一条
        insertNode(store, DistanceMetric::L2, &b, 0, 8);
        EXPECT_EQ(store.neighbors(0, 0), (std::vector<std::int64_t>{1}));
        EXPECT_EQ(store.neighbors(1, 0), (std::vector<std::int64_t>{0}));
        expectGraphSane(store, 2);
    }

    TEST(InsertNodeTest, FiveOnLineReachable)
    {
        // 数轴 0..4 依次插入（M=2，第 0 层容量 4）
        GraphStore store(1, 2, 5);
        for (int i = 0; i < 5; i++)
        {
            const float x = static_cast<float>(i);
            insertNode(store, DistanceMetric::L2, &x, 0, 8);
            expectGraphSane(store, 2);
        }
        const float q = 2.f;
        const auto hits = searchLayer(store, DistanceMetric::L2, &q,
                                      store.entryPoint(), 5, 0);
        ASSERT_EQ(hits.size(), 5u);
        EXPECT_EQ(hits[0].id, 2);
        // searchKnn 同一张图：距离升序 {2,1,3,0,4}
        const auto top = searchKnn(store, DistanceMetric::L2, &q, 5);
        ASSERT_EQ(top.size(), 5u);
        EXPECT_EQ(idsOf(top), (std::vector<std::int64_t>{2, 1, 3, 0, 4}));
    }

    TEST(InsertNodeTest, DuplicatePointsRespectCapacity)
    {
        // 20 个完全重合的点：候选全并列，Naive 按 id 截断。
        // 反复超容回炉剪枝必须把每张邻居表压在第 0 层容量 2M 以内
        GraphStore store(1, 2, 20);
        const float v = 3.f;
        for (int i = 0; i < 20; i++)
        {
            insertNode(store, DistanceMetric::L2, &v, 0, 8);
            expectGraphSane(store, 2);
        }

        const auto hits = searchLayer(store, DistanceMetric::L2, &v,
                                      store.entryPoint(), 8, 0);
        EXPECT_GE(hits.size(), 1u);
        EXPECT_LE(hits.size(), 8u);
    }

    TEST(InsertNodeTest, MultiLevelWiring)
    {
        // A(level2) → B(level1) → C(level0)：B 要在 0、1 两层都连上 A，C 只在第 0 层有边；入口保持 A
        GraphStore store(1, 2, 3);
        const float a = 0.f, b = 5.f, c = 9.f;
        insertNode(store, DistanceMetric::L2, &a, 2, 8);
        insertNode(store, DistanceMetric::L2, &b, 1, 8);
        insertNode(store, DistanceMetric::L2, &c, 0, 8);

        EXPECT_EQ(store.entryPoint(), 0);
        EXPECT_EQ(store.maxLevel(), 2u);
        EXPECT_FALSE(store.neighbors(1, 1).empty());  // B 的第 1 层连了东西
        EXPECT_EQ(store.nodeLevel(2), 0u);  // C 只有第 0 层（查 layer1 会抛）
        EXPECT_FALSE(store.neighbors(2, 0).empty());  // 但第 0 层必须连上
        expectGraphSane(store, 2);
    }

    TEST(BuildIndexTest, NullArgsThrow)
    {
        GraphStore store(2, 4, 8);
        LevelGenerator lv(4, 1u);
        const float* nul = nullptr;
        EXPECT_THROW(insertNode(store, DistanceMetric::L2, nul, 0, 8), ValueError);
        EXPECT_THROW(buildIndex(store, DistanceMetric::L2, nul, 3, lv, 8), ValueError);
    }

    TEST(BuildIndexTest, SameSeedSameGraph)
    {
        // 确定性回归锚：30 个合成点、同 seed 建两遍，逐节点逐层邻居表全等。
        std::mt19937 rng(77u);
        std::uniform_real_distribution<float> uni(0.f, 1.f);
        const std::size_t n = 30, dims = 2;
        std::vector<float> data(n * dims);
        for (auto& x : data) x = uni(rng);

        GraphStore a(dims, 4, n), b(dims, 4, n);
        LevelGenerator la(4, 555u), lb(4, 555u);
        buildIndex(a, DistanceMetric::L2, data.data(), n, la, 8);
        buildIndex(b, DistanceMetric::L2, data.data(), n, lb, 8);

        ASSERT_EQ(a.size(), b.size());
        EXPECT_EQ(a.entryPoint(), b.entryPoint());
        for (std::size_t i = 0; i < n; i++)
        {
            const std::int64_t id = static_cast<std::int64_t>(i);
            ASSERT_EQ(a.nodeLevel(id), b.nodeLevel(id)) << "node " << i;
            for (std::size_t lc = 0; lc <= a.nodeLevel(id); lc++)
                EXPECT_EQ(a.neighbors(id, lc), b.neighbors(id, lc))
                    << "node " << i << " layer " << lc;
        }
        expectGraphSane(a, 4);
    }

}
