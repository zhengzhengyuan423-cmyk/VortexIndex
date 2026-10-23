#include "hnsw_index.hpp"
#include <algorithm>
#include <functional>
#include <queue>
#include <utility>
#include "vortex/errors.hpp"

namespace vortex
{
    // 堆元素
    using HeapPair = std::pair<float, std::int64_t>;

    std::vector<KnnHit> searchLayer(const GraphStore& store,
                                    DistanceMetric metric, const float* query,
                                    std::int64_t entry, std::size_t ef,
                                    std::size_t layer)
    {
        if (query == nullptr) throw ValueError("query must not be null");
        const std::size_t n = store.size();
        if (ef == 0 || n == 0) return {};
        if (entry < 0 || static_cast<std::size_t>(entry) >= n)
            throw ValueError("entry point out of range");
        if (store.nodeLevel(entry) < layer)  // entry 层级不足
            throw ValueError("entry below requested layer");

        const std::size_t dims = store.dims();
        std::vector<char> visited(n, 0);

        // candidates 小顶堆（最近的先展开）；found 大顶堆封顶
        std::priority_queue<HeapPair, std::vector<HeapPair>, std::greater<HeapPair>> candidates;
        std::priority_queue<HeapPair> found;

        const float dEntry = evalDistance(metric, store.vector(entry), query, dims);
        candidates.emplace(dEntry, entry);
        found.emplace(dEntry, entry);
        visited[static_cast<std::size_t>(entry)] = 1;

        while (!candidates.empty())
        {
            const HeapPair c = candidates.top();
            candidates.pop();

            if (c.first > found.top().first) break;  // 重要剪枝

            for (const std::int64_t nb : store.neighbors(c.second, layer))
            {
                if (nb < 0 || static_cast<std::size_t>(nb) >= n)
                    throw ValueError("dangling neighbor id");
                if (visited[static_cast<std::size_t>(nb)]) continue;
                visited[static_cast<std::size_t>(nb)] = 1;

                const float d = evalDistance(metric, store.vector(nb), query, dims);
                if (found.size() < ef || d < found.top().first)
                {
                    candidates.emplace(d, nb);
                    found.emplace(d, nb);
                    if (found.size() > ef) found.pop();
                }
            }
        }

        std::vector<KnnHit> hits;
        hits.reserve(found.size());
        while (!found.empty())
        {
            hits.push_back(KnnHit{found.top().second, found.top().first});
            found.pop();
        }
        // 距离升序、并列按 id 升序
        std::sort(hits.begin(), hits.end(),
                  [](const KnnHit& a, const KnnHit& b)
                  {
                      return a.distance < b.distance ||
                             (a.distance == b.distance && a.id < b.id);
                  });
        return hits;
    }

    InsertPlan planInsertNode(const GraphStore& store, DistanceMetric metric,
                              const float* vec, std::size_t level,
                              std::size_t efConstruction)
    {
        if (vec == nullptr) throw ValueError("vec must not be null");
        InsertPlan plan;
        plan.level = level;

        const std::int64_t oldEntry = store.entryPoint();
        if (oldEntry == -1)  // 搜索时图是空的
        {
            plan.emptyGraph = true;
            return plan;
        }
        const std::size_t oldMaxLevel = store.maxLevel();

        std::int64_t ep = oldEntry;
        // 高层下降（ef=1），从旧 maxLevel 到 level+1
        for (std::size_t lc = oldMaxLevel; lc > level; lc--)
        {
            ep = searchLayer(store, metric, vec, ep, 1, lc)[0].id;
        }

        // 低层搜索，从 level 到 0，每层结果记进计划书
        const std::size_t top = std::min(level, oldMaxLevel);
        plan.found.resize(top + 1);
        for (std::size_t lc = top;; lc--)
        {
            plan.found[lc] = searchLayer(store, metric, vec, ep, efConstruction, lc);
            if (lc == 0) break;

            ep = plan.found[lc].empty() ? ep : plan.found[lc][0].id;
        }
        return plan;
    }

    std::int64_t applyInsertNode(GraphStore& store, DistanceMetric metric,
                                 const float* vec, const InsertPlan& plan,
                                 SelectStrategy strategy)
    {
        const std::size_t dims = store.dims();

        const PairDistance pairDist = [&](std::int64_t a, std::int64_t b)
        {
            return evalDistance(metric, store.vector(a), store.vector(b), dims);
        };

        const std::int64_t q = store.addNode(vec, plan.level);
        if (plan.emptyGraph) return q;

        for (std::size_t lc = plan.found.size(); lc-- > 0;)
        {
            const std::size_t cap = store.neighborCapacity(lc);
            const auto picked = selectNeighbors(plan.found[lc], cap, strategy, pairDist);
            for (const KnnHit& h : picked)
            {
                store.addLink(q, lc, h.id);
                auto theirs = store.neighbors(h.id, lc);
                std::vector<std::int64_t> list(theirs.begin(), theirs.end());
                list.push_back(q);
                if (list.size() > cap)
                {
                    std::vector<KnnHit> cand;
                    cand.reserve(list.size());
                    const float* hv = store.vector(h.id);
                    for (const std::int64_t nb : list)
                        cand.push_back({nb, evalDistance(metric, store.vector(nb), hv, dims)});
                    const auto kept = selectNeighbors(cand, cap, strategy, pairDist);
                    std::vector<std::int64_t> ids;
                    ids.reserve(kept.size());
                    for (const KnnHit& k : kept) ids.push_back(k.id);
                    store.setLinks(h.id, lc, ids);
                }
                else
                {
                    store.setLinks(h.id, lc, list);
                }
            }
        }
        return q;
    }

    std::int64_t insertNode(GraphStore& store, DistanceMetric metric,
                            const float* vec, std::size_t level,
                            std::size_t efConstruction, SelectStrategy strategy)
    {
        const InsertPlan plan =
            planInsertNode(store, metric, vec, level, efConstruction);
        return applyInsertNode(store, metric, vec, plan, strategy);
    }

    std::vector<KnnHit> searchKnn(const GraphStore& store,
                                  DistanceMetric metric, const float* query,
                                  std::size_t ef)
    {
        if (store.size() == 0) return {};
        std::int64_t ep = store.entryPoint();
        for (std::size_t lc = store.maxLevel(); lc >= 1; lc--)
        {
            std::vector<KnnHit> tmp = searchLayer(store, metric, query, ep, 1, lc);
            ep = tmp[0].id;
        }
        return searchLayer(store, metric, query, ep, ef, 0);
    }

    void buildIndex(GraphStore& store, DistanceMetric metric, const float* data,
                    std::size_t rows, LevelGenerator& levels,
                    std::size_t efConstruction, SelectStrategy strategy)
    {
        if (data == nullptr) throw ValueError("data must not be null");
        const std::size_t dims = store.dims();
        for (std::size_t i = 0; i < rows; i++)
        {
            insertNode(store, metric, data + i * dims, levels.nextLevel(),
                       efConstruction, strategy);
        }
    }

}
