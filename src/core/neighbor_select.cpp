#include "neighbor_select.hpp"
#include <algorithm>
#include "vortex/errors.hpp"

namespace vortex
{
    std::vector<KnnHit> selectNeighbors(const std::vector<KnnHit>& candidates,
                                        std::size_t M, SelectStrategy strategy,
                                        const PairDistance& pairDist)
    {
        std::vector<KnnHit> sorted = candidates;
        std::sort(sorted.begin(), sorted.end(),
                  [](const KnnHit& a, const KnnHit& b)
                  {
                      return a.distance < b.distance ||
                             (a.distance == b.distance && a.id < b.id);
                  });
        if (M == 0) return {};
        if (strategy == SelectStrategy::Naive)
        {
            if (sorted.size() > M) sorted.resize(M);
            return sorted;
        }

        if (!pairDist) throw ValueError("heuristic selection needs pairDist");

        std::vector<KnnHit> result;
        for (const KnnHit& c : sorted)
        {
            bool pruned = false;
            for (const KnnHit& nb : result)
            {
                if (pairDist(nb.id, c.id) < c.distance)
                {
                    pruned = true;
                    break;
                }
            }
            if (!pruned)
            {
                result.push_back(c);
                if (result.size() == M) break;
            }
        }

        // 回填：某个方向剪太狠名额用不满时，按距离升序补被剪掉的最近者，
        if (result.size() < M)
        {
            for (const KnnHit& c : sorted)
            {
                if (result.size() == M) break;
                bool already = false;
                for (const KnnHit& r : result)
                {
                    if (r.id == c.id)
                    {
                        already = true;
                        break;
                    }
                }
                if (!already) result.push_back(c);
            }
        }
        return result;
    }

}
