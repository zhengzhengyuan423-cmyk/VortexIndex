#include "bruteforce.hpp"
#include <algorithm>
#include <queue>
#include <utility>
#include "vortex/errors.hpp"

namespace vortex
{
    using HeapItem = std::pair<float, std::int64_t>; //[距离，外部id]

    // 仿函数，先比距离，再比 id
    static bool closerFirst(const KnnHit& a, const KnnHit& b)
    {
        if (a.distance != b.distance) return a.distance < b.distance;
        return a.id < b.id;
    }

    std::vector<KnnHit> bruteForceKnn(DistanceMetric metric, const float* data,
                                      std::size_t rows, std::size_t dims,
                                      const float* query, std::size_t k)
    {
        if (query == nullptr) throw ValueError("query must not be null");
        if (k == 0 || rows == 0) return {};
        if (data == nullptr) throw ValueError("data must not be null");


        std::priority_queue<HeapItem> topK;
        for (std::size_t i = 0; i < rows; i++)
        {
            const float dist = evalDistance(metric, data + i * dims, query, dims);
            if (topK.size() < k)
            {
                topK.emplace(dist, static_cast<std::int64_t>(i));
            }
            else if (dist < topK.top().first)
            {
                topK.pop();
                topK.emplace(dist, static_cast<std::int64_t>(i));
            }
        }

        // 堆里出来是逆序，倒进 vector 再排成正序
        std::vector<KnnHit> hits;
        hits.reserve(topK.size());
        while (!topK.empty())
        {
            hits.push_back(KnnHit{topK.top().second, topK.top().first});
            topK.pop();
        }
        std::sort(hits.begin(), hits.end(), closerFirst);
        return hits;
    }

}
