#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>
#include "knn_hit.hpp"

namespace vortex
{
    enum class SelectStrategy
    {
        Naive,      // 距离升序直取前 M
        Heuristic,  // RNG 剪枝
    };

    using PairDistance = std::function<float(std::int64_t, std::int64_t)>;
    
    std::vector<KnnHit> selectNeighbors(
        const std::vector<KnnHit>& candidates, std::size_t M,
        SelectStrategy strategy = SelectStrategy::Naive,
        const PairDistance& pairDist = nullptr);

}
