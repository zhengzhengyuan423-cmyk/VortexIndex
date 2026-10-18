#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include "knn_hit.hpp"
#include "vortex/distance.hpp"

namespace vortex
{
    std::vector<KnnHit> bruteForceKnn(DistanceMetric metric, const float* data,
                                      std::size_t rows, std::size_t dims,
                                      const float* query, std::size_t k);

}
