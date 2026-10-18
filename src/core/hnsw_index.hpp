#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include "graph_store.hpp"
#include "knn_hit.hpp"
#include "level_generator.hpp"
#include "neighbor_select.hpp"
#include "vortex/distance.hpp"

namespace vortex
{
    std::vector<KnnHit> searchLayer(const GraphStore& store,
                                    DistanceMetric metric, const float* query,
                                    std::int64_t entry, std::size_t ef,
                                    std::size_t layer);


    std::int64_t insertNode(GraphStore& store, DistanceMetric metric,
                            const float* vec, std::size_t level,
                            std::size_t efConstruction,
                            SelectStrategy strategy = SelectStrategy::Naive);


    std::vector<KnnHit> searchKnn(const GraphStore& store,
                                  DistanceMetric metric, const float* query,
                                  std::size_t ef);


    void buildIndex(GraphStore& store, DistanceMetric metric, const float* data,
                    std::size_t rows, LevelGenerator& levels,
                    std::size_t efConstruction,
                    SelectStrategy strategy = SelectStrategy::Naive);

}
