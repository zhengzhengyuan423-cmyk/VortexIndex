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



    struct InsertPlan
    {
        std::size_t level = 0;
        // found[lc] = 在 lc 层宽度为 efConstruction 的搜索结果
        std::vector<std::vector<KnnHit>> found;
        // 如果搜索时图是空的，第二步只需把点放进存储，没有任何边可连
        bool emptyGraph = false;
    };

    // store 必须在此期间不被改动
    InsertPlan planInsertNode(const GraphStore& store, DistanceMetric metric,
                              const float* vec, std::size_t level,
                              std::size_t efConstruction);

    // 按计划书写存储
    std::int64_t applyInsertNode(GraphStore& store, DistanceMetric metric,
                                 const float* vec, const InsertPlan& plan,
                                 SelectStrategy strategy);

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
