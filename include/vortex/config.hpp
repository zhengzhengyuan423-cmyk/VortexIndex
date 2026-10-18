#pragma once

#include <cstddef>
#include <cstdint>
#include "vortex/errors.hpp"

namespace vortex
{
    enum class DistanceMetric
    {
        L2,            // 欧氏距离
        InnerProduct,  // 内积距离
        Cosine         // 余弦距离
    };

    struct IndexParams
    {
        std::size_t dims = 0;      // 向量维度
        std::size_t capacity = 0;  // 可存储向量数上限，必填
        int M = 16;                // 上层每节点最大邻居数（第 0 层为其 2 倍）
        int efConstruction = 200;  // 插入时的搜索束宽
        std::uint32_t seed = 0x9E3779B9u;  // 随机数种子，黄金分割数
        void validate() const;
    };

    inline void IndexParams::validate() const
    {
        if (dims == 0) throw ValueError("dims must be > 0");
        if (capacity == 0) throw ValueError("capacity must be > 0");
        if (M < 2 || M > 128) throw ValueError("M must be in [2, 128]");
        if (efConstruction < M) throw ValueError("efConstruction must be >= M");
    }

}
