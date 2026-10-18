#pragma once

#include <cstdint>

namespace vortex
{
    // 一个命中 = 内部 id + 到 query 的距离
    struct KnnHit
    {
        std::int64_t id;
        float distance;
    };

}
