#pragma once

#include <cstddef>
#include "vortex/config.hpp"

namespace vortex
{
    float evalDistance(DistanceMetric metric, const float* a, const float* b,
                       std::size_t dims);
}
