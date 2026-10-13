#pragma once

#include <cstdint>
#include <functional>

namespace vortex 
{
    //业务筛选
    using IdFilter = std::function<bool(std::int64_t)>;
}
