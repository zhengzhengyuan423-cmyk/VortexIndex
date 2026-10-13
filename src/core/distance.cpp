#include "vortex/distance.hpp"
#include "vortex/errors.hpp"

namespace vortex
{

float evalDistance(DistanceMetric, const float*, const float*, std::size_t)
{
    throw NotImplementedError("evalDistance");
}

}
