#include "level_generator.hpp"
#include <cmath>
#include "vortex/errors.hpp"

namespace vortex
{
    LevelGenerator::LevelGenerator(std::size_t M, std::uint32_t seed)
        : mL_(1.0 / std::log(static_cast<double>(M))),
          rng_(seed),
          dist_(std::nextafter(0.0, 1.0), 1.0)
    {
        if (M < 2) throw ValueError("M must be at least 2");
    }

    std::size_t LevelGenerator::nextLevel()
    {
        const double u = dist_(rng_);
        return static_cast<std::size_t>(std::floor(-std::log(u) * mL_));
    }
}
