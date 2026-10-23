#pragma once

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <random>

namespace vortex
{
    class LevelGenerator
    {
    public:
        LevelGenerator(std::size_t M, std::uint32_t seed);
        std::size_t nextLevel();

    private:
        double mL_;                                   // 1 / ln(M)
        std::mt19937 rng_;                            // 伪随机数生成器
        std::uniform_real_distribution<double> dist_; // (0,1] 均匀分布
        std::mutex mu_;                               // 专用锁
    };

}
