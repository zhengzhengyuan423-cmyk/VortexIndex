#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace vortex
{
    // 浮点数据集
    struct Dataset
    {
        std::vector<float> data;  // rows * dims 个 float
        std::size_t rows = 0;
        std::size_t dims = 0;  // rows == 0 时为 0

        // 第 i 行的起始地址
        const float* row(std::size_t i) const 
        { 
            return data.data() + i * dims; 
        }
    };

    // 整数数据集
    struct IntDataset
    {
        std::vector<std::int32_t> data;
        std::size_t rows = 0;
        std::size_t dims = 0;

        const std::int32_t* row(std::size_t i) const
        {
            return data.data() + i * dims;
        }
    };

    Dataset readFvecs(const std::string& path, std::size_t maxRows = 0);
    IntDataset readIvecs(const std::string& path, std::size_t maxRows = 0);

}
