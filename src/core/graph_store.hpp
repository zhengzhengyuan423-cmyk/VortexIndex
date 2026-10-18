#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace vortex
{
    class GraphStore
    {
        public:
        using Neighbors = std::vector<std::int64_t>;

        GraphStore(std::size_t dims, std::size_t M, std::size_t capacity);

        std::int64_t addNode(const float* vec, std::size_t level);

        std::size_t size() const;
        std::size_t dims() const;
        std::size_t nodeLevel(std::int64_t id) const;
        const float* vector(std::int64_t id) const;

        // 邻居表可变引用
        Neighbors& neighbors(std::int64_t id, std::size_t layer);
        const Neighbors& neighbors(std::int64_t id, std::size_t layer) const;
        // 邻居表容量 = M 或 M+1，取决于层级
        std::size_t neighborCapacity(std::size_t layer) const;

        // 入口点
        std::int64_t entryPoint() const;
        std::size_t maxLevel() const;

        private:
        struct Node
        {
            std::vector<float> vec;
            std::vector<std::vector<std::int64_t>>
                links;  // links[l] = 第 l 层邻居
        };

        std::size_t dims_;
        std::size_t M_;
        std::size_t capacity_;
        std::vector<Node> nodes_;
        std::int64_t entry_ = -1;
        std::size_t maxLevel_ = 0;
    };

}
