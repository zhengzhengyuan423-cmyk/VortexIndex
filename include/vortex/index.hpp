#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include "vortex/config.hpp"
#include "vortex/filter.hpp"

namespace vortex
{

    struct SearchResult
    {
        std::int64_t id;
        float distance;
    };

    // 测试用
    inline bool operator==(const SearchResult& a, const SearchResult& b)
    {
        return a.id == b.id && a.distance == b.distance;
    }
    inline bool operator!=(const SearchResult& a, const SearchResult& b)
    {
        return !(a == b);
    }

    class Index
    {
    public:
        explicit Index(const IndexParams& params);  // 参数非法抛 ValueError
        ~Index();
        Index(Index&&) noexcept;
        Index& operator=(Index&&) noexcept;
        Index(const Index&) = delete;
        Index& operator=(const Index&) = delete;

        // 插入点，返回 id
        void addPoints(const float* data, const std::int64_t* ids, std::size_t rows, std::size_t threads = 1);

        // 查询 k 个最近邻，返回结果按距离升序排列。
        std::vector<SearchResult> searchKnn
        (const float* query, int k, int ef, const IdFilter& filter = IdFilter{}) const;

        std::size_t size() const;          // 活跃向量数
        std::size_t deletedCount() const;  // 墓碑槽位数

        void save(const std::string& path) const;    // 保存到磁盘
        static Index load(const std::string& path);  // 从磁盘加载

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

}
