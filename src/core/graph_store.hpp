#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace vortex
{
    class GraphStore
    {
    public:
        class LinkView
        {
        public:
            std::size_t size() const { return static_cast<std::size_t>(*count_); }
            bool empty() const { return *count_ == 0; }
            std::int64_t operator[](std::size_t i) const { return data_[i]; }
            const std::int64_t* begin() const { return data_; }
            const std::int64_t* end() const { return data_ + size(); }

            private:
            LinkView(const std::int64_t* count, const std::int64_t* data)
                : count_(count), data_(data)
            { }
            friend class GraphStore;

            const std::int64_t* count_;
            const std::int64_t* data_;
        };

        GraphStore(std::size_t dims, std::size_t M, std::size_t capacity);

        
        std::int64_t addNode(const float* vec, std::size_t level);

        std::size_t size() const;
        std::size_t dims() const;
        std::size_t nodeLevel(std::int64_t id) const;
        const float* vector(std::int64_t id) const;

        LinkView neighbors(std::int64_t id, std::size_t layer) const;
        // 追加一条边
        void addLink(std::int64_t id, std::size_t layer, std::int64_t nb);
        // 整表覆写（重剪枝回填用）
        void setLinks(std::int64_t id, std::size_t layer,
                      const std::vector<std::int64_t>& nbs);

        // 邻居表容量：第 0 层 2M，其余层 M
        std::size_t neighborCapacity(std::size_t layer) const;

        // 入口点
        std::int64_t entryPoint() const;
        std::size_t maxLevel() const;

    private:
        // 校验 id/layer 并返回该层块在 arena_ 的起始槽
        std::size_t blockStart(std::int64_t id, std::size_t layer) const;
        // 层 layer 的块起点相对节点块基址的偏移
        std::size_t blockOffset(std::size_t layer) const;
        // level 层节点的总槽数
        std::size_t nodeSlots(std::size_t level) const;

        std::size_t dims_;
        std::size_t M_;
        std::size_t capacity_;
        std::size_t n_ = 0;
        std::vector<float> vecs_;          // capacity*dims 预分配，行即向量
        std::vector<std::size_t> levels_;  // 每节点最高层
        std::vector<std::size_t> linkBase_;  // 每节点块在 arena_ 的起始槽
        std::vector<std::int64_t> arena_;
        std::int64_t entry_ = -1;
        std::size_t maxLevel_ = 0;
    };

}
