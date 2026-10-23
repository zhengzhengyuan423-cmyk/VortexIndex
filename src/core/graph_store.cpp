#include "graph_store.hpp"
#include <algorithm>
#include <cmath>
#include "vortex/errors.hpp"

namespace vortex
{
    GraphStore::GraphStore(std::size_t dims, std::size_t M,
                           std::size_t capacity)
        : dims_(dims), M_(M), capacity_(capacity)
    {
        if (dims == 0) throw ValueError("dims must be positive");
        if (M < 2) throw ValueError("M must be at least 2");
        vecs_.resize(capacity * dims);
        levels_.reserve(capacity);
        linkBase_.reserve(capacity);
        const double avgSlots = 1.0 + 2.0 * M_ + (1.0 + M_) / std::log(M_);
        arena_.reserve(static_cast<std::size_t>(capacity * avgSlots * 1.25));
    }

    std::size_t GraphStore::blockOffset(std::size_t layer) const
    {
        // 第 0 层块在最前（1+2M 槽），其上每层 1+M 槽
        return layer == 0 ? 0 : (1 + 2 * M_) + (layer - 1) * (1 + M_);
    }

    std::size_t GraphStore::nodeSlots(std::size_t level) const
    {
        return (1 + 2 * M_) + level * (1 + M_);
    }

    
    std::int64_t GraphStore::addNode(const float* vec, std::size_t level)
    {
        if (vec == nullptr) throw ValueError("vec must not be null");
        if (n_ >= capacity_) throw CapacityError("graph store is full");

        const std::int64_t id = static_cast<std::int64_t>(n_);
        std::copy(vec, vec + dims_, vecs_.begin() + id * dims_);
        linkBase_.push_back(arena_.size());
        arena_.resize(arena_.size() + nodeSlots(level), 0);
        levels_.push_back(level);
        n_++;

        if (entry_ == -1 || level > maxLevel_)
        {
            if (level > maxLevel_) maxLevel_ = level;
            entry_ = id;
            }
        return id;
    }

    std::size_t GraphStore::blockStart(std::int64_t id, std::size_t layer) const
    {
        if (id < 0 || static_cast<std::size_t>(id) >= n_)
            throw ValueError("node id out of range");
        if (layer > levels_[static_cast<std::size_t>(id)])
            throw ValueError("layer above node level");
        return linkBase_[static_cast<std::size_t>(id)] + blockOffset(layer);
    }

    GraphStore::LinkView GraphStore::neighbors(std::int64_t id, std::size_t layer) const
    {
        const std::size_t b = blockStart(id, layer);
        return LinkView(&arena_[b], &arena_[b + 1]);
    }

    void GraphStore::addLink(std::int64_t id, std::size_t layer, std::int64_t nb)
    {
        const std::size_t b = blockStart(id, layer);
        const std::int64_t cnt = arena_[b];
        if (static_cast<std::size_t>(cnt) >= neighborCapacity(layer))
            throw CapacityError("neighbor list is full");
        arena_[b + 1 + cnt] = nb;
        arena_[b] = cnt + 1;
    }

    void GraphStore::setLinks(std::int64_t id, std::size_t layer, const std::vector<std::int64_t>& nbs)
    {
        const std::size_t b = blockStart(id, layer);
        if (nbs.size() > neighborCapacity(layer))
            throw CapacityError("links exceed capacity");
        for (std::size_t i = 0; i < nbs.size(); i++) arena_[b + 1 + i] = nbs[i];
        arena_[b] = static_cast<std::int64_t>(nbs.size());
    }

    std::size_t GraphStore::neighborCapacity(std::size_t layer) const
    {
        return layer == 0 ? 2 * M_ : M_;
    }

    std::size_t GraphStore::nodeLevel(std::int64_t id) const
    {
        if (id < 0 || static_cast<std::size_t>(id) >= n_)
            throw ValueError("node id out of range");
        return levels_[static_cast<std::size_t>(id)];
    }

    const float* GraphStore::vector(std::int64_t id) const
    {
        if (id < 0 || static_cast<std::size_t>(id) >= n_)
            throw ValueError("node id out of range");
        return vecs_.data() + static_cast<std::ptrdiff_t>(id) * dims_;
    }

    std::int64_t GraphStore::entryPoint() const { return entry_; }
    std::size_t GraphStore::maxLevel() const { return maxLevel_; }
    std::size_t GraphStore::size() const { return n_; }
    std::size_t GraphStore::dims() const { return dims_; }

}
