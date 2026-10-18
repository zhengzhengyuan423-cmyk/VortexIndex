#include "graph_store.hpp"
#include "vortex/errors.hpp"

namespace vortex
{
    GraphStore::GraphStore(std::size_t dims, std::size_t M,
                           std::size_t capacity)
        : dims_(dims), M_(M), capacity_(capacity)
    {
        if (dims == 0) throw ValueError("dims must be positive");
        if (M < 2) throw ValueError("M must be at least 2");
        nodes_.reserve(capacity);
    }

    std::int64_t GraphStore::addNode(const float* vec, std::size_t level)
    {
        if (vec == nullptr) throw ValueError("vec must not be null");
        if (nodes_.size() >= capacity_)
            throw CapacityError("graph store is full");

        Node n;
        n.vec.assign(vec, vec + dims_);
        n.links.resize(level + 1);

        const std::int64_t id = static_cast<std::int64_t>(nodes_.size());

        if (entry_ == -1 || level > maxLevel_)
        {
            if (level > maxLevel_) maxLevel_ = level;
            entry_ = id;
        }
        nodes_.push_back(std::move(n));
        return id;
    }

    std::size_t GraphStore::nodeLevel(std::int64_t id) const
    {
        if (id < 0 || static_cast<std::size_t>(id) >= nodes_.size())
            throw ValueError("node id out of range");
        return nodes_[static_cast<std::size_t>(id)].links.size() - 1;
    }

    const float* GraphStore::vector(std::int64_t id) const
    {
        if (id < 0 || static_cast<std::size_t>(id) >= nodes_.size())
            throw ValueError("node id out of range");
        return nodes_[static_cast<std::size_t>(id)].vec.data();
    }

    const GraphStore::Neighbors& GraphStore::neighbors(std::int64_t id,
                                                       std::size_t layer) const
    {
        if (id < 0 || static_cast<std::size_t>(id) >= nodes_.size())
            throw ValueError("node id out of range");
        const Node& n = nodes_[static_cast<std::size_t>(id)];
        if (layer > n.links.size() - 1)
            throw ValueError("layer above node level");
        return n.links[layer];
    }

    GraphStore::Neighbors& GraphStore::neighbors(std::int64_t id,
                                                 std::size_t layer)
    {
        const GraphStore& self = *this;
        return const_cast<Neighbors&>(self.neighbors(id, layer));
    }

    std::size_t GraphStore::neighborCapacity(std::size_t layer) const
    {
        return layer == 0 ? 2 * M_ : M_;
    }

    std::int64_t GraphStore::entryPoint() const { return entry_; }
    std::size_t GraphStore::maxLevel() const { return maxLevel_; }
    std::size_t GraphStore::size() const { return nodes_.size(); }
    std::size_t GraphStore::dims() const { return dims_; }
}
