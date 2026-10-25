#include "api/slot_book.hpp"

namespace vortex
{
    void SlotBook::record(std::size_t slot, std::int64_t id)
    {
        if (slot >= slotToExternal_.size())
        {
            slotToExternal_.resize(slot + 1, -1);
            tomb_.resize(slot + 1, 0);
        }
        slotToExternal_[slot] = id;
        tomb_[slot] = 0;
        extToSlots_[id].push_back(slot);
    }

    std::size_t SlotBook::removeId(std::int64_t id)
    {
        const auto it = extToSlots_.find(id);
        if (it == extToSlots_.end()) return 0;
        std::size_t flipped = 0;
        for (const std::size_t slot : it->second)
        {
            if (slot < tomb_.size() && !tomb_[slot])
            {
                tomb_[slot] = 1;
                ++flipped;
            }
        }
        tombNum_ += flipped;
        return flipped;
    }

    bool SlotBook::live(std::size_t slot) const
    {
        return slot < tomb_.size() && !tomb_[slot] && slotToExternal_[slot] >= 0;
    }

    std::int64_t SlotBook::externalId(std::size_t slot) const
    {
        if (slot >= slotToExternal_.size()) return -1;
        return slotToExternal_[slot];
    }

    bool SlotBook::tombFlag(std::size_t slot) const
    {
        return slot < tomb_.size() && tomb_[slot] != 0;
    }

    void SlotBook::recordTomb(std::size_t slot, std::int64_t id)
    {
        record(slot, id);
        if (!tomb_[slot])
        {
            tomb_[slot] = 1;
            tombNum_++;
        }
    }

    std::size_t SlotBook::slots() const
    {
        return slotToExternal_.size();
    }

    std::size_t SlotBook::liveCount() const
    {
        return slots() - tombstones();
    }

    std::size_t SlotBook::tombstones() const
    {
        return tombNum_;
    }

    bool SlotBook::crowded(double ratio) const
    {
        const std::size_t total = slots();
        if (total == 0) return false;
        return static_cast<double>(tombstones()) > ratio * static_cast<double>(total);
    }

    void SlotBook::reset()
    {
        slotToExternal_.clear();
        tomb_.clear();
        extToSlots_.clear();
        tombNum_ = 0;
    }

}
