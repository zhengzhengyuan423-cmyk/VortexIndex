#pragma once

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace vortex
{
    class SlotBook
    {
    public:
        // 新落笔的槽入册；同 id 重复 record 即追加一槽
        void record(std::size_t slot, std::int64_t id);
        // 把 id 名下所有活槽翻墓碑，返回本次翻旗数
        std::size_t removeId(std::int64_t id);
        // 槽是否活跃且在册
        bool live(std::size_t slot) const;
        // 槽的墓碑旗原值
        bool tombFlag(std::size_t slot) const;
        // 读盘重建专用：落账同时直接埋旗
        void recordTomb(std::size_t slot, std::int64_t id);
        // 槽对应的外部 id；越界返回 -1
        std::int64_t externalId(std::size_t slot) const;
        // 已落笔槽数（含墓碑），容量检查用它
        std::size_t slots() const;
        // 活跃槽数 = slots - tombstones
        std::size_t liveCount() const;
        std::size_t tombstones() const;
        // 墓碑占比是否超过 ratio
        bool crowded(double ratio) const;
        // 全清零，供压实前换血
        void reset();

    private:
        std::vector<std::int64_t> slotToExternal_;            // 槽 -> id
        std::vector<std::uint8_t> tomb_;                      // 槽 -> 墓碑旗
        std::unordered_map<std::int64_t, std::vector<std::size_t>> extToSlots_;  // id -> 它的全部槽
        std::size_t tombNum_ = 0;  // 旗子总数
    };

}
