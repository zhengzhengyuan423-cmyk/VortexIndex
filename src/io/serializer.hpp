#pragma once

#include <cstddef>
#include <cstdint>
#include <iosfwd>

#include "api/slot_book.hpp"
#include "core/graph_store.hpp"
#include "vortex/config.hpp"

//   header 68B：magic u64 | version u32 | dims | M | efConstruction |
//               capacity | metric | seed | count（各 u64，count=槽数含墓碑）
//   body：向量区 count×dims×f32 | 层级 u8/槽 | 墓碑旗 u8/槽 |
//         外部 id i64/槽 | 邻接表 逐槽[u32 层数=level+1]{u32 个数 + 槽号 u32×n}
namespace vortex
{

    struct LoadedFile
    {
        IndexParams params;
        GraphStore store;
        SlotBook book;
    };

    void serializeIndex(std::ostream& os, const GraphStore& store, const SlotBook& book, const IndexParams& params);

    LoadedFile deserializeIndex(std::istream& is);

}
