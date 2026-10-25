#include "io/serializer.hpp"

#include <cstring>
#include <istream>
#include <iterator>
#include <limits>
#include <ostream>
#include <vector>

#include "vortex/errors.hpp"


namespace vortex
{
    static const std::uint64_t kMagic =
        (static_cast<std::uint64_t>('V') << 56) | (static_cast<std::uint64_t>('T') << 48) |
        (static_cast<std::uint64_t>('X') << 40) | static_cast<std::uint64_t>('1');
    static const std::uint32_t kFormatVersion = 1;
    static const std::size_t kHeaderBytes = 8 + 4 + 7 * 8;  
    // 且防恶意 header 的 capacity 撑爆内存
    static const std::uint64_t kMaxSlots = 1ull << 32;

    static void putU32(std::vector<std::uint8_t>& b, std::uint32_t v)
    {
        for (int i = 0; i < 4; i++) b.push_back(static_cast<std::uint8_t>(v >> (8 * i)));
    }
    static void putU64(std::vector<std::uint8_t>& b, std::uint64_t v)
    {
        for (int i = 0; i < 8; i++) b.push_back(static_cast<std::uint8_t>(v >> (8 * i)));
    }
    static void putF32(std::vector<std::uint8_t>& b, float f)
    {
        std::uint32_t u;
        std::memcpy(&u, &f, 4);
        putU32(b, u);
    }

    static std::uint32_t getU32(const std::vector<std::uint8_t>& b, std::size_t& pos)
    {
        if (pos + 4 > b.size()) throw IOError("load");
        std::uint32_t v = 0;
        for (int i = 0; i < 4; i++) v |= static_cast<std::uint32_t>(b[pos + i]) << (8 * i);
        pos += 4;
        return v;
    }
    static std::uint64_t getU64(const std::vector<std::uint8_t>& b, std::size_t& pos)
    {
        if (pos + 8 > b.size()) throw IOError("load");
        std::uint64_t v = 0;
        for (int i = 0; i < 8; i++) v |= static_cast<std::uint64_t>(b[pos + i]) << (8 * i);
        pos += 8;
        return v;
    }

    void serializeIndex(std::ostream& os, const GraphStore& store,
                        const SlotBook& book, const IndexParams& params)
    {
        const std::size_t count = store.size();
        if (count != book.slots()) throw IOError("save");  
        const std::size_t dims = store.dims();

        std::vector<std::uint8_t> out;
        out.reserve(kHeaderBytes + count * (dims * 4 + 10));  

        putU64(out, kMagic);
        putU32(out, kFormatVersion);
        putU64(out, dims);
        putU64(out, static_cast<std::uint64_t>(params.M));
        putU64(out, static_cast<std::uint64_t>(params.efConstruction));
        putU64(out, params.capacity);
        putU64(out, static_cast<std::uint64_t>(params.metric));
        putU64(out, static_cast<std::uint64_t>(params.seed));
        putU64(out, count);

        for (std::size_t s = 0; s < count; s++)
        {
            const float* v = store.vector(static_cast<std::int64_t>(s));
            for (std::size_t d = 0; d < dims; d++) putF32(out, v[d]);
        }
        for (std::size_t s = 0; s < count; s++)
        {
            const std::size_t lv = store.nodeLevel(static_cast<std::int64_t>(s));
            if (lv > 255) throw IOError("save");  // 层超 u8 段宽：这代格式养不起
            out.push_back(static_cast<std::uint8_t>(lv));
        }
        for (std::size_t s = 0; s < count; s++)
            out.push_back(book.tombFlag(s) ? 1 : 0);
        for (std::size_t s = 0; s < count; s++)
            putU64(out, static_cast<std::uint64_t>(book.externalId(s)));
        for (std::size_t s = 0; s < count; s++)
        {
            const std::size_t lv = store.nodeLevel(static_cast<std::int64_t>(s));
            putU32(out, static_cast<std::uint32_t>(lv) + 1);
            for (std::size_t layer = 0; layer <= lv; layer++)
            {
                const auto nb = store.neighbors(static_cast<std::int64_t>(s), layer);
                putU32(out, static_cast<std::uint32_t>(nb.size()));
                for (const std::int64_t x : nb)
                {
                    if (x < 0 || static_cast<std::uint64_t>(x) >= kMaxSlots)
                        throw IOError("save");
                    putU32(out, static_cast<std::uint32_t>(x));
                }
            }
        }

        os.write(reinterpret_cast<const char*>(out.data()),
                 static_cast<std::streamsize>(out.size()));
        os.flush();
        if (!os) throw IOError("save");
    }

    LoadedFile deserializeIndex(std::istream& is)
    {
        std::vector<std::uint8_t> f((std::istreambuf_iterator<char>(is)),
                                     std::istreambuf_iterator<char>());
        if (f.size() < kHeaderBytes) throw IOError("load");

        std::size_t pos = 0;
        if (getU64(f, pos) != kMagic) throw IOError("load");
        if (getU32(f, pos) != kFormatVersion) throw IOError("load");

        IndexParams params;
        const std::size_t dims = static_cast<std::size_t>(getU64(f, pos));
        const std::size_t m = static_cast<std::size_t>(getU64(f, pos));
        const std::size_t efc = static_cast<std::size_t>(getU64(f, pos));
        const std::size_t capacity = static_cast<std::size_t>(getU64(f, pos));
        const std::uint64_t metricRaw = getU64(f, pos);
        params.seed = static_cast<std::uint32_t>(getU64(f, pos));
        const std::size_t count = static_cast<std::size_t>(getU64(f, pos));
        // 越界的 enum 值转进去是未定义行为，先卡死再转
        if (metricRaw > static_cast<std::uint64_t>(DistanceMetric::Cosine))
            throw IOError("load");
        params.dims = dims;
        params.capacity = capacity;
        params.M = static_cast<int>(m);
        params.efConstruction = static_cast<int>(efc);
        params.metric = static_cast<DistanceMetric>(metricRaw);
        if (capacity > kMaxSlots) throw IOError("load");
        try { params.validate(); }
        catch (const VortexError&) { throw IOError("load"); }  // 参数被篡改＝文件问题

        GraphStore store(dims, m, params.capacity);
        SlotBook book;
        try
        {
            if (count > params.capacity) throw IOError("load");
            auto fits = [&](std::size_t n) { return n <= f.size() - pos; };

            const std::size_t vecBytes = count * dims * 4;
            if (!fits(vecBytes)) throw IOError("load");
            std::vector<float> vecs(count * dims);
            for (std::size_t i = 0; i < count * dims; i++)
            {
                const std::uint32_t u = getU32(f, pos);
                std::memcpy(&vecs[i], &u, 4);
            }
            if (!fits(count)) throw IOError("load");
            std::vector<std::uint8_t> levels(count);
            for (std::size_t s = 0; s < count; s++) levels[s] = f[pos++];
            if (!fits(count)) throw IOError("load");
            std::vector<std::uint8_t> tombs(count);
            for (std::size_t s = 0; s < count; s++) tombs[s] = f[pos++];
            if (!fits(count * 8)) throw IOError("load");
            std::vector<std::int64_t> ids(count);
            for (std::size_t s = 0; s < count; s++)
                ids[s] = static_cast<std::int64_t>(getU64(f, pos));

            for (std::size_t s = 0; s < count; s++)
                store.addNode(vecs.data() + s * dims, levels[s]);
            for (std::size_t s = 0; s < count; s++)
                if (tombs[s]) book.recordTomb(s, ids[s]); else book.record(s, ids[s]);

            for (std::size_t s = 0; s < count; s++)
            {
                const std::uint32_t nl = getU32(f, pos);
                if (nl != static_cast<std::uint32_t>(levels[s]) + 1) throw IOError("load");
                for (std::uint32_t layer = 0; layer < nl; layer++)
                {
                    const std::uint32_t n = getU32(f, pos);
                    if (n > store.neighborCapacity(layer)) throw IOError("load");
                    std::vector<std::int64_t> nbs;
                    nbs.reserve(n);
                    for (std::uint32_t i = 0; i < n; i++)
                    {
                        const std::uint32_t nb = getU32(f, pos);
                        if (nb >= count || nb == s) throw IOError("load");
                        nbs.push_back(static_cast<std::int64_t>(nb));
                    }
                    store.setLinks(static_cast<std::int64_t>(s), layer, nbs);
                }
            }
            if (pos != f.size()) throw IOError("load");  // 没吃完的字节＝长度谎报
        }
        catch (const VortexError&) { throw IOError("load"); }
        return LoadedFile{params, std::move(store), std::move(book)};
    }

}
