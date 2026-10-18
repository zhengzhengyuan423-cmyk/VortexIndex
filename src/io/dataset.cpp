#include "dataset.hpp"
#include <cstring>
#include <fstream>
#include "vortex/errors.hpp"

namespace vortex
{
    struct RawBytes
    {
        std::vector<char> data;
        std::size_t rows = 0;
        std::size_t dims = 0;
    };

    static RawBytes readVecsRaw(const std::string& path, const char* tag,
                                std::size_t elemSize, std::size_t maxRows)
    {
        std::ifstream in(path, std::ios::binary);  // 二进制文件打开
        if (!in) throw IOError("cannot open dataset file: " + path);

        RawBytes ds;

        while (maxRows == 0 || ds.rows < maxRows)
        {
            std::int32_t d = 0;
            in.read(reinterpret_cast<char*>(&d), sizeof(d));

            // in.gcount() 的返回类型是 std::streamsize（有符号整数）
            // 记录边界处的正常 EOF
            if (in.gcount() == 0) break;
            // 记录边界处的异常 EOF（不是 4）
            if (in.gcount() != static_cast<std::streamsize>(sizeof(d)))
                throw IOError(std::string("truncated dims field in ") + tag);
            if (d <= 0)
                throw IOError(std::string("non-positive dims in ") + tag +
                              " record");
            // 记录间维度不一致（和第一条不相等）
            if (ds.rows == 0)
                ds.dims = static_cast<std::size_t>(d);
            else if (static_cast<std::size_t>(d) != ds.dims)
                throw IOError(std::string("inconsistent dims across ") + tag +
                              " records");

            // 载荷直读进 buffer 尾部
            const std::streamsize payload =
                static_cast<std::streamsize>(ds.dims * elemSize);
            const std::size_t base = ds.data.size();
            ds.data.resize(base + static_cast<std::size_t>(payload));
            in.read(ds.data.data() + base, payload);
            if (in.gcount() != payload)
                throw IOError(std::string("truncated vector payload in ") +
                              tag);

            ds.rows++;
        }

        return ds;
    }

    Dataset readFvecs(const std::string& path, std::size_t maxRows)
    {
        RawBytes raw = readVecsRaw(path, "fvecs", sizeof(float), maxRows);
        Dataset ds;
        ds.rows = raw.rows;
        ds.dims = raw.dims;
        ds.data.resize(raw.data.size() / sizeof(float));
        std::memcpy(ds.data.data(), raw.data.data(), raw.data.size());
        return ds;
    }

    IntDataset readIvecs(const std::string& path, std::size_t maxRows)
    {
        RawBytes raw =
            readVecsRaw(path, "ivecs", sizeof(std::int32_t), maxRows);
        IntDataset ds;
        ds.rows = raw.rows;
        ds.dims = raw.dims;
        ds.data.resize(raw.data.size() / sizeof(std::int32_t));
        std::memcpy(ds.data.data(), raw.data.data(), raw.data.size());
        return ds;
    }

}
