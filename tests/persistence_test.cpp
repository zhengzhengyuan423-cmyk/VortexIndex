// save/load 往返一致、坏档拒读、墓碑随档走

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include "vortex/errors.hpp"
#include "vortex/index.hpp"

namespace vortex
{
    static IndexParams persistParams(std::size_t dims, std::size_t cap)
    {
        IndexParams p;
        p.dims = dims;
        p.capacity = cap;
        p.M = 8;
        p.efConstruction = 64;
        p.seed = 20261024u;
        return p;
    }

    // 固定种子的伪随机点：测试可复现
    static std::vector<float> makePoints(std::size_t n, std::size_t dims, std::uint64_t seed)
    {
        std::mt19937_64 rng(seed);
        std::uniform_real_distribution<float> unit(0.f, 1.f);
        std::vector<float> v(n * dims);
        for (auto& x : v) x = unit(rng);
        return v;
    }

    static std::string tmpPath(const char* tag)
    {
        return std::string("vortex_persist_") + tag + ".bin";
    }

    struct CleanFile
    {
        std::string path;
        explicit CleanFile(const std::string& p) : path(p) { std::filesystem::remove(p); }
        ~CleanFile() { std::filesystem::remove(path); }
    };

    static std::vector<std::uint8_t> readBytes(const std::string& p)
    {
        std::ifstream in(p, std::ios::binary);
        if (!in) throw IOError("readBytes");
        return std::vector<std::uint8_t>((std::istreambuf_iterator<char>(in)),
                                         std::istreambuf_iterator<char>());
    }

    static void writeBytes(const std::string& p, const std::vector<std::uint8_t>& b)
    {
        std::ofstream out(p, std::ios::binary | std::ios::trunc);
        out.write(reinterpret_cast<const char*>(b.data()), static_cast<std::streamsize>(b.size()));
        out.close();
        if (!out) throw IOError("writeBytes");
    }

    // 灌 n 个点的公共底座
    static void fill(Index& idx, const std::vector<float>& pts, std::size_t n)
    {
        std::vector<std::int64_t> ids(n);
        for (std::size_t i = 0; i < n; i++) ids[i] = static_cast<std::int64_t>(i);
        idx.addPoints(pts.data(), ids.data(), n, 1);
    }

    // 一批查询的答案全集（save 前的参照）
    static std::vector<std::vector<SearchResult>> runQueries(
        const Index& idx, const std::vector<float>& qs, std::size_t dims)
    {
        std::vector<std::vector<SearchResult>> all;
        const std::size_t n = qs.size() / dims;
        for (std::size_t q = 0; q < n; q++)
            all.push_back(idx.searchKnn(qs.data() + q * dims, 10, 50));
        return all;
    }

    // 1. 往返逐位一致：load 后同批查询与 save 前相等
    TEST(PersistenceTest, RoundTripIdentical)
    {
        const CleanFile cf{tmpPath("roundtrip")};
        const std::size_t dims = 3, n = 60;
        const auto pts = makePoints(n, dims, 11u);
        const auto qs = makePoints(20, dims, 12u);
        Index idx(persistParams(dims, 100));
        fill(idx, pts, n);
        const auto before = runQueries(idx, qs, dims);
        idx.save(cf.path);
        Index ld = Index::load(cf.path);
        EXPECT_EQ(ld.size(), n);
        EXPECT_EQ(ld.deletedCount(), 0u);
        const auto after = runQueries(ld, qs, dims);
        ASSERT_EQ(after.size(), before.size());
        for (std::size_t q = 0; q < before.size(); q++) EXPECT_EQ(after[q], before[q]);
    }

    // 2. 字节稳定：load→再 save，两文件逐字节相同
    TEST(PersistenceTest, SecondSaveByteIdentical)
    {
        const CleanFile cf1{tmpPath("stable1")};
        const CleanFile cf2{tmpPath("stable2")};
        const std::size_t dims = 3, n = 50;
        const auto pts = makePoints(n, dims, 21u);
        Index idx(persistParams(dims, 80));
        fill(idx, pts, n);
        idx.save(cf1.path);
        Index ld = Index::load(cf1.path);
        ld.save(cf2.path);
        EXPECT_EQ(readBytes(cf1.path), readBytes(cf2.path));
    }

    // 3. 残缺与改版必拒
    TEST(PersistenceTest, TruncatedAndWrongVersionRejected)
    {
        const CleanFile cf{tmpPath("bad")};
        const std::size_t dims = 3, n = 40;
        const auto pts = makePoints(n, dims, 31u);
        Index idx(persistParams(dims, 60));
        fill(idx, pts, n);
        idx.save(cf.path);
        const auto good = readBytes(cf.path);
        ASSERT_GT(good.size(), 68u);

        auto cut = good; cut.resize(good.size() / 2);          // body 吃一半：边界校验拦
        writeBytes(cf.path, cut);
        EXPECT_THROW(Index::load(cf.path), IOError);

        auto stub = good; stub.resize(40);                     // 连 header 都不完整
        writeBytes(cf.path, stub);
        EXPECT_THROW(Index::load(cf.path), IOError);

        auto ver = good; ver[8] = 2;                           // version u32 低字节 1->2
        writeBytes(cf.path, ver);
        EXPECT_THROW(Index::load(cf.path), IOError);
    }

    // 5. 墓碑随档走：删一半→save→load
    TEST(PersistenceTest, TombstoneRoundTrip)
    {
        const CleanFile cf{tmpPath("tomb")};
        const std::size_t dims = 3, n = 30;
        const auto pts = makePoints(n, dims, 51u);
        const auto qs = makePoints(10, dims, 52u);
        Index idx(persistParams(dims, 50));
        fill(idx, pts, n);
        std::vector<std::int64_t> evens;
        for (std::int64_t i = 0; i < 30; i += 2) evens.push_back(i);
        EXPECT_EQ(idx.removePoint(evens.data(), evens.size()), 15u);
        const auto before = runQueries(idx, qs, dims);
        idx.save(cf.path);
        Index ld = Index::load(cf.path);
        EXPECT_EQ(ld.size(), 15u);
        EXPECT_EQ(ld.deletedCount(), 15u);
        const auto after = runQueries(ld, qs, dims);
        for (std::size_t q = 0; q < before.size(); q++) EXPECT_EQ(after[q], before[q]);
        // 墓碑继续占槽：load 后按 30 槽（含 15 尸）起算，插到满 50 才抛
        const float nv[] = {0.5f, 0.5f, 0.5f};
        std::vector<std::int64_t> extra;
        for (std::int64_t i = 100; i < 100 + 20; i++) extra.push_back(i);
        std::vector<float> block(20 * dims);
        for (std::size_t i = 0; i < block.size(); i++) block[i] = 0.3f;
        EXPECT_NO_THROW(ld.addPoints(block.data(), extra.data(), 20, 1));  // 30+20=50 恰满
        const std::int64_t over = 200;
        EXPECT_THROW(ld.addPoints(nv, &over, 1, 1), CapacityError);        // 第 51 槽挤不进
        EXPECT_EQ(ld.deletedCount(), 15u);                                 // 尸位没被插活
    }

    // 6. 空图往返：0 点也能存读，load 后照常插入
    TEST(PersistenceTest, EmptyIndexRoundTrip)
    {
        const CleanFile cf{tmpPath("empty")};
        Index idx(persistParams(3, 10));
        idx.save(cf.path);
        Index ld = Index::load(cf.path);
        EXPECT_EQ(ld.size(), 0u);
        EXPECT_EQ(ld.deletedCount(), 0u);
        const float q[] = {0.f, 0.f, 0.f};
        EXPECT_TRUE(ld.searchKnn(q, 5, 10).empty());
        const float v[] = {0.1f, 0.2f, 0.3f};
        const std::int64_t id = 7;
        EXPECT_NO_THROW(ld.addPoints(v, &id, 1));
        EXPECT_EQ(ld.size(), 1u);
    }

    // 7. load 后继续活：load→再插→再 save→再 load→查询仍逐位一致
    TEST(PersistenceTest, LoadThenKeepLiving)
    {
        const CleanFile cf1{tmpPath("live1")};
        const CleanFile cf2{tmpPath("live2")};
        const std::size_t dims = 3;
        const auto pts = makePoints(40, dims, 71u);
        const auto more = makePoints(20, dims, 72u);
        const auto qs = makePoints(10, dims, 73u);
        Index idx(persistParams(dims, 100));
        fill(idx, pts, 40);
        idx.save(cf1.path);
        Index ld1 = Index::load(cf1.path);
        fill(ld1, more, 20);
        const auto before = runQueries(ld1, qs, dims);
        ld1.save(cf2.path);
        Index ld2 = Index::load(cf2.path);
        EXPECT_EQ(ld2.size(), 60u);
        const auto after = runQueries(ld2, qs, dims);
        for (std::size_t q = 0; q < before.size(); q++) EXPECT_EQ(after[q], before[q]);
    }

    // 8. 确定性验收：同 seed 同数据独立建两次图 → save 产物字节全等。
    // 边界写死：保证只属于串行构建（threads=1）。并行建图的落笔顺序跟着
    // 线程完成顺序走，跨运行逐位不可复现，对它只能
    // 要求质量区间，不能要求文件哈希
    TEST(PersistenceTest, IndependentBuildsByteIdentical)
    {
        const CleanFile cfA{tmpPath("detA")};
        const CleanFile cfB{tmpPath("detB")};
        const std::size_t dims = 3, n = 50;
        const auto pts = makePoints(n, dims, 81u);
        Index a(persistParams(dims, 70)), b(persistParams(dims, 70));
        fill(a, pts, n);
        fill(b, pts, n);
        a.save(cfA.path);
        b.save(cfB.path);
        EXPECT_EQ(readBytes(cfA.path), readBytes(cfB.path));
    }

}
