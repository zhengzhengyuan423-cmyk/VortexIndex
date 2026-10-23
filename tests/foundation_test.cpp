// 基础层测试：distance / dataset
#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
#include "io/dataset.hpp"
#include "vortex/distance.hpp"
#include "vortex/errors.hpp"

namespace vortex
{
    TEST(Distance, L2)
    {
        std::vector<float> a{1.f, 2.f}, b{4.f, 6.f};
        // (1-4)² + (2-6)² = 25
        EXPECT_NEAR(evalDistance(DistanceMetric::L2, a.data(), b.data(), 2), 25.f, 1e-4);
        std::vector<float> c{-1.f, 2.f}, d{3.f, -2.f};
        // (-1-3)² + (2-(-2))² = 32
        EXPECT_NEAR(evalDistance(DistanceMetric::L2, c.data(), d.data(), 2), 32.f, 1e-4);
        EXPECT_NEAR(evalDistance(DistanceMetric::L2, a.data(), a.data(), 2), 0.f, 1e-4);
        const float* nul = nullptr;
        EXPECT_THROW(evalDistance(DistanceMetric::L2, nul, a.data(), 2), ValueError);
    }

    TEST(Distance, InnerProduct)
    {
        std::vector<float> a{1.f, 2.f}, b{4.f, 6.f};
        // dot = 1*4 + 2*6 = 16 -> 1 - 16 = -15
        EXPECT_NEAR(evalDistance(DistanceMetric::InnerProduct, a.data(), b.data(), 2),-15.f, 1e-4);
        std::vector<float> c{1.f, 2.f}, d{-1.f, 2.f};
        // dot = -1 + 4 = 3 -> 1 - 3 = -2
        EXPECT_NEAR(evalDistance(DistanceMetric::InnerProduct, c.data(), d.data(), 2), -2.f, 1e-4);
    }

    TEST(Distance, Cosine)
    {
        std::vector<float> x{1.f, 0.f}, y{0.f, 1.f};
        // 正交：cos = 0 -> 距离 1
        EXPECT_NEAR(evalDistance(DistanceMetric::Cosine, x.data(), y.data(), 2), 1.f, 1e-4);
        // 零向量无方向，必须抛
        std::vector<float> z{0.f, 0.f};
        EXPECT_THROW(evalDistance(DistanceMetric::Cosine, z.data(), x.data(), 2), ValueError);
    }

    // dataset测试
    static std::filesystem::path tempPath(const std::string& name)
    {
        return std::filesystem::temp_directory_path() / ("vortex_test_" + name);
    }

    static void writeFvecsRaw(const std::filesystem::path& p,
                              const std::vector<std::vector<float>>& rows)
    {
        std::ofstream out(p, std::ios::binary);
        for (const auto& r : rows)
        {
            std::int32_t d = static_cast<std::int32_t>(r.size());
            out.write(reinterpret_cast<const char*>(&d), sizeof(d));
            out.write(reinterpret_cast<const char*>(r.data()),
                      static_cast<std::streamsize>(r.size() * sizeof(float)));
        }
    }

    static void writeIvecsRaw(
        const std::filesystem::path& p,
        const std::vector<std::vector<std::int32_t>>& rows)
    {
        std::ofstream out(p, std::ios::binary);
        for (const auto& r : rows)
        {
            std::int32_t d = static_cast<std::int32_t>(r.size());
            out.write(reinterpret_cast<const char*>(&d), sizeof(d));
            out.write(
                reinterpret_cast<const char*>(r.data()),
                static_cast<std::streamsize>(r.size() * sizeof(std::int32_t)));
        }
    }

    class DatasetTest : public ::testing::Test
    {
        protected:
        void TearDown() override
        {
            for (const auto& p : paths_) std::filesystem::remove(p);
        }
        std::filesystem::path makePath(const std::string& name)
        {
            paths_.push_back(tempPath(name));
            return paths_.back();
        }

        private:
        std::vector<std::filesystem::path> paths_;
    };

    // 验证 .fvecs 文件的读写往返正确
    TEST_F(DatasetTest, FvecRoundTrip)
    {
        auto p = makePath("roundtrip.fvecs");
        std::vector<std::vector<float>> rows{{1.f, 2.f, 3.f},
                                             {-4.f, 5.f, 0.f},
                                             {0.f, 0.f, 0.f},
                                             {7.f, 8.f, 9.f},
                                             {-1.f, -2.f, -3.f}};
        writeFvecsRaw(p, rows);

        Dataset ds = readFvecs(p.string());
        ASSERT_EQ(ds.rows, 5u);
        ASSERT_EQ(ds.dims, 3u);
        for (std::size_t i = 0; i < rows.size(); i++)
            for (std::size_t j = 0; j < 3; j++)
                EXPECT_EQ(ds.data[i * 3 + j], rows[i][j]);

        // maxRows 截断读取
        Dataset few = readFvecs(p.string(), 2);
        EXPECT_EQ(few.rows, 2u);
        EXPECT_EQ(few.data.size(), 6u);
    }

    // 验证 .fvecs 文件的格式错误处理
    TEST_F(DatasetTest, BadFormatThrows)
    {
        // 声明 3 维却只写 2 个 float：payload 截断
        auto p = makePath("trunc.fvecs");
        std::ofstream out(p, std::ios::binary);
        std::int32_t d = 3;
        out.write(reinterpret_cast<const char*>(&d), sizeof(d));
        float half[2] = {1.f, 2.f};
        out.write(reinterpret_cast<const char*>(half), sizeof(half));
        out.close();
        EXPECT_THROW(readFvecs(p.string()), IOError);

        // 第二条 dims=2 != 首条 3：维度不一致
        auto q = makePath("inconsist.fvecs");
        writeFvecsRaw(q, {{1.f, 2.f, 3.f}, {4.f, 5.f}});
        EXPECT_THROW(readFvecs(q.string()), IOError);
    }

    // 验证 .ivecs 文件的读写往返正确
    TEST_F(DatasetTest, IvecRoundTrip)
    {
        auto p = makePath("roundtrip.ivecs");
        std::vector<std::vector<std::int32_t>> rows{{7, 0, -1}, {2, 3, 4}};
        writeIvecsRaw(p, rows);

        IntDataset ds = readIvecs(p.string());
        ASSERT_EQ(ds.rows, 2u);
        ASSERT_EQ(ds.dims, 3u);
        EXPECT_EQ(ds.row(0)[0], 7);
        EXPECT_EQ(ds.row(0)[2], -1);
        EXPECT_EQ(ds.row(1)[1], 3);

        // 截断的 ivecs 同样必须抛
        auto t = makePath("trunc.ivecs");
        std::ofstream out(t, std::ios::binary);
        std::int32_t d = 3;
        out.write(reinterpret_cast<const char*>(&d), sizeof(d));
        std::int32_t half[2] = {1, 2};
        out.write(reinterpret_cast<const char*>(half), sizeof(half));
        out.close();
        EXPECT_THROW(readIvecs(t.string()), IOError);
    }

}
