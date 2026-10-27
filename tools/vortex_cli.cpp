// vortex-cli：build / query / bench 三个子命令的单文件命令行前台。
// 无交互模式、无进度条：每条命令一口气跑完，正常走 stdout，报错走 stderr 退出码 1。
// 用法：
//   vortex-cli build <数据集目录> <输出.bin> [threads] [M] [efC]
//   vortex-cli query <索引.bin> <数据集目录> <第几行> [k] [ef]
//   vortex-cli bench <数据集目录> [threads] [k] [ef]
// 数据集目录里按后缀找文件：*_base.fvecs / *_query.fvecs / *_groundtruth.ivecs
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <functional>
#include <string>
#include <vector>
#include "io/dataset.hpp"
#include "vortex/errors.hpp"
#include "vortex/index.hpp"

namespace
{
    // 第 i 个可选数字参数，缺省 def
    std::size_t num(int argc, char** argv, int i, std::size_t def)
    {
        return argc > i ? static_cast<std::size_t>(std::stoull(argv[i])) : def;
    }

    // 扫目录找以 tail 结尾的文件，返回全路径；找不到抛 IOError。
    std::string locate(const std::string& dir, const std::string& tail)
    {
        for (const auto& entry : std::filesystem::directory_iterator(dir))
        {
            const std::string name = entry.path().filename().string();
            if (name.size() >= tail.size() &&
                name.compare(name.size() - tail.size(), tail.size(), tail) == 0)
                return entry.path().string();
        }
        throw vortex::IOError("locate");
    }


    std::string stamp()
    {
        const std::time_t t = std::time(nullptr);
        std::tm tm{};
        localtime_r(&t, &tm);
        char buf[32];
        std::strftime(buf, sizeof buf, "%Y-%m-%d %H:%M:%S", &tm);
        return buf;
    }

    int showHelp(int, char**);

    int cmdBuild(int argc, char** argv)
    {
        if (argc < 4) return showHelp(argc, argv);
        const std::string dir = argv[2], out = argv[3];
        const std::size_t threads = num(argc, argv, 4, 1);
        const std::size_t M = num(argc, argv, 5, 16);
        const std::size_t efC = num(argc, argv, 6, 200);

        const vortex::Dataset base = vortex::readFvecs(locate(dir, "_base.fvecs"));
        vortex::IndexParams p;
        p.dims = base.dims;
        p.capacity = base.rows;
        p.M = static_cast<int>(M);
        p.efConstruction = static_cast<int>(efC);

        std::vector<std::int64_t> ids(base.rows);
        for (std::size_t i = 0; i < base.rows; i++) ids[i] = static_cast<std::int64_t>(i);

        const auto t0 = std::chrono::steady_clock::now();
        vortex::Index idx(p);
        idx.addPoints(base.data.data(), ids.data(), base.rows, threads);
        idx.save(out);
        const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

        std::printf("built rows=%zu dims=%zu M=%zu efC=%zu threads=%zu %.2fs -> %s\n",
                    base.rows, base.dims, M, efC, threads, sec, out.c_str());
        return 0;
    }

    int cmdQuery(int argc, char** argv)
    {
        if (argc < 5) return showHelp(argc, argv);
        const std::string bin = argv[2], dir = argv[3];
        const std::size_t row = num(argc, argv, 4, 0);
        const std::size_t k = num(argc, argv, 5, 10);
        const std::size_t ef = num(argc, argv, 6, std::max<std::size_t>(k, 64));

        const vortex::Dataset qs = vortex::readFvecs(locate(dir, "_query.fvecs"));
        if (row >= qs.rows)
        {
            std::fprintf(stderr, "vortex-cli: query row %zu out of range (%zu rows)\n", row, qs.rows);
            return 1;
        }
        vortex::Index idx = vortex::Index::load(bin);
        for (const vortex::SearchResult& h :
             idx.searchKnn(qs.row(row), static_cast<int>(k), static_cast<int>(ef)))
            std::printf("id=%lld dist=%.4f\n", static_cast<long long>(h.id), h.distance);
        return 0;
    }

    int cmdBench(int argc, char** argv)
    {
        if (argc < 3) return showHelp(argc, argv);
        const std::string dir = argv[2];
        const std::size_t threads = num(argc, argv, 3, 1);
        const std::size_t k = num(argc, argv, 4, 10);
        const std::size_t ef = num(argc, argv, 5, std::max<std::size_t>(k, 64));

        const vortex::Dataset base = vortex::readFvecs(locate(dir, "_base.fvecs"));
        const vortex::Dataset qs = vortex::readFvecs(locate(dir, "_query.fvecs"));
        const vortex::IntDataset gt = vortex::readIvecs(locate(dir, "_groundtruth.ivecs"));
        if (gt.rows < qs.rows || gt.dims < k)
        {
            std::fprintf(stderr, "vortex-cli: ground truth too small\n");
            return 1;
        }

        std::vector<std::int64_t> ids(base.rows);
        for (std::size_t i = 0; i < base.rows; i++) ids[i] = static_cast<std::int64_t>(i);
        vortex::IndexParams p;
        p.dims = base.dims;
        p.capacity = base.rows;

        const auto tb0 = std::chrono::steady_clock::now();
        vortex::Index idx(p);
        idx.addPoints(base.data.data(), ids.data(), base.rows, threads);
        const double buildSec = std::chrono::duration<double>(std::chrono::steady_clock::now() - tb0).count();

        double hit = 0;
        const auto tq0 = std::chrono::steady_clock::now();
        for (std::size_t q = 0; q < qs.rows; q++)
        {
            const auto res = idx.searchKnn(qs.row(q), static_cast<int>(k), static_cast<int>(ef));
            // recall@k：命中与真值前 k 名求交集。两边都换成排序 id 再数交集，
            // 不依赖次序（结果按距离序、真值也按距离序，但去序比更稳）
            std::vector<std::int64_t> ours;
            for (const vortex::SearchResult& h : res) ours.push_back(h.id);
            std::sort(ours.begin(), ours.end());
            std::vector<std::int64_t> truth(gt.row(q), gt.row(q) + k);
            std::sort(truth.begin(), truth.end());
            std::vector<std::int64_t> inter;
            std::set_intersection(ours.begin(), ours.end(), truth.begin(), truth.end(),
                                  std::back_inserter(inter));
            hit += static_cast<double>(inter.size()) / static_cast<double>(k);
        }
        const double qsec = std::chrono::duration<double>(std::chrono::steady_clock::now() - tq0).count();
        const double qps = static_cast<double>(qs.rows) / qsec;

        // 和 bench/results/*.jsonl 同一套字段名，脚本可以通吃两种来源
        std::printf("{\"time\":\"%s\",\"method\":\"cli\",\"threads\":%zu,"
                    "\"recall_at_k\":%.4f,\"qps\":%.1f,\"avg_latency_ms\":%.3f,"
                    "\"build_seconds\":%.1f}\n",
                    stamp().c_str(), threads, hit / static_cast<double>(qs.rows),
                    qps, 1000.0 * qsec / static_cast<double>(qs.rows), buildSec);
        return 0;
    }

    int showHelp(int, char**)
    {
        std::fprintf(stderr,
            "usage:\n"
            "  vortex-cli build <dataset_dir> <out.bin> [threads] [M] [efC]\n"
            "  vortex-cli query <index.bin> <dataset_dir> <row> [k] [ef]\n"
            "  vortex-cli bench <dataset_dir> [threads] [k] [ef]\n");
        return 1;
    }
}

int main(int argc, char** argv)
{
    if (argc < 2) return showHelp(argc, argv);
    const std::string cmd = argv[1];
    const std::function<int(int, char**)> fn =
        cmd == "build" ? cmdBuild :
        cmd == "query" ? cmdQuery :
        cmd == "bench" ? cmdBench : nullptr;
    if (!fn) return showHelp(argc, argv);
    try
    {
        return fn(argc, argv);
    }
    catch (const vortex::VortexError& e)
    {
        std::fprintf(stderr, "vortex-cli: %s\n", e.what());
        return 1;
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "vortex-cli: %s\n", e.what());
        return 1;
    }
}
