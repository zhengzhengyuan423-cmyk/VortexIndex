#include "vortex/index.hpp"
#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <shared_mutex>
#include <stdexcept>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

#include "api/slot_book.hpp"
#include "core/build_pipeline.hpp"
#include "io/dataset.hpp"
#include "io/serializer.hpp"
#include "core/graph_store.hpp"
#include "core/hnsw_index.hpp"
#include "core/knn_hit.hpp"
#include "core/level_generator.hpp"
#include "core/neighbor_select.hpp"
#include "vortex/errors.hpp"

namespace vortex
{
    // 批量删除后的墓碑占比警告阈值：超过就提 compact()，何时压实归业务
    static const double kCompactWarnRatio = 0.2;

    struct Index::Impl
    {
        IndexParams params;
        GraphStore store;              // 内核连续块存储
        LevelGenerator levels;         // 分层抽签，seed 固定即确定性构建
        SlotBook book;                 // 槽号<->id 映射 + 墓碑旗（api/slot_book.hpp）
        mutable std::shared_mutex mu;  // 数据主锁：搜索阶段共享、写表阶段独占

        explicit Impl(const IndexParams& p)
            : params(p), store(p.dims, static_cast<std::size_t>(p.M), p.capacity),
              levels(static_cast<std::size_t>(p.M), p.seed)
        {
            params.validate();
        }

        void stampRow(const float* data, const InsertPlan& plan, std::int64_t id);

        void addOne(const float* data, std::int64_t id);

        void workerLoop(WorkQueue& wq, Gate& gate, const float* data, std::size_t rows, std::size_t kWindow);
        
        void commitLoop(WorkQueue& wq, const float* data, const std::int64_t* ids, std::size_t rows, std::size_t kCommitBatch);
    };

    Index::Index(const IndexParams& params)
        : impl_(std::make_unique<Impl>(params))
    {}

    // 析构必须定义在 Impl 类型完整的翻译单元
    Index::~Index() = default;
    Index::Index(Index&&) noexcept = default;
    Index& Index::operator=(Index&&) noexcept = default;


    void Index::Impl::addOne(const float* data, std::int64_t id)
    {
        InsertPlan plan;
        {
            std::shared_lock<std::shared_mutex> guard(mu);
            if (book.slots() >= params.capacity)  throw CapacityError("index is full");
            const std::size_t level = levels.nextLevel();
            plan = planInsertNode(store, params.metric, data, level, static_cast<std::size_t>(params.efConstruction));
        }

        {
            std::unique_lock<std::shared_mutex> guard(mu);
            if (book.slots() >= params.capacity)  throw CapacityError("index is full");
            stampRow(data, plan, id);
        }
    }

    // 全库唯一落笔口：单线程、并行 commit、compact 重建都走这里登记
    void Index::Impl::stampRow(const float* data, const InsertPlan& plan, std::int64_t id)
    {
        const std::int64_t slot = applyInsertNode(store, params.metric, data, plan, SelectStrategy::Heuristic);
        book.record(static_cast<std::size_t>(slot), id);
    }

    void Index::addPoints(const std::string& fvecsPath, std::size_t threads)
    {
        const Dataset ds = readFvecs(fvecsPath);
        if (ds.dims != 0 && ds.dims != impl_->params.dims)
            throw ValueError("fvecs dims mismatch");
        std::vector<std::int64_t> ids(ds.rows);
        std::iota(ids.begin(), ids.end(), 0);
        addPoints(ds.data.data(), ids.data(), ds.rows, threads);
    }

    void Index::addPoints(const float* data, const std::int64_t* ids, std::size_t rows, std::size_t threads)
    {
        if (data == nullptr) throw ValueError("data must not be null");
        if (ids == nullptr) throw ValueError("ids must not be null");
        if (threads == 0) throw ValueError("threads must be >= 1");
        if (rows == 0) return;
        if (threads > rows) threads = rows;  

        {
            std::shared_lock<std::shared_mutex> guard(impl_->mu);
            // 墓碑也占槽：删而不压不会腾出容量，compact() 才还
            if (impl_->book.slots() + rows > impl_->params.capacity)
                throw CapacityError("index is full");
        }
        
        if (impl_->params.metric == DistanceMetric::Cosine)
        {
            for (std::size_t r = 0; r < rows; r++)
            {
                float norm = 0.f;
                const float* row = data + r * impl_->params.dims;
                for (std::size_t i = 0; i < impl_->params.dims; i++)
                    norm += row[i] * row[i];
                if (norm == 0.f) throw ValueError("cosine index rejects zero vector");
            }
        }

        if (threads == 1)
        {
            for (std::size_t i = 0; i < rows; i++)
                impl_->addOne(data + i * impl_->params.dims, ids[i]);
            return;
        }


        const std::size_t kWindow = std::max<std::size_t>(8 * threads, rows / 156);
        WorkQueue wq;
        Gate gate;

        // 起工作线程
        std::vector<std::thread> pool;
        try
        {
            for (std::size_t t = 0; t < threads; t++)
                pool.emplace_back([&, threads, rows, kWindow]
                { impl_->workerLoop(wq, gate, data, rows, kWindow); });
            gate.open(threads);
        }
        catch (...)
        {
            // 叫停已到闸机的，等全部退干净后原样抛出——索引零改动
            gate.abortNow();
            wq.wakeAll();
            for (auto& th : pool) th.join();
            throw;
        }

        // 起主线程
        impl_->commitLoop(wq, data, ids, rows, kWindow);
        for (auto& th : pool) th.join();
        if (const std::exception_ptr e = wq.firstError()) std::rethrow_exception(e);
    }

    // 工人主循环：领票 -> 共享锁里搜索 -> 交回计划书
    void Index::Impl::workerLoop(WorkQueue& wq, Gate& gate, const float* data,
                                 std::size_t rows, std::size_t kWindow)
    {
        if (!gate.arrive()) return;
        const std::size_t dims = params.dims;
        const std::size_t efC = static_cast<std::size_t>(params.efConstruction);
        while (!wq.failed())
        {
            std::size_t i = 0;
            if (!wq.takeTicket(kWindow, rows, i)) break;
            PlannedRow pr{i, InsertPlan{}};
            try
            {
                std::shared_lock<std::shared_mutex> guard(mu);
                const std::size_t level = levels.nextLevel();
                pr.plan = planInsertNode(store, params.metric, data + pr.row * dims, level, efC);
            }
            catch (...)
            {
                wq.recordError();
                break;
            }
            wq.pushPlan(std::move(pr));
        }
    }

    // 落笔人主循环：等攒满一整窗 -> 独占锁整批写表 -> 报完成腾窗口
    void Index::Impl::commitLoop(WorkQueue& wq, const float* data, const std::int64_t* ids, std::size_t rows, std::size_t kCommitBatch)
    {
        const std::size_t dims = params.dims;
        while(1)
        {
            std::vector<PlannedRow> batch;
            if (!wq.waitBatch(kCommitBatch, rows, batch)) break;
            if (batch.empty()) break;  // 全票已发、队列已空：收工
            try
            {
                std::unique_lock<std::shared_mutex> guard(mu);
                for (const PlannedRow& pr : batch)
                    stampRow(data + pr.row * dims, pr.plan, ids[pr.row]);
            }
            catch (...)
            {
                wq.recordError();
                break;
            }
            wq.commitDone(batch.size());
        }
    }

    std::vector<SearchResult> Index::searchKnn(const float* query, int k,
                                               int ef, const IdFilter& filter) const
    {
        if (query == nullptr) throw ValueError("query must not be null");
        if (k < 1) throw ValueError("k must be >= 1");
        if (ef < k) throw ValueError("ef must be >= k");

        std::shared_lock<std::shared_mutex> guard(impl_->mu);

        // 空图返回
        if (impl_->store.size() == 0) return {};

        // 内核按 ef 束宽搜，结果已距离升序，::vortex:: 限定是必需的
        const auto hits = ::vortex::searchKnn(
            impl_->store, impl_->params.metric, query,
            static_cast<std::size_t>(ef));

        std::vector<SearchResult> out;
        out.reserve(static_cast<std::size_t>(k));
        for (const KnnHit& h : hits)
        {
            // 墓碑照常参与内核路由，只在结果映射这层被扣下
            if (!impl_->book.live(static_cast<std::size_t>(h.id)))
                continue;
            const std::int64_t extId = impl_->book.externalId(static_cast<std::size_t>(h.id));
            if (filter && !filter(extId)) continue;  // post-filter
            out.push_back(SearchResult{extId, h.distance});
            if (out.size() == static_cast<std::size_t>(k)) break;
        }
        return out;
    }

    std::size_t Index::size() const
    {
        // 台账由写线程在独占锁内改动，读要共享锁
        std::shared_lock<std::shared_mutex> guard(impl_->mu);
        return impl_->book.liveCount();
    }

    std::size_t Index::deletedCount() const
    {
        std::shared_lock<std::shared_mutex> guard(impl_->mu);
        return impl_->book.tombstones();
    }

    std::size_t Index::removePoint(std::int64_t id)
    {
        std::unique_lock<std::shared_mutex> guard(impl_->mu);
        return impl_->book.removeId(id);
    }

    std::size_t Index::removePoint(const std::int64_t* ids, std::size_t rows)
    {
        if (ids == nullptr) throw ValueError("ids must not be null");
        std::size_t flipped = 0;
        {
            std::unique_lock<std::shared_mutex> guard(impl_->mu);
            for (std::size_t i = 0; i < rows; i++)
                flipped += impl_->book.removeId(ids[i]);
        }
        {
            // 批删收尾自查：单删不叫，一次一条就报警是狼来了
            std::shared_lock<std::shared_mutex> guard(impl_->mu);
            if (impl_->book.crowded(kCompactWarnRatio))
                std::fprintf(stderr,
                             "vortex: warning: tombstones %zu/%zu over %.0f%%, "
                             "consider compact()\n",
                             impl_->book.tombstones(), impl_->book.slots(),
                             kCompactWarnRatio * 100.0);
        }
        return flipped;
    }

    // 压实：按槽号原序把活点重灌成新图。外部 id 不变
    void Index::compact()
    {
        std::unique_lock<std::shared_mutex> guard(impl_->mu);
        if (impl_->book.tombstones() == 0) return;  // 无墓碑空操作，不抖图

        // 向量必须拷出来：下面一换 store，旧 arena 当场释放，指针全成悬垂
        struct LivePoint 
        { 
            std::vector<float> vec; 
            std::int64_t id; 
        };

        std::vector<LivePoint> live;
        live.reserve(impl_->book.liveCount());
        const std::size_t dims = impl_->params.dims;
        for (std::size_t s = 0; s < impl_->store.size(); s++)
        {
            if (!impl_->book.live(s)) continue;
            const float* v = impl_->store.vector(static_cast<std::int64_t>(s));
            live.push_back(LivePoint{std::vector<float>(v, v + dims),
                                     impl_->book.externalId(s)});
        }


        impl_->store = GraphStore(dims, static_cast<std::size_t>(impl_->params.M), impl_->params.capacity);
        impl_->book.reset();
        LevelGenerator gen(static_cast<std::size_t>(impl_->params.M), impl_->params.seed);
        const std::size_t efC = static_cast<std::size_t>(impl_->params.efConstruction);
        for (const LivePoint& lp : live)
        {
            const InsertPlan plan = planInsertNode(impl_->store, impl_->params.metric,
                                                   lp.vec.data(), gen.nextLevel(), efC);
            impl_->stampRow(lp.vec.data(), plan, lp.id);
        }
    }

    // 抄盘成文件，先写 .tmp，全部落盘后 rename 原子换正主
    void Index::save(const std::string& path) const
    {
        std::shared_lock<std::shared_mutex> guard(impl_->mu);
        const std::string tmp = path + ".tmp";
        {
            std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
            if (!out) throw IOError("save");
            serializeIndex(out, impl_->store, impl_->book, impl_->params);
            out.close();
            if (!out) throw IOError("save");  // 收尾写失败只在 close 后浮现
        }
        std::error_code ec;
        std::filesystem::rename(tmp, path, ec);
        if (ec)
        {
            std::error_code ign;
            std::filesystem::remove(tmp, ign);
            throw IOError("save");
        }
    }


    Index Index::load(const std::string& path)
    {
        std::ifstream in(path, std::ios::binary);
        if (!in) throw IOError("load");
        LoadedFile f = deserializeIndex(in);
        Index idx(f.params);  // 构造再过一遍 validate
        idx.impl_->store = std::move(f.store);
        idx.impl_->book = std::move(f.book);
        return idx;
    }

}
