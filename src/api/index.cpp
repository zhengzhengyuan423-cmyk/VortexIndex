#include "vortex/index.hpp"
#include <algorithm>
#include <numeric>
#include <random>
#include <shared_mutex>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#include "core/build_pipeline.hpp"
#include "core/graph_store.hpp"
#include "core/hnsw_index.hpp"
#include "core/knn_hit.hpp"
#include "core/level_generator.hpp"
#include "core/neighbor_select.hpp"
#include "vortex/errors.hpp"

namespace vortex
{
    struct Index::Impl
    {
        IndexParams params;
        GraphStore store;              // 内核连续块存储
        LevelGenerator levels;         // 分层抽签，seed 固定即确定性构建
        std::vector<std::int64_t> slotToExternal;  // 内部槽位 -> 调用方 id
        std::size_t count = 0;         // 活跃向量数
        mutable std::shared_mutex mu;  // 数据主锁：搜索阶段共享、写表阶段独占

        explicit Impl(const IndexParams& p)
            : params(p), store(p.dims, static_cast<std::size_t>(p.M), p.capacity),
              levels(static_cast<std::size_t>(p.M), p.seed)
        {
            params.validate();
        }

        void stampRow(const float* data, const InsertPlan& plan, std::int64_t id);

        void addOne(const float* data, std::int64_t id);

        void workerLoop(WorkQueue& wq, Gate& gate, const float* data, const std::vector<std::size_t>& order, std::size_t rows, std::size_t kWindow);
        
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
            if (count >= params.capacity)  throw CapacityError("index is full");
            const std::size_t level = levels.nextLevel();
            plan = planInsertNode(store, params.metric, data, level, static_cast<std::size_t>(params.efConstruction));
        }

        {
            std::unique_lock<std::shared_mutex> guard(mu);
            if (count >= params.capacity)  throw CapacityError("index is full");
            stampRow(data, plan, id);
        }
    }

    void Index::Impl::stampRow(const float* data, const InsertPlan& plan, std::int64_t id)
    {
        const std::int64_t slot = applyInsertNode(store, params.metric, data, plan, SelectStrategy::Heuristic);
        if (static_cast<std::size_t>(slot) >= slotToExternal.size())
            slotToExternal.resize(static_cast<std::size_t>(slot) + 1, -1);
        slotToExternal[static_cast<std::size_t>(slot)] = id;
        count++;
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
            if (impl_->count + rows > impl_->params.capacity)
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


        std::vector<std::size_t> order(rows);
        std::iota(order.begin(), order.end(), 0);
        std::mt19937_64 rng(static_cast<std::uint64_t>(impl_->params.seed));
        std::shuffle(order.begin(), order.end(), rng);

        const std::size_t kWindow = std::max<std::size_t>(8 * threads, rows / 156);
        WorkQueue wq;
        Gate gate;

        // 起工作线程
        std::vector<std::thread> pool;
        try
        {
            for (std::size_t t = 0; t < threads; t++)
                pool.emplace_back([&, threads, rows, kWindow]
                { impl_->workerLoop(wq, gate, data, order, rows, kWindow); });
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
                                 const std::vector<std::size_t>& order,
                                 std::size_t rows, std::size_t kWindow)
    {
        if (!gate.arrive()) return;
        const std::size_t dims = params.dims;
        const std::size_t efC = static_cast<std::size_t>(params.efConstruction);
        while (!wq.failed())
        {
            std::size_t i = 0;
            if (!wq.takeTicket(kWindow, rows, i)) break;
            PlannedRow pr{order[i], InsertPlan{}};
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
            if (static_cast<std::size_t>(h.id) >= impl_->slotToExternal.size())
                continue;  
            const std::int64_t extId = impl_->slotToExternal[static_cast<std::size_t>(h.id)];
            if (filter && !filter(extId)) continue;  // post-filter
            out.push_back(SearchResult{extId, h.distance});
            if (out.size() == static_cast<std::size_t>(k)) break;
        }
        return out;
    }

    std::size_t Index::size() const
    {
        // count 由写线程在独占锁内修改，读要共享锁
        std::shared_lock<std::shared_mutex> guard(impl_->mu);
        return impl_->count;
    }

    std::size_t Index::deletedCount() const { return 0; }

    void Index::save(const std::string&) const
    {
        throw NotImplementedError("save");
    }

    Index Index::load(const std::string&) { throw NotImplementedError("load"); }

}
