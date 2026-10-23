#include "build_pipeline.hpp"

namespace vortex
{
    bool Gate::arrive()
    {
        std::unique_lock<std::mutex> g(mu_);
        ready_++;
        cv_.notify_all();
        cv_.wait(g, [&] { return go_ || abort_; });
        return !abort_;
    }

    void Gate::open(std::size_t total)
    {
        std::unique_lock<std::mutex> g(mu_);
        cv_.wait(g, [&] { return ready_ == total; });
        go_ = true;
        cv_.notify_all();
    }

    void Gate::abortNow()
    {
        {
            std::lock_guard<std::mutex> g(mu_);
            abort_ = true;
        }
        cv_.notify_all();
    }

    bool WorkQueue::takeTicket(std::size_t kWindow, std::size_t rows, std::size_t& out)
    {
        std::unique_lock<std::mutex> g(mu_);
        cvWindow_.wait(g, [&]
        {
            return failed_.load() || ticketed_ - committed_ < kWindow ||
                   ticketed_ == rows;
        });
        if (failed_.load() || ticketed_ == rows) return false;
        out = nextRow_++;
        ticketed_++;
        return true;
    }

    void WorkQueue::pushPlan(PlannedRow pr)
    {
        {
            std::lock_guard<std::mutex> g(mu_);
            queue_.push_back(std::move(pr));
            returned_++;
        }
        cvPlan_.notify_one();
    }

    // 主线程等一批
    bool WorkQueue::waitBatch(std::size_t kCommitBatch, std::size_t rows,
                              std::vector<PlannedRow>& out)
    {
        std::unique_lock<std::mutex> g(mu_);
        if (failed_.load() || committed_ == rows) return false;
        cvPlan_.wait(g, [&]
        {
            return failed_.load() || queue_.size() >= kCommitBatch ||
                   (returned_ == rows && !queue_.empty());
        });
        out.swap(queue_);
        return true;
    }

    void WorkQueue::commitDone(std::size_t n)
    {
        {
            std::lock_guard<std::mutex> g(mu_);
            committed_ += n;
        }
        cvWindow_.notify_all();
    }

    void WorkQueue::recordError()
    {
        {
            std::lock_guard<std::mutex> g(mu_);
            if (!firstError_) firstError_ = std::current_exception();
            failed_.store(true, std::memory_order_release);
        }
        cvPlan_.notify_all();
        cvWindow_.notify_all();
    }

    bool WorkQueue::failed() const
    {
        return failed_.load(std::memory_order_acquire);
    }

    void WorkQueue::wakeAll()
    {
        cvPlan_.notify_all();
        cvWindow_.notify_all();
    }

    std::exception_ptr WorkQueue::firstError()
    {
        std::lock_guard<std::mutex> g(mu_);
        return firstError_;
    }

}
