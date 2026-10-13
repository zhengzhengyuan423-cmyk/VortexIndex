#include "vortex/index.hpp"
#include <utility>
#include "vortex/errors.hpp"

namespace vortex
{

struct Index::Impl
{
    IndexParams params;
    std::size_t count = 0;

    explicit Impl(const IndexParams& p) : params(p) { params.validate(); }
};

Index::Index(const IndexParams& params) : impl_(std::make_unique<Impl>(params)) {}

// 析构必须定义在 Impl 类型完整的翻译单元：
// unique_ptr<Impl> 删除时需要知道如何销毁 Impl（PImpl 的固定配套写法）。
Index::~Index() = default;
Index::Index(Index&&) noexcept = default;
Index& Index::operator=(Index&&) noexcept = default;

void Index::addPoint(const float*, std::int64_t)
{
    throw NotImplementedError("addPoint");
}

std::vector<SearchResult> Index::searchKnn(const float* query, int k, int ef, const IdFilter&) const
{
    if (query == nullptr)
        throw ValueError("query must not be null");
    if (k < 1)
        throw ValueError("k must be >= 1");
    if (ef < k)
        throw ValueError("ef must be >= k");
    throw NotImplementedError("searchKnn");
}

std::size_t Index::size() const { return impl_->count; }

std::size_t Index::deletedCount() const { return 0; }

void Index::save(const std::string&) const
{
    throw NotImplementedError("save");
}

Index Index::load(const std::string&)
{
    throw NotImplementedError("load");
}

}
