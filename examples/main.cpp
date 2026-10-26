// 三行级消费者：证明 find_package(vortex) + 一个 link 就能用，
// 不碰仓库里任何源码。跑对与否由 smoke_consumer.sh 断言（首个命中 id=11）
#include <cstdio>
#include <vector>
#include "vortex/index.hpp"

int main()
{
    vortex::IndexParams p;
    p.dims = 2;
    p.capacity = 8;
    p.M = 4;
    p.efConstruction = 16;
    vortex::Index idx(p);

    const float v[] = {0.f, 0.f, 1.f, 0.f, 0.f, 1.f, 5.f, 5.f};
    const std::int64_t ids[] = {10, 11, 12, 13};
    idx.addPoints(v, ids, 4);

    const float q[] = {0.9f, 0.1f};  // 离 (1,0) 最近
    for (const vortex::SearchResult& h : idx.searchKnn(q, 2, 8))
        std::printf("id=%lld dist=%.4f\n", static_cast<long long>(h.id), h.distance);
    return 0;
}
