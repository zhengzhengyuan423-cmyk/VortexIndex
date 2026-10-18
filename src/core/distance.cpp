#include "vortex/distance.hpp"
#include <cmath>

#include "vortex/errors.hpp"

namespace vortex
{

    float evalDistance(DistanceMetric metric, const float* a, const float* b,
                       std::size_t dims)
    {
        if (a == nullptr || b == nullptr)
            throw ValueError("distance inputs must not be null");
        if (dims == 0) throw ValueError("dims must be > 0");

        switch (metric)
        {
            case DistanceMetric::L2:
            {
                // 平方欧氏距离（不开方更快）
                float sum = 0.f;
                for (std::size_t i = 0; i < dims; i++)
                {
                    float diff = a[i] - b[i];
                    sum += diff * diff;
                }
                return sum;
            }
            case DistanceMetric::InnerProduct:
            {
                // 内积距离 = 1 - (a·b)
                float dot = 0.f;
                for (std::size_t i = 0; i < dims; i++) dot += a[i] * b[i];
                return 1.f - dot;
            }
            case DistanceMetric::Cosine:
            {
                // 余弦距离 = 1 - cos(θ) = 1 - (a·b / (||a|| * ||b||))
                float dot = 0.f, na = 0.f, nb = 0.f;
                for (std::size_t i = 0; i < dims; i++)
                {
                    dot += a[i] * b[i];
                    na += a[i] * a[i];
                    nb += b[i] * b[i];
                }
                // 除数不为0
                if (na * nb == 0.f)
                    throw ValueError(
                        "cosine distance is undefined for zero vector");
                return 1.f - dot / std::sqrt(na * nb);
            }
        }
        throw ValueError("unknown distance metric");
    }

}
