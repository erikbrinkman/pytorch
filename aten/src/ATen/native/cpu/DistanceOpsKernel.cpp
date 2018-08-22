#include "ATen/native/cpu/ReduceOpsKernel.h"

#include <numeric>
#include <iterator>
#include <algorithm>

#include "ATen/Dispatch.h"
#include "ATen/Parallel.h"
#include "ATen/core/optional.h"
#include "ATen/cpu/vec256/vec256.h"

namespace at { namespace native { namespace {

template<typename scalar_t>
struct PDist {

  static scalar_t zdist_calc(const scalar_t * a, const scalar_t * b, const int64_t size, const int64_t stride, const double p) {
    scalar_t result = 0.0;
    for (int64_t i = 0; i != size; i += 1, a += stride, b += stride) {
      result += *a != *b;
    }
    return result;
  }

  static scalar_t odist_calc(const scalar_t * a, const scalar_t * b, const int64_t size, const int64_t stride, const double p) {
    scalar_t result = 0.0;
    for (int64_t i = 0; i != size; i += 1, a += stride, b += stride) {
      result += std::abs(*a - *b);
    }
    return result;
  }

  static scalar_t tdist_calc(const scalar_t * a, const scalar_t * b, const int64_t size, const int64_t stride, const double p) {
    scalar_t result = 0.0;
    for (int64_t i = 0; i != size; i += 1, a += stride, b += stride) {
      scalar_t diff = *a - *b;
      result += diff * diff;
    }
    return std::sqrt(result);
  }

  static scalar_t pdist_calc(const scalar_t * a, const scalar_t * b, const int64_t size, const int64_t stride, const double p) {
    scalar_t result = 0.0;
    for (int64_t i = 0; i != size; i += 1, a += stride, b += stride) {
      result += std::pow(std::abs(*a - *b), p);
    }
    return std::pow(result, 1.0 / p);
  }

  static scalar_t idist_calc(const scalar_t * a, const scalar_t * b, const int64_t size, const int64_t stride, const double p) {
    scalar_t result = 0.0;
    for (int64_t i = 0; i != size; i += 1, a += stride, b += stride) {
      result = std::max(result, std::abs(*a - *b));
    }
    return result;
  }

  template <scalar_t (*F)(const scalar_t *, const scalar_t *, const int64_t, const int64_t, const double)>
  static void run_parallel(Tensor& result, const Tensor& self, const double p) {
    auto res_ = result.data<scalar_t>();
    auto self_ = self.data<scalar_t>();
    int64_t n = self.size(0);
    int64_t m = self.size(1);
    int64_t ns = self.stride(0);
    int64_t ms = self.stride(1);

    int64_t combs = n * (n - 1) / 2;
    parallel_for(0, combs, 1, [=](int64_t k, int64_t end) {
      float n2 = n - .5;
      // The -1 accounts for floating point truncation issues
      int64_t i = (int64_t) ((n2 - std::sqrt(n2 * n2 - 2 * k - 1)));
      int64_t j = k - n * i + i * (i + 1) / 2 + i + 1;
      for (; k < end; ++k) {
        res_[k] = F(self_ + i * ns, self_ + j * ns, m, ms, p);
        ++j;
        if (j == n) {
          ++i;
          j = i + 1;
        }
      }
    });
  }

  // Assumes self is nonempty and 2D
  static void apply(Tensor& result, const Tensor& self, const double p) {
    if (p == 0) {
      run_parallel<zdist_calc>(result, self, p);
    } else if (p == 1) {
      run_parallel<odist_calc>(result, self, p);
    } else if (p == 2) {
      run_parallel<tdist_calc>(result, self, p);
    } else if (std::isinf(p)) {
      run_parallel<idist_calc>(result, self, p);
    } else {
      run_parallel<pdist_calc>(result, self, p);
    }
  }

};

static void pdist_kernel_impl(Tensor& result, const Tensor& self, double p) {
  AT_DISPATCH_FLOATING_TYPES(self.type(), "pdist", [&] {
    PDist<scalar_t>::apply(result, self, p);
  });
}

}  // anonymous namespace

REGISTER_DISPATCH(pdist_kernel, &pdist_kernel_impl);

}}  // namespace at::native
