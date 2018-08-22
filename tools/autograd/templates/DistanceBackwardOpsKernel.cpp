#include "ATen/native/cpu/ReduceOpsKernel.h"

#include <numeric>
#include <iterator>
#include <algorithm>

#include "ATen/Dispatch.h"
#include "ATen/Parallel.h"
#include "ATen/core/optional.h"
#include "ATen/cpu/vec256/vec256.h"

namespace at { namespace native { namespace {

template <typename scalar_t>
struct PDistBackward {

  static inline scalar_t sign(scalar_t val) {
    return (0 < val) - (val < 0);
  }

  static void run_parallel(Tensor& result, const Tensor & grad, const Tensor & self) {
    const int64_t n = self.size(0);
    const int64_t ns = self.stride(0);
    const int64_t m = self.size(1);
    const int64_t ms = self.stride(1);
    const int64_t gs = grad.stride(0);

    const scalar_t * const grad_ = grad.data<scalar_t>();
    const scalar_t * const self_ = self.data<scalar_t>();
    scalar_t * const res_ = result.data<scalar_t>();

    at::parallel_for(0, grad.numel(), 1, [=](int64_t k, int64_t end) {
      float n2 = n - .5;
      // The -1 accounts for floating point truncation issues
      int64_t i = (int64_t) ((n2 - std::sqrt(n2 * n2 - 2 * k - 1)));
      int64_t j = k - n * i + i * (i + 1) / 2 + i + 1;
      for (; k < end; ++k) {
        const scalar_t pgrad = grad_[k * gs];
        const scalar_t * isc = self_ + i * ns;
        const scalar_t * jsc = self_ + j * ns;
        scalar_t * irc = res_ + i * m;
        scalar_t * jrc = res_ + j * m;
        const scalar_t * rend = irc + m;

        for (; irc != rend; isc += ms, jsc += ms, irc += 1, jrc += 1) {
          const scalar_t res = pgrad * sign(*isc - *jsc);
          *irc += res;
          *jrc -= res;
        }

        ++j;
        if (j == n) {
          ++i;
          j = i + 1;
        }
      }
    });
    return result;
  }

  static void apply(Tensor& result, const Tensor& grad, const Tensor& self, const double p, const Tensor& pdist) {
    if (p == 0) {
    } else if (p == 1) {
      run_parallel(result, grad, self);
    } else {
    }
  }

};

static void pdist_backward_kernel_impl(Tensor& result, const Tensor& grad, const Tensor& self, const double p, const Tensor& pdist) {
  AT_DISPATCH_FLOATING_TYPES(self.type(), "pdist_backward", [&] {
    PDistBackward<scalar_t>::apply(result, grad, self, p, pdist);
  });
}

}  // anonymous namespace

REGISTER_DISPATCH(pdist_backward_kernel, &pdist_backward_kernel_impl);

}}  // namespace at::native
