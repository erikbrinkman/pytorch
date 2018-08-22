#pragma once

#include <ATen/ATen.h>
#include <ATen/native/DispatchStub.h>

namespace at { namespace native {

using pdist_fn = void(*)(Tensor &, const Tensor &, const double);

DECLARE_DISPATCH(pdist_fn, pdist_kernel);

}} // namespace at::native
