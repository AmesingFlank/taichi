#include "taichi/backends/webgpu/kernel_utils.h"

#include <unordered_map>

#include "taichi/program/kernel.h"
#define TI_RUNTIME_HOST
#include "taichi/program/context.h"
#undef TI_RUNTIME_HOST

namespace taichi {
namespace lang {
namespace webgpu {

KernelContextAttributes::KernelContextAttributes(const Kernel &kernel)
    : args_bytes_(0),
      rets_bytes_(0),
      extra_args_bytes_(RuntimeContext::extra_args_size) {
  arg_attribs_vec_.reserve(kernel.args.size());
  for (const auto &ka : kernel.args) {
    ArgAttributes aa;
    aa.dt = ka.dt;
    const size_t dt_bytes = data_type_size(aa.dt);
    aa.is_array = ka.is_array;
    aa.stride = dt_bytes;
    aa.index = arg_attribs_vec_.size();
    arg_attribs_vec_.push_back(aa);
  }
  for (const auto &kr : kernel.rets) {
    RetAttributes ra;
    ra.dt = kr.dt;
    const size_t dt_bytes = data_type_size(ra.dt);
    ra.is_array = false;  // TODO(#909): this is a temporary limitation
    ra.stride = dt_bytes;
    ra.index = ret_attribs_vec_.size();
    ret_attribs_vec_.push_back(ra);
  }

  auto arange_args = [](auto *vec, size_t offset) -> size_t {
    std::vector<int> scalar_indices;
    std::vector<int> array_indices;
    size_t bytes = offset;
    for (int i = 0; i < vec->size(); ++i) {
      auto &attribs = (*vec)[i];
      const size_t dt_bytes = data_type_size(attribs.dt);
      // Align bytes to the nearest multiple of dt_bytes
      bytes = (bytes + dt_bytes - 1) / dt_bytes * dt_bytes;
      attribs.offset_in_mem = bytes;
      bytes += attribs.stride;
      TI_TRACE("  at={} {} offset_in_mem={} stride={}",
               (*vec)[i].is_array ? "vector ptr" : "scalar", i,
               attribs.offset_in_mem, attribs.stride);
    }
    return bytes - offset;
  };

  TI_TRACE("args:");
  args_bytes_ = arange_args(&arg_attribs_vec_, 0);
  TI_TRACE("rets:");
  rets_bytes_ = arange_args(&ret_attribs_vec_, args_bytes_);

  TI_TRACE("sizes: args={} rets={} ctx={} total={}", args_bytes(), rets_bytes(),
           ctx_bytes(), total_bytes());
  TI_ASSERT(has_rets() == (rets_bytes_ > 0));
}

}  // namespace spirv
}  // namespace lang
}  // namespace taichi