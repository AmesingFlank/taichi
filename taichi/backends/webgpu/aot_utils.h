#pragma once

#include <vector>

#include "taichi/backends/webgpu/kernel_utils.h"
#include "taichi/aot/module_data.h"
#include "taichi/aot/module_builder.h"

namespace taichi {
namespace lang {
namespace webgpu {

/**
 * AOT module data for the webgpu backend.
 */
struct TaichiAotData {
  //   BufferMetaData metadata;
  std::vector<std::vector<std::string>> wgsl_codes;
  std::vector<webgpu::TaichiKernelAttributes> kernels;
  std::vector<aot::CompiledFieldData> fields;
  size_t root_buffer_size;

  TI_IO_DEF(kernels, fields, root_buffer_size);
};

}  // namespace webgpu
}  // namespace lang
}  // namespace taichi
