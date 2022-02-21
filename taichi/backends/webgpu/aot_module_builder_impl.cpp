#include "taichi/backends/webgpu/aot_module_builder_impl.h"

#include <fstream>
#include <type_traits>

#include "taichi/aot/module_data.h"
#include "taichi/backends/webgpu/wgsl_codegen.h"
#include "taichi/program/program.h"


namespace taichi {
namespace lang {
namespace webgpu {

namespace {
class AotDataConverter {
 public:
  static aot::ModuleData convert(const TaichiAotData &in) {
    AotDataConverter c{};
    return c.visit(in);
  }

 private:
  explicit AotDataConverter() = default;

  aot::ModuleData visit(const TaichiAotData &in) const {
    aot::ModuleData res{};
    for (int i = 0; i < in.kernels.size(); ++i) {
      const auto &ker = in.kernels[i];
      auto val = visit(ker);
      for (int j = 0; j < ker.tasks_attribs.size(); ++j) {
        auto &code = in.wgsl_codes[i][j];
        printf("%s\n",code.c_str());
        size_t code_num_bytes = code.size();
        val.tasks[j].code = std::vector<unsigned char>(code_num_bytes);
        std::memcpy(val.tasks[j].code.data(), code.data(), code_num_bytes);
      }
      res.kernels[ker.name] = val;
    }
    res.fields = in.fields;
    res.root_buffer_size = in.root_buffer_size;
    return res;
  }

  aot::CompiledTaichiKernel visit(
      const webgpu::TaichiKernelAttributes &in) const {
    aot::CompiledTaichiKernel res{};
    res.tasks.reserve(in.tasks_attribs.size());
    for (const auto &t : in.tasks_attribs) {
      res.tasks.push_back(visit(t));
    }
    res.args_count = in.ctx_attribs.args().size();
    res.rets_count = in.ctx_attribs.rets().size();
    res.args_buffer_size = in.ctx_attribs.args_bytes();
    res.rets_buffer_size = in.ctx_attribs.rets_bytes();
    for (const auto &arg : in.ctx_attribs.args()) {
      res.scalar_args[arg.index] = visit(arg);
    }
    return res;
  }

  aot::CompiledOffloadedTask visit(const TaskAttributes &in) const {
    aot::CompiledOffloadedTask res{};
    res.type = offloaded_task_type_name(in.task_type);
    res.name = in.name;
    // TODO: update range_hint after ndarray is supported on webgpu.
    if (in.range_for_attribs && in.range_for_attribs->const_begin &&
        in.range_for_attribs->const_end) {
      res.range_hint = std::to_string(in.range_for_attribs->end -
                                      in.range_for_attribs->begin);
    }
    res.gpu_block_size = in.advisory_num_threads_per_group;
    return res;
  }

  aot::ScalarArg visit(
      const webgpu::KernelContextAttributes::ArgAttributes &in) const {
    aot::ScalarArg res{};
    res.dtype_name = in.dt.to_string();
    res.offset_in_args_buf = in.offset_in_mem;
    return res;
  }
};

}  // namespace
AotModuleBuilderImpl::AotModuleBuilderImpl(
    const std::vector<CompiledSNodeStructs> &compiled_structs)
    : compiled_structs_(compiled_structs) {
  if (!compiled_structs.empty()) {
    ti_aot_data_.root_buffer_size = compiled_structs[0].root_size;
  }
} 
 

aot::CompiledTaichiKernel AotModuleBuilderImpl::get_compiled_kernel(
    const std::string &name) const {
  auto converted = AotDataConverter::convert(ti_aot_data_);
  return converted.kernels.at(name);
}

void AotModuleBuilderImpl::dump(const std::string &output_dir,
                                const std::string &filename) const {
  
}

void AotModuleBuilderImpl::add_per_backend(const std::string &identifier,
                                           Kernel *kernel) {
  webgpu::lower(kernel);

  const auto id = Program::get_kernel_id();
  const auto taichi_kernel_name(fmt::format("{}_k{:04d}_vk", kernel->name, id));
  TI_TRACE("VK codegen for Taichi kernel={}", taichi_kernel_name);
  webgpu::KernelCodegen::Params params;
  params.ti_kernel_name = taichi_kernel_name;
  params.kernel = kernel;
  params.compiled_structs = compiled_structs_;
  
  webgpu::KernelCodegen codegen(params);
  TaichiKernelAttributes kernel_attribs;
  std::vector<std::string> generated_wgsl;
  codegen.run(kernel_attribs, generated_wgsl); 
  
  kernel_attribs.name = identifier;
  ti_aot_data_.kernels.push_back(kernel_attribs);
  ti_aot_data_.wgsl_codes.push_back(generated_wgsl);
}

void AotModuleBuilderImpl::add_field_per_backend(const std::string &identifier,
                                                 const SNode *rep_snode,
                                                 bool is_scalar,
                                                 DataType dt,
                                                 std::vector<int> shape,
                                                 int row_num,
                                                 int column_num) {
  // Note that currently we only support adding dense fields in AOT for all
  // backends. In opengl backend we only error out when a non dense field is
  // added to the aot module, but in metal backend we error out earlier when
  // constructing aot module. Ideally we will unify this behavior but it doesn't
  // matter too much for now.
  TI_ERROR_IF(!all_fields_are_dense_in_container(rep_snode->parent),
              "AOT: only supports dense field");
  const auto &dense_desc =
      compiled_structs_[0].snode_descriptors.at(rep_snode->parent->id);
  aot::CompiledFieldData field_data;
  field_data.field_name = identifier;
  field_data.is_scalar = is_scalar;
  field_data.dtype = 0;
  field_data.dtype_name = dt.to_string();
  field_data.shape = shape;
  field_data.mem_offset_in_parent = dense_desc.mem_offset_in_parent_cell;
  if (!is_scalar) {
    field_data.element_shape = {row_num, column_num};
  }
  ti_aot_data_.fields.push_back(field_data);
}

void AotModuleBuilderImpl::add_per_backend_tmpl(const std::string &identifier,
                                                const std::string &key,
                                                Kernel *kernel) {
  TI_ERROR("Templated kernels are not yet supported on webgpu aot.");
}

}  // namespace webgpu
}  // namespace lang
}  // namespace taichi
