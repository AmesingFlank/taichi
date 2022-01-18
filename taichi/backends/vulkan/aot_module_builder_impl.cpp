#include "taichi/backends/vulkan/aot_module_builder_impl.h"

#include <fstream>
#include <type_traits>

#include "taichi/codegen/spirv/spirv_codegen.h"

namespace taichi {
namespace lang {
namespace vulkan {

AotModuleBuilderImpl::AotModuleBuilderImpl(
    const std::vector<CompiledSNodeStructs> &compiled_structs)
    : compiled_structs_(compiled_structs) {
  aot_target_device_ = std::make_unique<AotTargetDevice>(Arch::vulkan);
  if (!compiled_structs.empty()) {
    ti_aot_data_.root_buffer_size = compiled_structs[0].root_size;
  }
}

uint32_t AotModuleBuilderImpl::to_vk_dtype_enum(DataType dt) {
  if (dt == PrimitiveType::u64) {
    return 0;
  } else if (dt == PrimitiveType::i64) {
    return 1;
  } else if (dt == PrimitiveType::u32) {
    return 2;
  } else if (dt == PrimitiveType::i32) {
    return 3;
  } else if (dt == PrimitiveType::u16) {
    return 4;
  } else if (dt == PrimitiveType::i16) {
    return 5;
  } else if (dt == PrimitiveType::u8) {
    return 6;
  } else if (dt == PrimitiveType::i8) {
    return 7;
  } else if (dt == PrimitiveType::f64) {
    return 8;
  } else if (dt == PrimitiveType::f32) {
    return 9;
  } else {
    TI_NOT_IMPLEMENTED
  }
}

void AotModuleBuilderImpl::write_spv_file(
    const std::string &output_dir,
    const TaskAttributes &k,
    const std::vector<uint32_t> &source_code) const {
  const std::string spv_path = fmt::format("{}/{}.spv", output_dir, k.name);
  std::ofstream fs(spv_path, std::ios_base::binary | std::ios::trunc);
  std::cout << " --------------------------- "<<k.name<<"-----------"<<std::endl;
  std::cout << "[";
  for(int i = 0;i<source_code.size();++i){
    std::cout << source_code[i];
    if(i < source_code.size()-1){
      std::cout << ",";
    }
  }
  std::cout << "]\n";
  std::cout << " ------------------------------- "<<std::endl;
  fs.write((char *)source_code.data(), source_code.size() * sizeof(uint32_t));
  fs.close();
}

void AotModuleBuilderImpl::dump(const std::string &output_dir,
                                const std::string &filename) const {
#if defined(TI_EMSCRIPTENED)
  for (int i = 0; i < ti_aot_data_.kernels.size(); ++i) {
    printf("kernel %d\n",i);
    auto k = ti_aot_data_.kernels[i];
    for (int j = 0; j < k.tasks_attribs.size(); ++j) {
      printf("kernel %d, task %d, %s \n",i,j,k.tasks_attribs[j].name.c_str());
      const auto& code = ti_aot_data_.spirv_codes[i][j];
      for(auto c:code){
        printf("%d ",c);
      }
      printf("\n");
    }
  }
#endif
  TI_WARN_IF(!filename.empty(),
             "Filename prefix is ignored on vulkan backend.");
  const std::string bin_path = fmt::format("{}/metadata.tcb", output_dir);
  write_to_binary_file(ti_aot_data_, bin_path);
  // Json format doesn't support multiple line strings.
  for (int i = 0; i < ti_aot_data_.kernels.size(); ++i) {
    auto k = ti_aot_data_.kernels[i];
    for (int j = 0; j < k.tasks_attribs.size(); ++j) {
      write_spv_file(output_dir, k.tasks_attribs[j],
                     ti_aot_data_.spirv_codes[i][j]);
    }
  }

  const std::string txt_path = fmt::format("{}/metadata.json", output_dir);
  TextSerializer ts;
  ts.serialize_to_json("aot_data", ti_aot_data_);
  ts.write_to_file(txt_path);
}


void AotModuleBuilderImpl::add_per_backend(const std::string &identifier,
                                           Kernel *kernel) {
   printf("calling vk add kernel per backend\n");
  spirv::lower(kernel);
  printf("lowered\n");
  auto compiled =
      run_codegen(kernel, aot_target_device_.get(), compiled_structs_);
  ti_aot_data_.kernels.push_back(compiled.kernel_attribs);
  ti_aot_data_.spirv_codes.push_back(compiled.task_spirv_source_codes);
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
  printf("calling vk add field per backend\n");
  printf("id  %d \n", rep_snode->parent->id);
  printf("num compiled structs %d \n", int(compiled_structs_.size()));
  printf("num compiled structs[0].snode descriptors %d \n", int(compiled_structs_[0].snode_descriptors.size()));
  const auto &dense_desc =
      compiled_structs_[0].snode_descriptors.at(rep_snode->parent->id);
  printf("1\n");
  aot::CompiledFieldData field_data;
  field_data.field_name = identifier;
  field_data.is_scalar = is_scalar;
  field_data.dtype = to_vk_dtype_enum(dt);
  field_data.dtype_name = dt.to_string();
  field_data.shape = shape;
  field_data.mem_offset_in_parent = dense_desc.mem_offset_in_parent_cell;
  field_data.row_num = row_num;
  field_data.column_num = column_num;
  printf("2\n");
  ti_aot_data_.fields.push_back(field_data);
   printf("3\n");
}

void AotModuleBuilderImpl::add_per_backend_tmpl(const std::string &identifier,
                                                const std::string &key,
                                                Kernel *kernel) {
}

}  // namespace vulkan
}  // namespace lang
}  // namespace taichi
