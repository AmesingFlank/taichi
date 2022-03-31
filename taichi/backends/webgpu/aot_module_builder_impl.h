#pragma once

#include <string>
#include <vector>

#include "taichi/backends/webgpu/aot_utils.h"
#include "taichi/backends/webgpu/snode_struct_compiler.h"
#include "taichi/backends/webgpu/kernel_utils.h"

#include "taichi/aot/module_data.h"
#include "taichi/aot/module_builder.h"
#include "taichi/texture/texture.h"

namespace taichi {
namespace lang {
namespace webgpu {

class AotModuleBuilderImpl : public AotModuleBuilder {
 public:
  explicit AotModuleBuilderImpl(
      const std::vector<CompiledSNodeStructs> &compiled_structs, const std::unordered_map<int, Texture*>& textures);

  void dump(const std::string &output_dir,
            const std::string &filename) const override;

  aot::CompiledTaichiKernel get_compiled_kernel(
      const std::string &name) const override;

 private:
  void add_per_backend(const std::string &identifier, Kernel *kernel) override;

  void add_field_per_backend(const std::string &identifier,
                             const SNode *rep_snode,
                             bool is_scalar,
                             DataType dt,
                             std::vector<int> shape,
                             int row_num,
                             int column_num) override;

  void add_per_backend_tmpl(const std::string &identifier,
                            const std::string &key,
                            Kernel *kernel) override;
 
  const std::vector<CompiledSNodeStructs> &compiled_structs_;
  const std::unordered_map<int, Texture*>& textures_;
  TaichiAotData ti_aot_data_;
};

}  // namespace webgpu
}  // namespace lang
}  // namespace taichi
