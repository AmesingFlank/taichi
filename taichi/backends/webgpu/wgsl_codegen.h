#pragma once

#include "taichi/lang_util.h"

#include "taichi/backends/webgpu/snode_struct_compiler.h"
#include "taichi/backends/webgpu/kernel_utils.h"
#include "taichi/texture/texture.h"
  
namespace taichi {
namespace lang {

class Kernel;

namespace webgpu {

void lower(Kernel *kernel);

class KernelCodegen {
 public:
  struct Params {
    std::string ti_kernel_name;
    Kernel *kernel;
    std::vector<CompiledSNodeStructs> compiled_structs;
    std::unordered_map<int, Texture*> textures;
   };

  explicit KernelCodegen(const Params &params);

  void run(TaichiKernelAttributes &kernel_attribs,
           std::vector<std::string> &generated_wgsl);

 private:
  Params params_;
  KernelContextAttributes ctx_attribs_;
 
};

}  // namespace spirv
}  // namespace lang
}  // namespace taichi
