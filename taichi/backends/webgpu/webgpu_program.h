#pragma once
#include "taichi/backends/webgpu/wgsl_codegen.h"
#include "taichi/backends/webgpu/snode_struct_compiler.h"
#include "taichi/backends/webgpu/kernel_utils.h"
  

#include "taichi/system/memory_pool.h"
#include "taichi/common/logging.h"
#include "taichi/struct/snode_tree.h"
#include "taichi/program/snode_expr_utils.h"
#include "taichi/program/program_impl.h"
#include "taichi/program/program.h"

#include <optional>

namespace taichi {
namespace lang {

 

class WebgpuProgramImpl : public ProgramImpl {
 public:
  WebgpuProgramImpl(CompileConfig &config);
  FunctionType compile(Kernel *kernel, OffloadedStmt *offloaded) override;

  std::size_t get_snode_num_dynamically_allocated(
      SNode *snode,
      uint64 *result_buffer) override {
    return 0;  
  }

  void compile_snode_tree_types(
      SNodeTree *tree,
      std::vector<std::unique_ptr<SNodeTree>> &snode_trees) override;

  void materialize_runtime(MemoryPool *memory_pool,
                           KernelProfilerBase *profiler,
                           uint64 **result_buffer_ptr) override;

  void materialize_snode_tree(SNodeTree *tree,
                              std::vector<std::unique_ptr<SNodeTree>> &,
                              uint64 *result_buffer) override;

  void synchronize() override {
   }

  std::unique_ptr<AotModuleBuilder> make_aot_module_builder() override;

  virtual void destroy_snode_tree(SNodeTree *snode_tree) override {
   }

  DeviceAllocation allocate_memory_ndarray(std::size_t alloc_size,
                                           uint64 *result_buffer) override;

  Device *get_compute_device() override {
    return nullptr;
  }

  Device *get_graphics_device() override {
    return nullptr;
  }

  DevicePtr get_snode_tree_device_ptr(int tree_id) override {
    return kDeviceNullPtr;
  }

  ~WebgpuProgramImpl();

 private: 
  std::vector<webgpu::CompiledSNodeStructs> aot_compiled_snode_structs_; 
};
}  // namespace lang
}  // namespace taichi
