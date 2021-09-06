#pragma once
#include "taichi/backends/vulkan/codegen_vulkan.h"
#include "taichi/backends/vulkan/runtime.h"
#include "taichi/backends/vulkan/snode_struct_compiler.h"
#include "taichi/system/memory_pool.h"
#include "taichi/common/logging.h"
#include "taichi/struct/snode_tree.h"
#include "taichi/program/snode_expr_utils.h"
#include "taichi/program/program_impl.h"

#include <optional>

namespace taichi {
namespace lang {
class OpenglProgramImpl : public ProgramImpl {
 public:
  OpenglProgramImpl(CompileConfig &config) : ProgramImpl(config) {
  }
  FunctionType compile(Kernel *kernel, OffloadedStmt *offloaded) override;

  std::size_t get_snode_num_dynamically_allocated(
      SNode *snode,
      uint64 *result_buffer) override {
    return 0;  // TODO: support dynamic snode alloc in vulkan
  }

  void materialize_runtime(MemoryPool *memory_pool,
                           KernelProfilerBase *profiler,
                           uint64 **result_buffer_ptr) override;

  void materialize_snode_tree(
      SNodeTree *tree,
      std::vector<std::unique_ptr<SNodeTree>> &snode_trees_,
      std::unordered_map<int, SNode *> &snodes,
      SNodeGlobalVarExprMap &snode_to_glb_var_exprs_,
      uint64 *result_buffer) override;

  void synchronize() override {
    vulkan_runtime_->synchronize();
  }

  std::unique_ptr<AotModuleBuilder> make_aot_module_builder() override {
    // TODO: implement vk aot
    return nullptr;
  }

  ~OpenglProgramImpl() {
  }

 private:
  std::optional<opengl::StructCompiledResult> opengl_struct_compiled_;
  std::unique_ptr<opengl::GLSLLauncher> opengl_kernel_launcher_;
};
}  // namespace lang
}  // namespace taichi
