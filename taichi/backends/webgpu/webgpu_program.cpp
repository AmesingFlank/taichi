#include "taichi/backends/webgpu/webgpu_program.h"
#include "taichi/backends/webgpu/aot_module_builder_impl.h"
 
using namespace taichi::lang::webgpu;

namespace taichi {
namespace lang {
 
WebgpuProgramImpl::WebgpuProgramImpl(CompileConfig &config)
    : ProgramImpl(config) {
}
 

FunctionType WebgpuProgramImpl::compile(Kernel *kernel,
                                        OffloadedStmt *offloaded) {
  TI_NOT_IMPLEMENTED;
}

void WebgpuProgramImpl::materialize_runtime(MemoryPool *memory_pool,
                                            KernelProfilerBase *profiler,
                                            uint64 **result_buffer_ptr) {
  *result_buffer_ptr = (uint64 *)memory_pool->allocate(
      sizeof(uint64) * taichi_result_buffer_entries, 8);
}

void WebgpuProgramImpl::compile_snode_tree_types(
    SNodeTree *tree,
    std::vector<std::unique_ptr<SNodeTree>> &snode_trees) {
 
    CompiledSNodeStructs compiled_structs =
        webgpu::compile_snode_structs(*tree->root());
    aot_compiled_snode_structs_.push_back(compiled_structs);
  
}

void WebgpuProgramImpl::materialize_snode_tree(
    SNodeTree *tree,
    std::vector<std::unique_ptr<SNodeTree>> &,
    uint64 *result_buffer) { 
        TI_NOT_IMPLEMENTED;
}

std::unique_ptr<AotModuleBuilder> WebgpuProgramImpl::make_aot_module_builder() {
  return std::make_unique<AotModuleBuilderImpl>(aot_compiled_snode_structs_);
}

DeviceAllocation WebgpuProgramImpl::allocate_memory_ndarray(
    std::size_t alloc_size,
    uint64 *result_buffer) {
      TI_NOT_IMPLEMENTED;
}

WebgpuProgramImpl::~WebgpuProgramImpl() {

}

}  // namespace lang
}  // namespace taichi
