#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <taichi/program/arch.h>
#include <taichi/program/program.h>
#include <taichi/ir/snode_types.h>
#include <taichi/ir/snode.h>
#include <taichi/ir/type.h>
#include <taichi/program/aot_module.h>
#include <taichi/program/callable.h>
#include <taichi/program/kernel.h>
#include <taichi/ir/ir_builder.h>
#include <taichi/ir/ir.h>
#include <taichi/ir/statements.h>

#include <emscripten.h>
#include <emscripten/bind.h>

using namespace emscripten;
using namespace taichi;
using namespace taichi::lang;
using namespace taichi::lang::aot;

float lerp(float a, float b, float t) {
  return (1 - t) * a + t * b;
}

Kernel *create_kernel(Program &program,
                      IRBuilder &builder,
                      const std::string &name,
                      bool grad) {
  return new Kernel(program, builder.extract_ir(), name, grad);
};

class TaskParams {
 public:
  std::vector<uint32_t> spirv;
  std::string range_hint;
  int gpu_block_size;

  std::vector<uint32_t> *get_spirv_ptr() {
    return &spirv;
  }
  std::string get_range_hint() {
    return range_hint;
  }
  int get_gpu_block_size() {
    return gpu_block_size;
  }
};

std::vector<TaskParams> get_kernel_params(AotModuleBuilder &aot_builder,
                                          const std::string &name) {
  std::vector<TaskParams> result;
  auto kernel = aot_builder.get_compiled_kernel(name);
  for (auto &task : kernel.tasks) {
    size_t num_words = task.code.size() / 4;
    std::vector<uint32_t> code(num_words);
    std::memcpy(code.data(), task.code.data(), num_words * 4);
    TaskParams params;
    params.spirv = std::move(code);
    params.range_hint = task.range_hint;
    params.gpu_block_size = task.gpu_block_size;
    result.push_back(params);
  }
  return result;
};

EMSCRIPTEN_BINDINGS(tint) {
  function("lerp", &lerp);

  register_vector<uint32_t>("VectorOfUnsignedInt32");
  register_vector<Stmt *>("VectorOfStmtPtr");
  register_vector<int>("VectorOfInt");
  register_vector<Axis>("VectorOfAxis");
  register_vector<TaskParams>("VectorOfTaskParams");

  enum_<Arch>("Arch").value("vulkan", Arch::vulkan);

  enum_<SNodeType>("SNodeType")
      .value("root", SNodeType::root)
      .value("dense", SNodeType::dense)
      .value("place", SNodeType::place);

  class_<SNodeTree>("SNodeTree");

  class_<AotModuleBuilder>("AotModuleBuilder")
      .function("add_field", &AotModuleBuilder::add_field, allow_raw_pointers())
      .function("add", &AotModuleBuilder::add, allow_raw_pointers())
      .function("dump", &AotModuleBuilder::dump, allow_raw_pointers());

  class_<TaskParams>("TaskParams")
      .function("get_spirv_ptr", &TaskParams::get_spirv_ptr,
                allow_raw_pointers())
      .function("get_range_hint", &TaskParams::get_range_hint,
                allow_raw_pointers())
      .function("get_gpu_block_size", &TaskParams::get_gpu_block_size,
                allow_raw_pointers());

  function("get_kernel_params", &get_kernel_params);

  class_<Program>("Program")
      .constructor<Arch>()
      .function("add_snode_tree", &Program::add_snode_tree,
                allow_raw_pointers())
      .function("make_aot_module_builder", &Program::make_aot_module_builder);

  class_<Axis>("Axis").constructor<int>();

  class_<SNode>("SNode")
      .constructor<int, SNodeType>()
      .function("dense",
                select_overload<SNode *(const std::vector<Axis> &,
                                        const std::vector<int> &, bool)>(
                    &SNode::dense_ptr),
                allow_raw_pointers())
      .function("insert_children", &SNode::insert_children_ptr,
                allow_raw_pointers())
      .function("dt_get", &SNode::dt_get, allow_raw_pointers())
      .function("dt_set", &SNode::dt_set, allow_raw_pointers());

  class_<DataType>("DataType");

  class_<PrimitiveType>("PrimitiveType")
      .class_property("i32", &PrimitiveType::i32)
      .class_property("f32", &PrimitiveType::f32);

  class_<Stmt>("Stmt");
  class_<ConstStmt, base<Stmt>>("ConstStmt");
  class_<RangeForStmt, base<Stmt>>("RangeForStmt");
  class_<LoopIndexStmt, base<Stmt>>("LoopIndexStmt");
  class_<AllocaStmt, base<Stmt>>("AllocaStmt");
  class_<LocalLoadStmt, base<Stmt>>("LocalLoadStmt");
  class_<GlobalPtrStmt, base<Stmt>>("GlobalPtrStmt");
  class_<GlobalLoadStmt, base<Stmt>>("GlobalLoadStmt");
  class_<BinaryOpStmt, base<Stmt>>("BinaryOpStmt");
  class_<UnaryOpStmt, base<Stmt>>("UnaryOpStmt");
  class_<WhileStmt, base<Stmt>>("WhileStmt");
  class_<IfStmt, base<Stmt>>("IfStmt");
  class_<WhileControlStmt, base<Stmt>>("WhileControlStmt");
  class_<ContinueStmt, base<Stmt>>("ContinueStmt");
  class_<ArgLoadStmt, base<Stmt>>("ArgLoadStmt");
  class_<RandStmt, base<Stmt>>("RandStmt");
  class_<ReturnStmt, base<Stmt>>("ReturnStmt");

  class_<IRBuilder::LoopGuard>("LoopGuard");
  class_<IRBuilder::IfGuard>("IfGuard");

  class_<IRNode>("IRNode");
  // class_<std::unique_ptr<IRNode>>("StdUniquePtrOfIRNode");
  //.smart_ptr<std::unique_ptr<IRNode>>("IRNode");

  class_<Block, base<IRNode>>("Block");
  // class_<std::unique_ptr<Block>>("StdUniquePtrOfBlock");
  //.smart_ptr<std::unique_ptr<Block>>("Block");

  class_<Callable>("Callable")
      .function("insert_arg", &Callable::insert_arg)
      .function("insert_ret", &Callable::insert_ret);

  class_<Kernel, base<Callable>>("Kernel").class_function(
      "create_kernel", &create_kernel, allow_raw_pointers());

  class_<IRBuilder>("IRBuilder")
      .constructor<>()
      .function("get_range_loop_guard",
                &IRBuilder::allocate_loop_guard<RangeForStmt>,
                allow_raw_pointers())
      .function("get_while_loop_guard",
                &IRBuilder::allocate_loop_guard<WhileStmt>,
                allow_raw_pointers())
      .function("get_if_guard", &IRBuilder::allocate_if_guard,
                allow_raw_pointers())
      .function("create_global_ptr_global_store",
                &IRBuilder::create_global_store<GlobalPtrStmt>,
                allow_raw_pointers())
      .function("create_global_ptr_global_load",
                &IRBuilder::create_global_load<GlobalPtrStmt>,
                allow_raw_pointers())
#define EXPORT_FUNCTION(f) .function(#f, &IRBuilder::f, allow_raw_pointers())

          EXPORT_FUNCTION(get_int32) EXPORT_FUNCTION(get_float32)
              EXPORT_FUNCTION(create_range_for) EXPORT_FUNCTION(
                  get_loop_index) EXPORT_FUNCTION(create_global_ptr)
                  EXPORT_FUNCTION(create_local_var) EXPORT_FUNCTION(
                      create_local_load) EXPORT_FUNCTION(create_local_store)

                      EXPORT_FUNCTION(create_add) EXPORT_FUNCTION(
                          create_sub) EXPORT_FUNCTION(create_mul)
                          EXPORT_FUNCTION(create_div) EXPORT_FUNCTION(
                              create_floordiv) EXPORT_FUNCTION(create_truediv)

                              EXPORT_FUNCTION(create_mod) EXPORT_FUNCTION(
                                  create_max) EXPORT_FUNCTION(create_min)
                                  EXPORT_FUNCTION(create_atan2) EXPORT_FUNCTION(
                                      create_pow)

                                      EXPORT_FUNCTION(
                                          create_and) EXPORT_FUNCTION(create_or)
                                          EXPORT_FUNCTION(create_xor)
                                              EXPORT_FUNCTION(create_shl)
                                                  EXPORT_FUNCTION(create_shr)
                                                      EXPORT_FUNCTION(
                                                          create_sar)
      // Comparisons.
      EXPORT_FUNCTION(create_cmp_lt) EXPORT_FUNCTION(
          create_cmp_le) EXPORT_FUNCTION(create_cmp_gt)
          EXPORT_FUNCTION(create_cmp_ge) EXPORT_FUNCTION(
              create_cmp_eq) EXPORT_FUNCTION(create_cmp_ne)

              EXPORT_FUNCTION(create_cast) EXPORT_FUNCTION(
                  create_bit_cast) EXPORT_FUNCTION(create_neg)
                  EXPORT_FUNCTION(create_not) EXPORT_FUNCTION(
                      create_logical_not) EXPORT_FUNCTION(create_round)
                      EXPORT_FUNCTION(create_floor) EXPORT_FUNCTION(
                          create_ceil) EXPORT_FUNCTION(create_abs)
                          EXPORT_FUNCTION(create_sgn) EXPORT_FUNCTION(
                              create_sqrt) EXPORT_FUNCTION(create_rsqrt)
                              EXPORT_FUNCTION(create_sin) EXPORT_FUNCTION(
                                  create_asin) EXPORT_FUNCTION(create_cos)
                                  EXPORT_FUNCTION(create_acos) EXPORT_FUNCTION(
                                      create_tan) EXPORT_FUNCTION(create_tanh)
                                      EXPORT_FUNCTION(
                                          create_exp) EXPORT_FUNCTION(create_log)

                                          EXPORT_FUNCTION(create_while_true)
                                              EXPORT_FUNCTION(create_if)
                                                  EXPORT_FUNCTION(create_break)
                                                      EXPORT_FUNCTION(
                                                          create_continue)

                                                          EXPORT_FUNCTION(
                                                              create_arg_load)
                                                              EXPORT_FUNCTION(
                                                                  create_rand)
                                                                  EXPORT_FUNCTION(
                                                                      create_return);
}
