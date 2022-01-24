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

struct TestClass {
  bool testProperty = false;
};

class TestBigClass {
 public:
  TestClass config;
};

std::unique_ptr<Kernel> create_kernel(Program &program,
                                      IRBuilder &builder,
                                      const std::string &name,
                                      bool grad) {
  return std::make_unique<Kernel>(program, builder.extract_ir(), name, grad);
};

std::vector<std::vector<uint32_t>> get_kernel_spirv(
    AotModuleBuilder &aot_builder,
    const std::string &name) {
  std::vector<std::vector<uint32_t>> result;
  auto kernel = aot_builder.get_compiled_kernel(name);
  for (auto &task : kernel.tasks) {
    size_t num_words = task.code.size() / 4;
    std::vector<uint32_t> code(num_words);
    std::memcpy(code.data(), task.code.data(), num_words * 4);
    result.push_back(std::move(code));
  }
  return result;
};

EMSCRIPTEN_BINDINGS(tint) {
  function("lerp", &lerp);

  enum_<Arch>("Arch").value("vulkan", Arch::vulkan);

  enum_<SNodeType>("SNodeType")
      .value("root", SNodeType::root)
      .value("dense", SNodeType::dense)
      .value("place", SNodeType::place);

  class_<SNodeTree>("SNodeTree");

  register_vector<uint32_t>("VectorOfUnsignedInt32");
  register_vector<std::vector<uint32_t>>("VectorOfVectorOfUnsignedInt32");

  class_<AotModuleBuilder>("AotModuleBuilder")
      .function("add_field", &AotModuleBuilder::add_field, allow_raw_pointers())
      .function("add", &AotModuleBuilder::add, allow_raw_pointers())
      .function("dump", &AotModuleBuilder::dump, allow_raw_pointers());

  function("get_kernel_spirv", &get_kernel_spirv);

  class_<Program>("Program")
      .constructor<Arch>()
      .function("add_snode_tree", &Program::add_snode_tree,
                allow_raw_pointers())
      .function("make_aot_module_builder", &Program::make_aot_module_builder);

  class_<Axis>("Axis").constructor<int>();

  class_<SNode>("SNode")
      .constructor<int, SNodeType>()
      .function(
          "dense",
          select_overload<SNode *(const Axis &, int, bool)>(&SNode::dense_ptr),
          allow_raw_pointers())
      .function("insert_children", &SNode::insert_children_ptr,
                allow_raw_pointers())
      .function("dt_get", &SNode::dt_get, allow_raw_pointers())
      .function("dt_set", &SNode::dt_set, allow_raw_pointers());

  class_<DataType>("DataType");

  class_<PrimitiveType>("PrimitiveType")
      .class_property(
          "i32",
          &PrimitiveType::i32)  //[](){return PrimitiveType::i32;}, [](const
                                //DataType& dt){PrimitiveType::i32 = dt;})
      ;

  class_<Stmt>("Stmt");
  class_<ConstStmt, base<Stmt>>("ConstStmt");
  class_<RangeForStmt, base<Stmt>>("RangeForStmt");
  class_<LoopIndexStmt, base<Stmt>>("LoopIndexStmt");
  class_<GlobalPtrStmt, base<Stmt>>("GlobalPtrStmt");

  class_<IRBuilder::LoopGuard>("LoopGuard");

  class_<IRNode>("IRNode");
  // class_<std::unique_ptr<IRNode>>("StdUniquePtrOfIRNode");
  //.smart_ptr<std::unique_ptr<IRNode>>("IRNode");

  class_<Block, base<IRNode>>("Block");
  // class_<std::unique_ptr<Block>>("StdUniquePtrOfBlock");
  //.smart_ptr<std::unique_ptr<Block>>("Block");

  class_<Callable>("Callable");

  class_<Kernel>("Kernel").class_function("create_kernel", &create_kernel);

  register_vector<Stmt *>("VectorOfStmtPtr");
  register_vector<int>("VectorOfInt");

  class_<IRBuilder>("IRBuilder")
      .constructor<>()
      .function("extract_ir", &IRBuilder::extract_ir)
      .function("get_int32", &IRBuilder::get_int32, allow_raw_pointers())
      .function("create_range_for", &IRBuilder::create_range_for,
                allow_raw_pointers())
      .function("get_range_loop_guard",
                &IRBuilder::allocate_loop_guard<RangeForStmt>,
                allow_raw_pointers())
      .function("get_loop_index", &IRBuilder::get_loop_index,
                allow_raw_pointers())
      .function("create_global_ptr", &IRBuilder::create_global_ptr,
                allow_raw_pointers())
      .function("create_global_ptr_global_store",
                &IRBuilder::create_global_store<GlobalPtrStmt>,
                allow_raw_pointers());
}
