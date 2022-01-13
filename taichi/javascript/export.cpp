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
#include <taichi/ir/ir_builder.h>
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
 
EMSCRIPTEN_BINDINGS(tint) {
    function("lerp", &lerp);

    enum_<Arch>("Arch")
        .value("vulkan", Arch::vulkan) 
    ;

    enum_<SNodeType>("SNodeType")
        .value("root", SNodeType::root) 
        .value("dense", SNodeType::dense) 
        .value("place", SNodeType::place) 
    ;

    class_<AotModuleBuilder>("AotModuleBuilder")
    ;

    class_<Program>("Program")
    .constructor<Arch>()
    .function("make_aot_module_builder", &Program::make_aot_module_builder);

    class_<Axis>("Axis")
    .constructor<int>() 
    ;

    class_<SNode>("SNode")
    .constructor<int, SNodeType>()
    .function("dense", select_overload<SNode*(const Axis &,int,bool)>( &SNode::dense_ptr), allow_raw_pointers())
    .function("insert_children",&SNode::insert_children_ptr, allow_raw_pointers())
    .property("dt",&SNode::dt)
    ;

    class_<DataType>("DataType")
    ;

    class_<PrimitiveType>("PrimitiveType")
    .class_property("i32", &PrimitiveType::i32)//[](){return PrimitiveType::i32;}, [](const DataType& dt){PrimitiveType::i32 = dt;})
    ;


    class_<Stmt>("Stmt");
    class_<ConstStmt, base<Stmt>>("ConstStmt");
    class_<RangeForStmt, base<Stmt>>("RangeForStmt");
    class_<LoopIndexStmt, base<Stmt>>("LoopIndexStmt");

    class_<IRBuilder::LoopGuard>("LoopGuard");


    class_<IRBuilder>("IRBuilder")
    .constructor<>()
    .function("get_int32",&IRBuilder::get_int32, allow_raw_pointers())
    .function("create_range_for",&IRBuilder::create_range_for, allow_raw_pointers())
    .function("get_range_loop_guard",&IRBuilder::get_loop_guard<RangeForStmt>, allow_raw_pointers())
    .function("get_loop_index",&IRBuilder::get_loop_index, allow_raw_pointers())
    ;

}
