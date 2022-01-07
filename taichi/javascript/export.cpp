#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <taichi/program/arch.h>
#include <taichi/program/program.h>

#include <emscripten.h>
#include <emscripten/bind.h>

using namespace emscripten;
using namespace taichi;
using namespace taichi::lang;
 
float lerp(float a, float b, float t) {
    return (1 - t) * a + t * b;
} 
 
EMSCRIPTEN_BINDINGS(tint) {
    function("lerp", &lerp);

    enum_<Arch>("Arch")
        .value("vulkan", Arch::vulkan) 
    ;

    class_<Program>("Program")
    .constructor<Arch>()
    .function("make_aot_module_builder", &Program::make_aot_module_builder);

}
