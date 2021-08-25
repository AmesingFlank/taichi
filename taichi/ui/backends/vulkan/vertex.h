#pragma once

namespace taichi {
namespace ui {

struct Vertex {
  template<typename T>
  struct vec3 {
    T x;
    T y;
    T z;
  };
  template<typename T>
  struct vec2 {
    float x;
    float y;
  };
  vec3<float> pos;
  vec3<float> normal;
  vec2<float> texCoord;
  vec3<unsigned char> color;
};

}  // namespace ui
}  // namespace taichi
