#pragma once
#include "taichi/ir/type.h"
namespace taichi {
namespace lang {

enum class TextureDimensionality {
  Dim2d,
  Dim3d,
  DimCube,
};

struct TextureParams {
  DataType primitive{PrimitiveType::f32};
  TextureDimensionality dimensionality{TextureDimensionality::Dim2d};
  bool is_depth{false};
  std::string format;
  TextureParams() {
  }
  TextureParams(DataType primitive,
                TextureDimensionality dimensionality,
                bool is_depth,
                std::string format)
      : primitive(primitive),
        dimensionality(dimensionality),
        is_depth(is_depth),
        format(format) {
  }
};

class Texture {
 public:
  TextureParams params;
  int id{-1};
  Texture(TextureParams params, int id) : params(params), id(id) {
  }
};

inline int get_texture_coords_num_components(TextureDimensionality dim) {
  switch (dim) {
    case TextureDimensionality::Dim2d: {
      return 2;
    }
    case TextureDimensionality::DimCube: {
      return 3;
    }
    case TextureDimensionality::Dim3d: {
      return 3;
    }
    default: {
      return -1;
    }
  }
}

}  // namespace lang
}  // namespace taichi
