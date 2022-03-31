#pragma once 
#include "taichi/ir/type.h"
namespace taichi {
namespace lang {

enum class TextureDimensionality {
    Dim2d,
};
 

struct TextureParams {
    PrimitiveTypeID primitive {PrimitiveTypeID::f32};
    TextureDimensionality dimensionality {TextureDimensionality::Dim2d};
    bool is_depth {false};
};

class Texture {
public:
    TextureParams params;
    int id{-1};
    Texture(TextureParams params, int id):params(params), id(id){

    }
};

}  // namespace lang
}  // namespace taichi
