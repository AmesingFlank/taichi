#pragma once
#include "field_info.h"
#include "../utils/utils.h"

struct RenderableInfo{
    FieldInfo vertices;
    FieldInfo normals;
    FieldInfo tex_coords;
    FieldInfo per_vertex_color;
    FieldInfo indices;
};