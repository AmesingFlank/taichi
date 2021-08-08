#pragma once
#include "field_info.h"
#include "../utils/utils.h"

TI_UI_NAMESPACE_BEGIN

struct RenderableInfo {
  FieldInfo vertices;
  FieldInfo normals;
  FieldInfo tex_coords;
  FieldInfo per_vertex_color;
  FieldInfo indices;
};


TI_UI_NAMESPACE_END