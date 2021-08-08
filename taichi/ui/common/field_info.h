#pragma once
#include "../utils/utils.h"
#include "constants.h"

TI_UI_NAMESPACE_BEGIN

struct FieldInfo {
  DEFINE_PROPERTY(bool, valid)
  DEFINE_PROPERTY(int, field_type);
  DEFINE_PROPERTY(int, matrix_rows);
  DEFINE_PROPERTY(int, matrix_cols);
  DEFINE_PROPERTY(std::vector<int>, shape);
  DEFINE_PROPERTY(int, field_source);
  DEFINE_PROPERTY(DType, dtype);
  DEFINE_PROPERTY(uint64_t, data);

  FieldInfo() {
    valid = false;
  }
};

TI_UI_NAMESPACE_END