#pragma once

#include <string>
#include "../utils/utils.h"

TI_UI_NAMESPACE_BEGIN

struct AppConfig {
  std::string name;
  int width;
  int height;
  bool vsync;
  std::string package_path;
  int ti_arch;
};


TI_UI_NAMESPACE_END