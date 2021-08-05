#pragma once

#include <string>

struct AppConfig {
  std::string name;
  int width;
  int height;
  bool vsync;
  std::string package_path;
  int ti_arch;
};
