#pragma once
#include "../utils/utils.h"
struct Event {
  int tag;

  DEFINE_PROPERTY(std::string, key);
};
