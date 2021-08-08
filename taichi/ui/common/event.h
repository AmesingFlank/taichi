#pragma once
#include "../utils/utils.h"

TI_UI_NAMESPACE_BEGIN

struct Event {
  EventType tag;

  DEFINE_PROPERTY(std::string, key);
};


TI_UI_NAMESPACE_END