#pragma once
#include "../utils/utils.h"

TI_UI_NAMESPACE_BEGIN


enum class EventType: int{
    EVENT_NONE = 0,
    EVENT_PRESS = 1,
    EVENT_RELEASE = 2
};

struct Event {
  EventType tag;

  DEFINE_PROPERTY(std::string, key);
};


TI_UI_NAMESPACE_END