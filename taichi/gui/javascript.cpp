#include "taichi/gui/gui.h"
#if defined(TI_EMSCRIPTENED)

TI_NAMESPACE_BEGIN

void GUI::process_event() {
  TI_ERROR("GGUI not supported on JS");
}

void GUI::create_window() {
  TI_ERROR("GGUI not supported on JS");
}

void GUI::redraw() {
  TI_ERROR("GGUI not supported on JS");
}

void GUI::set_title(std::string title) {
  TI_ERROR("GGUI not supported on JS");
}

GUI::~GUI() {
}

TI_NAMESPACE_END

#endif
