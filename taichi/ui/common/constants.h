#pragma once

#define DTYPE_I8 0
#define DTYPE_I16 1
#define DTYPE_I32 2
#define DTYPE_I64 3
#define DTYPE_U8 4
#define DTYPE_U16 5
#define DTYPE_U32 6
#define DTYPE_U64 7
#define DTYPE_F32 8
#define DTYPE_F64 9

#define EVENT_NONE 0
#define EVENT_PRESS 1

#define FIELD_SOURCE_X64 0
#define FIELD_SOURCE_CUDA 1

#define ARCH_X64 0
#define ARCH_CUDA 1

#define FIELD_TYPE_FIELD 0
#define FIELD_TYPE_MATRIX 1

#define PROJECTION_ORTHOGONAL 0
#define PROJECTION_PERSPECTIVE 1

#define GGUI_DEBUG 1

#define MAX_POINTLIGHTS 16

#define KEY_SHIFT std::string("Shift")
#define KEY_ALT std::string("Alt")
#define KEY_CTRL std::string("Control")
#define KEY_ESCAPE std::string("Escape")
#define KEY_RETURN std::string("Return")
#define KEY_TAB std::string("Tab")
#define KEY_BACKSPACE std::string("BackSpace")
#define KEY_SPACE std::string(" ")
#define KEY_UP std::string("Up")
#define KEY_DOWN std::string("Down")
#define KEY_LEFT std::string("Left")
#define KEY_RIGHT std::string("Right")
#define KEY_CAPSLOCK std::string("Caps_Lock")
#define KEY_LMB std::string("LMB")
#define KEY_MMB std::string("MMB")
#define KEY_RMB std::string("RMB")
