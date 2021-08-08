#pragma once

enum class DType: int{
    DType_I8 = 0,
    DType_I16 = 1,
    DType_I32 = 2,
    DType_I64 = 3,
    DType_U8 = 4,
    DType_U16 = 5,
    DType_U32 = 6,
    DType_U64 = 7,
    DType_F32 = 8,
    DType_F64 = 9
};


enum class EventType: int{
    EVENT_NONE = 0,
    EVENT_PRESS = 1,
    EVENT_RELEASE = 2
};


#define FIELD_SOURCE_X64 0
#define FIELD_SOURCE_CUDA 1

#define ARCH_X64 0
#define ARCH_CUDA 1

#define FIELD_TYPE_FIELD 0
#define FIELD_TYPE_MATRIX 1

#define PROJECTION_ORTHOGONAL 0
#define PROJECTION_PERSPECTIVE 1


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
