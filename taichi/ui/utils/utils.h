

#pragma once

#include <string>
#include <iostream>
#include <stdexcept>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <chrono>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <array>
#include <optional>
#include <set>
#include <unordered_map>

#ifdef _WIN64
#include <aclapi.h>
#include <dxgi1_2.h>
#include <windows.h>
#include <VersionHelpers.h>
#define _USE_MATH_DEFINES
#endif

#ifdef _WIN64
#define VK_USE_PLATFORM_WIN32_KHR 1
#endif
#include <volk.h>
#include <GLFW/glfw3.h>

#include <stdarg.h>
//#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include "../common/constants.h"

inline void initGLFW() {
  if (!glfwInit()) {
    printf("cannot initialize GLFW\n");
    exit(EXIT_FAILURE);
  }
}

inline GLFWwindow *create_glfw_window_(const std::string &name,
                                       int screenWidth,
                                       int screenHeight,
                                       bool vsync) {
  initGLFW();
  GLFWwindow *window;

  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

  window = glfwCreateWindow(screenWidth, screenHeight, name.c_str(), nullptr,
                            nullptr);

  if (!window) {
    glfwTerminate();
    exit(EXIT_FAILURE);
  }

  if (vsync) {
    glfwSwapInterval(1);
  } else {
    glfwSwapInterval(0);
  }
  return window;
}

inline std::unordered_map<std::string, int> get_keys_map() {
  std::unordered_map<std::string, int> keys;
  keys[KEY_SHIFT] = GLFW_KEY_LEFT_SHIFT;
  keys[KEY_ALT] = GLFW_KEY_LEFT_ALT;
  keys[KEY_CTRL] = GLFW_KEY_LEFT_CONTROL;
  keys[KEY_ESCAPE] = GLFW_KEY_ESCAPE;
  keys[KEY_RETURN] = GLFW_KEY_ENTER;
  keys[KEY_TAB] = GLFW_KEY_TAB;
  keys[KEY_BACKSPACE] = GLFW_KEY_BACKSPACE;
  keys[KEY_SPACE] = GLFW_KEY_SPACE;
  keys[KEY_UP] = GLFW_KEY_UP;
  keys[KEY_DOWN] = GLFW_KEY_DOWN;
  keys[KEY_LEFT] = GLFW_KEY_LEFT;
  keys[KEY_RIGHT] = GLFW_KEY_RIGHT;
  keys[KEY_CAPSLOCK] = GLFW_KEY_CAPS_LOCK;
  keys[KEY_LMB] = GLFW_MOUSE_BUTTON_LEFT;
  keys[KEY_MMB] = GLFW_MOUSE_BUTTON_MIDDLE;
  keys[KEY_RMB] = GLFW_MOUSE_BUTTON_RIGHT;
  return keys;
}

inline std::unordered_map<int, std::string> get_inv_keys_map() {
  auto keys = get_keys_map();
  std::unordered_map<int, std::string> keys_inv;
  for (auto kv : keys) {
    keys_inv[kv.second] = kv.first;
  }
  keys_inv[GLFW_KEY_RIGHT_SHIFT] = KEY_SHIFT;
  keys_inv[GLFW_KEY_RIGHT_CONTROL] = KEY_CTRL;
  keys_inv[GLFW_KEY_RIGHT_ALT] = KEY_ALT;
  return keys_inv;
}

inline int buttom_name_to_id(const std::string &name) {
  if (name.size() == 1) {
    char c = name[0];
    if (c >= 'a' && c <= 'z') {
      c = c - ('a' - 'A');
      return (int)c;
    }
  }

  auto keys = get_keys_map();

  if (keys.find(name) != keys.end()) {
    return keys.at(name);
  } else {
    throw std::runtime_error(std::string("unrecognized name: ") + name);
  }
}

inline std::string button_id_to_name(int id) {
  if (id >= 'A' && id <= 'Z') {
    char c = id + ('a' - 'A');
    std::string name;
    name += c;
    return name;
  }
  auto keys = get_inv_keys_map();

  if (keys.find(id) != keys.end()) {
    return keys.at(id);
  } else {
    throw std::runtime_error(std::string("unrecognized id: \n") +
                             std::to_string(id));
  }
}

inline int next_power_of_2(int n) {
  int count = 0;

  if (n && !(n & (n - 1)))
    return n;

  while (n != 0) {
    n >>= 1;
    count += 1;
  }

  return 1 << count;
}

#define DEFINE_PROPERTY(Type, name)    \
  Type name;                           \
  void set_##name(const Type &name_) { \
    name = name_;                      \
  }                                    \
  Type get_##name() {                  \
    return name;                       \
  }

inline std::vector<char> read_file(const std::string &filename) {
  std::ifstream file(filename, std::ios::ate | std::ios::binary);

  if (!file.is_open()) {
    throw std::runtime_error(filename + " failed to open file!");
  }

  size_t fileSize = (size_t)file.tellg();
  std::vector<char> buffer(fileSize);

  file.seekg(0);
  file.read(buffer.data(), fileSize);

  file.close();

  return buffer;
}
