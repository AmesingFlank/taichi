#pragma once
#include "field_info.h"
#include "scene_base.h"
#include "renderable_info.h"

struct SetImageInfo {
  FieldInfo img;
};

struct TrianglesInfo {
  RenderableInfo renderable_info;
  glm::vec3 color;
};

struct CirclesInfo {
  RenderableInfo renderable_info;
  glm::vec3 color;
  float radius;
};

struct LinesInfo {
  RenderableInfo renderable_info;
  glm::vec3 color;
  float width;
};

struct CanvasBase {
  virtual void set_background_color(const glm::vec3 &color) {
  }
  virtual void set_image(const SetImageInfo &info) {
  }
  virtual void triangles(const TrianglesInfo &info) {
  }
  virtual void circles(const CirclesInfo &info) {
  }
  virtual void lines(const LinesInfo &info) {
  }
  virtual void scene(SceneBase *scene) {
  }
};
