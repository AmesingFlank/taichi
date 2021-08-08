#pragma once

#include "../utils/utils.h"
#include "constants.h"

TI_UI_NAMESPACE_BEGIN

struct Camera {
  glm::vec3 position;
  glm::vec3 lookat;
  glm::vec3 up;
  int projection_mode = PROJECTION_PERSPECTIVE;

  float fov = 45;
  float left, right, top, bottom, z_near, z_far;

  glm::mat4 get_view_matrix() {
    return glm::lookAt(position, lookat, up);
  }
  glm::mat4 get_projection_matrix(float aspect_ratio) {
    if (projection_mode == PROJECTION_PERSPECTIVE) {
      return glm::perspective(fov, aspect_ratio, 0.1f, 1000.f);
    } else if (projection_mode == PROJECTION_ORTHOGONAL) {
      return glm::ortho(left, right, top, bottom, z_near, z_far);
    } else {
      throw std::runtime_error("invalid camera projection mode");
    }
  }
};


TI_UI_NAMESPACE_END