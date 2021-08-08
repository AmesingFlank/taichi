#pragma once
#include "taichi/ui/common/scene_base.h"

TI_UI_NAMESPACE_BEGIN

namespace vulkan {

class Scene final : public SceneBase {
 public:
  friend class Canvas;
  friend class Particles;
  friend class Mesh;

 private:
  struct SceneUniformBuffer {
    alignas(16) glm::vec3 camera_pos;
    alignas(16) glm::mat4 view;
    alignas(16) glm::mat4 projection;
    // vec4 instead of vec3 because of alignment in vulkan uniform buffer
    glm::vec4 point_light_positions[MAX_POINTLIGHTS];
    glm::vec4 point_light_colors[MAX_POINTLIGHTS];
    int point_light_count;
    alignas(16) glm::vec3 ambient_light;
  };
  SceneUniformBuffer current_ubo_;

  void update_ubo(float aspect_ratio) {
    current_ubo_.camera_pos = camera_.position;
    current_ubo_.view = camera_.get_view_matrix();
    current_ubo_.projection = camera_.get_projection_matrix(aspect_ratio);
    current_ubo_.point_light_count = point_lights_.size();
    for (int i = 0; i < point_lights_.size(); ++i) {
      current_ubo_.point_light_positions[i] =
          glm::vec4(point_lights_[i].pos, 1);
      current_ubo_.point_light_colors[i] = glm::vec4(point_lights_[i].color, 1);
    }
    current_ubo_.ambient_light = ambient_light_color_;
  }
};

}  // namespace vulkan

TI_UI_NAMESPACE_END
