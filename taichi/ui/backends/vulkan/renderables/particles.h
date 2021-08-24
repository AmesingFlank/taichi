#pragma once

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
#include "taichi/ui/utils/utils.h"
#include "taichi/ui/backends/vulkan/vertex.h"
#include "taichi/ui/backends/vulkan/vulkan_utils.h"
#include "taichi/ui/backends/vulkan/app_context.h"
#include "taichi/ui/backends/vulkan/swap_chain.h"
#include "taichi/ui/backends/vulkan/renderable.h"
#include "taichi/ui/common/field_info.h"
#include "taichi/ui/backends/vulkan/scene.h"

TI_UI_NAMESPACE_BEGIN

namespace vulkan {

class Particles final : public Renderable {
 public:
  Particles(class Renderer *renderer);

  void update_data(const ParticlesInfo &info, const Scene &scene);

 private:
  struct UniformBufferObject {
    Scene::SceneUniformBuffer scene;
    alignas(16) glm::vec3 color;
    int use_per_vertex_color;
    float radius;
    float window_width;
    float window_height;
    float tan_half_fov;
  };

  void init_particles(class Renderer *renderer, int vertices_count);

  void update_ubo(glm::vec3 color,
                  bool use_per_vertex_color,
                  float radius,
                  const Scene &scene);
  
  virtual void create_bindings() override;

  
};

}  // namespace vulkan

TI_UI_NAMESPACE_END
