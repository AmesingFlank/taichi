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
#include "../../../utils/utils.h"
#include "../vertex.h"
#include "../vulkan_utils.h"
#include "../app_context.h"
#include "../swap_chain.h"
#include "../renderable.h"
#include "../../../common/field_info.h"
#include "../scene.h"

TI_UI_NAMESPACE_BEGIN

namespace vulkan {

class Mesh final : public Renderable {
 public:
  Mesh(AppContext *app_context);

  void update_data(const MeshInfo &info, const Scene &scene);

 private:
  struct UniformBufferObject {
    Scene::SceneUniformBuffer scene;
    alignas(16) glm::vec3 color;
    int use_per_vertex_color;
    float shininess;
    int need_normal_generation;
  };

  void init_mesh(AppContext *app_context,
                 int vertices_count,
                 int indices_count);

  void update_ubo(const MeshInfo &info, const Scene &scene);

  virtual void create_descriptor_set_layout() override;

  virtual void create_descriptor_sets() override;
};

}  // namespace vulkan

TI_UI_NAMESPACE_END