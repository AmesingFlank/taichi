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
#include "taichi/ui/backend/vulkan/vertex.h"
#include "taichi/ui/backend/vulkan/vulkan_utils.h"
#include "taichi/ui/backend/vulkan/app_context.h"
#include "taichi/ui/backend/vulkan/swap_chain.h"
#include "../../common/renderable_info.h"

TI_UI_NAMESPACE_BEGIN

namespace vulkan {

enum class TopologyType : int { Triangles = 0, Lines = 1, Points = 2 };

struct RenderableConfig {
  int vertices_count;
  int indices_count;
  int ubo_size;
  std::string vertex_shader_path;
  std::string geometry_shader_path;
  std::string fragment_shader_path;
  TopologyType topology_type;
};

class Renderable {
 public:
  void update_data(const RenderableInfo &info);

  virtual void record_this_frame_commands(VkCommandBuffer command_buffer);

  virtual void recreate_swap_chain();

  void cleanup_swap_chain();

  virtual void cleanup();

  virtual ~Renderable();

 protected:
  RenderableConfig config_;

  AppContext *app_context_;

  VkPipelineLayout pipeline_layout_;
  VkPipeline graphics_pipeline_;

  VkBuffer vertex_buffer_;
  VkDeviceMemory vertex_buffer_memory_;
  VkBuffer index_buffer_;
  VkDeviceMemory index_buffer_memory_;

  // these staging buffers are used to copy data into the actual buffers when
  // `ti.cfg.arch==ti.cpu`
  VkBuffer staging_vertex_buffer_;
  VkDeviceMemory staging_vertex_buffer_memory_;
  VkBuffer staging_index_buffer_;
  VkDeviceMemory staging_index_buffer_memory_;

  std::vector<VkBuffer> uniform_buffers_;
  std::vector<VkDeviceMemory> uniform_buffer_memories_;

  VkDescriptorSetLayout descriptor_set_layout_;
  std::vector<VkDescriptorSet> descriptor_sets_;

  VkDescriptorPool descriptor_pool_;

  Vertex *vertex_buffer_device_ptr_;
  int *index_buffer_device_ptr_;

  bool indexed_{false};

 protected:
  void init(const RenderableConfig &config_, AppContext *app_context_);

  void init_render_resources();

  virtual void create_descriptor_set_layout() = 0;

  void create_descriptor_pool();

  void create_graphics_pipeline();

  void create_vertex_buffer();

  void create_index_buffer();

  void create_uniform_buffers();

  virtual void create_descriptor_sets() = 0;
};

}  // namespace vulkan

TI_UI_NAMESPACE_END