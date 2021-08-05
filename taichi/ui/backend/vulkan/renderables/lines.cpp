#include "lines.h"
#include "../vulkan_cuda_interop.h"
#include "../../../utils/utils.h"
#include "../../../common/constants.h"
#include "kernels.h"

namespace vulkan {

void Lines::update_data(const LinesInfo &info) {
  int N = info.renderable_info.vertices.shape[0] / 2;

  if (4 * N > config_.vertices_count || 6 * N > config_.indices_count) {
    cleanup_swap_chain();
    cleanup();
    init_lines(app_context_, 4 * N, 6 * N);
  }
  auto check_valid = [&](const FieldInfo &f) {
    if (f.dtype != DTYPE_F32) {
      throw std::runtime_error("dtype needs to be f32 for Lines");
    }
    if (f.matrix_rows != 2 || f.matrix_cols != 1) {
      throw std::runtime_error("Lines requres 2-d vector fields");
    }
  };
  check_valid(info.renderable_info.vertices);

  update_ubo(info.color, info.renderable_info.per_vertex_color.valid);

  float aspect_ratio = app_context_->swap_chain.swap_chain_extent.width /
                       (float)app_context_->swap_chain.swap_chain_extent.height;

  bool use_per_vertex_color = info.renderable_info.per_vertex_color.valid;

  if (info.renderable_info.vertices.field_source == FIELD_SOURCE_CUDA) {
    update_lines_vbo_cuda(vertex_buffer_device_ptr_, index_buffer_device_ptr_,
                          (float *)info.renderable_info.vertices.data, N,
                          info.width, aspect_ratio,
                          (float *)info.renderable_info.per_vertex_color.data,
                          use_per_vertex_color);
    CHECK_CUDA_ERROR("update lines data");
  } else if (info.renderable_info.vertices.field_source == FIELD_SOURCE_X64) {
    {
      MappedMemory mapped_vbo(app_context_->device(),
                              staging_vertex_buffer_memory_,
                              config_.vertices_count * sizeof(Vertex));
      MappedMemory mapped_ibo(app_context_->device(),
                              staging_index_buffer_memory_,
                              config_.indices_count * sizeof(int));
      update_lines_vbo_x64((Vertex *)mapped_vbo.data, (int *)mapped_ibo.data,
                           (float *)info.renderable_info.vertices.data, N,
                           info.width, aspect_ratio,
                           (float *)info.renderable_info.per_vertex_color.data,
                           use_per_vertex_color);
    }
    copy_buffer(staging_vertex_buffer_, vertex_buffer_,
                config_.vertices_count * sizeof(Vertex),
                app_context_->command_pool(), app_context_->device(),
                app_context_->graphics_queue());
    copy_buffer(staging_index_buffer_, index_buffer_,
                config_.indices_count * sizeof(int),
                app_context_->command_pool(), app_context_->device(),
                app_context_->graphics_queue());
  } else {
    throw std::runtime_error("unsupported field source");
  }
}

void Lines::init_lines(AppContext *app_context,
                       int vertices_count,
                       int indices_count) {
  RenderableConfig config = {
      vertices_count,
      indices_count,
      sizeof(UniformBufferObject),
      app_context->config.package_path + "/shaders/Lines_vk_vert.spv",
      "",
      app_context->config.package_path + "/shaders/Lines_vk_frag.spv",
      TopologyType::TriangleList,
  };

  Renderable::init(config, app_context);
  Renderable::init_render_resources();
}

Lines::Lines(AppContext *app_context) {
  init_lines(app_context, 4, 6);
}

void Lines::update_ubo(glm::vec3 color, bool use_per_vertex_color) {
  UniformBufferObject ubo{color, (int)use_per_vertex_color};

  MappedMemory mapped(
      app_context_->device(),
      uniform_buffer_memories_[app_context_->swap_chain.curr_image_index],
      sizeof(ubo));
  memcpy(mapped.data, &ubo, sizeof(ubo));
}

void Lines::create_descriptor_set_layout() {
  VkDescriptorSetLayoutBinding ubo_layout_binding{};
  ubo_layout_binding.binding = 0;
  ubo_layout_binding.descriptorCount = 1;
  ubo_layout_binding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  ubo_layout_binding.pImmutableSamplers = nullptr;
  ubo_layout_binding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT |
                                  VK_SHADER_STAGE_FRAGMENT_BIT |
                                  VK_SHADER_STAGE_GEOMETRY_BIT;

  std::array<VkDescriptorSetLayoutBinding, 1> bindings = {ubo_layout_binding};
  VkDescriptorSetLayoutCreateInfo layout_info{};
  layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  layout_info.bindingCount = static_cast<uint32_t>(bindings.size());
  layout_info.pBindings = bindings.data();

  if (vkCreateDescriptorSetLayout(app_context_->device(), &layout_info, nullptr,
                                  &descriptor_set_layout_) != VK_SUCCESS) {
    throw std::runtime_error("failed to create descriptor set layout!");
  }
}

void Lines::create_descriptor_sets() {
  std::vector<VkDescriptorSetLayout> layouts(
      app_context_->get_swap_chain_size(), descriptor_set_layout_);

  VkDescriptorSetAllocateInfo alloc_info{};
  alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  alloc_info.descriptorPool = descriptor_pool_;
  alloc_info.descriptorSetCount = app_context_->get_swap_chain_size();
  alloc_info.pSetLayouts = layouts.data();

  descriptor_sets_.resize(app_context_->get_swap_chain_size());

  if (vkAllocateDescriptorSets(app_context_->device(), &alloc_info,
                               descriptor_sets_.data()) != VK_SUCCESS) {
    throw std::runtime_error("failed to allocate descriptor sets!");
  }

  for (size_t i = 0; i < app_context_->get_swap_chain_size(); i++) {
    VkDescriptorBufferInfo buffer_info{};
    buffer_info.buffer = uniform_buffers_[i];
    buffer_info.offset = 0;
    buffer_info.range = config_.ubo_size;

    std::array<VkWriteDescriptorSet, 1> descriptor_writes{};

    descriptor_writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptor_writes[0].dstSet = descriptor_sets_[i];
    descriptor_writes[0].dstBinding = 0;
    descriptor_writes[0].dstArrayElement = 0;
    descriptor_writes[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    descriptor_writes[0].descriptorCount = 1;
    descriptor_writes[0].pBufferInfo = &buffer_info;

    vkUpdateDescriptorSets(app_context_->device(),
                           static_cast<uint32_t>(descriptor_writes.size()),
                           descriptor_writes.data(), 0, nullptr);
  }
}

void Lines::cleanup() {
  Renderable::cleanup();
}

}  // namespace vulkan
