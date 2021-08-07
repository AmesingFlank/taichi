#include "mesh.h"
#include "../vulkan_cuda_interop.h"
#include "../../../utils/utils.h"
#include "../../../common/constants.h"

namespace vulkan {

Mesh::Mesh(AppContext *app_context) {
  init_mesh(app_context, 3, 3);
}

void Mesh::update_ubo(const MeshInfo &info, const Scene &scene) {
  UniformBufferObject ubo;
  ubo.scene = scene.current_ubo_;
  ubo.color = info.color;
  ubo.use_per_vertex_color = info.renderable_info.per_vertex_color.valid;
  ubo.shininess = info.shininess;
  ubo.need_normal_generation = !info.renderable_info.normals.valid;

  MappedMemory mapped(
      app_context_->device(),
      uniform_buffer_memories_[app_context_->swap_chain.curr_image_index],
      sizeof(ubo));
  memcpy(mapped.data, &ubo, sizeof(ubo));
}

void Mesh::update_data(const MeshInfo &info, const Scene &scene) {
  if (info.renderable_info.vertices.matrix_rows != 3 ||
      info.renderable_info.vertices.matrix_cols != 1) {
    throw std::runtime_error("Mesh vertices requres 3-d vector fields");
  }

  Renderable::update_data(info.renderable_info);

  update_ubo(info, scene);
}

void Mesh::init_mesh(AppContext *app_context,
                     int vertices_count,
                     int indices_count) {
  RenderableConfig config = {
      vertices_count,
      indices_count,
      sizeof(UniformBufferObject),
      app_context->config.package_path + "/shaders/Mesh_vk_vert.spv",
      app_context->config.package_path + "/shaders/Mesh_vk_geom.spv",
      app_context->config.package_path + "/shaders/Mesh_vk_frag.spv",
      TopologyType::TriangleList,
  };

  Renderable::init(config, app_context);
  Renderable::init_render_resources();
}

void Mesh::create_descriptor_set_layout() {
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

void Mesh::create_descriptor_sets() {
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

}  // namespace vulkan
