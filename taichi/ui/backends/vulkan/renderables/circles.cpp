#include "circles.h"
#include "taichi/ui/backends/vulkan/vulkan_cuda_interop.h"
#include "taichi/ui/utils/utils.h"

TI_UI_NAMESPACE_BEGIN

namespace vulkan {

void Circles::update_data(const CirclesInfo &info) {
  if (info.renderable_info.vertices.matrix_rows != 2 ||
      info.renderable_info.vertices.matrix_cols != 1) {
    throw std::runtime_error("Circles vertices requres 2-d vector fields");
  }

  Renderable::update_data(info.renderable_info);

  update_ubo(info.color, info.renderable_info.per_vertex_color.valid,
             info.radius);
}

void Circles::init_circles(AppContext *app_context, int vertices_count) {
  RenderableConfig config = {
      vertices_count,
      1,
      sizeof(UniformBufferObject),
      app_context->config.package_path + "/shaders/Circles_vk_vert.spv",
      "",
      app_context->config.package_path + "/shaders/Circles_vk_frag.spv",
      TopologyType::Points,
  };

  Renderable::init(config, app_context);
  Renderable::init_render_resources();
}

Circles::Circles(AppContext *app_context) {
  init_circles(app_context, 1);
}

void Circles::update_ubo(glm::vec3 color,
                         bool use_per_vertex_color,
                         float radius) {
  UniformBufferObject ubo{
      color, (int)use_per_vertex_color,
      radius * app_context_->swap_chain.swap_chain_extent.height};

  MappedMemory mapped(
      app_context_->device(),
      uniform_buffer_memories_[app_context_->swap_chain.curr_image_index],
      sizeof(ubo));
  memcpy(mapped.data, &ubo, sizeof(ubo));
}

void Circles::create_descriptor_set_layout() {
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

void Circles::create_descriptor_sets() {
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

TI_UI_NAMESPACE_END
