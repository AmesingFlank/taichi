#pragma once

#define VOLK_IMPLEMENTATION
#include <volk.h>

#include "taichi/backends/vulkan/loader.h"

namespace taichi {
namespace lang {
namespace vulkan {

VulkanLoader::VulkanLoader() {
}

bool VulkanLoader::init() {
  std::call_once(init_flag_, [&]() {
    if (initialized) {
      return;
    }
    VkResult result = volkInitialize();
    initialized = result == VK_SUCCESS;
  });
  return initialized;
}

void VulkanLoader::load_instance(VkInstance instance) {
  vulkan_instance_ = instance;
  volkLoadInstance(instance);
}
void VulkanLoader::load_device(VkDevice device) {
  vulkan_device_ = device;
  volkLoadDevice(device);
}

PFN_vkVoidFunction VulkanLoader::load_function(const char * name){
    auto result = vkGetInstanceProcAddr(VulkanLoader::instance().vulkan_instance_, name);
    //printf("loading %s \n",name);
    if(result == nullptr){
        printf("%s is nullptr\n",name);
    }
    return result;
}

}  // namespace vulkan
}  // namespace lang
}  // namespace taichi
