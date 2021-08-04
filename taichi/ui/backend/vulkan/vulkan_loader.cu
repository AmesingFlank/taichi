#include "vulkan_loader.h"

#include <volk.h>

namespace vulkan{


VulkanLoader::VulkanLoader(){
    initialized_ = false;
}

bool VulkanLoader::init(){
    if(initialized_){
        return true;
    }
    VkResult result = volkInitialize();
    initialized_ = result == VK_SUCCESS;
    return initialized_;
}

void VulkanLoader::load_instance(VkInstance instance){
    vulkan_instance_ = instance;
    volkLoadInstance(instance);
}
void VulkanLoader::load_device(VkDevice device){
    vulkan_device_ = device;
    volkLoadDevice(device);
}

}