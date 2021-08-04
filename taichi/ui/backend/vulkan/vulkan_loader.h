#pragma once
#include "../../utils/utils.h"

namespace vulkan{

class VulkanLoader {
public:
    static VulkanLoader& instance()
    {
        static VulkanLoader   instance; 
        return instance;
    }

    static PFN_vkVoidFunction load_function(const char * name, void * userData){
        auto result = vkGetInstanceProcAddr(VulkanLoader::instance().vulkan_instance_, name);
        //printf("loading %s \n",name);
        if(result == nullptr){
            printf("%s is nullptr\n",name);
        }
        return result;
    }

public:
    VulkanLoader(VulkanLoader const&) = delete;
    void operator=(VulkanLoader const&) = delete;

    void load_instance(VkInstance instance_);
    void load_device(VkDevice device_);
    bool init();

private:

    bool initialized_;

    VulkanLoader();

    VkInstance vulkan_instance_;
    VkDevice vulkan_device_;
};

}