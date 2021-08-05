#pragma once
#include "vulkan_utils.h"
#include "../../common/app_config.h"
#include <memory>
#include "taichi/backends/vulkan/vulkan_api.h"
#include "taichi/backends/vulkan/loader.h"
#include "swap_chain.h"

namespace vulkan{


struct AppContext{

    VkInstance instance;
    VkDevice device;
    VkPhysicalDevice physical_device;
    taichi::lang::vulkan::VulkanQueueFamilyIndices queue_family_indices;
    VkQueue graphics_queue;
    VkQueue present_queue;
    
    VkCommandPool command_pool;
    
    std::unique_ptr<taichi::lang::vulkan::EmbeddedVulkanDevice> vulkan_device;

    VkRenderPass render_pass;

    VkDebugUtilsMessengerEXT debug_messenger;

    AppConfig config;

    GLFWwindow* glfw_window;

    SwapChain swap_chain;

    void init();
 
    void create_render_pass(VkRenderPass& render_pass,VkImageLayout final_color_layout );
    void create_render_passes();
     
    void cleanup_swap_chain();

    void cleanup();

    void recreate_swap_chain();

    int get_swap_chain_size();
};


}