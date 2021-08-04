#pragma once
#include "vulkan_utils.h"
#include "../../common/app_config.h"

namespace vulkan{





struct AppContext{
    VkInstance instance;
    VkDevice device;
    VkPhysicalDevice physical_device;
    QueueFamilyIndices queue_family_indices;
    VkQueue graphics_queue;
    VkQueue present_queue;
    
    VkCommandPool command_pool;
    
    VkRenderPass render_pass;

    VkDebugUtilsMessengerEXT debug_messenger;

    AppConfig config;

    bool enable_validation_layers = true;


    struct SwapChain* swap_chain;

    void init(QueueFamilyIndices queue_family_indices_);

    void create_instance(std::vector<const char*> extensions);

    void pick_physical_device(VkSurfaceKHR& surface);

    void create_logical_device() ;

    void create_render_pass(VkRenderPass& render_pass,VkImageLayout final_color_layout );
    void create_render_passes();

    void creat_command_pool();

    

    bool is_device_suitable(VkPhysicalDevice physical_device,VkSurfaceKHR surface);

    bool check_device_extension_support(VkPhysicalDevice device);

    bool check_validation_layer_support();

    void setup_debug_messenger();

    void populate_debug_messenger_create_info(VkDebugUtilsMessengerCreateInfoEXT& create_info);

    static VKAPI_ATTR VkBool32 VKAPI_CALL debug_callback(VkDebugUtilsMessageSeverityFlagBitsEXT severity, VkDebugUtilsMessageTypeFlagsEXT message_type, const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void* user_data);

    void cleanup_swap_chain();

    void cleanup();

    void recreate_swap_chain();

    int get_swap_chain_size();
};


}