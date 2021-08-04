#include "window.h"

namespace vulkan{

Window::Window(AppConfig config): WindowBase(config){
    if(!VulkanLoader::instance().init()){
        throw std::runtime_error("Error initializing vulkan");
    }
    app_context_.config = config;
    init();
}

void Window::init() {
    init_window();
    init_vulkan();  
    gui_.init(&app_context_,glfw_window_);  
    prepare_for_next_frame(); 
}

void Window::show()  {
    draw_frame();
    present_frame();
    WindowBase::show();
    prepare_for_next_frame();
}

void Window::prepare_for_next_frame(){
    update_image_index();
    canvas_ -> prepare_for_next_frame();
    gui_.prepare_for_next_frame();
}

CanvasBase* Window::get_canvas() {
    return canvas_.get();
}

GuiBase* Window::GUI()  {
    return &gui_;
}

void Window::init_window() {
    glfwSetFramebufferSizeCallback(glfw_window_, framebuffer_resize_callback);
}

void Window::framebuffer_resize_callback(GLFWwindow* glfw_window_, int width, int height) {
    auto window = reinterpret_cast<Window*>(glfwGetWindowUserPointer(glfw_window_));
    window -> recreate_swap_chain();
}

void Window::init_vulkan() {
    swap_chain_.app_context = &app_context_;
    app_context_.swap_chain = &swap_chain_;

    app_context_.create_instance(get_required_extensions());
    app_context_.setup_debug_messenger();
    create_surface();
    app_context_.pick_physical_device(swap_chain_.surface);

    printf("phys dev %d\n",app_context_.physical_device);

    app_context_.queue_family_indices = find_queue_families(app_context_.physical_device,swap_chain_.surface);

    app_context_.create_logical_device();

    swap_chain_.create_swap_chain();  
    swap_chain_.create_image_views();  
    swap_chain_.create_depth_resources();  
    swap_chain_.create_sync_objects();  

    app_context_.create_render_passes(); 
    app_context_.creat_command_pool(); 

    swap_chain_.create_framebuffers();  

    canvas_ = std::make_unique<Canvas>(&app_context_); 
}

void Window::cleanup_swap_chain() {
    swap_chain_.cleanup_swap_chain();

    canvas_ -> cleanup_swap_chain();
    app_context_.cleanup_swap_chain();
}

void Window::cleanup() {
    gui_.cleanup();
    cleanup_swap_chain();


    canvas_ -> cleanup();

    swap_chain_.cleanup();
    app_context_ . cleanup();

    glfwTerminate();
}

void Window::recreate_swap_chain() {

    int width = 0, height = 0;
    glfwGetFramebufferSize(glfw_window_, &width, &height);
    while (width == 0 || height == 0) {
        glfwGetFramebufferSize(glfw_window_, &width, &height);
        glfwWaitEvents();
    }
    app_context_.config.width = width;
    app_context_.config.height = height;

    vkDeviceWaitIdle(app_context_.device);

    cleanup_swap_chain();

    app_context_.recreate_swap_chain();

    swap_chain_.recreate_swap_chain();
    
    canvas_ -> recreate_swap_chain();

}


void Window::create_surface() {
    if (glfwCreateWindowSurface(app_context_.instance, glfw_window_, nullptr, &swap_chain_.surface) != VK_SUCCESS) {
        throw std::runtime_error("failed to create window surface!");
    }
}


void Window::draw_frame() {
    canvas_ ->draw_frame(&gui_);
}



void Window::present_frame(){
    swap_chain_.present_frame();
    if(swap_chain_.requires_recreate){
        recreate_swap_chain();
    }
}

void Window::update_image_index(){
    swap_chain_.update_image_index();
}



std::vector<const char*> Window::get_required_extensions() {
    uint32_t glfw_ext_count = 0;
    const char** glfw_extensions;
    glfw_extensions = glfwGetRequiredInstanceExtensions(&glfw_ext_count);

    std::vector<const char*> extensions(glfw_extensions, glfw_extensions + glfw_ext_count);

    extensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
    extensions.push_back(VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME);
    extensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME);
    extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    

    return extensions;
}


Window::~Window(){
    cleanup();
}

} // namespace vulkan