#include "set_image.h"
#include "../vulkan_cuda_interop.h"
#include "../vulkan_cuda_interop.h"
#include "../../../utils/utils.h"
#include "../../../common/constants.h"

namespace vulkan {


template<typename T>
__device__ __host__
inline unsigned char get_color_value(T x);

template<>
__device__ __host__
inline unsigned char get_color_value<unsigned char>(unsigned char x){
    return x;
}

template<>
__device__ __host__
inline unsigned char get_color_value<float>(float x){
    x = max(0.f,min(1.f,x));
    return (unsigned char)(x * 255);
}


template<typename T>
__global__
void copy_to_texture_fuffer_cuda(T* src, CUsurfObject surface, int width, int height, int actual_width, int actual_height, int channels){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= width * height) return;

    int y = i / width;
    int x = i % width;

    T* src_base_addr = src + (x*actual_height + y) * channels;
    uchar4 data = make_uchar4(0,0,0,0);
    
    data.x = get_color_value<T>(src_base_addr[0]);
    data.y = get_color_value<T>(src_base_addr[1]);
    data.z  = get_color_value<T>(src_base_addr[2]);
    data.w = 255;
    
    surf2Dwrite(data, surface, x* sizeof(uchar4), y);
}

template<typename T>
void copy_to_texture_fuffer_x64(T* src, uchar4* dest, int width, int height, int actual_width, int actual_height, int channels){
    for(int i = 0;i<width * height;++i){
        int y = i / width;
        int x = i % width;

        T* src_base_addr = src + (x*actual_height + y) * channels;
        uchar4 data = make_uchar4(0,0,0,0);
        
        data.x = get_color_value<T>(src_base_addr[0]);
        data.y = get_color_value<T>(src_base_addr[1]);
        data.z  = get_color_value<T>(src_base_addr[2]);
        data.w = 255;

        dest[y * width + x] = data;
    }
    
}

void SetImage::update_data(const SetImageInfo& info){
    const FieldInfo& img = info.img;
    if(img.shape.size() != 2){
        throw std::runtime_error("for set image, the image should have exactly two axis. e,g, ti.Vector.field(3,ti.u8,(1920,1080) ");
    }
    if( (img.matrix_rows != 3 && img.matrix_rows != 4) || img.matrix_cols != 1 ){
        throw std::runtime_error("for set image, the image should either a 3-D vector field (RGB) or a 4D vector field (RGBA) ");
    }
    int new_width  = img.shape[0];
    int new_height =  img.shape[1];

    if(new_width != width || new_height != height){
        cleanup_swap_chain();
        cleanup();
        init_set_image(app_context_,new_width,new_height);
    }
    
    int actual_width = next_power_of_2(width);
    int actual_height = next_power_of_2(height);

    int pixels = width * height;
    int num_blocks,num_threads;
    set_num_blocks_threads(pixels,num_blocks,num_threads);

    if(img.field_source == FIELD_SOURCE_CUDA){
        if(img.dtype == DTYPE_U8){
            copy_to_texture_fuffer_cuda<<<num_blocks,num_threads>>>((unsigned char*)img.data,(CUsurfObject)texture_surface_,width,height,actual_width,actual_height,img.matrix_rows);
        }
        else if (img.dtype == DTYPE_F32){
            copy_to_texture_fuffer_cuda<<<num_blocks,num_threads>>>((float*)img.data,(CUsurfObject)texture_surface_,width,height,actual_width,actual_height,img.matrix_rows);
        }
        else{
            throw std::runtime_error("for set image, dtype must be u8 or f32");
        }
        CHECK_CUDA_ERROR("copy to texture\n");
    }
    else if(img.field_source == FIELD_SOURCE_X64){
        transition_image_layout(texture_image_, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,app_context_->command_pool,app_context_->device,app_context_->graphics_queue);
        
        MappedMemory mapped_buffer(app_context_->device, staging_buffer_memory_ , pixels * sizeof(uchar4));
 
        if(img.dtype == DTYPE_U8){
            copy_to_texture_fuffer_x64 ((unsigned char*)img.data,(uchar4*)mapped_buffer.data,width,height,actual_width,actual_height,img.matrix_rows);
        }
        else if (img.dtype == DTYPE_F32){
            copy_to_texture_fuffer_x64((float*)img.data,(uchar4*)mapped_buffer.data,width,height,actual_width,actual_height,img.matrix_rows);
        }
        else{
            throw std::runtime_error("for set image, dtype must be u8 or f32");
        } 

        copy_buffer_to_image(staging_buffer_, texture_image_, width,height,app_context_->command_pool,app_context_->device,app_context_->graphics_queue);

        transition_image_layout(texture_image_, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,app_context_->command_pool,app_context_->device,app_context_->graphics_queue);
    }
    else{
        throw std::runtime_error("unsupported field source");
    }

    
}


SetImage::SetImage(AppContext* app_context){
    init_set_image(app_context,1,1);
}

void SetImage::init_set_image(AppContext* app_context,int img_width, int img_height){
    RenderableConfig config = {
        6,
        6,
        1,
        app_context ->config.package_path + "/shaders/SetImage_vk_vert.spv",
        "",
        app_context ->config.package_path + "/shaders/SetImage_vk_frag.spv",
        TopologyType::TriangleList,
    };


    Renderable::init(config,app_context);

    width = img_width;
    height = img_height;

    create_texture_image_(width,height);  
    create_texture_image_view_(); 
    create_texture_sampler_(); 

    Renderable::init_render_resources();  

    update_vertex_buffer_();  
    update_index_buffer_();  
}



void SetImage::create_texture_image_(int width, int height) {
        
    VkDeviceSize image_size = (int)(width * height * 4);

    create_image(width,height, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, texture_image_, texture_image_memory_,app_context_->device,app_context_->physical_device);

    transition_image_layout(texture_image_, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,app_context_->command_pool,app_context_->device,app_context_->graphics_queue);
    transition_image_layout(texture_image_, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,app_context_->command_pool,app_context_->device,app_context_->graphics_queue);

    if(app_context_->config.ti_arch == ARCH_CUDA){
        VkMemoryRequirements mem_requirements;
        vkGetImageMemoryRequirements(app_context_->device, texture_image_, &mem_requirements);

        auto handle = get_device_mem_handle(texture_image_memory_,app_context_->device);
        CUexternalMemory external_mem = import_vk_memory_object_from_handle(handle,mem_requirements.size,true);

        texture_surface_ = (uint64_t)get_image_surface_object_of_external_memory(external_mem,width,height);
    }
    create_buffer(image_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, staging_buffer_, staging_buffer_memory_,app_context_->device,app_context_->physical_device);
    
}



void SetImage::create_texture_image_view_() {
    texture_image_view_ = create_image_view(texture_image_, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_ASPECT_COLOR_BIT,app_context_->device);
}

void SetImage::create_texture_sampler_() {
    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(app_context_->physical_device, &properties);

    VkSamplerCreateInfo sampler_info{};
    sampler_info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    sampler_info.magFilter = VK_FILTER_LINEAR;
    sampler_info.minFilter = VK_FILTER_LINEAR;
    sampler_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sampler_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sampler_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sampler_info.anisotropyEnable = VK_TRUE;
    sampler_info.maxAnisotropy = properties.limits.maxSamplerAnisotropy;
    sampler_info.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    sampler_info.unnormalizedCoordinates = VK_FALSE;
    sampler_info.compareEnable = VK_FALSE;
    sampler_info.compareOp = VK_COMPARE_OP_ALWAYS;
    sampler_info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;

    if (vkCreateSampler(app_context_->device, &sampler_info, nullptr, &texture_sampler_) != VK_SUCCESS) {
        throw std::runtime_error("failed to create texture sampler!");
    }
}

void SetImage::update_vertex_buffer_(){
    const std::vector<Vertex> vertices = {
        {{-1.f,-1.f,0.f}, {0.f,0.f,1.f},  {0.f,1.f},{1.f,1.f,1.f}},
        {{-1.f,1.f,0.f}, {0.f,0.f,1.f}, {0.f,0.f},{1.f,1.f,1.f}},
        {{1.f,1.f,0.f}, {0.f,0.f,1.f}, {1.f,0.f},{1.f,1.f,1.f}},

        {{ -1.f,-1.f,0.f},{0.f,0.f,1.f}, { 0.f,1.f},{1.f,1.f,1.f}},
        {{ 1.f,1.f,0.f}, {0.f,0.f,1.f}, {1.f,0.f},{1.f,1.f,1.f}},
        {{ 1.f,-1.f,0.f},{0.f,0.f,1.f},  {1.f,1.f},{1.f,1.f,1.f}},
    };

    {
        MappedMemory mapped_vbo(app_context_->device, staging_vertex_buffer_memory_ , config_.vertices_count * sizeof(Vertex));
        memcpy(mapped_vbo.data, vertices.data(), (size_t) config_.vertices_count * sizeof(Vertex));
    }

    copy_buffer(staging_vertex_buffer_, vertex_buffer_, config_.vertices_count * sizeof(Vertex), app_context_ -> command_pool, app_context_ -> device, app_context_ -> graphics_queue) ;

}

void SetImage::update_index_buffer_() {
    const std::vector<uint32_t> indices = {
        0, 1, 2, 3,4,5,
    };
    {
        MappedMemory mapped_ibo(app_context_->device, staging_index_buffer_memory_ , config_.indices_count * sizeof(int));
        memcpy(mapped_ibo.data, indices.data(), (size_t) config_.indices_count * sizeof(int));
    }
    
    copy_buffer(staging_index_buffer_, index_buffer_, config_.indices_count * sizeof(int), app_context_ -> command_pool, app_context_ -> device, app_context_ -> graphics_queue) ;

}


void SetImage::create_descriptor_set_layout()  {
    VkDescriptorSetLayoutBinding ubo_layout_binding{};
    ubo_layout_binding.binding = 0;
    ubo_layout_binding.descriptorCount = 1;
    ubo_layout_binding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    ubo_layout_binding.pImmutableSamplers = nullptr;
    ubo_layout_binding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

    VkDescriptorSetLayoutBinding sampler_layout_binding{};
    sampler_layout_binding.binding = 1;
    sampler_layout_binding.descriptorCount = 1;
    sampler_layout_binding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    sampler_layout_binding.pImmutableSamplers = nullptr;
    sampler_layout_binding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    std::array<VkDescriptorSetLayoutBinding, 2> bindings = {ubo_layout_binding, sampler_layout_binding};
    VkDescriptorSetLayoutCreateInfo layout_info{};
    layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layout_info.bindingCount = static_cast<uint32_t>(bindings.size());
    layout_info.pBindings = bindings.data();

    if (vkCreateDescriptorSetLayout(app_context_->device, &layout_info, nullptr, &descriptor_set_layout_) != VK_SUCCESS) {
        throw std::runtime_error("failed to create descriptor set layout!");
    }
}

void SetImage::create_descriptor_sets()  {
    std::vector<VkDescriptorSetLayout> layouts(app_context_->get_swap_chain_size(), descriptor_set_layout_);

    VkDescriptorSetAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    alloc_info.descriptorPool = descriptor_pool_;
    alloc_info.descriptorSetCount = app_context_->get_swap_chain_size();
    alloc_info.pSetLayouts = layouts.data();

    descriptor_sets_.resize(app_context_->get_swap_chain_size());

    if (vkAllocateDescriptorSets(app_context_->device, &alloc_info, descriptor_sets_.data() ) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate descriptor sets!");
    }

    for (size_t i = 0; i < app_context_->get_swap_chain_size(); i++) {
        VkDescriptorBufferInfo buffer_info{};
        buffer_info.buffer = uniform_buffers_[i] ;
        buffer_info.offset = 0;
        buffer_info.range = config_.ubo_size;

        VkDescriptorImageInfo image_info{};
        image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        image_info.imageView = texture_image_view_;
        image_info.sampler = texture_sampler_;

        std::array<VkWriteDescriptorSet, 2> descriptor_writes{};

        descriptor_writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptor_writes[0].dstSet = descriptor_sets_[i] ;
        descriptor_writes[0].dstBinding = 0;
        descriptor_writes[0].dstArrayElement = 0;
        descriptor_writes[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        descriptor_writes[0].descriptorCount = 1;
        descriptor_writes[0].pBufferInfo = &buffer_info;

        descriptor_writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptor_writes[1].dstSet = descriptor_sets_[i] ;
        descriptor_writes[1].dstBinding = 1;
        descriptor_writes[1].dstArrayElement = 0;
        descriptor_writes[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        descriptor_writes[1].descriptorCount = 1;
        descriptor_writes[1].pImageInfo = &image_info;

        vkUpdateDescriptorSets(app_context_->device, static_cast<uint32_t>(descriptor_writes.size()), descriptor_writes.data(), 0, nullptr);
    }

    
}

void SetImage::cleanup() {
    Renderable::cleanup();

    vkDestroySampler(app_context_->device, texture_sampler_, nullptr);
    vkDestroyImageView(app_context_->device, texture_image_view_, nullptr);

    vkDestroyImage(app_context_->device, texture_image_, nullptr);
    vkFreeMemory(app_context_->device, texture_image_memory_, nullptr);

    vkDestroyBuffer(app_context_->device, staging_buffer_, nullptr);
    vkFreeMemory(app_context_->device, staging_buffer_memory_, nullptr);
}


}//namespace vulkan