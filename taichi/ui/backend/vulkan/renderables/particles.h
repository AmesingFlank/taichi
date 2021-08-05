#pragma once



#include <iostream>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <chrono>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <array>
#include <optional>
#include <set>
#include "../../../utils/utils.h"
#include "../vertex.h"
#include "../vulkan_utils.h"
#include "../app_context.h"
#include "../swap_chain.h"
#include "../renderable.h"
#include "../../../common/field_info.h"
#include "../scene.h"

namespace vulkan{


class Particles:public Renderable{

public:

    Particles(AppContext* app_context);

    void update_data(const ParticlesInfo& info,const Scene& scene);

private:

    struct UniformBufferObject {
        Scene::SceneUniformBuffer scene;
        alignas(16) glm::vec3 color;
        int use_per_vertex_color;
        float shininess;
        float radius;
        float window_width;
        float window_height;
        float tan_half_fov;
    };

    
    void init_particles(AppContext* app_context, int vertices_count);

    void update_ubo(glm::vec3 color,bool use_per_vertex_color,float radius,float shininess,const Scene& scene);

    virtual void create_descriptor_set_layout()override ;

    virtual void create_descriptor_sets() override;
    
};


}