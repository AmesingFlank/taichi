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
#include "../../../common/canvas_base.h"


namespace vulkan{


struct Circles:public Renderable{

public:
    Circles(AppContext* app_context);
    void update_data(const CirclesInfo& info);

private:
    
    struct UniformBufferObject {
        alignas(16) glm::vec3 color;
        int use_per_vertex_color;
        float radius;
    };

    void init_circles(AppContext* app_context, int vertices_count);

    void update_ubo(glm::vec3 color,bool use_per_vertex_color,float radius);

    virtual void create_descriptor_set_layout()override ;

    virtual void create_descriptor_sets() override;

 };


}