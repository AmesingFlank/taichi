#version 450
layout (triangles) in;
layout (triangle_strip, max_vertices = 3) out;

layout(location = 0) in vec3 fragPosIn[];
layout(location = 1) in vec3 fragNormalIn[];
layout(location = 2) in vec2 fragTexCoordIn[];
layout(location = 3) in vec3 selectedColorIn[];

layout(location = 0) out vec3 fragPos;
layout(location = 1) out vec3 fragNormal;
layout(location = 2) out vec2 fragTexCoord;
layout(location = 3) out vec3 selectedColor;

struct SceneUBO{
    vec3 camera_pos;
    mat4 view;
    mat4 projection;
    vec3 point_light_positions[16];
    vec3 point_light_colors[16];
    int point_light_count;
    vec3 ambient_light;
};

layout(binding = 0) uniform UBO {
    SceneUBO scene;
    vec3 color;
    int use_per_vertex_color;
    float shininess;
    int need_normal_generation;
} ubo;

vec3 gen_normal(){
   vec3 a = vec3(fragPosIn[0]) - vec3(fragPosIn[1]);
   vec3 b = vec3(fragPosIn[2]) - vec3(fragPosIn[1]);
   return normalize(cross(a, b));
}  

void main() {   

    for(int i = 0;i<3;++i){
        fragPos = fragPosIn[i];
        fragNormal = fragNormalIn[i];
        fragTexCoord = fragTexCoordIn[i];
        selectedColor = selectedColorIn[i];
        gl_Position = gl_in[i].gl_Position;

        if(ubo.need_normal_generation != 0){
            fragNormal = gen_normal();
        }

        EmitVertex();
    }
    
    EndPrimitive();
}  