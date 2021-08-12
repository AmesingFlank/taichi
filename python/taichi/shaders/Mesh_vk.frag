#version 450


layout(location = 0) in vec3 fragPos;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec2 fragTexCoord;



layout(location = 0) out vec4 outColor;

struct SceneUBO{
    vec3 camera_pos;
    mat4 view;
    mat4 projection;
    vec3 ambient_light;
    int point_light_count;
};

struct PointLight{
    vec3 pos;
    vec3 color;
};

layout(binding = 0) uniform UBO {
    SceneUBO scene;
    vec3 color;
    int use_per_vertex_color;
    PointLight point_lights[];
} ubo;

layout(location = 3) in vec3 selectedColor;

vec3 lambertian(){
    
    vec3 ambient = ubo.scene.ambient_light * selectedColor;
    vec3 result = ambient;

    for(int i = 0;i<ubo.scene.point_light_count;++i){
        vec3 lightColor = ubo.point_lights.color[i];

        vec3 lightDir = normalize(ubo.point_lights.pos[i] - fragPos);
        vec3 normal = normalize(fragNormal);
        vec3 diffuse = abs(dot(lightDir, normal)) * selectedColor * lightColor;
        result += diffuse;
    }

    return result;
}

void main() {
    outColor = vec4(lambertian(),1);
}