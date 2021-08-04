#version 450


layout(location = 0) in vec3 fragPos;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec2 fragTexCoord;



layout(location = 0) out vec4 outColor;

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

layout(location = 3) in vec3 selectedColor;

vec3 blinnPhong(){
    
    vec3 ambient = ubo.scene.ambient_light * selectedColor;
    vec3 result = ambient;

    for(int i = 0;i<ubo.scene.point_light_count;++i){
        vec3 lightColor = ubo.scene.point_light_colors[i];

        vec3 lightDir = normalize(ubo.scene.point_light_positions[i] - fragPos);
        vec3 normal = normalize(fragNormal);
        vec3 diffuse = abs(dot(lightDir, normal)) * selectedColor * lightColor;

        vec3 viewPos = ubo.scene.camera_pos;
        vec3 viewDir = normalize(viewPos - fragPos);
        vec3 reflectDir = reflect(-lightDir, normal);
        
        vec3 halfwayDir = normalize(lightDir + viewDir);  
        float spec = pow(abs(dot(normal, halfwayDir)), ubo.shininess);
        
        vec3 specular = lightColor * spec; 
        
        result += diffuse + specular;
    }

    return result;
}

void main() {
    outColor = vec4(blinnPhong(),1);
}