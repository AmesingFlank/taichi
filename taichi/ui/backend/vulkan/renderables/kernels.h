#include "taichi/ui/backend/vulkan/vertex.h"

void update_renderables_vertices_cuda(Vertex *vbo,
                                      float *vertices,
                                      int num_vertices,
                                      int num_components);
void update_renderables_vertices_x64(Vertex *vbo,
                                     float *vertices,
                                     int num_vertices,
                                     int num_components);

void update_renderables_indices_cuda(int *ibo, int *indices, int num_indices);
void update_renderables_indices_x64(int *ibo, int *indices, int num_indices);

void update_renderables_indices_unindexed_cuda(int *ibo, int num_indices);
void update_renderables_indices_unindexed_x64(int *ibo, int num_indices);

void update_renderables_colors_cuda(Vertex *vbo,
                                    float *colors,
                                    int num_vertices);
void update_renderables_colors_x64(Vertex *vbo,
                                   float *colors,
                                   int num_vertices);

void update_renderables_normals_cuda(Vertex *vbo,
                                     float *normals,
                                     int num_vertices);
void update_renderables_normals_x64(Vertex *vbo,
                                    float *normals,
                                    int num_vertices);

// We implement lines by generating rectangles. Note there this requires careful
// treatment of aspect ratios.
void update_lines_vbo_cuda(Vertex *vbo,
                           int *ibo,
                           float *vertices,
                           int N,
                           float width,
                           float aspect_ratio,
                           float *colors,
                           bool use_per_vertex_color);
void update_lines_vbo_x64(Vertex *vbo,
                          int *ibo,
                          float *vertices,
                          int N,
                          float width,
                          float aspect_ratio,
                          float *colors,
                          bool use_per_vertex_color);

template <typename T>
void copy_to_texture_fuffer_cuda(T *src,
                                 void *surface,
                                 int width,
                                 int height,
                                 int actual_width,
                                 int actual_height,
                                 int channels);

template <typename T>
void copy_to_texture_fuffer_x64(T *src,
                                unsigned char *dest,
                                int width,
                                int height,
                                int actual_width,
                                int actual_height,
                                int channels);
