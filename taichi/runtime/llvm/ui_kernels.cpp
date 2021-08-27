

extern "C" {
int thread_idx() {
  return 0;
}

int block_idx() {
  return 0;
}

int block_dim() {
  return 0;
}

int grid_dim() {
  return 0;
}


void update_renderables_vertices(float *vbo,
                                                      int stride,
                                                      float *data,
                                                      int num_vertices,
                                                      int num_components,
                                                      int offset) {
  int i = block_idx() *  block_dim() + thread_idx();

  //for(int i = i0; i < num_vertices; i += grid_dim() * block_dim()){

    float *dst = vbo + i*stride + offset;
    float *src = data + i * num_components;
    for (int c = 0; c < num_components; ++c) {
        dst[c] = src[c];
    }

  //}

  
}
}

