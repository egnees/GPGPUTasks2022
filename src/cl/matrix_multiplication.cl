#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define TILE 32
#define WPT 8

__kernel void matrix_multiplication(__global const float* a, __global const float* b, __global float* c, unsigned int M, unsigned int K, unsigned int N)
{
    unsigned int i = get_global_id(1);
    unsigned int j = get_global_id(0);
    unsigned int wpt_j = WPT * j;
    unsigned int local_i = get_local_id(1);
    unsigned int local_j = get_local_id(0);
    unsigned int wpt_local_j = WPT * local_j;
    __local float mem_a[TILE][TILE];
    __local float mem_b[TILE][TILE];
    float accum[WPT];
    for (int w = 0; w < WPT; ++w) accum[w] = 0.0f;
    const int tiles_number = K / TILE;
    for (int k = 0; k < tiles_number; ++k) {
        for (int w = 0; w < WPT; ++w) {
          mem_a[local_i][wpt_local_j + w] = a[i * K + k * TILE + wpt_local_j + w];
          mem_b[local_i][wpt_local_j + w] = b[(k * TILE + local_i) * N + wpt_j + w];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int t = 0; t < TILE; ++t) {
          for (int w = 0; w < WPT; ++w) {
            accum[w] += mem_a[local_i][t] * mem_b[t][wpt_local_j + w];
          }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    for (int w = 0; w < WPT; ++w) c[i * N + wpt_j + w] = accum[w];
}