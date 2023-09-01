#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define TILE 16

__kernel void matrix_transpose(__global const float* matrix, __global float* matrix_t, unsigned int M, unsigned int K)
{
  unsigned int j = get_global_id(0);
  unsigned int i = get_global_id(1);
  unsigned local_j = get_local_id(0);
  unsigned local_i = get_local_id(1);
  __local float mem[TILE * TILE];
  mem[local_j * TILE + (local_i + local_j) % TILE] = matrix[i * K + j];
  barrier(CLK_LOCAL_MEM_FENCE);
  matrix_t[(j/TILE*TILE + local_i) * M + i/TILE*TILE + local_j] = mem[local_i * TILE + (local_j + local_i) % TILE];
}