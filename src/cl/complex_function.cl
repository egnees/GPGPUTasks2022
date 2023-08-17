#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable

//#define GROUP_SIZE 128

// \sum_{i=0}^n e^cos(a[i])*sin(a[i] * 2)*log(1+a[i]), a[i] > 0

__kernel void complex_function_kernel(__global unsigned int* vram, unsigned int n, unsigned int work_size) {
    unsigned int local_id = get_local_id(0);
    unsigned int group_id = get_group_id(0);
    unsigned int group_size = get_local_size(0);
    unsigned int sum = 0;
    for (unsigned int i = 0; i < work_size; ++i) {
        unsigned int j = group_id * group_size * work_size + group_size * i + local_id;
        if (j < n) {
            float x = vram[j];
            sum += (unsigned int) (pow(2.71828f, cos(x) * sin(2.0f * x)));
        }
    }
    atomic_add(&vram[n], sum);
}

#define GROUP_SIZE 128

__kernel void complex_function_kernel_smart(__global unsigned int* vram, unsigned int n) {
    unsigned int group_id = get_group_id(0);
    unsigned int local_id = get_local_id(0);
    unsigned int start_i = group_id * GROUP_SIZE;
    __local unsigned int local_mem[GROUP_SIZE];

    if (start_i + local_id < n) {
        float x = vram[start_i + local_id];
        local_mem[local_id] = (unsigned int) (pow(2.71828f, cos(x) * sin(2.0f * x)));
    } else {
        local_mem[local_id] = 0;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int size = GROUP_SIZE; size > 1; size /= 2) {
        if (2 * local_id < size) {
            unsigned int x1 = local_mem[local_id];
            unsigned int x2 = local_mem[local_id + size / 2];
            local_mem[local_id] = x1 + x2;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (local_id == 0) {
        atomic_add(&vram[n], local_mem[0]);
    }
}