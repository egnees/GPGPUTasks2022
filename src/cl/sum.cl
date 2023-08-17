#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable

#define GROUP_SIZE 128

__kernel void sum(__global unsigned int* as, unsigned int size) {
    const unsigned int group_id = get_group_id(0);
    const unsigned int local_id = get_local_id(0);
    const unsigned int i = group_id * GROUP_SIZE + local_id;
    __local unsigned int local_as[GROUP_SIZE];
    if (i < size) {
        local_as[local_id] = as[i];
    } else {
        local_as[local_id] = 0;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (get_local_id(0) == 0) {
        // main worker
        unsigned int sum = 0;
        for (unsigned int j = 0; j < GROUP_SIZE; ++j) {
            sum += local_as[j];
        }
        atomic_add(&as[size], sum);
    }
}