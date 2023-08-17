
//
// Created by Sergei Yakovlev <3gnees@gmail.com> on 17.08.2023.
//

/*
 * f=(x) e^{cos(x)*sin(2*x)}*log(1+x), x > 0
 * Computes \sum_{i=0}^n f(a[i])
 * */

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "cl/complex_function_cl.h"

template<typename T>
void raiseFail(const T &a, const T &b, std::string message, std::string filename, int line)
{
    if (a != b) {
        std::cerr << message << " But " << a << " != " << b << ", " << filename << ":" << line << std::endl;
        throw std::runtime_error(message);
    }
}

#define EXPECT_THE_SAME(a, b, message) raiseFail(a, b, message, __FILE__, __LINE__)

#include <libutils/timer.h>
#include <libutils/fast_random.h>
#include <libutils/misc.h>

#include <libgpu/device.h>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>

inline unsigned int f(unsigned int x) {
    return (unsigned int) (powf(2.71828f, cosf((float) x) * sinf(2 * (float) x)));
}

int main(int argc, char** argv) {
    const unsigned int n = 100 * 1000 * 1000;
    std::vector<unsigned int> a(n + 1);
    FastRandom random(42);
    for (int i = 0; i < n; ++i) {
        a[i] = 0;
    }
    a[n] = 0.0f;

    unsigned int ref_sum = 0;
    for (int i = 0; i < n; ++i) {
        ref_sum += f(a[i]);
    }

    const unsigned int benchmark_iters = 5;
    const unsigned int f_ops = 5;

    {
        timer t;
        for (unsigned int iter = 0; iter < benchmark_iters; ++iter) {
            unsigned int sum = 0;
            for (int i = 0; i < n; ++i) {
                sum += f(a[i]);
            }
            EXPECT_THE_SAME(sum, ref_sum, "CPU results is not consistent!");
            t.nextLap();
        }

        std::cout << "CPU: " << t.lapAvg() << " +- " << t.lapStd() << " seconds.\n";
        std::cout << "CPU: " << (f_ops * n / 1000.0 / 1000.0 / 1000.0) / t.lapAvg() << " GFlops.\n";
    }

    gpu::Device device = gpu::chooseGPUDevice(argc, argv);
    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

    {
        // use default stupid kernel

        ocl::Kernel kernel(complex_function_kernel, complex_function_kernel_length, "complex_function_kernel");
        kernel.compile(true);

        gpu::gpu_mem_32u vram;
        vram.resizeN(n + 1);
        vram.writeN(a.data(), n + 1);

        const unsigned int group_size = 128;
        const unsigned int one_item_work_size = 128;
        const unsigned int rounded_size = gpu::divup(n, group_size) * group_size;

        timer t;
        for (unsigned int iter = 0; iter < benchmark_iters; ++iter) {
            kernel.exec(gpu::WorkSize(group_size, rounded_size / one_item_work_size), vram, n, one_item_work_size);
            if (iter == 0) {
                unsigned int sum;
                vram.readN(&sum, 1, n);
                EXPECT_THE_SAME(sum, ref_sum, "GPU results is not consistent!");
            }
            t.nextLap();
        }

        std::cout << "GPU (stupid): " << t.lapAvg() << " +- " << t.lapStd() << " seconds.\n";
        std::cout << "GPU (stupid): " << (f_ops * n / 1000.0 / 1000.0 / 1000.0) / t.lapAvg() << " GFlops.\n";
    }

    {
        // use default stupid kernel

        ocl::Kernel kernel(complex_function_kernel, complex_function_kernel_length, "complex_function_kernel_smart");
        kernel.compile(true);

        gpu::gpu_mem_32u vram;
        vram.resizeN(n + 1);
        vram.writeN(a.data(), n + 1);

        const unsigned int group_size = 128;
        const unsigned int rounded_size = gpu::divup(n, group_size) * group_size;

        timer t;
        for (unsigned int iter = 0; iter < benchmark_iters; ++iter) {
            kernel.exec(gpu::WorkSize(group_size, rounded_size), vram, n);
            if (iter == 0) {
                unsigned int sum;
                vram.readN(&sum, 1, n);
                EXPECT_THE_SAME(sum, ref_sum, "GPU results is not consistent!");
            }
            t.nextLap();
        }

        std::cout << "GPU (smart): " << t.lapAvg() << " +- " << t.lapStd() << " seconds.\n";
        std::cout << "GPU (smart): " << (f_ops * n / 1000.0 / 1000.0 / 1000.0) / t.lapAvg() << " GFlops.\n";
    }

    /*
     * Smart works slower than stupid
     * Probably, this is because bad function nature
     * If we were talking about matrix multiplication for example, the smart approach would be faster probably
     */

    return 0;
}