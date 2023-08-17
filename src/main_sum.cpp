#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>

#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>

#include "cl/sum_cl.h"

template<typename T>
void raiseFail(const T &a, const T &b, std::string message, std::string filename, int line)
{
    if (a != b) {
        std::cerr << message << " But " << a << " != " << b << ", " << filename << ":" << line << std::endl;
        throw std::runtime_error(message);
    }
}

#define EXPECT_THE_SAME(a, b, message) raiseFail(a, b, message, __FILE__, __LINE__)


int main(int argc, char **argv)
{
    int benchmarkingIters = 10;

    unsigned int reference_sum = 0;
    unsigned int n = 100*1000*1000;
    std::vector<unsigned int> as(n, 0);
    FastRandom r(42);
    for (int i = 0; i < n; ++i) {
        as[i] = (unsigned int) r.next(0, std::numeric_limits<unsigned int>::max() / n);
        reference_sum += as[i];
    }

    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            unsigned int sum = 0;
            for (int i = 0; i < n; ++i) {
                sum += as[i];
            }
            EXPECT_THE_SAME(reference_sum, sum, "CPU result should be consistent!");
            t.nextLap();
        }
        std::cout << "CPU:     " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU:     " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
        std::cout << "CPU:     " << (n/1000.0/1000.0/1000.0) / t.lapAvg() << " GFlops" << std::endl;
    }

    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            unsigned int sum = 0;
//            #pragma omp parallel for reduction(+:sum) default(none) shared(as, n)
            for (int i = 0; i < n; ++i) {
                sum += as[i];
            }
            EXPECT_THE_SAME(reference_sum, sum, "CPU OpenMP result should be consistent!");
            t.nextLap();
        }
        std::cout << "CPU OMP: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU OMP: " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
        std::cout << "CPU OMP: " << (n/1000.0/1000.0/1000.0) / t.lapAvg() << " GFlops" << std::endl;
    }

    {
        // TODO: implement on OpenCL
         gpu::Device device = gpu::chooseGPUDevice(argc, argv);
         gpu::Context context;
         context.init(device.device_id_opencl);
         context.activate();

         ocl::Kernel kernel(sum_kernel, sum_kernel_length, "sum");
         kernel.compile(true);

         unsigned int size = gpu::divup(n, 128) * 128;
         gpu::gpu_mem_32u vram;
         vram.resizeN(n + 1);
         vram.writeN(as.data(), n);

         const unsigned int GROUP_SIZE = 128;
         timer t;
         for (int iter = 0; iter < benchmarkingIters; ++iter) {
             kernel.exec(gpu::WorkSize(GROUP_SIZE, size), vram, n);
             if (iter == 0) {
                 unsigned int sum;
                 vram.readN(&sum, 1, n);

                 EXPECT_THE_SAME(sum, reference_sum, "GPU result should be consistent!");
             }
             t.nextLap();
         }

         std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
         std::cout << "GPU: " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
         std::cout << "GPU: " << (n/1000.0/1000.0/1000.0) / t.lapAvg() << " GFlops" << std::endl;
    }
}
