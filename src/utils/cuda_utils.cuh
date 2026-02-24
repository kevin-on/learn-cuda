#pragma once
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                                           \
    do {                                                                                           \
        cudaError_t err = (call);                                                                  \
        if (err != cudaSuccess) {                                                                  \
            fprintf(stderr, "CUDA error at %s:%d — %s\n", __FILE__, __LINE__,                      \
                    cudaGetErrorString(err));                                                      \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
    } while (0)

#define CUBLAS_CHECK(call)                                                                         \
    do {                                                                                           \
        cublasStatus_t stat = (call);                                                              \
        if (stat != CUBLAS_STATUS_SUCCESS) {                                                       \
            fprintf(stderr, "cuBLAS error at %s:%d: status=%d\n", __FILE__, __LINE__, (int)stat);  \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
    } while (0)

struct Stats {
    float mean;
    float median;
    float std_dev;
    float min;
    float max;
};

void printStats(const char *name, const Stats &s, double flops = 0.0) {
    printf("%-15s mean=%.3f ms  median=%.3f ms  std=%.3f ms  min=%.3f ms  max=%.3f ms", name,
           s.mean, s.median, s.std_dev, s.min, s.max);
    if (flops > 0.0)
        printf("  %.2f TFLOPS", flops / (s.median * 1e9));
    printf("\n");
}

Stats computeStats(std::vector<float> &times) {
    std::sort(times.begin(), times.end());
    int n = (int)times.size();

    float sum = std::accumulate(times.begin(), times.end(), 0.0f);
    float mean = sum / n;
    float median = (n % 2 == 0) ? (times[n / 2 - 1] + times[n / 2]) / 2.0f : times[n / 2];

    float sq_sum = 0.0f;
    for (float t : times)
        sq_sum += (t - mean) * (t - mean);
    float std_dev = sqrtf(sq_sum / n);

    return {mean, median, std_dev, times.front(), times.back()};
}

template <typename ResultT, typename LaunchFn, typename ResetFn, typename FetchResultFn>
ResultT runKernelBenchmark(LaunchFn &&launch_kernel, ResetFn &&reset_outputs,
                           FetchResultFn &&fetch_result, int warmup_iters, int timed_iters,
                           Stats &out_stats) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    auto launch = std::forward<LaunchFn>(launch_kernel);
    auto reset = std::forward<ResetFn>(reset_outputs);
    auto fetch = std::forward<FetchResultFn>(fetch_result);

    // Warmup
    for (int i = 0; i < warmup_iters; ++i) {
        reset();
        launch();
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed runs
    std::vector<float> times(timed_iters);
    for (int i = 0; i < timed_iters; ++i) {
        reset();

        CUDA_CHECK(cudaEventRecord(start));
        launch();
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        CUDA_CHECK(cudaEventElapsedTime(&times[i], start, stop));
    }

    out_stats = computeStats(times);

    // Return result for correctness checking
    reset();
    launch();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    ResultT result = fetch();

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return result;
}