#pragma once
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <utility>
#include <vector>

#include <cuda_bf16.h>
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

inline void printStats(const char *name, const Stats &s, double flops = 0.0) {
    printf("%-30s mean=%.3f ms  median=%.3f ms  std=%.3f ms  min=%.3f ms  max=%.3f ms", name,
           s.mean, s.median, s.std_dev, s.min, s.max);
    if (flops > 0.0)
        printf("  %.2f TFLOPS", flops / (s.median * 1e9));
    printf("\n");
}

template <typename T>
inline bool vectorApproximatelyEqual(const T *A, const T *B, int length, float abs_tol = 1e-4f,
                                     float rel_tol = 1e-4f) {
    float max_abs_err = 0.0f;
    float max_rel_err = 0.0f;
    int worst_idx = -1;

    for (int i = 0; i < length; i++) {
        float a = (float)A[i];
        float b = (float)B[i];
        float abs_err = fabsf(a - b);
        float rel_err = abs_err / fmaxf(fmaxf(fabsf(a), fabsf(b)), 1.0f);
        float tol = abs_tol + rel_tol * fmaxf(fabsf(a), fabsf(b));

        if (abs_err > max_abs_err) {
            max_abs_err = abs_err;
            max_rel_err = rel_err;
            worst_idx = i;
        }

        if (abs_err > tol) {
            printf("Error: A[%d] = %f, B[%d] = %f, abs_err = %.8f, rel_err = %.8f, tol = %.8f\n", i,
                   a, i, b, abs_err, rel_err, tol);
            return false;
        }
    }

    if (worst_idx >= 0) {
        printf("Max error: idx=%d, abs_err=%.8f, rel_err=%.8f\n", worst_idx, max_abs_err,
               max_rel_err);
    }
    return true;
}

inline bool vectorApproximatelyEqualBf16(const __nv_bfloat16 *A, const __nv_bfloat16 *B, int length,
                                         float abs_tol = 5e-2f, float rel_tol = 5e-2f) {
    return vectorApproximatelyEqual(A, B, length, abs_tol, rel_tol);
}

inline Stats computeStats(std::vector<float> &times) {
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