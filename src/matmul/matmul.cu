#include <cmath>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <cublas_v2.h>
#include <cuda/cmath>
#include <stdio.h>

#include "../utils/cuda_utils.cuh"

/**
 * Benchmark: 8192 x 8192 x 8192 matmul on A100
 *
 *   A100 FP32 peak (NVIDIA spec) : 19.5  TFLOPS
 *   cuBLAS                       : 19.17 TFLOPS (98.3%)
 *   This kernel                  :  3.52 TFLOPS (18.1%)
 */

#define INDX(row, col, ld) ((row) * (ld) + (col))

constexpr int kBlockSize = 16;

#define CUBLAS_CHECK(call)                                                                         \
    do {                                                                                           \
        cublasStatus_t stat = (call);                                                              \
        if (stat != CUBLAS_STATUS_SUCCESS) {                                                       \
            fprintf(stderr, "cuBLAS error at %s:%d: status=%d\n", __FILE__, __LINE__, (int)stat);  \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
    } while (0)

__global__ void matmulKernel(float *A, float *B, float *C, int m) {
    // Each thread computes C[blockIdx.y*kBlockSize + threadIdx.y][blockIdx.x*kBlockSize +
    // threadIdx.x]
    // TODO: handle index out of bounds
    __shared__ float tileA[kBlockSize][kBlockSize];
    __shared__ float tileB[kBlockSize][kBlockSize];

    float Cval = 0.0f;
    for (int i = 0; i < cuda::ceil_div(m, kBlockSize); i++) {
        // A tile origin: (blockRow, i*kBlockSize), B tile origin: (i*kBlockSize, blockCol)
        tileA[threadIdx.y][threadIdx.x] =
            A[INDX(blockIdx.y * kBlockSize + threadIdx.y, i * kBlockSize + threadIdx.x, m)];
        tileB[threadIdx.y][threadIdx.x] =
            B[INDX(i * kBlockSize + threadIdx.y, blockIdx.x * kBlockSize + threadIdx.x, m)];

        __syncthreads();
        for (int j = 0; j < kBlockSize; j++) {
            int colA = threadIdx.y + j;
            if (colA >= kBlockSize)
                colA -= kBlockSize;
            Cval += tileA[threadIdx.y][colA] * tileB[colA][threadIdx.x];
        }
        __syncthreads();
    }
    C[INDX(blockIdx.y * kBlockSize + threadIdx.y, blockIdx.x * kBlockSize + threadIdx.x, m)] = Cval;
}

void initArray(float *array, int len) {
    std::srand(std::time({}));
    for (int i = 0; i < len; i++) {
        array[i] = rand() / (float)RAND_MAX;
    }
}

void sequentialMatmul(float *A, float *B, float *C, int m) {
    // Precondition: C must be zeroed
    for (int i = 0; i < m; ++i) {
        for (int k = 0; k < m; ++k) {
            float a = A[INDX(i, k, m)];
            for (int j = 0; j < m; ++j) {
                C[INDX(i, j, m)] += a * B[INDX(k, j, m)];
            }
        }
    }
}

void cublasMatmul(cublasHandle_t handle, float *A, float *B, float *C, int m) {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Row-major A*B via column-major cuBLAS: C^T = B^T * A^T
    CUBLAS_CHECK(
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, m, m, &alpha, B, m, A, m, &beta, C, m));
}

bool vectorApproximatelyEqual(float *A, float *B, int length, float abs_tol = 1e-4f,
                              float rel_tol = 1e-4f) {
    float max_abs_err = 0.0f;
    float max_rel_err = 0.0f;
    int worst_idx = -1;

    for (int i = 0; i < length; i++) {
        float a = A[i];
        float b = B[i];
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

int main(int argc, char **argv) {
    if (argc != 2) {
        printf("Usage: %s <matrix size>\n", argv[0]);
        return 1;
    }
    int m = atoi(argv[1]);
    size_t numElems = (size_t)m * m;

    float *A = nullptr, *B = nullptr, *C = nullptr, *C_cublas = nullptr;
    CUDA_CHECK(cudaMallocManaged(&A, numElems * sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&B, numElems * sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&C, numElems * sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&C_cublas, numElems * sizeof(float)));
    initArray(A, numElems);
    initArray(B, numElems);

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    int warmup = 5, timed = 20;
    dim3 block(kBlockSize, kBlockSize);
    dim3 grid(cuda::ceil_div(m, kBlockSize), cuda::ceil_div(m, kBlockSize));

    Stats kernelStats;
    auto kernelResult = runKernelBenchmark<std::vector<float>>(
        [&]() { matmulKernel<<<grid, block>>>(A, B, C, m); },
        [&]() { CUDA_CHECK(cudaMemset(C, 0, numElems * sizeof(float))); },
        [&]() { return std::vector<float>(C, C + numElems); }, warmup, timed, kernelStats);

    Stats cublasStats;
    auto cublasResult = runKernelBenchmark<std::vector<float>>(
        [&]() { cublasMatmul(handle, A, B, C_cublas, m); },
        [&]() { CUDA_CHECK(cudaMemset(C_cublas, 0, numElems * sizeof(float))); },
        [&]() { return std::vector<float>(C_cublas, C_cublas + numElems); }, warmup, timed,
        cublasStats);

    double flops = 2.0 * m * m * m;
    printStats("matmulKernel", kernelStats, flops);
    printStats("cublasMatmul", cublasStats, flops);

    if (vectorApproximatelyEqual(kernelResult.data(), cublasResult.data(), numElems)) {
        printf("matmulKernel results are correct\n");
    } else {
        printf("Error: matmulKernel results are incorrect\n");
    }

    CUBLAS_CHECK(cublasDestroy(handle));

    CUDA_CHECK(cudaFree(A));
    CUDA_CHECK(cudaFree(B));
    CUDA_CHECK(cudaFree(C));
    CUDA_CHECK(cudaFree(C_cublas));

    return 0;
}
