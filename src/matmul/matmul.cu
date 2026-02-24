#include <cmath>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <cublas_v2.h>
#include <cuda/cmath>
#include <stdio.h>

#include "../utils/cuda_utils.cuh"

/**
 * Target: 8192 x 8192 x 8192 matmul in A100
 */

#define INDX(row, col, ld) ((row) * (ld) + (col))

constexpr int kBlockSize = 32;

#define CUBLAS_CHECK(call)                                                                         \
    do {                                                                                           \
        cublasStatus_t stat = (call);                                                              \
        if (stat != CUBLAS_STATUS_SUCCESS) {                                                       \
            fprintf(stderr, "cuBLAS error at %s:%d: status=%d\n", __FILE__, __LINE__, (int)stat);  \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
    } while (0)

__global__ void matmulKernel(float *A, float *B, float *C, int m) {
    // This kernel computes
    // C(blockIdx.y * kBlockSize + threadIdx.y, blockIdx.x * kBlockSize + threadIdx.x)
    __shared__ float tileA[kBlockSize][kBlockSize];
    float C_value = 0.0f;
    for (int i = 0; i < cuda::ceil_div(m, kBlockSize); i++) {
        // Copy A tile to shared memory
        // left top of A tile: (blockIdx.y * kBlockSize, i * kBlockSize)
        // left top of B tile: (i * kBlockSize, blockIdx.x * kBlockSize)
        tileA[threadIdx.y][threadIdx.x] =
            A[INDX(blockIdx.y * kBlockSize + threadIdx.y, i * kBlockSize + threadIdx.x, m)];
        __syncthreads();
        for (int j = 0; j < kBlockSize; j++) {
            int colA = threadIdx.y + j;
            if (colA >= kBlockSize)
                colA -= kBlockSize;
            C_value += tileA[threadIdx.y][colA] *
                       B[INDX(i * kBlockSize + colA, blockIdx.x * kBlockSize + threadIdx.x, m)];
        }
        __syncthreads();
    }
    C[INDX(blockIdx.y * kBlockSize + threadIdx.y, blockIdx.x * kBlockSize + threadIdx.x, m)] =
        C_value;
}

void initArray(float *A, int m) {
    std::srand(std::time({}));
    for (int i = 0; i < m; i++) {
        A[i] = rand() / (float)RAND_MAX;
    }
}

void sequentialMatmul(float *A, float *B, float *C, int m) {
    // Assume C is initialized to 0
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

    // A/B/C are stored row-major. cuBLAS expects column-major, so compute:
    // C^T = B^T * A^T  <=>  C = A * B in row-major.
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
    size_t n = (size_t)m * m;

    float *A = nullptr, *B = nullptr, *C = nullptr, *C_cublas = nullptr;
    CUDA_CHECK(cudaMallocManaged(&A, n * sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&B, n * sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&C, n * sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&C_cublas, n * sizeof(float)));
    initArray(A, n);
    initArray(B, n);

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    int warmup = 3, timed = 10;
    dim3 block(kBlockSize, kBlockSize);
    dim3 grid(cuda::ceil_div(m, kBlockSize), cuda::ceil_div(m, kBlockSize));

    Stats kernelStats;
    auto kernelResult = runKernelBenchmark<std::vector<float>>(
        [&]() { matmulKernel<<<grid, block>>>(A, B, C, m); },
        [&]() { CUDA_CHECK(cudaMemset(C, 0, n * sizeof(float))); },
        [&]() { return std::vector<float>(C, C + n); }, warmup, timed, kernelStats);

    Stats cublasStats;
    auto cublasResult = runKernelBenchmark<std::vector<float>>(
        [&]() { cublasMatmul(handle, A, B, C_cublas, m); },
        [&]() { CUDA_CHECK(cudaMemset(C_cublas, 0, n * sizeof(float))); },
        [&]() { return std::vector<float>(C_cublas, C_cublas + n); }, warmup, timed, cublasStats);

    double flops = 2.0 * m * m * m;
    printStats("matmulKernel", kernelStats, flops);
    printStats("cublasMatmul", cublasStats, flops);

    if (vectorApproximatelyEqual(kernelResult.data(), cublasResult.data(), n)) {
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
