#include <cmath>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <cublas_v2.h>
#include <cuda/cmath>
#include <getopt.h>
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

__global__ void matmulKernel(float *A, float *B, float *C, int m, int bm, int bn, int bk) {
    // TODO: handle index out of bounds
    extern __shared__ float smem[];
    float *tileA = smem;                 // bm x (bk+1)
    float *tileB = smem + bm * (bk + 1); // bk x bn

    int blockRow = blockIdx.y * bm;
    int blockCol = blockIdx.x * bn;

    float Cval = 0.0f;
    for (int i = 0; i < cuda::ceil_div(m, bk); i++) {
        // tileA origin: (blockRow, i*bk), tileB origin: (i*bk, blockCol)
        // FIXME: Current shared memory layout logic is only correct when bk == bn == bm.
        // Initializing shared memory with for loops is too slow.
        tileA[INDX(threadIdx.y, threadIdx.x, (bk + 1))] =
            A[INDX(blockRow + threadIdx.y, i * bk + threadIdx.x, m)];
        tileB[INDX(threadIdx.y, threadIdx.x, bn)] =
            B[INDX(i * bk + threadIdx.y, blockCol + threadIdx.x, m)];
        __syncthreads();

        for (int k = 0; k < bk; k++) {
            Cval += tileA[INDX(threadIdx.y, k, (bk + 1))] * tileB[INDX(k, threadIdx.x, bn)];
        }
        __syncthreads();
    }
    C[INDX(blockRow + threadIdx.y, blockCol + threadIdx.x, m)] = Cval;
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
    int warmup = 5;
    int iters = 10;

    static struct option long_options[] = {{"warmup", required_argument, nullptr, 'w'},
                                           {"iters", required_argument, nullptr, 'i'},
                                           {nullptr, 0, nullptr, 0}};

    int opt;
    while ((opt = getopt_long(argc, argv, "", long_options, nullptr)) != -1) {
        switch (opt) {
        case 'w':
            warmup = atoi(optarg);
            break;
        case 'i':
            iters = atoi(optarg);
            break;
        default:
            printf("Usage: %s <m> <bm> <bn> <bk> [--warmup N] [--iters N]\n", argv[0]);
            return 1;
        }
    }

    if (argc - optind != 4) {
        printf("Usage: %s <m> <bm> <bn> <bk> [--warmup N] [--iters N]\n", argv[0]);
        return 1;
    }
    int m = atoi(argv[optind]);
    int bm = atoi(argv[optind + 1]);
    int bn = atoi(argv[optind + 2]);
    int bk = atoi(argv[optind + 3]);

    if (m <= 0 || bm <= 0 || bn <= 0 || bk <= 0) {
        printf("Error: m, bm, bn, bk must be > 0\n");
        return 1;
    }
    if (warmup < 0 || iters <= 0) {
        printf("Error: warmup >= 0, iters > 0\n");
        return 1;
    }
    printf("Benchmark: m=%d, bm=%d, bn=%d, bk=%d, warmup=%d, iters=%d\n", m, bm, bn, bk, warmup,
           iters);

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

    dim3 block(bn, bm);
    dim3 grid(cuda::ceil_div(m, bn), cuda::ceil_div(m, bm));
    size_t smemSize = (bm * (bk + 1) + bk * bn) * sizeof(float);
    Stats matmulKernelStats;
    auto matmulKernelResult = runKernelBenchmark<std::vector<float>>(
        [&]() { matmulKernel<<<grid, block, smemSize>>>(A, B, C, m, bm, bn, bk); },
        [&]() { CUDA_CHECK(cudaMemset(C, 0, numElems * sizeof(float))); },
        [&]() { return std::vector<float>(C, C + numElems); }, warmup, iters, matmulKernelStats);

    Stats cublasStats;
    auto cublasResult = runKernelBenchmark<std::vector<float>>(
        [&]() { cublasMatmul(handle, A, B, C_cublas, m); },
        [&]() { CUDA_CHECK(cudaMemset(C_cublas, 0, numElems * sizeof(float))); },
        [&]() { return std::vector<float>(C_cublas, C_cublas + numElems); }, warmup, iters,
        cublasStats);

    double flops = 2.0 * m * m * m;
    printStats("matmulKernel", matmulKernelStats, flops);
    printStats("cublasMatmul", cublasStats, flops);

    if (vectorApproximatelyEqual(matmulKernelResult.data(), cublasResult.data(), numElems)) {
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
