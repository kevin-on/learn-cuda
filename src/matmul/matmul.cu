#include <cmath>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <cublas_v2.h>
#include <cuda/cmath>
#include <cuda/pipeline>
#include <cuda_pipeline.h>
#include <getopt.h>
#include <stdio.h>

#include "../utils/cuda_utils.cuh"

/**
 * Benchmark: 8192 x 8192 x 8192 matmul on A100
 *
 *   A100 FP32 peak (NVIDIA spec) : 19.5  TFLOPS
 *   cuBLAS                       : 19.17 TFLOPS (98.3%)
 */

/**
 * A100 specifications:
 *
 *   Max Warps / SM = 64
 *   Max Threads / SM = 2048
 *   Max Thread Blocks / SM = 32
 *   Max Thread Block Size = 1024
 *   Shared Memory per SM = 64 KB / Configureable up to 164KB
 */

#define INDX(row, col, ld) ((row) * (ld) + (col))

__global__ void squareTileMatmulKernel(float *A, float *B, float *C, int m, int bm) {
    /**
     * Each thread in this kernel computes a single element of the output matrix C.
     * The kernel uses a square tile of size bm x bm to load the input matrices A and B into shared
     * memory.
     *
     * Preconditions:
     * - m is divisible by bm  // TODO: handle index out of bounds (e.g. m % bm != 0)
     *
     * Benchmark (8192 x 8192 x 8192 matmul on A100)
     * - bm=16: 3.51 TFLOPS
     * - bm=32: 3.85 TFLOPS
     *
     * Analysis (bm=32):
     * - 32x32 threads per block
     * - 2 blocks per SM
     * - 64 warps per block (100% occupancy)
     *
     * In the for loop,
     * - Memory access latency: 400~600 cycles
     * - Compute latency: 3*bm cycles (96 cycles for bm=32)
     */
    extern __shared__ float smem[];
    float *tileA = smem;                     // 2 x bm x (bm + 1)
    float *tileB = smem + 2 * bm * (bm + 1); // 2 x bm x bm

    int blockRow = blockIdx.y * bm;
    int blockCol = blockIdx.x * bm;

    float Cval = 0.0f;

    // Initialize the first tile
    tileA[INDX(threadIdx.y, threadIdx.x, (bm + 1))] =
        A[INDX(blockRow + threadIdx.y, threadIdx.x, m)];
    tileB[INDX(threadIdx.y, threadIdx.x, bm)] = B[INDX(threadIdx.y, blockCol + threadIdx.x, m)];
    __syncthreads();

    for (int i = 0; i < cuda::ceil_div(m, bm); i++) {
        float *curTileA = tileA + (i & 1) * bm * (bm + 1);
        float *nextTileA = tileA + ((i + 1) & 1) * bm * (bm + 1);
        float *curTileB = tileB + (i & 1) * bm * bm;
        float *nextTileB = tileB + ((i + 1) & 1) * bm * bm;

        // prefetch the next tile to shared memory
        if (i < cuda::ceil_div(m, bm) - 1) {
            __pipeline_memcpy_async(&nextTileA[INDX(threadIdx.y, threadIdx.x, (bm + 1))],
                                    &A[INDX(blockRow + threadIdx.y, (i + 1) * bm + threadIdx.x, m)],
                                    sizeof(float));
            __pipeline_memcpy_async(&nextTileB[INDX(threadIdx.y, threadIdx.x, bm)],
                                    &B[INDX((i + 1) * bm + threadIdx.y, blockCol + threadIdx.x, m)],
                                    sizeof(float));
            __pipeline_commit();
        }

        // compute the current tile
        for (int k = 0; k < bm; k++) {
            Cval += curTileA[INDX(threadIdx.y, k, (bm + 1))] * curTileB[INDX(k, threadIdx.x, bm)];
        }

        __pipeline_wait_prior(0);
        __syncthreads();
    }
    C[INDX(blockRow + threadIdx.y, blockCol + threadIdx.x, m)] = Cval;
}

__global__ void float4SquareTileMatmulKernel(float *A, float *Bt, float *C, int m, int bm) {
    /**
     * This kernel is a variant of the squareTileMatmulKernel that uses float4 to load global memory
     * into shared memory.
     *
     * Preconditions:
     * - m and bm are divisible by 4
     * - Bt is a transpose of B
     *
     * Benchmark (8192 x 8192 x 8192 matmul on A100)
     * - The numbers below are without transpose latency.
     * - bm=16: 3.81 TFLOPS
     * - bm=32: 3.96 TFLOPS
     */
    // TODO: handle index out of bounds
    extern __shared__ float smem[];
    float *tileA = smem;                       // bm x (4*bm+1)
    float *tileB = smem + bm * ((4 * bm) + 1); // (4*bm) x (bm+1)

    float4 *A4 = reinterpret_cast<float4 *>(A);   // m x (m / 4)
    float4 *Bt4 = reinterpret_cast<float4 *>(Bt); // m x (m / 4)

    int mDiv4 = m >> 2;
    int blockRow = blockIdx.y * bm;
    int blockCol = blockIdx.x * bm;

    float Cval = 0.0f;
    for (int i = 0; i < cuda::ceil_div(m, 4 * bm); i++) {
        // tileA origin: (blockRow, i*4*bm)
        // tileA4 origin: (blockRow, i*bm)
        float4 a4 = A4[INDX(blockRow + threadIdx.y, i * bm + threadIdx.x, mDiv4)];
        tileA[INDX(threadIdx.y, 4 * threadIdx.x, (4 * bm + 1))] = a4.x;
        tileA[INDX(threadIdx.y, 4 * threadIdx.x + 1, (4 * bm + 1))] = a4.y;
        tileA[INDX(threadIdx.y, 4 * threadIdx.x + 2, (4 * bm + 1))] = a4.z;
        tileA[INDX(threadIdx.y, 4 * threadIdx.x + 3, (4 * bm + 1))] = a4.w;

        // // Sanity check
        // for (int j = 0; j < 4; j++) {
        //     int sr = threadIdx.y;
        //     int sc = 4 * threadIdx.x + j;
        //     if (tileA[INDX(sr, sc, (4 * bm + 1))] != A[INDX(blockRow + sr, i * 4 * bm + sc, m)])
        //     {
        //         printf("Error: tileA[%d][%d] != A[%d][%d] in block (%d, %d)\n", sr, sc,
        //                blockRow + sr, i * 4 * bm + sc, blockIdx.x, blockIdx.y);
        //         return;
        //     }
        // }
        __syncthreads();

        // tileB origin: (i*4*bm, blockCol)
        // tileBt origin: (blockCol, i*4*bm)
        // tileBt4 origin: (blockCol, i*bm)
        float4 bt4 = Bt4[INDX(blockCol + threadIdx.y, i * bm + threadIdx.x, mDiv4)];
        tileB[INDX(4 * threadIdx.x, threadIdx.y, bm + 1)] = bt4.x;
        tileB[INDX(4 * threadIdx.x + 1, threadIdx.y, bm + 1)] = bt4.y;
        tileB[INDX(4 * threadIdx.x + 2, threadIdx.y, bm + 1)] = bt4.z;
        tileB[INDX(4 * threadIdx.x + 3, threadIdx.y, bm + 1)] = bt4.w;
        __syncthreads();

        // // Sanity check
        // for (int j = 0; j < 4; j++) {
        //     int sr = 4 * threadIdx.x + j;
        //     int sc = threadIdx.y;
        //     if (tileB[INDX(sr, sc, bm)] != Bt[INDX(blockCol + sc, i * 4 * bm + sr, m)]) {
        //         printf("Error: tileB[%d][%d] != Bt[%d][%d] in block (%d, %d)\n", sr, sc,
        //                blockCol + sc, i * 4 * bm + sr, blockIdx.x, blockIdx.y);
        //         return;
        //     }
        // }

        for (int k = 0; k < 4 * bm; k++) {
            Cval += tileA[INDX(threadIdx.y, k, (4 * bm + 1))] * tileB[INDX(k, threadIdx.x, bm + 1)];
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

void sequentialTranspose(float *A, float *At, int m) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) {
            At[INDX(j, i, m)] = A[INDX(i, j, m)];
        }
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
            printf("Usage: %s <m> <bm> [--warmup N] [--iters N]\n", argv[0]);
            return 1;
        }
    }

    if (argc - optind != 2) {
        printf("Usage: %s <m> <bm> [--warmup N] [--iters N]\n", argv[0]);
        return 1;
    }
    int m = atoi(argv[optind]);
    int bm = atoi(argv[optind + 1]);

    if (m <= 0 || bm <= 0) {
        printf("Error: m, bm must be > 0\n");
        return 1;
    }
    if (warmup < 0 || iters <= 0) {
        printf("Error: warmup >= 0, iters > 0\n");
        return 1;
    }
    printf("Benchmark: m=%d, bm=%d, warmup=%d, iters=%d\n", m, bm, warmup, iters);

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

    double flops = 2.0 * m * m * m;

    // Run cublasMatmul
    Stats cublasStats;
    auto cublasResult = runKernelBenchmark<std::vector<float>>(
        [&]() { cublasMatmul(handle, A, B, C_cublas, m); },
        [&]() { CUDA_CHECK(cudaMemset(C_cublas, 0, numElems * sizeof(float))); },
        [&]() { return std::vector<float>(C_cublas, C_cublas + numElems); }, warmup, iters,
        cublasStats);
    printStats("cublasMatmul", cublasStats, flops);

    // Run squareTileMatmulKernel
    dim3 blockSquareTile(bm, bm);
    dim3 gridSquareTile(cuda::ceil_div(m, bm), cuda::ceil_div(m, bm));
    size_t smemSizeSquareTile = (2 * bm * (bm + 1) + 2 * bm * bm) * sizeof(float);
    Stats squareTileMatmulKernelStats;
    auto squareTileMatmulKernelResult = runKernelBenchmark<std::vector<float>>(
        [&]() {
            squareTileMatmulKernel<<<gridSquareTile, blockSquareTile, smemSizeSquareTile>>>(A, B, C,
                                                                                            m, bm);
        },
        [&]() { CUDA_CHECK(cudaMemset(C, 0, numElems * sizeof(float))); },
        [&]() { return std::vector<float>(C, C + numElems); }, warmup, iters,
        squareTileMatmulKernelStats);
    printStats("squareTileMatmulKernel", squareTileMatmulKernelStats, flops);
    if (vectorApproximatelyEqual(squareTileMatmulKernelResult.data(), cublasResult.data(),
                                 numElems)) {
        printf("squareTileMatmulKernel results are correct\n");
    } else {
        printf("Error: squareTileMatmulKernel results are incorrect\n");
    }

    // // Run float4SquareTileMatmulKernel
    // // TODO: Include transpose latency in the benchmark
    // float *Bt = nullptr;
    // CUDA_CHECK(cudaMallocManaged(&Bt, numElems * sizeof(float)));
    // sequentialTranspose(B, Bt, m);

    // dim3 blockFloat4SquareTile(bm, bm);
    // dim3 gridFloat4SquareTile(cuda::ceil_div(m, bm), cuda::ceil_div(m, bm));
    // size_t smemSizeFloat4SquareTile = (bm * (4 * bm + 1) + (4 * bm) * (bm + 1)) * sizeof(float);
    // Stats float4SquareTileMatmulKernelStats;
    // auto float4SquareTileMatmulKernelResult = runKernelBenchmark<std::vector<float>>(
    //     [&]() {
    //         float4SquareTileMatmulKernel<<<gridFloat4SquareTile, blockFloat4SquareTile,
    //                                        smemSizeFloat4SquareTile>>>(A, Bt, C, m, bm);
    //     },
    //     [&]() { CUDA_CHECK(cudaMemset(C, 0, numElems * sizeof(float))); },
    //     [&]() { return std::vector<float>(C, C + numElems); }, warmup, iters,
    //     float4SquareTileMatmulKernelStats);

    // printStats("float4SquareTileMatmulKernel", float4SquareTileMatmulKernelStats, flops);
    // if (vectorApproximatelyEqual(float4SquareTileMatmulKernelResult.data(), cublasResult.data(),
    //                              numElems)) {
    //     printf("float4SquareTileMatmulKernel results are correct\n");
    // } else {
    //     printf("Error: float4SquareTileMatmulKernel results are incorrect\n");
    // }
    // CUDA_CHECK(cudaFree(Bt));

    CUBLAS_CHECK(cublasDestroy(handle));

    CUDA_CHECK(cudaFree(A));
    CUDA_CHECK(cudaFree(B));
    CUDA_CHECK(cudaFree(C));
    CUDA_CHECK(cudaFree(C_cublas));

    return 0;
}
