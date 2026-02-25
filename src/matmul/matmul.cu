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

__global__ void baseMatmulKernel(float *A, float *B, float *C, int m, int bm) {
    /**
     * Each thread in this kernel computes a single element of the output matrix C.
     * The kernel uses a square tile of size bm x bm to load the input matrices A and B into shared
     * memory.
     *
     * Benchmark (8192 x 8192 x 8192 matmul on A100)
     * - bm=16: 3.51 TFLOPS
     * - bm=32: 3.85 TFLOPS
     */
    extern __shared__ float smem[];
    float *tileA = smem;                 // bm x (bm+1)
    float *tileB = smem + bm * (bm + 1); // bm x bm

    int blockRow = blockIdx.y * bm;
    int blockCol = blockIdx.x * bm;

    float Cval = 0.0f;
    for (int i = 0; i < cuda::ceil_div(m, bm); i++) {
        // tileA origin: (blockRow, i*bm), tileB origin: (i*bm, blockCol)
        // Commenting out all global memory loads => 4.70 TFLOPS
        tileA[INDX(threadIdx.y, threadIdx.x, (bm + 1))] =
            A[INDX(blockRow + threadIdx.y, i * bm + threadIdx.x, m)];
        tileB[INDX(threadIdx.y, threadIdx.x, bm)] =
            B[INDX(i * bm + threadIdx.y, blockCol + threadIdx.x, m)];
        __syncthreads();

        for (int k = 0; k < bm; k++) {
            Cval += tileA[INDX(threadIdx.y, k, (bm + 1))] * tileB[INDX(k, threadIdx.x, bm)];
            // Cval += 1.0f * tileB[INDX(k, threadIdx.x, bm)]; => 5.72 TFLOPS (reduce shared memory
            // access into half)
        }
        __syncthreads();
    }
    C[INDX(blockRow + threadIdx.y, blockCol + threadIdx.x, m)] = Cval;
}

#define MAX_BM 32
__global__ void mioOptimizedMatmulKernel(float *A, float *B, float *C, int m, int bm, int bn,
                                         int bk) {
    /**
     * Problem of base kernel:
     * - Every add instruction requires two shared memory accesses.
     * - ncu profiling shows that MIO Throttle is the main bottleneck of the base kernel.
     *
     * Solution:
     * - Idea: Store smem values in registers and reuse as much as possible.
     * - A single thread block computes bm x bn elements of the output matrix C.
     * - A single thread of this kernel:
     *   (1) loads bm x 1 column of A into registers.
     *   (2) loads 1 x bm row of B into registers.
     *   (3) computes bm x bn elements of the output matrix C.
     *   (4) A thread block aggregates the results of all threads.
     * - gridDim: (m / bm, n / bn)
     * - blockDim: any divisor of bk
     *   - Each thread loads & computes every threadIdx th (column of A, row of B) into registers.
     *
     * Benchmark (8192 x 8192 x 8192 matmul on A100):
     * - TBD
     */

    extern __shared__ float smem[];
    float *sA = smem;                 // bm x (bk + 1)
    float *sB = smem + bm * (bk + 1); // bk x bn
    int ldsa = bk + 1;
    int ldsb = bn;

    // global offset
    int y0 = blockIdx.y * bm;
    int x0 = blockIdx.x * bn;

    // Load bm x bk tile of A into shared memory
    for (int y = 0; y < bm; y++) {
        for (int x = threadIdx.x; x < bk; x += blockDim.x) {
            sA[INDX(y, x, ldsa)] = A[INDX(y0 + y, x0 + x, m)];
        }
    }
    // Load bk x bm tile of B into shared memory
    for (int y = 0; y < bk; y++) {
        // For simplicity, assume bn is divisible by blockDim.x
        for (int x = threadIdx.x; x < bn; x += blockDim.x) {
            sB[INDX(y, x, ldsb)] = B[INDX(y0 + y, x0 + x, m)];
        }
    }

    __syncthreads();

    // TODO: change MAX_BM to bm/bn to reduce register usage. Requires bm/bn to be constant.
    float regA[MAX_BM] = {0};
    float regB[MAX_BM] = {0};
    float regC[MAX_BM][MAX_BM] = {0};

    for (int k = threadIdx.x; k < bk; k += blockDim.x) {
        for (int y = 0; y < bm; y++) {
            regA[y] = sA[INDX(y, k, ldsa)];
        }
        for (int x = 0; x < bn; x++) {
            regB[x] = sB[INDX(k, x, ldsb)];
        }
        for (int x = 0; x < bn; x++) {
            for (int y = 0; y < bm; y++) {
                regC[y][x] += regA[y] * regB[x];
            }
        }
    }

    // Write back to global memory
    // TODO: compare with atomic add.
    for (int offset = 0; offset < blockDim.x; offset++) {
        int tx0 = threadIdx.x + offset;
        if (tx0 >= blockDim.x)
            tx0 -= blockDim.x;
        for (int y = 0; y < bm; y++) {
            for (int x = tx0; x < bn; x += blockDim.x) {
                C[INDX(y0 + y, x0 + x, m)] += regC[y][x];
            }
        }
        __syncthreads();
    }
}
