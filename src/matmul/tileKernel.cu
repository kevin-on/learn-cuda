#include <cuda/cmath>
#include <cuda_pipeline.h>

#include "../utils/cuda_utils.cuh"
#include "matmul.cuh"

#define INDX(row, col, ld) ((row) * (ld) + (col))

constexpr int kBlockTileM = 64;
constexpr int kBlockTileN = 64;
constexpr int kBlockTileK = 8;
constexpr int kThreadTileM = 8;
constexpr int kThreadTileN = 8;

__global__ void tileMatmulKernel(float *A, float *B, float *C, int m) {
    /**
     * - Each thread block computes a kBlockTileM x kBlockTileN tile of the output matrix C.
     * - Each thread computes a kThreadTileM x kThreadTileN tile of the output matrix C.
     * - In each iteration, a thread block loads a kBlockTileM x kBlockTileK tile of A and a
     * kBlockTileK x kBlockTileN tile of B into shared memory.
     * - gridDim: (m / kBlockTileN, m / kBlockTileM)
     * - blockDim: (kBlockTileN / kThreadTileN, kBlockTileM / kThreadTileM)
     *
     * Benchmark (8192 x 8192 x 8192 matmul on A100)
     * - kBlockTileM=64, kBlockTileN=64, kBlockTileK=64, kThreadTileM=8, kThreadTileN=8: 2.88 TFLOPS
     */

    __shared__ float sA[kBlockTileM][(kBlockTileK + 1)];
    __shared__ float sB[kBlockTileK][kBlockTileN];

    // origin of the thread block's output tile
    int bx0 = blockIdx.x * kBlockTileN;
    int by0 = blockIdx.y * kBlockTileM;

    float regC[kThreadTileM][kThreadTileN] = {0};
    float regA[kBlockTileK] = {0};

    for (int k0 = 0; k0 < m; k0 += kBlockTileK) {
        // Load kBlockTileM x kBlockTileK tile of A into shared memory
        for (int y = threadIdx.y; y < kBlockTileM; y += blockDim.y) {
            for (int x = threadIdx.x; x < kBlockTileK; x += blockDim.x) {
                __pipeline_memcpy_async(&sA[y][x], &A[INDX(by0 + y, k0 + x, m)], sizeof(float));
            }
        }
        // Load kBlockTileK x kBlockTileN tile of B into shared memory
        for (int y = threadIdx.y; y < kBlockTileK; y += blockDim.y) {
            for (int x = threadIdx.x; x < kBlockTileN; x += blockDim.x) {
                __pipeline_memcpy_async(&sB[y][x], &B[INDX(k0 + y, bx0 + x, m)], sizeof(float));
            }
        }
        __pipeline_commit();
        __pipeline_wait_prior(0);
        __syncthreads();

        for (int ry = 0; ry < kThreadTileM; ry++) {
            int sy = threadIdx.y + ry * blockDim.y;
            for (int k = 0; k < kBlockTileK; k++) {
                regA[k] = sA[sy][k];
            }
            for (int k = 0; k < kBlockTileK; k++) {
                for (int rx = 0; rx < kThreadTileN; rx++) {
                    int sx = threadIdx.x + rx * blockDim.x;
                    regC[ry][rx] += regA[k] * sB[k][sx];
                }
            }
        }
        __syncthreads();
    }

    // Write back to global memory
    for (int ry = 0; ry < kThreadTileM; ry++) {
        for (int rx = 0; rx < kThreadTileN; rx++) {
            int sx = threadIdx.x + rx * blockDim.x;
            int sy = threadIdx.y + ry * blockDim.y;
            C[INDX(by0 + sy, bx0 + sx, m)] = regC[ry][rx];
        }
    }
}

void runTileMatmul(const MatmulBenchCtx &ctx, const KernelSpec &spec) {
    int m = ctx.m;
    dim3 block(kBlockTileN / kThreadTileN, kBlockTileM / kThreadTileM);
    dim3 grid(cuda::ceil_div(m, kBlockTileN), cuda::ceil_div(m, kBlockTileM));

    auto reset = [&]() { CUDA_CHECK(cudaMemset(ctx.C, 0, ctx.numElems * sizeof(float))); };
    auto fetch = [&]() { return std::vector<float>(ctx.C, ctx.C + ctx.numElems); };
    Stats stats;
    auto result = runKernelBenchmark<std::vector<float>>(
        [&]() { tileMatmulKernel<<<grid, block>>>(ctx.A, ctx.B, ctx.C, m); }, reset, fetch,
        ctx.warmup, ctx.iters, stats);

    std::string label = specLabel(spec);
    printStats(label.c_str(), stats, ctx.flops);
    if (vectorApproximatelyEqual(result.data(), ctx.ref, ctx.numElems)) {
        printf("  -> correct\n");
    } else {
        printf("  -> INCORRECT\n");
    }
}
