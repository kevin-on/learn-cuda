#include <cuda/cmath>
#include <cuda_pipeline.h>

#include "../utils/cuda_utils.cuh"
#include "matmul.cuh"

#define INDX(row, col, ld) ((row) * (ld) + (col))

template <int BM, int BN, int BK, int TM, int TN>
__global__ void tileMatmulKernel(float *A, float *B, float *C, int m) {
    /**
     * - Each thread block computes a BM x BN tile of the output matrix C.
     * - Each thread computes a TM x TN tile of the output matrix C.
     * - In each iteration, a thread block loads a BM x BK tile of A and a
     * BK x BN tile of B into shared memory.
     * - gridDim: (m / BN, m / BM)
     * - blockDim: (BN / TN, BM / TM)
     *
     * Benchmark (8192 x 8192 x 8192 matmul on A100)
     * Note: cuBLAS=19.16 TFLOPS, theoretical peak=19.5 TFLOPS
     *   BM   BN   BK  TM  TN  TFLOPS
     *   64   64    8   8   8    6.31
     *   64   64   16   8   8    8.84
     *   64   64   32   8   8    7.11
     *  128  128    8   8   8    7.21
     *  128  128   16   8   8   10.19  🔥🔥🔥 HOLY $#@% 10 TFLOPS BABY LET'S GOOOOOOOO 🚀🚀🚀🚀🚀
     *  128  128   32   8   8    8.10
     *   64   64    8   4   8    7.39
     *   64   64   16   4   8    8.94
     *   64   64   32   4   8    6.80
     *  128  128    8   4   8    7.25
     *   64   64    8   4  16    6.30
     *   64   64   16   4  16    6.05
     *   64   64   32   4  16    5.66
     *  128  128    8   4  16    6.31
     *  128  128   16   4  16    6.84
     *  128  128   32   4  16    6.95
     */

    __shared__ float sA[BM][(BK + 1)];
    __shared__ float sB[BK][BN];

    int bx0 = blockIdx.x * BN;
    int by0 = blockIdx.y * BM;

    float regC[TM][TN] = {0};
    float regA[BK] = {0};

    for (int k0 = 0; k0 < m; k0 += BK) {
        for (int y = threadIdx.y; y < BM; y += blockDim.y) {
            for (int x = threadIdx.x; x < BK; x += blockDim.x) {
                __pipeline_memcpy_async(&sA[y][x], &A[INDX(by0 + y, k0 + x, m)], sizeof(float));
            }
        }
        for (int y = threadIdx.y; y < BK; y += blockDim.y) {
            for (int x = threadIdx.x; x < BN; x += blockDim.x) {
                __pipeline_memcpy_async(&sB[y][x], &B[INDX(k0 + y, bx0 + x, m)], sizeof(float));
            }
        }
        __pipeline_commit();
        __pipeline_wait_prior(0);
        __syncthreads();

        for (int ry = 0; ry < TM; ry++) {
            int sy = threadIdx.y + ry * blockDim.y;
            for (int k = 0; k < BK; k++) {
                regA[k] = sA[sy][k];
            }
            for (int k = 0; k < BK; k++) {
                for (int rx = 0; rx < TN; rx++) {
                    int sx = threadIdx.x + rx * blockDim.x;
                    regC[ry][rx] += regA[k] * sB[k][sx];
                }
            }
        }
        __syncthreads();
    }

    for (int ry = 0; ry < TM; ry++) {
        for (int rx = 0; rx < TN; rx++) {
            int sx = threadIdx.x + rx * blockDim.x;
            int sy = threadIdx.y + ry * blockDim.y;
            C[INDX(by0 + sy, bx0 + sx, m)] = regC[ry][rx];
        }
    }
}

template <int BM, int BN, int BK, int TM, int TN>
void launchTileMatmul(const MatmulBenchCtx &ctx, const KernelSpec &spec) {
    int m = ctx.m;
    dim3 block(BN / TN, BM / TM);
    dim3 grid(cuda::ceil_div(m, BN), cuda::ceil_div(m, BM));

    auto reset = [&]() { CUDA_CHECK(cudaMemset(ctx.C, 0, ctx.numElems * sizeof(float))); };
    auto fetch = [&]() { return std::vector<float>(ctx.C, ctx.C + ctx.numElems); };
    Stats stats;
    auto result = runKernelBenchmark<std::vector<float>>(
        [&]() { tileMatmulKernel<BM, BN, BK, TM, TN><<<grid, block>>>(ctx.A, ctx.B, ctx.C, m); },
        reset, fetch, ctx.warmup, ctx.iters, stats);

    std::string label = specLabel(spec);
    printStats(label.c_str(), stats, ctx.flops);
    if (vectorApproximatelyEqual(result.data(), ctx.ref, ctx.numElems)) {
        printf("  -> correct\n");
    } else {
        printf("  -> INCORRECT\n");
    }
}

// --- Config dispatch table ---

using TileFn = void (*)(const MatmulBenchCtx &, const KernelSpec &);

struct TileConfig {
    int bm, bn, bk, tm, tn;
    TileFn fn;
};

#define TILE_CFG(BM, BN, BK, TM, TN)                                                               \
    { BM, BN, BK, TM, TN, launchTileMatmul<BM, BN, BK, TM, TN> }

static const TileConfig tileConfigs[] = {
    // 8x8 tile
    TILE_CFG(64, 64, 8, 8, 8),
    TILE_CFG(64, 64, 16, 8, 8),
    TILE_CFG(64, 64, 32, 8, 8),
    TILE_CFG(128, 128, 8, 8, 8),
    TILE_CFG(128, 128, 16, 8, 8),
    TILE_CFG(128, 128, 32, 8, 8),
    // 4x8 tile
    TILE_CFG(64, 64, 8, 4, 8),
    TILE_CFG(64, 64, 16, 4, 8),
    TILE_CFG(64, 64, 32, 4, 8),
    TILE_CFG(128, 128, 8, 4, 8),
    // 4x16 tile
    TILE_CFG(64, 64, 8, 4, 16),
    TILE_CFG(64, 64, 16, 4, 16),
    TILE_CFG(64, 64, 32, 4, 16),
    TILE_CFG(128, 128, 8, 4, 16),
    TILE_CFG(128, 128, 16, 4, 16),
    TILE_CFG(128, 128, 32, 4, 16),
};

void runTileMatmul(const MatmulBenchCtx &ctx, const KernelSpec &spec) {
    int bm = spec.at("bm"), bn = spec.at("bn"), bk = spec.at("bk");
    int tm = spec.at("tm"), tn = spec.at("tn");
    for (auto &cfg : tileConfigs) {
        if (cfg.bm == bm && cfg.bn == bn && cfg.bk == bk && cfg.tm == tm && cfg.tn == tn) {
            cfg.fn(ctx, spec);
            return;
        }
    }
    fprintf(stderr, "No compiled config for bm=%d bn=%d bk=%d tm=%d tn=%d\n", bm, bn, bk, tm, tn);
}
