#include <cuda/cmath>
#include <cuda_pipeline.h>

#include "../utils/cuda_utils.cuh"
#include "matmul.cuh"

#define INDX(row, col, ld) ((row) * (ld) + (col))

template <int BM, int BN, int BK, int TM, int TN>
__global__ void tileMatmulKernel(bf16 *A, bf16 *B, bf16 *C, int m) {

    __shared__ bf16 sA[BM][(BK + 2)]; // TODO: Use swizzle (XOR) to avoid bank conflicts
    __shared__ bf16 sB[BK][BN];

    int bx0 = blockIdx.x * BN;
    int by0 = blockIdx.y * BM;

    float regC[TM][TN] = {0};
    bf16 regA[BK] = {0};

    for (int k0 = 0; k0 < m; k0 += BK) {

        /* -- DRAM -> SRAM -- */
        for (int y = threadIdx.y; y < BM; y += blockDim.y) {
            for (int x = threadIdx.x; 2 * x < BK; x += blockDim.x) {
                // __pipeline_memcpy_async requires the size parameter to be 4, 8, or 16 bytes.
                __pipeline_memcpy_async(&sA[y][2 * x], &A[INDX(by0 + y, k0 + 2 * x, m)],
                                        2 * sizeof(bf16));
            }
        }
        for (int y = threadIdx.y; y < BK; y += blockDim.y) {
            for (int x = threadIdx.x; 2 * x < BN; x += blockDim.x) {
                __pipeline_memcpy_async(&sB[y][2 * x], &B[INDX(k0 + y, bx0 + 2 * x, m)],
                                        2 * sizeof(bf16));
            }
        }
        __pipeline_commit();
        __pipeline_wait_prior(0);
        __syncthreads();

        /* -- SRAM -> Register  -- */
        /* -- Register @ Register => Register  -- */
#pragma unroll
        for (int ry = 0; ry < TM; ry++) {
            int sy = threadIdx.y + ry * blockDim.y;
            for (int k = 0; k < BK; k++) {
                regA[k] = sA[sy][k];
            }
            for (int k = 0; k < BK; k++) {
                for (int rx = 0; rx < TN; rx++) {
                    int sx = threadIdx.x + rx * blockDim.x;
                    regC[ry][rx] += (float)regA[k] * (float)sB[k][sx];
                }
            }
        }
        __syncthreads();
    }

    /* -- Register -> (Cache) -> DRAM  -- */
    for (int ry = 0; ry < TM; ry++) {
        for (int rx = 0; rx < TN; rx++) {
            int sx = threadIdx.x + rx * blockDim.x;
            int sy = threadIdx.y + ry * blockDim.y;
            C[INDX(by0 + sy, bx0 + sx, m)] = __float2bfloat16(regC[ry][rx]);
        }
    }
}

template <int BM, int BN, int BK, int TM, int TN>
void launchTileMatmul(const MatmulBenchCtx &ctx, const KernelSpec &spec) {
    int m = ctx.m;
    dim3 block(BN / TN, BM / TM);
    dim3 grid(cuda::ceil_div(m, BN), cuda::ceil_div(m, BM));

    auto reset = [&]() { CUDA_CHECK(cudaMemset(ctx.C, 0, ctx.numElems * sizeof(bf16))); };
    auto fetch = [&]() { return std::vector<bf16>(ctx.C, ctx.C + ctx.numElems); };
    Stats stats;
    auto result = runKernelBenchmark<std::vector<bf16>>(
        [&]() { tileMatmulKernel<BM, BN, BK, TM, TN><<<grid, block>>>(ctx.A, ctx.B, ctx.C, m); },
        reset, fetch, ctx.warmup, ctx.iters, stats);

    std::string label = specLabel(spec);
    printStats(label.c_str(), stats, ctx.flops);
    if (vectorApproximatelyEqualBf16(result.data(), ctx.ref, ctx.numElems)) {
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
    // TILE_CFG(64, 64, 8, 8, 8),
    // TILE_CFG(64, 64, 16, 8, 8),
    // TILE_CFG(64, 64, 32, 8, 8),
    TILE_CFG(128, 128, 8, 8, 8), TILE_CFG(128, 128, 16, 8, 8), TILE_CFG(128, 128, 32, 8, 8),
    // // 4x8 tile
    // TILE_CFG(64, 64, 8, 4, 8),
    // TILE_CFG(64, 64, 16, 4, 8),
    // TILE_CFG(64, 64, 32, 4, 8),
    // TILE_CFG(128, 128, 8, 4, 8),
    // // 4x16 tile
    // TILE_CFG(64, 64, 8, 4, 16),
    // TILE_CFG(64, 64, 16, 4, 16),
    // TILE_CFG(64, 64, 32, 4, 16),
    // TILE_CFG(128, 128, 8, 4, 16),
    // TILE_CFG(128, 128, 16, 4, 16),
    // TILE_CFG(128, 128, 32, 4, 16),
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
