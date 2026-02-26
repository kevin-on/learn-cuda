#include <cuda/cmath>
#include <vector>

#include "../utils/cuda_utils.cuh"
#include "matmul.cuh"

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
    float *sA = smem;                 // bm x (bm+1)
    float *sB = smem + bm * (bm + 1); // bm x bm
    int ldsa = bm + 1;
    int ldsb = bm;

    int blockRow = blockIdx.y * bm;
    int blockCol = blockIdx.x * bm;

    float valC = 0.0f;
    for (int i = 0; i < cuda::ceil_div(m, bm); i++) {
        // sA origin: (blockRow, i*bm), sB origin: (i*bm, blockCol)
        // Commenting out all global memory loads => 4.70 TFLOPS
        sA[INDX(threadIdx.y, threadIdx.x, ldsa)] =
            A[INDX(blockRow + threadIdx.y, i * bm + threadIdx.x, m)];
        sB[INDX(threadIdx.y, threadIdx.x, ldsb)] =
            B[INDX(i * bm + threadIdx.y, blockCol + threadIdx.x, m)];
        __syncthreads();

        for (int k = 0; k < bm; k++) {
            valC += sA[INDX(threadIdx.y, k, ldsa)] * sB[INDX(k, threadIdx.x, ldsb)];
            // Cval += 1.0f * sB[INDX(k, threadIdx.x, ldsb)]; => 5.72 TFLOPS (reduce shared memory
            // access into half)
        }
        __syncthreads();
    }
    C[INDX(blockRow + threadIdx.y, blockCol + threadIdx.x, m)] = valC;
}

void runBaseMatmul(const MatmulBenchCtx &ctx, const KernelSpec &spec) {
    int bm = spec.at("bm");
    int m = ctx.m;
    dim3 block(bm, bm);
    dim3 grid(cuda::ceil_div(m, bm), cuda::ceil_div(m, bm));
    size_t smem = (bm * (bm + 1) + bm * bm) * sizeof(float);

    auto reset = [&]() { CUDA_CHECK(cudaMemset(ctx.C, 0, ctx.numElems * sizeof(float))); };
    auto fetch = [&]() { return std::vector<float>(ctx.C, ctx.C + ctx.numElems); };

    Stats stats;
    auto result = runKernelBenchmark<std::vector<float>>(
        [&]() { baseMatmulKernel<<<grid, block, smem>>>(ctx.A, ctx.B, ctx.C, m, bm); }, reset,
        fetch, ctx.warmup, ctx.iters, stats);

    std::string label = specLabel(spec);
    printStats(label.c_str(), stats, ctx.flops);
    if (vectorApproximatelyEqual(result.data(), ctx.ref, ctx.numElems)) {
        printf("  -> correct\n");
    } else {
        printf("  -> INCORRECT\n");
    }
}
