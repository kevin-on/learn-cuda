#include <cuda/cmath>
#include <cuda/pipeline>
#include <cuda_pipeline.h>
#include <vector>

#include "../utils/cuda_utils.cuh"
#include "matmul.cuh"

#define INDX(row, col, ld) ((row) * (ld) + (col))

constexpr int kTileM = 8;
constexpr int kTileN = 8;
constexpr int kTileK = 64;

__global__ void mioOptimizedMatmulKernel(float *A, float *B, float *C, int m) {
  /**
   * Problem of base kernel:
   * - Every add instruction requires two shared memory accesses.
   * - ncu profiling shows that MIO Throttle is the main bottleneck of the base kernel.
   *
   * Solution:
   * - Idea: Store smem values in registers and reuse as much as possible.
   * - A single thread block computes kTileM x kTileN elements of the output matrix C.
   * - A single thread of this kernel:
   *   (1) loads kTileM x 1 column of A into registers.
   *   (2) loads 1 x kTileN row of B into registers.
   *   (3) computes kTileM x kTileN elements of the output matrix C.
   *   (4) A thread block aggregates the results of all threads.
   * - gridDim: (m / kTileN, m / kTileM)
   * - blockDim: any divisor of kTileK
   *   - Each thread loads & computes every threadIdx th (column of A, row of B) into registers.
   *
   * Important Notes:
   * - To keep kTileM x kTileN output matrix in registers, kTileM * kTileN should not exceed max
   * registers per thread = 256 for A100. If it exceeds, output matrix will spill to local memory,
   * which makes the kernel significantly slower.
   *
   * Benchmark (8192 x 8192 x 8192 matmul on A100):
   * - kTileM=8, kTileN=8, kTileK=64, blockDim=16: 1.87 TFLOPS
   * - kTileM=8, kTileN=8, kTileK=64, blockDim=32: 2.14 TFLOPS
   * - kTileM=8, kTileN=8, kTileK=64, blockDim=64: 2.05 TFLOPS & INCORRECT RESULTS
   */

  __shared__ float sA[kTileM * (kTileK + 1)];
  __shared__ float sB[kTileK * kTileN];
  int ldsa = kTileK + 1;
  int ldsb = kTileN;

  // global offset
  int y0 = blockIdx.y * kTileM;
  int x0 = blockIdx.x * kTileN;

  float regC[kTileM][kTileN] = {0};

  for (int k0 = 0; k0 < m; k0 += kTileK) {
    // Load kTileM x kTileK tile of A into shared memory
    // Load kTileK x kTileN tile of B into shared memory
    for (int y = 0; y < kTileM; y++) {
      for (int x = threadIdx.x; x < kTileK; x += blockDim.x) {
        __pipeline_memcpy_async(&sA[INDX(y, x, ldsa)], &A[INDX(y0 + y, k0 + x, m)],
                                sizeof(float));
      }
    }

    // Assume blockDim.x is divisible by kTileN
    int bLoadBatchSize = blockDim.x / kTileN;
    int bLoadRow = threadIdx.x / kTileN;
    int bLoadCol = threadIdx.x % kTileN;
    for (int y = bLoadRow; y < kTileK; y += bLoadBatchSize) {
      __pipeline_memcpy_async(&sB[INDX(y, bLoadCol, ldsb)],
                              &B[INDX(k0 + y, x0 + bLoadCol, m)], sizeof(float));
    }

    __pipeline_commit();
    __pipeline_wait_prior(0);
    __syncthreads();

    float regA[kTileM] = {0};
    float regB[kTileN] = {0};

    for (int k = threadIdx.x; k < kTileK; k += blockDim.x) {
      for (int y = 0; y < kTileM; y++) {
        regA[y] = sA[INDX(y, k, ldsa)];
      }
      for (int x = 0; x < kTileN; x++) {
        regB[x] = sB[INDX(k, x, ldsb)];
      }
      for (int x = 0; x < kTileN; x++) {
        for (int y = 0; y < kTileM; y++) {
          regC[y][x] += regA[y] * regB[x];
        }
      }
    }
  }
  __syncthreads();

  // Write back to global memory

  /**
   * Method 1: Manual offset rotation
   * x is runtime-computed, so the compiler cannot map regC[y][x] to a specific register.
   * This forces regC to spill to local memory (stack).
   */
  // for (int offset = 0; offset < blockDim.x; offset++) {
  //     int tx0 = threadIdx.x + offset;
  //     if (tx0 >= blockDim.x)
  //         tx0 -= blockDim.x;
  //     for (int y = 0; y < kTileM; y++) {
  //         for (int x = tx0; x < kTileN; x += blockDim.x) {
  //             C[INDX(y0 + y, x0 + x, m)] += regC[y][x];
  //         }
  //     }
  //     __syncthreads();
  // }

  /**
   * Method 2: Atomic add
   * Both y and x are compile-time constants, so the compiler maps each regC[y][x] to a dedicated
   * register. No spill.
   * But latency is similar because all threads in the block access the same addresses in the same
   * order, maximizing atomic contention.
   */
  for (int y = 0; y < kTileM; y++) {
    for (int x = 0; x < kTileN; x++) {
      atomicAdd(&C[INDX(y0 + y, x0 + x, m)], regC[y][x]);
    }
  }
}

void runMioMatmul(const MatmulBenchCtx &ctx, const KernelSpec &spec) {
  int bdim = spec.at("bdim");
  int m = ctx.m;
  dim3 block(bdim);
  dim3 grid(cuda::ceil_div(m, kTileN), cuda::ceil_div(m, kTileM));

  auto reset = [&]() { CUDA_CHECK(cudaMemset(ctx.C, 0, ctx.numElems * sizeof(float))); };
  auto fetch = [&]() { return std::vector<float>(ctx.C, ctx.C + ctx.numElems); };

  Stats stats;
  auto result = runKernelBenchmark<std::vector<float>>(
      [&]() { mioOptimizedMatmulKernel<<<grid, block>>>(ctx.A, ctx.B, ctx.C, m); },
      reset, fetch, ctx.warmup, ctx.iters, stats);

  std::string label = specLabel(spec);
  printStats(label.c_str(), stats, ctx.flops);
  if (vectorApproximatelyEqual(result.data(), ctx.ref, ctx.numElems)) {
    printf("  -> correct\n");
  } else {
    printf("  -> INCORRECT\n");
  }
}
