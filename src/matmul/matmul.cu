#include <cstdlib>
#include <cstring>
#include <ctime>
#include <cublas_v2.h>
#include <cuda/cmath>
#include <cuda/pipeline>
#include <cuda_pipeline.h>
#include <getopt.h>
#include <stdio.h>
#include <string>
#include <unordered_map>

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

// ---------------------------------------------------------------------------
// Benchmark runner
// ---------------------------------------------------------------------------

struct KernelSpec {
    std::string name;
    std::unordered_map<std::string, int> params;

    int at(const char *key) const {
        auto it = params.find(key);
        if (it == params.end()) {
            fprintf(stderr, "Missing param '%s' for kernel '%s'\n", key, name.c_str());
            exit(1);
        }
        return it->second;
    }
};

KernelSpec parseKernelSpec(const char *str) {
    KernelSpec spec;
    std::string s(str);
    size_t colon = s.find(':');
    spec.name = s.substr(0, colon);
    if (colon != std::string::npos) {
        std::string rest = s.substr(colon + 1);
        size_t pos = 0;
        while (pos < rest.size()) {
            size_t eq = rest.find('=', pos);
            size_t comma = rest.find(',', eq);
            if (comma == std::string::npos)
                comma = rest.size();
            std::string key = rest.substr(pos, eq - pos);
            int val = atoi(rest.substr(eq + 1, comma - eq - 1).c_str());
            spec.params[key] = val;
            pos = comma + 1;
        }
    }
    return spec;
}

std::string specLabel(const KernelSpec &spec) {
    std::string label = spec.name;
    for (auto &kv : spec.params)
        label += " " + kv.first + "=" + std::to_string(kv.second);
    return label;
}

void initArray(float *arr, int len) {
    std::srand(std::time(nullptr));
    for (int i = 0; i < len; i++)
        arr[i] = rand() / (float)RAND_MAX;
}

void cublasMatmul(cublasHandle_t handle, float *A, float *B, float *C, int m) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    CUBLAS_CHECK(
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, m, m, &alpha, B, m, A, m, &beta, C, m));
}

void runKernel(const KernelSpec &spec, float *A, float *B, float *C, int m, int warmup, int iters,
               double flops, const float *refResult, size_t numElems) {
    Stats stats;
    std::vector<float> result;
    std::string label = specLabel(spec);

    auto reset = [&]() { CUDA_CHECK(cudaMemset(C, 0, numElems * sizeof(float))); };
    auto fetch = [&]() { return std::vector<float>(C, C + numElems); };

    if (spec.name == "base") {
        int bm = spec.at("bm");
        dim3 block(bm, bm);
        dim3 grid(cuda::ceil_div(m, bm), cuda::ceil_div(m, bm));
        size_t smem = (bm * (bm + 1) + bm * bm) * sizeof(float);
        result = runKernelBenchmark<std::vector<float>>(
            [&]() { baseMatmulKernel<<<grid, block, smem>>>(A, B, C, m, bm); }, reset, fetch,
            warmup, iters, stats);
    } else if (spec.name == "mio") {
        int bdim = spec.at("bdim");
        dim3 block(bdim);
        dim3 grid(cuda::ceil_div(m, kTileN), cuda::ceil_div(m, kTileM));
        result = runKernelBenchmark<std::vector<float>>(
            [&]() { mioOptimizedMatmulKernel<<<grid, block>>>(A, B, C, m); }, reset, fetch, warmup,
            iters, stats);
    } else {
        printf("Unknown kernel: %s\n", spec.name.c_str());
        return;
    }

    printStats(label.c_str(), stats, flops);
    if (vectorApproximatelyEqual(result.data(), refResult, numElems)) {
        printf("  -> correct\n");
    } else {
        printf("  -> INCORRECT\n");
    }
}

void printUsage(const char *prog) {
    printf("Usage: %s <m> [--warmup N] [--iters N] --run SPEC [--run SPEC ...]\n", prog);
    printf("  SPEC = name:key=val,key=val,...\n");
    printf("  e.g.  --run base:bm=32 --run mio:bm=16,bn=16,bk=32\n");
}

int main(int argc, char **argv) {
    int warmup = 5;
    int iters = 10;
    std::vector<KernelSpec> specs;

    static struct option long_options[] = {{"warmup", required_argument, nullptr, 'w'},
                                           {"iters", required_argument, nullptr, 'i'},
                                           {"run", required_argument, nullptr, 'r'},
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
        case 'r':
            specs.push_back(parseKernelSpec(optarg));
            break;
        default:
            printUsage(argv[0]);
            return 1;
        }
    }

    if (argc - optind != 1) {
        printUsage(argv[0]);
        return 1;
    }
    int m = atoi(argv[optind]);

    if (m <= 0 || warmup < 0 || iters <= 0) {
        printf("Error: m > 0, warmup >= 0, iters > 0\n");
        return 1;
    }
    if (specs.empty()) {
        printf("Error: at least one --run SPEC required\n");
        printUsage(argv[0]);
        return 1;
    }

    printf("m=%d, warmup=%d, iters=%d, kernels=%zu\n", m, warmup, iters, specs.size());

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

    Stats cublasStats;
    auto cublasResult = runKernelBenchmark<std::vector<float>>(
        [&]() { cublasMatmul(handle, A, B, C_cublas, m); },
        [&]() { CUDA_CHECK(cudaMemset(C_cublas, 0, numElems * sizeof(float))); },
        [&]() { return std::vector<float>(C_cublas, C_cublas + numElems); }, warmup, iters,
        cublasStats);
    printStats("cublas", cublasStats, flops);

    for (auto &spec : specs)
        runKernel(spec, A, B, C, m, warmup, iters, flops, cublasResult.data(), numElems);

    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaFree(A));
    CUDA_CHECK(cudaFree(B));
    CUDA_CHECK(cudaFree(C));
    CUDA_CHECK(cudaFree(C_cublas));
    return 0;
}
