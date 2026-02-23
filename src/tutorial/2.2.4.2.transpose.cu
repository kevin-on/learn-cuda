#include "cuda_utils.cuh"
#include <ctime>
#include <cuda/cmath>
#include <stdio.h>

#define INDX(row, col, ld) (row * ld + col)

constexpr int kThreadsPerBlockX = 32;
constexpr int kThreadsPerBlockY = 32;

__global__ void naiveTransposeKernel(int m, float *A, float *C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m && col < m) {
        C[INDX(row, col, m)] = A[INDX(col, row, m)]; // Transposed read from A is strided across a
                                                     // warp, so global loads are not coalesced.
    }
}

__global__ void smemTransposeKernel(int m, float *A, float *C) {
    // Use shared memory to make global loads coalesced.
    __shared__ float smem[kThreadsPerBlockX][kThreadsPerBlockY];
    int rowA = blockIdx.x * blockDim.y + threadIdx.y;
    int colA = blockIdx.y * blockDim.x + threadIdx.x;
    if (rowA < m && colA < m) {
        smem[threadIdx.x][threadIdx.y] = A[INDX(
            rowA, colA, m)]; // For a warp (x=0..31, y fixed), smem[x][y] has stride 32 floats, so
                             // bank=(x*32+y)%32=y: all lanes hit one bank (32-way conflict).
    }
    __syncthreads();
    int rowC = blockIdx.y * blockDim.y + threadIdx.y;
    int colC = blockIdx.x * blockDim.x + threadIdx.x;
    if (rowC < m && colC < m) {
        C[INDX(rowC, colC, m)] = smem[threadIdx.y][threadIdx.x];
    }
}

__global__ void smemTransposeWithoutBankConflictKernel(int m, float *A, float *C) {
    __shared__ float smem[kThreadsPerBlockX + 1]
                         [kThreadsPerBlockY + 1]; // +1 padding changes the stride and removes
                                                  // shared-memory bank conflicts.
    int rowA = blockIdx.x * blockDim.y + threadIdx.y;
    int colA = blockIdx.y * blockDim.x + threadIdx.x;
    if (rowA < m && colA < m) {
        smem[threadIdx.x][threadIdx.y] = A[INDX(rowA, colA, m)];
    }
    __syncthreads();
    int rowC = blockIdx.y * blockDim.y + threadIdx.y;
    int colC = blockIdx.x * blockDim.x + threadIdx.x;
    if (rowC < m && colC < m) {
        C[INDX(rowC, colC, m)] = smem[threadIdx.y][threadIdx.x];
    }
}

void sequentialTranspose(int m, float *A, float *C) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) {
            C[INDX(i, j, m)] = A[INDX(j, i, m)];
        }
    }
}

void initArray(float *A, int array_size) {
    std::srand(std::time({}));
    for (int i = 0; i < array_size; i++) {
        A[i] = rand() / (float)RAND_MAX;
    }
}

int main(int argc, char **argv) {
    if (argc < 2 || argc > 4) {
        printf("Usage: %s <matrix size> [warmup_iters] [timed_iters]\n", argv[0]);
        return 1;
    }
    int m = atoi(argv[1]);

    const size_t matrix_bytes = m * m * sizeof(float);
    int warmup_iters = (argc >= 3) ? atoi(argv[2]) : 5;
    int timed_iters = (argc >= 4) ? atoi(argv[3]) : 50;
    if (m <= 0 || warmup_iters < 0 || timed_iters <= 0) {
        printf("Error: matrix size must be > 0, warmup_iters >= 0, timed_iters > 0\n");
        return 1;
    }
    printf("Benchmark configuration: matrix size = %d, warmup_iters = %d, timed_iters = %d\n", m,
           warmup_iters, timed_iters);

    float *A = nullptr;
    float *d_C = nullptr;
    float *h_C = (float *)malloc(matrix_bytes);

    CUDA_CHECK(cudaMallocManaged(&A, matrix_bytes));
    CUDA_CHECK(cudaMallocManaged(&d_C, matrix_bytes));
    initArray(A, m * m);

    // sequential transpose in host
    sequentialTranspose(m, A, h_C);

    dim3 blockDim(kThreadsPerBlockX, kThreadsPerBlockY);
    dim3 gridDim(cuda::ceil_div(m, kThreadsPerBlockX), cuda::ceil_div(m, kThreadsPerBlockY));

    auto verifyTranspose = [&](const char *kernel_name) -> bool {
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < m; j++) {
                if (h_C[INDX(i, j, m)] != d_C[INDX(i, j, m)]) {
                    printf("[%s] Error: h_C[%d][%d] = %f, d_C[%d][%d] = %f\n", kernel_name, i, j,
                           h_C[INDX(i, j, m)], i, j, d_C[INDX(i, j, m)]);
                    return false;
                }
            }
        }
        printf("[%s] Results are correct\n", kernel_name);
        return true;
    };

    auto runBenchmark = [&](const char *kernel_name, auto launch_kernel) {
        Stats stats{};
        runKernelBenchmark<float>(
            launch_kernel, [&]() { CUDA_CHECK(cudaMemset(d_C, 0, matrix_bytes)); },
            []() {
                return 0.0f; // dummy fetch result
            },
            warmup_iters, timed_iters, stats);

        printf("[%s] Benchmark (%d warmup, %d timed): mean=%.3f ms, median=%.3f ms, std=%.3f ms, "
               "min=%.3f ms, max=%.3f ms\n",
               kernel_name, warmup_iters, timed_iters, stats.mean, stats.median, stats.std_dev,
               stats.min, stats.max);
    };

    CUDA_CHECK(cudaMemset(d_C, 0, matrix_bytes));
    naiveTransposeKernel<<<gridDim, blockDim>>>(m, A, d_C);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    if (!verifyTranspose("Naive transpose"))
        return 1;

    CUDA_CHECK(cudaMemset(d_C, 0, matrix_bytes));
    smemTransposeKernel<<<gridDim, blockDim>>>(m, A, d_C);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    if (!verifyTranspose("Smem transpose"))
        return 1;

    CUDA_CHECK(cudaMemset(d_C, 0, matrix_bytes));
    smemTransposeWithoutBankConflictKernel<<<gridDim, blockDim>>>(m, A, d_C);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    if (!verifyTranspose("Smem transpose without bank conflict"))
        return 1;

    runBenchmark("Naive transpose",
                 [&]() { naiveTransposeKernel<<<gridDim, blockDim>>>(m, A, d_C); });
    runBenchmark("Smem transpose",
                 [&]() { smemTransposeKernel<<<gridDim, blockDim>>>(m, A, d_C); });
    runBenchmark("Smem transpose without bank conflict", [&]() {
        smemTransposeWithoutBankConflictKernel<<<gridDim, blockDim>>>(m, A, d_C);
    });

    CUDA_CHECK(cudaFree(A));
    CUDA_CHECK(cudaFree(d_C));
    free(h_C);

    return 0;
}
