#include <cstdlib>
#include <ctime>
#include <cublas_v2.h>
#include <getopt.h>
#include <stdio.h>
#include <vector>

#include "../utils/cuda_utils.cuh"
#include "matmul.cuh"

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

void runBaseMatmul(const MatmulBenchCtx &ctx, const KernelSpec &spec);
void runMioMatmul(const MatmulBenchCtx &ctx, const KernelSpec &spec);

static const std::unordered_map<std::string, RunFn> kernels = {
    {"base", runBaseMatmul},
    {"mio", runMioMatmul},
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

void printUsage(const char *prog) {
  printf("Usage: %s <m> [--warmup N] [--iters N] --run SPEC [--run SPEC ...]\n", prog);
  printf("  SPEC = name:key=val,key=val,...\n");
  printf("  e.g.  --run base:bm=32 --run mio:bdim=32\n");
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

  MatmulBenchCtx ctx = {A, B, C, m, warmup, iters, flops,
                         cublasResult.data(), numElems};

  for (auto &spec : specs) {
    auto it = kernels.find(spec.name);
    if (it == kernels.end()) {
      printf("Unknown kernel: %s\n", spec.name.c_str());
      continue;
    }
    it->second(ctx, spec);
  }

  CUBLAS_CHECK(cublasDestroy(handle));
  CUDA_CHECK(cudaFree(A));
  CUDA_CHECK(cudaFree(B));
  CUDA_CHECK(cudaFree(C));
  CUDA_CHECK(cudaFree(C_cublas));
  return 0;
}
