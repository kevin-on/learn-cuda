# CUDA Naming Conventions (NVIDIA-flavored)

This note summarizes practical naming patterns commonly seen across NVIDIA codebases and CUDA projects.
There is no single universal NVIDIA standard, but the rules below cover the most common and readable choices.

## 1) C++ naming (Carbonite / Omniverse style)
Use these when you want a consistent C++ codebase style.

- Types (class/struct/enum/typedef): `PascalCase`
  - Example: `GemmConfig`, `MemoryPool`, `DataType`
- Functions: `camelCase()`
  - Private or protected functions: `_camelCase()`
- Constants: `kCamelCase`
  - Example: `constexpr int kBlockSize = 256;`
- Enum values: `eCamelCase`
  - Example: `enum class Layout { eRowMajor, eColMajor };`
- Members:
  - Public members: `camelCase`
  - Private or protected members: `m_camelCase`
  - Private or protected static members: `s_camelCase`
- Globals (file scope): `g_camelCase`
- Locals: `camelCase`
- Acronyms: keep the same casing rule
  - Prefer `gpuBuffer`, `cudaStream` over `GPUBuffer`, `CUDAStream`

## 2) CUDA kernel naming
Pick one and apply consistently.

- Suffix `Kernel`: `reduceSumKernel`, `gemmKernel`, `fusedAttentionKernel`
- Or prefix `kernel`: `kernelReduceSum`

Tip: adding `Kernel` makes call sites and symbol searches easier.

## 3) CUDA indexing variables (common short forms)
These names are widely understood in CUDA code reviews.

- `tid` for `threadIdx.x`
- `bid` for `blockIdx.x`
- `idx` for global linear index
- `lane` for warp lane id
- `warpId` for warp index in a block

Example:

    __global__ void reduceSumKernel(const float* inPtr, float* outPtr, int n) {
      int tid = threadIdx.x;
      int bid = blockIdx.x;
      int idx = bid * blockDim.x + tid;

      int lane = tid & 31;
      int warpId = tid >> 5;
      // ...
    }

## 4) Pointer naming (general)
Use `Ptr` suffix if you want the pointer-ness to be obvious.

- `inPtr`, `outPtr`, `workspacePtr`
- If you prefer shorter: `in`, `out`, `workspace` is also fine, but be consistent.

## 5) Matrix pointer naming
Two good options exist. Choose based on your code style.

Option A: camelCase + pointer clarity (recommended for general C++ style)
- `aPtr`, `bPtr`, `cPtr`
- BLAS-style leading dimensions: `lda`, `ldb`, `ldc`
- Sizes: `m`, `n`, `k` are fine

    __global__ void gemmKernel(
        const float* aPtr, const float* bPtr, float* cPtr,
        int m, int n, int k,
        int lda, int ldb, int ldc) {
      // ...
    }

Option B: BLAS notation (common in GEMM-focused code)
- `A`, `B`, `C` for matrix pointers
- `lda`, `ldb`, `ldc`

    __global__ void gemmKernel(
        const float* A, const float* B, float* C,
        int m, int n, int k,
        int lda, int ldb, int ldc) {
      // ...
    }

Avoid `a`, `b` for long-lived pointers unless the scope is tiny and meaning is obvious.

## 6) Host vs Device pointers (classic CUDA convention)
If your project uses separate host and device allocations, these prefixes reduce mistakes.

- Host pointer: `h_A`, `h_B` (NVIDIA samples often use underscore)
- Device pointer: `d_A`, `d_B`
- Alternative if your project is camelCase-heavy: `hA`, `dA`

Optional: add `Ptr` for clarity: `hAPtr`, `dAPtr`.

## 7) Unified Memory pointer naming (managed memory)
Unified Memory is a single pointer usable on CPU and GPU. Three practical conventions exist.

Pattern A: no memory tag, name by meaning (cleanest if you mostly use Unified Memory)
- Examples: `aPtr`, `data`, `result`, `weightsPtr`

Pattern B: keep `hptr` and `dptr` naming for structure, alias them for Unified Memory
- Use when you support multiple memory paths (Memcpy path, Unified Memory path, etc)
- Example idea: `hptrA = dptrA` when using managed allocation

Pattern C: explicitly mark Unified Memory
- Prefix examples: `umA`, `umData`, `managedA`, `managedPtr`
- Use when your code mixes multiple memory types and you want a clear visual signal.

Recommendation:
- If Unified Memory is your default: Pattern A
- If you benchmark or switch between memory modes: Pattern B

## 8) constexpr in CUDA code
`constexpr` is for values that can be evaluated at compile time.

- Prefer `constexpr` over macros for typed constants with scope.
- Use `kCamelCase` for constants if you follow Carbonite style.

Examples:

    constexpr int kWarpSize = 32;
    constexpr int kNumWarps = 8;
    constexpr int kBlockSize = kWarpSize * kNumWarps;

    constexpr int divUp(int a, int b) { return (a + b - 1) / b; }
    constexpr int blocks = divUp(1024, 256);

`if constexpr` can remove unused branches at compile time in templated device code.

    template<bool kUseVectorized>
    __device__ void load(...) {
      if constexpr (kUseVectorized) {
        // vectorized path
      } else {
        // scalar path
      }
    }

## 9) Practical defaults (recommended)
If you want a simple, consistent baseline:

- C++ code: Carbonite style (`PascalCase`, `camelCase`, `kX`, `m_`, `s_`, `g_`)
- Kernels: `*Kernel` suffix
- Device pointers: `d_A` / `dA`, host pointers: `h_A` / `hA`
- Unified Memory: either meaning-only names (`dataPtr`) or `umDataPtr`
- Matrices: `aPtr/bPtr/cPtr` or `A/B/C`, plus `lda/ldb/ldc`