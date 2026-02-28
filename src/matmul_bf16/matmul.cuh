#pragma once

#include <cstdio>
#include <cstdlib>
#include <string>
#include <unordered_map>

#include <cuda_bf16.h>

using bf16 = __nv_bfloat16;

struct MatmulBenchCtx {
    bf16 *A, *B, *C;
    int m;
    int warmup, iters;
    double flops;
    const bf16 *ref;
    size_t numElems;
};

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

inline std::string specLabel(const KernelSpec &spec) {
    std::string label = spec.name;
    for (auto &kv : spec.params)
        label += " " + kv.first + "=" + std::to_string(kv.second);
    return label;
}

using RunFn = void (*)(const MatmulBenchCtx &ctx, const KernelSpec &spec);
