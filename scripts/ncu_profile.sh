#!/bin/bash
set -euo pipefail

BIN="build/matmul"
M="${1:-4096}"
SPEC="${2:?Usage: $0 [m] <kernel-spec> [ncu-extra-args...]}"
shift 2

mkdir -p build profiles

nvcc -O2 -lineinfo -std=c++17 -arch=sm_80 src/matmul/*.cu -o "$BIN" -lcublas

REPORT="profiles/ncu_$(echo "$SPEC" | tr ':=,' '_').ncu-rep"

ncu --set full \
    --export "$REPORT" \
    -f \
    "$@" \
    "$BIN" "$M" --warmup 0 --iters 1 --run "$SPEC"

echo "Report: $REPORT"
echo "Open with: ncu-ui $REPORT"
