#!/bin/bash
set -euo pipefail

BIN="build/matmul_bf16"
M="${1:-4096}"
SPEC="${2:?Usage: $0 [m] <kernel-spec> [label] [ncu-extra-args...]}"
LABEL="${3:-}"
shift 2
[[ -n "$LABEL" ]] && shift

mkdir -p build profiles

nvcc -O2 -lineinfo -std=c++17 -arch=sm_80 src/matmul_bf16/*.cu -o "$BIN" -lcublas

SUFFIX=$(echo "$SPEC" | tr ':=,' '_')
[[ -n "$LABEL" ]] && SUFFIX="${SUFFIX}_${LABEL}"
REPORT="profiles/ncu_${SUFFIX}.ncu-rep"

ncu --set full \
    --export "$REPORT" \
    -f \
    "$@" \
    "$BIN" "$M" --warmup 0 --iters 1 --run "$SPEC"

echo "Report: $REPORT"
echo "Open with: ncu-ui $REPORT"
