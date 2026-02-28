#!/bin/bash
set -euo pipefail

BIN="build/matmul_bf16"

mkdir -p build

nvcc -O2 -std=c++17 -arch=sm_80 src/matmul_bf16/*.cu -o "$BIN" -lcublas
"$BIN" "$@"
