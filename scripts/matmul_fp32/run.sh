#!/bin/bash
set -euo pipefail

BIN="build/matmul"

mkdir -p build

nvcc -O2 -std=c++17 -arch=sm_80 src/matmul_fp32/*.cu -o "$BIN" -lcublas
"$BIN" "$@"
