#!/bin/bash
set -euo pipefail

SRC="src/matmul/matmul.cu"
BIN="build/matmul"

mkdir -p build

nvcc -O2 -std=c++17 -arch=sm_80 "$SRC" -o "$BIN" -lcublas
"$BIN" "$@"
