#!/bin/bash
set -euo pipefail

BIN="build/matmul"
M="${1:-8192}"
shift || true
EXTRA_ARGS=("$@")

mkdir -p build
nvcc -O2 -std=c++17 -arch=sm_80 src/matmul_fp32/*.cu -o "$BIN" -lcublas

CONFIGS=(
  # 8x8 tile
  "bm=64,bn=64,bk=8,tm=8,tn=8"
  "bm=64,bn=64,bk=16,tm=8,tn=8"
  "bm=64,bn=64,bk=32,tm=8,tn=8"
  "bm=128,bn=128,bk=8,tm=8,tn=8"
  "bm=128,bn=128,bk=16,tm=8,tn=8"
  "bm=128,bn=128,bk=32,tm=8,tn=8"
  # 4x8 tile
  "bm=64,bn=64,bk=8,tm=4,tn=8"
  "bm=64,bn=64,bk=16,tm=4,tn=8"
  "bm=64,bn=64,bk=32,tm=4,tn=8"
  "bm=128,bn=128,bk=8,tm=4,tn=8"
  # 4x16 tile
  "bm=64,bn=64,bk=8,tm=4,tn=16"
  "bm=64,bn=64,bk=16,tm=4,tn=16"
  "bm=64,bn=64,bk=32,tm=4,tn=16"
  "bm=128,bn=128,bk=8,tm=4,tn=16"
  "bm=128,bn=128,bk=16,tm=4,tn=16"
  "bm=128,bn=128,bk=32,tm=4,tn=16"
)

ARGS=()
for cfg in "${CONFIGS[@]}"; do
  ARGS+=(--run "tile:$cfg")
done

"$BIN" "$M" "${EXTRA_ARGS[@]}" "${ARGS[@]}"
