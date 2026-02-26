#!/bin/bash
set -euo pipefail

NAME="${1:?Usage: $0 <name>}"
SRC="src/matmul/matmul.cu"

mkdir -p build logs

nvcc -g -lineinfo -cubin -arch=sm_80 "$SRC" -o "build/${NAME}.cubin"
nvdisasm -g "build/${NAME}.cubin" > "logs/${NAME}.cubin.log"

echo "Output: logs/${NAME}.cubin.log"
