#!/usr/bin/env bash
# Build ERNIE 5.5 SM121a CUDA fatbins for TensorFold.
set -euo pipefail

export PATH=/usr/local/cuda/bin:$PATH
TF_DIR="${TF_DIR:?Set TF_DIR to TensorFold/zig}"
FATBIN_OUT="${FATBIN_OUT:-$PWD/fatbins}"
mkdir -p "$FATBIN_OUT"

TORCH="-D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ --expt-relaxed-constexpr -std=c++20"
GLUE="-O3 --fmad=false --ftz=false --prec-div=true --prec-sqrt=true"
GEN="-gencode=arch=compute_121a,code=sm_121a -gencode=arch=compute_121a,code=compute_121a -gencode=arch=compute_121,code=compute_121"

build() {
  local name="$1" src="$2" flags="$3"
  nvcc -fatbin $TORCH $flags -I"$TF_DIR/kernels/cuda" $GEN -o "$FATBIN_OUT/$name.fatbin" "$TF_DIR/kernels/cuda/$src.cu"
  echo "$name: $(cuobjdump -symbols "$FATBIN_OUT/$name.fatbin" 2>/dev/null | grep -c 'tf_ernie55_') symbols"
}

build ernie55_ops   ernie55/ops        "$GLUE"
build ernie55_mxfp4 ernie55/mxfp4_situ "-O3"
build ernie55_dsa   ernie55/dsa_attn   "-O3"
build ernie55_hca   ernie55/hca_attn   "-O3"
