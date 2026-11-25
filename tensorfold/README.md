# Native TensorFold Backend for ERNIE-5.5-Flash-Preview (4× DGX Spark, TP=4 RoCE)

This directory documents how to build, deploy, and run [`baidu/ERNIE-5.5-Flash-Preview`](https://huggingface.co/baidu/ERNIE-5.5-Flash-Preview) natively on **TensorFold** (Zig 0.17.0 + SM121a CUDA) across four NVIDIA DGX Sparks (GB10, `sm_121a`, 128 GB unified memory each) over lossless 100G RoCE v2.

- **Fork / branch:** [`1890peachypeachy/TensorFold@feat/ernie55`](https://github.com/1890peachypeachy/TensorFold/tree/feat/ernie55)
- **Precision:** Exact checkpoint fidelity — native QAT MXFP4 routed experts + block-128 FP8 projections + BF16 `lm_head` + bundled 3-layer DSpark speculative drafter (no lossy re-quantization)
- **Context & features:** 262,144-token context (CUDA-graphed end-to-end via bucketed sparse graphs), thinking mode, ERNIE parallel multi-arg tool calling, prefix cache

## Verified Quiet-Lane Benchmarks (4× DGX Spark, TP=4, RoCE v2)

Measured on a quiet 4-Spark cluster (`0` external requests, `+0` RoCE `packet_seq_err` across all 4 nodes, `~69.5 GiB` `MemAvailable` per Spark):

| Workload | vLLM Baseline | **Native TensorFold (`feat/ernie55`)** | Delta vs vLLM |
|---|---:|---:|---:|
| **`c1` Prose (`temp=0.7`)** | `41.0 tok/s` | **`45.1 – 47.4 tok/s`** | **+10–16%** |
| **`c1` Code (`temp=0.7`)** | `53.6 tok/s` | **`58.0 – 76.4 tok/s`** | **+8–43%** |
| **`c1` Parallel Tool Calling** | — | **`74.1 – 81.9 tok/s`** | exact JSON |
| **`c2` Code (`temp=0.7`, agg decode)** | `74.7 tok/s` | **`90.4 tok/s`** (`[45.2, 48.8]`) | **+21%** |
| **`c2` Mixed prose+code (agg decode)** | `74.7 tok/s` | **`69.1 tok/s`** (`78.2` simultaneous) | — |
| **`c4` Code (`temp=0.7`, agg decode)** | `114.4 tok/s` | **`121.8 tok/s`** (`[30.4, 36.8, 32.1, 33.7]`) | **+6.5%** |
| **`c4` Mixed 2 prose+2 code (agg decode)** | `114.4 tok/s` | **`94.9 tok/s`** (`114.6` simultaneous) | — |
| **Long-Context Decode (3K / 12K / 30K)** | — | **`42.2 / 39.8 / 40.3 tok/s`** | flat to 262K |
| **Cold Prefill (3K / 12K / 30K)** | `~1,500 / ~1,450` | **`1,128 / 1,187 / 1,146 tok/s`** | 100G RoCE ring |
| **Warm 12K Prefix Cache Hit** | — | **`0.57 s` TTFT** (`12,544` cached tokens) | instant |

## Key SM121a Optimizations Included

1. **SM121a Tensor-Core MMA FlashAttention + Split-K (`hca_attn.cu`, `dsa_attn.cu`):** `mma.sync.aligned.m16n8k16` for HCA (`14 µs/layer`), DSpark HCA, and MLA-DSA (`44 µs/layer`), with zero-padded tail K-tiles and bucketed sparse CUDA graphs (`4K..262K`).
2. **Fused Decode GEMVs, Dedicated LM Head, & Fast mHC (`cuda_weights.zig`, `ops.cu`, `cuda_forward.zig`):** Fused `qdown_kv`, `qup_ogate`, `comp_wkv_wgate`, `qa_kva`, `qb_ogate`, `gate_up`, `fc1_shared_w13`; dedicated `250 GB/s` `tf_ernie55_lm_head_bf16` (4.3× faster than `qmmf` at decode); `213 GB/s` fused BF16 router; 16-lane parallel-warp Sinkhorn; 16-row Tensor-Core `mhc_pre_mma` for prefill; last-token-only prompt logits (`forwardWithTapsLastOnly`).
3. **RoCE Fabric & Sampler (`cuda_forward.zig`, `cuda_sampler.zig`):** BF16 VHA post-mix AllReduce (halves VHA wire bytes on 100G RoCE); single RoCE `allGather` + `unpackVocabBf16` + single-sync `drawBatch`.
4. **Batched Concurrent DSpark (`cuda_dspark.zig`, `cuda_engine.zig`, `cuda_follow.zig`, `cuda_lanes.zig`, `ops.cu`):** Batched `absorbMany`/`draftMany`, 8-entry LRU CUDA graph caches for `verifyMany` and `blockMany`, fused `keep` advances, exact quadratic MultiMax in `tf_ernie55_dspark_markov_multimax_batch`, and calibrated 4× GB10 TP=4 cost tables.

## Build & Run

### 1. Clone and build on rank 0 (Spark)

Requires **Zig 0.17.0** and **CUDA 13.0+ (`nvcc` with `sm_121a`)**.

```sh
git clone -b feat/ernie55 https://github.com/1890peachypeachy/TensorFold.git
cd TensorFold

# Compile SM121a CUDA fatbins
TF_DIR=$PWD/zig FATBIN_OUT=$PWD/fatbins bash /path/to/recipe/tensorfold/build_fatbins.sh

# Build tensorfold-native
zig build native -Dfatbins=$PWD/fatbins -Doptimize=ReleaseSafe
```

### 2. Copy binary to all 4 Sparks and launch

Copy `zig-out/native/bin/tensorfold-native` (and your NCCL `libnccl.so.2` if not in system `LD_LIBRARY_PATH`) to all four Sparks, then run:

```sh
MODEL_DIR=/path/to/ERNIE-5.5-Flash-Preview \
TF_BIN=/path/to/tensorfold-native \
MASTER_IP=192.0.2.1 \
RANK1_HOST=user@192.0.2.2 \
RANK2_HOST=user@192.0.2.3 \
RANK3_HOST=user@192.0.2.4 \
bash tensorfold/start_tensorfold.sh
```

Wait ~5 minutes for the 4-rank TP=4 weight load (`42.16 GiB/rank` including bundled DSpark). Rank 0 serves the OpenAI-compatible API on `http://${BIND_HOST:-127.0.0.1}:${PORT:-8888}/v1`.
