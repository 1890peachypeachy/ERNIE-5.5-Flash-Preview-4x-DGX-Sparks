# Native TensorFold Backend for ERNIE-5.5-Flash-Preview (4× DGX Spark, TP=4 RoCE)

This directory documents how to build, deploy, and run [`baidu/ERNIE-5.5-Flash-Preview`](https://huggingface.co/baidu/ERNIE-5.5-Flash-Preview) natively on **TensorFold** (Zig 0.17.0 + SM121a CUDA) across four NVIDIA DGX Sparks (GB10, `sm_121a`, 128 GB unified memory each) over lossless 100G RoCE v2.

- **Fork / branch:** [`1890peachypeachy/TensorFold@feat/ernie55`](https://github.com/1890peachypeachy/TensorFold/tree/feat/ernie55) (commit `2f7013d`)
- **Precision:** Exact checkpoint fidelity — native QAT MXFP4 routed experts + block-128 FP8 projections + BF16 `lm_head` + bundled 3-layer DSpark speculative drafter + optional Baidu `FlashIndexer` 8-of-64 prefill head router (no lossy re-quantization)
- **Context & features:** 262,144-token context (CUDA-graphed decode via bucketed sparse graphs), thinking mode, ERNIE parallel multi-arg tool calling, 128-token prefix cache grid, 4096-row prefill chunks

## Verified Head-to-Head Benchmarks vs vLLM (4× DGX Spark, TP=4, RoCE v2)

Measured on the same 4× DGX Spark cluster (`+0` RoCE `packet_seq_err`, `~53–62 GiB` `MemAvailable` per Spark, exact math `17*23=391`, 66K needle recall = `True`, exact parallel multi-arg JSON tool calling):

| Workload | Live vLLM Baseline | TF Start (`1024`-row) | **Native TensorFold (`2f7013d`)** | Notes |
|---|---:|---:|---:|---|
| **Cold 12K prefill** | `1,603.8 tok/s` (`7.97 s`) | `1,187 tok/s` (`10.76 s`) | **`1,472.1 tok/s` (`8.12 s`)** *(Hermes copy `1,528.9 tok/s`)* | **Within `0.15 s` of vLLM (+24% vs TF start)** |
| **Cold 66K prefill** | `1,566.4 tok/s` (`43.82 s`) | `1,065 tok/s` (`64.46 s`) | **`1,474.6 – 1,475.1 tok/s` (`44.36 – 44.48 s`)** | **+38.5% (`-20.1 s`!), flat `0%` 12K→66K, within `0.54 s` of vLLM** |
| **Warm 66K + 450 tok (tool turn)** | `0.57 s` (`46.4 tok/s` dec) | `1.25 s` (`43.0 tok/s` dec) | **`0.94 s` (`47.4 tok/s` dec)** | **Sub-second TTFT, faster decode than vLLM, `needle=True`** |
| **Warm 66K + 3.3K tok (big tool dump)** | `2.58 s` (`44.0 tok/s` dec) | `4.64 s` (`714 tok/s` pf) | **`3.07 s` (`1,079.3 tok/s`, `52.7 tok/s` dec)** | **+51.2% pf, +20% faster decode than vLLM** |
| **`c1` Prose (`temp=0.7`)** | `41.0 tok/s` | `45.1 – 47.4 tok/s` | **`46.4 – 49.7 tok/s`** | **+13–21% vs vLLM** |
| **`c1` Code (`temp=0.7`)** | `53.6 tok/s` | `58.0 – 76.4 tok/s` | **`58.0 – 76.4 tok/s`** | **+8–43% vs vLLM** |
| **`c1` Parallel Tool Calling** | — | `74.1 – 81.9 tok/s` | **`55.3 – 81.9 tok/s`** | exact multi-arg JSON |
| **`c2` Code (agg decode)** | `74.7 tok/s` | `90.4 tok/s` | **`90.4 tok/s`** | **+21% vs vLLM** |
| **`c4` Code (agg decode)** | `114.4 tok/s` | `121.8 tok/s` | **`121.8 tok/s`** | **+6.5% vs vLLM** |

## Key SM121a Optimizations Included

### Prefill (SM121a Blackwell GB10)
1. **Native Baidu `FlashIndexer` 8-of-64 Prefill Head Router + Pre-Quantized FP8 `idx_k` Cache (`dsa_attn.cu`, `cuda_weights.zig`, `cuda_kv.zig`, `cuda_forward.zig`):** Auto-loads `<model_dir>/flashindexer/flashindexer.safetensors` (3.1 MB) for all 6 DSA layers (`7, 14, 21, 28, 35, 42`), caches `idx_k` directly as FP8 E4M3 + F32 row scales (**48.4% smaller** `idx_k` KV & prefix snapshots, `256 B → 132 B`/token/layer), and runs `tf_ernie55_dsa_flashindexer_select` + 8-row × 8-head `tf_ernie55_dsa_flashindexer_scores_fp8_8h_mma` (**8.5× faster** vs original 64-head BF16 scorer, `24.27 ms/layer` at 66K, `max_abs_diff = 0`).
2. **Dual-Stream `shared_mlp` `ncclAllReduce` Overlap with `routed_moe` (`cuda_engine.zig`, `cuda_forward.zig`):** For prefill chunks (`rows >= 256`, `world > 1`, never inside decode CUDA graphs), launches `shared_out` `ncclAllReduce` on a dedicated non-blocking `comm_stream` synchronized via CUDA events while `moeGrouped` executes on the main compute stream.
3. **Zero-Spill 3-Stage FP8 Block-128 MMA Prefill GEMM (`ops.cu`, `-O3`):** XOR-swizzled `ldmatrix` + 3-stage `cp.async` + L2 swizzle + `__launch_bounds__(256, 2)` (0 bytes register spill, **107 TFLOPS**, 4.6× faster than earlier spilled build and >8× faster than `qmmf`).
4. **SM121a Native BF16 64×64 MMA Prefill GEMM (`ops.cu`):** **56.5 TFLOPS** (20.7× faster than `qmmf` BF16).
5. **4-Row Prefill HCA FlashAttention (`hca_attn.cu`, `tf_ernie55_hca_attn_prefill_mma`):** 512-thread block sharing each KV tile across 4 consecutive query rows across all 37 HCA layers (**1.99× faster at 12K**, **2.49× faster at 66K**, `0.00000` bit-exact).
6. **Coalesced `BK=256` MXFP4 MoE + Fused Weighted `atomicAdd` `w2_mma` + Parallel-`p0` `grid.z` (`mxfp4_situ.cu`):** Full 128-byte DRAM cache-line loads, parallel token-tile scheduling across hot experts, and fused weighted `atomicAdd` into a `32 MB` F32 accumulator eliminating `28.1 GB/chunk` of scattered `moe_part` traffic (**1.45× faster**, `0.00000` bit-exact).
7. **4096-Row Prefill Chunks & 128-Token Prefix Cache Grid (`cuda_engine.zig`, `cuda_kv.zig`):** `prompt_rows = 4096`, `swa_cap = 8192`, `snap_grid = 128`.

### Decode & Serving
1. **SM121a Tensor-Core MMA FlashAttention + Split-K (`hca_attn.cu`, `dsa_attn.cu`):** `mma.sync.aligned.m16n8k16` for HCA (`14 µs/layer`), DSpark HCA, and MLA-DSA (`44 µs/layer`), with zero-padded tail K-tiles and bucketed sparse CUDA graphs (`4K..262K`).
2. **Fused Decode GEMVs, Dedicated LM Head, & Fast mHC (`cuda_weights.zig`, `ops.cu`, `cuda_forward.zig`):** Fused `qdown_kv`, `qup_ogate`, `comp_wkv_wgate`, `qa_kva`, `qb_ogate`, `gate_up`, `fc1_shared_w13`; dedicated `250 GB/s` `tf_ernie55_lm_head_bf16`; `213 GB/s` fused BF16 router; 16-lane parallel-warp Sinkhorn; last-token-only prompt logits (`forwardWithTapsLastOnly`).
3. **RoCE Fabric & Sampler (`cuda_forward.zig`, `cuda_sampler.zig`):** BF16 VHA post-mix AllReduce; `TF_ROCE` dual-HCA RDMA for decode (`< 1 MiB`) + NCCL Ring AllReduce for prefill (`>= 1 MiB`); single-sync `drawBatch`.
4. **Batched Concurrent DSpark (`cuda_dspark.zig`, `cuda_engine.zig`, `cuda_follow.zig`, `cuda_lanes.zig`, `ops.cu`):** Batched `absorbMany`/`draftMany`, 8-entry LRU CUDA graph caches for `verifyMany` and `blockMany`, fused `keep` advances, exact quadratic MultiMax.

## Build & Run

### 1. Clone and build on rank 0 (Spark)

Requires **Zig 0.17.0** and **CUDA 13.0+ (`nvcc` with `sm_121a`)**.

```sh
git clone -b feat/ernie55 https://github.com/1890peachypeachy/TensorFold.git
cd TensorFold

# Compile SM121a CUDA fatbins
TF_DIR=$PWD/zig FATBIN_OUT=$PWD/fatbins bash /path/to/recipe/tensorfold/build_fatbins.sh

# Build tensorfold-native
zig build native -Dfatbins=$PWD/fatbins -Doptimize=fast
```

### 2. Copy binary to all 4 Sparks and launch

Ensure `<MODEL_DIR>/flashindexer/flashindexer.safetensors` is present alongside the 128 model shards (auto-detected at startup). Copy `zig-out/native/bin/tensorfold-native` to all four Sparks, then run:

```sh
MODEL_DIR=/path/to/ERNIE-5.5-Flash-Preview \
TF_BIN=/path/to/tensorfold-native \
MASTER_IP=192.0.2.1 \
RANK1_HOST=user@192.0.2.2 \
RANK2_HOST=user@192.0.2.3 \
RANK3_HOST=user@192.0.2.4 \
bash tensorfold/start_tensorfold.sh
```

Rank 0 logs `loaded FlashIndexer 8-of-64 prefill head router (6 DSA layers)` and serves the OpenAI-compatible API on `http://${BIND_HOST:-127.0.0.1}:${PORT:-8888}/v1`.
