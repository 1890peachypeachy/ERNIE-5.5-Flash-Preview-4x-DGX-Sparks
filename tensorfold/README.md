# Native TensorFold Backend for ERNIE-5.5-Flash-Preview (4× DGX Spark, TP=4 RoCE)

This directory documents how to build, deploy, and run [`baidu/ERNIE-5.5-Flash-Preview`](https://huggingface.co/baidu/ERNIE-5.5-Flash-Preview) natively on **TensorFold** (Zig 0.17.0 + SM121a CUDA) across four NVIDIA DGX Sparks (GB10, `sm_121a`, 128 GB unified memory each) over lossless 100G RoCE v2.

- **Fork / branch:** [`1890peachypeachy/TensorFold@feat/ernie55`](https://github.com/1890peachypeachy/TensorFold/tree/feat/ernie55) (commit `6ade1fe` = Wave-5 `45b4c7e` + the greedy-in-sampled-rounds correctness fix below)

> **Correctness fix, 2026-11-27 (`6ade1fe`). Builds before it are affected; rebuild.**
>
> - **Symptom:** a greedy request (temperature 0) that decodes while a sampled request (temperature > 0) is also
>   decoding turns into `<|ERNIE_PAD|>` tokens after its first token, then stops.
> - **Cause:** a shared verify that holds a sampled stream drew its greedy streams with a zero-temperature rule
>   (`inv_t = 0`). `sample.cu` ranks candidates by `logit * inv_t`, so the all-zero tie went to token 0, which is
>   PAD.
> - **Fix:** greedy streams in such rounds now take the same F32 argmax as an all-greedy round, so their tokens
>   no longer depend on which streams share the round. `greedyRule()` is fixed (`inv_t = 1`) for grammar-masked
>   draws.
> - **Gate for every build:** 4 greedy + 4 sampled requests at once (spark-bench `probe_concurrency.py`, mode
>   `c8-mix`). Each greedy reply must keep its answer and match its solo reply. All-greedy and all-sampled tests
>   cannot see this bug.
- **Precision:** Exact checkpoint fidelity — native QAT MXFP4 routed experts + block-128 FP8 projections + BF16 `lm_head` + bundled 3-layer DSpark speculative drafter + optional Baidu `FlashIndexer` 8-of-64 prefill head router (no lossy re-quantization)
- **Context & features:** 262,144-token context (CUDA-graphed decode via bucketed sparse graphs), thinking mode, ERNIE parallel multi-arg tool calling, 128-token prefix cache grid, 4096-row sequence-parallel prefill chunks

## Verified Head-to-Head Benchmarks vs vLLM (4× DGX Spark, TP=4, RoCE v2)

Measured on the same 4× DGX Spark cluster (`+0` RoCE `packet_seq_err`, `~46.3–46.7 GiB` `MemAvailable` per Spark after `c8` with `--prompt-cache-gib 6`, exact math `17*23=391`, 66K needle recall = `True`, exact parallel multi-arg JSON tool calling):

| Workload | Live vLLM Baseline | TF Start (`1024`-row) | **Native TensorFold (`45b4c7e`)** | Notes |
|---|---:|---:|---:|---|
| **Cold 9K–12K prefill** | `1,603.8 tok/s` (`7.97 s` @ 12K) | `1,187 tok/s` (`10.76 s`) | **`1,727 tok/s` (`5.21 s` @ 9K, `6.93 s` @ 12K)** | **Beats vLLM (`-1.04 s` @ 12K, +45% vs TF start)** |
| **Cold 66K prefill** | `1,566.4 tok/s` (`43.82 s`) | `1,065 tok/s` (`64.46 s`) | **`1,718.7 tok/s` (`38.125 s`)** | **Beats vLLM by `5.70 s` (`-40.8%` vs TF start!)** |
| **Warm 66K + ~550 tok (tool turn)** | `0.57 s` *(450 tok, `46.4` dec)* | `1.25 s` (`43.0 tok/s` dec) | **`0.556 – 0.584 s`** | **Ties/beats vLLM TTFT (on 24% more tokens!), `needle=True`** |
| **Warm 66K + 3.4K tok (big tool dump)** | `2.58 s` (`1,294 tok/s`, `44.0` dec) | `4.64 s` (`714 tok/s` pf) | **`2.239 s` (`1,527.4 tok/s`)** | **Beats vLLM prefill by +18% (`-0.34 s`), `needle=True`** |
| **`c1` Prose (`temp=0.7`)** | `41.0 tok/s` | `45.1 – 47.4 tok/s` | **`46.4 – 49.7 tok/s`** | **+13–21% vs vLLM** |
| **`c1` Code (`temp=0.7`)** | `53.6 tok/s` | `58.0 – 76.4 tok/s` | **`58.0 – 76.4 tok/s`** | **+8–43% vs vLLM** |
| **`c1` Parallel Tool Calling** | — | `74.1 – 81.9 tok/s` | **`55.3 – 81.9 tok/s`** | exact multi-arg JSON |
| **`c2` Code (agg decode)** | `74.7 tok/s` | `90.4 tok/s` | **`90.4 tok/s`** | **+21% vs vLLM** |
| **`c2` Mixed Code/Tech (agg decode)** | `74.7 tok/s` *(code)* | `50.1 tok/s` *(pre-W5)* | **`74.0 tok/s`** (`44.9 tok/s`/stream, `0.22 s` avg TTFT) | **+47.7% vs pre-W5 (`c2/c1 = 1.39×`)** |
| **`c4` Mixed Code/Tech (agg decode)** | `114.4 tok/s` | `80.0 tok/s` *(pre-W4)* | **`106.3 – 121.8 tok/s`** (`32.2 – 38.3 tok/s`/stream, `0.34 s` TTFT) | **+35% vs pre-W4, `-60%` max TTFT** |
| **`c8` Mixed Code/Tech (agg decode)** | — | `99.8 tok/s` *(pre-W4)* | **`118.7 tok/s`** (`19.0 tok/s`/stream med, `1.14 s` avg TTFT, `45 GiB` free) | **+11% vs Wave-4 via W1 bucketed shared verify** |

## Key SM121a Optimizations Included

### Prefill (SM121a Blackwell GB10)
1. **Native Baidu `FlashIndexer` 8-of-64 Prefill Head Router + Pre-Quantized FP8 `idx_k` Cache (`dsa_attn.cu`, `cuda_weights.zig`, `cuda_kv.zig`, `cuda_forward.zig`):** Auto-loads `<model_dir>/flashindexer/flashindexer.safetensors` (3.1 MB) for all 6 DSA layers (`7, 14, 21, 28, 35, 42`), caches `idx_k` directly as FP8 E4M3 + F32 row scales (**48.4% smaller** `idx_k` KV & prefix snapshots, `256 B → 132 B`/token/layer), and runs `tf_ernie55_dsa_flashindexer_select` + 8-row × 8-head `tf_ernie55_dsa_flashindexer_scores_fp8_8h_mma` (**8.5× faster** vs original 64-head BF16 scorer, `24.27 ms/layer` at 66K, `max_abs_diff = 0`).
2. **Dual-Stream `shared_mlp` `ncclAllReduce` Overlap with `routed_moe` (`cuda_engine.zig`, `cuda_forward.zig`):** For prefill chunks (`rows >= 256`, `world > 1`, never inside decode CUDA graphs), launches `shared_out` `ncclAllReduce` on a dedicated non-blocking `comm_stream` synchronized via CUDA events while `moeGrouped` executes on the main compute stream.
3. **Zero-Spill 3-Stage FP8 Block-128 MMA Prefill GEMM (`ops.cu`, `-O3`):** XOR-swizzled `ldmatrix` + 3-stage `cp.async` + L2 swizzle + `__launch_bounds__(256, 2)` (0 bytes register spill, **107 TFLOPS**, 4.6× faster than earlier spilled build and >8× faster than `qmmf`).
4. **SM121a Native BF16 64×64 MMA Prefill GEMM (`ops.cu`):** **56.5 TFLOPS** (20.7× faster than `qmmf` BF16).
5. **4-Row Prefill HCA FlashAttention (`hca_attn.cu`, `tf_ernie55_hca_attn_prefill_mma`):** 512-thread block sharing each KV tile across 4 consecutive query rows across all 37 HCA layers (**1.99× faster at 12K**, **2.49× faster at 66K**, `0.00000` bit-exact).
6. **Coalesced `BK=256` MXFP4 MoE + Fused Weighted `atomicAdd` `w2_mma` + Parallel-`p0` `grid.z` (`mxfp4_situ.cu`):** Full 128-byte DRAM cache-line loads, parallel token-tile scheduling across hot experts, and fused weighted `atomicAdd` into a `32 MB` F32 accumulator eliminating `28.1 GB/chunk` of scattered `moe_part` traffic (**1.45× faster**, `0.00000` bit-exact).
7. **4096-Row Prefill Chunks & 128-Token Prefix Cache Grid (`cuda_engine.zig`, `cuda_kv.zig`):** `prompt_rows = 4096`, `swa_cap = 8192`, `snap_grid = 128`.

8. **Warm `Seq` Slot Pooling (`slot < 4`) & Single-Chunk `late_mark` Snapshots (`cuda_engine.zig`, `cuda_lanes.zig`, `cuda_follow.zig`, `cuda_native.zig`):** Pools up to 4 sequence slots across requests (eliminating `~150 ms` of `cuMemAlloc` + `4.3 GiB` zero-fill and keeping captured CUDA graphs hot) and merges `<= 4096`-token remaining warm turns into a single chunk with a stashed DSpark draft ring at `late_mark` (eliminating extra `~250–550 ms` tail passes), cutting Warm 66K tool-turn TTFT by **40%** (`0.94 s → 0.556 s`).
9. **W4 Sequence-Parallel Row-Sharded Prefill (`1a00a03`, `cuda_forward.zig`, `nccl.zig`, `dsa_attn.cu`):** Shards residual, mHC (`mhcPreFast`/`mhcPost` 84×), MoE post-combine `rmsNorm`/`fc2_latent`/`addBf16`, HCA prep + `C128` flush, and DSA prep + `FlashIndexer`/TopK across the 4 TP ranks (`ncclReduceScatter` + `ncclAllGather`, with `IDX_SHARD` overlapped on `comm_stream` behind `qb_ogate`/`absorbQ`), pushing Cold 66K prefill to **`38.125 s` (`1,718.7 tok/s`, `-5.70 s` faster than vLLM)**.

### Decode & Serving
1. **SM121a Tensor-Core MMA FlashAttention + Split-K (`hca_attn.cu`, `dsa_attn.cu`):** `mma.sync.aligned.m16n8k16` for HCA (`14 µs/layer`), DSpark HCA, and MLA-DSA (`44 µs/layer`), with zero-padded tail K-tiles and bucketed sparse CUDA graphs (`4K..262K`).
2. **Fused Decode GEMVs, Dedicated LM Head, & Fast mHC (`cuda_weights.zig`, `ops.cu`, `cuda_forward.zig`):** Fused `qdown_kv`, `qup_ogate`, `comp_wkv_wgate`, `qa_kva`, `qb_ogate`, `gate_up`, `fc1_shared_w13`; dedicated `250 GB/s` `tf_ernie55_lm_head_bf16`; `213 GB/s` fused BF16 router; 16-lane parallel-warp Sinkhorn; last-token-only prompt logits (`forwardWithTapsLastOnly`).
3. **RoCE Fabric & Sampler (`cuda_forward.zig`, `cuda_sampler.zig`):** BF16 VHA post-mix AllReduce; `TF_ROCE` dual-HCA RDMA for decode (`< 1 MiB`) + NCCL Ring AllReduce for prefill (`>= 1 MiB`); single-sync `drawBatch`.
4. **W1 Bucketed Shared Verify & Padding Guard (`db1e499`, `1a00a03`, `ops.cu`, `hca_attn.cu`, `dsa_attn.cu`, `cuda_kv.zig`, `cuda_forward.zig`, `cuda_engine.zig`):** Keys shared-verify CUDA graphs solely on `(bucket_rows, any_sparse, older_ceil)` (32-entry LRU cache) using table-driven `_tab` attention/prep/stash kernels, with zero-initialized scratch (`Scratch.alloc`) and finite/clamp guards for `row_seg < 0` padding rows — jumping `c2` **+47.7%** (`50.1 → 74.0 tok/s`) and `c8` **+11%** (`106.8 → 119.0 tok/s`).
5. **Tensor-Core Concurrent MoE/FP8 & `window_rows = 64` (`cuda_forward.zig`, `cuda_engine.zig`, `cuda_lanes.zig`):** `grouped_min_rows = 24` and `fp8GemmB128Mma` at `rows >= 24` put `c4`/`c8` onto CUDA-graphed Tensor Cores; `window_rows = 64` fits all 8 streams × 8 DSpark rows in one shared verify; `pooled_slots = 4` keeps 4 warm slots and their CUDA graphs hot (`+35.4%` `c4` decode, `-60%` `c4` max TTFT).
6. **W5+W2 DSpark Sharding & Cost Env Overrides (`45b4c7e`, `cuda_weights.zig`, `cuda_dspark.zig`, `cuda_lanes.zig`):** K-shards `feature_proj` across ranks (`12288 → 3072/rank`, saving `~80 MiB` VRAM per Spark and 4× smaller absorb reads) and adds `TF_ERNIE55_WINDOW_MS`, `TF_ERNIE55_SHARED_MS`, `TF_ERNIE55_BLOCK_MS`, `TF_ERNIE55_LEARN_ROWS` overrides.
7. **Batched Concurrent DSpark (`cuda_dspark.zig`, `cuda_engine.zig`, `cuda_follow.zig`, `cuda_lanes.zig`, `ops.cu`):** Batched `absorbMany`/`draftMany`, 8-entry LRU CUDA graph caches for `verifyMany` and `blockMany`, fused `keep` advances, exact quadratic MultiMax.

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
