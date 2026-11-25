# Hill-climb, 2026-11-25: where the step goes, what was tried, what was kept

Method: the [inference-optimisation skill](https://github.com/sybil-solutions/inference-optimisation-skill) (fda968e, MIT):
measure the hardware, compute the ceiling, profile, change one thing at a time between controls, keep a ledger.
Bench: the skill's `bench.py` (fixed 6-prompt panel, seed 1234, temperature 0.6, thinking on, 512-token replies; c1 = median
of 6; c4 = wall-clock aggregate; prefill = prompt tokens / TTFT on unique cold prompts). Every candidate ran between two
controls (`C` = this recipe as published at the time), each its own boot, the lane drained and idle otherwise.

**Kept: patch 11 (low-M GEMVs through `ll_bf16_gemm` on GB10). c1 +4.6%, prefill +4.6-4.8%, c4 +2.5% (inside noise).** Everything else was
rejected or could not run. Details below.

## Measured hardware (one GB10, in the lane's own image)

| | per GB10 | x4 |
|---|---:|---:|
| read bandwidth (4 GiB reduction) | **238.6 GB/s** | 954 GB/s |
| copy bandwidth (read + write) | 224.4 GB/s | |
| BF16 GEMM 8192^3 / FP8 GEMM 8192^3 | 95.0 / 196.4 TFLOPS | |

## Ceiling (bytes per token / bandwidth, communication free)

From the checkpoint's own tensor sizes (safetensors headers): routed experts 143.75 GB (6.69 MB each, MXFP4), target dense
7.81 GB, LM head 1.65 GB, DSpark draft 11.03 GB (3 MoE layers: experts 10.27, dense 0.56, Markov head 0.10).

- No speculation: 12.26 GB/token -> 3.07 GB per rank -> **77.8 tok/s** c1.
- DSpark k=3 step (verify of 4 rows, expert union 7.6%, plus a 3-position draft): 23.3 GB -> 5.81 GB per rank -> **24.4 ms**,
  i.e. **100 tok/s at 2.45 accepted tokens per step** (long sampled essays) and 131 tok/s at 3.2 (agentic traffic).
- Measured before this pass: **53.9 ms per step**, 45-46 tok/s c1: about 46% of the ceiling.

## Where the 54 ms goes (torch profiler, rank 0, 80 decode steps at c1)

GPU busy 96%, so the lane is not CPU- or scheduling-bound (async scheduling is already on by default in this vLLM). Wall
time per step, with overlapping graph branches shared between the kernels that overlap:

| Kernel group | ms / step | Note |
|---|---:|---|
| FP8 128x128 block linears (deep_gemm sm120) | 12.2 | ~53% of bandwidth at M=4 |
| routed MoE (b12x, MXFP4 W4A8) | 10.0 | at the bandwidth floor |
| NCCL all-reduce, bf16 `[4,4096]` (139 per step) | 8.6 | ~49 us each: latency, not bytes |
| HCA compressor projection, bf16 `[4096 -> 1024]` (cuBLAS SM80 WMMA kernel) | 7.0 | 155 us per call in the model, 40 us alone |
| MoE router, fp32 `[4096 -> 1024]` (`gemmSN_TN<float>`) | 3.9 | fp32 params, so GateLinear's tier 7 |
| NCCL all-reduce, fp32 `[4,16,512]` (46 per step, attention) | 3.2 | |
| mHC, sparse MLA, quant, other | ~7 | |
| DSpark Markov head `[1,256] x [256,201216]` bf16 (3 per step) | 1.4 | replicated on every rank, at bandwidth |

Why the two GEMV groups are slow on GB10: vLLM's `GateLinear` gates its low-latency router tiers to Hopper and
Blackwell-100, and ERNIE declares fp32 router params, so on sm_121 the router runs as fp32 `F.linear` over a 16 MB weight.
The compressor's `torch.mm(out_dtype=fp32)` gets cuBLAS's 16x16 WMMA kernel (vLLM's own note: "on sm_12x cuBLAS drops
to a 16x16-tiled kernel for narrow N at 2-4 tokens"). A cold-weight microbench (24 rotating copies, so every call reads
DRAM) at this shape and M=4: cuBLAS bf16->fp32 40.9 us, cuBLASLt 44.7 us, fp32 `F.linear` 81.5 us, vLLM's cuteDSL
`ll_bf16_gemm` **35.8 us, the 35.2 us DRAM floor**. At M=8-16 `ll_bf16_gemm` takes its split-K path and is slower than
cuBLAS (56.7 vs 44.8 us).

## Experiments

| Run | One change | c1 | c4 | prefill 5K / 20K | Verdict |
|---|---|---:|---:|---:|---|
| controls (9 boots) | — | 45.01 +- 0.87 (42.99-45.98) | 104.51 +- 1.84 | 1,511 +- 21 / 1,537 +- 4 | the noise band |
| `NCCL_ALGO=Tree` | tree all-reduce | — | — | — | invalid: AllGather has no tree algorithm |
| `NCCL_ALGO=allreduce:tree;...:ring` | tree for all-reduce only | 42.56 | 93.0 | 1,006 / 1,013 | **rejected**: -6% c1, -13% c4, -34% prefill |
| `NCCL_MIN/MAX_NCHANNELS=2` | fewer channels | 44.86 | 104.7 | 1,477 / 1,510 | **rejected**: inside noise |
| `NCCL_PROTO=LL` | low-latency protocol | 43.02 | 103.9 | 1,024 / 1,018 | **rejected**: -4.5% c1, -33% prefill |
| `--linear-backend b12x` | FP8 linears | — | — | — | invalid: no ScaledMM kernel for 128x128 block FP8 |
| `--linear-backend cutlass` | FP8 linears | — | — | — | invalid: deep_gemm scale-layout assertion at load |
| `--linear-backend flashinfer_cutlass` | FP8 linears | — | — | — | invalid: no ScaledMM kernel |
| `--linear-backend triton` | FP8 linears | — | — | — | invalid |
| `TORCH_BLAS_PREFER_CUBLASLT=1` | cuBLASLt | — | — | — | **rejected from the microbench** (slower at M=2-4), no boot spent |
| **patch 11**, run 1 | GEMVs through `ll_bf16_gemm` | **47.17** | 110.4 | **1,571 / 1,606** | beats both neighbouring controls on every metric |
| **patch 11**, run 2 (repeat) | same | **46.42** | 104.5 | **1,593 / 1,613** | beats both on c1 and prefill; c4 flat |
| **patch 11**, clean qualification build | this repo, image `17721a5a950a` | **47.67** | 106.5 | **1,580 / 1,613** | same as the tested build; gates passed |
| **patch 11, all three** | | **47.09 (+4.6%)** | 107.1 (+2.5%) | **1,581 / 1,611 (+4.6% / +4.8%)** | every c1 and prefill run above every control |

So the NCCL settings this recipe already pins (ring, 4 channels, default protocol) are the best of those tried, and
the 128x128 block-FP8 linears run only on deep_gemm on this stack.

## Patch 11, and what "lossless" means on this stack

On sm_12x, for M <= 16: (a) the DeepSeek-V4 compressor's fused wkv/wgate projection and (b) the ERNIE router, now kept in
the checkpoint's bf16 so `GateLinear` can take tier 1, run through `ll_bf16_gemm`. Before and after, the products are bf16 x
bf16 (exact in fp32), accumulated in fp32, written as fp32: only the order of the fp32 sums changes. In prefill, the bf16
router weight lets cuBLAS use bf16 tensor cores where it ran fp32 CUDA-core GEMMs before; that is the prefill gain.

The greedy check could not prove bit-identity, and could not for any configuration: **two boots of the unchanged recipe
produced 0 of 8 identical greedy outputs** (temperature 0, thinking on), diverging between the first and the ~60th token,
at positions whose top-1/top-2 logprob gap was 0.005-0.82 nats. The patched build diverged from a control at the same kind
of positions with gaps of 0.016-0.72 nats: indistinguishable from the lane's own run-to-run variation. Answers checked in
those captures (planets in order, the French translation) were correct in every run that finished. A task-level quality
evaluation was not run.

## Remaining levers, ranked (not done)

| Lever | Estimated gain | Cost |
|---|---|---|
| FP8 block linears at M=4: a skinny block-FP8 GEMV, or tuning deep_gemm's sm120 decode config (12.2 ms at ~53%) | up to ~10% c1 | kernel work |
| Fewer all-reduces per step (186 now; e.g. one reduce for shared + routed experts) | up to ~5-8% c1 | model-code change, numerics review |
| Vocab-shard the DSpark Markov head across the four ranks (1.4 ms -> ~0.4 ms + a gather) | ~2% c1 | plugin change |
| Compressor GEMV at M=16 back on cuBLAS (only the router keeps `ll_bf16_gemm` there) | <1% c4 | trivial, below this harness's noise |
