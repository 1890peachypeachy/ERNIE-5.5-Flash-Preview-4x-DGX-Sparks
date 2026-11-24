# Tuning sweep (reported)

Numbers from the tuning session that produced this lane's settings. **Provenance:** supplied by the operator of the lane from their own
runs, not produced by `scripts/bench/bench.py` and not re-run by this repository; the workloads (prose, code, structured JSON,
needle-in-a-haystack, a live agentic coding session) are theirs. Treat them as one operator's measurements. The
independent, reproducible numbers are in the README's Performance section and `results/`. The two sets use different prompts, so
they are not expected to match (for example, one stream is 40.9 tok/s on prose here and 46.6 tok/s on long essays there).

## Test setup

- **Model:** `baidu/ERNIE-5.5-Flash-Preview` (280B total / 14B active; official MXFP4 routed experts, blockwise FP8 dense and attention, FP8 KV, bundled 3-layer DSpark draft head; 154.5 GiB on disk).
- **Hardware:** 4 DGX Spark (GB10 `sm_121a`, 121.7 GiB unified memory each), measured memory bandwidth 233.14 GB/s per node, 932.56 GB/s aggregate, TP=4 over a lossless 100G RoCE v2 switch (`NCCL_IB_TC=106`, `NCCL_ALGO=Ring`, 4 channels, no `packet_seq_err` during the runs).
- **Stack:** `eugr/spark-vllm` (vLLM 0.34.1rc1, CUDA 13.0, FlashInfer 0.7.5, `b12x`, TileLang 0.1.12) + `baidu/vLLM-ERNIE` + the SM121 FlashInfer HCA/DSA and `b12x` `SITU` patches in `patches/`.
- **Winning config:** `--tensor-parallel-size 4`, `--moe-backend b12x`, `--kv-cache-dtype fp8`, `--max-model-len 262144`, `--kv-cache-memory-bytes 17179869184`, `--max-num-batched-tokens 8192`, DSpark with `num_speculative_tokens: 3`.
- **Memory:** 41.74 GiB of weights per Spark (target plus draft head); 46.0 to 49.0 GiB of unified memory still free per Spark after a 180K-token prompt.

> **Reconcile before relying on the KV figure.** The reported run says the 16 GiB pool holds 1.49M tokens (5.67x a full 256K request) and weights took 41.74 GiB. The lane's own boot log, read on 2026-11-24, says
> 572,021 tokens (2.18x) and 42.02 GiB. These are different runs or different builds; which one a fresh `start.sh` reproduces is not yet checked.

## Decode

| Workload | Baseline (`DeepGemmFP4`, k=7, `mnbt=4096`) | Optimized (`b12x`, k=3, `mnbt=8192`) | Change | Notes |
| --- | ---: | ---: | ---: | --- |
| c1 prose (median) | 25.1 tok/s (23.7-31.8) | **40.9 tok/s** (37.9-44.2) | +63% | 6 prompts x 256 tok; median TTFT 0.204 s |
| c1 code (median) | 37.2 tok/s (28.9-45.9) | **53.6 tok/s** (37.8-56.6) | +44% | 5 runs x 256 tok; median TTFT 0.221 s |
| c1 structured / JSON (median) | 48.4 tok/s (37.7-58.1) | **58.4 tok/s** (48.5-61.9) | +21% | 6 prompts x 256 tok; median TTFT 0.224 s |
| c1 deterministic (temperature 0, 3 prompts) | not run | **42.3 tok/s** (39.8-46.4 end to end) | n/a | 256 tok each; 0 suspect characters |
| c1 live agentic coding session | not run | **52-57 tok/s** | n/a | 71-75% DSpark acceptance (3.14-3.25 tokens per step), 85% prefix-cache hit |
| c2 concurrent (aggregate wall) | 36.6 tok/s (19.4 per stream) | **64.5 tok/s** (35.0 per stream) | +76% | 2 x 256 tok in parallel; max TTFT 0.251 s |
| c4 concurrent (aggregate wall) | 45.5 tok/s (13.0 per stream) | **104.2 tok/s** (28.1 per stream) | +129% | 4 x 256 tok in parallel; max TTFT 2.33 s down to 0.50 s |

For scale: with no speculative decoding, plain TP=4 decode is bounded by memory bandwidth at about 76.0 tok/s (12.27 GB of active weights read per token at 932.56 GB/s aggregate). That is a ceiling estimate, not a measurement.

## Prefill and long context (cold, salted prompts, 0 cached tokens)

| Context | Baseline (`DeepGemmFP4`, `mnbt=4096`) | Optimized (`b12x`, `mnbt=8192`) | Wall / TTFT | Notes |
| --- | ---: | ---: | ---: | --- |
| ~11K tokens (10,803) | 1,397 tok/s (7.81 s) | **1,527 tok/s** | 7.07 s | cold prefill, +9.3% |
| ~44K tokens (43,717) | 1,391 tok/s (31.44 s) | **1,510 tok/s** | 28.96 s | cold prefill, +8.6% |
| ~88K tokens (88,959) | 1,327 tok/s (65.27 s) | **1,452 tok/s** | 61.26 s | cold prefill, +9.4%; nearly flat from 11K to 88K |
| ~180K needle in a haystack (179,889) | not run | **1,303 tok/s** effective | 138.07 s | exact passkey retrieved (106 completion tokens including thinking); 46.0 GiB minimum free memory |
| warm ~14K multi-turn (13,848 then 13,898) | 10.74 s then 0.378 s | **8.96 s then 0.212 s** | 0.212 s | turn-2 TTFT down 44% with prefix caching |

## How the settings were chosen

Each row adds one change to the row above it. "Keep" means it was kept.

| Step | Change | c1 prose | c1 code | c1 struct | c2 / c4 aggregate | Cold prefill 11K / 44K / 88K | Verdict |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Baseline | `DeepGemmFP4`, DSpark k=7, `mnbt=4096`, Ring | 25.1 | 37.2 | 48.4 | 36.6 / 45.5 | 1397 / 1391 / 1327 | baseline |
| + `b12x` MoE | `--moe-backend b12x` (native SM12x MXFP4 x MXFP8 fused `SITU`) | 36.4 | 56.2 | 62.9 | 57.3 / 64.3 | 1500 / 1486 / 1405 | keep |
| + DSpark k=3 | `num_speculative_tokens: 3` (best of k = 2, 3, 4, 7) | 42.2 | 47.4 | 59.5 | 66.4 / 105.1 | 1499 / 1496 / 1443 | keep |
| + `mnbt=8192` | `--max-num-batched-tokens 8192` | 40.9 | 53.6 | 58.4 | 64.5 / 104.2 | 1527 / 1510 / 1452 | keep |

Reading it: `b12x` is the big win (it also lifts single-stream code and structured output well above the baseline). Moving from k=7 to k=3 costs
some single-stream code speed (56.2 to 47.4) but roughly doubles four-stream throughput (64.3 to 105.1), which is why k=3 is the default here. Raising `mnbt` to
8192 gives a small prefill gain (about 2%) with decode roughly unchanged.
