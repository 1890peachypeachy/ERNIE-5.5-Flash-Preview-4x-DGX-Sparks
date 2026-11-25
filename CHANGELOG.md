# Changelog

- **2026-11-25 (Native TensorFold backend)** -- Added [`tensorfold/`](tensorfold/README.md) recipe for running ERNIE-5.5-Flash-Preview natively on TensorFold (`1890peachypeachy/TensorFold@feat/ernie55`) across 4× DGX Sparks (TP=4 over lossless 100G RoCE v2). Includes SM121a Tensor-Core MMA HCA/DSA FlashAttention, 250 GB/s BF16 `lm_head`, BF16 VHA AllReduce, batched concurrent DSpark with exact quadratic MultiMax, and bucketed sparse CUDA graphs up to 262K context. Verified quiet-lane results: `c1` 45.1–47.4 tok/s prose / 58.0–76.4 tok/s code, `c2` 90.4 tok/s code, `c4` 121.8 tok/s code, 1,187 tok/s cold 12K prefill, 0.57 s warm 12K cache hit, +0 RoCE `packet_seq_err`.

- **2026-11-25 (first cold start, now verified)** -- `./start.sh` started the lane from stopped nodes: preflight passed on all four, the image built from the pinned base in ~3 s
  (base already local), ranks 3,2,1,0 started, `/health` 200 after about four minutes on warm caches, `READY`, and a chat completion returned the right answer.
  It first stopped in preflight on two bugs of ours, both fixed: it required a registered `nvidia`
  Docker runtime (these nodes expose GPUs through CDI, `--gpus all` still works), and it required 129 safetensors files (the weights are 128
  `model-*-of-00128.safetensors` shards; the 129th file is the optional `flashindexer/flashindexer.safetensors`, unused here).

- **2026-11-24 (tuning sweep)** -- Added `docs/tuning.md`: the operator-reported decode, prefill and long-context results (to 180K tokens) and the step-by-step
  case for `b12x`, DSpark `k=3` and `mnbt=8192`. Labelled as reported, not re-run here. Flagged one open mismatch with the live boot log (KV pool tokens, weight GiB).

- **2026-11-24 (benchmark)** -- Added `scripts/bench/bench.py` and the first throughput numbers: 46.6 tok/s per stream at one stream,
  235 tok/s aggregate at 16, prefill about 1,525 tok/s to a ~24k-token prompt, about 2.5 tokens per speculative step.
  Raw results in `results/`. Made the repository public.

- **2026-11-24** -- First publication. Recipe reconstructed from the running four-Spark lane (`docker inspect` of all four containers,
  the image's build files, the rank-0 boot log) and checked for parity: `DRY_RUN=1 ./start.sh` regenerates every rank's arguments,
  environment and mounts exactly. Weights `baidu/ERNIE-5.5-Flash-Preview@de3ffd3e8d55` verified 142/142 files. vLLM
  `0.34.1rc1.dev37+g3e72879e5` on the pinned `eugr/spark-vllm` digest, vLLM-ERNIE `ba322a56`, ten compatibility patches.
  **Not yet done:** a cold start driven by `start.sh`, a quality measurement, context beyond 262,144.
  `BIND_HOST` defaults to `127.0.0.1`; `SERVED_NAME` defaults to the model name.
