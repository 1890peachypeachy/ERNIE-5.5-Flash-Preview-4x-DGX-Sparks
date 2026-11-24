# Changelog

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
