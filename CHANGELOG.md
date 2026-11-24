# Changelog

- **2026-11-24** -- First publication. Recipe reconstructed from the running four-Spark lane (`docker inspect` of all four containers,
  the image's build files, the rank-0 boot log) and checked for parity: `DRY_RUN=1 ./start.sh` regenerates every rank's arguments,
  environment and mounts exactly. Weights `baidu/ERNIE-5.5-Flash-Preview@de3ffd3e8d55` verified 142/142 files. vLLM
  `0.34.1rc1.dev37+g3e72879e5` on the pinned `eugr/spark-vllm` digest, vLLM-ERNIE `ba322a56`, ten compatibility patches.
  **Not yet done:** a cold start driven by `start.sh`, any throughput or quality measurement, context beyond 262,144.
  `BIND_HOST` defaults to `127.0.0.1` here (the live lane binds `0.0.0.0`); `SERVED_NAME` defaults to the model name (the live lane
  keeps an older alias for its clients).
