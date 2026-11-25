<h1 align="center">ERNIE-5.5-Flash-Preview on four DGX Sparks</h1>

<p align="center"><sub>Native TensorFold (SM121a CUDA + DSpark) &amp; vLLM + vLLM-ERNIE · tensor parallel 4 · QAT MXFP4/FP8 · over the ConnectX-7 RoCE v2 fabric</sub></p>

> [!TIP]
> **New: Native TensorFold backend ([`tensorfold/README.md`](tensorfold/README.md))** — runs `baidu/ERNIE-5.5-Flash-Preview` directly in Zig + SM121a CUDA (`1890peachypeachy/TensorFold@feat/ernie55`) with **45.1–47.4 tok/s** `c1` prose (+10–16% vs vLLM), **58.0–76.4 tok/s** `c1` code (+8–43% vs vLLM), **90.4 tok/s** `c2` code (+21% vs vLLM), **121.8 tok/s** `c4` code (+6.5% vs vLLM), flat **~40 tok/s** sparse-graphed decode out to 30K+, and **0.57 s** warm 12K prefix cache hits. See [`tensorfold/README.md`](tensorfold/README.md) for build & launch instructions.

> [!IMPORTANT]
> **Status: serving, measured for throughput, not for quality.** This recipe describes a lane that is up and answering on four GB10 Sparks (see
> [What is checked](#what-is-checked), including what is **not**). It is a **reconstruction**: the launcher in this repo was
> rebuilt from `docker inspect` of the running containers, then proven to regenerate every rank's command line, environment
> and mounts exactly. It has not been used to cold-start the lane from an empty node, and `start.sh`'s checks, image build
> and wait loop have not run end to end yet. Verify against your own nodes before relying on it. Throughput is measured (see [Performance](#performance)); output
> quality is not. The model is a Baidu *preview* checkpoint (post-training is unfinished).

This recipe serves [`baidu/ERNIE-5.5-Flash-Preview`](https://huggingface.co/baidu/ERNIE-5.5-Flash-Preview) (280B total, 14B
active, 512 routed experts top-10, 43 layers, 1M-token window) on **four NVIDIA DGX Sparks** (GB10, `sm_121`, 128 GB unified
memory each), one tensor-parallel rank per Spark, joined by their ConnectX-7 links. Rank 0 serves the OpenAI-compatible API;
ranks 1-3 run headless.

```sh
cp scripts/local.sh.example scripts/local.sh   # nodes, fabric addresses, weights path (once)
scripts/fetch-weights.sh                        # 142 files, 165.9 GB, checked against the Hub (skip if you have them)
./start.sh                                      # check all four Sparks, build the image on each, start ranks 3,2,1,0, smoke-test
./stop.sh                                       # stop rank 0 then the rest, archive the logs
DRY_RUN=1 ./start.sh                            # print all four `docker run` lines and change nothing (no ssh, no docker)
```

The engine is stock vLLM (`0.34.1rc1.dev37`, commit `3e72879e`, torch `2.14.0+cu130`) from [eugr's Spark build](https://github.com/eugr/spark-vllm-docker)
pinned by digest, plus Baidu's [vLLM-ERNIE](https://github.com/baidu/vLLM-ERNIE) plugin at commit `ba322a56`, plus ten small
compatibility patches (`patches/patch_ernie_sm121.py`) that this vLLM and this GPU need. Baidu's model card installs
`vllm==0.32.0` and targets B300; neither is what runs here.

## Why this stack on a Spark

- **Four nodes, TP=4, not Baidu's layout.** The card's vLLM command is `--data-parallel-size 4 --enable-expert-parallel`
  (attention DP plus EP), written for one box with NVLink. Four Sparks reach each other over RoCE, so this lane runs plain
  tensor parallel across four hosts (`--tensor-parallel-size 4 --nnodes 4`), NCCL ring, custom all-reduce off.
- **The weights fit with room.** The FP4-expert checkpoint is 165.9 GB; each rank holds 42.02 GiB, leaving the rest of
  its 128 GB for a 16 GiB KV pool, CUDA graphs, the OS and (on a shared box) other tenants. The two attention types (37 HCA
  layers that keep one entry per 128 tokens, 6 sparse DSA layers) keep the KV cache small: the 16 GiB pool holds 572,021 tokens.
- **sm_121 needs the attention path rerouted.** The plugin picks Hopper/Blackwell-datacenter FlashMLA kernels; on
  `sm_12x` the patch routes the HCA and DSA layers to vLLM's FlashInfer SM120 sparse-MLA backend, with an exact attention-sink
  merge over the 2048 + 128 key split. The rest of the patch file adapts the plugin to renamed vLLM 0.34 internals
  ([docker/README.md](docker/README.md) lists all ten).
- **b12x for the MoE.** `--moe-backend b12x` runs the W4A8 experts. The kernels ship inside the base image's FlashInfer
  (`flashinfer/experimental/b12x`); one patch (no. 8) makes its `situ` activation class accept the extra keyword arguments this model's
  W4A8 path passes.

## What is checked

Checked on 2026-11-24 against the live lane, by reading it back, not by trusting this repo's own scripts:

- **Parity.** With a `local.sh` mirroring the live topology, `DRY_RUN=1 ./start.sh` generates, for each of the four ranks, an
  argument list, an environment (the 42 variables the launcher sets) and a set of mounts that are byte-identical to
  `docker inspect` of the corresponding running container. Entrypoint (`vllm`), `--gpus all`, `--network host`, `--ipc host`,
  `IPC_LOCK`, `/dev/infiniband`, the three ulimits and `label=disable` are included. Restart policy is `no`.
- **Same software on all four nodes.** Base image ID, `vllm 0.34.1rc1.dev37+g3e72879e5.d20261124.cu130`, `vllm-ernie 0.1.0`,
  `flashinfer-python 0.7.5` and the md5 of the two patched plugin files agree on every node. The image *IDs* differ node to
  node, because each node built its own copy from the same pinned base; that is expected.
- **Weights.** 142 of 142 files match the Hub at `de3ffd3e8d55` by size and sha256 (checked once when downloaded; the size-only
  pass was repeated). `scripts/verify-weights.py` is the checker.
- **Serving.** `/health` 200, `/v1/models` lists the served name with `max_model_len` 262144, tool and reasoning parsers loaded,
  prefix caching on, and a plain arithmetic prompt returned the right answer (`17*23` → `391`, thinking on, 22 tokens).
  The benchmark below ran 64 streamed requests (1 to 16 at a time) and two long prompts with no errors.
- **Speculative decoding is doing work.** See the table below: about 2.5 tokens per verify step on long thinking-heavy
  essays, and a higher 61% acceptance (2.8 tokens per step) over the lane's earlier mixed short traffic.
- **Offline suite.** `scripts/tests/check.sh` (syntax, shellcheck, the DRY_RUN shape, and an exposure scan) passes.

**Not checked, so do not read it as verified:**

- A cold start from this repo: `start.sh`'s preflight, remote image build, worker launch order and wait loop are written from
  the same inspection but have never driven a real start. The first run on a fresh set of nodes is the real test.
- **Throughput beyond what is in the table.** One run per concurrency level, one prompt family (long essays), prompts up to
  23,913 tokens, no agent-loop trace, no repeat runs for variance.
- **Quality.** No eval has been run on this deployment. Baidu's benchmark table is Baidu's own.
- **Long context.** The model supports 1,048,576 tokens. This lane is configured for 262,144; `scripts/bench/bench.py` ran to a ~24k-token prompt, and the reported tuning sweep reached 180K (needle retrieved, [docs/tuning.md](docs/tuning.md)). Nothing here covers 262K or beyond.
- **`SPEC_TOKENS` re-check.** The model card suggests up to 7 draft tokens at low concurrency; the reported tuning sweep found 3 best across its workloads. This repository has not re-run that comparison.
- **FP8/BF16 repositories**, the SGLang path, and the plugin commits newer than `ba322a56`.
- First-boot time on empty caches. A start with warm caches took about four minutes from `docker run` to the API listening
  (weights load 107 s, KV and warm-up 36 s).

## Performance

Measured 2026-11-24 on the lane described here (four GB10 Sparks, TP=4, `SPEC_TOKENS=3`, 262,144 context, the settings in the table
below), with `scripts/bench/bench.py` (standard library only; raw output in
[`results/2026-11-24-tp4-c1-16.json`](results/2026-11-24-tp4-c1-16.json)). **Thinking is always on**, so every request is a
long thinking trace followed by an answer; the 768-token cap was reached by all 64 decode requests. Sampling was temperature 0.6, top-p 0.95.
The lane had no other requests running at the start. Prompts are unique per request, so the prefix cache did not help.

**Decode, one prompt family (explain a topic in depth), 768 new tokens per request**

| Concurrent streams | Requests | Decode tok/s per stream, median (min-max) | **Aggregate tok/s** | TTFT median (s) | Tokens per verify step | Draft accept rate (by position, %) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 4 | 46.6 (41.8-48.5) | **45.2** | 0.21 | 2.45 | 48% (75 / 45 / 25) |
| 2 | 4 | 39.1 (36.5-39.6) | **74.7** | 0.27 | 2.48 | 49% (77 / 47 / 24) |
| 4 | 8 | 30.1 (27.2-32.2) | **114.4** | 0.31 | 2.44 | 48% (75 / 45 / 24) |
| 8 | 16 | 21.2 (19.0-26.7) | **162.0** | 0.45 | 2.47 | 49% (76 / 46 / 24) |
| 16 | 32 | 15.5 (14.0-17.3) | **235.2** | 0.68 | 2.46 | 49% (76 / 46 / 24) |

**Prefill, one request at a time, 128 new tokens (the answer finished earlier)**

| Prompt tokens | TTFT (s) | Prefill tok/s (upper bound) | Decode tok/s after |
| ---: | ---: | ---: | ---: |
| 6,005 | 3.93 | 1,526 | 49.7 |
| 23,913 | 15.69 | 1,524 | 50.1 |

How to read it:

- **Per stream, it is fast for a 280B model, and it drops as streams are added.** One stream decodes at about 47 tok/s; sixteen
  streams share the lane at about 235 tok/s in total, 5.2 times one stream, with each stream slowing to about 15 tok/s. Aggregate
  throughput was still rising at 16 streams; where it flattens was not tested.
- **Speculative decoding earns about 2.5 tokens per step** at these settings: 75% of first draft tokens are accepted, 45% of
  second, 25% of third. Acceptance was higher (61% overall) on the lane's earlier short, mixed prompts, so it depends on the text.
  The tuning sweep compared k = 2, 3, 4 and 7 and kept 3 ([docs/tuning.md](docs/tuning.md)); that comparison is reported, not re-run here.
- **Prefill is about 1,525 tok/s**, flat from 6k to 24k tokens, so a 24k-token prompt waits roughly 16 s before the first
  token. The prefill figure is an upper bound on speed, because the time to first token also contains the first decode step.
- **Not a quality score.** These numbers say how fast tokens come out, not whether they are good tokens.

A second, larger set from the tuning session (prose, code, JSON, deterministic and live agentic runs, concurrency 2 and 4, prefill to 88K
and a 180K needle test, plus how `b12x`, `k=3` and `mnbt=8192` were chosen over the baseline) is in
[docs/tuning.md](docs/tuning.md). Its headline: **+44% to +63% single-stream decode over the first working config, +129% at four
streams, 1.3-1.5K tok/s prefill out to 180K tokens.** Those figures were supplied by the lane's operator and were not re-run here, so they are labelled as reported.

Reproduce: `python3 -I scripts/bench/bench.py --url http://127.0.0.1:$PORT --model <served name> --out results/run.json`
(about 8 minutes). Compare lanes only with the same script, the same prompts and the same sampling.

## Requirements

- **Four DGX Sparks** (GB10, 128 GB, CUDA 13 driver, Linux aarch64) with their ConnectX-7 ports cabled on one fabric and an IPv4
  on each, all four on one subnet. This lane runs NCCL as a ring (`NCCL_ALGO=Ring`) over the two RDMA devices of the cabled port;
  set `FABRIC_IFACE`, `RDMA_HCAS` and `NCCL_GID_INDEX` for your cabling (see [Settings](#settings)).
- Key-based ssh from rank 0 to the other three; your login in the `docker` group on all four; Docker with the NVIDIA Container
  Toolkit and BuildKit; `curl`, `python3`, `tar`, `ss`, `ip`, `nvidia-smi`; `hf` for the download.
- The weights at the **same path on every node**: a local copy on each (165.9 GB each), or one copy exported read-only over NFS
  from the fabric. Spark with a smaller disk should take the NFS route.
- Disk: about 29 GB for the image on each node (the base is most of it), plus `MIN_DISK_GIB` (60) free under `CACHE_DIR`.
- Network egress on the first build only (it pulls the pinned base image and clones the plugin commit).

## How it starts

1. **Topology check** (`check_topology`): exactly four nodes, rank 0 is `local`, four IPv4 fabric addresses.
2. **Preflight on every node**: aarch64, Docker with GPU access (an `nvidia` runtime or CDI `nvidia.com/gpu` devices), a GB10 visible, `/dev/infiniband`, the node's fabric
   address on the fabric netdev, the weights (config, index, all 128 `model-*-of-00128.safetensors` shards; the optional `flashindexer/` file is not needed), a writable cache with free space, no leftover
   container. Then rank 0's API port must be free.
3. **Image** `ernie55-vllm:<hash>` on each node, built from `docker/Dockerfile` and the patch (the hash covers both and both pins).
   A node that already has the tag is skipped; the others build locally from a copy of `docker/` and `patches/` sent over ssh.
4. **Launch**: ranks 3, 2, 1 first (`--headless`, they wait for rank 0), then rank 0. Each is one container
   (`--network host --ipc host --gpus all --device /dev/infiniband`) running `vllm serve`; none has a restart policy.
5. **Ready**: `/health`, then `/v1/models` must list the served name, then a one-word completion must return a usage block.
   If any rank's container exits while waiting, its last log lines are printed and the start stops.

Rank 0's `--master-addr` is its own fabric address; every rank sets `VLLM_HOST_IP` to its own.

## Settings

Defaults live in `scripts/config.sh`; override in `scripts/local.sh` (which wins) or the environment.

| Setting | Default | Meaning |
| --- | --- | --- |
| `NODES`, `FABRIC_IPS` | none | Required. Rank-ordered ssh targets (`local` first) and each node's fabric IPv4 |
| `MODEL_DIR` | `$HOME/models/ERNIE-5.5-Flash-Preview` | Weights, same path everywhere, mounted read-only at `/model` |
| `MODEL_REV` | `de3ffd3e…` | Hub commit the weights were verified on |
| `CACHE_DIR` | `$HOME/.cache/ernie55-tp4` | Per-node compile/JIT caches (inductor, triton, FlashInfer, tilelang, b12x), mounted at `/cache` |
| `SERVED_NAME` | `ERNIE-5.5-Flash-Preview` | Model name clients send. See the note below |
| `BIND_HOST` | `127.0.0.1` | API bind address. **The API has no authentication.** Use `0.0.0.0` only on a network you trust |
| `PORT` | `8888` | API port on rank 0 |
| `CONTEXT` | `262144` | `--max-model-len`. The model supports 1,048,576; only 262,144 was run |
| `MAX_SEQS` | `16` | `--max-num-seqs` |
| `MAX_BATCHED_TOKENS` | `8192` | `--max-num-batched-tokens` |
| `KV_CACHE_BYTES` | `17179869184` | 16 GiB of KV per rank, as an explicit size. vLLM then skips memory profiling and **ignores** `GPU_UTIL` |
| `GPU_UTIL` | `0.80` | Passed, but inert while `KV_CACHE_BYTES` is set (the engine says so in its log) |
| `BLOCK_SIZE` | `256` | KV block size |
| `SPEC_TOKENS` | `3` | DSpark draft tokens per step (the card suggests up to 7 at low concurrency; untested) |
| `CUDAGRAPH_SIZES` | `1 2 4 8 16 24 32 48 64` | CUDA graph capture sizes (vLLM 0.34 warns the flag is deprecated; it still applies) |
| `MOE_BACKEND` | `b12x` | W4A8 MoE kernels |
| `FABRIC_IFACE`, `RDMA_HCAS`, `NCCL_GID_INDEX`, `FABRIC_PREFIX_LEN` | `enp1s0f1np1`, `rocep1s0f1,roceP2p1s0f1`, `3`, `24` | The fabric as this lane was cabled. Yours may differ: check `ip -br addr`, `ibdev2netdev`, `show_gids` |
| `WAIT_TIMEOUT` | `3600` | Seconds `start.sh` waits for `/health`; also passed as `VLLM_ENGINE_READY_TIMEOUT_S` |
| `BASE_IMAGE`, `VLLM_ERNIE_REF` | pinned (see `scripts/config.sh`) | Base image digest and plugin commit |

**Pinning the served name.** The default is the model's own name. If your clients already send a fixed model name and must not
be reconfigured when you swap lanes, set `SERVED_NAME` to that name instead; remember then that `/v1/models` no longer tells you which
model is answering, so check the container (`docker ps`, `docker inspect`) or the log.

The reply puts the model's thinking in **`message.reasoning`** on this vLLM build, not `reasoning_content` as the model card
shows. Thinking is always on: budget `max_tokens` for it (the 22-token arithmetic answer used 19 for thinking).

## Memory on a 128 GB unified Spark

From the live rank-0 log (the tuning sweep reports different figures, 41.74 GiB and 1.49M tokens: see [docs/tuning.md](docs/tuning.md)): initial free memory 113.39 GiB; model loading took 42.02 GiB; a 16.0 GiB KV pool was reserved by
size; CUDA graphs took 1.48 GiB; the pool holds **572,021 tokens** (2.18 full 262,144-token requests at once). `MAX_SEQS=16`
and `MAX_BATCHED_TOKENS=8192` bound the rest. The Spark's CPU and GPU share one pool, so anything else on the box (a ComfyUI
tenant, a page cache that has not been dropped) comes out of the same 128 GB; leave headroom before raising `KV_CACHE_BYTES`
or `CONTEXT`.

## Logs and stopping

`./stop.sh` finds each node's container by label (`ernie55.recipe=ernie55-tp4`), saves its log to `logs/current/rankN.log`,
stops rank 0 first, then the others (30 s grace), removes the containers, and archives the logs as
`logs/ernie55-<time>.tar.gz`. It never touches a container without that label.

## Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| `[E_CONFIG]` | `NODES`/`FABRIC_IPS` not four entries, rank 0 not `local`, or a bad address or interface name: see `scripts/local.sh.example` |
| `[E_NODE] rankN: …` | The message names the failed check: no ssh, wrong arch, no GPU access in Docker, no GB10, no `/dev/infiniband`, the fabric IP not on `FABRIC_IFACE`, weights missing or fewer than 128 shards, cache not writable or too full, or a leftover `ernie55-tp4` container (`./stop.sh`) |
| `[E_PORT]` | `PORT` is already listening on rank 0 |
| `[E_LAUNCH]`, `[E_RANK]` | `docker run` failed, or a rank's container exited while the API was coming up; the last 40 log lines are printed |
| `[E_TIMEOUT]` | No `/health` within `WAIT_TIMEOUT`. The first start compiles kernels into `CACHE_DIR`; raise it. `docker logs ernie55-tp4` on each node |
| `[E_MODEL]`, `[E_SMOKE]` | The API is up but the served name or a completion is wrong; query `/v1/models` and the log |
| `Module torchcodec.decoders was found but failed to import` + a traceback at start-up | Harmless: vLLM probes an optional video-decode module the text-only model never uses. Seen on every start of the live lane |
| `SymmMemCommunicator: Device capability 12.1 not supported` and `FlashInfer All Reduce is disabled … world_size=4` | Expected on GB10 at TP=4; collectives go over NCCL |
| `cudagraph_capture_sizes … are deprecated` | A vLLM 0.34 notice; the sizes still apply |
| Image IDs differ between nodes | Expected: each node builds its own. Compare `pip list` and the patched files' md5, not image IDs |
| `reasoning_content` is empty or missing | This build names the field `reasoning` |
| Rank 0 waits forever, workers idle | A rank cannot reach rank 0 on the fabric: check `FABRIC_IPS`, `FABRIC_IFACE`, MTU and that all four share one subnet; `NCCL_DEBUG=INFO` (edit `lib.sh`) for detail |
| Out of memory on one rank | Something else holds unified memory there, or `KV_CACHE_BYTES`/`CONTEXT` is too large for the room; `nvidia-smi` is not informative on GB10, use `free -g` and `/proc/meminfo` |

## Credits

The model, the DSpark draft head and the vLLM-ERNIE plugin are Baidu's. The serving engine is the vLLM project's, by way of
[eugr's DGX Spark build](https://github.com/eugr/spark-vllm-docker). This repository's layout, launcher conventions and
README structure are modeled on [Mia's AI Lab](https://github.com/MiaAI-Lab)'s DGX Spark recipes. Full credits and licence
notes: [CREDITS.md](CREDITS.md), [NOTICE.md](NOTICE.md). How the lane differs from the model card and what is still open:
[docs/HOW-WE-GOT-HERE.md](docs/HOW-WE-GOT-HERE.md). Details of what was and was not tested: [docs/validation.md](docs/validation.md).

## License

Apache License 2.0: see [LICENSE](LICENSE). The weights are Apache-2.0 (Baidu); the base image, plugin and engine keep their own terms.
