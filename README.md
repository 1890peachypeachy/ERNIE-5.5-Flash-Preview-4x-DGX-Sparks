<h1 align="center">ERNIE-5.5-Flash-Preview on four DGX Sparks</h1>

<p align="center"><sub>vLLM + Baidu's vLLM-ERNIE plugin · tensor parallel 4 · W4A8 experts · DSpark speculative decoding · over the ConnectX-7 fabric</sub></p>

> [!IMPORTANT]
> **Status: serving, thinly measured.** This recipe describes a lane that is up and answering on four GB10 Sparks (see
> [What is checked](#what-is-checked), including what is **not**). It is a **reconstruction**: the launcher in this repo was
> rebuilt from `docker inspect` of the running containers, then proven to regenerate every rank's command line, environment
> and mounts exactly. It has not been used to cold-start the lane from an empty node, and `start.sh`'s checks, image build
> and wait loop have not run end to end yet. Verify against your own nodes before relying on it. No throughput benchmark
> exists yet. The model is a Baidu *preview* checkpoint (post-training is unfinished).

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
  A 600-token code request returned `200` and stopped at `finish_reason: length` mid-answer, after 549 of its 600 tokens went on thinking;
  that is the cap, not a fault.
- **Speculative decoding is doing work.** From the lane's own `/metrics` since its start (mixed traffic, a few thousand drafts):
  per-position acceptance 82.9% / 60.2% / 40.2% for the three draft tokens, 61.1% of drafted tokens accepted overall,
  i.e. about 2.8 tokens emitted per verify step.
- **Offline suite.** `scripts/tests/check.sh` (syntax, shellcheck, the DRY_RUN shape, and an exposure scan) passes.

**Not checked, so do not read it as verified:**

- A cold start from this repo: `start.sh`'s preflight, remote image build, worker launch order and wait loop are written from
  the same inspection but have never driven a real start. The first run on a fresh set of nodes is the real test.
- **Throughput.** There is no benchmark, no concurrency sweep and no long-context run. The one 600-token request took 27.6 s
  end to end (about 22 tok/s counting 549 thinking tokens), measured once on a lane that may have been serving other traffic.
  Treat that as a sanity figure, not a number.
- **Quality.** No eval has been run on this deployment. Baidu's benchmark table is Baidu's own.
- **Context beyond 262,144.** The model supports 1,048,576. This lane was configured and observed at 262,144 only.
- **`SPEC_TOKENS` above 3.** The model card suggests up to 7 draft tokens at low concurrency; this lane runs 3.
- **FP8/BF16 repositories**, the SGLang path, and the plugin commits newer than `ba322a56`.
- First-boot time on empty caches. A start with warm caches took about four minutes from `docker run` to the API listening
  (weights load 107 s, KV and warm-up 36 s).

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
2. **Preflight on every node**: aarch64, Docker with the NVIDIA runtime, a GB10 visible, `/dev/infiniband`, the node's fabric
   address on the fabric netdev, the weights (config, index, at least 129 shards), a writable cache with free space, no leftover
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

**The served name is a contract.** Clients that must not change configuration when a lane is swapped can pin `SERVED_NAME` to
the name they already use; the live lane does exactly that, so the name in `/v1/models` says nothing about which model
answers. Check the container (`docker ps`, `docker inspect`) or the log, not the alias.

The reply puts the model's thinking in **`message.reasoning`** on this vLLM build, not `reasoning_content` as the model card
shows. Thinking is always on: budget `max_tokens` for it (the 22-token arithmetic answer used 19 for thinking).

## Memory on a 128 GB unified Spark

From the live rank-0 log: initial free memory 113.39 GiB; model loading took 42.02 GiB; a 16.0 GiB KV pool was reserved by
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
| `[E_NODE] rankN: …` | The message names the failed check: no ssh, wrong arch, no NVIDIA runtime, no GB10, no `/dev/infiniband`, the fabric IP not on `FABRIC_IFACE`, weights missing or fewer than 129 shards, cache not writable or too full, or a leftover `ernie55-tp4` container (`./stop.sh`) |
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
