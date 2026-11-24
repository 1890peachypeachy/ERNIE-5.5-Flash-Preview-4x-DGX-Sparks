# How we got here

A short record of where this recipe came from and what it does differently from the model card. Read it before changing
anything, and read the last section before believing it.

## What the model card gives you, and what it does not

Baidu's card ([baidu/ERNIE-5.5-Flash-Preview](https://huggingface.co/baidu/ERNIE-5.5-Flash-Preview)) documents two servers:

| | SGLang | vLLM |
| --- | --- | --- |
| Parallelism | `--tp 4 --attn-dp-size 4 --enable-dp-lm-head` | `--data-parallel-size 4 --enable-expert-parallel` |
| Speculative | `--speculative-algorithm DSPARK` | `{"method":"dspark","num_speculative_tokens":7,...}` |
| Version | `lmsysorg/sglang:latest` | `vllm==0.32.0` + `vLLM-ERNIE` |
| Hardware named | B300 (cookbook link) | not stated |

Neither runs unmodified on four Sparks:

- **The layouts assume NVLink.** Attention data-parallel plus expert parallel moves tokens between GPUs constantly. Four Sparks
  talk over 100 Gb/s-class RoCE links, so the lane uses plain tensor parallel across four hosts with NCCL in ring mode and
  custom all-reduce off. (How the four are cabled is yours to set: `FABRIC_IFACE`, `RDMA_HCAS`, `NCCL_GID_INDEX`.)
- **The plugin targets datacenter Blackwell and Hopper kernels** for the 37 HCA layers and 6 DSA layers. GB10 (`sm_121`) is a
  different part from datacenter Blackwell (community reports put its shared memory near 99 KB per block, and vLLM's own log here
  says its symmetric-memory all-reduce does not support capability 12.1). The patch moves those layers onto FlashInfer's SM120
  sparse-MLA backend, which the plugin did not select on its own.
- **vLLM moved.** The plugin was written against 0.32. The Spark base image carries 0.34.1rc1, in which several internals the plugin
  touches were renamed. The patch file's ten sections are mostly that.

## What is recorded, and what is not

**Recorded** (from the running lane and its build files, 2026-11-24):

- The image build: `FROM eugr/spark-vllm:latest`, `uv pip install --no-deps` of `vLLM-ERNIE@ba322a56`, then the patch script.
- The full command, environment, mounts and capability set of all four containers (the basis of `scripts/lib.sh`).
- The boot log of rank 0: weights 42.02 GiB per rank, 107 s to load, a 16 GiB KV pool of 572,021 tokens, CUDA graphs 1.48 GiB,
  engine initialisation 36 s, API listening about four minutes after `docker run` with warm caches.
- Speculative decoding counters from `/metrics`, and the answers to two small prompts.

**Not recorded.** The bring-up that produced the patch (which crash led to which of the ten edits, in what order, with what dead
ends) was done in another working session and left no log that this repo could read. The numbers `KV_CACHE_BYTES=16 GiB`,
`MAX_SEQS=16`, `SPEC_TOKENS=3`, `BLOCK_SIZE=256` and the CUDA graph sizes were chosen then; whether they are tuned or merely
the first values that booted is **unknown**. The patch comments say what each edit does, not what failed without it.

If you tune this lane, record the failure and the fix here as you go. That is the part of a recipe that cannot be reconstructed
afterwards.

## Decisions that held (visible in the running config)

- **Explicit KV size instead of a utilisation fraction.** `--kv-cache-memory-bytes` makes vLLM skip profiling (its log says the
  utilisation setting is then ignored). On a unified-memory Spark that avoids profiling against a pool other tenants share.
- **`--disable-custom-all-reduce`** and the NCCL set (`ALGO=Ring`, `P2P_DISABLE`, `SHM_DISABLE`, channel count pinned to 4, a fixed GID
  index, subnet-aware routing). vLLM's own log says its symmetric-memory and FlashInfer all-reduce paths are unavailable at capability
  12.1 / world size 4, so collectives are NCCL over RoCE regardless.
- **No restart policy.** A lane that has not been through a long soak should not come back by itself after a crash or a reboot.
- **Per-node image builds** from a pinned base instead of shipping one image: same inputs, same patched-file hashes, different image IDs.

## Open questions

1. How fast is it? There is no benchmark: single stream, concurrency 4/8/16, long prompts.
2. Is `SPEC_TOKENS=3` right? The card recommends up to 7 at low concurrency. Acceptance at three positions is 83% / 60% / 40%, so a
   fourth position would accept under 40% of the time; whether the extra verification cost pays is untested.
3. Does the sm_121 attention merge match the reference kernels numerically? Only end-to-end answers have been looked at.
4. Does context beyond 262,144 work, and what does the KV pool need for it?
5. How does a first start on empty caches behave (time, memory peaks)?
6. Newer `vLLM-ERNIE` commits (the HALOps indexer path) and the FP8/BF16 checkpoints were not tried.
