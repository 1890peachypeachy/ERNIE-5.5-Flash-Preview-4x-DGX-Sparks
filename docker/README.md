# docker/

`docker/Dockerfile` adds Baidu's vLLM-ERNIE plugin and this recipe's compatibility patch to a pinned DGX Spark vLLM image. The
build context is the repository root; the root `.dockerignore` sends only `docker/` and `patches/`, so weights, logs and `.git`
stay out. `start.sh` builds it on every node; to build by hand:

```sh
docker build -f docker/Dockerfile \
  --build-arg BASE_IMAGE=eugr/spark-vllm@sha256:afbe9787f5ef6eb3be5747902bf04bdc3515ec34e7fca2659aea34ef8221ea6a \
  --build-arg VLLM_ERNIE_REF=ba322a5634f91d3e6f96ae711df72ddaf3c38b6e \
  -t ernie55-vllm:dev .
docker run --rm --entrypoint python3 ernie55-vllm:dev -c "import vllm, vllm_ernie; print(vllm.__version__)"
```

| Arg | Default | Meaning |
| --- | --- | --- |
| `BASE_IMAGE` | `eugr/spark-vllm@sha256:afbe9787…` | The Spark vLLM build: vLLM `0.34.1rc1.dev37+g3e72879e5`, torch `2.14.0+cu130`, FlashInfer `0.7.5`, transformers `5.23.0`, CUDA 13.0, kernels for `10.3a 12.0a 12.1a` |
| `VLLM_ERNIE_REF` | `ba322a5634f9…` | `baidu/vLLM-ERNIE` commit installed with `--no-deps` |

The live lane was built from `eugr/spark-vllm:latest` on 2026-11-24; the digest above is the one that tag pointed to on the
lane's nodes (read from `RepoDigests`), pinned so a rebuild cannot drift. The pinned base and plugin commit both still resolve.

## What `patches/patch_ernie_sm121.py` changes

The file is the one that was baked into the live image, unchanged. It edits the installed plugin in place and asserts that every
target text is present, so a plugin or vLLM that has moved on fails the build instead of silently skipping a patch.

| # | File | Change |
| --- | --- | --- |
| 1 | `nvidia/mtp.py` | `mtp_validation` moved in vLLM 0.34: import it if present, else treat the check as off |
| 2 | `patches.py` | `SpeculativeConfig.hf_config_override` now takes extra arguments: pass them through |
| 3 | `config.py` | Give the config the `layer_types` and `compress_rates` that vLLM 0.34's DeepSeek-V4 attention reads (derived from `compress_ratios`) |
| 4 | `nvidia/attention.py` | On `sm_12x`, route the HCA layers and the DSA layers to FlashInfer's SM120 sparse-MLA backend, with an exact attention-sink merge across the 2048 + 128 key split (a/b/c/d/e) |
| 5 | `nvidia/model.py` | vLLM 0.34 replaced `use_eagle()`/`uses_draft_model()` with `needs_mtp_hidden_buffer` |
| 6 | `nvidia/attention.py` | `_prepare_and_attn_fn` was renamed `_prepare_and_attn` |
| 7 | `nvidia/attention.py` | `get_kv_cache_spec` became `get_kv_cache_config` (the DSA page spec is re-applied to it) |
| 8 | `b12x/moe/_shared/kernels/situ.py` | The `situ` MoE class forwards every keyword argument to its parent (the W4A8 path passes more than it accepted) |
| 9 | `nvidia/flashindexer.py` | vLLM 0.34 passes extra keyword arguments (such as `logits_dtype`) to the FP8×FP4 MQA-logits call: forward them |
| 10 | `parser.py` | Turn off vLLM's `arg_value_extender` for the ERNIE tool-call parser: it conflicts with the parser's own argument converter and corrupted streamed multi-argument JSON tool calls |
| 11 | `vllm/models/deepseek_v4/compressor.py`, `nvidia/moe.py` | On sm_12x, for M <= 16: the HCA compressor's fused wkv/wgate GEMV and the router GEMV (router weights kept in the checkpoint's bf16 so GateLinear's tier 1 applies) run through vLLM's cuteDSL `ll_bf16_gemm`. bf16 x bf16, fp32 accumulation and output, as before; only the summation order changes. c1 +4.6%, prefill +4.6-4.8% ([docs/hillclimb.md](../docs/hillclimb.md)) |

Numbers 1-3, 5-7 and 9 are renames and signature changes between vLLM 0.32 (what Baidu's card pins) and 0.34. Numbers 4 and 8
are what GB10 needs. Number 10 is a correctness fix for tool calls. Number 11 is a GB10 speed change, measured between controls.

**Not covered by any test in this repo:** the patches are known to apply and to boot; their numerical effect (the sm_121
attention merge in particular) has no isolated check, only the end-to-end replies noted in `docs/validation.md`.
