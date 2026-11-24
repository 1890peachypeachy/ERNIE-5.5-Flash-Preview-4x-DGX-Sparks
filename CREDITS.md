# Credits

This recipe is a thin layer over other people's work. Licence terms and required notices are in [NOTICE.md](NOTICE.md); this
repository's own licence is in [LICENSE](LICENSE).

## The model

**[ERNIE-5.5-Flash-Preview](https://huggingface.co/baidu/ERNIE-5.5-Flash-Preview)** by the Baidu ERNIE team: the weights, the
W4A8 quantisation-aware training, the DSpark draft head and the tokenizer (Apache-2.0 for the Preview weights; the Base model has its
own licence). This repository does not contain weights.

## The serving stack

- **[vLLM](https://github.com/vllm-project/vllm)** and its contributors: the engine, OpenAI-compatible server, speculative decoding,
  multi-node tensor parallel.
- **[baidu/vLLM-ERNIE](https://github.com/baidu/vLLM-ERNIE)** (Apache-2.0): the plugin that teaches vLLM the ERNIE 5.5 architecture,
  reasoning parser and tool-call parser. The recipe installs it at a pinned commit and patches the installed copy at image-build
  time (`patches/patch_ernie_sm121.py`); the plugin's source is not vendored here.
- **[eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker)**: the DGX Spark vLLM image this recipe builds on, with
  Spark-targeted torch, FlashInfer and CUDA 13 builds.
- **[FlashInfer](https://github.com/flashinfer-ai/flashinfer)**: the SM120 sparse-MLA attention backend and the `b12x` MoE kernels the
  patch routes to.
- **NVIDIA** NCCL and the Container Toolkit; the DGX Spark platform.

## The recipe's shape

The layout of this repository (README sections, `start.sh` / `stop.sh` with `DRY_RUN`, `scripts/`, `docker/`, `patches/`,
`CHANGELOG.md`, `CREDITS.md`, `NOTICE.md`) and several launcher conventions follow
**[Mia's AI Lab](https://github.com/MiaAI-Lab)**'s DGX Spark recipes, in particular the 4-Spark TP=4 GLM-5.3-Flash recipe. No code was copied
from them; the structure and the habit of stating what was and was not checked are theirs.

## This recipe

The compatibility patch, the reconstruction of the launcher from the running lane, and the checks are this repository's own work.
