# NOTICE

ERNIE-5.5-Flash-Preview on four DGX Sparks (vLLM TP=4 recipe)
Copyright 2026 the recipe authors. Licensed under the Apache License, Version 2.0 (see [LICENSE](LICENSE)).

This recipe builds on third-party work that keeps its own licence. If you redistribute it or a modified version, keep this notice
and the ones below, and state what you changed (Apache-2.0, section 4). People and projects are credited in [CREDITS.md](CREDITS.md).

## What is in this repository, and what is not

This repository holds a recipe: scripts, a Dockerfile, documentation and one patch script. It does not contain model weights, vLLM,
FlashInfer, the vLLM-ERNIE plugin or any binary. The licence above does not cover third-party software or weights that the recipe
downloads, and does not change their terms.

## Third-party terms to check before you use or redistribute

| Component | Where it comes from | Terms |
| --- | --- | --- |
| ERNIE-5.5-Flash-Preview weights | `huggingface.co/baidu/ERNIE-5.5-Flash-Preview` | Apache-2.0 per the model card. The card states this covers the Preview weights only; ERNIE 5.5 Flash Base has its own licence |
| vLLM-ERNIE plugin | `github.com/baidu/vLLM-ERNIE`, pinned commit | Apache-2.0 per its repository |
| vLLM | inside the base image | Apache-2.0 |
| Base image | `eugr/spark-vllm` | See `github.com/eugr/spark-vllm-docker`; this notice does not restate its terms |
| FlashInfer, torch, NCCL, CUDA | inside the base image | Their own licences; CUDA and NCCL are NVIDIA's |

`patches/patch_ernie_sm121.py` edits files of the vLLM-ERNIE plugin and of FlashInfer's `b12x` module at image-build time. If you
redistribute an image built with it, the modified files remain under their original licences and you must say they were changed.
