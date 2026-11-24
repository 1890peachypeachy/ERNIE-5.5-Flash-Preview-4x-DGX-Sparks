# ERNIE-5.5-Flash-Preview on a DGX Spark (GB10, sm_121): eugr's Spark vLLM build + Baidu's vLLM-ERNIE plugin + the
# recipe's compatibility patches. Build context is the repository root (start.sh builds it on every node).
#
#   docker build -f docker/Dockerfile -t ernie55-vllm:dev .
#
# The base is pinned by digest. It is vLLM 0.34.1rc1.dev37 (commit 3e72879e), torch 2.14.0+cu130, FlashInfer 0.7.5,
# transformers 5.23.0. Baidu's model card installs vllm==0.32.0; the plugin does not run unmodified on this newer
# vLLM, so patches/patch_ernie_sm121.py adapts it (see docker/README.md).
ARG BASE_IMAGE=eugr/spark-vllm@sha256:afbe9787f5ef6eb3be5747902bf04bdc3515ec34e7fca2659aea34ef8221ea6a
FROM ${BASE_IMAGE}

ARG VLLM_ERNIE_REF=ba322a5634f91d3e6f96ae711df72ddaf3c38b6e
COPY patches/patch_ernie_sm121.py /tmp/patch_ernie_sm121.py
RUN uv pip install --no-deps "git+https://github.com/baidu/vLLM-ERNIE.git@${VLLM_ERNIE_REF}" && \
    python3 /tmp/patch_ernie_sm121.py
