#!/usr/bin/env bash
# Defaults for the ERNIE-5.5-Flash-Preview TP=4 recipe. Sourced by start.sh and stop.sh; never run on its own.
#
# scripts/local.sh (yours, git-ignored) is sourced BEFORE this file, so a value it sets beats the environment, and the
# environment beats the defaults below. Topology (NODES, FABRIC_IPS) has no default: you must set it in local.sh.

# --- topology (required) -------------------------------------------------------------------------------------------
# NODES      one ssh target per rank, in rank order. Rank 0 serves the API; write "local" when you run start.sh on it.
# FABRIC_IPS the IPv4 each node holds on the ConnectX-7 fabric (the subnet NCCL talks over), same order as NODES.
# (declared by scripts/lib.sh so a value from local.sh is never reset here)

# --- weights ---------------------------------------------------------------------------------------------------------
MODEL_REPO="${MODEL_REPO:-baidu/ERNIE-5.5-Flash-Preview}"
MODEL_REV="${MODEL_REV:-de3ffd3e8d55ec0f68ca51cbcec52d2a6490cf3b}"      # Hub commit the recipe was verified on
MODEL_DIR="${MODEL_DIR:-$HOME/models/ERNIE-5.5-Flash-Preview}"          # same path on every node (local copy or NFS)
CACHE_DIR="${CACHE_DIR:-$HOME/.cache/ernie55-tp4}"                      # compile/JIT caches, per node, writable

# --- image -----------------------------------------------------------------------------------------------------------
# The base is eugr's DGX Spark vLLM build, pinned by digest (the live lane was built from its :latest of 2026-11-24).
BASE_IMAGE="${BASE_IMAGE:-eugr/spark-vllm@sha256:afbe9787f5ef6eb3be5747902bf04bdc3515ec34e7fca2659aea34ef8221ea6a}"
VLLM_ERNIE_REF="${VLLM_ERNIE_REF:-ba322a5634f91d3e6f96ae711df72ddaf3c38b6e}"   # baidu/vLLM-ERNIE commit
IMAGE_NAME="${IMAGE_NAME:-ernie55-vllm}"
CONTAINER="${CONTAINER:-ernie55-tp4}"
# shellcheck disable=SC2034  # used by lib.sh and stop.sh
RECIPE_LABEL="ernie55-tp4"                                               # stop.sh only touches containers with this label

# --- serving ---------------------------------------------------------------------------------------------------------
SERVED_NAME="${SERVED_NAME:-ERNIE-5.5-Flash-Preview}"   # the name clients send as "model"
BIND_HOST="${BIND_HOST:-127.0.0.1}"                      # 0.0.0.0 exposes an unauthenticated API: choose it on purpose
PORT="${PORT:-8888}"
MASTER_PORT="${MASTER_PORT:-29688}"
CONTEXT="${CONTEXT:-262144}"                             # the model supports 1,048,576; this is what was verified
GPU_UTIL="${GPU_UTIL:-0.80}"
MAX_SEQS="${MAX_SEQS:-16}"
MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-8192}"
KV_CACHE_BYTES="${KV_CACHE_BYTES:-17179869184}"          # 16 GiB of KV per rank, an explicit size (not a fraction)
BLOCK_SIZE="${BLOCK_SIZE:-256}"
SPEC_TOKENS="${SPEC_TOKENS:-3}"                          # DSpark draft block; the model card suggests up to 7 at low concurrency (untested here)
CUDAGRAPH_SIZES="${CUDAGRAPH_SIZES:-1 2 4 8 16 24 32 48 64}"
MOE_BACKEND="${MOE_BACKEND:-b12x}"

# --- tuning hooks (empty = the recipe as published) ---------------------------------------------------------------
# VLLM_EXTRA_ARGS  newline-separated vllm serve arguments, one argv element per line, appended last (later flags win).
# EXTRA_ENV        newline-separated KEY=VALUE container env; a key the recipe already sets is replaced, not duplicated.
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"
EXTRA_ENV="${EXTRA_ENV:-}"

# --- fabric ----------------------------------------------------------------------------------------------------------
FABRIC_IFACE="${FABRIC_IFACE:-enp1s0f1np1}"              # netdev of the CX7 port that carries the fabric subnet
RDMA_HCAS="${RDMA_HCAS:-rocep1s0f1,roceP2p1s0f1}"        # the two RDMA devices of that cabled port
NCCL_GID_INDEX="${NCCL_GID_INDEX:-3}"                    # RoCE v2 GID holding the IPv4
FABRIC_PREFIX_LEN="${FABRIC_PREFIX_LEN:-24}"

# --- start-up behaviour ----------------------------------------------------------------------------------------------
WAIT_TIMEOUT="${WAIT_TIMEOUT:-3600}"                     # seconds to wait for /health (first boot compiles kernels)
MIN_DISK_GIB="${MIN_DISK_GIB:-60}"                       # free GiB wanted under CACHE_DIR and for the image on each node
SSH_OPTS="${SSH_OPTS:--o BatchMode=yes -o ConnectTimeout=8 -o ServerAliveInterval=15}"
DRY_RUN="${DRY_RUN:-0}"
LOG_DIR="${LOG_DIR:-$ROOT/logs}"
