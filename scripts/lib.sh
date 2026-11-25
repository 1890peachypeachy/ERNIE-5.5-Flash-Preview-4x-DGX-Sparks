#!/usr/bin/env bash
# Shared helpers for start.sh and stop.sh. Sourced, never run.
[[ "${_ERNIE_LIB_LOADED:-0}" == 1 ]] && return 0
_ERNIE_LIB_LOADED=1

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
declare -a NODES=() FABRIC_IPS=()
# shellcheck disable=SC1091
[[ -f "$ROOT/scripts/local.sh" ]] && source "$ROOT/scripts/local.sh"
# shellcheck disable=SC1091
source "$ROOT/scripts/config.sh"

say()  { printf '%s\n' "$*" >&2; }
fail() { local code=$1; shift; printf '[%s] %s\n' "$code" "$*" >&2; exit 1; }
warn() { printf '[%s] %s\n' "$1" "${*:2}" >&2; }

# --- topology ---------------------------------------------------------------------------------------------------------
check_topology() {
  (( ${#NODES[@]} == 4 )) || fail E_CONFIG "NODES must list exactly 4 targets (rank 0 first); got ${#NODES[@]}. See scripts/local.sh.example."
  (( ${#FABRIC_IPS[@]} == 4 )) || fail E_CONFIG "FABRIC_IPS must list exactly 4 addresses, one per node; got ${#FABRIC_IPS[@]}."
  [[ "${NODES[0]}" == local ]] || fail E_CONFIG "NODES[0] must be 'local': run start.sh on the node that serves the API (rank 0)."
  local ip
  for ip in "${FABRIC_IPS[@]}"; do [[ "$ip" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]] || fail E_CONFIG "FABRIC_IPS entry '$ip' is not an IPv4 address."; done
  [[ "$FABRIC_IFACE" =~ ^[A-Za-z0-9_.-]+$ ]] || fail E_CONFIG "FABRIC_IFACE must be a netdev name, got '$FABRIC_IFACE'."
}

# node_sh <rank> <command string>: run on that rank's host (rank 0 runs here).
node_sh() {
  local i=$1; shift
  if [[ "${NODES[$i]}" == local ]]; then bash -c "$*"; else
    # shellcheck disable=SC2086
    ssh $SSH_OPTS -- "${NODES[$i]}" "$*"
  fi
}
node_name() { [[ "${NODES[$1]}" == local ]] && echo "rank0(local)" || echo "rank$1"; }

# --- image ------------------------------------------------------------------------------------------------------------
sha256_stdin() { if command -v sha256sum >/dev/null; then sha256sum; else shasum -a 256; fi; }
image_hash() { cat "$ROOT/docker/Dockerfile" "$ROOT/patches/patch_ernie_sm121.py" <(printf '%s\n%s' "$BASE_IMAGE" "$VLLM_ERNIE_REF") | sha256_stdin | cut -c1-12; }
image_tag()  { printf '%s:%s' "$IMAGE_NAME" "$(image_hash)"; }

# --- the container command (the one place every flag lives) ------------------------------------------------------------
# build_run_cmd <rank>: fills RUN_CMD with the full `docker run` argv for that rank. Rank 0 serves; ranks 1-3 are headless.
# Reconstructed from `docker inspect` of the running lane: see README "What is checked".
build_run_cmd() {
  local r=$1 tag; tag=$(image_tag)
  RUN_CMD=(docker run -d --name "$CONTAINER" --label "ernie55.recipe=$RECIPE_LABEL" --label "ernie55.rank=$r"
    --gpus all --network host --ipc host --cap-add IPC_LOCK --security-opt label=disable
    --ulimit memlock=-1 --ulimit nofile=1048576 --ulimit stack=67108864
    --device /dev/infiniband
    -v "$CACHE_DIR:/cache" -v "$MODEL_DIR:/model:ro"
    --entrypoint vllm)
  local kv; local -a envs=()
  # Everything below is set by the launcher (the image does not carry it). VLLM_HOST_IP differs per rank.
  for kv in \
    "CUDA_CACHE_PATH=/cache/nv" "FLASHINFER_CUDA_ARCH_LIST=12.1a" "FLASHINFER_DISABLE_VERSION_CHECK=1" \
    "FLASHINFER_WORKSPACE_BASE=/cache/fi" "GLOO_SOCKET_IFNAME=$FABRIC_IFACE" "HF_HOME=/cache/hf" \
    "HF_HUB_DISABLE_IMPLICIT_TOKEN=1" "HF_HUB_OFFLINE=1" "NCCL_ALGO=Ring" "NCCL_CROSS_NIC=1" "NCCL_CUMEM_ENABLE=1" \
    "NCCL_DEBUG=ERROR" "NCCL_IB_DISABLE=0" "NCCL_IB_GID_INDEX=$NCCL_GID_INDEX" "NCCL_IB_HCA=$RDMA_HCAS" \
    "NCCL_IB_MERGE_NICS=0" "NCCL_IB_SUBNET_AWARE_ROUTING=1" "NCCL_IB_SUBNET_PREFIX_LEN=$FABRIC_PREFIX_LEN" \
    "NCCL_IB_TC=106" "NCCL_IGNORE_CPU_AFFINITY=1" "NCCL_MAX_NCHANNELS=4" "NCCL_MIN_NCHANNELS=4" "NCCL_P2P_DISABLE=1" \
    "NCCL_RUNTIME_CONNECT=1" "NCCL_SHM_DISABLE=1" "NCCL_SOCKET_IFNAME=$FABRIC_IFACE" \
    "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" "TILELANG_CACHE_DIR=/cache/tilelang" \
    "TORCHINDUCTOR_CACHE_DIR=/cache/inductor" "TORCH_CUDA_ARCH_LIST=12.1a" "TRANSFORMERS_OFFLINE=1" \
    "TRITON_CACHE_DIR=/cache/triton" "VLLM_ALLOW_LONG_MAX_MODEL_LEN=1" "VLLM_CACHE_ROOT=/cache/vllm" \
    "VLLM_ENGINE_READY_TIMEOUT_S=$WAIT_TIMEOUT" "VLLM_HOST_IP=${FABRIC_IPS[$r]}" \
    "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0" "VLLM_NO_USAGE_STATS=1" "VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=256" \
    "VLLM_USE_BREAKABLE_CUDAGRAPH=1" "VLLM_WORKER_MULTIPROC_METHOD=spawn" "XDG_CACHE_HOME=/cache"; do
    envs+=("$kv")
  done
  # EXTRA_ENV: newline-separated KEY=VALUE lines (tuning experiments). A key already set above is replaced in place.
  local line key i found
  while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    [[ "$line" =~ ^[A-Za-z_][A-Za-z0-9_]*= ]] || fail E_CONFIG "EXTRA_ENV line is not KEY=VALUE: '$line'"
    key=${line%%=*}; found=0
    for i in "${!envs[@]}"; do [[ "${envs[$i]%%=*}" == "$key" ]] && { envs[i]=$line; found=1; }; done
    (( found )) || envs+=("$line")
  done <<< "$EXTRA_ENV"
  for kv in "${envs[@]}"; do RUN_CMD+=(-e "$kv"); done
  # shellcheck disable=SC2206
  local -a sizes=($CUDAGRAPH_SIZES)
  RUN_CMD+=("$tag" serve /model --served-model-name "$SERVED_NAME" --trust-remote-code --dtype bfloat16
    --kv-cache-dtype fp8 --max-model-len "$CONTEXT" --gpu-memory-utilization "$GPU_UTIL"
    --max-num-seqs "$MAX_SEQS" --max-num-batched-tokens "$MAX_BATCHED_TOKENS"
    --reasoning-parser ernie5 --tool-call-parser ernie5 --enable-auto-tool-choice --disable-custom-all-reduce
    --host "$BIND_HOST" --port "$PORT" --kv-cache-memory-bytes "$KV_CACHE_BYTES" --block-size "$BLOCK_SIZE"
    --enable-prefix-caching --moe-backend "$MOE_BACKEND"
    --speculative-config "{\"method\":\"dspark\",\"num_speculative_tokens\":$SPEC_TOKENS,\"draft_sample_method\":\"probabilistic\"}"
    --cudagraph-capture-sizes "${sizes[@]}"
    --tensor-parallel-size 4 --nnodes 4 --node-rank "$r" --master-addr "${FABRIC_IPS[0]}" --master-port "$MASTER_PORT"
    --distributed-executor-backend mp)
  # VLLM_EXTRA_ARGS: newline-separated, one argv element per line (so JSON values keep their spaces). Appended last, so a
  # repeated flag overrides the one above (argparse keeps the last value).
  while IFS= read -r line; do [[ -n "$line" ]] && RUN_CMD+=("$line"); done <<< "$VLLM_EXTRA_ARGS"
  (( r == 0 )) || RUN_CMD+=(--headless)
}
quote_cmd() { local a out=''; for a in "$@"; do out+="$(printf '%q' "$a") "; done; printf '%s' "${out% }"; }
