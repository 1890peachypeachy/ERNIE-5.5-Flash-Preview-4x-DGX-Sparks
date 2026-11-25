#!/usr/bin/env bash
# Serve ERNIE-5.5-Flash-Preview on four DGX Sparks at tensor parallel 4 (vLLM + the vLLM-ERNIE plugin, W4A8 experts,
# DSpark speculative decoding), one rank per Spark, over the ConnectX-7 fabric. Run it on the Spark that serves the API
# (rank 0). It checks every node, builds the image on each, starts ranks 3, 2, 1 and then 0, and waits for the API.
#
# Usage: ./start.sh [restart]      (./stop.sh stops it)
#        DRY_RUN=1 ./start.sh      prints every node's `docker run` and changes nothing (no ssh, no docker)
# Setup: cp scripts/local.sh.example scripts/local.sh   and fill in NODES, FABRIC_IPS, MODEL_DIR, CACHE_DIR.
# README: "How it starts".
set -euo pipefail
case "${1:-}" in
  help|-h|--help) awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; exit 0 ;;
esac
# shellcheck disable=SC1091
source "$(dirname -- "$(readlink -f -- "$0" 2>/dev/null || echo "$0")")/scripts/lib.sh"
check_topology

if [[ "$DRY_RUN" == 1 ]]; then
  say "DRY_RUN: image tag $(image_tag) (base $BASE_IMAGE, vLLM-ERNIE $VLLM_ERNIE_REF); nothing is run."
  for r in 3 2 1 0; do build_run_cmd "$r"; printf '# rank %s on %s\n%s\n\n' "$r" "${NODES[$r]}" "$(quote_cmd "${RUN_CMD[@]}")"; done
  exit 0
fi

[[ "${1:-}" == restart ]] && "$ROOT/stop.sh"

# --- 1. checks on every node -------------------------------------------------------------------------------------------
preflight() {
  local r=$1 out
  out=$(node_sh "$r" "
    set -u
    [ \"\$(uname -m)\" = aarch64 ] || echo 'not aarch64'
    command -v docker >/dev/null || echo 'no docker'
    { docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q nvidia || docker info 2>/dev/null | grep -q 'cdi: nvidia.com/gpu'; } || echo 'docker has neither an nvidia runtime nor CDI nvidia.com/gpu devices'
    nvidia-smi -L 2>/dev/null | grep -q GB10 || echo 'no GB10 visible'
    [ -d /dev/infiniband ] || echo 'no /dev/infiniband'
    ip -4 addr show dev '$FABRIC_IFACE' 2>/dev/null | grep -q 'inet ${FABRIC_IPS[$r]}/' || echo 'fabric IP ${FABRIC_IPS[$r]} not on $FABRIC_IFACE'
    [ -f '$MODEL_DIR/config.json' ] && [ -f '$MODEL_DIR/model.safetensors.index.json' ] || echo 'weights missing under $MODEL_DIR'
    n=\$(ls '$MODEL_DIR'/model-*-of-00128.safetensors 2>/dev/null | wc -l); [ \"\$n\" -eq 128 ] || echo \"\$n of 128 model shards present\"
    mkdir -p '$CACHE_DIR' 2>/dev/null && [ -w '$CACHE_DIR' ] || echo 'CACHE_DIR not writable'
    free=\$(df -BG --output=avail '$CACHE_DIR' 2>/dev/null | tail -1 | tr -dc 0-9); [ \"\${free:-0}\" -ge $MIN_DISK_GIB ] || echo \"only \${free:-0} GiB free under CACHE_DIR (want $MIN_DISK_GIB)\"
    docker ps -a --format '{{.Names}}' | grep -qx '$CONTAINER' && echo 'container $CONTAINER already exists (./stop.sh first)'
    true" 2>&1) || fail E_NODE "Cannot reach $(node_name "$r") (${NODES[$r]}): $out"
  [[ -z "$out" ]] || fail E_NODE "$(node_name "$r"): ${out//$'\n'/; }"
  say "ok   $(node_name "$r")"
}
for r in 0 1 2 3; do preflight "$r"; done
node_sh 0 "ss -ltn | grep -q ':$PORT '" && fail E_PORT "Port $PORT is already in use on rank 0."

# --- 2. image on every node (built locally; the patch is deterministic over a pinned base) -----------------------------------
TAG=$(image_tag)
build_image() {
  local r=$1
  if node_sh "$r" "docker image inspect '$TAG' >/dev/null 2>&1"; then say "ok   $(node_name "$r") has $TAG"; return 0; fi
  say "build $(node_name "$r"): $TAG (the base is ~25 GB the first time)"
  if [[ "${NODES[$r]}" == local ]]; then
    docker build -f "$ROOT/docker/Dockerfile" --build-arg "BASE_IMAGE=$BASE_IMAGE" --build-arg "VLLM_ERNIE_REF=$VLLM_ERNIE_REF" -t "$TAG" "$ROOT"
  else
    # shellcheck disable=SC2086
    tar -C "$ROOT" -cf - docker patches | ssh $SSH_OPTS -- "${NODES[$r]}" "d=\$(mktemp -d) && tar -C \"\$d\" -xf - && docker build -f \"\$d/docker/Dockerfile\" --build-arg 'BASE_IMAGE=$BASE_IMAGE' --build-arg 'VLLM_ERNIE_REF=$VLLM_ERNIE_REF' -t '$TAG' \"\$d\"; rc=\$?; rm -rf \"\$d\"; exit \$rc"
  fi
}
for r in 0 1 2 3; do build_image "$r"; done

# --- 3. launch: workers first (they wait for rank 0), then rank 0 ---------------------------------------------------------------
mkdir -p "$LOG_DIR/current"
launch() {
  local r=$1; build_run_cmd "$r"
  node_sh "$r" "$(quote_cmd "${RUN_CMD[@]}")" >/dev/null || fail E_LAUNCH "docker run failed on $(node_name "$r")"
  say "up   $(node_name "$r")"
}
for r in 3 2 1 0; do launch "$r"; done

# --- 4. ready: /health, then /v1/models, then a short completion ------------------------------------------------------------------
say "waiting for the API (the first boot compiles kernels: up to ${WAIT_TIMEOUT}s)"
deadline=$(( SECONDS + WAIT_TIMEOUT ))
until curl -fsS -o /dev/null "http://127.0.0.1:$PORT/health" 2>/dev/null; do
  for r in 0 1 2 3; do
    node_sh "$r" "docker ps --format '{{.Names}}' | grep -qx '$CONTAINER'" || { node_sh "$r" "docker logs --tail 40 '$CONTAINER' 2>&1" >&2 || true; fail E_RANK "$(node_name "$r") exited during start-up (log above). ./stop.sh archives the logs."; }
  done
  (( SECONDS < deadline )) || fail E_TIMEOUT "No /health within ${WAIT_TIMEOUT}s: docker logs $CONTAINER on each node."
  sleep 10
done
curl -fsS "http://127.0.0.1:$PORT/v1/models" | grep -q "\"$SERVED_NAME\"" || fail E_MODEL "/v1/models does not list $SERVED_NAME."
reply=$(curl -fsS "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
  -d "{\"model\":\"$SERVED_NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with the single word PONG.\"}],\"max_tokens\":512,\"temperature\":0}") \
  || fail E_SMOKE "The smoke request failed."
grep -q '"completion_tokens"' <<<"$reply" || fail E_SMOKE "The smoke reply has no usage block: $reply"
say "READY  http://$BIND_HOST:$PORT/v1  model=$SERVED_NAME  (thinking is always on; reasoning arrives in the reply's reasoning field)"
