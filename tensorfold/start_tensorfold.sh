#!/usr/bin/env bash
# Launch native TensorFold ERNIE-5.5-Flash-Preview across 4 DGX Sparks (TP=4 over RoCE v2).
set -euo pipefail

MODEL_DIR="${MODEL_DIR:?Set MODEL_DIR to ERNIE-5.5-Flash-Preview weights}"
TF_BIN="${TF_BIN:?Set TF_BIN to tensorfold-native binary path on all 4 nodes}"
LIB_DIR="${LIB_DIR:-}"
MASTER_IP="${MASTER_IP:?Set MASTER_IP to rank-0 RoCE IP}"
MASTER_PORT="${MASTER_PORT:-29688}"
RANK1_HOST="${RANK1_HOST:?Set RANK1_HOST (e.g. user@192.0.2.2)}"
RANK2_HOST="${RANK2_HOST:?Set RANK2_HOST (e.g. user@192.0.2.3)}"
RANK3_HOST="${RANK3_HOST:?Set RANK3_HOST (e.g. user@192.0.2.4)}"

BIND_HOST="${BIND_HOST:-127.0.0.1}"
PORT="${PORT:-8888}"
SERVED_NAME="${SERVED_NAME:-ERNIE-5.5-Flash-Preview}"
CONTEXT="${CONTEXT:-262144}"
IFACE="${IFACE:-enp1s0f1np1}"
IB_HCA="${IB_HCA:-rocep1s0f1,roceP2p1s0f1}"
LOG_DIR="${LOG_DIR:-/tmp/tensorfold-ernie55-logs}"
mkdir -p "$LOG_DIR"

ENV_COMMON="NCCL_SOCKET_IFNAME=$IFACE NCCL_IB_HCA=$IB_HCA NCCL_IB_GID_INDEX=3 NCCL_IB_TC=106 NCCL_CROSS_NIC=1 NCCL_IB_SUBNET_AWARE_ROUTING=1 NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 NCCL_ALGO=Ring NCCL_PROTO=Simple NCCL_MIN_NCHANNELS=4 NCCL_MAX_NCHANNELS=4 TF_ROCE=1 TF_ROCE_HCA=$IB_HCA TF_ROCE_GID_INDEX=3"
if [ -n "$LIB_DIR" ]; then
  ENV_COMMON="LD_LIBRARY_PATH=$LIB_DIR $ENV_COMMON"
fi

ARGS_COMMON="$MODEL_DIR --backend cuda --host $BIND_HOST --port $PORT --name $SERVED_NAME --context $CONTEXT --thinking --tp 4 --master $MASTER_IP --master-port $MASTER_PORT"

# Launch workers (ranks 3, 2, 1) first, then rank 0 (API server)
ssh -o BatchMode=yes "$RANK3_HOST" "nohup env $ENV_COMMON $TF_BIN serve $ARGS_COMMON --rank 3 >/tmp/tf_ernie_r3.log 2>&1 </dev/null &"
echo "launched rank 3 on $RANK3_HOST"
ssh -o BatchMode=yes "$RANK2_HOST" "nohup env $ENV_COMMON $TF_BIN serve $ARGS_COMMON --rank 2 >/tmp/tf_ernie_r2.log 2>&1 </dev/null &"
echo "launched rank 2 on $RANK2_HOST"
ssh -o BatchMode=yes "$RANK1_HOST" "nohup env $ENV_COMMON $TF_BIN serve $ARGS_COMMON --rank 1 >/tmp/tf_ernie_r1.log 2>&1 </dev/null &"
echo "launched rank 1 on $RANK1_HOST"
nohup env $ENV_COMMON "$TF_BIN" serve $ARGS_COMMON --rank 0 >"$LOG_DIR/tf_ernie_r0.log" 2>&1 </dev/null &
echo "launched rank 0 locally (log: $LOG_DIR/tf_ernie_r0.log)"

echo "waiting for http://127.0.0.1:$PORT/v1/models ..."
for i in $(seq 1 140); do
  if curl -fsS --max-time 2 "http://127.0.0.1:$PORT/v1/models" >/tmp/tf_models.json 2>/dev/null; then
    echo "READY after $((i*3))s:"
    cat /tmp/tf_models.json
    echo ""
    exit 0
  fi
  sleep 3
done
echo "TIMEOUT waiting for /v1/models! Tail of r0 log:"
tail -n 30 "$LOG_DIR/tf_ernie_r0.log"
exit 1
