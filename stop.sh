#!/usr/bin/env bash
# Stop the ERNIE lane: rank 0 first, then ranks 1-3. Each rank's log is saved to logs/current/ and the set is archived to
# logs/ernie55-<time>.tar.gz. Only a container named $CONTAINER that carries the label ernie55.recipe=ernie55-tp4, or runs an
# image tagged $IMAGE_NAME:*, is ever stopped or removed (so a lane started by hand from the same image is found too).
#
# Usage: ./stop.sh        DRY_RUN=1 ./stop.sh   prints what it would do
set -euo pipefail
case "${1:-}" in
  help|-h|--help) awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; exit 0 ;;
esac
# shellcheck disable=SC1091
source "$(dirname -- "$(readlink -f -- "$0" 2>/dev/null || echo "$0")")/scripts/lib.sh"
check_topology

mkdir -p "$LOG_DIR/current"
for r in 0 1 2 3; do
  find_cmd="docker ps -a --filter 'name=^/$CONTAINER\$' --format '{{.ID}}|{{.Image}}|{{.Label \"ernie55.recipe\"}}' | awk -F'|' -v lab='$RECIPE_LABEL' -v img='$IMAGE_NAME:' 'index(\$2, img) == 1 || \$3 == lab { print \$1 }'"
  if [[ "$DRY_RUN" == 1 ]]; then say "would stop $(node_name "$r"): $find_cmd"; continue; fi
  id=$(node_sh "$r" "$find_cmd" 2>/dev/null || true)
  if [[ -z "$id" ]]; then say "none $(node_name "$r")"; continue; fi
  node_sh "$r" "docker logs '$id' 2>&1" > "$LOG_DIR/current/rank$r.log" || true
  node_sh "$r" "docker stop -t 30 '$id' >/dev/null && docker rm '$id' >/dev/null" || warn W_STOP "$(node_name "$r"): stop failed"
  say "down $(node_name "$r")"
done
[[ "$DRY_RUN" == 1 ]] && exit 0
stamp=$(date +%Y%m%d-%H%M%S)
tar -C "$LOG_DIR" -czf "$LOG_DIR/ernie55-$stamp.tar.gz" current && rm -f "$LOG_DIR"/current/rank*.log
say "logs archived: $LOG_DIR/ernie55-$stamp.tar.gz"
