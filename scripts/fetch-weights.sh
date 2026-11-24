#!/usr/bin/env bash
# Download the pinned ERNIE-5.5-Flash-Preview weights (public, ungated, Apache-2.0; 142 files, 165.9 GB) into MODEL_DIR,
# then check them against the Hub (scripts/verify-weights.py). Run it on ONE node when MODEL_DIR is shared over NFS, or
# on every node that keeps its own copy. Needs `hf` (huggingface_hub) on PATH.
#
# Usage: scripts/fetch-weights.sh [--size-only]
# A single `hf download` stream measured 28-37 MB/s here, so the download takes a couple of hours; to go faster,
# split the shard list across nodes (each downloads its share, then rsync them together) as docs/HOW-WE-GOT-HERE.md does.
set -euo pipefail
# shellcheck disable=SC1091
source "$(dirname -- "$(readlink -f -- "$0" 2>/dev/null || echo "$0")")/lib.sh"
command -v hf >/dev/null || fail E_TOOL "hf not found: pip install -U huggingface_hub"
mkdir -p "$MODEL_DIR"
hf download "$MODEL_REPO" --revision "$MODEL_REV" --local-dir "$MODEL_DIR"
python3 -I "$ROOT/scripts/verify-weights.py" "$MODEL_DIR" --repo "$MODEL_REPO" --rev "$MODEL_REV" "$@"
