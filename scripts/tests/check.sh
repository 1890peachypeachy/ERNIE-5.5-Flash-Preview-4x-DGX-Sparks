#!/usr/bin/env bash
# Offline checks: no GPU, no docker, no network, no ssh. Run before every commit:  scripts/tests/check.sh
# 1. every shell script parses (and passes shellcheck when it is installed)
# 2. DRY_RUN=1 ./start.sh prints four rank commands with the settings TP=4 needs
# 3. the repository carries nothing that would expose the people or machines behind it
set -uo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT" || exit 1
fails=0
ok()  { printf 'ok    %s\n' "$*"; }
bad() { printf 'FAIL  %s\n' "$*"; fails=$((fails + 1)); }

for f in start.sh stop.sh scripts/*.sh scripts/tests/*.sh; do bash -n "$f" && ok "syntax $f" || bad "syntax $f"; done
python3 -I -c 'import ast,sys; [ast.parse(open(f).read(), f) for f in sys.argv[1:]]' scripts/verify-weights.py scripts/bench/bench.py patches/patch_ernie_sm121.py && ok "python parses" || bad "python parse"
if command -v shellcheck >/dev/null; then
  shellcheck -x -S warning start.sh stop.sh scripts/lib.sh scripts/config.sh scripts/fetch-weights.sh scripts/tests/check.sh && ok shellcheck || bad shellcheck
fi

# DRY_RUN against the example topology, in a scratch copy so no scripts/local.sh of yours is read.
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
cp -R start.sh stop.sh scripts docker patches "$tmp/"; rm -f "$tmp/scripts/local.sh"; cp "$tmp/scripts/local.sh.example" "$tmp/scripts/local.sh"
out=$(cd "$tmp" && DRY_RUN=1 ./start.sh 2>/dev/null) || bad "DRY_RUN start.sh exits non-zero"
[[ $(grep -c '^docker run' <<<"$out") == 4 ]] && ok "four docker run lines" || bad "expected four docker run lines"
for r in 0 1 2 3; do grep -q -- "--node-rank $r " <<<"$out" || grep -q -- "--node-rank $r\$" <<<"$out" || bad "rank $r missing"; done
[[ $(grep -c -- '--headless' <<<"$out") == 3 ]] && ok "ranks 1-3 are headless, rank 0 is not" || bad "--headless count is not 3"
[[ $(grep -c -- '--tensor-parallel-size 4 --nnodes 4' <<<"$out") == 4 ]] && ok "TP=4 on 4 nodes" || bad "TP/nnodes"
grep -q -- '--restart' <<<"$out" && bad "a restart policy is set (never auto-restart an unverified lane)" || ok "no restart policy"
grep -q -- '--host 127.0.0.1' <<<"$out" && ok "binds 127.0.0.1 by default" || bad "default bind is not 127.0.0.1"
(cd "$tmp" && DRY_RUN=1 ./stop.sh >/dev/null 2>&1) && ok "DRY_RUN stop.sh" || bad "DRY_RUN stop.sh"

# Exposure scan over tracked-or-trackable files. Placeholders (RFC 5737 192.0.2.x) are fine.
scan=$(git ls-files -co --exclude-standard 2>/dev/null | grep -v '^scripts/tests/check.sh$' | grep -v '^LICENSE$')
hits=$(xargs grep -nIE \
  '(^|[^0-9.])(10\.[0-9]+\.[0-9]+\.[0-9]+|172\.(1[6-9]|2[0-9]|3[01])\.[0-9]+\.[0-9]+|192\.168\.[0-9]+\.[0-9]+|100\.(6[4-9]|[7-9][0-9]|1[01][0-9]|12[0-7])\.[0-9]+\.[0-9]+)([^0-9]|$)|tailscale|\.ts\.net|/Users/|/home/[a-z0-9_-]+/|hf_[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,}|BEGIN [A-Z ]*PRIVATE KEY|api[_-]?key *[:=] *[A-Za-z0-9]{12,}|password *[:=]|@[a-z0-9-]+\.(local|lan)\b|(^|[^a-z])spark[1-4]([^0-9a-z]|$)' \
  <<<"$scan" 2>/dev/null || true)
[[ -z "$hits" ]] && ok "no private addresses, hostnames, home paths or credentials" || { bad "exposure scan:"; printf '%s\n' "$hits"; }

(( fails == 0 )) && { echo "ALL CHECKS PASSED"; exit 0; } || { echo "$fails CHECK(S) FAILED"; exit 1; }
