# Validation

What was run against the live lane, how to run it again, and what each result does and does not establish. Everything here was
observed on 2026-11-24; nothing is carried over from a benchmark or another lane.

## Re-run it

```sh
# 1. The recipe still regenerates the live containers (changes nothing).
DRY_RUN=1 ./start.sh                         # then diff each rank's args/env/mounts against `docker inspect` of that rank

# 2. Weights match the Hub.
python3 -I scripts/verify-weights.py "$MODEL_DIR"            # sizes + sha256 (166 GB: minutes)
python3 -I scripts/verify-weights.py "$MODEL_DIR" --size-only

# 3. The lane answers.
curl -s localhost:$PORT/health -o /dev/null -w '%{http_code}\n'
curl -s localhost:$PORT/v1/models | python3 -m json.tool | head
curl -s localhost:$PORT/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"<served name>","messages":[{"role":"user","content":"What is 17*23? Answer with just the number."}],"max_tokens":400,"temperature":0}'

# 4. Throughput (about 8 minutes; adds real load, so not on a lane clients are using).
python3 -I scripts/bench/bench.py --url http://127.0.0.1:$PORT --model <served name> --out results/run.json

# 5. Speculative decoding is accepting drafts (cumulative since the engine started).
curl -s localhost:$PORT/metrics | grep -E '^vllm:spec_decode_(num_drafts|num_draft_tokens|num_accepted_tokens)(_per_pos)?_total'
```

## Results

| Check | Result | Establishes |
| --- | --- | --- |
| Generated command vs `docker inspect`, ranks 0-3 | Argument lists equal, environment equal (42 launcher variables, `VLLM_HOST_IP` per rank), mounts equal | The repo reproduces the running configuration |
| Software on each node | Same base image ID, vllm, vllm-ernie, flashinfer, and md5 of the two patched plugin files | The four ranks run the same code |
| Weights vs Hub (`de3ffd3e8d55`) | 142 of 142 files; sha256 once, sizes again | The bytes are Baidu's |
| `/health`, `/v1/models` | 200; served name listed; `max_model_len` 262144 | The API is up as configured |
| `17*23`, temperature 0 | `391`, 22 tokens (19 thinking), `finish_reason: stop`, answer in `content`, thinking in `reasoning` | End-to-end text generation and the reasoning parser work |
| `scripts/bench/bench.py`, concurrency 1/2/4/8/16 (64 + 2 requests) and 6k/24k-token prompts | No errors; 46.6 tok/s per stream at one stream, 235 tok/s aggregate at 16; prefill about 1,525 tok/s; about 2.5 tokens per speculative step | Throughput at these settings, on one prompt family, one run per level (README "Performance") |
| Spec-decode counters | 2,527 drafts, 7,581 drafted, 4,633 accepted; per position 2,095 / 1,521 / 1,017 | DSpark drafts are accepted at about 61%; speed-up not measured |
| `scripts/tests/check.sh` | passes | Scripts parse, DRY_RUN has the right shape, nothing sensitive is committed |

## Reported, not re-run here

[docs/tuning.md](docs/tuning.md): one operator's tuning-sweep numbers (workload-specific decode, prefill to 88K, a 180K needle test, a k and `mnbt` comparison). Not reproduced by this repository.

## Not established

No repeat runs (so no variance across runs), no agent-loop trace, no prompt beyond about 24k tokens, no quality evaluation, no cold start from this repo, no first-boot-on-empty-cache timing,
no fault injection (a rank dying mid-request). The recipe's quality-gate step, in the sense of a scored task panel, does not exist
yet for this model; the two prompts above are smoke tests, not a gate.
