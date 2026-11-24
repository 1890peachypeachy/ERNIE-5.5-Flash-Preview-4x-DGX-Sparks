#!/usr/bin/env python3
"""Decode / prefill / speculative-decoding benchmark for the ERNIE lane. Standard library only; run with python3 -I.

    python3 -I scripts/bench/bench.py --url http://127.0.0.1:8888 --model <served name> --out results/run.json

What it measures (every request streams, thinking is always on, sampling is temperature 0.6, top_p 0.95):
  * decode tok/s per stream  = completion_tokens / (last chunk - first chunk). Completion tokens include the thinking tokens,
    because the server generates and bills them; "first chunk" is the first reasoning OR content token.
  * TTFT                     = request sent -> first reasoning or content token.
  * aggregate tok/s          = all completion tokens of a wave / wall time of the wave (the number that grows with concurrency).
  * prefill tok/s            = prompt_tokens / TTFT for the long-context runs (an upper bound on prefill time, TTFT also holds
    the first decode step).
  * speculative decoding     = the delta of the server's /metrics counters across each phase: accepted/drafted tokens and
    per-position acceptance. Mean tokens per verify step = 1 + accepted/drafts.
Every prompt starts with a unique tag so the prefix cache cannot serve one run's prompt to another.
Output lengths are bounded by --max-tokens; a request that stops earlier is reported with its own token count.
"""
import argparse
import json
import re
import statistics
import sys
import threading
import time
import urllib.request

TOPICS = [
    "the history of the printing press", "how a transformer language model is trained", "the water cycle",
    "the rules and strategy of chess", "how vaccines train the immune system", "the causes of the French Revolution",
    "how a compiler turns source code into machine code", "the physics of rainbows", "the economics of container shipping",
    "how a mechanical watch keeps time", "the life cycle of a star", "how public-key cryptography works",
    "the design of a suspension bridge", "the origins of jazz", "how bread rises", "the migration of monarch butterflies",
]
FILLER = ("The committee reviewed the quarterly logistics report and noted that regional throughput, warehouse utilisation, "
          "and carrier on-time rates were within their expected ranges, with minor seasonal variation in the northern depots. ")


def prompt_for(i, tag):
    return [{"role": "user", "content": f"[run {tag} request {i}] Write a detailed, well-structured explanation of {TOPICS[i % len(TOPICS)]}, "
             f"with examples, as long as you can."}]


def long_prompt(tokens_target, tag):
    # ~1 token per 4.6 chars for this filler; the server reports the exact prompt_tokens.
    n = int(tokens_target * 4.6 / len(FILLER))
    body = FILLER * n
    return [{"role": "user", "content": f"[run {tag}] {body}\n\nIn exactly three sentences, summarise what the report above is about."}]


def stream_request(url, model, messages, max_tokens):
    body = json.dumps({"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0.6, "top_p": 0.95,
                       "stream": True, "stream_options": {"include_usage": True}}).encode()
    req = urllib.request.Request(url + "/v1/chat/completions", body, {"Content-Type": "application/json"})
    t0 = time.perf_counter()
    first = last = None
    usage = None
    finish = None
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            ev = json.loads(data)
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices") or []:
                d = ch.get("delta") or {}
                if d.get("content") or d.get("reasoning") or d.get("reasoning_content"):
                    now = time.perf_counter()
                    first = first or now
                    last = now
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
    if usage is None or first is None:
        raise RuntimeError("no usage block or no tokens in the stream")
    ct = usage["completion_tokens"]
    decode_s = max(last - first, 1e-9)
    return {"prompt_tokens": usage["prompt_tokens"], "completion_tokens": ct,
            "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
            "ttft_s": first - t0, "decode_s": decode_s, "decode_tok_s": (ct - 1) / decode_s, "finish": finish,
            "total_s": time.perf_counter() - t0}


def spec_counters(url):
    out = {}
    try:
        text = urllib.request.urlopen(url + "/metrics", timeout=30).read().decode()
    except Exception:
        return out
    for line in text.splitlines():
        if not line.startswith("vllm:spec_decode_") or "_created" in line:
            continue
        m = re.match(r'^(vllm:spec_decode_\w+?)(?:_total)?(\{[^}]*\})? ([0-9.eE+-]+)$', line)
        if not m:
            continue
        name, labels, val = m.group(1), m.group(2) or "", float(m.group(3))
        pos = re.search(r'position="(\d+)"', labels)
        key = name + (f"@{pos.group(1)}" if pos else "")
        out[key] = out.get(key, 0.0) + val
    return out


def spec_delta(a, b):
    d = {k: b.get(k, 0) - a.get(k, 0) for k in b}
    drafts = d.get("vllm:spec_decode_num_drafts", 0)
    drafted = d.get("vllm:spec_decode_num_draft_tokens", 0)
    acc = d.get("vllm:spec_decode_num_accepted_tokens", 0)
    pos = {k.split("@")[1]: v for k, v in d.items() if k.startswith("vllm:spec_decode_num_accepted_tokens_per_pos@")}
    res = {"drafts": int(drafts), "drafted": int(drafted), "accepted": int(acc)}
    if drafts:
        res["mean_tokens_per_step"] = round(1 + acc / drafts, 2)
        res["accept_rate"] = round(acc / drafted, 3) if drafted else None
        res["accept_by_position"] = {p: round(v / drafts, 3) for p, v in sorted(pos.items())}
    return res


def wave(url, model, concurrency, n_requests, max_tokens, tag):
    results, errors, lock = [], [], threading.Lock()
    idx = iter(range(n_requests))

    def worker():
        while True:
            with lock:
                i = next(idx, None)
            if i is None:
                return
            try:
                r = stream_request(url, model, prompt_for(i, f"{tag}-c{concurrency}"), max_tokens)
                with lock:
                    results.append(r)
            except Exception as e:  # keep going; report at the end
                with lock:
                    errors.append(repr(e))

    t0 = time.perf_counter()
    threads = [threading.Thread(target=worker) for _ in range(concurrency)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    wall = time.perf_counter() - t0
    return results, errors, wall


def summarise(results, wall):
    if not results:
        return {}
    dec = [r["decode_tok_s"] for r in results]
    ttft = sorted(r["ttft_s"] for r in results)
    return {"requests": len(results), "wall_s": round(wall, 1),
            "completion_tokens": sum(r["completion_tokens"] for r in results),
            "aggregate_tok_s": round(sum(r["completion_tokens"] for r in results) / wall, 1),
            "decode_tok_s_per_stream_median": round(statistics.median(dec), 1),
            "decode_tok_s_per_stream_min": round(min(dec), 1), "decode_tok_s_per_stream_max": round(max(dec), 1),
            "ttft_s_median": round(statistics.median(ttft), 2), "ttft_s_max": round(ttft[-1], 2),
            "finish": {f: sum(1 for r in results if r["finish"] == f) for f in {r["finish"] for r in results}}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8888")
    ap.add_argument("--model", required=True, help="served model name to send")
    ap.add_argument("--label", default="ERNIE-5.5-Flash-Preview", help="name recorded in the results file")
    ap.add_argument("--out", required=True)
    ap.add_argument("--concurrency", default="1,2,4,8,16")
    ap.add_argument("--max-tokens", type=int, default=768)
    ap.add_argument("--waves", type=int, default=2, help="requests per concurrency level = concurrency x waves (min 4)")
    ap.add_argument("--long", default="8000,32000", help="prompt sizes in tokens for the prefill runs ('' to skip)")
    a = ap.parse_args()
    tag = time.strftime("%H%M%S")
    out = {"label": a.label, "date": time.strftime("%Y-%m-%d"), "max_tokens": a.max_tokens, "sampling": "temperature 0.6, top_p 0.95",
           "thinking": "always on", "decode": [], "prefill": []}
    # warm-up: one short request so the first measured run is not the first request of the process
    stream_request(a.url, a.model, prompt_for(0, tag + "-warm"), 64)
    for c in [int(x) for x in a.concurrency.split(",") if x]:
        n = max(4, c * a.waves)
        s0 = spec_counters(a.url)
        res, errs, wall = wave(a.url, a.model, c, n, a.max_tokens, tag)
        row = {"concurrency": c, **summarise(res, wall), "spec": spec_delta(s0, spec_counters(a.url)), "errors": errs[:3]}
        out["decode"].append(row)
        print(json.dumps(row), flush=True)
    for size in [int(x) for x in a.long.split(",") if x]:
        s0 = spec_counters(a.url)
        try:
            r = stream_request(a.url, a.model, long_prompt(size, f"{tag}-L{size}"), 128)
            row = {"target_prompt_tokens": size, "prompt_tokens": r["prompt_tokens"], "ttft_s": round(r["ttft_s"], 2),
                   "prefill_tok_s_upper_bound": round(r["prompt_tokens"] / r["ttft_s"]), "decode_tok_s": round(r["decode_tok_s"], 1),
                   "completion_tokens": r["completion_tokens"], "finish": r["finish"]}
        except Exception as e:
            row = {"target_prompt_tokens": size, "error": repr(e)}
        row["spec"] = spec_delta(s0, spec_counters(a.url))
        out["prefill"].append(row)
        print(json.dumps(row), flush=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print("wrote", a.out)


if __name__ == "__main__":
    sys.exit(main())
