#!/usr/bin/env python3
"""Benchmark any OpenAI-compatible server the same way, so every recipe's numbers compare.

    python3 bench/bench.py http://127.0.0.1:8080 --recipe qwen3.8-27b --suites standard,long,prefill

Standard library only: it runs on a bare DGX Spark host, in any container, or on a Mac.

Suites (see bench/README.md for why each exists):
  standard   TensorFold's published protocol: code/chat x sampled/greedy, 64 tokens, seeds 1234-1238, medians.
             Directly comparable with the TensorFold and vLLM numbers in bench/reference.json.
  long       512-token replies on realistic work (code, prose, structured JSON), greedy and sampled.
  prefill    cold prompts of increasing length (a random salt defeats prefix caches): TTFT and prefill tok/s.
  exactness  TensorFold only: each reply drafted vs the same request with "draft": false must be byte-identical.
  context    coding-agent long context: cold synthetic codebase of 32k..850k tokens with a hidden fact, then a
             follow-up turn. Prefill tok/s, decode tok/s at depth, needle found, and follow-up (cache reuse) TTFT.
  concurrency  N parallel streams (vLLM batches them; TensorFold queues them): per-stream and aggregate tok/s.

Decode tok/s = (completion tokens - 1) / (last token time - first token time), the span TensorFold's and
vLLM's own benches use: it excludes prefill and the first token.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import hashlib
import json
import os
import platform
import random
import re
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SEEDS = [1234, 1235, 1236, 1237, 1238]
SAMPLED = {"temperature": 1.0, "top_k": 20, "top_p": 0.95}
GREEDY = {"temperature": 0.0}
# Every template reads its own key and ignores the rest: Qwen/GLM use enable_thinking, DeepSeek thinking.
NO_THINK = {"enable_thinking": False, "thinking": False, "reasoning_effort": "low"}

# The exact prompts of TensorFold's tools/bench_openai.py, so the standard suite reproduces its published cells.
CODE = "Write a short Python function that computes the Fibonacci sequence and explain it."
CHAT = "Explain how matrix multiplication uses a GPU in plain English, then give a small numerical example."

LONG_PROMPTS = [
    ("code", "Write a complete Python module implementing an LRU cache class with get, put, resize and "
             "statistics methods, full type hints and docstrings, followed by pytest unit tests for every method."),
    ("prose", "Write a detailed essay on the history of the Silk Road: its origins, the goods and ideas that "
              "travelled along it, the empires that controlled it, and why it declined."),
    ("json", "Return a JSON array of 15 fictional employees. Each object has id, name, email, department, "
             "title, salary, start_date, skills (a list of 3 strings) and manager_id. Output only the JSON."),
]

FILLER = ("The quarterly report covers revenue, logistics, staffing and product updates across every region. "
          "Each section lists the figures, the changes since last quarter and the risks the team sees ahead. ")


WORDS = ("account alert audit batch billing buffer cache catalog channel checkout client cluster config consumer "
         "cursor customer dataset device digest dispatch document driver event export feature filter gateway graph "
         "handler health index inventory invoice job journal key ledger limit listener loader lock manifest metric "
         "migration monitor node notifier order owner packet parser partition payment pipeline policy pool profile "
         "queue quota record region registry replica report request resolver route runner scheduler schema segment "
         "session shard signal snapshot source storage stream subscriber task tenant token topic tracker transfer "
         "upload user validator vault version worker workspace").split()


def synthetic_codebase(chars: int, secret: str, rng: random.Random) -> str:
    """Varied, deterministic Python-like source of about `chars` characters, with DEPLOY_CODE = secret halfway.
    Varied on purpose: repetitive filler would let copy-drafting and n-gram tables look unrealistically good."""
    parts, n, placed, i = [], 0, False, 0
    while n < chars:
        a, b, c = rng.sample(WORDS, 3)
        cls = "".join(w.capitalize() for w in (a, b))
        lines = [f"# ===== file: src/{a}/{b}_{c}_{i}.py =====",
                 f'"""{a.capitalize()} {b} {c} utilities: keeps the {c} of each {a} in sync with its {b}."""',
                 "from __future__ import annotations", "import logging", "from dataclasses import dataclass, field", "",
                 f"LOG = logging.getLogger('{a}.{b}')", f"MAX_{c.upper()} = {rng.randint(2, 4096)}", "",
                 "@dataclass", f"class {cls}:", f"    {a}_id: int", f"    {c}: list[str] = field(default_factory=list)",
                 f"    retries: int = {rng.randint(1, 9)}", ""]
        for _ in range(rng.randint(2, 5)):
            v1, v2 = rng.sample(WORDS, 2)
            k = rng.randint(3, 97)
            lines += [f"    def {v1}_{v2}(self, {v2}: dict, limit: int = {k}) -> list[str]:",
                      f'        """Return up to `limit` {v2} keys that need a {v1} pass for this {a}."""',
                      f"        out = [key for key, value in {v2}.items() if value and len(key) % {k % 7 + 2}]",
                      f"        if len(out) > MAX_{c.upper()}:",
                      f"            LOG.warning('%s: {v1} backlog %d', self.{a}_id, len(out))",
                      f"        return sorted(out)[:limit]", ""]
        if not placed and n > chars // 2:
            lines += ["", f'DEPLOY_CODE = "{secret}"  # read by the release pipeline, never change by hand', ""]
            placed = True
        block = "\n".join(lines) + "\n\n"
        parts.append(block)
        n += len(block)
        i += 1
    return "".join(parts)


# --------------------------------------------------------------------------------------------------- HTTP ---

def post_stream(base: str, path: str, body: dict, timeout: float = 1800) -> dict:
    """Send one streamed request; return timings, token count, text and any engine stats it reports."""
    body = {**body, "stream": True, "stream_options": {"include_usage": True}}
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json",
                                          **({"Authorization": f"Bearer {os.environ['API_KEY']}"}
                                             if os.environ.get("API_KEY") else {})})
    start = time.perf_counter()
    first = last = None
    usage, extra, pieces, chunks = None, {}, [], 0
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[5:])
            if chunk.get("error"):
                raise RuntimeError(chunk["error"])
            usage = chunk.get("usage") or usage
            for key in ("tensorfold", "speculative", "exact_mode"):
                if chunk.get(key):
                    extra[key] = chunk[key]
            for choice in chunk.get("choices", []):
                delta = choice.get("delta") or {}
                piece = choice.get("text") or delta.get("content") or delta.get("reasoning_content") or ""
                if piece:
                    now = time.perf_counter()
                    first = now if first is None else first
                    last = now
                    pieces.append(piece)
                    chunks += 1
    text = "".join(pieces)
    n = int(usage["completion_tokens"]) if usage and usage.get("completion_tokens") else chunks
    decode_s = (last - first) if first is not None and last is not None else 0.0
    return {
        "ttft_s": (first - start) if first is not None else None,
        "decode_s": decode_s,
        "total_s": time.perf_counter() - start,
        "tokens": n,
        "prompt_tokens": (usage or {}).get("prompt_tokens"),
        "cached_tokens": ((usage or {}).get("prompt_tokens_details") or {}).get("cached_tokens"),
        "decode_tps": (n - 1) / decode_s if n > 1 and decode_s > 0 else None,
        "text_sha256": hashlib.sha256(text.encode()).hexdigest()[:16],
        "text": text,
        **extra,
    }


def get_json(base: str, path: str) -> dict | None:
    try:
        with urllib.request.urlopen(base + path, timeout=10) as r:
            return json.load(r)
    except (urllib.error.URLError, OSError, ValueError):
        return None


def public_url(base: str) -> str:
    """The server URL as results keep it: local addresses as they are, any other host name replaced (results get shared)."""
    u = urllib.parse.urlsplit(base)
    if u.hostname in ("127.0.0.1", "localhost", "::1"):
        return base
    return urllib.parse.urlunsplit((u.scheme, "<server>" + (f":{u.port}" if u.port else ""), u.path, "", ""))


def completion(model: str, prompt: str, tokens: int, sampling: dict, seed: int | None, **more) -> tuple[str, dict]:
    # TensorFold renders a completion's prompt through the chat template, and with thinking on the reply would
    # be spent in a think block its text stream never shows. Thinking off, as in every published cell.
    body = {"model": model, "prompt": prompt, "max_tokens": tokens, "ignore_eos": True,
            "chat_template_kwargs": NO_THINK, **sampling, **more}
    if seed is not None:
        body["seed"] = seed
    return "/v1/completions", body


def chat(model: str, prompt: str, tokens: int, sampling: dict, seed: int | None, ignore_eos=True, **more):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": tokens,
            "chat_template_kwargs": NO_THINK, "ignore_eos": ignore_eos, **sampling, **more}
    if seed is not None:
        body["seed"] = seed
    return "/v1/chat/completions", body


# ------------------------------------------------------------------------------------------------ helpers ---

def summarize(runs: list[dict]) -> dict:
    tps = [r["decode_tps"] for r in runs if r.get("decode_tps")]
    ttft = [r["ttft_s"] for r in runs if r.get("ttft_s") is not None]
    out = {
        "decode_tps_median": round(statistics.median(tps), 2) if tps else None,
        "decode_tps_min": round(min(tps), 2) if tps else None,
        "decode_tps_max": round(max(tps), 2) if tps else None,
        "decode_tps_all": [round(x, 2) for x in tps],
        "ttft_s_median": round(statistics.median(ttft), 3) if ttft else None,
        "tokens": runs[0]["tokens"] if runs else None,
    }
    acc = [r["speculative"] for r in runs if isinstance(r.get("speculative"), dict)]
    if acc and all("drafted" in a for a in acc):
        drafted = sum(a.get("drafted", 0) for a in acc)
        out["draft_acceptance"] = round(sum(a.get("accepted", 0) for a in acc) / drafted, 3) if drafted else None
    return out


def slow_runs(tps: list[float]) -> bool:
    """GB10 page-migration bursts can halve a run (TensorFold's GLM recipe). Flag a run under 85% of the best."""
    return len(tps) > 1 and min(tps) < 0.85 * max(tps)


class Bench:
    def __init__(self, args):
        self.a = args
        self.base = args.url.rstrip("/")
        self.model = args.model + (args.model_suffix or "")
        self.cells: list[dict] = []
        self.flush = lambda: None             # main() replaces it: results are saved after every suite

    def say(self, msg: str) -> None:
        print(msg, file=sys.stderr, flush=True)

    def run_cell(self, suite: str, name: str, make, seeds: list[int | None], greedy: bool, **meta) -> dict:
        """One cell: a warm-up request, then one request per seed. Greedy cells whose runs disagree by more than
        15% are run again (up to --retries times), because every greedy run decodes the same text."""
        path, body = make(seeds[0])
        post_stream(self.base, path, body)                                            # warm-up
        attempts = []
        for attempt in range(1 + self.a.retries):
            runs = []
            for seed in seeds:
                path, body = make(seed)
                runs.append(post_stream(self.base, path, body))
            attempts.append(runs)
            tps = [r["decode_tps"] for r in runs if r.get("decode_tps")]
            if not (greedy and slow_runs(tps)):
                break
            self.say(f"    {name}: runs disagree ({min(tps):.1f} vs {max(tps):.1f} tok/s), measuring again")
        runs = max(attempts, key=lambda rs: statistics.median([r["decode_tps"] or 0 for r in rs]))
        cell = {"suite": suite, "cell": name, "greedy": greedy, **meta, **summarize(runs),
                "slow_runs_flagged": greedy and slow_runs([r["decode_tps"] for r in runs if r.get("decode_tps")]),
                "attempts": len(attempts), "sample": runs[0]["text"][:200],
                "text_sha256": [r["text_sha256"] for r in runs]}
        self.cells.append(cell)
        if cell["decode_tps_median"] is None:
            self.say(f"  {suite:<11} {name:<26}  no text streamed: check the reply ({runs[0]['tokens']} tokens)")
            return cell
        self.say(f"  {suite:<11} {name:<26} {cell['decode_tps_median'] or 0:7.1f} tok/s"
                 f"   (min {cell['decode_tps_min'] or 0:.1f}, max {cell['decode_tps_max'] or 0:.1f},"
                 f" ttft {cell['ttft_s_median'] or 0:.2f} s{', acc ' + str(cell['draft_acceptance']) if cell.get('draft_acceptance') else ''})")
        return cell

    # --------------------------------------------------------------------------------------------- suites ---

    def standard(self):
        """TensorFold's protocol (tools/bench_openai.py): 64 tokens, ignore_eos, one warm-up, seeds 1234-1238."""
        m, seeds = self.model, SEEDS[: self.a.reps]
        for greedy, samp in ((False, SAMPLED), (True, GREEDY)):
            mode = "greedy" if greedy else "sampled"
            self.run_cell("standard", f"code-{mode}", lambda s: completion(m, CODE, 64, samp, s), seeds, greedy)
            self.run_cell("standard", f"chat-{mode}", lambda s: chat(m, CHAT, 64, samp, s), seeds, greedy)

    def codechat(self):
        """The standard code prompt as a chat turn, thinking off: code a coding assistant writes. TensorFold 0.3.5 sends
        /v1/completions to the model as raw text (0.3.4 applied the chat template), so the standard code cells now
        time the model continuing the prompt, mostly a think block, rather than writing code."""
        m, seeds = self.model, SEEDS[: self.a.reps]
        for greedy, samp in ((False, SAMPLED), (True, GREEDY)):
            mode = "greedy" if greedy else "sampled"
            self.run_cell("codechat", f"code-chat-{mode}", lambda s: chat(m, CODE, 64, samp, s), seeds, greedy)

    def long(self):
        m, seeds = self.model, SEEDS[: max(3, self.a.reps - 2)]
        for kind, prompt in LONG_PROMPTS:
            for greedy, samp in ((True, GREEDY), (False, SAMPLED)):
                mode = "greedy" if greedy else "sampled"
                self.run_cell("long", f"{kind}-{mode}-512", lambda s, p=prompt: chat(m, p, 512, samp, s), seeds, greedy)

    def prefill(self):
        """Cold prompts: a fresh random salt at the start of each one, so no prefix cache can serve it."""
        sizes = [int(x) for x in self.a.prefill_sizes.split(",") if x]
        for size in sizes:
            if size + 64 > self.a.max_context:
                self.say(f"  prefill     {size:>6} tokens          skipped (server context {self.a.max_context})")
                continue
            reps, rows = 3, []
            for i in range(reps + 1):                                              # first one is a warm-up
                salt = f"[session {random.getrandbits(64):016x}] "
                words = int(size * 0.72)                                            # ~1.35 tokens a word
                body_text = (FILLER * (words // len(FILLER.split()) + 1))
                body_text = " ".join(body_text.split()[:words])
                prompt = salt + body_text + "\n\nSummarize the report above in one sentence."
                path, body = chat(self.model, prompt, 16, GREEDY, None, ignore_eos=True)
                try:
                    r = post_stream(self.base, path, body)
                except urllib.error.HTTPError as e:
                    self.say(f"  prefill     {size:>6} tokens          HTTP {e.code}: {e.read()[:160]!r}")
                    break
                if i:
                    rows.append(r)
            if not rows:
                continue
            ttft = statistics.median(r["ttft_s"] for r in rows)
            ptoks = rows[0]["prompt_tokens"] or size
            cell = {"suite": "prefill", "cell": f"prefill-{size}", "prompt_tokens": ptoks,
                    "ttft_s_median": round(ttft, 3), "prefill_tps": round(ptoks / ttft, 1),
                    "cached_tokens": rows[0].get("cached_tokens")}
            self.cells.append(cell)
            self.say(f"  prefill     {ptoks:>6} tokens          {cell['prefill_tps']:7.1f} tok/s prefill"
                     f"   (ttft {ttft:.2f} s)")

    def exactness(self):
        """TensorFold's guarantee, checked end to end: drafted replies equal serial ("draft": false) replies."""
        m, results = self.model, []
        prompts = [("code", lambda s, d: completion(m, CODE, 96, s, d)),
                   ("chat", lambda s, d: chat(m, CHAT, 96, s, d)),
                   ("json", lambda s, d: chat(m, LONG_PROMPTS[2][1], 96, s, d))]
        for name, make in prompts:
            for label, samp, seed in (("greedy", GREEDY, None), ("seed1234", SAMPLED, 1234), ("seed1235", SAMPLED, 1235)):
                p1, b1 = make(samp, seed)
                drafted = post_stream(self.base, p1, b1)
                serial = post_stream(self.base, p1, {**b1, "draft": False})
                same = drafted["text"] == serial["text"]
                results.append({"prompt": name, "sampling": label, "identical": same,
                                "drafted_tps": drafted["decode_tps"], "serial_tps": serial["decode_tps"]})
        passed = sum(r["identical"] for r in results)
        speedup = statistics.median(r["drafted_tps"] / r["serial_tps"] for r in results
                                    if r["drafted_tps"] and r["serial_tps"])
        self.cells.append({"suite": "exactness", "cell": "drafted-vs-serial", "passed": passed,
                           "total": len(results), "draft_speedup_median": round(speedup, 2), "runs": results})
        mark = "PASS" if passed == len(results) else "FAIL"
        self.say(f"  exactness   drafted == serial           {passed}/{len(results)} {mark}"
                 f"   (drafts {speedup:.2f}x faster than serial)")

    def context(self):
        """Long context, the way a coding agent uses it. For each size: a cold prompt of synthetic source code
        (salted, so no prefix cache helps) with a fact hidden in the middle, then a follow-up turn that appends a
        short question to the same conversation. Measures time to first token (prefill tok/s), decode speed at
        that depth, whether the hidden fact was found, and how much a follow-up turn reuses the cache."""
        sizes = [int(x) for x in self.a.context_sizes.split(",") if x]
        ratio = self.chars_per_token()
        self.say(f"  context     tokenizer: {ratio:.2f} characters a token in the synthetic code")
        for size in sizes:
            if size + 1024 > self.a.max_context:
                self.say(f"  context     {size:>7} tokens         skipped (server context {self.a.max_context})")
                continue
            rng = random.Random(size)
            secret = f"{rng.choice(['ORCA', 'LYNX', 'KITE', 'MOTH'])}-{rng.randint(1000, 9999)}"
            code = synthetic_codebase(int(size * ratio), secret, rng)
            question = ("\n\nYou have the whole repository above. First answer on one line: what is the value of "
                        "DEPLOY_CODE? Then explain the repository's architecture module by module.")
            messages = [{"role": "user", "content": code + question}]
            body = {"model": self.model, "messages": messages, "max_tokens": 256, "temperature": 0,
                    "chat_template_kwargs": NO_THINK, "ignore_eos": True}
            try:
                cold = post_stream(self.base, "/v1/chat/completions", body, timeout=7200)
            except urllib.error.HTTPError as e:
                self.say(f"  context     {size:>7} tokens         HTTP {e.code}: {e.read()[:200]!r}")
                continue
            found = secret in cold["text"]
            # The follow-up turn: the conversation so far plus one short new message.
            follow = {**body, "messages": messages + [
                {"role": "assistant", "content": cold["text"]},
                {"role": "user", "content": "Now list the three largest modules by line count, briefly."}]}
            warm = post_stream(self.base, "/v1/chat/completions", follow, timeout=7200)
            pt = cold["prompt_tokens"] or size
            cell = {"suite": "context", "cell": f"context-{size}", "prompt_tokens": pt,
                    "ttft_s": round(cold["ttft_s"], 2), "prefill_tps": round(pt / cold["ttft_s"], 1),
                    "decode_tps_median": round(cold["decode_tps"], 2) if cold["decode_tps"] else None,
                    "needle_found": found, "followup_ttft_s": round(warm["ttft_s"], 2),
                    "followup_prompt_tokens": warm["prompt_tokens"], "followup_cached_tokens": warm["cached_tokens"],
                    "followup_decode_tps": round(warm["decode_tps"], 2) if warm["decode_tps"] else None,
                    "sample": cold["text"][:200]}
            self.cells.append(cell)
            self.flush()
            self.say(f"  context     {pt:>7} tokens  prefill {cell['prefill_tps']:7.1f} tok/s (ttft {cold['ttft_s']:.1f} s)"
                     f"  decode {cell['decode_tps_median'] or 0:5.1f} tok/s  needle {'found' if found else 'MISSED'}"
                     f"  follow-up ttft {warm['ttft_s']:.1f} s")

    def chars_per_token(self) -> float:
        """Measure the server's tokenizer on the synthetic code once, so sizes land near their token targets."""
        sample = synthetic_codebase(40000, "X-0", random.Random(0))
        path, body = chat(self.model, sample, 1, GREEDY, None, ignore_eos=True)
        r = post_stream(self.base, path, body)
        return len(sample) / r["prompt_tokens"] if r.get("prompt_tokens") else 3.2

    def concurrency(self):
        m = self.model
        for n in [int(x) for x in self.a.concurrency.split(",") if x]:
            prompts = [chat(m, LONG_PROMPTS[1][1], 256, GREEDY, None) for _ in range(n)]
            post_stream(self.base, *prompts[0])
            t = time.perf_counter()
            with cf.ThreadPoolExecutor(n) as ex:
                runs = list(ex.map(lambda pb: post_stream(self.base, *pb), prompts))
            wall = time.perf_counter() - t
            ok = [r for r in runs if r.get("decode_tps") and r.get("ttft_s") is not None]
            failed = len(runs) - len(ok)
            per = statistics.median(r["decode_tps"] for r in ok) if ok else 0.0
            agg = sum(r.get("tokens") or 0 for r in ok) / wall
            self.cells.append({"suite": "concurrency", "cell": f"streams-{n}", "streams": n, "failed": failed,
                               "decode_tps_median": round(per, 2), "aggregate_tps": round(agg, 2),
                               "ttft_s_median": round(statistics.median(r["ttft_s"] for r in ok), 3) if ok else None})
            note = f", {failed} FAILED" if failed else ""
            self.say(f"  concurrency {n} streams                  {per:7.1f} tok/s per stream, {agg:.1f} aggregate{note}")


# ------------------------------------------------------------------------------------------------- main ---

def git_rev() -> str | None:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, timeout=5).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("url", help="server base URL, e.g. http://127.0.0.1:8080")
    p.add_argument("--model", default="", help="model id (default: the first one /v1/models lists)")
    p.add_argument("--model-suffix", default="", help="appended to the model id, e.g. '@c3:0.35' (GLM draft policy)")
    p.add_argument("--recipe", default="adhoc", help="results go to results/<recipe>/")
    p.add_argument("--label", default="", help="a variant name kept in the results (e.g. 'policy c3:0.35')")
    p.add_argument("--suites", default="standard", help="comma list: standard,codechat,long,prefill,exactness,concurrency,context")
    p.add_argument("--reps", type=int, default=5, help="seeds per cell (standard: 5, like the published numbers)")
    p.add_argument("--retries", type=int, default=1, help="re-measure greedy cells whose runs disagree >15%%")
    p.add_argument("--prefill-sizes", default="1024,4096,16384,32768")
    p.add_argument("--max-context", type=int, default=32768, help="skip prefill sizes the server cannot hold")
    p.add_argument("--concurrency", default="1,2,4")
    p.add_argument("--context-sizes", default="32768,131072",
                   help="token sizes for the context suite (skipped above --max-context)")
    p.add_argument("--meta", action="append", default=[], help="key=value stored with the results")
    p.add_argument("--output", help="results file (default: results/<recipe>/<time>-<host>.json)")
    a = p.parse_args()

    base = a.url.rstrip("/")
    models = get_json(base, "/v1/models")
    if not models:
        sys.exit(f"no server answers at {base}/v1/models. Is it running? (./status.sh)")
    if not a.model:
        a.model = models["data"][0]["id"]

    b = Bench(a)
    started = dt.datetime.now(dt.timezone.utc)
    meta = dict(kv.split("=", 1) for kv in a.meta)
    out = Path(a.output) if a.output else (
        ROOT / "results" / a.recipe / f"{started.strftime('%Y%m%d-%H%M%S')}"
        f"{'-' + re.sub(r'[^A-Za-z0-9._-]+', '_', a.label).strip('_') if a.label else ''}.json")
    out.parent.mkdir(parents=True, exist_ok=True)

    def flush() -> None:
        result = {
            "schema": 1, "recipe": a.recipe, "label": a.label, "model": b.model, "url": public_url(base),
            "started_utc": started.isoformat(timespec="seconds"),
            "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "client": {"python": platform.python_version()},
            "recipes_git": git_rev(), "meta": meta, "cells": b.cells,
        }
        out.write_text(json.dumps(result, indent=1) + "\n")

    b.flush = flush
    b.say(f"\n{a.recipe}{' [' + a.label + ']' if a.label else ''}: {b.model} at {base}\n")
    for suite in [s.strip() for s in a.suites.split(",") if s.strip()]:
        getattr(b, suite)()
        flush()
    flush()
    b.say(f"\nsaved {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}"
          f"\ncompare: python3 bench/compare.py\n")

if __name__ == "__main__":
    main()
