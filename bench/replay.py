#!/usr/bin/env python3
"""Send a recorded agent session (`record.py`) to a server again, with the agent's own pauses between calls, and
record what the server did: the same traffic before and after a server change.

    python3 bench/replay.py recordings/task.jsonl --target http://<spark1>:8080 --out recordings/task-replay.jsonl
    python3 bench/report.py recordings/task.jsonl recordings/task-replay.jsonl

A call that began while the one before it was still running (parallel sub-agents) starts at the same offset after
it again; any other call starts the recorded pause after the previous call ended, so a faster server finishes the
session sooner. The agent's replies are not re-run: a changed reply does not change the next request.

--chains replays each conversation (its first user message) as its own chain: a chain's calls follow one another
with the recorded pause after each (the agent's tools), whatever else was queued on the server when it was
recorded, and chains start at their recorded offsets; --from/--to keep the calls that began in that window.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import urllib.request
from pathlib import Path


def send(target: str, row: dict, started: float, out: Path, lock: threading.Lock) -> float:
    body = dict(row["body"] or {})
    body["stream"] = True
    body["stream_options"] = {"include_usage": True}
    req = urllib.request.Request(target + row["path"], json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.time()
    first, stats, usage, reasoning, content, finish = None, {}, {}, 0, 0, None
    with urllib.request.urlopen(req, timeout=24 * 3600) as r:
        for line in r:
            if first is None:
                first = time.time()
            line = line.strip()
            if not line.startswith(b"data: ") or line == b"data: [DONE]":
                continue
            try:
                d = json.loads(line[6:])
            except ValueError:
                continue
            stats = d.get("tensorfold") or stats
            usage = d.get("usage") or usage
            for ch in d.get("choices") or []:
                delta = ch.get("delta") or {}
                reasoning += len(delta.get("reasoning_content") or "")
                content += len(delta.get("content") or "")
                finish = ch.get("finish_reason") or finish
    end = time.time()
    out_row = {"start": t0 - started, "first_byte": (first or end) - t0, "seconds": end - t0, "path": row["path"],
               "status": 200, "body": row["body"], "stats": stats, "usage": usage, "reasoning_chars": reasoning,
               "content_chars": content, "finish": finish}
    with lock, out.open("a") as f:
        f.write(json.dumps(out_row) + "\n")
    print(f"{t0 - started:8.1f}s  {usage.get('prompt_tokens', '?'):>7} prompt ({stats.get('cached', '?')} resumed)  "
          f"{end - t0:6.1f}s", flush=True)
    return end


def chain_key(row: dict) -> str:
    msgs = (row.get("body") or {}).get("messages") or []
    users = [m for m in msgs if m.get("role") == "user"]
    c = users[0].get("content") if users else ""
    if isinstance(c, list):
        c = " ".join(p.get("text", "") for p in c if isinstance(p, dict))
    return " ".join(str(c).split()[:30])


def run_chains(rows: list[dict], a, lock: threading.Lock, started: float) -> int:
    chains: dict[str, list[dict]] = {}
    for r in rows:
        chains.setdefault(chain_key(r), []).append(r)
    t0 = min(r["start"] for r in rows)
    ends: dict[str, float] = {}

    def run(key: str, rs: list[dict]) -> None:
        time.sleep(max(0.0, rs[0]["start"] - t0 - (time.time() - started)))
        for i, row in enumerate(rs):
            send(a.target.rstrip("/"), row, started, a.out, lock)
            if i + 1 < len(rs) and not a.no_pauses:
                time.sleep(max(0.0, rs[i + 1]["start"] - (row["start"] + row["seconds"])))
        ends[key] = time.time() - started

    threads = [threading.Thread(target=run, args=(k, rs)) for k, rs in chains.items()]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for k, rs in chains.items():
        print(f"chain of {len(rs):3d} calls: began {rs[0]['start'] - t0:6.1f}s, ended {ends[k]:7.1f}s  | {k[:60]}")
    print(f"replayed {len(rows)} calls in {len(chains)} chains in {time.time() - started:.1f}s -> {a.out}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("recording", type=Path)
    p.add_argument("--target", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--no-pauses", action="store_true", help="send each call as soon as the one before it ends")
    p.add_argument("--chains", action="store_true", help="each conversation as its own chain (parallel sub-agents)")
    p.add_argument("--from", dest="t_from", type=float, default=0.0)
    p.add_argument("--to", dest="t_to", type=float, default=float("inf"))
    a = p.parse_args()
    rows = sorted((json.loads(l) for l in a.recording.read_text().splitlines() if l.strip()),
                  key=lambda r: r["start"])
    rows = [r for r in rows if r.get("status") == 200 and r.get("body") and a.t_from <= r["start"] <= a.t_to]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("")
    lock, started = threading.Lock(), time.time()
    if a.chains:
        return run_chains(rows, a, lock, started)
    threads: list[threading.Thread] = []
    prev = None                 # the previous recorded row and its thread / replayed start
    prev_start = started
    prev_thread = None
    result: dict[int, float] = {}
    for i, row in enumerate(rows):
        if prev is not None and row["start"] < prev["start"] + prev["seconds"]:      # began beside the previous one
            time.sleep(max(0.0, prev_start + (row["start"] - prev["start"]) - time.time()))
        elif prev_thread is not None:
            prev_thread.join()
            gap = 0.0 if a.no_pauses else row["start"] - (prev["start"] + prev["seconds"])
            time.sleep(max(0.0, gap))
        prev_start = time.time()

        def run(i=i, row=row):
            result[i] = send(a.target.rstrip("/"), row, started, a.out, lock)

        t = threading.Thread(target=run)
        t.start()
        threads.append(t)
        prev, prev_thread = row, t
    for t in threads:
        t.join()
    print(f"replayed {len(rows)} calls in {time.time() - started:.1f}s -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
