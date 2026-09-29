#!/usr/bin/env python3
"""Where an agent session's time went, from a recording (`record.py` or `replay.py`); two recordings side by side.

    python3 bench/report.py recordings/task.jsonl [recordings/task-after.jsonl]

Wall time splits into time the server worked on at least one request and time it had none (the agent's own work:
tools, browsing, its own thinking between calls). Server time splits into reading prompts (and how many prompt tokens
it read afresh rather than resumed) and writing replies (and how much of that was thinking). Calls are grouped by the
start of their system prompt, so the main agent, sub-agents and utility calls show up apart.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path


def load(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return sorted((r for r in rows if r.get("status") == 200), key=lambda r: r["start"])


def label(body: dict | None) -> str:
    """The call's kind: the first words of its system prompt (or of its first message)."""

    msgs = (body or {}).get("messages") or []
    for m in msgs:
        c = m.get("content")
        if isinstance(c, list):
            c = " ".join(p.get("text", "") for p in c if isinstance(p, dict))
        if isinstance(c, str) and c.strip():
            words = " ".join(c.split()[:7])
            return (m.get("role", "?")[:1] + ": " + words)[:60]
    return "(completion)"


def busy(rows: list[dict]) -> float:
    """Seconds with at least one request in flight (the union of their intervals)."""

    spans = sorted((r["start"], r["start"] + r["seconds"]) for r in rows)
    total, cur_a, cur_b = 0.0, None, None
    for a, b in spans:
        if cur_b is None or a > cur_b:
            if cur_b is not None:
                total += cur_b - cur_a
            cur_a, cur_b = a, b
        else:
            cur_b = max(cur_b, b)
    return total + ((cur_b - cur_a) if cur_b is not None else 0.0)


def summary(rows: list[dict]) -> dict:
    s = defaultdict(float)
    if not rows:
        return s
    s["calls"] = len(rows)
    s["wall"] = max(r["start"] + r["seconds"] for r in rows) - rows[0]["start"]
    s["busy"] = busy(rows)
    s["idle"] = s["wall"] - s["busy"]
    most = 1
    for r in rows:
        st, us = r.get("stats") or {}, r.get("usage") or {}
        prompt, cached = int(us.get("prompt_tokens") or 0), int(st.get("cached") or 0)
        s["prompt_tokens"] += prompt
        s["read_afresh"] += max(0, prompt - cached)
        s["prefill_s"] += float(st.get("prefill_s") or 0)
        s["decode_s"] += float(st.get("decode_s") or 0)
        s["reply_tokens"] += int(us.get("completion_tokens") or 0)
        s["thinking_chars"] += int(r.get("reasoning_chars") or 0)
        s["answer_chars"] += int(r.get("content_chars") or 0)
        s["images"] += sum(1 for m in (r.get("body") or {}).get("messages") or [] if isinstance(m.get("content"), list)
                           for p in m["content"] if isinstance(p, dict) and "image" in str(p.get("type")))
        s["queued_s"] += max(0.0, r["seconds"] - float(st.get("prefill_s") or 0) - float(st.get("decode_s") or 0))
        overlap = sum(1 for o in rows if o is not r and o["start"] <= r["start"] < o["start"] + o["seconds"])
        most = max(most, overlap + 1)
    s["most_at_once"] = most
    return s


def groups(rows: list[dict]) -> dict:
    g = defaultdict(list)
    for r in rows:
        g[label(r.get("body"))].append(r)
    return g


def fmt_min(sec: float) -> str:
    return f"{sec / 60:5.1f} min" if sec >= 90 else f"{sec:5.1f} s  "


def show(paths: list[Path]) -> None:
    sets = [load(p) for p in paths]
    sums = [summary(rows) for rows in sets]
    heads = [p.stem[:22] for p in paths]
    print(f"{'':34}" + "".join(f"{h:>24}" for h in heads))
    lines = [
        ("calls", lambda s: f"{int(s['calls'])}"),
        ("wall time", lambda s: fmt_min(s["wall"])),
        ("  server working", lambda s: f"{fmt_min(s['busy'])} {100 * s['busy'] / max(s['wall'], 1e-9):3.0f}%"),
        ("  agent / tools (server idle)", lambda s: f"{fmt_min(s['idle'])} {100 * s['idle'] / max(s['wall'], 1e-9):3.0f}%"),
        ("prompt reading", lambda s: fmt_min(s["prefill_s"])),
        ("  prompt tokens (read afresh)", lambda s: f"{int(s['prompt_tokens']):,} ({int(s['read_afresh']):,})"),
        ("decoding", lambda s: fmt_min(s["decode_s"])),
        ("  reply tokens", lambda s: f"{int(s['reply_tokens']):,}"),
        ("  thinking / answer chars", lambda s: f"{int(s['thinking_chars']):,} / {int(s['answer_chars']):,}"),
        ("queue, images, overhead", lambda s: fmt_min(s["queued_s"])),
        ("images", lambda s: f"{int(s['images'])}"),
        ("most calls at once", lambda s: f"{int(s['most_at_once'])}"),
    ]
    for name, f in lines:
        print(f"{name:34}" + "".join(f"{f(s):>24}" for s in sums))
    for path, rows in zip(paths, sets):
        print(f"\n{path.name}: by kind of call (the start of its prompt)")
        print(f"  {'calls':>5} {'read s':>7} {'afresh':>9} {'decode s':>8} {'think ch':>9}  kind")
        for name, rs in sorted(groups(rows).items(), key=lambda kv: -sum(r["seconds"] for r in kv[1])):
            s = summary(rs)
            print(f"  {len(rs):5d} {s['prefill_s']:7.1f} {int(s['read_afresh']):9,} {s['decode_s']:8.1f} "
                  f"{int(s['thinking_chars']):9,}  {name}")
        slow = sorted(rows, key=lambda r: -r["seconds"])[:5]
        print("  slowest calls:")
        for r in slow:
            st, us = r.get("stats") or {}, r.get("usage") or {}
            print(f"    at {r['start']:7.1f}s  {r['seconds']:6.1f}s: {us.get('prompt_tokens', '?')} prompt "
                  f"({st.get('cached', '?')} resumed, read {float(st.get('prefill_s') or 0):.1f}s), "
                  f"{us.get('completion_tokens', '?')} reply ({float(st.get('decode_s') or 0):.1f}s), "
                  f"thinking {r.get('reasoning_chars', 0):,} chars  [{label(r.get('body'))[:40]}]")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    show([Path(p) for p in sys.argv[1:]])
