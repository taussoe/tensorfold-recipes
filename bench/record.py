#!/usr/bin/env python3
"""Record an agent's traffic to a TensorFold server: a small proxy in front of it.

Point the agent (Glyph, opencode, any OpenAI-compatible client) at the proxy instead of the server, run a real task,
and every request is written to a JSON-lines file with its timing and the server's own figures (prompt tokens,
tokens resumed, prompt reading and decoding time, thinking). `report.py` turns a recording into where the time
went; `replay.py` sends a recording to a server again, to measure a server change on the same traffic.

    python3 bench/record.py --target http://<spark1>:8080 --port 8090 --out recordings/task.jsonl
    # then set the agent's base URL to http://127.0.0.1:8090/v1

Recordings hold your prompts, code and screenshots: they stay on this machine (recordings/ is not committed).
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

LOCK = threading.Lock()


def make_handler(target: str, out: Path, started: float):
    class Proxy(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):  # quiet
            pass

        def _forward(self, method: str) -> None:
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else None
            headers = {k: v for k, v in self.headers.items()
                       if k.lower() not in ("host", "content-length", "connection", "accept-encoding")}
            req = urllib.request.Request(target + self.path, data=raw, method=method, headers=headers)
            t0 = time.time()
            first = None
            stats, usage, reasoning, content, finish = {}, {}, 0, 0, None
            try:
                resp = urllib.request.urlopen(req, timeout=24 * 3600)
                status = resp.status
            except urllib.error.HTTPError as err:
                resp, status = err, err.code
            self.send_response(status)
            for k, v in resp.headers.items():
                if k.lower() not in ("transfer-encoding", "connection", "content-length"):
                    self.send_header(k, v)
            self.send_header("Connection", "close")
            self.end_headers()
            buffer = b""
            try:
                while True:
                    chunk = resp.read1(65536) if hasattr(resp, "read1") else resp.read(65536)
                    if not chunk:
                        break
                    if first is None:
                        first = time.time()
                    self.wfile.write(chunk)
                    self.wfile.flush()
                    buffer += chunk
                    while b"\n" in buffer:
                        line, buffer = buffer.split(b"\n", 1)
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
            except (BrokenPipeError, ConnectionResetError):
                pass
            end = time.time()
            if buffer.strip().startswith(b"{"):          # a plain JSON reply (not streamed)
                try:
                    d = json.loads(buffer)
                    stats = d.get("tensorfold") or stats
                    usage = d.get("usage") or usage
                    for ch in d.get("choices") or []:
                        msg = ch.get("message") or {}
                        reasoning += len(msg.get("reasoning_content") or "")
                        content += len(msg.get("content") or "")
                        finish = ch.get("finish_reason") or finish
                except ValueError:
                    pass
            if method == "POST" and self.path.rstrip("/").endswith("completions"):
                try:
                    body = json.loads(raw or b"{}")
                except ValueError:
                    body = None
                row = {"start": t0 - started, "first_byte": (first or end) - t0, "seconds": end - t0,
                       "path": self.path, "status": status, "body": body, "stats": stats, "usage": usage,
                       "reasoning_chars": reasoning, "content_chars": content, "finish": finish}
                with LOCK, out.open("a") as f:
                    f.write(json.dumps(row) + "\n")
                p = stats.get("prefill_s")
                print(f"{t0 - started:8.1f}s  {usage.get('prompt_tokens', '?'):>7} prompt ({stats.get('cached', '?')} "
                      f"resumed)  prefill {p if p is None else round(p, 1)}s  {usage.get('completion_tokens', '?')} "
                      f"reply in {end - t0:.1f}s  thinking {reasoning} chars", flush=True)
            self.close_connection = True

        def do_GET(self):
            self._forward("GET")

        def do_POST(self):
            self._forward("POST")

    return Proxy


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target", required=True, help="the server, e.g. http://<spark1>:8080")
    p.add_argument("--port", type=int, default=8090)
    p.add_argument("--out", type=Path, required=True, help="JSON-lines file to append to")
    a = p.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(a.target.rstrip("/"), a.out, time.time()))
    print(f"recording to {a.out}: point the agent at http://127.0.0.1:{a.port}/v1 (Ctrl-C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
