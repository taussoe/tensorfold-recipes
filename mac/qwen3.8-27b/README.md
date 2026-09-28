# Qwen3.8-27B on a Mac (TensorFold, MLX + Metal)

Qwen3.8-27B, 4-bit, with the DFlash2 draft model, served by TensorFold on Apple Silicon. On M5 GPUs it uses the
tensor-unit lane kernels; on M1 to M4 (this recipe's first target is an **M3 Max, 128 GB**) it uses TensorFold's
row-exact lane decoder and simdgroup matmul. Either way drafted replies are byte-identical to serial decoding.

| | |
| --- | --- |
| Checkpoint | [`Vontra/Qwen3.8-27B-MLX-4bit`](https://huggingface.co/Vontra/Qwen3.8-27B-MLX-4bit), 16.1 GB |
| Draft model | [`z-lab/Qwen3.8-27B-DFlash2`](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2), 3.8 GB |
| Mac | Apple Silicon, 32 GB or more, macOS with Python 3.11+ (or `uv`) |
| API | `http://127.0.0.1:8080/v1`, model id `Qwen3.8-27B` |
| Context | 65,536 in this recipe (`CONTEXT`); the model supports 262,144 |

## Run it

```bash
./setup.sh      # .venv-mac with the pinned TensorFold; checks chip, memory, power
./pull.sh       # 20 GB into ~/.cache/huggingface
./start.sh      # background server, log in .run/server.log   (./start.sh --fg: in this terminal)
./chat.sh "Write a Python function that merges two sorted lists."
./bench.sh
./stop.sh
```

Use it from any OpenAI client: base URL `http://127.0.0.1:8080/v1`, any API key, model `Qwen3.8-27B`.
To reach it from other machines: `HOST=0.0.0.0 ./start.sh`.

## What to expect

TensorFold's published numbers on an **M3 Ultra** (64 tokens, median of seeds, thinking off):

| | Code, sampled | Chat, sampled | Code, greedy | Chat, greedy |
| --- | ---: | ---: | ---: | ---: |
| Serial (no drafts) | 38.2 | 38.2 | 39.3 | 39.3 |
| 0.3.4 with DFlash2 | 141.3 | 73.9 | 158.4 | 74.2 |

An M3 Max (40-core GPU) has half an M3 Ultra's memory bandwidth (400 against 800 GB/s) and half its GPU cores.
**Measured with this recipe** on an M3 Max, 128 GB, plugged in (27 September 2026, `./bench.sh`,
[results/qwen3.8-27b-mac/](../../results/qwen3.8-27b-mac/)):

| | Code, sampled | Chat, sampled | Code, greedy | Chat, greedy |
| --- | ---: | ---: | ---: | ---: |
| M3 Max, **High Power** energy mode, TensorFold 0.3.4 | **90.5** | 46.4 | **87.0** | 46.0 |
| M3 Max, Automatic energy mode, TensorFold 0.3.4 | 53.3 | 26.9 | 52.1 | 26.8 |
| M3 Max, High Power, TensorFold 0.3.5.1 (28 September) | 72.6 | **46.9** | 66.4 | **50.4** |

TensorFold 0.3.5 sends `/v1/completions` to the model as raw text (0.3.4 applied the chat template), so the code
cells above no longer time the same text: on 0.3.5 the model continues the prompt, mostly with a think block. The
same code prompt as a chat turn (`./bench.sh --suites codechat`), measured the same day on this Mac:

| Code written as a chat reply | Sampled | Greedy |
| --- | ---: | ---: |
| TensorFold 0.3.4 | 92.6 | 96.1 |
| **TensorFold 0.3.5.1** | **109.7** | **110.9** |

0.3.5.1 also reads prompts faster here: 211–218 tok/s up to 14k tokens, 197 at 28k (a 33k prompt: first token
after 186 s, hidden fact found).

| 512-token replies | Code, greedy | Prose, greedy | JSON, greedy | Prose, sampled |
| --- | ---: | ---: | ---: | ---: |
| M3 Max, High Power | **67.1** | **35.8** | **96.3** | **36.3** |
| M3 Max, Automatic | 48.3 | 27.9 | 35.1 | 8.4 |

- **High Power mode is worth 1.4–4.3×.** In Automatic mode the same request's rounds took anywhere from 70 to
  212 ms as the Mac throttled under sustained load; in High Power they stay steady. `start.sh` warns when it is off.
- Exactness: 9/9 drafted replies byte-identical to serial ones; drafts made decoding 4.0× faster than serial.
- Code and JSON draft well (61–84% acceptance), prose less (27–36%).
- Cold prefill: 100–150 tok/s, so a 14k-token prompt waits ~107 s before the first token. Follow-up turns reuse
  the prompt cache and only prefill what is new. For big cold prompts a DGX Spark is the better machine.
- At load TensorFold timed its verify windows here: 1 row 69 ms, 2–8 rows 93–106 ms, 9–16 rows 142–162 ms.

## Getting the most out of the Mac

- **Plugged in, High Power.** The single biggest setting: 1.4–4.3× measured on an M3 Max (above). On a 16" MacBook
  Pro: System Settings → Battery → Energy Mode → High Power, or `sudo pmset -a powermode 2`. On battery a Mac runs
  about 3× slower. `setup.sh` and `start.sh` warn about battery and Low
  Power Mode.
- **Quiet machine.** Close GPU-heavy apps (browsers with video, other model servers). A laptop also slows 20–40%
  after long, heavy decoding (heat), so benchmark cool and compare like with like.
- **Memory.** The model plus drafter take ~20 GB; the KV cache grows with `CONTEXT`. 65,536 leaves most of 128 GB
  free. `--prompt-cache-gib` (default: an eighth of RAM, at most 16) holds recent conversation prefixes so agent
  turns only prefill what's new.
- **Thinking.** Qwen3.8 thinks by default. For quick answers send `"chat_template_kwargs": {"enable_thinking": false}`,
  or cap it: `SERVE_ARGS="--thinking-budget 2048" ./start.sh`, or `--reasoning-effort low`.

## Settings (`recipe.env`)

| Variable | Default | Notes |
| --- | --- | --- |
| `CONTEXT` | `65536` | prompt plus reply; empty = the model's 262,144 |
| `PORT` | `8080` | |
| `SERVE_ARGS` | empty | e.g. `--thinking-budget 2048`, `--no-drafts` (serial reference), `--prompt-cache-gib 24` |

Source: [TensorFold's Qwen3.8-27B recipe](https://github.com/ashhart/TensorFold/blob/main/docs/recipes/qwen3.8-27b.md#macs-without-tensor-units-m1-to-m4).
