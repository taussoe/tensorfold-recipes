# Qwen3.8-27B on 1 or 2× DGX Spark (TensorFold)

The dense Qwen3.8-27B (DeltaNet + full attention), 4-bit, with the DFlash2 draft model proposing token trees
that TensorFold verifies 12 rows at a time. Replies are byte-identical to serial decoding.

| | |
| --- | --- |
| Checkpoint | [`Vontra/Qwen3.8-27B-MLX-4bit`](https://huggingface.co/Vontra/Qwen3.8-27B-MLX-4bit), 16.1 GB |
| Draft model | [`z-lab/Qwen3.8-27B-DFlash2`](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2), 3.8 GB |
| Machines | 2 Sparks by default; `NODES=1` for one |
| API | `http://<spark1>:8080/v1`, model id `Qwen3.8-27B` |
| First start | 1–2 minutes (kernel compile); later starts are faster |

## Run it

```bash
./setup.sh && ./pull.sh
./start.sh                 # two Sparks, the fastest
NODES=1 ./start.sh         # one Spark (use NODES=1 for setup/pull too if Spark 2 is busy)
./chat.sh "Write a Python function that merges two sorted lists."
./bench.sh
./stop.sh
```

## What to expect

Published by TensorFold (decode tok/s, one stream, 64 tokens, median of 5 seeds):

| | Code, sampled | Chat, sampled | Code, greedy | Chat, greedy |
| --- | ---: | ---: | ---: | ---: |
| TensorFold, 1 Spark | 49.6 | 45.8 | 49.2 | 45.9 |
| TensorFold, 2 Sparks | 82.4 | 58.9 | 76.2 | 71.1 |
| vLLM NVFP4 + MTP=3, 1 Spark | 17.7 | 15.0 | 17.7 | 17.0 |
| vLLM NVFP4 + MTP=3, 2 Sparks | 33.1 | 30.4 | 31.6 | 30.5 |

Serial decoding (no drafts) is 13.1 tok/s on one Spark and 22.5 on two: the model reads its 16 GB of weights
every step at ~240 GB/s, so drafting is where the speed comes from. Code and file edits draft best, fresh prose
least.

Measured on our Sparks on 28 September 2026 with TensorFold 0.3.6.1 (this repo's benchmark, drafted == serial 9/9,
hidden fact found at 32k):

| | Standard cells | Code as a chat reply (`codechat`) | 32k prompt | Decode at 32k |
| --- | ---: | ---: | ---: | ---: |
| **2 Sparks, MLX 4-bit** | 82 / 71 / 82 / 72 | **142 / 164** | **1,945 tok/s** (17 s) | 63.5 |
| 1 Spark, MLX 4-bit | 59 / 52 / 56 / 52 | 96 / 111 | 1,579 tok/s (21 s) | 36.6 |
| 1 Spark, EXL3 3.00bpw (`turboderp/Qwen3.8-27B-exl3`) | 83 / 46 / 68 / 42 | 97 / 98 | 921 tok/s (36 s) | 40.0 |

(0.3.4: 80 / 57 / 73 / 69 and 721 tok/s at 32k on two Sparks.) The EXL3 pack is no faster where it counts here
(code as chat, chat cells, prompts) and runs on one Spark only, so this recipe keeps the MLX checkpoint. Serving it
needed a packaging fix (`qwen3_5/cuda/b16.cpp` was missing from the wheel), which our engine branch carries.

**Several requests at once** (`PARALLEL=8 ./start.sh`, up to 16). Two Sparks, 256-token prose replies, each stream with a
147k context, drafted == serial 9/9:

| Concurrent streams | Per stream | Together |
| --- | ---: | ---: |
| 1 | 64.7 tok/s | 63.7 tok/s |
| 2 | 50.4 | 99.0 |
| 4 | 39.7 | 155.1 |
| 8 | 30.6 | **236.2** |

## Which to choose: 1 or 2 Sparks?

Two Sparks give +30–65% (most on code). With `NODES=1` the model runs on Spark 1 only and Spark 2 stays free
for other work (`start.sh` then only checks Spark 1).

## Settings (`recipe.env`)

| Variable | Default | Notes |
| --- | --- | --- |
| `NODES` | `2` | `1` or `2` |
| `CONTEXT` | empty (the model's 262,144) | the capacity is fixed at start; a cap frees memory |
| `PARALLEL` | empty | `8`: up to 8 requests together (16 at most) |
| `SERVE_ARGS` | empty | e.g. `--thinking-budget 4096`, `--reasoning-effort low`, `--no-thinking` |

## Limits

- TensorFold 0.3.5 also serves concurrent requests for this model (shared drafting rounds); `./bench.sh`'s
  concurrency suite measures it.
- Prefix reuse keeps two states (the last prompt's and the last reply's): an agent that continues its
  conversation resumes; a different conversation starts over.

Source: [TensorFold's Qwen3.8-27B recipe](https://github.com/ashhart/TensorFold/blob/main/docs/recipes/qwen3.8-27b.md#dgx-spark-cuda).
