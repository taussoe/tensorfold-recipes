# Qwen3.8 Flash Next on a Mac (TensorFold, MLX + Metal)

Qwen3.8 Flash Next, 4-bit: 512 routed experts, hashed n-gram tables and an MTP head, served by TensorFold on
Apple Silicon. Its 80 GB of weights stay in memory; its 32 GB of n-gram tables are read from the SSD
(`--ple-on-ssd`, new in TensorFold 0.3.6), so it runs on a **128 GB Mac**. Drafted replies are byte-identical to
serial decoding.

| | |
| --- | --- |
| Checkpoint | [`Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP`](https://huggingface.co/Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP), 113 GB |
| Mac | Apple Silicon with 128 GB, plugged in, High Power energy mode; ~115 GB free on the SSD |
| API | `http://127.0.0.1:8080/v1`, model id `Qwen3.8-Flash-Next` |
| Context | 65,536 in this recipe (`CONTEXT`) |

## Run it

```bash
./setup.sh      # .venv-mac with the pinned TensorFold (shared with the 27B recipe)
./pull.sh       # 113 GB into ~/.cache/huggingface (about 20 minutes on a fast line, hours on Wi-Fi)
./start.sh      # background server, log in .run/server.log
./chat.sh "Write a Python function that merges two sorted lists."
./bench.sh
./stop.sh
```

One model at a time: stop the 27B recipe first (`../qwen3.8-27b/stop.sh`), both use port 8080.

## What to expect

**Measured with this recipe** on an M3 Max (40-core GPU, 128 GB), High Power, TensorFold 0.3.6.1, 28 September
2026 (`./bench.sh`, [results/qwen3.8-flash-next-mac/](../../results/qwen3.8-flash-next-mac/)):

| | Code, sampled | Chat, sampled | Code, greedy | Chat, greedy |
| --- | ---: | ---: | ---: | ---: |
| Standard cells (64 tokens, 5 seeds) | 71.8 | 69.6 | 63.6 | 68.1 |
| Code written as a chat reply (`--suites codechat`) | 75.8 | | 73.0 | |

| Prompt | Prompt reading | First token | Decode at that depth | Hidden fact |
| --- | ---: | ---: | ---: | :---: |
| 3,482 tokens | 567 tok/s | 6.1 s | | |
| 13,804 tokens | 544 tok/s | 25 s | | |
| 33,141 tokens (synthetic codebase) | 546 tok/s | 61 s | 60.5 tok/s | found |

Drafted == serial 9/9. Compared with the 27B on the same Mac, Flash Next reads prompts ~2.7x faster (546 against
~200 tok/s) and decodes chat faster (68-70 against 47-54), while the 27B with DFlash2 writes code faster (110
against 73-76 tok/s as a chat reply). Choose Flash Next for long prompts, the 27B for code-heavy replies.

## What matters for speed

- **High Power energy mode.** `start.sh` warns when it is off; Automatic throttled sustained decoding 1.4-4.3x on
  this Mac with the 27B.
- **Free memory.** TensorFold keeps the whole process inside 70% of RAM (89.6 GiB on 128 GB) and counts what other
  apps use. With ~14 GB used elsewhere it logged room for fewer concurrent streams; single requests were not
  affected. Close large apps for long prompts.
- **The SSD.** Each token reads 16 rows of the n-gram tables from disk. An internal SSD is assumed; an external
  disk would slow decoding.

## Settings (`recipe.env`)

| Variable | Default | Notes |
| --- | --- | --- |
| `CONTEXT` | `65536` | prompt + reply tokens; empty lets TensorFold fit what it can |
| `SERVE_ARGS` | `--ple-on-ssd` | drop it only with 192 GB or more of memory |

Source: [TensorFold's Flash Next recipe](https://github.com/ashhart/TensorFold/blob/main/docs/recipes/qwen3.8-flash-next.md).
