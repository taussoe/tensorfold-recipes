# Qwen3.8 Flash Next on 1 or 2× DGX Spark (TensorFold)

Qwen3.8 Flash Next: 512 routed experts, hyper-connections, sparse attention, hashed n-gram tables and an MTP
head, 4-bit. TensorFold drafts up to 6 tokens a round with the MTP head and verifies them in one CUDA graph.

| | |
| --- | --- |
| Checkpoint | [`Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP`](https://huggingface.co/Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP), 113 GB |
| Machines | 2 Sparks by default (40.7 GB of weights each); `NODES=1` for one (80.4 GB) |
| API | `http://<spark1>:8080/v1`, model id `Qwen3.8-Flash-Next` |
| Context | what fits (TensorFold 0.3.5): 157,607 tokens on one Spark, the full 262,144 on two; 128k prompts measured on both |
| Start | about 80 s on two Sparks, 90 s on one |

## Run it

```bash
./setup.sh && ./pull.sh
./start.sh                 # two Sparks
NODES=1 ./start.sh         # one Spark
./chat.sh "Give me three names for a coffee shop run by robots."
./bench.sh
./stop.sh
```

## What to expect

Published by TensorFold (decode tok/s, one stream, 64 tokens, median of 5 seeds):

| | Code, sampled | Chat, sampled | Code, greedy | Chat, greedy |
| --- | ---: | ---: | ---: | ---: |
| TensorFold, 1 Spark | 68.3 | 58.5 | 73.1 | 60.2 |
| TensorFold, 2 Sparks | 103.8 | 84.0 | 96.2 | 100.2 |
| vLLM NVFP4 + MTP=3, 1 Spark | 42.4 | 33.2 | 40.9 | 37.6 |
| vLLM NVFP4 + MTP=3, 2 Sparks (TP2 + EP) | 46.4 | 41.4 | 55.2 | 50.7 |

This is the fastest model in the set: each token reads only the experts it routes to.

### Measured: TensorFold 0.3.5.1 (engine/)

Measured with this repo's benchmark on 28 September 2026. Flash Next runs TensorFold 0.3.5.1's own code in our
`engine/` branch (only GLM has changes there): its prompt path reads the 4-bit weights with FP8 tensor cores, and
drafted replies still equal serial ones (9/9).

| | Context | Prompt reading, 32k | First token, 32k | Prompt reading, 128k | First token, 128k | Decode, standard cells | Decode at 32k / 128k |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TensorFold 0.3.4, 1 Spark | 8k | 104 tok/s | 318 s | – | – | 67 / 60 / 72 / 59 | 52.6 |
| Our 0.3.4-based branch, 1 Spark | 40k | 634 tok/s | 52 s | – | – | 65 / 57 / 72 / 59 | 49.5 |
| **0.3.5.1, 1 Spark** | 157k | **1,901 tok/s** | **17 s** | **1,656 tok/s** | **79 s** | 75 / 61 / 75 / 73 | 38.9 / 40.0 |
| **0.3.5.1, 2 Sparks** | 262k | **2,450 tok/s** | **13.5 s** | **2,031 tok/s** | **65 s** | **100 / 91 / 91 / 88** | 59.0 / 51.9 |
| **0.3.6.1, 1 Spark** | 157k | **1,890 tok/s** | **17.5 s** | **1,800 tok/s** | **73 s** | 75 / 60 / 75 / 72 | 30.5 / 40.0 |
| **0.3.6.1, 2 Sparks** | 262k | **2,599 tok/s** | **12.8 s** | **2,258 tok/s** | **58 s** | **103 / 90 / 91 / 86** | 57.7 / 51.5 |
| **0.3.6.2, 1 Spark** | 157k | 1,772 tok/s | 18.7 s | 1,660 tok/s | 79 s | 71 / 59 / 74 / 73 | 39.3 / 40.0 |
| **0.3.6.2, 2 Sparks** | 262k | 2,466 tok/s | 13.4 s | 2,041 tok/s | 64 s | **104 / 91 / 92 / 89** | 58.9 / 52.3 |
| **0.3.6.3 + agent turns, 1 Spark** | 157k | 2,102 tok/s | 15.8 s | 1,840 tok/s | 71 s | 77 / 61 / 75 / 72 | 52.2 / 47.3 |
| vLLM + MTP (published, 1 Spark) | | 2,314 tok/s | | | | 42 / 33 / 41 / 38 | |

Every run found the hidden fact at 32k and 128k. Code written as a chat reply (`--suites codechat`): 0.3.6.1
80.0 / 81.2 tok/s on one Spark and 114.4 / 110.0 on two, 0.3.6.2 79.8 / 81.0 and 114.8 / 112.3 (sampled / greedy).

**Where a decode step goes** (one Spark, `engine/tools/profile_flashnext_decode.py`, 29 September 2026): 25.0 ms
for one row. The routed and shared experts read ~1.63 GB in 8.5 ms (~79% of the ~243 GB/s GB10 reads in practice),
the other 4-bit projections and the head ~2.1 GB in 9.8 ms (~88%), the GDN chain 1.2 ms. The hyper-connection mixes
(97 small down and up products of 2 MB) take 3.8 ms at ~43%: their launch settings that keep the bits save 0.2 ms
(`engine/tools/tune_flashnext_hc.py`), the rest is the cost of many small calls. Decoding is close to what the
memory reads allow.

Prompt reading moves with the machine's state from day to day, not only with the release: side by side on one
Spark on 29 September, 0.3.6.1 read a 36,870-token prompt at 2,096-2,138 tok/s and 0.3.6.2 at 2,089-2,123, a
138,467-token one at 1,656-1,665 and 1,652-1,655. Compare releases in one session.

**Several requests at once** (`NODES=1 PARALLEL=8 ./start.sh`; TensorFold 0.3.6 runs concurrent Flash Next streams on
one GPU). One Spark, 256-token prose replies, each stream with a 32k context, drafted == serial 9/9:

| Concurrent streams | Per stream | Together |
| --- | ---: | ---: |
| 1 | 54.0 tok/s | 52.3 tok/s |
| 2 | 48.5 | 93.1 |
| 4 | 36.3 | 137.6 |
| 8 | 24.1 | **181.3** (vLLM + MTP on one Spark, published: 163) |

## Agent turns

An agent (Glyph, a coding harness) sends each reply back without its reasoning. Qwen's template then renders that
turn's empty reasoning block as `<think>` and two newlines, one token, where the prompt before ended in `<think>` and
one newline: the next prompt parts from the kept one at its very last token. Flash Next resumes only from a whole kept
prompt, so in TensorFold 0.3.6.3 as released it reads every agent turn from the start (0 of 42 calls of a recorded Glyph
session resumed). Our `engine/` branch keeps the state before the prompt's last token instead (the 27B does the same
in 0.3.6.3); the prompt's state and reply stay byte-identical (CUDA tests on a Spark).

The same 42 calls of a recorded Glyph session (three sub-agents reviewing a repo, then the lead agent), replayed with
`bench/replay.py --chains` on one Spark, 29 September 2026:

| Setup | Session time | Prompt tokens read afresh | Prompt reading |
| --- | ---: | ---: | ---: |
| 0.3.6.2, one stream | 16.0 min | 1,154,408 of 1,154,408 | 9.3 min |
| 0.3.6.2, `PARALLEL=4` | 15.4 min | 1,154,408 | 9.0 min |
| [MiaAI-Lab's single-Spark setup](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold) (0.3.6.3, 5 streams, int8 cache) | 16.9 min | 1,154,408 | 9.0 min |
| **0.3.6.3 + agent turns, one stream** | 10.8 min | 507,455 | 4.0 min |
| **0.3.6.3 + agent turns, `PARALLEL=4`** | **7.6 min** | **115,222** | **67 s** |

One stream keeps one conversation's state: sub-agents that take turns push each other's out, so for parallel
sub-agents start with `NODES=1 PARALLEL=4 CONTEXT=65536 ./start.sh` (each stream keeps its own). A 32k-token turn then
reads its prompt in 0.5 s instead of 15 s.

MiaAI-Lab's setup decodes faster deep in a long prompt in `bench.py`'s context suite (62.6 / 56.5 tok/s at 32k / 128k,
ours 52.2 / 47.3); prompt reading and the standard cells are the same within a few percent.

## What matters for speed

- **Keep the n-gram tables in the page cache.** The checkpoint carries 32 GB of hashed n-gram tables that stay
  memory-mapped; each token reads 16 rows of them. If the kernel evicts them, every token waits ~8 ms on the
  disk. On one Spark (80 GB weights + 32 GB tables of 128) that is tight: stop everything else, and prefer two
  Sparks. Never drop caches while it serves.
- **MTP depth.** `--mtp-drafts N` (default 6 on CUDA; the chain also stops under 30% confidence). Compare with
  `SERVE_ARGS="--mtp-drafts 4" ./start.sh` then `./bench.sh --label mtp4`.
- The draft head reads a 79,591-token subset of the vocabulary (code and docs heavy). Text in rare scripts drafts
  less, never wrongly.

## Settings (`recipe.env`)

| Variable | Default | Notes |
| --- | --- | --- |
| `NODES` | `2` | `1` or `2` |
| `PARALLEL` | empty | `8`: up to 8 requests together (one Spark: `NODES=1`) |
| `SERVE_ARGS` | empty | e.g. `--mtp-drafts 4`, `--no-thinking` |

## Limits

- Long context: `CONTEXT=N ./start.sh` sizes the caches for N tokens. We measured 32k prompts (needle found,
  drafted == serial); larger windows are untested.
- One stream; requests queue.

Source: [TensorFold's Flash Next recipe](https://github.com/ashhart/TensorFold/blob/main/docs/recipes/qwen3.8-flash-next.md#dgx-spark-cuda).
