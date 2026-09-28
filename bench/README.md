# Benchmarking

One benchmark for every recipe and engine, so the numbers compare: TensorFold or vLLM, Spark or Mac.

```bash
cd dgx-spark/qwen3.8-27b && ./bench.sh          # from a recipe folder: the right URL, model id and metadata
python3 bench/compare.py                        # every recipe side by side → results/LEADERBOARD.md
```

`bench.py` uses only Python's standard library: it runs on a Spark host, in a container, or on a laptop
(`BENCH_URL=http://spark1.local:8080 ./bench.sh`). A full run takes 10–25 minutes.

## What it measures

**Decode tok/s** = (completion tokens − 1) / (time of last token − time of first token), from the stream: the
speed you see while text arrives, without prefill and the first token. TensorFold's and vLLM's own benchmarks use
the same span. **TTFT** is time to first token (prefill + first step).

| Suite | Cells | Why |
| --- | --- | --- |
| `standard` | code / chat × sampled / greedy, 64 tokens, seeds 1234–1238, median | TensorFold's exact published protocol (`tools/bench_openai.py`), so your numbers sit next to the published ones in [reference.json](reference.json). Short replies favour drafting; that is why the next suite exists |
| `codechat` | the standard code prompt as a chat turn, thinking off, sampled / greedy | code as an assistant writes it: from TensorFold 0.3.5 the standard code cells send raw text (`/v1/completions` without the chat template), so the model mostly continues with a think block there |
| `long` | code, prose, JSON × greedy / sampled, 512 tokens, 3 seeds | closer to real use: long answers, where acceptance drifts and caches grow |
| `prefill` | cold prompts of ~1k, 4k, 16k, 32k tokens (up to the recipe's context), 3 each | TTFT and prefill tok/s. A random salt at the start of every prompt defeats prefix caches, so this is the cold case (a coding agent's first turn); later turns reuse the cache |
| `exactness` | 3 prompts × (greedy, 2 seeds), drafted vs `"draft": false` | TensorFold only: its promise that drafts never change the output, checked end to end. Also shows what drafting buys over serial |
| `concurrency` | 1, 2, 4 parallel streams (opt-in: `BENCH_SUITES=concurrency`) | vLLM batches streams; TensorFold queues them. Per-stream and aggregate tok/s |

Thinking is switched off in every cell (`chat_template_kwargs`: `enable_thinking` / `thinking` false), and every
request sets `ignore_eos` so each reply has exactly the requested length.

## Reading the results

- **Medians over seeds.** A sampled cell decodes a different text per seed, and drafting speed depends on the
  text: 27B code on two Sparks ranged 70–136 tok/s over five seeds. Compare medians.
- **Greedy cells decode the same text every run**, so their runs should agree. When one run is under 85% of the
  best (GB10 page-migration bursts can halve a run), the cell is measured again (`--retries`, default 1). A `*`
  in the leaderboard means it still disagreed: run it again before believing it.
- **Draft acceptance** is printed when the server reports it (TensorFold's `speculative` field).
- Results are JSON in `results/<recipe>/<time>-<host>[-label].json`, with the recipe, engine, nodes, checkpoint
  and engine commit in `meta`.

## Options

```bash
./bench.sh --suites standard              # only the published protocol (4 cells, ~3 min)
BENCH_SUITES=standard,long ./bench.sh     # the same through the environment
./bench.sh --label "mtp4"                 # tag a variant; compare.py keeps variants apart
./bench.sh --model-suffix "@c3:0.35"      # GLM: a draft policy for every request
./bench.sh --reps 3 --retries 0           # quicker, noisier
./bench.sh --prefill-sizes 2048,8192
BENCH_SUITES=concurrency ./bench.sh --concurrency 1,2
python3 bench/bench.py --help
```

## Comparing

```bash
python3 bench/compare.py                           # newest run per recipe/label/platform + published numbers
python3 bench/compare.py --recipe glm-5.3-flash --all   # every GLM run over time
python3 bench/compare.py --csv > all.csv           # for a spreadsheet
./tools/bench-all.sh                               # from the repo root on Spark 1: start, bench, stop every model, then the leaderboard
```

## A fair comparison

- Nothing else running on the machine(s); after warm-up (the script sends one warm-up request per cell).
- The same prompts, seeds, reply length and client for every engine: that is what `bench.py` is for.
- Note power and thermal state on a Mac (plugged in, High Power, cool).
- One change at a time, labelled, and compare labelled runs from the same session.
