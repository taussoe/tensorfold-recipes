# GLM-5.3-Flash on 2× DGX Spark (TensorFold)

Z.ai's GLM-5.3-Flash, 4-bit, split over two Sparks with TensorFold's CUDA engine. Drafts with the model's own
MTP head, and optionally with a DFlash2 draft model; every reply is byte-identical to one-token-at-a-time
decoding, drafts only change the speed.

| | |
| --- | --- |
| Checkpoint | [`Vontra/GLM-5.3-Flash-MLX-4bit-MTP`](https://huggingface.co/Vontra/GLM-5.3-Flash-MLX-4bit-MTP), 182 GB (MIT) |
| Draft model (opt-in) | [`incoai/GLM-5.3-Flash-DFlash2`](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2), **CC BY-NC-ND 4.0, non-commercial** |
| Machines | 2 Sparks (one has 128 GB; the model needs 182). Each rank holds 90.8 GB |
| API | `http://<spark1>:8080/v1`, model id `GLM-5.3-Flash` |
| Context | 458,752 tokens (`CONTEXT`), tested with a 449k prompt; `CONTEXT=0` allocates what fits, 465,768 on our Sparks (our `engine/` branch's latent cache, image input on) |
| First start | up to ~13 minutes (kernel compile + load); later starts about 4–5 minutes |
| Images | yes, with our `engine/` branch: `image_url` parts in user and tool messages (see [Images](#images)) |

## Run it

```bash
./setup.sh      # builds the TensorFold image on both Sparks (once)
./pull.sh       # downloads 182 GB on Spark 1, writes each rank's 91 GB half, sends rank 1's half to Spark 2
./start.sh      # rank 1 on Spark 2, rank 0 on Spark 1; waits for /health
./chat.sh "Explain KV caches in two sentences."
./bench.sh
./stop.sh
```

With the DFlash2 draft model (only if its non-commercial license fits your use):

```bash
ACCEPT_NONCOMMERCIAL_DRAFTER=1 ./pull.sh
ACCEPT_NONCOMMERCIAL_DRAFTER=1 ./start.sh
```

or set `ACCEPT_NONCOMMERCIAL_DRAFTER=1` in `recipe.env`.

## What to expect

Published by TensorFold (decode tok/s, one stream, 64 tokens, median of 5 seeds; `./bench.sh`'s standard suite
reproduces this protocol):

| | Code, sampled | Chat, sampled | Code, greedy | Chat, greedy |
| --- | ---: | ---: | ---: | ---: |
| TensorFold, MTP + DFlash2 (`auto`) | 49.4 | 43.3 | 66.3 | 45.2 |
| TensorFold, MTP only (this recipe's default) | 49.3 | 43.2 | 57.8 | 37.2 |
| vLLM, Mia-AiLab EXL3, MTP=3 | 24.5 | 24.3 | 32.2 | 24.7 |

### Baseline: what we run today (vLLM, measured with this repo's benchmark)

Mia-AiLab's [GLM-5.3-Flash EXL3 recipe](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks) on our
two Sparks (vLLM, EXL3 4 bpw experts, fp8 KV, DFlash2 drafts, 850k context), measured on 27 September 2026 with
`tools/bench-baseline.sh` ([results/glm-5.3-flash/](../../results/glm-5.3-flash/)). This is the bar the
TensorFold recipe has to clear on the same benchmark.

| Standard cells (64 tokens, 5 seeds) | Code, sampled | Chat, sampled | Code, greedy | Chat, greedy |
| --- | ---: | ---: | ---: | ---: |
| vLLM, Mia EXL3, 850k context | 35.1 | 26.9 | 48.2* | 29.7 |

\* Unstable: greedy runs spanned 42–75 tok/s even after a re-measure.

| Long context (synthetic codebase, cold) | Wait for first token | Prompt reading speed | Writing speed at that depth | Hidden fact found | Follow-up turn, first token |
| --- | ---: | ---: | ---: | :---: | ---: |
| 32,763 tokens | 24.3 s | 1,350 tok/s | 28.8 tok/s | yes | 0.9 s |
| 130,832 tokens | 105.2 s | 1,243 tok/s | 34.9 tok/s | yes | 2.3 s |

| 512-token replies | Code, greedy | Code, sampled | Prose, greedy | Prose, sampled | JSON, greedy | JSON, sampled |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| vLLM, Mia EXL3, 850k context | 50.4 | 45.7 | 29.0 | 27.5 | 54.2 | 56.8 |

| Concurrent streams (256-token prose each) | Per stream | Together |
| --- | ---: | ---: |
| 1 | 29.8 tok/s | 27.2 tok/s |
| 2 | 22.5 tok/s | 41.5 tok/s |
| 4 | 17.2 tok/s | 61.6 tok/s |

vLLM's strength is the last table: four agents at once get 2.3x the total throughput of one. TensorFold serves one
request at a time today, so there a second agent waits.

### TensorFold with our engine/ branch, same benchmark

Measured on the same two Sparks on 28 September 2026, MTP drafts only (no DFlash2). `engine/` is our branch
`glm-long-context`: TensorFold 0.3.5.1 (its CUDA prefill matmuls and grouped CUDA experts) with our long-context
work for GLM on top:

- GLM's attention cache kept as its 512-wide latent (~1 KB a token a layer instead of ~16 KB per head set), with
  the query absorbed into kv_b and the value projection applied after attention, reading kv_b's 4-bit rows as
  stored. (On 0.3.5.1 prompts past 2,051 tokens answered "!!!!"; we reported it as #53 and Ash fixed it in 0.3.6 the
  same day.)
- DSA's token selection for all rows of a chunk at once, over the pools it can see, with the top 512 pools found
  by one radix-select kernel; CUDA graphs past the dense limit; attention tiles of all 32 heads in prompt chunks;
  KDA chains of prompt chunks in three kernels.
- Memory admission that sizes the latent cache (so `CONTEXT=0` finds 487k), and a prompt cache for several
  conversations.

| Standard cells (64 tokens, 5 seeds) | Code, sampled | Chat, sampled | Code, greedy | Chat, greedy |
| --- | ---: | ---: | ---: | ---: |
| **TensorFold, engine/** | 49.0 | **44.0** | **63.1** | 47.6 |
| TensorFold 0.3.5.1 as released (36k context) | 51.0 | 42.7 | 55.9 | 47.8 |
| vLLM baseline (above) | 35.1 | 26.9 | 48.2* | 29.7 |
| engine/ / vLLM | 1.40x | 1.64x | 1.31x | 1.60x |

| Long context | Prompt reading | Wait for first token | Writing speed at depth | Hidden fact | Follow-up turn |
| --- | ---: | ---: | ---: | :---: | ---: |
| **TensorFold engine/, 32,770 tokens** | 1,292 tok/s | 25 s | **45.3 tok/s** | yes | **0.5 s** |
| vLLM, 32,763 tokens | **1,350 tok/s** | **24 s** | 28.8 tok/s | yes | 0.9 s |
| **TensorFold engine/, 130,839 tokens** | 1,231 tok/s | 106 s | **44.2 tok/s** | yes | **0.7 s** |
| vLLM, 130,832 tokens | **1,243 tok/s** | **105 s** | 34.9 tok/s | yes | 2.3 s |
| **TensorFold engine/, 257,711 tokens** | 1,139 tok/s | 226 s | 42.8 tok/s | yes | 0.9 s |
| **TensorFold engine/, 449,088 tokens** (images on, `CONTEXT=0`) | 975 tok/s | 461 s | 38.1 tok/s | yes | 1.2 s |
| TensorFold 0.3.5.1 as released, 32,770 tokens | 626 tok/s | 52 s | – | no ("!!!!", fixed in 0.3.6: #53) | – |

Exactness: 9/9 drafted replies byte-identical to serial ones. The 256k run left at least 14 GB free on each Spark.
How the 128k case moved (the first rows are the 0.3.4-based branch):

| Build | Fits 128k | Prompt reading | Wait at 128k |
| --- | :---: | ---: | ---: |
| TensorFold 0.3.4 (per-head cache, 64-token chunks) | no (~143 GB a Spark) | – | – |
| + latent cache | yes | 202 tok/s | 647 s |
| + 1,024-token prefill chunks | yes | 493 tok/s | 265 s |
| + 4-bit absorb kernels, visible-pool token selection | yes | 535 tok/s | 245 s |
| + our MoE prefill kernels, 2,048-token chunks | yes | 577 tok/s | 227 s |
| + top-512 pools by radix select | yes | 703 tok/s | 186 s |
| + 32-head attention tiles, KDA chain in three kernels | yes | 739–756 tok/s | 173–177 s |
| all of it moved onto TensorFold 0.3.5.1 (its prefill matmuls and CUDA experts) | yes | 1,103–1,110 tok/s | 118 s |
| + absorb and expand of prompt chunks on one bf16 head GEMM | yes | **1,212–1,231 tok/s** | **106–108 s** |

TensorFold 0.3.5 computes prompt chunks with their own arithmetic (weights rounded once to bf16, one fp32 chain),
which is the same for any chunking but not decode's. Drafted replies still equal serial ones; a follow-up turn
resumes at the end of its previous prompt and prefills the previous reply again.

**Several requests at once** (`PARALLEL=4 ./start.sh`, each request with 64k of context). Our branch decodes the
streams together: one forward verifies every stream's MTP drafts, attention and KDA run on each stream's own state,
and every stream still equals its serial decoding (tested; drafted == serial 9/9 served). Same benchmark as vLLM's
row above (256-token prose replies, cold):

| Concurrent streams | vLLM, per stream | vLLM, together | **TensorFold, per stream** | **TensorFold, together** |
| --- | ---: | ---: | ---: | ---: |
| 1 | 29.8 | 27.2 | **45.9** | **44.0** |
| 2 | 22.5 | 41.5 | **40.0** | **76.2** |
| 4 | 17.2 | 61.6 | **32.8** | **122.4** |

With `PARALLEL` the rounds run without CUDA graphs and without DFlash2; a single stream still decodes at 49 / 44 /
63 / 48 tok/s and reads a 32k prompt at 1,184 tok/s.

**Several conversations.** The engine keeps up to 8 conversations' prompts (`TF_GLM_CACHE_ENTRIES`); when one
conversation takes the attention caches, the others' rows are saved (~20 KB a token). Everything the kept
conversations hold (their ~70 MB of KDA state each and their saved rows) stays within `TF_GLM_CACHE_GIB`, default 3.

Prompt reading is now within 1% of vLLM at 128k and 4% at 32k, and with `PARALLEL` TensorFold serves several requests at once too (above). Code written as a chat reply (`--suites codechat`): 55.0 sampled, 57.9 greedy.

## Images

Our `engine/` branch reads images: OpenAI `image_url` parts (data: or http URLs) in user and tool messages, as
Glyph and most agent harnesses send screenshots. TensorFold 0.3.6.1 as released answers them with HTTP 400.

- **Same preprocessing as the checkpoint's processor.** The image is fitted on a canvas rounded up to 28 pixels
  (zero padding right and bottom), 16 to 8,000 image tokens, one token per 28×28 pixels. Checked equal to
  vLLM's `Glm5NextImageProcessor` (largest difference 5e-7).
- **The vision tower runs on rank 0**, its 1 GB of bf16 weights in rank 0's folder (`./pull.sh` adds
  `vision.safetensors` to an existing split). Its products are fp32: the tower amplifies rounding, and run all in
  bf16 (as vLLM runs it) its output for a photo is 7% off the exact one. Ours matches a float64 reference to the
  final bf16 rounding.
- **The prompt cache knows the images.** An image's tokens carry its hash, so a follow-up turn resumes the
  conversation (its images are not encoded again) and a different image never matches a cached one.
- **One stream.** Images need `PARALLEL` empty. `ENGINE_ENV="TF_GLM_VISION=0 ..."` turns the tower off.

Measured on the two Sparks, 28 September 2026, thinking off, one image and a one-line question:

| Image | Image tokens | Prompt tokens | Prefill (tower included) | Answer |
| --- | ---: | ---: | ---: | --- |
| 1000×700 test drawing | 900 | 927 | 1.4–2.2 s | shapes, colors, positions and the caption right |
| same, follow-up turn | (cached) | 1,073 | 0.5 s (927 resumed) | which shape is bigger and the number in the caption |
| 1280×800 screenshot | 1,334 | 1,361 | 2.2 s | top-bar title and button label, small default font |
| 1920×1080 screenshot | 2,691 | 2,718 | 5.1 s | same |
| 2880×1800 screenshot | 6,695 | 6,722 | 16.3 s | same |
| 3900×2600 drawing | 7,957 (the cap) | 7,972 | 19.9 s, of which the tower 13 s | right |

Images take ~2.3 GB on rank 0 (tower and scratch), counted in the startup estimate; the 458,752-token context
fits beside them. We tried attention in split fp16 pieces on tensor cores to speed the tower up: 1.5x faster on the
largest image, but 1.5% off the exact rows, so the tower stays exact.

## Agent speed

Four things in our `engine/` branch aim at the time an agent (Glyph, a coding harness) waits, measured on the two
Sparks, 28 September 2026:

**Reasoning effort.** GLM-5.3's template writes a "Reasoning Effort" line: Low, High or Max. TensorFold 0.3.6.1's
CUDA server does not pass `reasoning_effort` on, so every request thinks at Max. The branch maps it: `minimal`/`low`
to Low, `medium`/`high` to High, `xhigh`/`max` to Max, `none` turns thinking off; without the field it stays Max.
Three coding prompts, one seed:

| `reasoning_effort` | Thinking | Time for the three |
| --- | ---: | ---: |
| `low` | 182 chars | 57 s |
| `high` | 14,160 chars | 160 s |
| none sent (Max) | 109,517 chars | 685 s |

**Prompts kept on disk** (`TF_GLM_DISK_DIR`). Every prompt's rows also go to disk, as the rows it adds to the prompt
before it, so a long conversation that another request pushed off the GPU, or one from before a restart, resumes
instead of being read again. A 102,934-token conversation, 85 s to read cold: resumed in 0.8 s after another
conversation, 3.1 s after a server restart. Files are ~20 KB a token per Spark (64 GiB cap, `TF_GLM_DISK_GIB`),
and a new engine build starts them afresh.

**Checkpoints part way through a prompt** (with `TF_GLM_DISK_DIR`). A prefill also writes its state every 4,096
tokens up to 32,768 and every 16,384 after, so a prompt that shares only the start of a kept one resumes from the
last checkpoint before they part: a client that trimmed or edited earlier text, or another agent with the same tool
list (GLM's template puts the tools first) and its own system prompt. Three agents with one 60-tool list (17,936
tokens) and different system prompts: 18.7 s for the first, 2.1 s and 3.2 s for the others (16,384 resumed).

**Drafts copied from the context** (`TF_GLM_LOOKUP`). When the last 8 tokens also stand earlier in the prompt or
the reply, a round verifies up to 7 tokens that followed them there, as a model rewriting a file does. Replies stay
byte-identical (checked with the same token counts, and `drafted == serial` 9/9). Thinking off, greedy:

| Task | MTP drafts only | With copied drafts |
| --- | ---: | ---: |
| Rewrite a 660-token file with a rename | 60.0 tok/s | 73.5 tok/s |
| Rewrite a 1,305-token file with a rename | 61.0 tok/s | 83.9 tok/s |
| Rewrite a 5,599-token file with a rename | 58.8 tok/s | 65.6 tok/s |
| Add docstrings, return the whole file | 59.9 tok/s | 69.6 tok/s |
| New code, prose | 54.1, 47.2 tok/s | 52.7, 46.1 tok/s (no copied rounds: run-to-run noise) |

**The client's prompts.** A server that caches prompts resumes only from an exact prefix. In Glyph we changed two
things that rewrote the start of a running prompt: tools shown mid-run (the tool list sits at the start of GLM's
prompt) and compaction that fired every ~5% of the window. Other harnesses may do the same: keep the prompt
append-only and send `reasoning_effort`.

### Where a decode step goes

Profiled on rank 0 (`engine/tools/profile_glm_decode.py`, `engine/tools/bench_glm_comm.py`), 28 September 2026:
one row takes 27.8 ms without the network and 29.1 ms across both Sparks, so the 90 all-gathers of a step cost
~1.3 ms (alone they take 4-6 ms; they overlap the kernels). Of the 28.2 ms of kernels, the routed and shared
experts take 12.3 ms, reading ~2.7 GB at ~220 GB/s (80% of GB10's 273 GB/s); the other 4-bit projections run at
~220 GB/s too with weights not in L2 (`engine/tools/tune_glm_qmm.py`: the best K split of every shape would save
0.8 ms a step, under 3%, so we kept TensorFold's); the rest ~5 ms. Four rows take 2.6x the expert time, because
four tokens route to ~3x as many experts. Decoding is bound by memory bandwidth and close to it.

Fewer bytes a token is what is left, so we measured Mia-AiLab's EXL3 checkpoint (`GLM-5.3-Flash-EXL3-TR3-4bpw`:
4-bit EXL3 experts, every other weight BF16) on the same branch: a 1-row step takes 58.1 ms (29.1 on MLX), standard
cells 30.6 / 28.3 / 36.9 / 27.2 tok/s and code as a chat reply 35.4 / 40.2 (MLX: 45.7 / 42.8 / 60.4 / 45.9 and
54.1 / 58.0), drafted == serial 9/9; its window fits 507,018 tokens against 465,768. TensorFold's own log says as
much ("the MLX checkpoint ... runs faster"). The MLX checkpoint stays.

## What this recipe does for speed

- **Per-rank halves** (`RANK_SPLIT=1`). `pull.sh` runs TensorFold's splitter once; each rank then loads 91 GB
  instead of reading through 182 GB (the down projections' halves interleave in the original files). TensorFold's
  own measurements were taken this way. With `WEIGHTS_SYNC=rsync` Spark 2 only ever receives its half.
- **Draft policy per request.** The default `auto` measures MTP against DFlash2 on each greedy request and keeps
  the faster one. Any request can pick another policy with a suffix on the model id:

  | Model id | Drafts a round |
  | --- | --- |
  | `GLM-5.3-Flash` | `auto` (default) |
  | `GLM-5.3-Flash@a:0.6:0.85` | 1 to 3 MTP drafts, by running acceptance |
  | `GLM-5.3-Flash@c3:0.35` | up to 3 MTP drafts while their joint probability stays ≥ 0.35 |
  | `GLM-5.3-Flash@fc5:0.3` | up to 5 DFlash2 drafts (needs the draft model) |
  | `GLM-5.3-Flash@0` | none: the serial reference |

  `./sweep-policies.sh` benchmarks them all on the running server and prints the comparison.
- **Kernel cache on the host.** Compiled kernels live in `KERNEL_CACHE` (from `cluster.env`), so only the very
  first start pays for compilation.
- **No page-cache drop.** TensorFold measured that dropping the page cache after loading set off ~25 GB of page
  migration during decoding. The recipe never does it.

## Settings (`recipe.env`)

| Variable | Default | Notes |
| --- | --- | --- |
| `ACCEPT_NONCOMMERCIAL_DRAFTER` | `0` | `1` pulls and uses DFlash2 (non-commercial license) |
| `CONTEXT` | `458752` | prompt + reply tokens, tested with a 449k prompt (lowest free memory 12 GB); `0` allocates what fits (465,768 on our Sparks). GB10 counts the page cache as used memory: when a start finds less free, `start.sh` starts with the window that fits (411,648 after an image build) if it holds `CONTEXT_MIN` (262,144) |
| `ENGINE_ENV` | see `recipe.env` | `TF_GLM_CACHE_GIB=1`, `TF_GLM_DISK_DIR`/`TF_GLM_DISK_GIB` (prompts on disk), `TF_GLM_LOOKUP=1` (copied drafts) |
| `RANK_SPLIT` | `1` | `0` serves the full checkpoint on both ranks (needs 182 GB of disk on each) |
| `PARALLEL` | empty | `4`: up to 4 requests decoded together, `CONTEXT` each (default 65,536 then) |
| `SERVE_ARGS` | `--drafter none` without the opt-in | more `tensorfold serve` flags, e.g. `--mtp-drafts 3`, `--thinking-budget 2048` |

## Limits

- One request at a time unless `PARALLEL` is set (then up to that many, each with its own `CONTEXT`).
- A client that disconnects does not stop its reply early: both ranks finish it.
- On GB10, bursts of page migration can slow a run to half speed. `bench.sh` re-measures greedy cells whose runs
  disagree and marks the ones that still do.
- TensorFold's server does not support `stop`, `n > 1` or `logprobs` (see its `docs/api.md`); images only
  with our `engine/` branch and without `PARALLEL`. Video and audio are not supported.

Source: [TensorFold's GLM recipe](https://github.com/ashhart/TensorFold/blob/main/docs/recipes/glm-5.3-flash.md).
