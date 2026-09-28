# Fast local LLMs on DGX Spark and Mac: recipes

Serve frontier open models on **two NVIDIA DGX Sparks** and on a **Mac** with
[TensorFold](https://github.com/ashhart/TensorFold), and nothing else. Every recipe works the same way: one folder, the same commands, and a benchmark whose
numbers compare across all of them.

> **Thank you, Ash.** Everything here stands on [TensorFold](https://github.com/ashhart/TensorFold), the engine
> [ashhart](https://github.com/ashhart) ([@ashxhart](https://x.com/ashxhart)) builds with the TensorFold contributors:
> exact speculative decoding, kernels written for each model, and some of the most carefully measured documentation
> we have seen. The speed in these tables is TensorFold's. This repo only makes it easy to run, measures it next to
> vLLM on the same machines, and adds a few things for GLM that we have offered back upstream. If these recipes help
> you, please star [TensorFold](https://github.com/ashhart/TensorFold). More in [Why TensorFold](docs/why-tensorfold.md).

| Model | Where | Context | Decode, one stream | Reading a 32k prompt | Measured on |
| --- | --- | ---: | ---: | ---: | --- |
| [GLM-5.3-Flash](dgx-spark/glm-5.3-flash/) | 2× Spark | 448k, with images | 44–63 tok/s (vLLM: 27–48) | 1,292 tok/s (vLLM: 1,350) | our Sparks, same benchmark |
| [Qwen3.8 Flash Next](dgx-spark/qwen3.8-flash-next/) | 1 or 2× Spark | 157k on 1, 262k on 2 | 60–75 on 1, 86–103 on 2 | 1,890 on 1, 2,599 on 2 (vLLM, 1 Spark: 2,314) | our Sparks |
| [Qwen3.8-27B](dgx-spark/qwen3.8-27b/) | 1 or 2× Spark | 262k | 71–82 tok/s on 2, code as chat 142–164 | 1,945 tok/s on 2 | our Sparks |
| [Qwen3.8 Flash Next](mac/qwen3.8-flash-next/) | Mac, 128 GB (n-gram tables on the SSD) | 64k | 64–76 tok/s on an M3 Max | 546 tok/s | this Mac |
| [Qwen3.8-27B](mac/qwen3.8-27b/) | Mac (Apple Silicon) | 64k | 47–54 chat, code as chat 110 tok/s on an M3 Max | ~200 tok/s | this Mac |

Several agents at once: `PARALLEL=N ./start.sh` in each Spark recipe (GLM 122 tok/s together at 4 streams, Flash Next
181 at 8 on one Spark, the 27B 236 at 8 on two). All numbers and their protocols are in each recipe's README and in [results/LEADERBOARD.md](results/LEADERBOARD.md);
measure your own with `./bench.sh`.

## Our TensorFold branch (`engine/`)

The Spark recipes build TensorFold from `engine/` (`setup.sh` clones it the first time): branch `glm-long-context`, TensorFold 0.3.6.1 with our
long-context work for GLM-5.3-Flash on top (Flash Next and the 27B run 0.3.6.1's own code). Set
`TENSORFOLD_SOURCE=pinned` for a released TensorFold, or `ENGINE_DIR=<checkout>` to build another tree. To clone it
yourself: `git clone -b glm-long-context https://github.com/taussoe/TensorFold.git engine`.

What it adds to 0.3.6.1, every change tested so drafted replies stay byte-identical to serial ones:

| Change | Effect |
| --- | --- |
| GLM's attention cache as its 512-wide latent (~20 KB a token per Spark instead of ~0.4 MB) | 449k tested (448k the default, beside image input) on two Sparks, where the per-head cache holds ~40k; offered to TensorFold in [#54](https://github.com/ashhart/TensorFold/pull/54) |
| DSA token selection for all rows of a chunk over the pools it can see, top 512 by one radix-select kernel; CUDA graphs past 2,051 tokens | selection at 128k: 45 → 6 ms a layer and chunk |
| Latent attention tiles of all 32 heads in prompt chunks; KDA chains of prompt chunks in three kernels | sparse attention 28 → 20 ms, a KDA chain 6.9 → 4.0 ms (2,048 rows) |
| Memory admission that sizes the latent cache (and the prompt scratch as allocated) | `CONTEXT=0` finds 465,768 tokens with image input |
| Prompt cache for several conversations (rows saved when another conversation takes the caches, 3 GiB budget) | switching back to a long conversation resumes instead of prefilling again |
| Concurrent GLM streams (`PARALLEL=4`): every stream's drafts verified in one forward, each equal to its serial decoding | 122 tok/s together at 4 streams, 33 each (vLLM: 62 together, 17 each) |
| Image input for GLM (`image_url` parts): the checkpoint's processor, its vision tower on rank 0 with fp32 products, image-aware prompt cache | screenshots read right; a 1920×1080 screenshot prefills in 5.1 s ([details](dgx-spark/glm-5.3-flash/README.md#images)) |
| GLM reads `reasoning_effort` (the template's Low/High/Max; before, always Max) | three coding prompts: 685 s at Max, 160 s at High ([agent speed](dgx-spark/glm-5.3-flash/README.md#agent-speed)) |
| Prompts kept on disk, written as the rows each adds | a 103k-token conversation resumes in 0.8 s after another one, 3.1 s after a restart (85 s cold) |
| Drafts copied from the context when the last 8 tokens stand earlier | file rewrites 1.12–1.38x faster, replies byte-identical |

With 0.3.5.1's prompt kernels and a bf16 head GEMM for absorb/expand, GLM reads a 32k prompt at 1,292 tok/s and a
128k one at 1,231 (vLLM: 1,350 and 1,243); on our 0.3.4-based branch it was 772 and 739, on TensorFold 0.3.4 187 and none. The older branch is
`long-context` (0.3.4.1).

## Start here

**On a Mac** (Qwen3.8-27B):

```bash
cd mac/qwen3.8-27b
./setup.sh      # Python env with TensorFold, checks the Mac
./pull.sh       # 20 GB download
./start.sh      # serves http://127.0.0.1:8080/v1
./chat.sh "Write a haiku about GPUs"
./bench.sh      # measures, saves to results/
./stop.sh
```

**On two DGX Sparks**, once (about 15 minutes, [the hardware guide](docs/01-hardware-setup.md) has every step):

```bash
git clone <this repo> && cd <repo>             # on Spark 1 only; Spark 2 is driven over SSH
cp config/cluster.env.example config/cluster.env && nano config/cluster.env
./tools/doctor.sh                               # checks GPUs, Docker, SSH, the 200 Gb/s link, RDMA
```

then for any model:

```bash
cd dgx-spark/qwen3.8-27b
./setup.sh && ./pull.sh && ./start.sh           # clone engine/ (first time), build the image, download, serve
./bench.sh
./stop.sh
```

Every recipe folder has the same scripts:

| Script | What it does |
| --- | --- |
| `setup.sh` | builds or installs the engine (pinned version) on every machine the recipe uses |
| `pull.sh` | downloads the weights; with two Sparks, copies them over the direct link |
| `start.sh` | starts the server and waits until it answers; prints the URL and model id |
| `chat.sh "…"` | one test request |
| `bench.sh` | the benchmark suites; results land in `results/<recipe>/` |
| `status.sh`, `logs.sh [1]`, `stop.sh` | what runs, its log (rank 1 with `1`), stop everything |
| `recipe.env` | every setting of the recipe, commented; override any of them per run: `NODES=1 ./start.sh` |

One model runs at a time: each one uses most of a Spark's 128 GB. `start.sh` refuses to start next to another.

## Layout

```
config/cluster.env.example   your Sparks: SSH name, link addresses, NCCL adapters (copy to cluster.env)
dgx-spark/<model>/           one folder per model on DGX Spark
mac/<model>/                 one folder per model on the Mac
bench/                       bench.py (the benchmark), compare.py (the leaderboard), reference.json (published numbers)
results/                     your benchmark runs (JSON) and LEADERBOARD.md
docs/                        hardware setup, optimization notes, troubleshooting, the TensorFold review
lib/                         the shared scripts behind every recipe folder (you don't need to read them)
docker/tensorfold-spark/     the pinned TensorFold image for the Sparks
tools/                       doctor.sh (checks), bench-all.sh (benchmark every model back to back)
```

## Docs

1. [Hardware setup](docs/01-hardware-setup.md): cabling two Sparks, link addresses, SSH, RDMA, `cluster.env`.
2. [Optimization](docs/02-optimization.md): what makes each model fast, what we switched on, and what not to touch.
3. [Benchmarking](bench/README.md): the suites, how to read them, how to compare fairly.
4. [Troubleshooting](docs/03-troubleshooting.md).
5. [Why TensorFold](docs/why-tensorfold.md): what makes TensorFold special, in our words.

## Pinned versions

| Component | Version |
| --- | --- |
| TensorFold (Mac) | 0.3.4, commit `2f8e514` (`lib/common.sh`) |
| TensorFold (Sparks) | `engine/`, branch `glm-long-context` on 0.3.6.1 (`TENSORFOLD_SOURCE=local`, the default) |
| NVIDIA PyTorch container | `nvcr.io/nvidia/pytorch:26.07-py3` |

Pinning is deliberate: both Sparks and every benchmark run the same code. [Updating](docs/02-optimization.md#updating-the-pins).

## Credits

- **[TensorFold](https://github.com/ashhart/TensorFold)** by [ashhart](https://github.com/ashhart) ([@ashxhart](https://x.com/ashxhart) on X) and the
  TensorFold contributors is the engine behind every recipe here: exact speculative decoding (every drafted reply is
  byte-identical to serial decoding), the CUDA and MLX engines, and the models' kernels. This repo only packages,
  measures and extends it. Our `engine/` branch
  ([taussoe/TensorFold, `glm-long-context`](https://github.com/taussoe/TensorFold/tree/glm-long-context)) is a fork of
  TensorFold 0.3.6.1; its GLM long-context work is proposed upstream in
  [#54](https://github.com/ashhart/TensorFold/pull/54).
- **[Mia-AiLab](https://github.com/MiaAI-Lab)**'s vLLM recipes for DGX Spark are the baselines the benchmarks compare
  against, measured on the same machines, and the starting point for this repo's layout.
- **The models:** Z.ai's GLM-5.3-Flash and Qwen's Qwen3.8 models, in the MLX conversions by
  [Vontra](https://huggingface.co/Vontra); the DFlash2 draft models by [z-lab](https://huggingface.co/z-lab) (Qwen3.8-27B)
  and [incoai](https://huggingface.co/incoai) (GLM-5.3-Flash); the EXL3 packs measured in the 27B recipe by
  [turboderp](https://huggingface.co/turboderp).

## Licenses

The scripts, benchmark and docs here are MIT ([LICENSE](LICENSE)). TensorFold, and so our `engine/` branch, is MIT
(Copyright (c) 2026 TensorFold contributors) with the third-party notices in its repo. Each model keeps its own
license (see its Hugging Face page). GLM's DFlash2 draft model is **CC BY-NC-ND 4.0, non-commercial**: the GLM recipe
only uses it after you opt in.
