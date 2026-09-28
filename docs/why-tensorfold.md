# Why TensorFold

Every recipe in this repo runs on [TensorFold](https://github.com/ashhart/TensorFold), written by
[ashhart](https://github.com/ashhart) ([@ashxhart](https://x.com/ashxhart)) with the TensorFold contributors. This page
explains, in our own words, what makes it special. The authoritative source is always
[TensorFold's own docs](https://github.com/ashhart/TensorFold/tree/main/docs), which are some of the most carefully
measured documentation we have read.

## Speculative decoding that never changes the output

Most engines draft tokens with a small model or an MTP head and accept them when the big model agrees, but "agrees"
is fuzzy: batching and kernel choices change the numbers a little, so a drafted reply can differ from the one the
model would have written alone. TensorFold makes it exact. Sampling is keyed (the token at position *p* is the argmax
of `logit / T + Gumbel(seed, p, token)` over the top-k/top-p set), and every kernel on the verify path is written so a
row gets the same bits alone or inside a window of drafted rows. A drafted reply is therefore byte-identical to
serial decoding: drafting changes only the speed. `bench.py`'s exactness suite checks this on your own hardware, and
it has passed on every build we have measured.

## Kernels written for each model

TensorFold supports a hand-picked set of models and writes the fast path for each one: Metal/MLX on Apple Silicon,
CUDA and Triton on NVIDIA. DFlash2 draft trees for the 27B, MTP chains for Flash Next and GLM, FP8 prompt kernels,
grouped expert kernels, CUDA graphs, and on two DGX Sparks tensor parallelism with partial sums added in a fixed
order, so even two machines give the exact serial result.

## Moving fast

In the two days we built these recipes TensorFold shipped 0.3.4.1, 0.3.5 and 0.3.6: concurrent streams, FP8 prompt
processing (Flash Next's prompts went from ~100 to ~2,000 tok/s on one Spark), Macs with less memory, EXL3
checkpoints. A GLM bug we reported ([#53](https://github.com/ashhart/TensorFold/issues/53)) was found and fixed within
hours, with a clear explanation of the cause. We pin a commit (`lib/common.sh`) and re-benchmark on each release.

## What this repo adds

Very little, compared with TensorFold itself: scripts that set up two Sparks with one command, a benchmark that
compares engines on the same machines, and, in our `engine/` branch, a latent attention cache and concurrent streams
for GLM-5.3-Flash, offered back to TensorFold in [#54](https://github.com/ashhart/TensorFold/pull/54).

## Things to know

- TensorFold serves the models it has a family for; others are refused before anything downloads.
- The CUDA server does not take images yet (on TensorFold's roadmap); the Mac server's features are listed in
  TensorFold's `docs/api.md`.
- It is young (0.3.x) and changes quickly: pin a commit, as these recipes do.
