# DeepSeek-V4-Flash on TensorFold: feasibility

Written 27 September 2026. Checkpoint: `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp` at `86f746b`, the one
[Mia's vLLM recipe](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark) serves. Sizes below are
read from the checkpoint's safetensors headers.

## Status upstream

TensorFold has no DeepSeek family today. On 27 September 2026 its author answered
[issue #14](https://github.com/ashhart/TensorFold/issues/14): DeepSeek support is being started now, targeting
0.3.6, "a lane family for its MLA attention and MoE layers, exact against its own serial decode, on Mac and CUDA".

## Does it fit on two Sparks?

Yes.

| Part | Size |
| --- | ---: |
| Routed experts (43 layers × 256, MXFP4: 4-bit values, one E8M0 scale per 32) | 147.2 GB |
| Everything else in the decoder (attention FP8 with 128×128 block scales, shared experts FP8, compressors BF16, head BF16) | 7.8 GB |
| MTP (3 layers plus DSpark's Markov head) | 10.9 GB |
| Embedding | 1.1 GB |
| Vision encoder | 0.9 GB |
| **Total** | **167.8 GB**, ~84 GB a Spark at TP=2 |

That is less than GLM-5.3-Flash's MLX 4-bit checkpoint (182 GB). The attention cache is very small, because V4
compresses its keys (see below). A 1M context costs a few GB, not the hundreds MLA or GQA would need.

## How fast could it be?

A decoded token reads 11.2 GB: 3.4 GB of routed experts (6 of 256 in each layer) and 7.8 GB of dense weights. That
is ~5.6 GB a Spark at TP=2, against GLM-5.3-Flash's ~5.0 GB. The serial speed limit is therefore about 10% below
GLM's.

With drafts at GLM's acceptance, that points to roughly 45–55 tok/s. This is an estimate, not a measurement.

Mia's vLLM recipe measures 62–83 tok/s at one stream. It uses DSpark drafts, 6 tokens deep, and native NVFP4. It
reads a 128k prompt in 75–80 s (~1,650 tok/s). Our GLM branch takes 177 s for 128k.

On GLM, TensorFold won decode by 1.3–1.4× because vLLM's GLM path was weak (MTP only). For DeepSeek, vLLM
already has a strong drafter. TensorFold would only win if its exact tree verification accepts more tokens per
step than DSpark's, which is not given.

## What is new compared with GLM

| Part | V4-Flash | Reuse from our GLM work |
| --- | --- | --- |
| Hyper-connections | 4 streams, 20 Sinkhorn steps | **yes**, the same scheme |
| Attention | One 512-wide KV head shared by 64 query heads. A 128-token sliding window of raw keys. Two kinds of compressed memory: every 4 tokens gated into one entry, with an indexer picking the top 512 entries (21 layers), and every 128 tokens into one entry, read densely (20 layers). An attention sink and a grouped low-rank output (8 groups, rank 1,024) | partly: the DSA indexer, top-k selection, sparse graphs and pool bucketing. The compressor, the window and the sink are new |
| Compression state | An entry forms only when its block of 4 or 128 tokens completes | new. Drafted rows that cross a block boundary must give the bits serial steps give. This is the hardest exactness problem here |
| MoE | 256 experts, top 6 plus a shared expert. `sqrtsoftplus` scores. SwiGLU clamped at 10. The first 3 layers route by a token-id hash table | the MoE kernels' structure (tight grid, two tiles a program) carries over. The routing is new |
| Weight formats | MXFP4 experts, FP8 E4M3 dense weights | **new kernels**: our MoE kernels read MLX 4-bit groups of 64 |
| Drafts | 3 MTP layers plus a DSpark Markov head (block of 5) | new |
| Vision | 32-layer ViT | can be left out for text-only use |

## Recommendation

Do not build our own DeepSeek family now. The author has just started the same work for 0.3.6. A second
implementation beside it would be weeks of work that is likely to be replaced, and would be hard to merge.

What we can do instead:

1. Prepare `dgx-spark/deepseek-v4-flash/` as soon as 0.3.6 ships, then benchmark it with `./bench.sh` against
   Mia's vLLM numbers above (measured from the outside only, as with GLM).
2. Offer to test on two Sparks in issue #14. Our long-context suite (32k and 128k, with a needle) and our exactness
   checks are ready to run.
3. After 0.3.6 ships, carry our GLM long-context work across to it where the parts match: long prefill chunks,
   bucketed DSA selection, sparse CUDA graphs and the multi-conversation cache.
