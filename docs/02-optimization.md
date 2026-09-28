# 2. Optimization: what makes these models fast, and what we switched on

Decoding one stream is limited by how many bytes of weights each token reads, divided by memory bandwidth
(GB10: ~240 GB/s measured; M3 Max: 400 GB/s; M3 Ultra: 800 GB/s). Everything that makes these recipes fast
is one of three things:

1. **Read fewer bytes a token**: 4-bit or 2-bit weights, MoE models that read only their routed experts, and
   splitting the model over two Sparks so each reads half.
2. **Get more tokens per read**: speculative decoding. A cheap drafter guesses several tokens, and the big model
   checks all of them in one forward that costs little more than one token. Everything accepted is free.
3. **Stop wasting time between reads**: fused kernels, CUDA graphs, host work overlapped with the GPU.

| Model | Bytes a token (per Spark) | Drafter | Why it is fast here |
| --- | --- | --- | --- |
| Qwen3.8-27B | 16 GB (8 on two) | DFlash2 draft model, trees of 12 rows | dense: drafting is everything (13 → 49 tok/s on one Spark) |
| Qwen3.8 Flash Next | a few GB (routed experts only) | MTP head, up to 6 a round | small active set + CUDA graphs |
| GLM-5.3-Flash | 5.0 GB each on two | MTP head and DFlash2, chosen per request | 8 of 288 experts; TensorFold's row-invariant kernels |

## What every recipe does

| Optimization | Where | Effect |
| --- | --- | --- |
| Pinned engine builds baked into an image | `docker/tensorfold-spark` | no reinstall per start; both ranks provably run the same code (TensorFold refuses mismatched ranks) |
| Compiled kernels cached on the host | `KERNEL_CACHE` mounted at `/kernel-cache` | the 1–4 minute kernel compile happens once per Spark, not per container |
| Weights copied over the 200 Gb/s link | `WEIGHTS_SYNC=rsync` | 180–340 GB reach Spark 2 in minutes instead of a second internet download |
| NCCL pinned to the live adapters | `NCCL_SOCKET_IFNAME`, `NCCL_IB_HCA` | NCCL uses RoCE on the cable instead of hanging on a dead adapter or falling back to TCP |
| `--ipc=host`, `memlock=-1`, `/dev/infiniband`, `IPC_LOCK` | every two-Spark container | what NCCL's RDMA transport needs inside Docker |
| One model at a time | `start.sh` checks both Sparks | two models on one unified 128 GB fight for memory and bandwidth |
| TensorFold's own defaults kept | no NCCL tuning | TensorFold measured: `NCCL_PROTO=LL` made 27B forwards 56% slower; channel/protocol changes moved Flash Next ≤2%. The defaults were best |

## Per model

- **GLM-5.3-Flash**: per-rank halves (`RANK_SPLIT=1`, each rank loads 91 GB), per-request draft policy
  (`./sweep-policies.sh` to find the best for your prompts), DFlash2 after opt-in (+25% greedy code).
- **Qwen3.8-27B**: two Sparks +30–65%. Its 12-row verify window was chosen by measurement (wider trees accept
  ~one more token but cost 30–40 ms more a round).
- **Qwen3.8 Flash Next**: keep its 32 GB n-gram tables in the page cache (don't run anything memory-hungry next
  to it, never drop caches); try `--mtp-drafts 4..6`.
- **Mac**: plugged in and in **High Power** energy mode (1.4–4.3× on an M3 Max, measured), quiet, cool. `--no-drafts` shows the serial baseline to see what drafting buys.

## Don'ts (each one measured by someone, somewhere)

- **Don't drop the page cache after loading** (`echo 3 > /proc/sys/vm/drop_caches`) with TensorFold: it set off
  ~25 GB of page migration while GLM decoded.
- **Don't read memory from `nvidia-smi`** on GB10: memory is unified. Use `free -h`.
- **Don't trust one run.** A sampled cell decodes a different text per seed (27B code on two Sparks: 70 to 136
  tok/s over five seeds). Page-migration bursts on GB10 slow single runs to half speed. Use medians;
  `bench.py` re-measures greedy cells whose runs disagree.
- **Don't benchmark while something else runs** on either Spark, or while the server is still warming up.
- **Don't send more concurrent streams than an engine is built for**: TensorFold decodes one at a time (others
  queue).

## Measuring a change

Every optimization question is answered the same way: change one thing, benchmark, compare.

```bash
./bench.sh --label baseline
SERVE_ARGS="--mtp-drafts 4" ./start.sh     # after ./stop.sh
./bench.sh --label mtp4
python3 ../../bench/compare.py --recipe qwen3.8-flash-next --all
```

## Updating the pins

TensorFold moves fast. To try a newer commit:

1. Set `TENSORFOLD_VERSION` and `TENSORFOLD_COMMIT` in `lib/common.sh`.
2. `./setup.sh` in the recipe builds a new image on both Sparks (the tag includes the commit, so the old one
   stays for rollback).
3. For GLM, re-split (`rm -rf ~/.cache/huggingface/tensorfold-splits/glm-5.3-flash` on both, then `./pull.sh`):
   the split rules may change between versions.
4. `./bench.sh`, and compare with the previous run before keeping it.

