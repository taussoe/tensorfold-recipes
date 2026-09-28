# 3. Troubleshooting

Start with `./tools/doctor.sh` (Sparks) and `./status.sh` + `./logs.sh` in the recipe folder. `./logs.sh 1`
shows rank 1 on Spark 2.

## Starting

| Symptom | Cause and fix |
| --- | --- |
| `config/cluster.env is missing` | `cp config/cluster.env.example config/cluster.env` and edit it |
| `image missing. Run ./setup.sh first` | the engine image is built per recipe engine and TensorFold commit |
| `Spark N is already serving: …` | another recipe runs. Stop it with its `./stop.sh` (or `docker rm -f <name>` on that Spark) |
| Two-Spark start hangs, then times out | rank 0 waits at the rendezvous for rank 1. `./logs.sh 1`. Check: rank 1 running? `ping` over the link both ways? port 29551 open? `NCCL_IB_HCA` lists only `(Up)` adapters? |
| Ranks refuse to start: their settings differ | TensorFold compares both ranks' flags and files: e.g. a draft model on one Spark only. Run `./pull.sh` again (it copies drafters too), or `SERVE_ARGS="--drafter none"` |
| NCCL `unhandled system error` / falls back to sockets | the container lacks RDMA: the recipes pass `--device /dev/infiniband --cap-add IPC_LOCK --ulimit memlock=-1`; check `/dev/infiniband` exists on the host |
| Out of memory while loading | something else holds unified memory: `free -h`, `docker ps` on both Sparks. One model at a time |
| First start slow | kernel compilation (1–4 min), cached afterwards in `KERNEL_CACHE` |
| GLM: `this request needs a N-token context` (HTTP 400) | restart with a larger window: `CONTEXT=65536 ./start.sh` |
| Flash Next: long prompts rejected | its CUDA default is 8,192 tokens: `CONTEXT=65536 ./start.sh` |

## Speed

| Symptom | Cause and fix |
| --- | --- |
| One run in a cell at half speed | GB10 page migration bursts (measured by TensorFold). `bench.py` re-measures greedy cells; if a `*` stays, run the cell again |
| Everything slower than published | another container or process on a Spark; a benchmark started before warm-up ended; for Flash Next, n-gram tables evicted from the page cache |
| Sampled cells vary a lot | expected: each seed decodes a different text. Compare medians, not single runs |
| Mac slower over time | heat; plug in, High Power mode, let it cool, compare cool with cool |
| Mac 3× slower | on battery |

## Requests

| Symptom | Cause and fix |
| --- | --- |
| Long `reasoning_content` before every answer (Qwen, GLM) | thinking is on by default. `"chat_template_kwargs": {"enable_thinking": false}` per request, `--no-thinking` or `--thinking-budget N` for the server |
| `stop`, `n`, `logprobs`, images ignored or rejected | TensorFold's server does not support them yet (its `docs/api.md`) |
| A second request waits | TensorFold decodes one request at a time; a request with `"priority": "background"` yields to others |

## Recovering

```bash
./stop.sh                                    # both Sparks, whatever NODES is
docker ps -a                                 # on each Spark: leftover containers?
docker rm -f $(docker ps -aq --filter label=tf-recipes.recipe)   # every TensorFold recipe container on this Spark
```
