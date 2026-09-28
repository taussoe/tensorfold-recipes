# 1. Hardware setup: two DGX Sparks

Once, before the first recipe. About 15 minutes. At the end `./tools/doctor.sh` passes every check.

```
            your network (LAN / Wi-Fi)                        ← SSH, downloads, the API from your laptop
        ┌──────────────┴──────────────┐
   ┌────┴─────┐   QSFP 200 Gb/s   ┌────┴─────┐
   │ Spark 1  │═══════════════════│ Spark 2  │               ← tensor-parallel traffic (NCCL over RoCE) + rsync
   │  head    │  192.168.100.1/2  │  worker  │
   │ rank 0   │                   │ rank 1   │
   │ API :8080│                   │          │
   └──────────┘                   └──────────┘
```

Spark 1 runs the recipes, serves the API and drives Spark 2 over SSH. Spark 2 only needs Docker and the weights.

## The cable

Connect one QSFP cable between the ConnectX-7 ports on the back of the two Sparks (NVIDIA sells the matching
cable for Spark pairs; any QSFP56/QSFP112 DAC that the CX-7 accepts works). One cable is enough.

## The link

Each Spark gets a fixed address on the cable's interface. Find the interface that has a link:

```bash
ibdev2netdev          # e.g. "rocep1s0f1 port 1 ==> enp1s0f1np1 (Up)"
```

The name after `==>` marked `(Up)` is the interface (`enp1s0f1np1` above), the name before it the RDMA adapter
(`rocep1s0f1`). GB10 exposes four adapters for its two ports and some show `(Down)`: only use the Up ones.

Give it an address with netplan, on Spark 1 (use `.2` on Spark 2):

```bash
sudo tee /etc/netplan/60-spark-link.yaml >/dev/null <<'EOF'
network:
  version: 2
  ethernets:
    enp1s0f1np1:
      addresses: [192.168.100.1/24]
      mtu: 9000
EOF
sudo chmod 600 /etc/netplan/60-spark-link.yaml
sudo netplan apply
ping -c 3 192.168.100.2      # from Spark 1, once both are done
```

`mtu: 9000` (jumbo frames) is the usual choice for RoCE between two hosts; set it on both ends or on neither.

## SSH

From Spark 1 to Spark 2, without a password, by name and over the link (rsync of weights uses the link address):

```bash
ssh-keygen -t ed25519            # if you have no key yet
ssh-copy-id you@spark2           # the LAN name
ssh-copy-id you@192.168.100.2    # the link address
ssh -o BatchMode=yes spark2 true && echo ok
```

## Docker

DGX OS ships Docker and the NVIDIA container runtime. Let your user run Docker without sudo, on both Sparks:

```bash
sudo usermod -aG docker $USER    # then log out and in again
docker run --rm --gpus all nvcr.io/nvidia/pytorch:26.07-py3 nvidia-smi
```

## The recipes

On Spark 1 only:

```bash
git clone <this repo> ~/llm-recipes && cd ~/llm-recipes
cp config/cluster.env.example config/cluster.env
nano config/cluster.env          # SPARK2_SSH, the link addresses, NCCL_SOCKET_IFNAME, NCCL_IB_HCA
./tools/doctor.sh
```

`NCCL_IB_HCA` takes the RDMA adapters that are Up, comma separated (`ibdev2netdev` output). NCCL may otherwise
pick a dead one and hang at the rendezvous.

## From your laptop

The API listens on Spark 1 at port 8080 on all interfaces. Set `SPARK1_LAN` in `cluster.env` to the name your
laptop uses (for the printed URLs), and benchmark from the laptop with:

```bash
BENCH_URL=http://spark1.local:8080 ./dgx-spark/qwen3.8-27b/bench.sh
```

The API has no authentication. Keep port 8080 on a trusted network, or put a reverse proxy with auth in front.

## Disk space per Spark

| Recipe | Spark 1 | Spark 2 |
| --- | --- | --- |
| GLM-5.3-Flash (`RANK_SPLIT=1`, rsync) | 182 GB checkpoint + 2 × 91 GB halves | 91 GB half |
| Qwen3.8 Flash Next | 113 GB | 113 GB |
| Qwen3.8-27B | 20 GB | 20 GB |
| TensorFold image (NVIDIA PyTorch container) | ~20 GB | ~20 GB |

After GLM's halves exist, the full checkpoint on Spark 1 can go (`rm -rf ~/.cache/huggingface/hub/models--Vontra--GLM-5.3-Flash-MLX-4bit-MTP`);
you need it again only to re-split for a new TensorFold version.
