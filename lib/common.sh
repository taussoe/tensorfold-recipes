#!/usr/bin/env bash
# Shared helpers for every recipe. Sourced by lib/recipe.sh, never run on its own.
# Keep this file bash 3.2 compatible: the Mac recipes source it too.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# TensorFold is pinned to one commit so every Spark and every benchmark runs the same code.
# Bump both lines together (see docs/02-optimization.md#updating-tensorfold).
TENSORFOLD_VERSION=0.3.6.1
TENSORFOLD_COMMIT=34bae79ac97da6c3ab3fe10159cf49633ce8112a
TENSORFOLD_PIP="git+https://github.com/ashhart/TensorFold.git@${TENSORFOLD_COMMIT}"

# Where the Spark image's TensorFold comes from:
#   local   this repo's engine/ checkout: our branch with the latent-cache long-context work (the default)
#   pinned  upstream at TENSORFOLD_COMMIT
TENSORFOLD_SOURCE="${TENSORFOLD_SOURCE:-local}"

LABEL_KEY=tf-recipes.recipe   # docker label on every container a recipe starts

if [ -t 2 ]; then C_B=$'\033[1;34m'; C_Y=$'\033[1;33m'; C_R=$'\033[1;31m'; C_G=$'\033[1;32m'; C_0=$'\033[0m'
else C_B=; C_Y=; C_R=; C_G=; C_0=; fi

log()  { printf '%s[%s]%s %s\n' "$C_B" "${RECIPE_NAME:-recipes}" "$C_0" "$*" >&2; }
ok()   { printf '%s[%s] ✓%s %s\n' "$C_G" "${RECIPE_NAME:-recipes}" "$C_0" "$*" >&2; }
warn() { printf '%s[%s] !%s %s\n' "$C_Y" "${RECIPE_NAME:-recipes}" "$C_0" "$*" >&2; }
die()  { printf '%s[%s] ✗%s %s\n' "$C_R" "${RECIPE_NAME:-recipes}" "$C_0" "$*" >&2; exit 1; }

load_cluster() {
  local f="$REPO_ROOT/config/cluster.env"
  [ -f "$f" ] || die "config/cluster.env is missing. Run: cp config/cluster.env.example config/cluster.env  (then edit it)"
  set -a; . "$f"; set +a
  : "${PORT:=8080}" "${WEIGHTS_SYNC:=rsync}"
  : "${HF_CACHE:=$HOME/.cache/huggingface}" "${KERNEL_CACHE:=$HOME/.cache/tensorfold-kernels}"
}

load_recipe() {
  RECIPE_DIR="$1"
  [ -f "$RECIPE_DIR/recipe.env" ] || die "no recipe.env in $RECIPE_DIR"
  # A variable set on the command line wins over recipe.env:  CONTEXT=65536 NODES=1 ./start.sh
  local n overrides=""
  for n in $(sed -nE 's/^([A-Z_][A-Z0-9_]*)=.*/\1/p' "$RECIPE_DIR/recipe.env"); do
    if [ -n "${!n+x}" ]; then overrides="$overrides $n=$(printf '%q' "${!n}")"; fi
  done
  set -a; eval "$overrides"; . "$RECIPE_DIR/recipe.env"; eval "$overrides"; set +a
  RECIPE_NAME="${RECIPE_NAME:-$(basename "$RECIPE_DIR")}"
}

# run_on NODE CMD...   NODE 1 is this machine (Spark 1), NODE 2 is $SPARK2_SSH. stdin is passed through.
run_on() {
  local node="$1"; shift
  if [ "$node" = 1 ]; then "$@"
  else ssh -T -o BatchMode=yes -o ConnectTimeout=10 "$SPARK2_SSH" "$(printf '%q ' "$@")"
  fi
}

# The nodes a recipe uses: "1" or "1 2".
nodes() { if [ "${NODES:-1}" = 2 ]; then echo "1 2"; else echo 1; fi; }

require_nodes_supported() {
  case " ${SUPPORTED_NODES:-1} " in *" ${NODES} "*) ;; *) die "$RECIPE_NAME runs on ${SUPPORTED_NODES} Spark(s), not NODES=${NODES}";; esac
}

wait_http() {
  # wait_http URL SECONDS [CONTAINER NODE [CONTAINER2 NODE2]]: fails early when a watched container stops (with
  # two ranks, rank 1 refusing its context leaves rank 0 waiting at the rendezvous, so both are watched).
  local url="$1" limit="$2" start now c n
  start=$(date +%s)
  while :; do
    if curl -fsS -m 5 "$url" >/dev/null 2>&1; then return 0; fi
    now=$(date +%s)
    set -- "$url" "$limit" "${3:-}" "${4:-1}" "${5:-}" "${6:-2}"
    for pair in "$3:$4" "$5:$6"; do
      c="${pair%:*}" n="${pair##*:}"
      [ -n "$c" ] || continue
      if ! run_on "$n" docker ps -q --filter "name=^${c}\$" | grep -q .; then
        warn "container $c on Spark $n stopped. Last log lines:"
        run_on "$n" docker logs --tail 40 "$c" 2>&1 | sed 's/^/    /' >&2 || true
        return 1
      fi
    done
    if [ $((now - start)) -ge "$limit" ]; then return 1; fi
    printf '.' >&2; sleep 5
  done
}

# Refuse to start when another recipe holds a GPU: two models on one GB10 fight over its 128 GB.
ensure_gpus_free() {
  local node running ours other
  for node in $(nodes); do
    running=$(run_on "$node" docker ps --format '{{.Names}}' || true)
    # Recipe containers are tf-<recipe>-rankN. Mia-AiLab's vLLM GLM (glm53-exl3-*) holds the GPUs as well.
    ours=$(echo "$running" | grep -E '^(tf-.*-rank[01]|glm53-exl3-.*)$' || true)
    other=$(echo "$running" | grep -vE '^(tf-.*-rank[01]|glm53-exl3-.*)$' | grep . || true)
    if [ -n "$ours" ]; then
      die "Spark $node is already serving: $(echo $ours). Stop it first (a recipe's ./stop.sh, or ./stop.sh in Mia's vLLM repo): one model at a time."
    fi
    [ -z "$other" ] || warn "Spark $node also runs: $(echo $other). Make sure none of them uses the GPU or much memory."
  done
}

# rsync_model_to_spark2 DIRNAME   copy $HF_CACHE/hub/DIRNAME to Spark 2 over the direct link.
rsync_model_to_spark2() {
  local d="$1" src="$HF_CACHE/hub/$1"
  [ -d "$src" ] || die "$src not found on Spark 1"
  run_on 2 mkdir -p "$HF_CACHE/hub"
  log "copying $d to Spark 2 over the direct link ($SPARK2_LINK_IP)"
  # The HF cache stores blobs plus relative symlinks; -a keeps both. --inplace avoids a second copy on disk.
  rsync -a --info=progress2 --inplace -e "ssh -T -o BatchMode=yes -c aes128-gcm@openssh.com" \
    "$src/" "$(ssh -G "$SPARK2_SSH" | awk '/^user /{print $2}')@${SPARK2_LINK_IP}:$HF_CACHE/hub/$d/"
}

# fix_owner NODE IMAGE PATH...   containers write as root; give the files back to the user who runs the recipes,
# so rsync, rm and the next pull work without sudo.
fix_owner() {
  local node="$1" image="$2" ug p; shift 2
  ug="$(run_on "$node" id -u):$(run_on "$node" id -g)"
  local mounts=""
  for p in "$@"; do mounts="$mounts -v $p:$p"; done
  # shellcheck disable=SC2086
  run_on "$node" docker run --rm --entrypoint chown $mounts "$image" -R "$ug" "$@"
}

hf_dirname() { echo "models--$(echo "$1" | sed 's#/#--#g')"; }

print_endpoint() {
  local host="${SPARK1_LAN:-<spark1>}"
  ok "serving ${SERVED_MODEL:-$MODEL}"
  cat >&2 <<EOF

    API        http://${host}:${PORT}/v1$([ "$host" = 127.0.0.1 ] || echo "          (on Spark 1: http://127.0.0.1:${PORT}/v1)")
    Model id   $(curl -fsS -m 5 "http://127.0.0.1:${PORT}/v1/models" 2>/dev/null | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null || echo "${SERVED_MODEL:-?}")
    Try it     ./chat.sh "Say hello in one sentence."
    Benchmark  ./bench.sh
    Stop       ./stop.sh

EOF
}
