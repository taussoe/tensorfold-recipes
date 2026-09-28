#!/usr/bin/env bash
# TensorFold on one or two DGX Sparks. Sourced by lib/recipe.sh after common.sh, cluster.env and recipe.env.
#
# Recipe variables (set in dgx-spark/<model>/recipe.env):
#   MODEL            Hugging Face repo of the checkpoint
#   DRAFTERS         draft-model repos to pull next to it (optional, space separated)
#   SUPPORTED_NODES  "1 2", or "2" for models that need both Sparks
#   NODES            default node count (override: NODES=1 ./start.sh)
#   CONTEXT          --context for tensorfold serve (empty: TensorFold's default for the family)
#   SERVE_ARGS       more `tensorfold serve` flags, the same on both ranks
#   RANK_SPLIT       1: serve per-rank halves written by ./pull.sh (GLM only: each Spark reads 91 GB, not 182)
#   SERVED_NAME      the model id clients send (default: the repo's name)
#   STARTUP_TIMEOUT  seconds to wait for /health
#   ENGINE_ENV       engine environment variables, KEY=VALUE words passed to both ranks
#   ENGINE_REPO, ENGINE_BRANCH  where setup.sh clones engine/ from (default: our fork, branch glm-long-context)
#   ENGINE_DIR       the TensorFold tree built with TENSORFOLD_SOURCE=local (default: engine/), e.g. another
#                    checkout for an A/B: ENGINE_DIR=~/tensorfold-upstream ./setup.sh

if [ -n "${TF_IMAGE:-}" ]; then
  :   # an exact image, e.g. an older build for an A/B: TF_IMAGE=tensorfold-spark:local-297c3a9d46 ./start.sh
elif [ "$TENSORFOLD_SOURCE" = local ]; then
  ENGINE_DIR="${ENGINE_DIR:-$REPO_ROOT/engine}"
  # Our TensorFold branch; setup.sh clones it into engine/ the first time.
  : "${ENGINE_REPO:=https://github.com/taussoe/TensorFold.git}" "${ENGINE_BRANCH:=glm-long-context}"
  if [ ! -d "$ENGINE_DIR/src/tensorfold" ]; then
    [ "$cmd" = setup ] || die "$ENGINE_DIR is missing: run ./setup.sh (it clones $ENGINE_REPO, branch $ENGINE_BRANCH), or TENSORFOLD_SOURCE=pinned"
    log "cloning TensorFold ($ENGINE_BRANCH) into $ENGINE_DIR"
    git clone -q -b "$ENGINE_BRANCH" "$ENGINE_REPO" "$ENGINE_DIR" || die "could not clone $ENGINE_REPO"
  fi
  # Tag by a hash of the engine's source, so a code change builds a new image and an unchanged tree reuses it.
  # The Dockerfile counts too: a change to how the image builds must not reuse an old image.
  ENGINE_HASH=$( (cd "$ENGINE_DIR" && find src pyproject.toml -type f \( -name '*.py' -o -name '*.cu' -o -name '*.cuh' \
    -o -name '*.cpp' -o -name '*.txt' -o -name '*.toml' \) -print0 | sort -z | xargs -0 cat;
    cat "$REPO_ROOT/docker/tensorfold-spark/Dockerfile.local") | sha256sum | cut -c1-10)
  TF_IMAGE="tensorfold-spark:local-$ENGINE_HASH"
else
  TF_IMAGE="tensorfold-spark:${TENSORFOLD_VERSION}-${TENSORFOLD_COMMIT:0:7}"
fi
MASTER_PORT="${MASTER_PORT:-29551}"
: "${SERVED_NAME:=${MODEL##*/}}" "${STARTUP_TIMEOUT:=900}" "${NODES:=${SUPPORTED_NODES##* }}"
SERVED_MODEL="$SERVED_NAME"
SPLIT_ROOT=/root/.cache/huggingface/tensorfold-splits/$RECIPE_NAME   # inside the container

container_name() { echo "tf-${RECIPE_NAME}-rank$1"; }

# tf_docker NODE [docker run flags...] -- CMD...   one container on one Spark, with the caches mounted.
tf_docker() {
  local node="$1"; shift
  local flags=() ; while [ "$1" != -- ]; do flags+=("$1"); shift; done; shift
  local net=()
  [ -n "${HF_TOKEN:-}" ] && net+=(-e "HF_TOKEN=$HF_TOKEN")
  net+=(-e "HF_XET_HIGH_PERFORMANCE=${HF_XET_HIGH_PERFORMANCE:-1}")
  # ENGINE_ENV: engine settings as KEY=VALUE words (the same on both ranks), e.g. "TF_GLM_PREFILL_ROWS=1024"
  local kv
  for kv in ${ENGINE_ENV:-}; do net+=(-e "$kv"); done
  if [ "$NODES" = 2 ]; then
    net+=(--device /dev/infiniband --cap-add IPC_LOCK
         -e "NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME" -e "NCCL_IB_HCA=$NCCL_IB_HCA"
         -e "GLOO_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME")
    [ -n "${NCCL_IB_GID_INDEX:-}" ] && net+=(-e "NCCL_IB_GID_INDEX=$NCCL_IB_GID_INDEX")
  fi
  run_on "$node" mkdir -p "$HF_CACHE" "$KERNEL_CACHE"
  run_on "$node" docker run "${flags[@]}" \
    --gpus all --ipc=host --network host --ulimit memlock=-1 --ulimit stack=67108864 \
    "${net[@]}" \
    -v "$HF_CACHE:/root/.cache/huggingface" \
    -v "$KERNEL_CACHE:/kernel-cache" \
    -v "$KERNEL_CACHE/tensorfold:/root/.cache/tensorfold" \
    "$TF_IMAGE" "$@"
}

cmd_setup() {
  require_nodes_supported
  local node
  for node in $(nodes); do
    if run_on "$node" docker image inspect "$TF_IMAGE" >/dev/null 2>&1; then
      ok "Spark $node has $TF_IMAGE"
    else
      log "building $TF_IMAGE on Spark $node (pulls nvcr.io/nvidia/pytorch:26.07-py3, ~20 GB, once)"
      if [ "$TENSORFOLD_SOURCE" = local ]; then
        # engine/ goes over as the build context (a tar stream), with our Dockerfile added to it.
        tar -C "$ENGINE_DIR" --exclude .git --exclude '__pycache__' --exclude '.pytest_cache' -cf - . \
            -C "$REPO_ROOT/docker/tensorfold-spark" Dockerfile.local \
          | run_on "$node" docker build -t "$TF_IMAGE" -f Dockerfile.local -
      else
        run_on "$node" docker build -t "$TF_IMAGE" --build-arg "TENSORFOLD_COMMIT=$TENSORFOLD_COMMIT" - \
          < "$REPO_ROOT/docker/tensorfold-spark/Dockerfile"
      fi
      ok "Spark $node: $TF_IMAGE built"
    fi
  done
  tf_docker 1 --rm -- tensorfold info "$MODEL" || warn "tensorfold info could not read $MODEL"
}

cmd_pull() {
  require_nodes_supported
  local repos="$MODEL ${DRAFTERS:-}" node pids=""
  log "pulling: $repos"
  # While another model serves, download inside a memory cap so the page cache and xet buffers cannot starve it:
  #   PULL_ONLY=1 PULL_MEMORY_LIMIT=4g HF_XET_HIGH_PERFORMANCE=0 ./pull.sh
  local cap=()
  [ -n "${PULL_MEMORY_LIMIT:-}" ] && cap=(--memory "$PULL_MEMORY_LIMIT")
  if [ "$NODES" = 2 ] && [ "$WEIGHTS_SYNC" = pull ]; then
    for node in 1 2; do
      # shellcheck disable=SC2086
      tf_docker "$node" --rm "${cap[@]}" -- tensorfold pull $repos & pids="$pids $!"
    done
    for p in $pids; do wait "$p" || die "a pull failed"; done
  else
    # shellcheck disable=SC2086
    tf_docker 1 --rm "${cap[@]}" -- tensorfold pull $repos
  fi
  # Only our own model folders change owner: other tools' files in the same cache stay as they are.
  local own=() r
  for r in $repos; do own+=("$HF_CACHE/hub/$(hf_dirname "$r")"); done
  fix_owner 1 "$TF_IMAGE" "${own[@]}"
  if [ "${PULL_ONLY:-0}" = 1 ]; then ok "downloaded on Spark 1 (PULL_ONLY: no split, no copy to Spark 2). Run ./pull.sh again later"; return; fi
  [ "$NODES" = 2 ] && [ "$WEIGHTS_SYNC" = pull ] && fix_owner 2 "$TF_IMAGE" "${own[@]}"
  if [ "${RANK_SPLIT:-0}" = 1 ]; then
    split_ranks
    fix_owner 1 "$TF_IMAGE" "$HF_CACHE/tensorfold-splits"
    [ "$WEIGHTS_SYNC" = pull ] && fix_owner 2 "$TF_IMAGE" "$HF_CACHE/tensorfold-splits"
  fi
  if [ "$NODES" = 2 ] && [ "$WEIGHTS_SYNC" = rsync ]; then
    local r
    for r in ${DRAFTERS:-}; do rsync_model_to_spark2 "$(hf_dirname "$r")"; done
    if [ "${RANK_SPLIT:-0}" = 1 ]; then
      run_on 2 mkdir -p "$HF_CACHE/tensorfold-splits/$RECIPE_NAME"
      log "copying rank 1's half to Spark 2"
      rsync -a --info=progress2 --inplace -e "ssh -T -o BatchMode=yes" \
        "$HF_CACHE/tensorfold-splits/$RECIPE_NAME/rank1/" \
        "$(ssh -G "$SPARK2_SSH" | awk '/^user /{print $2}')@${SPARK2_LINK_IP}:$HF_CACHE/tensorfold-splits/$RECIPE_NAME/rank1/"
    else
      rsync_model_to_spark2 "$(hf_dirname "$MODEL")"
    fi
  fi
  ok "weights ready. Next: ./start.sh"
}

# Write each rank's half of the checkpoint once (GLM): a rank then loads 91 GB instead of reading 182.
split_ranks() {
  local r node
  for r in 0 1; do
    node=1; [ "$WEIGHTS_SYNC" = pull ] && [ "$r" = 1 ] && node=2
    # rank 0 also keeps the vision tower (vision.safetensors, 1 GB): an older split gets it added
    if run_on "$node" test -f "$HF_CACHE/tensorfold-splits/$RECIPE_NAME/rank$r/.complete" && { [ "$r" = 1 ] ||
        run_on "$node" test -f "$HF_CACHE/tensorfold-splits/$RECIPE_NAME/rank$r/vision.safetensors"; }; then
      ok "rank $r half already on Spark $node"; continue
    fi
    log "writing rank $r's half on Spark $node (about 91 GB, a few minutes)"
    tf_docker "$node" --rm -- bash -c "set -e
      snap=\$(python -c 'from huggingface_hub import snapshot_download as s; print(s(\"$MODEL\", local_files_only=True))')
      python -m tensorfold.families.glm5_next.cuda.split \"\$snap\" --rank $r $SPLIT_ROOT/rank$r
      touch $SPLIT_ROOT/rank$r/.complete"
  done
}

serve_cmd() {  # serve_cmd RANK
  local r="$1" model="$MODEL"
  [ "${RANK_SPLIT:-0}" = 1 ] && model="$SPLIT_ROOT/rank$r"
  local a=(tensorfold serve "$model" --name "$SERVED_NAME" --no-update-check)
  [ -n "${CONTEXT:-}" ] && a+=(--context "$CONTEXT")
  # The reply length a request gets when it names none (TensorFold's own default is 4,096, which cuts off
  # agents writing whole files with thinking on); capped by what the context has left.
  a+=(--max-tokens "${MAX_TOKENS:-32768}")
  if [ "$NODES" = 2 ]; then a+=(--tp 2 --rank "$r" --master "$SPARK1_LINK_IP" --master-port "$MASTER_PORT"); fi
  [ "$r" = 0 ] && a+=(--host 0.0.0.0 --port "$PORT")
  # shellcheck disable=SC2206
  a+=(${SERVE_ARGS:-})
  printf '%s\n' "${a[@]}"
}

start_rank() {  # start_rank RANK NODE
  local r="$1" node="$2" args=()
  while IFS= read -r line; do args+=("$line"); done < <(serve_cmd "$r")
  log "Spark $node, rank $r: ${args[*]}"
  tf_docker "$node" -d --name "$(container_name "$r")" --label "$LABEL_KEY=$RECIPE_NAME" -- "${args[@]}" >/dev/null
}

# The largest window a refused start named (both ranks agree on it), or nothing.
refused_window() {
  local r node
  for r in 0 1; do
    node=1; [ "$r" = 1 ] && node=2
    [ "$r" = 1 ] && [ "$NODES" != 2 ] && continue
    run_on "$node" docker logs --tail 20 "$(container_name "$r")" 2>&1 |
      sed -n 's/.*largest fitting prompt-plus-reply window: \([0-9][0-9]*\) tokens.*/\1/p' | tail -1
  done | sort -n | head -1
}

cmd_start() {
  require_nodes_supported
  run_on 1 docker image inspect "$TF_IMAGE" >/dev/null 2>&1 || die "image missing. Run ./setup.sh first"
  ensure_gpus_free
  local attempt fit
  for attempt in 1 2 3; do
    if [ "$NODES" = 2 ]; then
      start_rank 1 2           # rank 1 first: rank 0 is the rendezvous and waits for it
      start_rank 0 1
    else
      start_rank 0 1
    fi
    log "loading (first start also compiles kernels; GLM ~4 min, Flash Next ~90 s, 27B ~1-2 min)"
    local watch2=""
    [ "$NODES" = 2 ] && watch2="$(container_name 1)"
    if wait_http "http://127.0.0.1:$PORT/health" "$STARTUP_TIMEOUT" "$(container_name 0)" 1 "$watch2" 2; then
      echo >&2
      print_endpoint
      return 0
    fi
    # Free memory varies with the page cache (GB10 counts it as used): a CONTEXT that just missed starts again
    # with the window TensorFold names, when that still holds CONTEXT_MIN tokens.
    fit=$(refused_window)
    # (3% under it: the page cache moves between attempts)
    if [ "$attempt" != 3 ] && [ -n "$fit" ] && [ "${CONTEXT:-0}" != 0 ] && [ "$fit" -lt "${CONTEXT}" ] &&
        [ "$fit" -ge "${CONTEXT_MIN:-262144}" ]; then
      CONTEXT=$(( fit * 97 / 100 / 1024 * 1024 ))
      warn "free memory now holds a ${fit}-token window, less than CONTEXT: starting with ${CONTEXT}"
      cmd_stop >/dev/null 2>&1 || true
      continue
    fi
    [ "$NODES" = 2 ] && { warn "rank 1 log:"; run_on 2 docker logs --tail 30 "$(container_name 1)" 2>&1 | sed 's/^/    /' >&2 || true; }
    die "not ready. ./logs.sh shows why; ./stop.sh cleans up"
  done
}

cmd_stop() {
  local node ids
  for node in 1 2; do   # both Sparks, whatever NODES says: a stale rank 1 would block the next start
    ids=$(run_on "$node" docker ps -aq --filter "label=$LABEL_KEY=$RECIPE_NAME" 2>/dev/null || true)
    if [ -n "$ids" ]; then
      # shellcheck disable=SC2086
      run_on "$node" docker rm -f $ids >/dev/null && ok "stopped on Spark $node"
    fi
  done
}

cmd_logs() {
  local r="${1:-0}" node=1
  [ "$r" = 1 ] && node=2
  run_on "$node" docker logs -f --tail 200 "$(container_name "$r")"
}

cmd_status() {
  local node
  for node in $(nodes); do
    echo "Spark $node:"; run_on "$node" docker ps --filter "label=$LABEL_KEY" --format '  {{.Names}}  {{.Status}}' || true
  done
  if curl -fsS -m 5 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    ok "API up at http://127.0.0.1:$PORT/v1"; curl -fsS "http://127.0.0.1:$PORT/v1/models"; echo
  else warn "API not answering on :$PORT"; fi
}
