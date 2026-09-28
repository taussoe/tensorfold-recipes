#!/usr/bin/env bash
# The one entry point behind every recipe folder's scripts:
#
#   lib/recipe.sh COMMAND RECIPE_DIR [ARGS...]
#
# COMMAND: setup | pull | start | stop | logs | status | chat | bench
# Each recipe folder has a small script per command (start.sh, bench.sh, ...) that calls this.
set -euo pipefail

. "$(dirname "${BASH_SOURCE[0]}")/common.sh"

cmd="${1:?usage: lib/recipe.sh COMMAND RECIPE_DIR [ARGS...]}"
load_recipe "$(cd "${2:?recipe dir}" && pwd)"
shift 2

case "$ENGINE" in
  tensorfold-spark) load_cluster; . "$REPO_ROOT/lib/engine-tensorfold-spark.sh" ;;
  tensorfold-mac)   . "$REPO_ROOT/lib/engine-tensorfold-mac.sh" ;;
  *) die "unknown ENGINE=$ENGINE in recipe.env" ;;
esac

cmd_chat() {
  local prompt="${*:-Say hello in one sentence.}" url="${BENCH_URL:-http://127.0.0.1:$PORT}"
  PROMPT="$prompt" MODEL_ID="$SERVED_MODEL" URL="$url" python3 - <<'EOF'
import json, os, time, urllib.request
body = {"model": os.environ["MODEL_ID"], "max_tokens": 512, "stream": False,
        "messages": [{"role": "user", "content": os.environ["PROMPT"]}],
        # Thinking off for a quick check; each template reads its own key and ignores the others.
        "chat_template_kwargs": {"enable_thinking": False, "thinking": False}}
req = urllib.request.Request(os.environ["URL"] + "/v1/chat/completions", data=json.dumps(body).encode(),
                             headers={"Content-Type": "application/json"})
t = time.perf_counter()
out = json.load(urllib.request.urlopen(req, timeout=600))
dt = time.perf_counter() - t
print(out["choices"][0]["message"].get("content") or out["choices"][0]["message"])
u = out.get("usage", {})
print(f"\n[{u.get('completion_tokens', '?')} tokens in {dt:.1f} s]", end="")
tf = out.get("tensorfold") or {}
if tf.get("tokens_per_second"): print(f"  decode {tf['tokens_per_second']:.1f} tok/s", end="")
spec = out.get("speculative") or {}
if spec.get("drafted"): print(f"  drafts accepted {spec.get('accepted')}/{spec.get('drafted')}", end="")
print()
EOF
}

cmd_bench() {
  local url="${BENCH_URL:-http://127.0.0.1:$PORT}" suites="${BENCH_SUITES:-standard,long,prefill}" extra=""
  case "$ENGINE" in
    tensorfold-*) suites="$suites,exactness"; extra="--meta tensorfold_source=$TENSORFOLD_SOURCE --meta tensorfold_image=${TF_IMAGE:-} --meta tensorfold_commit=${TENSORFOLD_COMMIT:0:7}" ;;
  esac
  case "$ENGINE" in
    # macOS energy mode: 0 automatic, 1 low power, 2 high power. It changes sustained speed by up to 3x.
    tensorfold-mac) extra="$extra --meta powermode=$(pmset -g 2>/dev/null | awk '/powermode/{print $2}')" ;;
  esac
  python3 "$REPO_ROOT/bench/bench.py" "$url" \
    --recipe "$RECIPE_NAME" --model "$SERVED_MODEL" --suites "$suites" \
    --max-context "${BENCH_MAX_CONTEXT:-32768}" --prefill-sizes "${BENCH_PREFILL_SIZES:-1024,4096,16384,32768}" \
    --context-sizes "${BENCH_CONTEXT_SIZES:-32768,131072}" \
    --meta "engine=$ENGINE" --meta "nodes=${NODES:-1}" --meta "checkpoint=$MODEL" \
    --meta "platform=$([ "$ENGINE" = tensorfold-mac ] && sysctl -n machdep.cpu.brand_string || echo "${NODES:-1}x DGX Spark")" \
    $extra "$@"
}

"cmd_$cmd" "$@"
