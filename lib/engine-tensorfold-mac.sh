#!/usr/bin/env bash
# TensorFold on a Mac with Apple Silicon (MLX + Metal). Sourced by lib/recipe.sh; needs no config/cluster.env.
#
# Recipe variables: MODEL, DRAFTERS, CONTEXT, SERVE_ARGS, SERVED_NAME, PORT, STARTUP_TIMEOUT.

VENV="$REPO_ROOT/.venv-mac"
TF="$VENV/bin/tensorfold"
: "${PORT:=8080}" "${SERVED_NAME:=${MODEL##*/}}" "${STARTUP_TIMEOUT:=600}"
SERVED_MODEL="$SERVED_NAME"
SPARK1_LAN=127.0.0.1
RUN_DIR="$RECIPE_DIR/.run"
PID_FILE="$RUN_DIR/server.pid"
LOG_FILE="$RUN_DIR/server.log"

mac_checks() {
  [ "$(uname -s)" = Darwin ] && [ "$(uname -m)" = arm64 ] || die "this recipe needs a Mac with Apple Silicon"
  local chip mem
  chip=$(sysctl -n machdep.cpu.brand_string); mem=$(( $(sysctl -n hw.memsize) / 1073741824 ))
  log "$chip, ${mem} GB"
  [ "$mem" -ge 32 ] || die "Qwen3.8-27B needs 32 GB or more"
  case "$chip" in
    *M5*) ok "M5 GPU: TensorFold's tensor-unit lane kernels" ;;
    *)    ok "no M5 tensor units: TensorFold's row-exact lane decoder + simdgroup matmul, drafts stay exact" ;;
  esac
  if pmset -g batt 2>/dev/null | grep -q "Battery Power"; then
    warn "on battery: expect ~3x slower. Plug in before benchmarking"
  fi
  case "$(pmset -g 2>/dev/null | awk '/[[:space:]]powermode/{print $2}')" in
    2) ok "High Power energy mode" ;;
    1) warn "Low Power energy mode: expect several times slower. System Settings > Battery > Energy Mode > High Power" ;;
    0) warn "Automatic energy mode: sustained decoding throttles (1.4-4.3x slower measured on an M3 Max)." \
            "Set System Settings > Battery > Energy Mode > High Power, or: sudo pmset -a powermode 2" ;;
  esac
}

cmd_setup() {
  mac_checks
  local stamp="$VENV/.tensorfold-commit"
  if [ ! -x "$TF" ]; then
    if command -v uv >/dev/null; then
      log "creating .venv-mac with uv (Python 3.12)"
      uv venv -q --python 3.12 "$VENV"
    else
      log "creating .venv-mac with python3 (needs 3.11+)"
      python3 -c 'import sys; sys.exit(sys.version_info < (3, 11))' || die "Python 3.11+ needed (brew install python, or install uv)"
      python3 -m venv "$VENV"
    fi
  fi
  # (re)install when the pinned commit changed since the last setup
  if [ "$(cat "$stamp" 2>/dev/null)" != "$TENSORFOLD_COMMIT" ]; then
    log "installing TensorFold ${TENSORFOLD_VERSION} (${TENSORFOLD_COMMIT:0:7})"
    if command -v uv >/dev/null; then
      uv pip install -q --reinstall-package tensorfold --python "$VENV/bin/python" "$TENSORFOLD_PIP"
    else
      "$VENV/bin/pip" install -q --force-reinstall --no-deps "$TENSORFOLD_PIP" && "$VENV/bin/pip" install -q "$TENSORFOLD_PIP"
    fi
    echo "$TENSORFOLD_COMMIT" > "$stamp"
  fi
  ok "$("$TF" --version) in .venv-mac (pinned ${TENSORFOLD_COMMIT:0:7})"
  "$TF" info "$MODEL" || true
}

cmd_pull() {
  [ -x "$TF" ] || die "run ./setup.sh first"
  # shellcheck disable=SC2086
  "$TF" pull "$MODEL" ${DRAFTERS:-}
  ok "weights ready. Next: ./start.sh"
}

serve_args() {
  local a=(serve "$MODEL" --name "$SERVED_NAME" --host "${HOST:-127.0.0.1}" --port "$PORT" --no-update-check)
  [ -n "${CONTEXT:-}" ] && a+=(--context "$CONTEXT")
  # The reply length a request gets when it names none (TensorFold's own default is 4,096, which cuts off
  # agents writing whole files with thinking on); capped by what the context has left.
  a+=(--max-tokens "${MAX_TOKENS:-32768}")
  # shellcheck disable=SC2206
  a+=(${SERVE_ARGS:-})
  printf '%s\n' "${a[@]}"
}

cmd_start() {
  [ -x "$TF" ] || die "run ./setup.sh first"
  mac_checks
  if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then die "already running (pid $(cat "$PID_FILE")). ./stop.sh first"; fi
  curl -fsS -m 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && die "something already answers on :$PORT"
  local args=() line
  while IFS= read -r line; do args+=("$line"); done < <(serve_args)
  mkdir -p "$RUN_DIR"
  if [ "${1:-}" = --fg ]; then log "tensorfold ${args[*]}"; exec "$TF" "${args[@]}"; fi
  log "tensorfold ${args[*]}  (log: ${LOG_FILE#$REPO_ROOT/})"
  nohup "$TF" "${args[@]}" >"$LOG_FILE" 2>&1 &
  echo $! >"$PID_FILE"
  local start; start=$(date +%s)
  until curl -fsS -m 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; do
    kill -0 "$(cat "$PID_FILE")" 2>/dev/null || { tail -30 "$LOG_FILE" >&2; die "the server exited"; }
    [ $(( $(date +%s) - start )) -lt "$STARTUP_TIMEOUT" ] || die "not ready after ${STARTUP_TIMEOUT}s (./logs.sh)"
    printf '.' >&2; sleep 3
  done
  echo >&2
  grep -iE "draft|lane|row|exact" "$LOG_FILE" | tail -5 | sed 's/^/    /' >&2 || true
  print_endpoint
}

cmd_stop() {
  if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    kill "$(cat "$PID_FILE")"; ok "stopped (pid $(cat "$PID_FILE"))"
  else warn "not running"; fi
  rm -f "$PID_FILE"
}

cmd_logs()   { tail -n 200 -f "$LOG_FILE"; }
cmd_status() {
  if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then ok "running, pid $(cat "$PID_FILE")"; else warn "not running"; fi
  curl -fsS -m 3 "http://127.0.0.1:$PORT/v1/models" && echo || true
}
