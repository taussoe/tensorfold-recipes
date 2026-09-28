#!/usr/bin/env bash
# Benchmark several DGX Spark recipes back to back: start, bench, stop, next. Then print the leaderboard.
# Run on Spark 1 after ./pull.sh for each recipe. Takes 15-40 minutes per model.
#
#   ./tools/bench-all.sh                                   every Spark recipe (two Sparks each)
#   ./tools/bench-all.sh qwen3.8-27b qwen3.8-flash-next    only these
#   NODES=1 ./tools/bench-all.sh qwen3.8-27b              one Spark
set -uo pipefail
cd "$(dirname "$0")/.."
recipes=("$@")
[ ${#recipes[@]} -gt 0 ] || recipes=(qwen3.8-27b qwen3.8-flash-next glm-5.3-flash)
failed=()
for r in "${recipes[@]}"; do
  d="dgx-spark/$r"
  [ -d "$d" ] || { echo "no recipe $d"; failed+=("$r"); continue; }
  echo; echo "================ $r ================"
  if "$d/start.sh" && "$d/bench.sh"; then :; else failed+=("$r"); fi
  "$d/stop.sh" || true
  sleep 20   # let unified memory settle before the next model loads
done
python3 bench/compare.py
[ ${#failed[@]} -eq 0 ] || { echo "failed: ${failed[*]}"; exit 1; }
