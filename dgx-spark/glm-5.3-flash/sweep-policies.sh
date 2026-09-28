#!/usr/bin/env bash
# GLM's draft policy is chosen per request (a suffix on the model id), so one running server can compare them
# all. Runs the standard suite once per policy; compare with: python3 bench/compare.py --recipe glm-5.3-flash
#   ./sweep-policies.sh                         the policies TensorFold measured
#   ./sweep-policies.sh "@c2:0.4" "@a:0.5:0.8"  your own (the table in README.md explains the syntax)
set -euo pipefail
cd "$(dirname "$0")"
policies=("$@")
if [ ${#policies[@]} -eq 0 ]; then
  policies=("" "@a:0.6:0.85" "@c3:0.35")
  grep -q '^ACCEPT_NONCOMMERCIAL_DRAFTER=1' recipe.env 2>/dev/null || [ "${ACCEPT_NONCOMMERCIAL_DRAFTER:-0}" = 1 ] \
    && policies+=("@fc5:0.3")
fi
for p in "${policies[@]}"; do
  BENCH_SUITES=standard ./bench.sh --model-suffix "$p" --label "policy ${p:-auto}"
done
python3 ../../bench/compare.py --recipe glm-5.3-flash --no-write
