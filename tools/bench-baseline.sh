#!/usr/bin/env bash
# Measure a server this project does not run (e.g. Mia-AiLab's vLLM GLM on the Sparks) with the same benchmark,
# so TensorFold's numbers have a baseline measured the same way. Only sends OpenAI requests; changes nothing.
#
#   ./tools/bench-baseline.sh http://<spark1>:8888 glm-5.3-flash "vLLM Mia EXL3"
#   QUICK=1 ./tools/bench-baseline.sh ...     standard + context up to 128k only (~15 min)
#
# The full run takes about 30 minutes (cold prompts up to 128k tokens). Nothing else should use the server meanwhile.
set -euo pipefail
url="${1:?usage: bench-baseline.sh URL RECIPE LABEL}" recipe="${2:?recipe name}" label="${3:?label}"
cd "$(dirname "$0")/.."
max=$(curl -fsS "$url/v1/models" | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0].get("max_model_len") or 32768)')
running=$(curl -fsS -m 5 "$url/metrics" 2>/dev/null | awk '/^vllm:num_requests_running/{s+=$2} END{print s+0}')
[ "${running:-0}" = 0 ] || echo "!! the server is running $running request(s) right now: results will be disturbed and so will that user" >&2
if [ "${QUICK:-0}" = 1 ]; then suites=standard,context; sizes=32768,131072
else suites=standard,long,context,concurrency; sizes=32768,131072; fi
exec python3 bench/bench.py "$url" --recipe "$recipe" --label "$label" --suites "$suites" \
  --max-context "$max" --context-sizes "$sizes" --concurrency 1,2,4 \
  --meta "platform=${PLATFORM:-2x DGX Spark}" --meta "engine=${ENGINE:-baseline (external)}"
