#!/usr/bin/env bash
exec "$(dirname "$0")/../../lib/recipe.sh" bench "$(dirname "$0")" "$@"
