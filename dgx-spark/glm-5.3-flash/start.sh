#!/usr/bin/env bash
exec "$(dirname "$0")/../../lib/recipe.sh" start "$(dirname "$0")" "$@"
