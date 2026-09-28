#!/usr/bin/env bash
exec "$(dirname "$0")/../../lib/recipe.sh" pull "$(dirname "$0")" "$@"
