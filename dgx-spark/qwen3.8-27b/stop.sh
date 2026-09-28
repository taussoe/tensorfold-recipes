#!/usr/bin/env bash
exec "$(dirname "$0")/../../lib/recipe.sh" stop "$(dirname "$0")" "$@"
