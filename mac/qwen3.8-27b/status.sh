#!/usr/bin/env bash
exec "$(dirname "$0")/../../lib/recipe.sh" status "$(dirname "$0")" "$@"
