#!/usr/bin/env bash
exec "$(dirname "$0")/../../lib/recipe.sh" logs "$(dirname "$0")" "$@"
