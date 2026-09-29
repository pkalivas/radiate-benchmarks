#!/usr/bin/env bash
# Python runner: radiate (python bindings), DEAP, pymoo, PyGAD.
# Contract (see schema.md): run.sh --spec <dir> --out <dir>
#   exit 0 = ok, 3 = toolchain missing (skipped), anything else = failed.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"

if ! command -v uv >/dev/null 2>&1; then
    echo "uv not found on PATH" >&2
    exit 3
fi

cd "$ROOT"
uv sync --quiet
PYTHONPATH="$ROOT:$HERE" exec uv run --quiet python "$HERE/runner.py" "$@"
