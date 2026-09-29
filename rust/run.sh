#!/usr/bin/env bash
# Rust runner: radiate (native), built against the local checkout at ../../radiate.
# Contract (see schema.md): run.sh --spec <dir> --out <dir>
#   exit 0 = ok, 3 = toolchain missing (skipped), anything else = failed.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RADIATE_DIR="$(cd "$HERE/../../radiate" 2>/dev/null && pwd || true)"

if ! command -v cargo >/dev/null 2>&1; then
    echo "cargo not found on PATH" >&2
    exit 3
fi
if [ -z "$RADIATE_DIR" ]; then
    echo "local radiate checkout not found at $HERE/../../radiate" >&2
    exit 3
fi

cd "$HERE"
cargo build --release --quiet

# Crate version plus the local checkout's commit (and "-dirty" if it has local edits),
# since a path dependency's version number alone doesn't pin what was benchmarked.
VERSION="$(cargo pkgid -p radiate | sed -E 's/.*[#@]//')"
SHA="$(git -C "$RADIATE_DIR" rev-parse --short HEAD 2>/dev/null || echo unknown)"
if [ -n "$(git -C "$RADIATE_DIR" status --porcelain 2>/dev/null)" ]; then
    SHA="$SHA-dirty"
fi

exec "$HERE/target/release/radiate-benchmarks-runner" --library-version "$VERSION+$SHA" "$@"
