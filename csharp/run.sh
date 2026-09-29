#!/usr/bin/env bash
# C# runner: GeneticSharp (single-objective only).
# Contract (see schema.md): run.sh --spec <dir> --out <dir>
#   exit 0 = ok, 3 = toolchain missing (skipped), anything else = failed.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! command -v dotnet >/dev/null 2>&1 || ! dotnet --list-sdks | grep -q '^9\.'; then
    echo ".NET 9 SDK not found" >&2
    exit 3
fi

cd "$HERE"
dotnet build -c Release -nologo -v q >/dev/null
exec dotnet "$HERE/bin/Release/net9.0/csharp-runner.dll" "$@"
