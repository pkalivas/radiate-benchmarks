#!/usr/bin/env bash
# Java runner: Jenetics and jMetal.
# Contract (see schema.md): run.sh --spec <dir> --out <dir>
#   exit 0 = ok, 3 = toolchain missing (skipped), anything else = failed.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Jenetics 9 is compiled for Java 25. Prefer a registered JDK 25, else Homebrew's openjdk.
JAVA_HOME="$(/usr/libexec/java_home -F -v 25 2>/dev/null || true)"
if [ -z "$JAVA_HOME" ] && command -v brew >/dev/null 2>&1; then
    candidate="$(brew --prefix openjdk 2>/dev/null)/libexec/openjdk.jdk/Contents/Home"
    [ -x "$candidate/bin/java" ] && JAVA_HOME="$candidate"
fi
if [ -z "$JAVA_HOME" ]; then
    echo "JDK 25 not found (brew install openjdk)" >&2
    exit 3
fi
if ! command -v mvn >/dev/null 2>&1; then
    echo "mvn not found on PATH (brew install maven)" >&2
    exit 3
fi
export JAVA_HOME

cd "$HERE"
mvn -q -B package
exec "$JAVA_HOME/bin/java" -jar "$HERE/target/java-runner.jar" "$@"
