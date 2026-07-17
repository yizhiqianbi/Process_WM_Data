#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  printf 'Usage: %s OUTPUT.tar.gz\n' "$0" >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PROJECT_PARENT="$(dirname -- "$PROJECT_ROOT")"
PROJECT_NAME="$(basename -- "$PROJECT_ROOT")"
OUTPUT="$(realpath -m -- "$1")"

if [[ "$OUTPUT" == "$PROJECT_ROOT" || "$OUTPUT" == "$PROJECT_ROOT/"* ]]; then
  printf 'Output archive must be outside the project directory: %s\n' "$OUTPUT" >&2
  exit 2
fi

mkdir -p -- "$(dirname -- "$OUTPUT")"
tar \
  --exclude="$PROJECT_NAME/work" \
  --exclude="$PROJECT_NAME/.pytest_cache" \
  --exclude="$PROJECT_NAME/.fastwam_download" \
  --exclude='*/__pycache__' \
  --exclude='*.pyc' \
  --exclude='*.pyo' \
  --exclude='*.token' \
  --exclude='*token*.txt' \
  -czf "$OUTPUT" \
  -C "$PROJECT_PARENT" \
  "$PROJECT_NAME"

printf '%s\n' "$OUTPUT"
