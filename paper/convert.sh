#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CALL_DIR="$(pwd)"
if [[ $# -eq 0 ]]; then
  OUTPUT="$SCRIPT_DIR/paper.docx"
elif [[ "$1" = /* ]]; then
  OUTPUT="$1"
else
  OUTPUT="$CALL_DIR/$1"
fi

if ! command -v pandoc >/dev/null 2>&1; then
  echo "Error: pandoc is required but was not found in PATH." >&2
  exit 1
fi

cd "$SCRIPT_DIR"
pandoc paper.tex \
  --from=latex \
  --to=docx \
  --citeproc \
  --bibliography=literatur.bib \
  --resource-path=.:Abbildungen \
  -o "$OUTPUT"

echo "Wrote $OUTPUT"
