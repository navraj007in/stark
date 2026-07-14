#!/usr/bin/env bash
# Extract every ```stark code block from the normative spec into individual
# fixture files, for use as a parser test corpus.
#
# Usage: tools/extract-spec-examples.sh [output-dir]
#
# Output: <output-dir>/<spec-file>__NN.stark  (default: tests/spec-fixtures/)
#
# Note: some blocks are intentionally non-compilable (error demonstrations
# marked with "// Error:" comments, and API-notation listings in
# 06-Standard-Library.md — see its "Notation" section). When the parser
# lands, triage fixtures into expect-pass/ and expect-fail/ sets; blocks
# containing "// Error:" are expect-fail candidates.

set -euo pipefail

SPEC_DIR="$(cd "$(dirname "$0")/../docs/spec" && pwd)"
OUT_DIR="${1:-$(dirname "$0")/../tests/spec-fixtures}"
mkdir -p "$OUT_DIR"
rm -f "$OUT_DIR"/*.stark

total=0
for f in "$SPEC_DIR"/0[0-7]-*.md; do
  base="$(basename "$f" .md)"
  count=$(awk -v base="$base" -v out="$OUT_DIR" '
    /^```stark$/ { inblock=1; n++; file=sprintf("%s/%s__%02d.stark", out, base, n); next }
    /^```/       { inblock=0; next }
    inblock      { print > file }
    END          { print n+0 }
  ' "$f")
  echo "$base: $count blocks"
  total=$((total + count))
done
echo "Extracted $total stark blocks into $OUT_DIR"
