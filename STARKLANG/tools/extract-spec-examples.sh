#!/usr/bin/env bash
# Extract every ```stark code block from the normative spec into individual
# fixture files, for use as a parser test corpus.
#
# Usage: tools/extract-spec-examples.sh [output-dir]
#
# Output: <output-dir>/<spec-file>__NN.stark  (default: tests/spec-fixtures/)
#
# Every fixture has a triage verdict in <output-dir>/manifest.toml
# (parse-pass / parse-fail / semantic-error / lex-pass / notation), enforced
# by starkc/tests/conformance.rs. After extraction this script fails if the
# extracted file set diverges from the manifest — a spec edit that adds,
# removes, or renumbers ```stark blocks requires re-triaging the manifest in
# the same change.

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

MANIFEST="$OUT_DIR/manifest.toml"
if [[ -f "$MANIFEST" ]]; then
  if ! diff <(grep -o '^\["[^"]*"\]' "$MANIFEST" | sed 's/^\["//; s/"\]$//' | sort) \
            <(cd "$OUT_DIR" && ls -- *.stark | sort); then
    echo "ERROR: extracted fixtures diverge from $MANIFEST (< manifest, > extracted)." >&2
    echo "Re-triage the manifest in the same change as the spec edit." >&2
    exit 1
  fi
  echo "Manifest is in sync with the extracted fixture set."
else
  echo "WARNING: no manifest at $MANIFEST — verdicts not checked." >&2
fi
