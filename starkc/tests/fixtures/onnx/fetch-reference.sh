#!/usr/bin/env bash
# Fetch and verify the Gate 4 reference computer-vision model.
#
# The model is a large binary and is deliberately NOT committed to Git
# (GEMINI_GATE4_IMPLEMENTATION.md §M4.0). This script downloads it on demand
# for the Gate 4 exit demonstration and verifies its SHA-256 before use.
#
# Usage:  starkc/tests/fixtures/onnx/fetch-reference.sh [DEST_DIR]
# Default DEST_DIR is the system temp dir; the artifact is never written into
# the repository.
set -euo pipefail

URL="https://github.com/onnx/models/raw/main/validated/vision/classification/resnet/model/resnet50-v1-7.onnx"
SHA256="af16a04a6ec48ac494065d4439fe9dea590d337b9ca6dc328160ccf04a217b9c"
NAME="resnet50-v1-7.onnx"
DEST="${1:-${TMPDIR:-/tmp}}"
OUT="${DEST%/}/${NAME}"

echo "Fetching ${NAME} -> ${OUT}"
curl -fSL --retry 3 -o "${OUT}" "${URL}"

# Portable SHA-256 verification (macOS shasum / Linux sha256sum).
if command -v sha256sum >/dev/null 2>&1; then
  echo "${SHA256}  ${OUT}" | sha256sum -c -
else
  echo "${SHA256}  ${OUT}" | shasum -a 256 -c -
fi

echo "OK: ${OUT}"
echo "Reference signature:  input Float32 [B, 3, 224, 224]  ->  output Float32 [B, 1000]"
