#!/usr/bin/env bash
# Fetch and verify the Gate 5 reference image.
set -euo pipefail

URL="https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"
SHA256="f3f87bb8ab3c26c7ecfd3ac60421d7f32b0503d1d6c5baf8bac42ed93d86351a"
NAME="dog.jpg"
DEST="${1:-${TMPDIR:-/tmp}}"
OUT="${DEST%/}/${NAME}"

if [ ! -f "${OUT}" ]; then
  echo "Fetching ${NAME} -> ${OUT}"
  curl -fSL --retry 3 -o "${OUT}" "${URL}"
fi

# Portable SHA-256 verification (macOS shasum / Linux sha256sum).
if command -v sha256sum >/dev/null 2>&1; then
  echo "${SHA256}  ${OUT}" | sha256sum -c -
else
  echo "${SHA256}  ${OUT}" | shasum -a 256 -c -
fi

echo "OK: ${OUT}"
