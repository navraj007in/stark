#!/bin/sh
set -eu

prefix="${STARK_INSTALL_PREFIX:-$HOME/.local}"

if [ "${1:-}" = "--prefix" ]; then
    if [ "$#" -ne 2 ] || [ -z "$2" ]; then
        echo "usage: uninstall.sh [--prefix <directory>]" >&2
        exit 2
    fi
    prefix=$2
elif [ "$#" -ne 0 ]; then
    echo "usage: uninstall.sh [--prefix <directory>]" >&2
    exit 2
fi

rm -f "$prefix/bin/stark" "$prefix/bin/starkc" "$prefix/bin/starkide"
rm -rf "$prefix/lib/stark/stark-runtime"
rm -f "$prefix/lib/stark/uninstall.sh"
rmdir "$prefix/lib/stark" 2>/dev/null || true

echo "Removed STARK from $prefix"
