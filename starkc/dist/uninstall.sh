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
if [ -L "$prefix/lib/stark/current" ]; then
    current_target=$(readlink "$prefix/lib/stark/current")
    case "$current_target" in
        versions/*)
            rm -rf "$prefix/lib/stark/$current_target"
            ;;
    esac
fi
rm -f "$prefix/lib/stark/current"
rm -f "$prefix/lib/stark/uninstall.sh"
rm -rf "$prefix/lib/stark/stark-runtime"
rm -rf "$prefix/lib/stark/stark-provider-abi"
rmdir "$prefix/lib/stark/versions" 2>/dev/null || true
rmdir "$prefix/lib/stark" 2>/dev/null || true

echo "Removed STARK from $prefix"
