#!/bin/sh
set -eu

prefix="${STARK_INSTALL_PREFIX:-$HOME/.local}"

if [ "${1:-}" = "--prefix" ]; then
    if [ "$#" -ne 2 ] || [ -z "$2" ]; then
        echo "usage: ./install.sh [--prefix <directory>]" >&2
        exit 2
    fi
    prefix=$2
elif [ "$#" -ne 0 ]; then
    echo "usage: ./install.sh [--prefix <directory>]" >&2
    exit 2
fi

package_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

if [ ! -x "$package_dir/bin/stark" ] ||
   [ ! -f "$package_dir/lib/stark/stark-runtime/Cargo.toml" ]; then
    echo "error: install.sh must be run from an extracted STARK release package" >&2
    exit 1
fi

mkdir -p "$prefix/bin" "$prefix/lib/stark"
cp "$package_dir/bin/stark" "$prefix/bin/stark"
cp "$package_dir/bin/starkc" "$prefix/bin/starkc"
cp "$package_dir/bin/starkide" "$prefix/bin/starkide"
chmod 755 "$prefix/bin/stark" "$prefix/bin/starkc" "$prefix/bin/starkide"

runtime_dest="$prefix/lib/stark/stark-runtime"
rm -rf "$runtime_dest"
mkdir -p "$runtime_dest"
cp "$package_dir/lib/stark/stark-runtime/Cargo.toml" "$runtime_dest/Cargo.toml"
cp -R "$package_dir/lib/stark/stark-runtime/src" "$runtime_dest/src"
cp "$package_dir/uninstall.sh" "$prefix/lib/stark/uninstall.sh"
chmod 755 "$prefix/lib/stark/uninstall.sh"

echo "Installed STARK in $prefix"
echo "Run: $prefix/bin/stark --help"
case ":${PATH:-}:" in
    *":$prefix/bin:"*) ;;
    *) echo "Add $prefix/bin to PATH to invoke 'stark' directly." ;;
esac
