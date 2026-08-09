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

# The mirror layout writes the runtime under `starkc/`; packages built before that move wrote it
# flat. Accept either, so this installer can still lay down an older package.
if [ -f "$package_dir/lib/stark/starkc/stark-runtime/Cargo.toml" ] &&
   [ -f "$package_dir/lib/stark/starkc/stark-provider-abi/Cargo.toml" ]; then
    runtime_present=yes
elif [ -f "$package_dir/lib/stark/stark-runtime/Cargo.toml" ] &&
     [ -f "$package_dir/lib/stark/stark-provider-abi/Cargo.toml" ]; then
    runtime_present=yes
else
    runtime_present=no
fi

if [ ! -x "$package_dir/bin/stark" ] ||
   [ ! -f "$package_dir/manifest.json" ] ||
   [ "$runtime_present" != yes ]; then
    echo "error: install.sh must be run from an extracted STARK release package" >&2
    exit 1
fi

version=$(sed -n 's/.*"stark_version": "\([^"]*\)".*/\1/p' "$package_dir/manifest.json" | head -n 1)
if [ -z "$version" ]; then
    echo "error: manifest.json does not declare stark_version" >&2
    exit 1
fi

install_root="$prefix/lib/stark"
version_root="$install_root/versions/$version"
staging_root="$install_root/.staging-$version-$$"
mkdir -p "$prefix/bin" "$install_root/versions"
rm -rf "$staging_root"
mkdir -p "$staging_root"

cp -R "$package_dir/bin" "$staging_root/bin"
cp -R "$package_dir/lib" "$staging_root/lib"
cp "$package_dir/manifest.json" "$staging_root/manifest.json"
cp "$package_dir/BUILD-INFO.txt" "$staging_root/BUILD-INFO.txt"
cp "$package_dir/LICENSE" "$staging_root/LICENSE"
cp "$package_dir/README.md" "$staging_root/README.md"
cp "$package_dir/install.sh" "$staging_root/install.sh"
cp "$package_dir/uninstall.sh" "$staging_root/uninstall.sh"
chmod 755 "$staging_root/bin/stark" "$staging_root/bin/starkc" "$staging_root/bin/starkide" "$staging_root/install.sh" "$staging_root/uninstall.sh"

checksum_program=""
if command -v shasum >/dev/null 2>&1; then
    checksum_program="shasum -a 256"
elif command -v sha256sum >/dev/null 2>&1; then
    checksum_program="sha256sum"
fi
if [ -z "$checksum_program" ]; then
    echo "error: install.sh requires shasum or sha256sum to verify manifest files" >&2
    rm -rf "$staging_root"
    exit 1
fi
check_file="$staging_root/.manifest-checksums"
awk '
    /"path":/ {
        path = $0
        sub(/^.*"path": "/, "", path)
        sub(/",?$/, "", path)
    }
    /"sha256":/ {
        sha = $0
        sub(/^.*"sha256": "/, "", sha)
        sub(/",?$/, "", sha)
        if (path != "") {
            print sha "  '"$staging_root"'/" path
        }
    }
' "$staging_root/manifest.json" > "$check_file"
if ! $checksum_program -c "$check_file" >/dev/null; then
    echo "error: staged STARK installation failed manifest verification" >&2
    rm -rf "$staging_root"
    exit 1
fi
rm -f "$check_file"

rm -rf "$version_root"
mv "$staging_root" "$version_root"
ln -sfn "versions/$version" "$install_root/current"

for binary in stark starkc starkide; do
    rm -f "$prefix/bin/$binary"
    if ln -s "../lib/stark/current/bin/$binary" "$prefix/bin/$binary" 2>/dev/null; then
        :
    else
        cp "$version_root/bin/$binary" "$prefix/bin/$binary"
        chmod 755 "$prefix/bin/$binary"
    fi
done
ln -sfn "current/uninstall.sh" "$install_root/uninstall.sh"

echo "Installed STARK in $prefix"
echo "Version: $version"
echo "Run: $prefix/bin/stark --help"
case ":${PATH:-}:" in
    *":$prefix/bin:"*) ;;
    *) echo "Add $prefix/bin to PATH to invoke 'stark' directly." ;;
esac
