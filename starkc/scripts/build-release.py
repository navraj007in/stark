#!/usr/bin/env python3
"""Build and package standalone STARK compiler/IDE binaries.

The script is intentionally dependency-free and runs on Python 3. It builds
one Rust target per invocation; native/cross linkers and the requested Rust
standard-library target must already be installed.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile


CRATE_DIR = Path(__file__).resolve().parent.parent
REPO_DIR = CRATE_DIR.parent
BINARIES = ("starkc", "starkide")


def run(command: list[str], *, capture: bool = False) -> str:
    print("+", " ".join(command), flush=True)
    try:
        result = subprocess.run(
            command,
            cwd=CRATE_DIR,
            check=True,
            text=True,
            stdout=subprocess.PIPE if capture else None,
        )
    except FileNotFoundError as error:
        raise SystemExit(f"required command not found: {command[0]}") from error
    except subprocess.CalledProcessError as error:
        raise SystemExit(error.returncode) from error
    return result.stdout.strip() if capture else ""


def host_target() -> str:
    for line in run(["rustc", "-vV"], capture=True).splitlines():
        if line.startswith("host: "):
            return line.removeprefix("host: ")
    raise SystemExit("could not determine the Rust host target")


def package_version() -> str:
    metadata = json.loads(
        run(["cargo", "metadata", "--no-deps", "--format-version", "1"], capture=True)
    )
    packages = [package for package in metadata["packages"] if package["name"] == "starkc"]
    if len(packages) != 1:
        raise SystemExit("cargo metadata did not contain exactly one starkc package")
    return packages[0]["version"]


def add_tar_tree(archive: tarfile.TarFile, root: Path, archive_root: str) -> None:
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        info = archive.gettarinfo(str(path), f"{archive_root}/{relative.as_posix()}")
        info.uid = info.gid = 0
        info.uname = info.gname = ""
        info.mtime = 0
        if path.is_file():
            with path.open("rb") as source:
                archive.addfile(info, source)
        else:
            archive.addfile(info)


def create_tar_gz(staging: Path, output: Path, archive_root: str) -> None:
    with output.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as archive:
                add_tar_tree(archive, staging, archive_root)


def create_zip(staging: Path, output: Path, archive_root: str) -> None:
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(staging.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(staging).as_posix()
            info = zipfile.ZipInfo(f"{archive_root}/{relative}", (1980, 1, 1, 0, 0, 0))
            mode = 0o755 if path.name in BINARIES or path.suffix == ".exe" else 0o644
            info.external_attr = mode << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED)


def write_checksum(archive: Path) -> Path:
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    checksum = archive.with_name(f"{archive.name}.sha256")
    checksum.write_text(f"{digest}  {archive.name}\n", encoding="utf-8", newline="\n")
    return checksum


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and package the starkc and starkide release binaries."
    )
    parser.add_argument(
        "--target",
        help="Rust target triple; defaults to the current host target.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=CRATE_DIR / "target" / "packages",
        help="Package output directory (default: starkc/target/packages).",
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip the native test suite before building the release package.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    target = args.target or host_target()
    version = package_version()
    windows = "windows" in target
    executable_suffix = ".exe" if windows else ""
    package_name = f"stark-{version}-{target}"
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_tests:
        run(["cargo", "test", "--locked", "--all-targets", "--all-features"])

    run(
        [
            "cargo",
            "build",
            "--release",
            "--locked",
            "--all-features",
            "--bins",
            "--target",
            target,
        ]
    )

    release_dir = CRATE_DIR / "target" / target / "release"
    with tempfile.TemporaryDirectory(prefix="stark-release-") as temporary:
        staging = Path(temporary)
        for binary in BINARIES:
            source = release_dir / f"{binary}{executable_suffix}"
            if not source.is_file():
                raise SystemExit(f"expected release binary was not produced: {source}")
            destination = staging / source.name
            shutil.copy2(source, destination)
            if not windows:
                destination.chmod(0o755)

        shutil.copy2(REPO_DIR / "LICENSE", staging / "LICENSE")
        shutil.copy2(CRATE_DIR / "README.md", staging / "README.md")
        (staging / "BUILD-INFO.txt").write_text(
            "\n".join(
                [
                    f"STARK {version}",
                    f"Rust target: {target}",
                    "Included binaries: starkc, starkide",
                    "These binaries are unsigned development releases.",
                    "",
                ]
            ),
            encoding="utf-8",
            newline="\n",
        )

        if windows:
            archive = out_dir / f"{package_name}.zip"
            create_zip(staging, archive, package_name)
        else:
            archive = out_dir / f"{package_name}.tar.gz"
            create_tar_gz(staging, archive, package_name)

    checksum = write_checksum(archive)
    print(f"Release package: {archive}")
    print(f"SHA-256 file:   {checksum}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
