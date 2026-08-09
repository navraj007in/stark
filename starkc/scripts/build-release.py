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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import target_matrix  # noqa: E402  (path is set immediately above)


CRATE_DIR = Path(__file__).resolve().parent.parent
REPO_DIR = CRATE_DIR.parent
BINARIES = ("stark", "starkc", "starkide")
RUNTIME_FILES = ("Cargo.toml",)
RUNTIME_DIRS = ("src",)
RUNTIME_PATH_DEPENDENCIES = ("stark-provider-abi",)
# A provider crate needs its manifest, its pinned lockfile and its sources. `target/` is a build
# artefact of the checkout, not payload: including it multiplied the package by two orders of
# magnitude and shipped one host's object files to every other.
PROVIDER_FILES = ("Cargo.toml", "Cargo.lock")
PROVIDER_DIRS = ("src",)


def provider_crate_paths() -> list[str]:
    """The built-in providers' crate paths, read from the manifests the compiler embeds.

    `provider_registry.rs` states the rule this follows: adding a provider is adding a manifest,
    and nothing else changes. Reading `crate_path` here rather than repeating the six names keeps
    the packager from being the one place that has to be edited in step — a list that drifts silent
    would ship a package missing exactly the capability someone added.
    """
    manifest_dir = CRATE_DIR / "providers"
    paths = []
    for manifest in sorted(manifest_dir.glob("*.json")):
        provider = json.loads(manifest.read_text(encoding="utf-8"))["provider"]
        crate_path = provider.get("crate_path")
        if not crate_path:
            raise SystemExit(f"provider manifest declares no crate_path: {manifest}")
        paths.append(crate_path)
    if not paths:
        raise SystemExit(f"no provider manifests found under {manifest_dir}")
    return paths


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
            # NTFS has no POSIX executable bit, so `gettarinfo`'s host-`stat`-derived mode loses
            # it when packaging runs on a Windows host; set it explicitly from the archive layout
            # instead of trusting the host filesystem to have preserved a `chmod` call.
            executable = path.parent.name == "bin" or path.name in {
                "install.sh",
                "uninstall.sh",
            }
            info.mode = 0o755 if executable else 0o644
            with path.open("rb") as source:
                archive.addfile(info, source)
        else:
            info.mode = 0o755
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
            mode = (
                0o755
                if path.parent.name == "bin"
                or path.name in {"install.sh", "uninstall.sh"}
                or path.suffix == ".exe"
                else 0o644
            )
            info.external_attr = mode << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED)


def copy_payload_file(source: Path, destination: Path) -> Path:
    shutil.copyfile(source, destination)
    shutil.copymode(source, destination)
    return destination


def create_install_tree(staging: Path, root: Path, *, version: str, windows: bool) -> None:
    version_root = root / "usr/local/stark/versions" / version
    shutil.copytree(staging, version_root, copy_function=copy_payload_file)
    current = root / "usr/local/stark/current"
    current.parent.mkdir(parents=True, exist_ok=True)
    current.symlink_to(f"versions/{version}")
    bin_dir = root / "usr/local/bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    suffix = ".exe" if windows else ""
    for binary in BINARIES:
        (bin_dir / f"{binary}{suffix}").symlink_to(
            f"../stark/current/bin/{binary}{suffix}"
        )


def write_checksum(archive: Path) -> Path:
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    checksum = archive.with_name(f"{archive.name}.sha256")
    checksum.write_text(f"{digest}  {archive.name}\n", encoding="utf-8", newline="\n")
    return checksum


def write_ar_archive(output: Path, members: list[Path]) -> None:
    with output.open("wb") as archive:
        archive.write(b"!<arch>\n")
        for member in members:
            data = member.read_bytes()
            name = member.name.encode("ascii")
            if len(name) > 15:
                raise SystemExit(f"ar member name is too long: {member.name}")
            header = (
                name.ljust(16, b" ")
                + b"0".ljust(12, b" ")
                + b"0".ljust(6, b" ")
                + b"0".ljust(6, b" ")
                + b"100644".ljust(8, b" ")
                + str(len(data)).encode("ascii").ljust(10, b" ")
                + b"`\n"
            )
            archive.write(header)
            archive.write(data)
            if len(data) % 2:
                archive.write(b"\n")


def create_deb(
    *, staging: Path, output: Path, package_name: str, version: str, target: str
) -> Path:
    with tempfile.TemporaryDirectory(prefix="stark-deb-") as temporary:
        work = Path(temporary)
        data_root = work / "data"
        create_install_tree(staging, data_root, version=version, windows=False)
        data_tar = work / "data.tar.gz"
        create_tar_gz(data_root, data_tar, ".")

        control = work / "control"
        control.mkdir()
        control_text = "\n".join(
            [
                "Package: stark",
                f"Version: {debian_version(version)}",
                "Section: devel",
                "Priority: optional",
                "Architecture: amd64",
                "Maintainer: STARK Project <noreply@starklang.local>",
                f"Description: STARK compiler toolchain ({target})",
                " This unsigned development package wraps the canonical STARK release payload.",
                "",
            ]
        )
        (control / "control").write_text(control_text, encoding="utf-8", newline="\n")
        control_tar = work / "control.tar.gz"
        create_tar_gz(control, control_tar, ".")
        debian_binary = work / "debian-binary"
        debian_binary.write_text("2.0\n", encoding="ascii", newline="\n")
        output = output.with_name(f"{package_name}.deb")
        write_ar_archive(output, [debian_binary, control_tar, data_tar])
        return output


def debian_version(version: str) -> str:
    allowed = set("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.+-~:")
    return "".join(ch if ch in allowed else "+" for ch in version)


def create_macos_pkg(
    *, staging: Path, output: Path, package_name: str, version: str
) -> Path | None:
    if shutil.which("pkgbuild") is None:
        return None
    with tempfile.TemporaryDirectory(prefix="stark-pkg-") as temporary:
        work = Path(temporary)
        payload = work / "payload"
        create_install_tree(staging, payload, version=version, windows=False)
        if shutil.which("xattr") is not None:
            subprocess.run(["xattr", "-cr", str(payload)], check=True)
        package = output.with_name(f"{package_name}.pkg")
        env = os.environ.copy()
        env["COPYFILE_DISABLE"] = "1"
        subprocess.run(
            [
                "pkgbuild",
                "--root",
                str(payload),
                "--identifier",
                "org.starklang.stark",
                "--version",
                version,
                "--install-location",
                "/",
                str(package),
            ],
            check=True,
            env=env,
        )
        return package


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_DIR,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unknown"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(staging: Path, *, target: str, version: str) -> None:
    files = []
    for path in sorted(staging.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(staging).as_posix()
        files.append(
            {
                "path": relative,
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
                "component": component_for_path(relative),
                "executable": is_executable_payload(relative, path),
            }
        )
    manifest = {
        "schema_version": 1,
        "stark_version": version,
        "release_channel": "dev",
        "build_commit": git_commit(),
        "build_timestamp": "unspecified",
        "host_target": target,
        "compiler": {
            "version": version,
            "sha256": next(
                file["sha256"]
                for file in files
                if file["path"] in {"bin/stark", "bin/stark.exe"}
            ),
        },
        "mir_version": "unknown",
        "runtime_version": version,
        "backend_version": version,
        "packages": [],
        # Declared, not inferred from the file list: a reader asking "which capabilities can this
        # package build?" gets an answer that stays true even if the payload is later trimmed —
        # and a mismatch between this and `files` is then a detectable defect rather than silence.
        "providers": sorted(
            {crate_path.split("/", 1)[0] for crate_path in provider_crate_paths()}
        ),
        "files": files,
        "signing": {
            "scheme": "unsigned-development",
            "key_id": "none",
        },
    }
    (staging / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def component_for_path(relative: str) -> str:
    if relative.startswith("bin/"):
        return "binary"
    # The mirror layout first, then the flat one an older package used, so a manifest written for
    # either classifies the same way.
    if relative.startswith(("lib/stark/starkc/stark-runtime/", "lib/stark/stark-runtime/")):
        return "runtime"
    if relative.startswith(
        ("lib/stark/starkc/stark-provider-abi/", "lib/stark/stark-provider-abi/")
    ):
        return "provider-abi"
    if relative.startswith("lib/stark/providers/"):
        return "provider"
    if relative.startswith("lib/stark/packages/"):
        return "package"
    if relative in {"install.sh", "install.ps1", "uninstall.sh", "uninstall.ps1"}:
        return "installer"
    if relative in {"README.md", "LICENSE", "BUILD-INFO.txt"}:
        return "metadata"
    return "other"


def is_executable_payload(relative: str, path: Path) -> bool:
    return (
        relative.startswith("bin/")
        or relative in {"install.sh", "uninstall.sh"}
        or path.suffix == ".exe"
    )


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


def package_release(
    *, target: str, version: str, release_dir: Path, out_dir: Path
) -> tuple[Path, Path, list[Path]]:
    # WP-C6.4: exact named-target lookup, never a substring test.
    #
    # This used to be `windows = "windows" in target`, which is wrong in two directions: it would
    # classify any unknown triple containing the word as Windows, and — worse — it would happily
    # package a triple the compiler does not name at all, producing an artifact nothing can
    # qualify. `require` raises on an unknown triple instead, and the executable suffix, archive
    # format and installer pair all come from the entry rather than from a boolean.
    entry = target_matrix.require(target)
    windows = entry.archive == "zip"
    executable_suffix = entry.executable_suffix
    package_name = f"stark-{version}-{target}"
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="stark-release-") as temporary:
        staging = Path(temporary)
        package_bin = staging / "bin"
        package_bin.mkdir()
        for binary in BINARIES:
            source = release_dir / f"{binary}{executable_suffix}"
            if not source.is_file():
                raise SystemExit(f"expected release binary was not produced: {source}")
            destination = package_bin / source.name
            shutil.copy2(source, destination)
            if not windows:
                destination.chmod(0o755)

        # **The installed tree mirrors the repository.** The runtime and the provider ABI go under
        # `lib/stark/starkc/`, and the providers under `lib/stark/packages/` — the same two levels
        # they occupy in the checkout. That correspondence is load-bearing, not cosmetic: the
        # runtime's `../stark-provider-abi` and every provider's `../../../starkc/stark-provider-abi`
        # then name the same directory, so Cargo sees one `stark-provider-abi` and not two.
        #
        # It is also what `native_build::provider_root_beside_runtime` reads: it derives the
        # provider root from the runtime's own location, so a runtime installed flat at
        # `lib/stark/stark-runtime` shifts every candidate up a level and finds no providers at
        # all. The flat layout stays readable by `native_toolchain::discover_runtime`, which lists
        # it as a fallback for installations made before this move — but it is no longer written.
        starkc_root = staging / "lib" / "stark" / "starkc"
        runtime_source = CRATE_DIR / "stark-runtime"
        runtime_destination = starkc_root / "stark-runtime"
        runtime_destination.mkdir(parents=True)
        for filename in RUNTIME_FILES:
            shutil.copy2(runtime_source / filename, runtime_destination / filename)
        for dirname in RUNTIME_DIRS:
            shutil.copytree(runtime_source / dirname, runtime_destination / dirname)
        for crate_name in RUNTIME_PATH_DEPENDENCIES:
            source = CRATE_DIR / crate_name
            destination = starkc_root / crate_name
            destination.mkdir(parents=True)
            for filename in RUNTIME_FILES:
                shutil.copy2(source / filename, destination / filename)
            for dirname in RUNTIME_DIRS:
                shutil.copytree(source / dirname, destination / dirname)

        # The provider crates themselves. Without these the package can `check` and `test` but
        # cannot `build` any program that declares a capability — the compiler selects a provider
        # from its built-in registry and then finds no crate to compile.
        packages_root = staging / "lib" / "stark" / "packages"
        for crate_path in provider_crate_paths():
            source = REPO_DIR / "packages" / crate_path
            if not (source / "Cargo.toml").is_file():
                raise SystemExit(f"provider crate is missing from the checkout: {source}")
            destination = packages_root / crate_path
            destination.mkdir(parents=True)
            for filename in PROVIDER_FILES:
                candidate = source / filename
                if candidate.is_file():
                    shutil.copy2(candidate, destination / filename)
            for dirname in PROVIDER_DIRS:
                shutil.copytree(source / dirname, destination / dirname)

        dist_dir = CRATE_DIR / "dist"
        for installer_name in entry.installers:
            destination = staging / installer_name
            shutil.copy2(dist_dir / installer_name, destination)
            if not windows:
                destination.chmod(0o755)

        shutil.copy2(REPO_DIR / "LICENSE", staging / "LICENSE")
        shutil.copy2(CRATE_DIR / "README.md", staging / "README.md")
        (staging / "BUILD-INFO.txt").write_text(
            "\n".join(
                [
                    f"STARK {version}",
                    f"Rust target: {target}",
                    "Included binaries: stark, starkc, starkide",
                    "Installed runtime: lib/stark/starkc/stark-runtime",
                    "Installed provider ABI: lib/stark/starkc/stark-provider-abi",
                    "Installed providers: lib/stark/packages/<name>/native — "
                    + ", ".join(
                        sorted(
                            {path.split("/", 1)[0] for path in provider_crate_paths()}
                        )
                    ),
                    "These binaries are unsigned development releases.",
                    "",
                ]
            ),
            encoding="utf-8",
            newline="\n",
        )
        write_manifest(staging, target=target, version=version)

        archive = out_dir / f"{package_name}.{entry.archive}"
        if entry.archive == "zip":
            create_zip(staging, archive, package_name)
        else:
            create_tar_gz(staging, archive, package_name)
        native_installers = []
        if target.endswith("-apple-darwin"):
            package = create_macos_pkg(
                staging=staging,
                output=archive,
                package_name=package_name,
                version=version,
            )
            if package is not None:
                native_installers.append(package)
        if target == "x86_64-unknown-linux-gnu":
            native_installers.append(
                create_deb(
                    staging=staging,
                    output=archive,
                    package_name=package_name,
                    version=version,
                    target=target,
                )
            )

    checksum = write_checksum(archive)
    print(f"Release package: {archive}")
    print(f"SHA-256 file:   {checksum}")
    for installer in native_installers:
        installer_checksum = write_checksum(installer)
        print(f"Native installer: {installer}")
        print(f"SHA-256 file:    {installer_checksum}")
    return archive, checksum, native_installers


def main() -> int:
    args = parse_args()
    target = args.target or host_target()
    version = package_version()

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
    package_release(
        target=target,
        version=version,
        release_dir=release_dir,
        out_dir=args.out_dir,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
