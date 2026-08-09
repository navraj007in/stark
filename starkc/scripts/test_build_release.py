#!/usr/bin/env python3
"""Hermetic release-package structure and installer tests."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import subprocess
import tarfile
import tempfile
import unittest
import zipfile


SCRIPT = Path(__file__).with_name("build-release.py")
SPEC = importlib.util.spec_from_file_location("build_release", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
build_release = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(build_release)


class ReleasePackageTests(unittest.TestCase):
    def fake_release(self, root: Path, *, windows: bool) -> Path:
        release = root / "release"
        release.mkdir()
        suffix = ".exe" if windows else ""
        for binary in build_release.BINARIES:
            path = release / f"{binary}{suffix}"
            path.write_bytes(f"fake {binary}\n".encode())
            if not windows:
                path.chmod(0o755)
        return release

    def assert_checksum(self, archive: Path, checksum: Path) -> None:
        expected = hashlib.sha256(archive.read_bytes()).hexdigest()
        self.assertEqual(
            checksum.read_text(encoding="utf-8"),
            f"{expected}  {archive.name}\n",
        )

    def test_unix_package_has_runner_runtime_installers_and_installs(self) -> None:
        with tempfile.TemporaryDirectory(prefix="stark-release-test-") as temporary:
            root = Path(temporary)
            release = self.fake_release(root, windows=False)
            archive, checksum, native_installers = build_release.package_release(
                target="x86_64-unknown-linux-gnu",
                version="0.1.0-test",
                release_dir=release,
                out_dir=root / "packages",
            )
            self.assert_checksum(archive, checksum)
            self.assertEqual(len(native_installers), 1)
            self.assertEqual(native_installers[0].suffix, ".deb")
            self.assertTrue(native_installers[0].is_file())
            extracted = root / "extracted"
            package_root_name = "stark-0.1.0-test-x86_64-unknown-linux-gnu"
            with tarfile.open(archive, "r:gz") as package:
                for binary in build_release.BINARIES:
                    # Check the archive's own metadata, not a post-extraction os.stat(): NTFS has
                    # no POSIX executable bit, so a Windows host can never observe one on an
                    # extracted file regardless of what the tar entry says.
                    member = package.getmember(f"{package_root_name}/bin/{binary}")
                    self.assertTrue(
                        member.mode & stat.S_IXUSR,
                        f"{binary} is missing the executable bit in the archive",
                    )
                package.extractall(extracted, filter="data")
            package_root = extracted / package_root_name
            for binary in build_release.BINARIES:
                path = package_root / "bin" / binary
                self.assertTrue(path.is_file())
                if os.name != "nt":
                    self.assertTrue(path.stat().st_mode & stat.S_IXUSR)
            self.assertTrue(
                (package_root / "lib/stark/starkc/stark-runtime/Cargo.toml").is_file()
            )
            self.assertTrue(
                (package_root / "lib/stark/starkc/stark-runtime/src/lib.rs").is_file()
            )
            self.assertTrue(
                (package_root / "lib/stark/starkc/stark-provider-abi/src/lib.rs").is_file()
            )
            # Without the provider crates the package can `check` and `test` but cannot `build`
            # anything that declares a capability, so their presence is part of what "packaged"
            # means. The ABI path is asserted from the provider's own perspective: its manifest
            # says `../../../starkc/stark-provider-abi`, and that has to land on a real directory.
            for crate_path in build_release.provider_crate_paths():
                native = package_root / "lib/stark/packages" / crate_path
                self.assertTrue(
                    (native / "Cargo.toml").is_file(), f"missing provider crate: {crate_path}"
                )
                self.assertTrue(
                    (native / "src/lib.rs").is_file(), f"missing provider sources: {crate_path}"
                )
                self.assertTrue(
                    (native / "../../../starkc/stark-provider-abi/Cargo.toml").resolve().is_file(),
                    f"provider ABI dependency does not resolve for {crate_path}",
                )
            for package_name in build_release.toolchain_package_paths():
                package = package_root / "lib/stark/packages" / package_name
                self.assertTrue((package / "starkpkg.json").is_file())
                self.assertTrue(
                    any((package / "src").rglob("*.stark")),
                    f"missing STARK sources for {package_name}",
                )
            # `target/` is a build artefact of the checkout, never payload.
            self.assertEqual(
                [], list((package_root / "lib/stark/packages").rglob("target")),
                "provider build artefacts must not be packaged",
            )
            self.assertTrue((package_root / "manifest.json").is_file())
            install_manifest = json.loads(
                (package_root / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                install_manifest["packages"], build_release.toolchain_package_paths()
            )
            self.assertEqual(
                install_manifest["providers"],
                sorted(
                    {
                        path.split("/", 1)[0]
                        for path in build_release.provider_crate_paths()
                    }
                ),
            )
            if os.name != "nt":
                prefix = root / "prefix with spaces"
                subprocess.run(
                    [str(package_root / "install.sh"), "--prefix", str(prefix)],
                    check=True,
                )
                self.assertTrue((prefix / "bin/stark").is_file())
                installed = prefix / "lib/stark/versions/0.1.0-test"
                self.assertTrue((prefix / "lib/stark/current").exists())
                self.assertTrue((installed / "manifest.json").is_file())
                self.assertTrue(
                    (installed / "lib/stark/starkc/stark-runtime/src/lib.rs").is_file()
                )
                self.assertTrue(
                    (installed / "lib/stark/starkc/stark-provider-abi/src/lib.rs").is_file()
                )
                self.assertTrue(
                    (installed / "lib/stark/packages").is_dir(),
                    "an installed tree must carry packages and provider crates",
                )
                for package_name in build_release.toolchain_package_paths():
                    self.assertTrue(
                        (installed / "lib/stark/packages" / package_name / "starkpkg.json").is_file()
                    )
                subprocess.run(
                    [
                        str(prefix / "lib/stark/uninstall.sh"),
                        "--prefix",
                        str(prefix),
                    ],
                    check=True,
                )
                self.assertFalse((prefix / "bin/stark").exists())
                self.assertFalse(installed.exists())
                self.assertFalse((prefix / "lib/stark/current").exists())

    def test_windows_package_has_runner_runtime_and_powershell_installers(self) -> None:
        with tempfile.TemporaryDirectory(prefix="stark-release-test-") as temporary:
            root = Path(temporary)
            release = self.fake_release(root, windows=True)
            archive, checksum, _native_installers = build_release.package_release(
                target="x86_64-pc-windows-msvc",
                version="0.1.0-test",
                release_dir=release,
                out_dir=root / "packages",
            )
            self.assert_checksum(archive, checksum)
            with zipfile.ZipFile(archive) as package:
                names = set(package.namelist())
                package_root = "stark-0.1.0-test-x86_64-pc-windows-msvc"
                for binary in build_release.BINARIES:
                    self.assertIn(f"{package_root}/bin/{binary}.exe", names)
                self.assertIn(
                    f"{package_root}/lib/stark/starkc/stark-runtime/Cargo.toml", names
                )
                self.assertIn(
                    f"{package_root}/lib/stark/starkc/stark-runtime/src/lib.rs", names
                )
                self.assertIn(
                    f"{package_root}/lib/stark/starkc/stark-provider-abi/src/lib.rs", names
                )
                for crate_path in build_release.provider_crate_paths():
                    self.assertIn(
                        f"{package_root}/lib/stark/packages/{crate_path}/Cargo.toml", names
                    )
                for package_name in build_release.toolchain_package_paths():
                    self.assertIn(
                        f"{package_root}/lib/stark/packages/{package_name}/starkpkg.json",
                        names,
                    )
                    self.assertTrue(
                        any(
                            name.startswith(
                                f"{package_root}/lib/stark/packages/{package_name}/src/"
                            )
                            and name.endswith(".stark")
                            for name in names
                        ),
                        f"missing STARK sources for {package_name}",
                    )
                self.assertIn(f"{package_root}/manifest.json", names)
                self.assertIn(f"{package_root}/install.ps1", names)
                self.assertIn(f"{package_root}/uninstall.ps1", names)

    def test_macos_package_builds_pkg_when_pkgbuild_is_available(self) -> None:
        if build_release.shutil.which("pkgbuild") is None:
            self.skipTest("pkgbuild is not available")
        with tempfile.TemporaryDirectory(prefix="stark-release-test-") as temporary:
            root = Path(temporary)
            release = self.fake_release(root, windows=False)
            archive, checksum, native_installers = build_release.package_release(
                target="aarch64-apple-darwin",
                version="0.1.0-test",
                release_dir=release,
                out_dir=root / "packages",
            )
            self.assert_checksum(archive, checksum)
            self.assertEqual(len(native_installers), 1)
            self.assertEqual(native_installers[0].suffix, ".pkg")
            self.assertTrue(native_installers[0].is_file())


class TargetClassificationTests(unittest.TestCase):
    """WP-C6.4: packaging classifies by exact named target, never by substring.

    The replaced code was `windows = "windows" in target`. These tests pin the two ways that was
    wrong — a triple the compiler does not name must not be packaged at all, and a triple's shape
    must not decide its packaging.
    """

    def test_an_unknown_triple_is_refused_rather_than_packaged(self) -> None:
        with tempfile.TemporaryDirectory(prefix="stark-release-test-") as temporary:
            root = Path(temporary)
            release = root / "release"
            release.mkdir()
            for triple in (
                "x86_64-unknown-linux-musl",
                "i686-pc-windows-msvc",
                "aarch64-unknown-linux-gnu",
                # The substring trap: contains "windows", is not a target STARK names.
                "sparc64-windows-unknown",
            ):
                with self.assertRaises(
                    build_release.target_matrix.UnknownTarget, msg=triple
                ):
                    build_release.package_release(
                        target=triple,
                        version="0.1.0-test",
                        release_dir=release,
                        out_dir=root / "packages",
                    )

    def test_packaging_shape_comes_from_the_named_entry(self) -> None:
        matrix = build_release.target_matrix
        windows = matrix.require("x86_64-pc-windows-msvc")
        self.assertEqual(windows.executable_suffix, ".exe")
        self.assertEqual(windows.archive, "zip")
        self.assertEqual(windows.installers, ("install.ps1", "uninstall.ps1"))
        for triple in ("aarch64-apple-darwin", "x86_64-unknown-linux-gnu"):
            entry = matrix.require(triple)
            self.assertEqual(entry.executable_suffix, "")
            self.assertEqual(entry.archive, "tar.gz")
            self.assertEqual(entry.installers, ("install.sh", "uninstall.sh"))
            self.assertTrue(entry.is_tier1)

    def test_classification_is_exact_not_prefix_or_substring(self) -> None:
        matrix = build_release.target_matrix
        self.assertIsNotNone(matrix.classify("x86_64-unknown-linux-gnu"))
        for near_miss in (
            "x86_64-unknown-linux-gnux32",
            "x86_64-unknown-linux",
            "86_64-unknown-linux-gnu",
        ):
            self.assertIsNone(matrix.classify(near_miss), near_miss)

    def test_tier1_is_exactly_the_two_gate_c6_targets(self) -> None:
        self.assertEqual(
            sorted(build_release.target_matrix.tier1_triples()),
            ["aarch64-apple-darwin", "x86_64-unknown-linux-gnu"],
        )


if __name__ == "__main__":
    unittest.main()
