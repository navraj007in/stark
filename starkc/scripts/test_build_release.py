#!/usr/bin/env python3
"""Hermetic release-package structure and installer tests."""

from __future__ import annotations

import hashlib
import importlib.util
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
            archive, checksum = build_release.package_release(
                target="x86_64-unknown-linux-gnu",
                version="0.1.0-test",
                release_dir=release,
                out_dir=root / "packages",
            )
            self.assert_checksum(archive, checksum)
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
                (package_root / "lib/stark/stark-runtime/Cargo.toml").is_file()
            )
            self.assertTrue(
                (package_root / "lib/stark/stark-runtime/src/lib.rs").is_file()
            )
            if os.name != "nt":
                prefix = root / "prefix with spaces"
                subprocess.run(
                    [str(package_root / "install.sh"), "--prefix", str(prefix)],
                    check=True,
                )
                self.assertTrue((prefix / "bin/stark").is_file())
                self.assertTrue(
                    (prefix / "lib/stark/stark-runtime/src/lib.rs").is_file()
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
                self.assertFalse((prefix / "lib/stark/stark-runtime").exists())

    def test_windows_package_has_runner_runtime_and_powershell_installers(self) -> None:
        with tempfile.TemporaryDirectory(prefix="stark-release-test-") as temporary:
            root = Path(temporary)
            release = self.fake_release(root, windows=True)
            archive, checksum = build_release.package_release(
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
                    f"{package_root}/lib/stark/stark-runtime/Cargo.toml", names
                )
                self.assertIn(
                    f"{package_root}/lib/stark/stark-runtime/src/lib.rs", names
                )
                self.assertIn(f"{package_root}/install.ps1", names)
                self.assertIn(f"{package_root}/uninstall.ps1", names)


if __name__ == "__main__":
    unittest.main()
