#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

REQUIRED = {"linux-x64", "macos-arm64", "windows-x64"}
ALLOWED = {"pending", "implemented", "qualified", "unsupported"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("records", nargs="+")
    args = parser.parse_args()

    records = [json.loads(Path(path).read_text()) for path in args.records]
    by_platform = {record["platform"]: record for record in records}
    missing = REQUIRED - by_platform.keys()
    if missing:
        raise SystemExit(f"missing C7.8 records for: {', '.join(sorted(missing))}")
    if len(by_platform) != len(records):
        raise SystemExit("duplicate C7.8 platform record")

    schemas = {record["schema_version"] for record in records}
    abi_versions = {record["provider_abi_version"] for record in records}
    mir_surfaces = {record["mir_runtime_surface"] for record in records}
    if schemas != {1}:
        raise SystemExit(f"schema mismatch: {schemas}")
    if abi_versions != {"0.1"}:
        raise SystemExit(f"ABI version mismatch: {abi_versions}")
    if mir_surfaces != {"0.1-A10"}:
        raise SystemExit(f"MIR runtime surface mismatch: {mir_surfaces}")

    capability_keys = None
    for platform, record in sorted(by_platform.items()):
        tests = record["tests"]
        if any(tests[key] < 0 for key in ("passed", "failed", "ignored")):
            raise SystemExit(f"{platform} has negative test counts")
        capabilities = record["capabilities"]
        if capability_keys is None:
            capability_keys = set(capabilities)
        elif set(capabilities) != capability_keys:
            raise SystemExit(f"{platform} capability keys differ")
        for key, state in capabilities.items():
            if state not in ALLOWED:
                raise SystemExit(f"{platform}.{key} has invalid state {state!r}")

    for key in sorted(capability_keys or []):
        states = {platform: record["capabilities"][key] for platform, record in by_platform.items()}
        if "qualified" in states.values() and any(value == "pending" for value in states.values()):
            raise SystemExit(f"{key} is qualified on one platform and pending on another: {states}")

    print("C7.8 qualification records agree semantically")


if __name__ == "__main__":
    sys.exit(main())
