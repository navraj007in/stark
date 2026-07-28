#!/usr/bin/env python3
import argparse
import json
import platform
import subprocess
import sys
import tomllib
from pathlib import Path

ALLOWED = {"pending", "implemented", "qualified", "unsupported"}


def run(args):
    return subprocess.check_output(args, text=True).strip()


def load_manifest(path):
    with path.open("rb") as f:
        manifest = tomllib.load(f)
    if manifest.get("version") != 1:
        raise SystemExit("capability manifest version must be 1")
    for capability, fields in manifest.items():
        if capability == "version":
            continue
        for key, value in fields.items():
            if key.endswith("_reason"):
                continue
            if value not in ALLOWED:
                raise SystemExit(f"{capability}.{key} has invalid state {value!r}")
            if value == "unsupported" and not fields.get(f"{key}_reason"):
                raise SystemExit(f"{capability}.{key}=unsupported requires {key}_reason")
    return manifest


def capability_summary(manifest):
    return {
        "time": manifest["stark_time"]["provider_metadata"],
        "args_env": manifest["stark_env"]["provider_metadata"],
        "file": manifest["stark_file"]["provider_metadata"],
        "tcp": manifest["stark_net"]["loopback_provider"],
        "stark_time_e2e": manifest["stark_time"]["native_e2e"],
        "args_env_e2e": manifest["stark_env"]["native_e2e"],
        "file_e2e": manifest["stark_file"]["native_e2e"],
        "tcp_e2e": manifest["stark_net"]["native_e2e"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--platform", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--manifest", default="c78/capabilities.toml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--passed", type=int, default=0)
    parser.add_argument("--failed", type=int, default=0)
    parser.add_argument("--ignored", type=int, default=0)
    args = parser.parse_args()

    manifest = load_manifest(Path(args.manifest))
    record = {
        "schema_version": 1,
        "commit": args.commit,
        "platform": args.platform,
        "host": platform.system().lower(),
        "rustc": run(["rustc", "--version"]),
        "cargo": run(["cargo", "--version"]),
        "provider_abi_version": "0.1",
        "mir_runtime_surface": "0.1-A10",
        "capabilities": capability_summary(manifest),
        "tests": {
            "passed": args.passed,
            "failed": args.failed,
            "ignored": args.ignored,
        },
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    sys.exit(main())
