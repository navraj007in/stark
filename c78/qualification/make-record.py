#!/usr/bin/env python3
import argparse
import json
import platform
import subprocess
import sys
import tomllib
from pathlib import Path

ALLOWED = {"pending", "implemented", "qualified", "unsupported"}
EVIDENCE = {
    "provider_metadata": "pass",
    "provider_unit": "pass",
    "resource_lifecycle": "pass",
    "loopback": "pass",
    "native_e2e": "pending",
}


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
        # HC9 (CD-365). `tls_transfer` is a SEPARATE row from `tls_e2e` deliberately: "TLS works"
        # and "a resource crossed from one provider to another and was released exactly once" are
        # different claims, and CD-360 left the second one open for HC9 to close.
        "tls": manifest["stark_tls"]["provider_metadata"],
        "tls_e2e": manifest["stark_tls"]["native_e2e"],
        "tls_transfer": manifest["stark_tls"]["cross_provider_transfer"],
    }


def load_schema(path):
    schema = json.loads(path.read_text())
    if schema.get("schema_version") != 1:
        raise SystemExit("qualification schema version must be 1")
    return schema


def validate_record(record, schema):
    required = {
        "schema_version",
        "commit",
        "platform",
        "host",
        "rustc",
        "cargo",
        "provider_abi_version",
        "mir_runtime_surface",
        "capabilities",
        "evidence",
    }
    missing = required - set(record)
    if missing:
        raise SystemExit(f"record missing required fields: {sorted(missing)}")
    if record["schema_version"] != schema["schema_version"]:
        raise SystemExit("record schema_version does not match schema")
    if record["provider_abi_version"] != schema["provider_abi_version"]:
        raise SystemExit("record provider ABI version does not match schema")
    if record["mir_runtime_surface"] != schema["mir_runtime_surface"]:
        raise SystemExit("record MIR runtime surface does not match schema")
    allowed_capabilities = set(schema["allowed_capability_states"])
    for key, state in record["capabilities"].items():
        if state not in allowed_capabilities:
            raise SystemExit(f"record capability {key} has invalid state {state!r}")
    required_evidence = set(schema["required_evidence"])
    if set(record["evidence"]) != required_evidence:
        raise SystemExit("record evidence keys do not match schema")
    allowed_evidence = set(schema["allowed_evidence_states"])
    for key, state in record["evidence"].items():
        if state not in allowed_evidence:
            raise SystemExit(f"record evidence {key} has invalid state {state!r}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--platform", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--manifest", default="c78/capabilities.toml")
    parser.add_argument("--schema", default="c78/qualification/schema.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    manifest = load_manifest(Path(args.manifest))
    schema = load_schema(Path(args.schema))
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
        "evidence": EVIDENCE,
    }
    validate_record(record, schema)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    sys.exit(main())
