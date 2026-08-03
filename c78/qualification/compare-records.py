#!/usr/bin/env python3
import argparse
import json
import sys
import tomllib
from pathlib import Path

from capability_map import capability_summary

ALLOWED = {"pending", "implemented", "qualified", "unsupported"}


def load_schema(path):
    schema = json.loads(path.read_text())
    if schema.get("schema_version") != 1:
        raise SystemExit("qualification schema version must be 1")
    return schema


def load_manifest(path):
    with path.open("rb") as f:
        manifest = tomllib.load(f)
    if manifest.get("version") != 1:
        raise SystemExit("capability manifest version must be 1")
    return manifest


def validate_record(record, schema, expected_commit, expected_capabilities):
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
        raise SystemExit(f"{record.get('platform', '<unknown>')} missing fields: {sorted(missing)}")
    if "tests" in record:
        raise SystemExit(f"{record['platform']} still contains fabricated numeric tests field")
    if record["schema_version"] != schema["schema_version"]:
        raise SystemExit(f"{record['platform']} schema_version mismatch")
    if record["provider_abi_version"] != schema["provider_abi_version"]:
        raise SystemExit(f"{record['platform']} provider ABI mismatch")
    if record["mir_runtime_surface"] != schema["mir_runtime_surface"]:
        raise SystemExit(f"{record['platform']} MIR surface mismatch")
    if record["commit"] != expected_commit:
        raise SystemExit(
            f"{record['platform']} commit {record['commit']} != expected {expected_commit}"
        )
    if record["capabilities"] != expected_capabilities:
        raise SystemExit(f"{record['platform']} capabilities differ from manifest")
    if set(record["evidence"]) != set(schema["required_evidence"]):
        raise SystemExit(f"{record['platform']} evidence keys differ from schema")
    allowed_evidence = set(schema["allowed_evidence_states"])
    for key, state in record["evidence"].items():
        if state not in allowed_evidence:
            raise SystemExit(f"{record['platform']}.{key} evidence state {state!r} invalid")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--schema", default="c78/qualification/schema.json")
    parser.add_argument("--manifest", default="c78/capabilities.toml")
    parser.add_argument("--commit", required=True)
    parser.add_argument("records", nargs="+")
    args = parser.parse_args()

    schema = load_schema(Path(args.schema))
    manifest = load_manifest(Path(args.manifest))
    required = set(schema["required_platforms"])
    expected_capabilities = capability_summary(manifest)
    records = [json.loads(Path(path).read_text()) for path in args.records]
    by_platform = {record["platform"]: record for record in records}
    missing = required - by_platform.keys()
    if missing:
        raise SystemExit(f"missing C7.8 records for: {', '.join(sorted(missing))}")
    if len(by_platform) != len(records):
        raise SystemExit("duplicate C7.8 platform record")

    capability_keys = set(expected_capabilities)
    first_capabilities = None
    first_evidence = None
    for platform, record in sorted(by_platform.items()):
        validate_record(record, schema, args.commit, expected_capabilities)
        capabilities = record["capabilities"]
        if set(capabilities) != capability_keys:
            raise SystemExit(f"{platform} capability keys differ")
        for key, state in capabilities.items():
            if state not in ALLOWED:
                raise SystemExit(f"{platform}.{key} has invalid state {state!r}")
        if first_capabilities is None:
            first_capabilities = capabilities
            first_evidence = record["evidence"]
        elif capabilities != first_capabilities:
            raise SystemExit(f"{platform} capability states differ without declared exception")
        elif record["evidence"] != first_evidence:
            raise SystemExit(f"{platform} evidence states differ without declared exception")

    print("C7.8 qualification records agree semantically")


if __name__ == "__main__":
    sys.exit(main())
