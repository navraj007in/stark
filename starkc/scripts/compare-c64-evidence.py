#!/usr/bin/env python3
"""WP-C6.4c — compare two Tier-1 evidence records.

§10.4 requires "a comparison job [that] verifies both Tier-1 records refer to the same commit and
compatible versions", and §17.2 requires every disagreement to be resolved or to block closure.
Listing two platforms' results side by side is not comparing them; this script is what turns two
independent runs into one Tier-1 agreement claim.

What must be IDENTICAL across the two records (a difference is a defect):

    commit_sha, compiler_version, mir_version, mir_runtime_surface, backend_version,
    runtime_version, layout_contract, profile, determinism_result, failed_count, ignored_count,
    skipped_count, overall_result

What is EXPECTED to differ (recording them, comparing them would be wrong):

    host_triple, selected_target_triple, os_name, os_version, architecture, rustc/cargo version,
    runner, timestamps, durations

`passed_count` is deliberately in neither list: it is compared per named command rather than in
aggregate, because two platforms legitimately run the same commands and an aggregate mismatch says
nothing about which observation diverged.

Usage:

    python3 scripts/compare-c64-evidence.py macos-arm64.json linux-x64.json [--out summary.md]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Fields whose disagreement means the two runs are not describing the same thing.
IDENTICAL = (
    "schema_version",
    "commit_sha",
    "compiler_version",
    "mir_version",
    "mir_runtime_surface",
    "backend_version",
    "runtime_version",
    "layout_contract",
    "profile",
    "determinism_result",
    "failed_count",
    "ignored_count",
    "skipped_count",
    # An ignore classified on one platform and not the other would mean the two runs made
    # different observations while reporting the same totals.
    "unclassified_ignores",
    "overall_result",
)

# Fields that are supposed to differ. Recorded in the summary so a reader can see *what* the two
# platforms were, and never compared.
EXPECTED_TO_DIFFER = (
    "host_triple",
    "selected_target_triple",
    "os_name",
    "os_version",
    "architecture",
    "cargo_version",
    "runner_provider",
    "python_version",
)

TIER1 = {"aarch64-apple-darwin", "x86_64-unknown-linux-gnu"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare two C6.4 Tier-1 evidence records")
    parser.add_argument("records", nargs=2, type=Path)
    parser.add_argument("--out", type=Path, help="write a Markdown summary here")
    args = parser.parse_args()

    records = []
    for path in args.records:
        try:
            records.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError) as error:
            print(f"error: cannot read {path}: {error}", file=sys.stderr)
            return 2
    a, b = records
    problems: list[str] = []

    # Both must be Tier-1, and they must be the two DIFFERENT Tier-1 targets: two records from the
    # same platform would satisfy every field comparison below while proving nothing about
    # cross-platform agreement.
    triples = {a["host_triple"], b["host_triple"]}
    if not triples <= TIER1:
        problems.append(f"records are not both Tier-1 targets: {sorted(triples)}")
    elif triples != TIER1:
        problems.append(
            f"both records are for the same target ({sorted(triples)[0]}); a Tier-1 claim needs "
            "one record from each"
        )

    for key in IDENTICAL:
        if a.get(key) != b.get(key):
            problems.append(f"{key}: {a['host_triple']}={a.get(key)!r} vs {b['host_triple']}={b.get(key)!r}")

    if a.get("dirty_worktree") or b.get("dirty_worktree"):
        problems.append("at least one record was produced from a dirty worktree")
    if a.get("quick_mode") or b.get("quick_mode"):
        problems.append("at least one record is a quick-mode run, which is not a qualification claim")

    # Per-command comparison. Same command names, same pass/fail verdict on each.
    commands_a = {c["name"]: c for c in a.get("commands", [])}
    commands_b = {c["name"]: c for c in b.get("commands", [])}
    if set(commands_a) != set(commands_b):
        only_a = sorted(set(commands_a) - set(commands_b))
        only_b = sorted(set(commands_b) - set(commands_a))
        problems.append(f"command sets differ: only on {a['host_triple']}={only_a}, only on {b['host_triple']}={only_b}")
    for name in sorted(set(commands_a) & set(commands_b)):
        ca, cb = commands_a[name], commands_b[name]
        if ca["ok"] != cb["ok"]:
            problems.append(f"{name}: ok={ca['ok']} on {a['host_triple']}, ok={cb['ok']} on {b['host_triple']}")
        if ca["passed"] != cb["passed"]:
            problems.append(
                f"{name}: {ca['passed']} passed on {a['host_triple']} vs {cb['passed']} on "
                f"{b['host_triple']} — the same commands must make the same observations"
            )

    for record in (a, b):
        for deviation in record.get("deviations", []):
            problems.append(f"[{record['host_triple']}] {deviation}")

    agreed = not problems
    lines = [
        "# C6.4 Tier-1 qualification summary",
        "",
        f"- Commit: `{a.get('commit_sha')}`",
        f"- Records: `{a.get('host_triple')}` and `{b.get('host_triple')}`",
        f"- Generated corpus: {a.get('generated_corpus_status')} (see WP-C6.4 §1.2)",
        "",
        "## Platform identities",
        "",
        f"| Field | {a.get('host_triple')} | {b.get('host_triple')} |",
        "|---|---|---|",
    ]
    for key in EXPECTED_TO_DIFFER:
        lines.append(f"| {key} | {a.get(key)} | {b.get(key)} |")
    lines += ["", "## Required agreement", "", "| Field | Value | Agrees |", "|---|---|---|"]
    for key in IDENTICAL:
        lines.append(f"| {key} | {a.get(key)} | {'yes' if a.get(key) == b.get(key) else '**NO**'} |")
    lines += ["", "## Per-command results", "", "| Command | " + a.get("host_triple", "?") + " | " + b.get("host_triple", "?") + " |", "|---|---|---|"]
    for name in sorted(set(commands_a) | set(commands_b)):
        def cell(cs: dict) -> str:
            c = cs.get(name)
            return "absent" if c is None else f"{'PASS' if c['ok'] else 'FAIL'} ({c['passed']})"
        lines.append(f"| {name} | {cell(commands_a)} | {cell(commands_b)} |")
    lines += ["", "## Result", "", f"**{'TIER-1 AGREEMENT' if agreed else 'TIER-1 DISAGREEMENT'}**", ""]
    if problems:
        lines += ["### Problems", ""] + [f"- {p}" for p in problems] + [""]

    summary = "\n".join(lines)
    print(summary)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(summary, encoding="utf-8")
    return 0 if agreed else 1


if __name__ == "__main__":
    sys.exit(main())
