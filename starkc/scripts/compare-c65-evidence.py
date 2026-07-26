#!/usr/bin/env python3
"""WP-C6.5 §16.4 — Tier-1 corpus agreement.

Two green corpus jobs are not agreement, for the same reason two green platform jobs are not: each
says the corpus passed *there*. Agreement is the claim that both targets ran **the same corpus at the
same commit** and **observed the same things**, and only a comparison can say that.

What it compares (§16.4):

* same exact commit, both worktrees clean;
* same corpus version, generator version and seed;
* same manifest and generator hashes;
* same required case IDs — no case missing from either side, none extra;
* same per-case outcome class;
* same per-case **observation hash** — the strongest clause, because two records can agree on
  "PASS" while having observed different bytes;
* same pass/fail/skip totals;
* both records in FULL evidence mode, not filtered or sharded;
* two DIFFERENT Tier-1 triples — a "comparison" of one platform against itself is not one.

Platform metadata that is *expected* to differ (OS, architecture, toolchain build strings) is
recorded in the report and never treated as disagreement. Everything else that differs is.

Exit code 0 means agreement; 1 means a named disagreement. A missing or unreadable record is a
disagreement with a cause, not a crash — the comparison must still produce a report when the jobs
that feed it failed.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

# Fields that must be identical: they describe WHAT was run, not WHERE.
IDENTITY_FIELDS = (
    "schema_version",
    "commit_sha",
    "corpus_version",
    "generator_version",
    "seed",
    "manifest_sha256",
    "generator_sha256",
    "mir_version",
    "backend_version",
    "runtime_version",
)

COUNT_FIELDS = (
    "case_count",
    "handwritten_count",
    "generated_count",
    "retained_count",
    "metamorphic_family_count",
    "metamorphic_group_count",
    "passed_count",
    "failed_count",
    "skipped_count",
    "quarantined_count",
)

# Recorded, reported, and never a disagreement.
PLATFORM_FIELDS = ("target_triple", "os", "architecture", "rustc", "cargo", "python")


def load(path: pathlib.Path) -> tuple[dict | None, str | None]:
    if not path.is_file():
        return None, f"{path} does not exist"
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"{path} is unreadable: {exc}"


def per_case_index(path: pathlib.Path | None) -> tuple[dict, str | None]:
    if path is None:
        return {}, None
    records, problem = load(path)
    if problem:
        return {}, problem
    if not isinstance(records, list):
        return {}, f"{path} is not a list of per-case records"
    return {r["case_id"]: r for r in records}, None


def compare(a_path: pathlib.Path, b_path: pathlib.Path, per_case: list[pathlib.Path]) -> tuple[list[str], list[str]]:
    problems: list[str] = []
    notes: list[str] = []

    a, a_problem = load(a_path)
    b, b_problem = load(b_path)
    for problem in (a_problem, b_problem):
        if problem:
            problems.append(problem)
    if a is None or b is None:
        problems.append(
            "TIER-1 DISAGREEMENT: at least one corpus record is missing, so agreement cannot be "
            "claimed. A missing record is not a pass."
        )
        return problems, notes

    for field in IDENTITY_FIELDS:
        if a.get(field) != b.get(field):
            problems.append(
                f"{field} differs: {a.get(field)!r} vs {b.get(field)!r}"
            )
    for field in COUNT_FIELDS:
        if a.get(field) != b.get(field):
            problems.append(f"{field} differs: {a.get(field)!r} vs {b.get(field)!r}")

    for record, label in ((a, a_path.name), (b, b_path.name)):
        if record.get("result") != "PASS":
            problems.append(f"[{label}] result is {record.get('result')!r}, not PASS")
        if not record.get("full_evidence"):
            problems.append(
                f"[{label}] is not FULL evidence — a filtered or sharded run is a diagnostic run "
                "(§12.6) and cannot stand as Tier-1 evidence"
            )
        if str(record.get("dirty_worktree", "true")).lower() != "false":
            problems.append(f"[{label}] was produced from a DIRTY worktree")
        if record.get("failed_count"):
            problems.append(f"[{label}] reports {record['failed_count']} failed case(s)")
        if record.get("skipped_count"):
            problems.append(
                f"[{label}] reports {record['skipped_count']} skipped case(s); §16.3 makes a "
                "required skip a qualification failure"
            )

    triple_a = a.get("target_triple")
    triple_b = b.get("target_triple")
    if triple_a == triple_b:
        problems.append(
            f"both records report the same target triple ({triple_a!r}); comparing a platform "
            "against itself is not Tier-1 agreement"
        )
    for field in PLATFORM_FIELDS:
        if a.get(field) != b.get(field):
            notes.append(f"{field}: {a.get(field)!r} / {b.get(field)!r}")

    # The strongest clause: per-case observations. Two records can agree on every count while having
    # observed different bytes.
    if len(per_case) == 2:
        left, left_problem = per_case_index(per_case[0])
        right, right_problem = per_case_index(per_case[1])
        for problem in (left_problem, right_problem):
            if problem:
                problems.append(problem)
        if left and right:
            only_left = sorted(set(left) - set(right))
            only_right = sorted(set(right) - set(left))
            for case_id in only_left:
                problems.append(f"case {case_id} ran only on {a.get('target_triple')}")
            for case_id in only_right:
                problems.append(f"case {case_id} ran only on {b.get('target_triple')}")
            for case_id in sorted(set(left) & set(right)):
                l, r = left[case_id], right[case_id]
                if l.get("result") != r.get("result"):
                    problems.append(
                        f"case {case_id}: outcome class differs "
                        f"({l.get('result')!r} vs {r.get('result')!r})"
                    )
                elif l.get("observation_hash") != r.get("observation_hash"):
                    problems.append(
                        f"case {case_id}: SAME outcome class but DIFFERENT observation "
                        f"({l.get('observation_hash')} vs {r.get('observation_hash')}) — the two "
                        "targets ran the same program and saw different things"
                    )
    else:
        notes.append(
            "per-case records were not supplied, so only summary-level agreement was checked"
        )
    return problems, notes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first", type=pathlib.Path)
    parser.add_argument("second", type=pathlib.Path)
    parser.add_argument("--per-case", type=pathlib.Path, nargs=2, default=None)
    parser.add_argument("--output", type=pathlib.Path, default=None)
    args = parser.parse_args()

    problems, notes = compare(args.first, args.second, args.per_case or [])
    verdict = "TIER-1 CORPUS AGREEMENT" if not problems else "TIER-1 CORPUS DISAGREEMENT"

    lines = [f"# {verdict}", ""]
    if problems:
        lines.append("## Disagreements")
        lines.extend(f"- {problem}" for problem in problems)
        lines.append("")
    if notes:
        lines.append("## Platform metadata (expected to differ; not disagreement)")
        lines.extend(f"- {note}" for note in notes)
        lines.append("")
    report = "\n".join(lines)
    print(report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
