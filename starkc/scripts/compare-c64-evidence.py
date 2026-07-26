#!/usr/bin/env python3
"""WP-C6.4c — compare two Tier-1 evidence records.

§10.4 requires "a comparison job [that] verifies both Tier-1 records refer to the same commit and
compatible versions", and §17.2 requires every disagreement to be resolved or to block closure.
Listing two platforms' results side by side is not comparing them; this script is what turns two
independent runs into one Tier-1 agreement claim, or refuses to.

# The rule that shapes everything below

**Agreement must not be reachable from incomplete or differently-shaped evidence.** Two records
that both omit a field agree on nothing; two records whose ignored-test *counts* match while their
ignored-test *identities* differ describe different runs. So every check below is written to fail
on absence, not only on difference:

- required metadata must be present and non-blank, per record, before any comparison happens;
- the two records must be the two DIFFERENT Tier-1 triples, each self-consistent
  (`selected_target_triple == host_triple`, tier-1, 64-bit);
- every command in the fixed qualification set must be present in both, and compared on verdict,
  exit code, and all four counts — plus the full identities of ignored and unclassified-ignored
  tests, not merely how many there were;
- an ignored count that cannot be fully attributed to names fails, because an unattributed ignore
  cannot have been classified;
- any deviation, dirty worktree, quick mode, filtered run, unclassified ignore or self-skipped
  required test fails outright.

# What is expected to differ, and is therefore never compared

Host triple, selected target, OS, architecture, rustc/Cargo version, runner identity, Python
version, timestamps, durations, and the absolute interpreter path inside a command's argv. These
are recorded so a reader can see what the two platforms were. Comparing them would fail every
honest run.

Usage:

    python3 scripts/compare-c64-evidence.py macos-arm64.json linux-x64.json [--out summary.md]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import target_matrix  # noqa: E402  (path is set immediately above)

# From `starkc/target-matrix.json`, pinned to `src/target.rs` in both directions by
# `tests/c64_platform_matrix.rs::target_matrix_json_matches_the_compiler`.
TIER1 = frozenset(target_matrix.tier1_triples())

SCHEMA = "c6.4-evidence-1"

# Fields whose disagreement means the two runs are not describing the same thing.
IDENTICAL = (
    "schema_version",
    "work_package",
    "commit_sha",
    "compiler_version",
    "mir_version",
    "mir_runtime_surface",
    "backend_version",
    "runtime_version",
    "layout_contract",
    "layout_contract_version",
    "compiler_layout_revision",
    "target_tier",
    "target_pointer_width",
    "profile",
    "determinism_result",
    "failed_count",
    "ignored_count",
    "skipped_count",
    # An ignore classified on one platform and not the other would mean the two runs made
    # different observations while reporting the same totals.
    "unclassified_ignores",
    "classified_ignores",
    "generated_corpus_status",
    "generated_corpus_version",
    "generated_corpus_case_count",
    "required_steps",
    "overall_result",
)

# Present and non-blank in EACH record, checked before any comparison. A field both records omit
# would otherwise "agree".
REQUIRED_NON_BLANK = (
    "commit_sha",
    "compiler_version",
    "mir_version",
    "mir_runtime_surface",
    "backend_version",
    "runtime_version",
    "layout_contract",
    "profile",
    "host_triple",
    "selected_target_triple",
    "target_tier",
    "rustc_version_verbose",
    "cargo_version",
    "determinism_result",
    "generated_corpus_status",
    "overall_result",
)

# Present and a positive integer in each record.
REQUIRED_POSITIVE_INT = ("layout_contract_version", "compiler_layout_revision", "target_pointer_width")

# Fields that are supposed to differ. Recorded in the summary so a reader can see *what* the two
# platforms were, and never compared.
EXPECTED_TO_DIFFER = (
    "host_triple",
    "selected_target_triple",
    "os_name",
    "os_version",
    "architecture",
    "rustc_version_verbose",
    "cargo_version",
    "runner_provider",
    "python_version",
)

# WP-C6.5 §16.5 handoff. Until the C6.5 corpus existed this was `BLOCKED-BY-C6.5` and a nonzero case
# count was itself a problem; the corpus now exists and is replayed by this harness, so the required
# state inverts: `PASS`, with a real case count. A record reporting `BLOCKED-BY-C6.5` today means the
# qualification run did not execute the corpus steps — which is a missing observation, not a legacy
# state to tolerate.
EXPECTED_CORPUS_STATUS = "PASS"

# Per-command fields compared for exact equality across the two platforms.
COMMAND_FIELDS = ("ok", "exit_code", "passed", "failed", "ignored", "skipped")


def blank(value: object) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def normative_argv(argv: list[str]) -> list[str]:
    """The part of a command's argv that must agree across platforms.

    A `cargo` invocation must be identical — the flags are the observation. A Python step is
    launched through `sys.executable`, an absolute path that legitimately differs per runner, so
    its interpreter is dropped and the script and its arguments are compared.
    """
    if not argv:
        return []
    if argv[0] == "cargo":
        return list(argv)
    return list(argv[1:])


def validate_record(record: dict, label: str, problems: list[str]) -> None:
    """Everything checkable about ONE record, before the two are compared."""
    if record.get("schema_version") != SCHEMA:
        problems.append(
            f"[{label}] schema_version is {record.get('schema_version')!r}, expected {SCHEMA!r}"
        )

    for field in REQUIRED_NON_BLANK:
        if field not in record:
            problems.append(f"[{label}] required field `{field}` is absent")
        elif blank(record[field]):
            problems.append(f"[{label}] required field `{field}` is blank")

    for field in REQUIRED_POSITIVE_INT:
        value = record.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            problems.append(f"[{label}] `{field}` must be a positive integer, got {value!r}")

    # Platform identity, self-consistency first.
    host = record.get("host_triple", "")
    selected = record.get("selected_target_triple", "")
    if host and selected and host != selected:
        problems.append(
            f"[{label}] selected target `{selected}` differs from host `{host}`; C6.4 qualifies "
            "host builds only"
        )
    entry = target_matrix.classify(host) if host else None
    if entry is None:
        problems.append(f"[{label}] host `{host}` is not a target STARK names")
    else:
        if not entry.is_tier1:
            problems.append(f"[{label}] host `{host}` is {entry.tier}, not tier-1")
        if record.get("target_tier") != entry.tier:
            problems.append(
                f"[{label}] records tier {record.get('target_tier')!r} but `{host}` is "
                f"{entry.tier!r}"
            )
        if record.get("target_pointer_width") != entry.pointer_width:
            problems.append(
                f"[{label}] records pointer width {record.get('target_pointer_width')!r} but "
                f"`{host}` is {entry.pointer_width}"
            )
        if record.get("layout_contract") != entry.layout_contract:
            problems.append(
                f"[{label}] records layout contract {record.get('layout_contract')!r} but "
                f"`{host}` declares {entry.layout_contract!r}"
            )
    if record.get("target_pointer_width") not in (None, 64):
        problems.append(
            f"[{label}] pointer width {record.get('target_pointer_width')!r}: every tier-1 target "
            "is 64-bit, and the runtime's checked-index surface is qualified on that basis"
        )

    # Validity of the run itself.
    if record.get("dirty_worktree"):
        problems.append(f"[{label}] produced from a dirty worktree")
    if record.get("quick_mode"):
        problems.append(f"[{label}] is a quick-mode run, which is not a qualification claim")
    if record.get("overall_result") != "PASS":
        problems.append(f"[{label}] overall_result is {record.get('overall_result')!r}, not PASS")
    for deviation in record.get("deviations") or []:
        problems.append(f"[{label}] deviation: {deviation}")
    if record.get("unclassified_ignores"):
        problems.append(
            f"[{label}] unclassified ignored test(s): "
            f"{', '.join(record['unclassified_ignores'])}"
        )
    if record.get("skipped_count"):
        problems.append(f"[{label}] {record['skipped_count']} self-skipped required test(s)")
    if record.get("failed_count"):
        problems.append(f"[{label}] {record['failed_count']} failed test(s)")

    # The fixed qualification set must all be present, and each must have passed.
    commands = {c["name"]: c for c in record.get("commands") or []}
    required = record.get("required_steps") or []
    if not required:
        problems.append(f"[{label}] declares no required_steps; cannot tell a complete run from a filtered one")
    for name in required:
        if name not in commands:
            problems.append(f"[{label}] required command `{name}` is missing from the record")
        elif not commands[name].get("ok"):
            problems.append(f"[{label}] required command `{name}` did not pass")

    # Ignored counts must be fully attributable to names, per command.
    for name, command in commands.items():
        names = command.get("ignored_names")
        if names is None:
            problems.append(f"[{label}] command `{name}` records no ignored_names list")
        elif len(names) != command.get("ignored", 0):
            problems.append(
                f"[{label}] command `{name}` reports {command.get('ignored')} ignored but names "
                f"{len(names)}; an unattributed ignore cannot have been classified"
            )

    # Generated corpus: the one state C6.4 may be in.
    status = record.get("generated_corpus_status")
    if status != EXPECTED_CORPUS_STATUS:
        problems.append(
            f"[{label}] generated_corpus_status is {status!r}, expected "
            f"{EXPECTED_CORPUS_STATUS!r}. If C6.5 has landed, matrix row 24 and this constant "
            "must be revisited together."
        )
    if not record.get("generated_corpus_case_count"):
        problems.append(
            f"[{label}] reports no generated corpus cases; row 24 requires the corpus to have been "
            "replayed, and a zero count means it was not"
        )
    if not record.get("generated_corpus_version"):
        problems.append(f"[{label}] records no generated_corpus_version")

    # Determinism.
    if record.get("determinism_result") != "match":
        problems.append(
            f"[{label}] determinism rerun is {record.get('determinism_result')!r}, not 'match'"
        )
    first = record.get("determinism_first_hash")
    second = record.get("determinism_second_hash")
    if blank(first) or blank(second):
        problems.append(f"[{label}] determinism hashes are missing")
    elif first != second:
        problems.append(f"[{label}] determinism hashes differ: {first!r} vs {second!r}")


def compare_records(a: dict, b: dict, problems: list[str]) -> None:
    """Everything that must agree BETWEEN the two records."""
    triples = {a.get("host_triple"), b.get("host_triple")}
    if triples != TIER1:
        if len(triples) == 1:
            problems.append(
                f"both records are for the same target ({next(iter(triples))}); a Tier-1 claim "
                "needs one record from each"
            )
        else:
            problems.append(
                f"records are not the two Tier-1 targets: got {sorted(str(t) for t in triples)}, "
                f"expected {sorted(TIER1)}"
            )

    for key in IDENTICAL:
        if a.get(key) != b.get(key):
            problems.append(
                f"{key}: {a.get('host_triple')}={a.get(key)!r} vs "
                f"{b.get('host_triple')}={b.get(key)!r}"
            )

    commands_a = {c["name"]: c for c in a.get("commands") or []}
    commands_b = {c["name"]: c for c in b.get("commands") or []}
    if set(commands_a) != set(commands_b):
        only_a = sorted(set(commands_a) - set(commands_b))
        only_b = sorted(set(commands_b) - set(commands_a))
        problems.append(
            f"command sets differ: only on {a.get('host_triple')}={only_a}, only on "
            f"{b.get('host_triple')}={only_b}"
        )
    for name in sorted(set(commands_a) & set(commands_b)):
        ca, cb = commands_a[name], commands_b[name]
        for field in COMMAND_FIELDS:
            if ca.get(field) != cb.get(field):
                problems.append(
                    f"{name}.{field}: {a.get('host_triple')}={ca.get(field)!r} vs "
                    f"{b.get('host_triple')}={cb.get(field)!r} — the same commands must make the "
                    "same observations"
                )
        if normative_argv(ca.get("argv") or []) != normative_argv(cb.get("argv") or []):
            problems.append(
                f"{name}.argv differs where it is normative: "
                f"{normative_argv(ca.get('argv') or [])} vs {normative_argv(cb.get('argv') or [])}"
            )
        for field in ("ignored_names", "unclassified_ignores"):
            if sorted(ca.get(field) or []) != sorted(cb.get(field) or []):
                problems.append(
                    f"{name}.{field}: {sorted(ca.get(field) or [])} vs "
                    f"{sorted(cb.get(field) or [])} — identities differ even where counts may not"
                )


def render(a: dict | None, b: dict | None, problems: list[str], missing: list[str]) -> str:
    agreed = not problems and not missing
    a = a or {}
    b = b or {}
    label_a = a.get("host_triple", "(absent)")
    label_b = b.get("host_triple", "(absent)")
    lines = [
        "# C6.4 Tier-1 qualification summary",
        "",
        f"- Commit: `{a.get('commit_sha') or b.get('commit_sha') or '(absent)'}`",
        f"- Records: `{label_a}` and `{label_b}`",
        f"- Generated corpus: {a.get('generated_corpus_status') or '(absent)'} "
        "(see `WP-C6.4.md` §1.2)",
        "",
    ]
    if missing:
        lines += ["## Missing records", ""] + [f"- {m}" for m in missing] + [""]

    lines += [
        "## Platform identities",
        "",
        f"| Field | {label_a} | {label_b} |",
        "|---|---|---|",
    ]
    for key in EXPECTED_TO_DIFFER:
        va = str(a.get(key, "—")).splitlines()[0] if a.get(key) else "—"
        vb = str(b.get(key, "—")).splitlines()[0] if b.get(key) else "—"
        lines.append(f"| {key} | {va} | {vb} |")

    lines += ["", "## Required agreement", "", "| Field | Value | Agrees |", "|---|---|---|"]
    for key in IDENTICAL:
        agrees = "yes" if (key in a and key in b and a.get(key) == b.get(key)) else "**NO**"
        lines.append(f"| {key} | {a.get(key, '—')} | {agrees} |")

    commands_a = {c["name"]: c for c in a.get("commands") or []}
    commands_b = {c["name"]: c for c in b.get("commands") or []}
    lines += [
        "",
        "## Per-command results",
        "",
        f"| Command | {label_a} | {label_b} |",
        "|---|---|---|",
    ]

    def cell(cs: dict, name: str) -> str:
        c = cs.get(name)
        if c is None:
            return "absent"
        return f"{'PASS' if c['ok'] else 'FAIL'} ({c['passed']} passed, {c['ignored']} ignored)"

    for name in sorted(set(commands_a) | set(commands_b)):
        lines.append(f"| {name} | {cell(commands_a, name)} | {cell(commands_b, name)} |")

    lines += [
        "",
        "## Result",
        "",
        f"**{'TIER-1 AGREEMENT' if agreed else 'TIER-1 DISAGREEMENT'}**",
        "",
    ]
    if problems:
        lines += ["### Problems", ""] + [f"- {p}" for p in problems] + [""]
    return "\n".join(lines)


def load(path: Path, missing: list[str]) -> dict | None:
    """Load a record, or record why it could not be loaded.

    Returning `None` rather than exiting is deliberate: §10.4's comparison job runs even when a
    qualification job failed, and a summary that says WHICH record is absent is more useful than
    a job that was skipped.
    """
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        missing.append(f"{path.name}: no such file — its qualification job did not produce one")
    except json.JSONDecodeError as error:
        missing.append(f"{path.name}: not valid JSON ({error})")
    except OSError as error:
        missing.append(f"{path.name}: unreadable ({error})")
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare two C6.4 Tier-1 evidence records")
    parser.add_argument("records", nargs=2, type=Path)
    parser.add_argument("--out", type=Path, help="write a Markdown summary here")
    args = parser.parse_args(argv)

    problems: list[str] = []
    missing: list[str] = []
    a = load(args.records[0], missing)
    b = load(args.records[1], missing)

    for record, path in ((a, args.records[0]), (b, args.records[1])):
        if record is not None:
            validate_record(record, record.get("host_triple") or path.name, problems)
    if a is not None and b is not None:
        compare_records(a, b, problems)

    summary = render(a, b, problems, missing)
    print(summary)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(summary, encoding="utf-8")
    return 0 if (not problems and not missing) else 1


if __name__ == "__main__":
    sys.exit(main())
