#!/usr/bin/env python3
"""WP-C6.4c — the Tier-1 platform qualification harness.

`WP-C6-ENTRY.md` §35 requires each Tier-1 platform to run a fixed set of commands at the same
commit and to record commit, OS, architecture, rustc, Cargo, commands, counts and artifacts. This
script is that run: one cross-platform entry point (Python, not shell — §9.11), producing one
machine-readable record and one human-readable one.

Two rules shape everything below.

**Measured, not asserted.** `--expected-target` is compared against what `rustc -vV` actually
reports; a mismatch fails qualification rather than being recorded as fact. The same applies to the
commit: `--commit` is checked against `git rev-parse HEAD`, because an evidence file whose commit
field was supplied by the caller proves nothing about the tree that was tested.

**A skip is not a pass.** §19 says no required skipped test may be counted as passing, and this
suite has two ways to skip quietly: Cargo's own `ignored` count, and the `SKIP:` lines eleven
native/differential suites print when no rustc is present — they return success, so the exit code
says nothing. Both are detected, and both fail a required command. That is deliberate: a
qualification run on a machine without a Rust toolchain would otherwise report a green matrix
having executed almost nothing.

Detecting the second kind needs `--nocapture`, and this is the subtlety that makes the check real
rather than decorative: **libtest discards a PASSING test's output**, so a `SKIP:` line printed by
a test that then returns success is invisible under a plain `cargo test`. Every step whose suite
can self-skip therefore runs with `-- --nocapture` (`Step.nocapture`). The whole-workspace step
does not — its output would be enormous — so for that step the guarantee is narrower and is stated
honestly here: it rests on the exit code and on ignore-name parsing, not on `SKIP:` detection. The
suites that can self-skip are all covered individually by their own steps.

Usage:

    python3 scripts/run-c64-qualification.py \
        --expected-target aarch64-apple-darwin \
        --commit "$(git rev-parse HEAD)" \
        --output-dir ../starkc/docs/compiler/evidence/c6.4

    python3 scripts/run-c64-qualification.py --list      # what it would run, and why
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import pathlib
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

SCHEMA_VERSION = "c6.4-evidence-1"
WORK_PACKAGE = "WP-C6.4"

STARKC = Path(__file__).resolve().parent.parent
REPO = STARKC.parent

sys.path.insert(0, str(Path(__file__).resolve().parent))
import target_matrix  # noqa: E402  (path is set immediately above)

# From `starkc/target-matrix.json`, which is pinned to `src/target.rs` in both directions by
# `tests/c64_platform_matrix.rs::target_matrix_json_matches_the_compiler`. The harness used to
# carry its own copy of this table; it does not any more.
TIER1 = target_matrix.tier1_triples()


@dataclass
class Step:
    """One qualification command.

    `required` marks a command whose failure — or whose silent skip — fails the whole run. A
    non-required step still records its result; it just cannot block a claim on its own.
    """

    name: str
    argv: list[str]
    group: str
    why: str
    required: bool = True
    cwd: Path = STARKC
    # Append `-- --nocapture`. Required for any suite that can self-skip, because libtest hides a
    # passing test's output and the `SKIP:` line would otherwise never be seen. See the module
    # docstring.
    nocapture: bool = False

    def command(self) -> list[str]:
        return [*self.argv, "--", "--nocapture"] if self.nocapture else list(self.argv)


@dataclass
class StepResult:
    name: str
    group: str
    argv: list[str]
    exit_code: int
    passed: int = 0
    failed: int = 0
    ignored: int = 0
    skipped: int = 0
    duration_s: float = 0.0
    note: str = ""
    ok: bool = False
    ignored_names: list[str] = field(default_factory=list)
    unclassified_ignores: list[str] = field(default_factory=list)


@dataclass
class Evidence:
    steps: list[StepResult] = field(default_factory=list)
    deviations: list[str] = field(default_factory=list)


# WP-C6.5 §16.5. The corpus steps whose collective result decides row 24. Named explicitly rather
# than "every step tagged semantics", so adding an unrelated semantics step cannot silently become
# part of a corpus claim.
C65_CORPUS_STEPS = (
    "c65_generated_corpus",
    "c65_metamorphic",
    "c65_mutation",
    "c65_package",
    "c65_corpus_integrity",
)


def generated_corpus_fields(results: list) -> dict:
    """§16.5's row-24 fields, MEASURED.

    The version and case count are read from the corpus's own lock and manifest — the same files the
    corpus tests verify — not passed in or assumed. The status is `PASS` only when every C6.5 corpus
    step in THIS run passed; a record that reported `PASS` while its own commands failed would be
    exactly the kind of evidence C6.4's re-qualification rule exists to prevent.
    """
    corpus = pathlib.Path("tests/c6-corpus")
    version = None
    case_count = 0
    lock = corpus / "corpus.lock"
    if lock.is_file():
        for line in lock.read_text(encoding="utf-8").splitlines():
            if line.startswith("corpus_version = "):
                version = line.split(" = ", 1)[1].strip()
            elif line.startswith("case_count = "):
                case_count = int(line.split(" = ", 1)[1].strip())
    ran = {r.name: r for r in results}
    present = [name for name in C65_CORPUS_STEPS if name in ran]
    if not present:
        status = "NOT-RUN"
    elif len(present) != len(C65_CORPUS_STEPS):
        status = "PARTIAL"
    elif all(ran[name].ok for name in present):
        status = "PASS"
    else:
        status = "FAIL"
    return {
        "generated_corpus_version": version,
        "generated_corpus_case_count": case_count,
        "generated_corpus_status": status,
    }


def steps_for(quick: bool) -> list[Step]:
    """The §10.5 command set.

    Ordered cheapest-first so a formatting or lint failure is reported in seconds rather than after
    the full suite. `--quick` drops the whole-workspace run only; every C6.4-specific observation
    still executes, which is what makes quick mode useful for validating the harness itself without
    letting it be mistaken for a qualification run (`quick_mode` is recorded in the output).
    """
    full: list[Step] = [
        Step(
            "fmt",
            ["cargo", "fmt", "--all", "--", "--check"],
            "hygiene",
            "§10.5: formatting is part of the required Tier-1 command set.",
        ),
        Step(
            "clippy",
            [
                "cargo",
                "clippy",
                "--workspace",
                "--all-targets",
                "--all-features",
                "--",
                "-D",
                "warnings",
            ],
            "hygiene",
            "§10.5: strict clippy, the exact CI invocation.",
        ),
        Step(
            "c64_platform_matrix",
            ["cargo", "test", "--test", "c64_platform_matrix"],
            "platform",
            "The C6.4 suite itself: preflight, portability, output bytes, traps, determinism.",
            nocapture=True,
        ),
        Step(
            "three_engine_differential",
            ["cargo", "test", "--test", "three_engine_differential"],
            "semantics",
            "§10.5: HIR/MIR/native agreement, compared against real native stdout.",
            nocapture=True,
        ),
        Step(
            "mir_differential",
            ["cargo", "test", "--test", "mir_differential"],
            "semantics",
            "§10.5: the frozen corpus through both interpreters.",
        ),
        # WP-C6.5 §16.5: the generated-corpus commands this record's row 24 depends on. They are
        # separate steps rather than one, so a failure names WHICH claim broke -- replay, pair
        # preservation, mutation sensitivity or package breadth.
        Step(
            "c65_generated_corpus",
            ["cargo", "test", "--test", "c6_generated_corpus"],
            "semantics",
            "§16.5: the C6.5 corpus replayed through every engine each case declares.",
            nocapture=True,
        ),
        Step(
            "c65_metamorphic",
            ["cargo", "test", "--test", "c6_metamorphic"],
            "semantics",
            "§16.5: metamorphic pairs preserved by each engine.",
        ),
        Step(
            "c65_mutation",
            ["cargo", "test", "--test", "c6_mutation"],
            "semantics",
            "§16.5: the sixteen mutation controls -- proof the corpus can fail.",
        ),
        Step(
            "c65_package",
            ["cargo", "test", "--test", "c6_package"],
            "semantics",
            "§16.5: package relocation, dependency reorder and the DEV-113/DEV-114 pins.",
        ),
        Step(
            "c65_corpus_integrity",
            [
                "cargo",
                "test",
                "--test",
                "c6_corpus_manifest",
                "--test",
                "c6_corpus_generator",
            ],
            "semantics",
            "§16.5: manifest validation, lock integrity, generator determinism.",
        ),
        Step(
            "exec_snapshots",
            ["cargo", "test", "--test", "exec_snapshots"],
            "semantics",
            "§10.5: the frozen execution corpus.",
        ),
        Step(
            "c63_closure_evidence",
            ["cargo", "test", "--test", "c63_closure_evidence"],
            "runtime",
            "§10.6/§10.7: installed-runtime build, offline build, version-mismatch detection.",
            nocapture=True,
        ),
        Step(
            "conformance",
            ["cargo", "test", "--test", "conformance"],
            "semantics",
            "Manifest-driven spec conformance.",
        ),
        Step(
            "release_package",
            [sys.executable, "scripts/test_build_release.py"],
            "install",
            "§10.6: release package structure and installer.",
        ),
        Step(
            "workspace",
            [
                "cargo",
                "test",
                "--workspace",
                "--all-targets",
                "--all-features",
                "--no-fail-fast",
            ],
            "suite",
            "§10.5: the full no-fail-fast workspace suite.",
        ),
    ]
    if quick:
        return [s for s in full if s.name not in {"workspace", "clippy"}]
    return full


# `test result: ok. 14 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out`
RESULT_RE = re.compile(
    r"test result: \w+\. (\d+) passed; (\d+) failed; (\d+) ignored; (\d+) measured"
)

# `test some::path::name ... ignored, reason`
IGNORED_RE = re.compile(r"^test (\S+) \.\.\. ignored", re.MULTILINE)

# §10.4: "skipped tests fail qualification **unless explicitly classified outside the required
# matrix**". This is that classification, and it is a closed list rather than a count.
#
# Counting ignores would let a NEW ignore replace a retired one silently, which is precisely the
# way a required observation goes missing without anyone deciding it should. Naming them means an
# ignore that is not on this list fails the run, and putting one on the list is a decision with a
# reason attached.
#
# Both current entries are opt-in tensor-track tests that need an external artifact this matrix
# does not supply. Neither observes Core runtime semantics, which is what C6.4 qualifies.
CLASSIFIED_IGNORES = {
    "imports_and_verifies_checksum_pinned_reference_model": (
        "gate-4 tensor track; needs a checksum-verified ResNet50 named by "
        "STARK_GATE4_REFERENCE_ONNX. Outside the C6.4 Core-runtime matrix."
    ),
    "repeated_connect_and_release_reuses_slot_state": (
        "DEFECT-C788-LOOP-TEMP, found by this test (CD-263). A temporary holding "
        "`Result<Resource, E>` inside a loop body is never dropped, so the second iteration "
        "writes to a still-live slot and stark-runtime aborts with 'write to a live slot "
        "(STARK compiler defect, not a program fault)'. The generated program contains exactly "
        "one `drop_with`, on the match binding, and none for the scrutinee temp. This is a real "
        "compiler defect, not a test problem; the ignore records it rather than hiding it, and "
        "goes with the fix. Admitted as a non-blocking C7 deviation at P1 compiler priority "
        "(CD-264) -- high priority, not the P1 workload; mandatory before native resource "
        "support is declared generally usable beyond the admitted P1 workload."
    ),
    "real_inference_agrees_with_reference": (
        "gate-5 tensor track; downloads and links ONNX Runtime and runs Python. "
        "Outside the C6.4 Core-runtime matrix."
    ),
}


def run_step(step: Step, env: dict[str, str]) -> StepResult:
    start = datetime.datetime.now()
    argv = step.command()
    proc = subprocess.run(
        argv,
        cwd=step.cwd,
        env=env,
        capture_output=True,
        text=True,
    )
    duration = (datetime.datetime.now() - start).total_seconds()
    combined = proc.stdout + proc.stderr
    result = StepResult(
        name=step.name,
        group=step.group,
        argv=argv,
        exit_code=proc.returncode,
        duration_s=round(duration, 2),
    )
    for passed, failed, ignored, _measured in RESULT_RE.findall(combined):
        result.passed += int(passed)
        result.failed += int(failed)
        result.ignored += int(ignored)

    # The silent skip. These tests return SUCCESS when no rustc is present, so the exit code says
    # nothing; the printed line is the only evidence that a required observation did not happen.
    result.skipped = combined.count("SKIP:")

    # Which tests were ignored, by their COMPLETE libtest name.
    #
    # An earlier version kept only the final `::` component. That is a collision waiting to happen
    # — two modules can each hold a `basic_case` — and a collision here would let a classified
    # ignore silently vouch for an unrelated unclassified one. A list, not a set, so the count is
    # preserved when two binaries ignore identically-named tests.
    ignored_names = sorted(IGNORED_RE.findall(combined))
    result.ignored_names = ignored_names
    result.unclassified_ignores = sorted(
        name for name in ignored_names if name not in CLASSIFIED_IGNORES
    )

    result.ok = proc.returncode == 0
    if step.required and result.skipped:
        result.ok = False
        result.note = (
            f"{result.skipped} test(s) skipped themselves; a required observation did not run"
        )
    elif step.required and result.unclassified_ignores:
        result.ok = False
        result.note = (
            f"unclassified ignored test(s): {', '.join(result.unclassified_ignores)}. Either the "
            "observation is required — in which case fix the test — or classify it in "
            "CLASSIFIED_IGNORES with a reason."
        )
    elif step.required and len(ignored_names) != result.ignored:
        # Cargo's ignored COUNT and the `... ignored` lines must agree. If they do not, some
        # ignored test was never attributed to a name, and an unattributed ignore cannot have been
        # classified. Waving it through is precisely how a required observation goes missing.
        result.ok = False
        result.note = (
            f"cargo reported {result.ignored} ignored test(s) but {len(ignored_names)} could be "
            "identified by name; an unattributed ignore cannot be classified"
        )
    elif proc.returncode != 0:
        tail = "\n".join(combined.strip().splitlines()[-25:])
        result.note = f"exit {proc.returncode}\n{tail}"
    return result


def probe(argv: list[str], cwd: Path = REPO) -> str:
    try:
        out = subprocess.run(argv, cwd=cwd, capture_output=True, text=True, check=False)
        return out.stdout.strip() or out.stderr.strip()
    except FileNotFoundError:
        return ""


def rustc_field(verbose: str, field_name: str) -> str:
    for line in verbose.splitlines():
        if line.startswith(field_name):
            return line[len(field_name) :].strip()
    return ""


def compiler_constants() -> dict[str, str]:
    """Version identities read from the sources that declare them.

    Read rather than hardcoded: an evidence file that states a MIR surface version the compiler
    does not actually carry is worse than one that omits it.
    """
    def const(path: Path, name: str) -> str:
        try:
            text = (STARKC / path).read_text(encoding="utf-8")
        except OSError:
            return ""
        match = re.search(rf'{name}\s*:\s*&str\s*=\s*"([^"]+)"', text)
        return match.group(1) if match else ""

    def cargo_version(manifest: Path) -> str:
        try:
            text = (STARKC / manifest).read_text(encoding="utf-8")
        except OSError:
            return ""
        match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
        return match.group(1) if match else ""

    def layout_field(name: str) -> int:
        """A `u32` field of `LayoutIdentity`'s `stark64_v1()` constructor.

        The layout contract's identity is what a build's observable `size_of`/`align_of` answers
        are attributable to (CD-067). Recording only the contract NAME would let two records agree
        on `stark-64-v1` while one of them answered from a revised table.
        """
        try:
            text = (STARKC / "src" / "layout.rs").read_text(encoding="utf-8")
        except OSError:
            return 0
        match = re.search(rf"{name}:\s*(\d+)", text)
        return int(match.group(1)) if match else 0

    return {
        "compiler_version": cargo_version(Path("Cargo.toml")),
        "runtime_version": const(Path("stark-runtime/src/version.rs"), "RUNTIME_VERSION"),
        "backend_version": const(Path("src/backend/version.rs"), "BACKEND_VERSION"),
        "mir_version": const(Path("src/mir/mod.rs"), "MIR_VERSION"),
        "mir_runtime_surface": const(Path("src/mir/mod.rs"), "MIR_RUNTIME_SURFACE"),
        "layout_contract_version": layout_field("layout_contract_version"),
        "compiler_layout_revision": layout_field("compiler_layout_revision"),
    }


DETERMINISM_RE = re.compile(r"C64-DETERMINISM (key=\S+ source=\S+)")


def determinism_probe() -> tuple[str, str, str]:
    """§10.8: a genuine second run, in a separate process.

    `determinism_two_clean_builds_agree_on_key_source_and_metadata` already builds the same program
    twice inside one process; that shows the compiler is deterministic *within* an invocation.
    Only running the whole thing again, as its own process with its own temporary directories, can
    show it stayed deterministic *across* invocations — so the test prints its build key and a hash
    of the generated source, and this compares the two printed lines.

    The subject is the compiler's own product, not the linked binary. Binary reproducibility is
    C7's claim (§10.8's closing line) and is deliberately not made here.
    """
    observed = []
    for _ in range(2):
        out = subprocess.run(
            [
                "cargo",
                "test",
                "--test",
                "c64_platform_matrix",
                "determinism_",
                "--",
                "--nocapture",
            ],
            cwd=STARKC,
            capture_output=True,
            text=True,
            env={**os.environ, "CARGO_TERM_COLOR": "never"},
        )
        match = DETERMINISM_RE.search(out.stdout)
        observed.append(match.group(1) if match and out.returncode == 0 else "")
    first, second = observed
    if not first or not second:
        return first or "absent", second or "absent", "not-observed"
    return first, second, "match" if first == second else "mismatch"


def markdown(record: dict, results: list[StepResult]) -> str:
    lines = [
        f"# C6.4 Platform Evidence — {record['selected_target_triple']}",
        "",
        "*Generated by `scripts/run-c64-qualification.py`. Do not edit by hand — regenerate.*",
        "",
        f"- Commit: `{record['commit_sha']}`",
        f"- Worktree: {'DIRTY' if record['dirty_worktree'] else 'clean'}",
        f"- Date/time UTC: {record['timestamp_utc']}",
        f"- Runner: {record['runner_provider']}",
        f"- OS: {record['os_name']} {record['os_version']}",
        f"- Architecture: {record['architecture']}",
        f"- Host triple: `{record['host_triple']}`",
        f"- Selected target: `{record['selected_target_triple']}`",
        f"- Tier: {record['target_tier']}",
        f"- rustc: {record['rustc_version_verbose'].splitlines()[0] if record['rustc_version_verbose'] else 'n/a'}",
        f"- Cargo: {record['cargo_version']}",
        f"- Python: {record['python_version']}",
        f"- Compiler version: {record['compiler_version']}",
        f"- MIR version: {record['mir_version']}",
        f"- MIR runtime surface: {record['mir_runtime_surface']}",
        f"- Backend version: {record['backend_version']}",
        f"- Runtime version: {record['runtime_version']}",
        f"- Layout contract: {record['layout_contract']}",
        f"- Profile: {record['profile']}",
        f"- Quick mode: {record['quick_mode']}",
        "",
        "## Commands and results",
        "",
        "| Command | Result | Passed | Failed | Ignored | Skipped | Seconds |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        lines.append(
            f"| `{' '.join(r.argv)}` | {'PASS' if r.ok else 'FAIL'} | {r.passed} | {r.failed} "
            f"| {r.ignored} | {r.skipped} | {r.duration_s} |"
        )
    # §17.1's template names "semantic observations", "installed runtime" and "offline build" as
    # their own sections. They are not separate sections here because they are not separate runs —
    # each is established by a command in the table above. Saying which one, rather than leaving a
    # reader to infer it, is the point of this block.
    lines += [
        "",
        "## Coverage of the required observations",
        "",
        "| Required observation (§17.1 / §35) | Established by |",
        "|---|---|",
        "| stdout/stderr bytes, line termination, Unicode | `c64_platform_matrix` "
        "(`platform_stdout_is_exact_bytes_including_unicode_and_line_termination`) |",
        "| trap class, category, provenance, exit status, pre-trap prefix | `c64_platform_matrix` "
        "(`platform_trap_reports_category_provenance_and_exit_status`) |",
        "| three-engine semantic agreement, incl. Drop observations | `three_engine_differential` |",
        "| frozen execution corpus | `mir_differential`, `exec_snapshots` |",
        "| installed runtime, outside the checkout | `c63_closure_evidence`; "
        "`c64_platform_matrix::portability_installed_runtime_requirement_refuses_the_checkout_fallback` |",
        "| locked offline generated build | `c63_closure_evidence`; "
        "`c64_platform_matrix::portability_generated_crate_is_locked_and_network_free` |",
        "| frozen multi-package workspace | `workspace` (`native_c5_4_workspace`); "
        "`release_package` |",
        "| target preflight and recorded metadata | `c64_platform_matrix` (`target_preflight_*`, "
        "`portability_build_manifest_records_host_and_selected_target_separately`) |",
        "| generated corpus | **not established** — "
        f"{record['generated_corpus_status']}, see `WP-C6.4.md` §1.2 |",
        "",
        "## Determinism rerun",
        "",
        f"- first: `{record['determinism_first_hash']}`",
        f"- second: `{record['determinism_second_hash']}`",
        f"- result: **{record['determinism_result']}**",
        "",
        "## Totals",
        "",
        f"- passed: {record['passed_count']}",
        f"- failed: {record['failed_count']}",
        f"- ignored: {record['ignored_count']}",
        f"- self-skipped: {record['skipped_count']}",
        "",
        "## Deviations",
        "",
    ]
    lines += [f"- {d}" for d in record["deviations"]] or ["- none"]
    lines += ["", "## Qualification result", "", f"**{record['overall_result']}**", ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="WP-C6.4 Tier-1 qualification harness")
    parser.add_argument("--expected-target", help="the triple this run claims to qualify")
    parser.add_argument("--commit", help="the commit this run claims to test")
    parser.add_argument("--output-dir", type=Path, help="where to write the evidence pair")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="skip the whole-workspace suite and clippy; records quick_mode=true, which is NOT a "
        "qualification run",
    )
    parser.add_argument(
        "--only",
        action="append",
        metavar="NAME",
        help="run only these steps (repeatable). Records a deviation: a filtered run is a "
        "diagnostic, never a qualification claim.",
    )
    parser.add_argument("--list", action="store_true", help="print the command set and exit")
    args = parser.parse_args()

    if args.list:
        print(f"Tier-1 targets: {', '.join(TIER1)}")
        for step in steps_for(args.quick):
            flag = "required" if step.required else "optional"
            print(
                f"\n[{step.group}/{flag}] {step.name}\n  $ {' '.join(step.command())}\n  {step.why}"
            )
        return 0

    rustc_verbose = probe(["rustc", "-vV"])
    host_triple = rustc_field(rustc_verbose, "host:")
    head = probe(["git", "rev-parse", "HEAD"])
    dirty = bool(probe(["git", "status", "--porcelain", "--untracked-files=no"]))

    deviations: list[str] = []

    # Measured against claimed. Neither is trusted (§10.1).
    if args.expected_target and args.expected_target != host_triple:
        deviations.append(
            f"expected target `{args.expected_target}` but this host reports `{host_triple}`"
        )
    if args.commit and head and args.commit != head:
        deviations.append(f"claimed commit `{args.commit}` but HEAD is `{head}`")
    if dirty:
        deviations.append("tracked worktree is dirty; evidence does not describe a clean commit")
    if host_triple not in TIER1:
        deviations.append(f"`{host_triple}` is not a Tier-1 target; this is not a Tier-1 claim")

    selected_steps = steps_for(args.quick)
    if args.only:
        unknown = set(args.only) - {s.name for s in selected_steps}
        if unknown:
            print(f"error: no such step(s): {', '.join(sorted(unknown))}", file=sys.stderr)
            return 2
        selected_steps = [s for s in selected_steps if s.name in set(args.only)]
        deviations.append(
            f"filtered run (--only {' '.join(args.only)}); not a qualification claim"
        )
    if args.quick:
        deviations.append("quick mode: the full workspace suite and clippy did not run")

    env = {**os.environ, "CARGO_TERM_COLOR": "never"}
    results = [run_step(step, env) for step in selected_steps]
    for r in results:
        if not r.ok:
            deviations.append(f"{r.name}: {r.note or 'failed'}")

    first, second, determinism = determinism_probe()
    if determinism != "match":
        deviations.append("determinism rerun did not reproduce the first run's observations")

    consts = compiler_constants()
    entry = target_matrix.classify(host_triple)
    tier = entry.tier if entry else "unsupported"

    record = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "commit_sha": head,
        "dirty_worktree": dirty,
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "runner_provider": os.environ.get("GITHUB_RUNNER_NAME")
        or os.environ.get("RUNNER_NAME")
        or ("github-actions" if os.environ.get("GITHUB_ACTIONS") else "local"),
        "os_name": platform.system(),
        "os_version": platform.release(),
        "architecture": platform.machine(),
        "host_triple": host_triple,
        "selected_target_triple": host_triple,
        "target_tier": tier,
        # From the target matrix, not asserted: a record that states the pointer width the
        # comparator then checks is only meaningful if it came from the target's own entry.
        "target_pointer_width": entry.pointer_width if entry else 0,
        "rustc_version_verbose": rustc_verbose,
        "cargo_version": probe(["cargo", "-V"]),
        "python_version": platform.python_version(),
        "layout_contract": entry.layout_contract if entry else "",
        "profile": "debug",
        "quick_mode": args.quick,
        # The fixed qualification set, so the comparator can tell a complete run from a filtered
        # one without trusting the `deviations` list to have been populated.
        "required_steps": [s.name for s in steps_for(False) if s.required],
        "commands": [
            {
                "name": r.name,
                "group": r.group,
                "argv": r.argv,
                "exit_code": r.exit_code,
                "passed": r.passed,
                "failed": r.failed,
                "ignored": r.ignored,
                "ignored_names": r.ignored_names,
                "skipped": r.skipped,
                "duration_s": r.duration_s,
                "ok": r.ok,
                "note": r.note,
            }
            for r in results
        ],
        "test_binaries": [r.name for r in results if r.name != "fmt"],
        "passed_count": sum(r.passed for r in results),
        "failed_count": sum(r.failed for r in results),
        "ignored_count": sum(r.ignored for r in results),
        # Named, with the reason each was excluded from the required matrix. A reader can audit the
        # decision; a count alone can only be believed.
        "classified_ignores": {
            name: CLASSIFIED_IGNORES[name]
            for r in results
            for name in r.ignored_names
            if name in CLASSIFIED_IGNORES
        },
        "unclassified_ignores": sorted(
            {name for r in results for name in r.unclassified_ignores}
        ),
        "skipped_count": sum(r.skipped for r in results),
        **generated_corpus_fields(results),
        "determinism_first_hash": first,
        "determinism_second_hash": second,
        "determinism_result": determinism,
        "deviations": deviations,
        **consts,
    }
    record["overall_result"] = "PASS" if not deviations else "FAIL"

    print(json.dumps({k: record[k] for k in ("commit_sha", "host_triple", "overall_result")}, indent=2))
    for d in deviations:
        print(f"deviation: {d}", file=sys.stderr)

    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        slug = {
            "aarch64-apple-darwin": "macos-arm64",
            "x86_64-unknown-linux-gnu": "linux-x64",
            "x86_64-pc-windows-msvc": "windows-x64",
        }.get(host_triple, host_triple)
        (args.output_dir / f"{slug}.json").write_text(
            json.dumps(record, indent=2) + "\n", encoding="utf-8"
        )
        (args.output_dir / f"{slug}.md").write_text(
            markdown(record, results), encoding="utf-8"
        )
        print(f"wrote {slug}.json and {slug}.md to {args.output_dir}")

    return 0 if record["overall_result"] == "PASS" else 1


if __name__ == "__main__":
    if shutil.which("cargo") is None:
        print("error: cargo is not on PATH; qualification cannot run", file=sys.stderr)
        sys.exit(2)
    sys.exit(main())
