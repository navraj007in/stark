#!/usr/bin/env python3
"""WP-C6.4 — fixture-driven tests for the qualification harness and the comparison gate.

These two scripts are the only thing standing between "two CI jobs went green" and a Tier-1
qualification claim. Until now they were exercised by hand-built shell invocations, which is
exactly the kind of evidence C6.4 refuses everywhere else: it proves the happy path and nothing
about the refusals, and it leaves no artifact a reviewer can re-run.

Every test below builds a record in memory, mutates ONE thing, and asserts the comparator's
verdict. A test that mutates two things cannot say which one was detected.

Run: `python3 scripts/test_c64_scripts.py`
"""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


compare = _load("compare_c64", "compare-c64-evidence.py")
qualify = _load("qualify_c64", "run-c64-qualification.py")
target_matrix = _load("target_matrix_under_test", "target_matrix.py")


def command(name: str, **overrides) -> dict:
    base = {
        "name": name,
        "group": "semantics",
        "argv": ["cargo", "test", "--test", name],
        "exit_code": 0,
        "passed": 10,
        "failed": 0,
        "ignored": 0,
        "ignored_names": [],
        "skipped": 0,
        "duration_s": 1.0,
        "ok": True,
        "note": "",
        "unclassified_ignores": [],
    }
    base.update(overrides)
    return base


def record(host: str) -> dict:
    """A minimal but VALID record — one the comparator must accept when paired with its opposite.

    Built from the real required-step list, so a step added to the harness makes these fixtures
    fail rather than quietly testing a stale set.
    """
    steps = [s.name for s in qualify.steps_for(False) if s.required]
    entry = target_matrix.require(host)
    return {
        "schema_version": compare.SCHEMA,
        "work_package": "WP-C6.4",
        "commit_sha": "a" * 40,
        "dirty_worktree": False,
        "timestamp_utc": "2026-07-26T00:00:00+00:00",
        "runner_provider": "github-actions",
        "os_name": "Linux" if "linux" in host else "Darwin",
        "os_version": "1.0",
        "architecture": "x86_64" if "x86_64" in host else "arm64",
        "host_triple": host,
        "selected_target_triple": host,
        "target_tier": entry.tier,
        "target_pointer_width": entry.pointer_width,
        "rustc_version_verbose": "rustc 1.97.1 (abc 2026-07-14)\nhost: " + host,
        "cargo_version": "cargo 1.97.1",
        "python_version": "3.14.4",
        "layout_contract": entry.layout_contract,
        "layout_contract_version": 1,
        "compiler_layout_revision": 1,
        "profile": "debug",
        "quick_mode": False,
        "required_steps": steps,
        "commands": [command(name) for name in steps],
        "test_binaries": steps,
        "passed_count": 10 * len(steps),
        "failed_count": 0,
        "ignored_count": 0,
        "classified_ignores": {},
        "unclassified_ignores": [],
        "skipped_count": 0,
        "generated_corpus_version": None,
        "generated_corpus_case_count": 0,
        "generated_corpus_status": "BLOCKED-BY-C6.5",
        "determinism_first_hash": "key=1 source=2",
        "determinism_second_hash": "key=1 source=2",
        "determinism_result": "match",
        "deviations": [],
        "compiler_version": "0.1.0",
        "runtime_version": "0.1",
        "backend_version": "0.1",
        "mir_version": "0.1",
        "mir_runtime_surface": "0.1-A9",
        "overall_result": "PASS",
    }


MAC = "aarch64-apple-darwin"
LINUX = "x86_64-unknown-linux-gnu"


class ComparatorTests(unittest.TestCase):
    def run_compare(self, a: dict | None, b: dict | None) -> tuple[int, str]:
        """Run the comparator through its real entry point, on real files."""
        with tempfile.TemporaryDirectory(prefix="c64-compare-") as tmp:
            root = Path(tmp)
            paths = []
            for i, rec in enumerate((a, b)):
                path = root / f"record{i}.json"
                if rec is not None:
                    path.write_text(json.dumps(rec), encoding="utf-8")
                paths.append(path)
            out = root / "summary.md"
            stdout = sys.stdout
            sys.stdout = open(root / "stdout.txt", "w", encoding="utf-8")
            try:
                code = compare.main([str(paths[0]), str(paths[1]), "--out", str(out)])
            finally:
                sys.stdout.close()
                sys.stdout = stdout
            return code, out.read_text(encoding="utf-8")

    def assert_rejected(self, a, b, needle: str) -> None:
        code, summary = self.run_compare(a, b)
        self.assertEqual(code, 1, f"expected rejection; summary:\n{summary}")
        self.assertIn("TIER-1 DISAGREEMENT", summary)
        self.assertIn(needle, summary, f"summary did not explain the failure:\n{summary}")

    # --- the case that must PASS ------------------------------------------------------------

    def test_valid_tier1_agreement(self) -> None:
        code, summary = self.run_compare(record(MAC), record(LINUX))
        self.assertEqual(code, 0, summary)
        self.assertIn("TIER-1 AGREEMENT", summary)
        # A summary must always emit both artefact sections, so a reader never has to guess.
        self.assertIn("## Per-command results", summary)
        self.assertIn("## Required agreement", summary)

    # --- platform identity -----------------------------------------------------------------

    def test_same_platform_supplied_twice(self) -> None:
        self.assert_rejected(record(MAC), record(MAC), "both records are for the same target")

    def test_selected_target_differs_from_host(self) -> None:
        bad = record(LINUX)
        bad["selected_target_triple"] = MAC
        self.assert_rejected(record(MAC), bad, "differs from host")

    def test_a_non_tier1_host_is_refused(self) -> None:
        bad = record(LINUX)
        bad["host_triple"] = "x86_64-pc-windows-msvc"
        bad["selected_target_triple"] = "x86_64-pc-windows-msvc"
        self.assert_rejected(record(MAC), bad, "not tier-1")

    def test_an_unnamed_host_is_refused(self) -> None:
        bad = record(LINUX)
        bad["host_triple"] = "x86_64-unknown-linux-musl"
        bad["selected_target_triple"] = "x86_64-unknown-linux-musl"
        self.assert_rejected(record(MAC), bad, "is not a target STARK names")

    def test_wrong_pointer_width_is_refused(self) -> None:
        bad = record(LINUX)
        bad["target_pointer_width"] = 32
        self.assert_rejected(record(MAC), bad, "pointer width")

    # --- metadata --------------------------------------------------------------------------

    def test_blank_version_metadata(self) -> None:
        for field in ("mir_runtime_surface", "backend_version", "commit_sha", "layout_contract"):
            with self.subTest(field=field):
                bad = record(LINUX)
                bad[field] = ""
                self.assert_rejected(record(MAC), bad, f"`{field}` is blank")

    def test_absent_version_metadata(self) -> None:
        bad = record(LINUX)
        del bad["runtime_version"]
        self.assert_rejected(record(MAC), bad, "`runtime_version` is absent")

    def test_missing_layout_contract_version(self) -> None:
        bad = record(LINUX)
        bad["layout_contract_version"] = 0
        self.assert_rejected(record(MAC), bad, "layout_contract_version")

    def test_differing_commit_sha(self) -> None:
        bad = record(LINUX)
        bad["commit_sha"] = "b" * 40
        self.assert_rejected(record(MAC), bad, "commit_sha")

    # --- per-command agreement --------------------------------------------------------------

    def test_differing_ignored_identities_with_equal_counts(self) -> None:
        """The case a count-only comparison cannot see."""
        a, b = record(MAC), record(LINUX)
        step = a["required_steps"][0]
        for rec, name in ((a, "mod_one::case"), (b, "mod_two::case")):
            cmd = next(c for c in rec["commands"] if c["name"] == step)
            cmd["ignored"] = 1
            cmd["ignored_names"] = [name]
            rec["ignored_count"] = 1
            rec["classified_ignores"] = {name: "classified for the test"}
        self.assert_rejected(a, b, "ignored_names")

    def test_differing_skipped_counts(self) -> None:
        bad = record(LINUX)
        cmd = bad["commands"][0]
        cmd["skipped"] = 1
        bad["skipped_count"] = 1
        self.assert_rejected(record(MAC), bad, "self-skipped required test")

    def test_differing_passed_counts(self) -> None:
        bad = record(LINUX)
        bad["commands"][0]["passed"] = 9
        self.assert_rejected(record(MAC), bad, "passed")

    def test_differing_exit_codes(self) -> None:
        bad = record(LINUX)
        bad["commands"][0]["exit_code"] = 2
        self.assert_rejected(record(MAC), bad, "exit_code")

    def test_normative_argv_must_agree(self) -> None:
        bad = record(LINUX)
        bad["commands"][0]["argv"] = ["cargo", "test", "--test", "something-else"]
        self.assert_rejected(record(MAC), bad, "argv differs where it is normative")

    def test_interpreter_path_is_not_compared(self) -> None:
        """A Python step is launched through `sys.executable`, which legitimately differs."""
        a, b = record(MAC), record(LINUX)
        a["commands"][0]["argv"] = ["/opt/py/bin/python3", "scripts/test_build_release.py"]
        b["commands"][0]["argv"] = ["/usr/bin/python3", "scripts/test_build_release.py"]
        code, summary = self.run_compare(a, b)
        self.assertEqual(code, 0, summary)

    def test_unattributed_ignore_count(self) -> None:
        bad = record(LINUX)
        bad["commands"][0]["ignored"] = 2
        bad["commands"][0]["ignored_names"] = ["only_one"]
        self.assert_rejected(bad, record(MAC), "an unattributed ignore cannot have been classified")

    def test_missing_required_command(self) -> None:
        bad = record(LINUX)
        dropped = bad["required_steps"][2]
        bad["commands"] = [c for c in bad["commands"] if c["name"] != dropped]
        self.assert_rejected(record(MAC), bad, f"required command `{dropped}` is missing")

    def test_failed_required_command(self) -> None:
        bad = record(LINUX)
        bad["commands"][0]["ok"] = False
        bad["commands"][0]["exit_code"] = 101
        self.assert_rejected(record(MAC), bad, "did not pass")

    # --- run validity -----------------------------------------------------------------------

    def test_one_failed_qualification_record(self) -> None:
        bad = record(LINUX)
        bad["overall_result"] = "FAIL"
        bad["deviations"] = ["workspace: 3 test(s) failed"]
        self.assert_rejected(record(MAC), bad, "not PASS")

    def test_dirty_worktree(self) -> None:
        bad = record(LINUX)
        bad["dirty_worktree"] = True
        self.assert_rejected(record(MAC), bad, "dirty worktree")

    def test_quick_mode(self) -> None:
        bad = record(LINUX)
        bad["quick_mode"] = True
        self.assert_rejected(record(MAC), bad, "quick-mode run")

    def test_filtered_run_is_refused_through_its_deviation(self) -> None:
        bad = record(LINUX)
        bad["deviations"] = ["filtered run (--only fmt); not a qualification claim"]
        self.assert_rejected(record(MAC), bad, "filtered run")

    def test_unclassified_ignore(self) -> None:
        bad = record(LINUX)
        bad["unclassified_ignores"] = ["some_new_thing"]
        self.assert_rejected(record(MAC), bad, "unclassified ignored test")

    # --- corpus and determinism --------------------------------------------------------------

    def test_differing_generated_corpus_status(self) -> None:
        bad = record(LINUX)
        bad["generated_corpus_status"] = "COMPLETE"
        self.assert_rejected(record(MAC), bad, "generated_corpus_status")

    def test_nonzero_generated_corpus_case_count_while_blocked(self) -> None:
        bad = record(LINUX)
        bad["generated_corpus_case_count"] = 12
        self.assert_rejected(record(MAC), bad, "generated corpus case")

    def test_determinism_mismatch(self) -> None:
        bad = record(LINUX)
        bad["determinism_result"] = "mismatch"
        bad["determinism_second_hash"] = "key=9 source=9"
        self.assert_rejected(record(MAC), bad, "determinism")

    def test_determinism_hashes_absent(self) -> None:
        bad = record(LINUX)
        bad["determinism_first_hash"] = ""
        bad["determinism_second_hash"] = ""
        self.assert_rejected(record(MAC), bad, "determinism hashes are missing")

    # --- missing and malformed files ----------------------------------------------------------

    def test_missing_evidence_file(self) -> None:
        code, summary = self.run_compare(record(MAC), None)
        self.assertEqual(code, 1)
        self.assertIn("no such file", summary)
        self.assertIn("TIER-1 DISAGREEMENT", summary)
        # A summary must still be produced: a skipped comparison job is exactly what §10.4 forbids
        # substituting for an explicit disagreement report.
        self.assertIn("## Platform identities", summary)

    def test_both_records_missing_still_produces_a_summary(self) -> None:
        code, summary = self.run_compare(None, None)
        self.assertEqual(code, 1)
        self.assertIn("TIER-1 DISAGREEMENT", summary)

    def test_malformed_json(self) -> None:
        with tempfile.TemporaryDirectory(prefix="c64-compare-") as tmp:
            root = Path(tmp)
            good = root / "good.json"
            good.write_text(json.dumps(record(MAC)), encoding="utf-8")
            bad = root / "bad.json"
            bad.write_text("{ not json", encoding="utf-8")
            out = root / "summary.md"
            stdout = sys.stdout
            sys.stdout = open(root / "stdout.txt", "w", encoding="utf-8")
            try:
                code = compare.main([str(good), str(bad), "--out", str(out)])
            finally:
                sys.stdout.close()
                sys.stdout = stdout
            self.assertEqual(code, 1)
            self.assertIn("not valid JSON", out.read_text(encoding="utf-8"))

    def test_wrong_schema_version(self) -> None:
        bad = record(LINUX)
        bad["schema_version"] = "c6.4-evidence-0"
        self.assert_rejected(record(MAC), bad, "schema_version")


class HarnessTests(unittest.TestCase):
    """The parsing the harness does on `cargo` output, without running cargo."""

    def test_ignored_names_are_complete_libtest_names(self) -> None:
        output = (
            "test mod_a::case ... ignored, reason\n"
            "test mod_b::case ... ignored, reason\n"
            "test result: ok. 0 passed; 0 failed; 2 ignored; 0 measured; 0 filtered out\n"
        )
        names = qualify.IGNORED_RE.findall(output)
        # Both survive: truncating to the final component would collapse them to one.
        self.assertEqual(sorted(names), ["mod_a::case", "mod_b::case"])

    def test_known_ignores_are_classified_and_a_rogue_one_is_not(self) -> None:
        known = sorted(qualify.CLASSIFIED_IGNORES)
        self.assertTrue(known, "the classification list must not be empty")
        for name in known:
            self.assertTrue(qualify.CLASSIFIED_IGNORES[name].strip(), f"{name} has no reason")
        self.assertNotIn("some_new_thing", qualify.CLASSIFIED_IGNORES)

    def test_result_regex_reads_counts(self) -> None:
        line = "test result: ok. 14 passed; 1 failed; 2 ignored; 0 measured; 3 filtered out"
        self.assertEqual(qualify.RESULT_RE.findall(line), [("14", "1", "2", "0")])

    def test_self_skipping_suites_run_with_nocapture(self) -> None:
        """libtest hides a passing test's output, so a `SKIP:` line is invisible without it."""
        steps = {s.name: s for s in qualify.steps_for(False)}
        for name in ("c64_platform_matrix", "three_engine_differential", "c63_closure_evidence"):
            self.assertIn("--nocapture", steps[name].command(), name)

    def test_tier1_comes_from_the_target_matrix(self) -> None:
        self.assertEqual(sorted(qualify.TIER1), sorted(target_matrix.tier1_triples()))
        self.assertEqual(sorted(compare.TIER1), sorted(target_matrix.tier1_triples()))

    def test_required_steps_are_all_required(self) -> None:
        steps = qualify.steps_for(False)
        self.assertTrue(all(s.required for s in steps), "an optional step would weaken the claim")
        names = [s.name for s in steps]
        self.assertEqual(len(names), len(set(names)), "duplicate step names")

    def test_quick_mode_drops_the_broad_steps(self) -> None:
        quick = {s.name for s in qualify.steps_for(True)}
        self.assertNotIn("workspace", quick)
        self.assertNotIn("clippy", quick)


class TargetMatrixTests(unittest.TestCase):
    def test_exact_match_only(self) -> None:
        self.assertIsNotNone(target_matrix.classify(MAC))
        self.assertIsNone(target_matrix.classify("aarch64-apple-darwin-sim"))
        self.assertIsNone(target_matrix.classify("aarch64-apple"))

    def test_require_raises_on_unknown(self) -> None:
        with self.assertRaises(target_matrix.UnknownTarget):
            target_matrix.require("sparc64-windows-unknown")

    def test_every_named_target_is_64_bit(self) -> None:
        for entry in target_matrix.all_targets():
            self.assertEqual(entry.pointer_width, 64, entry.triple)
            self.assertEqual(entry.layout_contract, "stark-64-v1", entry.triple)

    def test_tier_of_unknown_is_unsupported(self) -> None:
        self.assertEqual(target_matrix.tier_of("mips-unknown-linux-gnu"), "unsupported")


if __name__ == "__main__":
    unittest.main(verbosity=2)
