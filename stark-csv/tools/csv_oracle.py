#!/usr/bin/env python3
"""Host-oracle differential test for stark-csv v0.1 (WP §22).

Generates a deterministic corpus of CSV record sets, computes each one's canonical
representation with Python's standard `csv` module under the frozen dialect, and cross-checks
stark-csv's parser/writer against it by generating and running one STARK program that performs
the comparison natively (STARK Core v1 has no stdin/file I/O, so this is the only way to feed
generated cases into a compiled/interpreted STARK program).

Required comparison, per case (WP §22 "Required comparison"):
  1. Python writes `records` -> `python_canonical` (this script, via `csv.writer`).
  2. STARK parses `python_canonical` -> must equal `records` (asserted inside the generated
     STARK program: this is parse() correctness).
  3. (same assertion as 2 -- "STARK data equals original" is what the equality check verifies.)
  4. STARK writes `records` -> must equal `python_canonical` BYTE FOR BYTE (asserted inside the
     generated STARK program: this is write() correctness, and simultaneously proves item 7
     directly, since the comparison target literally IS python_canonical).
  5/6. "Python reads STARK's output, data equals original" -- STARK's write() output is proven
     byte-identical to python_canonical by check 4 above, and this script independently verifies
     (in `self_check_case`, Python-side, before ever generating STARK source for a case) that
     `csv.reader(python_canonical, strict=True, ...)` reads back to exactly `records`. By
     transitivity, reading STARK's output with Python's reader also yields `records` -- checking
     it again by literally re-invoking csv.reader on a string already proven identical would not
     exercise any code path the two checks above have not already exercised, so this script
     performs the equality proof once via item 4's string comparison rather than round-tripping
     large multi-line CSV blobs back through a subprocess's stdout (which would need its own
     escaping layer to survive embedded CR/LF, adding failure surface without adding coverage).
  7. Proven directly by check 4 (STARK's write output IS compared against python_canonical).

KNOWN ORACLE DIVERGENCES (found by running this script against the real implementation), both
involving a record consisting of exactly one empty-string field:

1. Quoting (`has_lone_empty_field_record`, any position). Python's `csv.writer` quotes such a
   record (writing `""` rather than nothing), unconditionally, regardless of where it falls in
   the document, specifically to disambiguate it on the wire from a zero-field record. WP §17.2
   explicitly forbids this for stark-csv: "Do not quote only because it: is empty" -- no
   exception for that shape. stark-csv's writer follows the WP's explicit rule (verified
   directly by `test_write_zero_field_record_normalization` and `test_write_unquoted_empty` in
   `src/lib.stark`). For a case with this shape but where the lone-empty-field record is NOT
   last, the generated STARK program checks a round trip (`parse(write(records)) == records`)
   instead of byte equality against python_canonical, and reports
   `EXPLAINED <idx> lone-empty-field-quoting`.
2. Representational limit (`is_trailing_lone_empty_field_record`, last record only). When the
   lone-empty-field record is the LAST record, stark-csv's own write->parse round trip is
   genuinely lossy, independent of Python entirely: writing it contributes zero bytes, and
   "nothing" after the previous record's separator is indistinguishable from "no further
   record" (WP §12: `"" -> []`; "a trailing separator terminates the existing record but does
   not add another"). This is WP §17.5's own documented, accepted exception ("A zero-field
   record cannot be represented distinctly in CSV from a one-empty-field record"), not a defect.
   For this shape, the generated STARK program does not attempt a round trip at all -- it is
   known by construction not to hold -- and reports
   `EXPLAINED <idx> trailing-lone-empty-field-representational-limit`.

Confirmed to be the sole sources of divergence: every one of the 110 mismatches in the first full
1000-case run had a lone-empty-field record; checking all 1000 generated cases programmatically
found exactly 55 with that record last (limitation 2) and exactly 55 with it elsewhere
(divergence 1), summing to the original 110 with zero unexplained residue.

Usage:
    python3 tools/csv_oracle.py
"""

from __future__ import annotations

import csv
import io
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

SEED = 20260724
CASE_COUNT_TARGET = 1000

DIALECT = {
    "delimiter": ",",
    "quotechar": '"',
    "doublequote": True,
    "skipinitialspace": False,
    "lineterminator": "\r\n",
    "quoting": csv.QUOTE_MINIMAL,
}

BOUNDARY_LENGTHS = [0, 1, 2, 15, 16, 17, 255, 256, 257, 1023, 1024, 1025]
DIMENSION_COUNTS = [0, 1, 2, 3, 10, 100]
FIELD_COUNTS = [1, 2, 3, 10]

STRUCTURAL_POOL = [",", '"', "\r", "\n", "\r\n"]
UNICODE_POOL = [
    "ਪੰਜਾਬੀ",
    "हिन्दी",
    "中文字",
    "😀🎉",
    "é",  # combining mark
    "\U0001F600",  # supplementary plane
    "\x00",  # NUL
    " \t ",
]


def python_canonical(records: list[list[str]]) -> str:
    buf = io.StringIO(newline="")
    writer = csv.writer(buf, **DIALECT)
    writer.writerows(records)
    text = buf.getvalue()
    # csv.writerows always terminates the LAST row too; stark-csv's writer never emits a
    # trailing separator (WP §17.1), so strip exactly one trailing lineterminator to match.
    if records and text.endswith(DIALECT["lineterminator"]):
        text = text[: -len(DIALECT["lineterminator"])]
    return text


def self_check_case(records: list[list[str]], canonical: str) -> bool:
    """Python-side sanity check: does canonical actually read back to records under the frozen
    reader dialect? A failure here means the *test generator* built an inconsistent case, not
    that stark-csv is broken -- such a case is skipped and reported separately."""
    reader_dialect = dict(DIALECT)
    reader_dialect.pop("quoting")
    buf = io.StringIO(canonical, newline="")
    reader = csv.reader(buf, strict=True, **reader_dialect)
    read_back = [list(row) for row in reader]
    return read_back == records


def make_field(rng: random.Random, length: int) -> str:
    if length == 0:
        return ""
    pool = ["x", "y", "z", "0", "1", " "]
    chars = []
    remaining = length
    while remaining > 0:
        if rng.random() < 0.15 and remaining >= 1:
            piece = rng.choice(UNICODE_POOL + STRUCTURAL_POOL)
        else:
            piece = rng.choice(pool)
        chars.append(piece)
        remaining -= len(piece.encode("utf-8"))
    return "".join(chars)


def generate_cases(seed: int, target: int) -> list[list[list[str]]]:
    rng = random.Random(seed)
    cases: list[list[list[str]]] = []

    # Deterministic structural/boundary coverage (WP §21/§22 required categories).
    for length in BOUNDARY_LENGTHS:
        cases.append([["x" * 0 if length == 0 else "x" * length, "sentinel"]])
    for count in DIMENSION_COUNTS:
        cases.append([["r" + str(i), "v" + str(i)] for i in range(count)])
    for count in FIELD_COUNTS:
        cases.append([["f" + str(i) for i in range(count)]])
    for s in STRUCTURAL_POOL:
        cases.append([["a" + s + "b", "sentinel"]])
    for u in UNICODE_POOL:
        cases.append([[u, "sentinel"]])
    cases.append([["", ""], ["a", "b", "c"], [""], ["x" * 300]])

    # Deterministic-seed random fuzz cases to reach the target count.
    while len(cases) < target:
        n_records = rng.choice([1, 1, 1, 2, 3, 5])
        records = []
        for _ in range(n_records):
            n_fields = rng.choice([1, 1, 2, 3, 5])
            row = []
            for _ in range(n_fields):
                length = rng.choice([0, 1, 2, 5, 16, 64, rng.randint(0, 300)])
                row.append(make_field(rng, length))
            records.append(row)
        cases.append(records)

    return cases[:target] if len(cases) > target else cases


def escape_stark_string(s: str) -> str:
    out = []
    for ch in s:
        if ch == "\\":
            out.append("\\\\")
        elif ch == '"':
            out.append('\\"')
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\r":
            out.append("\\r")
        elif ch == "\t":
            out.append("\\t")
        elif ch == "\x00":
            out.append("\\0")
        elif 0x20 <= ord(ch) < 0x7F:
            out.append(ch)
        else:
            out.append("\\u{%x}" % ord(ch))
    return "".join(out)


def stark_row_literal(row: list[str], var: str) -> list[str]:
    lines = [f"    let mut {var}: Vec<String> = Vec::new();"]
    for field in row:
        lines.append(f'    {var}.push(String::from("{escape_stark_string(field)}"));')
    return lines


def stark_records_literal(records: list[list[str]], var: str) -> list[str]:
    lines = [f"    let mut {var}: Vec<Vec<String>> = Vec::new();"]
    for i, row in enumerate(records):
        row_var = f"{var}_r{i}"
        lines.extend(stark_row_literal(row, row_var))
        lines.append(f"    {var}.push({row_var});")
    return lines


def rows_eq_helper() -> str:
    return """
fn oracle_rows_eq(a: &Vec<Vec<String>>, b: &Vec<Vec<String>>) -> Bool {
    if a.len() != b.len() {
        return false;
    }
    let mut i: UInt64 = 0u64;
    while i < a.len() {
        let ra = &a[i];
        let rb = &b[i];
        if ra.len() != rb.len() {
            return false;
        }
        let mut j: UInt64 = 0u64;
        while j < ra.len() {
            if ra[j].as_str() != rb[j].as_str() {
                return false;
            }
            j = j + 1u64;
        }
        i = i + 1u64;
    }
    true
}
"""


def has_lone_empty_field_record(records: list[list[str]]) -> bool:
    """A record with exactly one field, and that field empty, is where Python's csv.writer and
    the frozen dialect (WP §17.2: never quote solely for emptiness) are known to diverge --
    Python quotes it (writing `""`) *unconditionally, regardless of position*, to disambiguate a
    one-empty-field record from a zero-field record; the frozen dialect never does. See the
    module docstring's "known oracle divergence" note."""
    return any(len(r) == 1 and r[0] == "" for r in records)


def is_trailing_lone_empty_field_record(records: list[list[str]]) -> bool:
    """Stricter than `has_lone_empty_field_record`: true only when the *last* record is a lone
    empty field. Only in that position is stark-csv's own write->parse round trip actually lossy
    -- writing it contributes zero bytes, and a trailing "nothing" after the previous record's
    separator is indistinguishable from "no further record" (WP §12's own "" -> [] and "a
    trailing separator terminates the existing record but does not add another" rules), exactly
    the exception WP §17.5 documents and accepts ("A zero-field record cannot be represented
    distinctly in CSV from a one-empty-field record"). A lone empty field that is NOT last
    round-trips fine, because the following record's separator disambiguates it -- confirmed by
    running both shapes through this oracle."""
    return bool(records) and len(records[-1]) == 1 and records[-1][0] == ""


def generate_stark_program(cases: list[tuple[int, list[list[str]], str]]) -> str:
    parts = [
        "use stark_csv::parse;",
        "use stark_csv::write;",
        "",
        rows_eq_helper(),
    ]
    case_fn_names = []
    for idx, records, canonical in cases:
        fn_name = f"oracle_case_{idx}"
        case_fn_names.append(fn_name)
        trailing_limit = is_trailing_lone_empty_field_record(records)
        quoting_divergence = has_lone_empty_field_record(records) and not trailing_limit
        lines = [f"fn {fn_name}() -> Bool {{"]
        lines.extend(stark_records_literal(records, "expected"))
        escaped_canonical = escape_stark_string(canonical)
        lines.append(f'    let canonical = "{escaped_canonical}";')
        lines.append("    match parse(canonical) {")
        lines.append("        Ok(rows) => {")
        lines.append("            if !oracle_rows_eq(&rows, &expected) {")
        lines.append(f'                println("FAIL {idx} parse-mismatch");')
        lines.append("                return false;")
        lines.append("            }")
        lines.append("        }")
        lines.append("        Err(_) => {")
        lines.append(f'            println("FAIL {idx} parse-errored");')
        lines.append("            return false;")
        lines.append("        }")
        lines.append("    }")
        lines.append("    match write(&expected) {")
        if trailing_limit:
            # WP §17.5's own documented, accepted exception: a trailing lone-empty-field record
            # cannot round-trip at all (writing it is indistinguishable from omitting it). Not
            # compared against canonical, and no round trip is attempted -- both are known to
            # differ/lose information by the format's own construction, not by a stark-csv defect.
            lines.append("        Ok(_) => {")
            lines.append(f'            println("EXPLAINED {idx} trailing-lone-empty-field-representational-limit");')
            lines.append("        }")
        elif quoting_divergence:
            # WP §17.2 forbids quoting solely for emptiness; Python's csv.writer quotes a lone
            # empty field anyway (to disambiguate it from a zero-field record) -- a real, known
            # divergence between the frozen dialect and Python's QUOTE_MINIMAL. Don't compare
            # against `canonical` byte-for-byte here; verify stark-csv's own round trip instead,
            # which is the check that actually matters for this shape (and, unlike the trailing
            # case above, is expected to hold, since a following record's separator disambiguates
            # a non-trailing lone empty field).
            lines.append("        Ok(s) => {")
            lines.append("            match parse(s.as_str()) {")
            lines.append("                Ok(back) => {")
            lines.append("                    if !oracle_rows_eq(&back, &expected) {")
            lines.append(f'                        println("FAIL {idx} write-round-trip-mismatch");')
            lines.append("                        return false;")
            lines.append("                    }")
            lines.append(f'                    println("EXPLAINED {idx} lone-empty-field-quoting");')
            lines.append("                }")
            lines.append("                Err(_) => {")
            lines.append(f'                    println("FAIL {idx} write-round-trip-parse-errored");')
            lines.append("                    return false;")
            lines.append("                }")
            lines.append("            }")
            lines.append("        }")
        else:
            lines.append("        Ok(s) => {")
            lines.append("            if s.as_str() != canonical {")
            lines.append(f'                println("FAIL {idx} write-mismatch");')
            lines.append("                return false;")
            lines.append("            }")
            lines.append("        }")
        lines.append("        Err(_) => {")
        lines.append(f'            println("FAIL {idx} write-errored");')
        lines.append("            return false;")
        lines.append("        }")
        lines.append("    }")
        lines.append("    true")
        lines.append("}")
        parts.append("\n".join(lines))

    main_lines = ["fn main() {", "    let mut passed: UInt64 = 0u64;", "    let mut failed: UInt64 = 0u64;"]
    for fn_name in case_fn_names:
        main_lines.append(f"    if {fn_name}() {{")
        main_lines.append("        passed = passed + 1u64;")
        main_lines.append("    } else {")
        main_lines.append("        failed = failed + 1u64;")
        main_lines.append("    }")
    main_lines.append('    println("ORACLE_SUMMARY");')
    main_lines.append("    println(passed);")
    main_lines.append("    println(failed);")
    main_lines.append("}")
    parts.append("\n".join(main_lines))
    return "\n\n".join(parts)


def main() -> int:
    print(f"Python version: {sys.version}")
    print(f"Seed: {SEED}")

    cases = generate_cases(SEED, CASE_COUNT_TARGET)
    print(f"Generated case count: {len(cases)}")

    max_records = max(len(c) for c in cases)
    max_fields = max((len(r) for c in cases for r in c), default=0)
    max_field_bytes = max(
        (len(f.encode("utf-8")) for c in cases for r in c for f in r), default=0
    )
    print(f"Max dimensions: {max_records} records, {max_fields} fields in a record")
    print(f"Max field bytes: {max_field_bytes}")

    verified_cases = []
    generator_bugs = []
    for idx, records in enumerate(cases):
        canonical = python_canonical(records)
        if not self_check_case(records, canonical):
            generator_bugs.append(idx)
            continue
        verified_cases.append((idx, records, canonical))

    if generator_bugs:
        print(f"WARNING: {len(generator_bugs)} generated cases failed self-check and were "
              f"skipped (generator bug, not a stark-csv result): {generator_bugs[:10]}")

    print(f"Cases sent to STARK: {len(verified_cases)}")

    program = generate_stark_program(verified_cases)

    repo_root = Path(__file__).resolve().parents[2]
    starkc_manifest = repo_root / "starkc" / "Cargo.toml"
    stark_csv_root = repo_root / "stark-csv"

    # The compiler confines a path dependency to `parent(consumer_manifest_dir)`
    # (starkc/src/package.rs `get_workspace_root`) -- a fixed one-directory-up rule, not a
    # nearest-common-ancestor search. To depend on the real stark-csv/starkpkg.json, this scratch
    # runner package must therefore live exactly one directory level under stark-csv/, not under
    # the system temp directory (which has no ancestor relationship to stark-csv/ at all).
    pkg_dir = stark_csv_root / ".oracle_runner_scratch"
    if pkg_dir.exists():
        shutil.rmtree(pkg_dir)
    try:
        (pkg_dir / "src").mkdir(parents=True)
        (pkg_dir / "starkpkg.json").write_text(
            '{\n'
            '  "name": "stark-csv-oracle-runner",\n'
            '  "version": "0.1.0",\n'
            '  "entry": "src/main.stark",\n'
            '  "dependencies": {\n'
            '    "stark_csv": { "package": "stark-csv", "path": ".." }\n'
            '  }\n'
            '}\n',
            encoding="utf-8",
        )
        (pkg_dir / "src" / "main.stark").write_text(program, encoding="utf-8")

        t0 = time.time()
        result = subprocess.run(
            ["cargo", "run", "--quiet", "--manifest-path", str(starkc_manifest), "--bin", "stark", "--", "run"],
            cwd=str(pkg_dir),
            capture_output=True,
            text=True,
        )
        elapsed = time.time() - t0
    finally:
        shutil.rmtree(pkg_dir, ignore_errors=True)

    stdout = result.stdout
    stderr = result.stderr
    print(f"STARK run exit code: {result.returncode}")
    print(f"STARK run wall time: {elapsed:.1f}s")

    fail_lines = [line for line in stdout.splitlines() if line.startswith("FAIL ")]
    explained_lines = [line for line in stdout.splitlines() if line.startswith("EXPLAINED ")]
    summary_idx = stdout.find("ORACLE_SUMMARY")
    passed = failed = None
    if summary_idx != -1:
        after = stdout[summary_idx:].splitlines()
        if len(after) >= 3:
            passed = after[1].strip()
            failed = after[2].strip()

    print(f"STARK-reported passed: {passed}")
    print(f"STARK-reported failed: {failed}")
    print(f"FAIL lines: {len(fail_lines)}")
    print(f"EXPLAINED lines (known lone-empty-field quoting divergence, see docstring): "
          f"{len(explained_lines)}")
    if fail_lines:
        print("First failure:", fail_lines[0])
        for line in fail_lines[:20]:
            print("  ", line)

    if result.returncode != 0 and not fail_lines:
        print("STARK process exited nonzero with no FAIL lines -- unexpected crash:")
        print("---- stdout ----")
        print(stdout[-4000:])
        print("---- stderr ----")
        print(stderr[-4000:])

    overall_pass = (
        result.returncode == 0
        and len(fail_lines) == 0
        and passed == str(len(verified_cases))
        and failed == "0"
    )
    print(f"Overall result: {'PASS' if overall_pass else 'FAIL'}")
    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
