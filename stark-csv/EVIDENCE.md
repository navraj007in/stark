# stark-csv v0.1 evidence

## Repository identities

```text
Implementation baseline: 8de4d41159083349133dc0696784a725a99aaacb
Implementation commit:   UNCOMMITTED
Review head:              8de4d41159083349133dc0696784a725a99aaacb
```

## Exact file list

`git diff --name-only` is not yet meaningful (nothing committed). The package is entirely new;
every file under `stark-csv/` is new except `tools/__pycache__/` (removed, gitignored) and
`*/target/` build directories (gitignored, never created except transiently by `stark build`).

```text
stark-csv/BLOCKERS.md
stark-csv/EVIDENCE.md
stark-csv/README.md
stark-csv/docs/CsvError/index.html
stark-csv/docs/CsvLimits/index.html
stark-csv/docs/default_limits/index.html
stark-csv/docs/index.html
stark-csv/docs/limits/index.html
stark-csv/docs/parse/index.html
stark-csv/docs/parse_with_limits/index.html
stark-csv/docs/search.html
stark-csv/docs/search.json
stark-csv/docs/style.css
stark-csv/docs/write/index.html
stark-csv/docs/write_with_limits/index.html
stark-csv/fixtures/src/main.stark
stark-csv/fixtures/stark.lock
stark-csv/fixtures/starkpkg.json
stark-csv/src/lib.stark
stark-csv/src/tests.stark
stark-csv/stark.lock
stark-csv/starkpkg.json
stark-csv/tools/csv_oracle.py
```

No file outside `stark-csv/` was modified. `stark-csv/.oracle_runner_scratch/` is created and
deleted by `tools/csv_oracle.py` at run time (see BLOCKERS.md's "Secondary finding") and is not a
committed artifact.

## Readiness probes

| Probe | Result | Evidence |
|---|---|---|
| A: nested `Vec<Vec<String>>` return | PASS (interpreter) | `stark check`/`stark test` on the probe scaffold; superseded by `parse`'s real return type, exercised by all 107 package tests |
| B: cross-file free function | PASS (interpreter) | package's own `lib.stark` (impl) / `tests.stark` (107 tests, all calling `parse`/`write`/etc. from a different file) |
| C: cross-package free function | PASS (interpreter AND native) | `fixtures/` — `stark check`/`stark run` exit 0; `stark build` also succeeds and the produced binary exits 0 for the *trivial* `answer(): UInt64` case (see "Native evidence" below for where nesting breaks) |
| D: nested writer argument `&Vec<Vec<String>>` | PASS (interpreter); BLOCKED (native) | interpreter: same as A; native: `stark build` fails, see BLOCKERS.md |

Classification: **READY_INTERPRETER** (WP §5.1's own definition: "pure, cross-file and
cross-package use work but real native execution is blocked by C6.3/package tooling" — exactly
what was observed).

## Implemented

- `CsvError` (10 variants, exact match to WP §8), `CsvLimits` (6 private fields), `default_limits`,
  `limits`, `parse`, `parse_with_limits`, `write`, `write_with_limits` — all free functions.
- Four-state parser (`StartField`/`InUnquoted`/`InQuoted`/`AfterClosingQuote`, WP §15) with
  overflow-guarded field/limit accounting (§16) and exact error payloads (§14).
- Deterministic writer: CRLF separators, minimal quoting, quote doubling, overflow-guarded output
  limits, zero-field-record normalization (§17).
- 107 required tests (§19/§20), all passing.
- External path-dependency consumer (`fixtures/`).
- Python host-oracle differential (`tools/csv_oracle.py`), 1,000 deterministic cases.

## Public API audit

- [x] exact: `CsvError` (10 variants), `CsvLimits` (private fields), `default_limits`, `limits`,
      `parse`, `parse_with_limits`, `write`, `write_with_limits` — no extra public items, no
      associated functions, no `CsvParser`.
- [x] no extra public items (`grep -c "^pub "` on `src/lib.stark` = 8, matching exactly the 8
      frozen API items: 1 enum + 1 struct + 6 functions).
- [x] tests call API cross-file: every one of the 107 tests in `src/tests.stark` calls `parse`,
      `parse_with_limits`, `write`, `write_with_limits`, `default_limits`, or `limits`, imported
      via `use super::X;` from `src/lib.stark`.
- [x] consumer calls API cross-package: `fixtures/src/main.stark` imports `stark_csv::{parse,
      write, CsvError}` and uses no private internals (no field access on `CsvLimits`, no
      internal helper functions).

## Requirement ledger

### §19.1 Basic and separators

| Requirement | Test function(s) | Result |
|---|---|---|
| empty input | `test_parse_empty_input` | PASS |
| one field | `test_parse_one_field` | PASS |
| two/three fields | `test_parse_two_fields`, `test_parse_three_fields` | PASS |
| two records | `test_parse_two_records` | PASS |
| LF | `test_parse_lf_separator` | PASS |
| CRLF | `test_parse_crlf_separator` | PASS |
| mixed LF/CRLF | `test_parse_mixed_lf_crlf` | PASS |
| trailing LF/CRLF | `test_parse_trailing_lf`, `test_parse_trailing_crlf` | PASS |

### §19.2 Empty cases (exact structures pinned)

`test_parse_empty_string_is_zero_records`, `test_parse_bare_lf_is_one_blank_record`,
`test_parse_bare_crlf_is_one_blank_record`, `test_parse_two_lf_is_two_blank_records`,
`test_parse_two_crlf_is_two_blank_records`, `test_parse_bare_comma_is_two_empty_fields`,
`test_parse_two_commas_is_three_empty_fields`, `test_parse_a_comma_is_a_then_empty`,
`test_parse_comma_a_is_empty_then_a`, `test_parse_a_comma_comma_b`, `test_parse_a_lf_lf` — all 11
of §12's table rows pinned. PASS.

### §19.3 Quoted fields

`test_parse_quoted_ordinary_field`, `test_parse_quoted_empty_field`, `test_parse_quoted_comma`,
`test_parse_quoted_lf`, `test_parse_quoted_cr`, `test_parse_quoted_crlf`,
`test_parse_quoted_leading_trailing_spaces`, `test_parse_quoted_tab`, `test_parse_quoted_nul`,
`test_parse_one_escaped_quote`, `test_parse_several_escaped_quotes`,
`test_parse_escaped_quote_beside_comma`, `test_parse_escaped_quote_beside_newline` — all 13
required cases. PASS.

### §19.4 Unicode preservation

`test_parse_unicode_punjabi`, `test_parse_unicode_hindi`, `test_parse_unicode_cjk`,
`test_parse_unicode_emoji`, `test_parse_unicode_combining_mark`,
`test_parse_unicode_supplementary_plane`, `test_parse_unicode_mixed_ascii`,
`test_parse_unicode_beside_comma_in_quotes`, `test_parse_unicode_beside_escaped_quote` — all 9
required cases. PASS.

### §19.5 Unequal widths

`test_parse_one_field_then_three`, `test_parse_three_fields_then_one`,
`test_parse_blank_record_between_nonblank`. PASS.

### §19.6 Quote errors (exact variant/offset/record/field pinned)

`test_parse_quote_in_unquoted_field` (`a"b`), `test_parse_quote_in_unquoted_field_leading_space`
(` "a"`), `test_parse_quote_in_unquoted_field_second_field` (`a,b"c`). PASS.

### §19.7 Post-quote errors (exact byte payload pinned)

`test_parse_unexpected_after_closing_quote_letter` (`"a"b`),
`test_parse_unexpected_after_closing_quote_space` (`"a" `),
`test_parse_unexpected_after_closing_quote_tab` (`"a"\t`),
`test_parse_unexpected_after_closing_quote_after_escaped_quote` (`"a""b"x`). PASS.

### §19.8 CR errors (exact CR offset pinned)

`test_parse_bare_cr_mid_field` (`a\rb`), `test_parse_bare_cr_at_eof` (`a\r`),
`test_parse_bare_cr_between_commas` (`,\r,`), `test_parse_cr_inside_quotes_succeeds`. PASS.

### §19.9 Unterminated fields (exact opening-quote offset pinned)

`test_parse_unterminated_just_quote` (`"`), `test_parse_unterminated_quote_a` (`"a`),
`test_parse_unterminated_second_field` (`a,"b`),
`test_parse_unterminated_after_escaped_quote` (`"a""`). PASS.

### §19.10 Limits (boundary succeeds, first excess returns exact payload)

`test_limit_input_bytes_boundary_and_excess`, `test_limit_records_boundary_and_excess`,
`test_limit_fields_per_record_boundary_and_excess`, `test_limit_total_fields_boundary_and_excess`,
`test_limit_field_bytes_ascii_boundary_and_excess`,
`test_limit_field_bytes_multibyte_boundary_and_excess`,
`test_limit_output_bytes_boundary_and_excess` — all 7 required limit categories. PASS.

### §19.11 State reset

`test_parse_failed_then_successful`, `test_parse_repeated_successful`. PASS.

### §20 Writer — basic

`test_write_zero_records`, `test_write_one_field`, `test_write_multiple_fields`,
`test_write_multiple_records`, `test_write_unequal_widths`, `test_write_empty_fields`,
`test_write_zero_field_record_normalization`. PASS.

### §20 Writer — minimal quoting

`test_write_unquoted_ordinary_ascii`, `test_write_unquoted_empty`, `test_write_unquoted_spaces`,
`test_write_unquoted_tab`, `test_write_unquoted_unicode`, `test_write_unquoted_nul`,
`test_write_quoted_comma`, `test_write_quoted_quote`, `test_write_quoted_cr`,
`test_write_quoted_lf`, `test_write_quoted_crlf`. PASS.

### §20 Quote doubling

`test_write_quote_doubling_a_quote_b`, `test_write_quote_doubling_single_quote`,
`test_write_quote_doubling_two_quotes`, `test_write_quote_doubling_two_quoted_fields`. PASS.

### §20 CRLF

`test_write_always_crlf_no_trailing`. PASS.

### §20 Limits

`test_write_limit_field_content_excess`, `test_write_limit_comma_excess`,
`test_write_limit_crlf_excess`, `test_write_limit_quote_excess`,
`test_write_limit_quote_doubling_excess`, `test_write_limit_multibyte_utf8_excess` — all 6
required excess categories. PASS.

### §20 Input immutability

`test_write_input_unchanged_after_success`, `test_write_input_unchanged_after_failure`. PASS.

### §18 Canonicalization

`test_canonicalization_write_then_parse_round_trips`, `test_canonicalization_examples` (all 4
worked examples from §18). PASS.

### §21 Round-trip corpus

`test_round_trip_boundary_field_lengths` (12 boundary lengths: 0,1,2,15,16,17,255,256,257,1023,
1024,1025), `test_round_trip_record_counts` (0,1,2,3,10,100), `test_round_trip_field_counts`
(1,2,3,10), `test_round_trip_structural_and_unicode_and_nul`. PASS.

**Total: 107 test functions, 107 passing, 0 failing.**

## Command evidence

| Command | Working directory | Exit | Test count | Tool version | Target | Engine |
|---|---|---|---|---|---|---|
| `cargo run --manifest-path ../starkc/Cargo.toml --bin stark -- check` | `stark-csv/` | 0 | — | starkc 0.1.0 @ 8de4d41 | host (aarch64-apple-darwin) | front end only |
| `cargo run --manifest-path ../starkc/Cargo.toml --bin stark -- test` | `stark-csv/` | 0 | 107/107 | starkc 0.1.0 @ 8de4d41 | host | HIR interpreter |
| `cargo run --manifest-path ../starkc/Cargo.toml --bin stark -- fmt --check` | `stark-csv/` | 0 | — | starkc 0.1.0 @ 8de4d41 | — | — |
| `cargo run --manifest-path ../starkc/Cargo.toml --bin stark -- doc` | `stark-csv/` | 0 | 18 items | starkc 0.1.0 @ 8de4d41 | — | — |
| `cargo run --manifest-path ../starkc/Cargo.toml --bin stark -- build` | `stark-csv/` | 1 | — | starkc 0.1.0 @ 8de4d41 | host | N/A — no `main`, expected: `error: native build does not yet support this program: program without a main function` |
| `cargo run --manifest-path ../starkc/Cargo.toml --bin stark -- check` | `stark-csv/fixtures/` | 0 | — | starkc 0.1.0 @ 8de4d41 | host | front end only |
| `cargo run --manifest-path ../starkc/Cargo.toml --bin stark -- run` | `stark-csv/fixtures/` | 0 | — | starkc 0.1.0 @ 8de4d41 | host | HIR interpreter |
| `cargo run --manifest-path ../starkc/Cargo.toml --bin stark -- build` | `stark-csv/fixtures/` | 1 | — | starkc 0.1.0 @ 8de4d41 | host | native — FAILS, see "Native evidence" |
| `python3 tools/csv_oracle.py` | `stark-csv/tools/` | 0 | 1000/1000 | Python 3.14.4 | host | HIR interpreter (generated STARK program) |

## Native evidence

`stark build` on the real consumer (`fixtures/`, which calls `parse`/`write` — this package's
actual `Vec<Vec<String>>` data model) fails:

```text
error: native build does not yet support this program: method bytes on Str (a later C4.5e sub-slice)
```

An isolated reproducer using only nested-`Vec` types (no `str::bytes()`) fails earlier, at
linkage:

```text
error: native build does not yet support this program: linkage: body
`stark_csv::probe_count_rows@[]` param 0 is not C5-representable: MirTy Core(Vec,
[Core(Vec, [String])]) has no C5.3a generated-Rust representation yet
```

Both are native generated-Rust backend limitations (`starkc/src/backend/generated_rust`), out of
this package's authorized scope to fix (§6: "Prohibited: `starkc/**`"). See BLOCKERS.md for the
full record.

## Execution engines

| Evidence | HIR | MIR | native |
|---|---|---|---|
| package tests (107) | PASS | not run — `stark test` drives the HIR interpreter only; no MIR-level test harness is part of this package's authorized scope | BLOCKED (see above) |
| consumer (`fixtures/`) | PASS (`stark run`) | not run | BLOCKED (see above) |
| oracle (1000 cases) | PASS (1000/1000, see below) | not run | BLOCKED (transitively — the oracle runner itself calls `parse`/`write`, same blocker) |

## Host-oracle differential

- Python version: 3.14.4 (main, Apr 7 2026, 13:13:20) [Clang 21.0.0]
- Seed: 20260724
- Cases: 1000 (generated: boundary lengths, dimension counts, field counts, each structural byte,
  each Unicode pool entry, plus deterministic-seed random fuzz to reach 1000)
- Command: `python3 tools/csv_oracle.py` (run from `stark-csv/tools/`)
- Max dimensions observed: 100 records, 10 fields in a record
- Max field bytes observed: 1025
- Result: **PASS — 1000/1000, 0 unexplained failures**

### Investigation: the first run found 110 real mismatches, not a stark-csv bug

The first run (before any classification logic existed) reported 890 exact byte-for-byte matches
against Python's canonical output and 110 `write-mismatch` failures. Every one of the 110 failing
cases turned out to contain a record consisting of exactly one empty-string field — confirmed by
checking all 1000 generated cases programmatically, with zero unexplained residue. Splitting that
group further by position (checked programmatically against all 1000 cases) found it decomposes
into exactly two sub-shapes, 55 cases each, both root-caused and neither a `stark-csv` defect:

1. **Quoting divergence (any position, 55 cases).** Python's `csv.writer` quotes a lone empty
   field (writing `""`) unconditionally, regardless of position, to disambiguate it from a
   zero-field record. WP §17.2 explicitly forbids this for `stark-csv`: "Do not quote only
   because it: is empty" — no exception for this shape. `stark-csv`'s writer correctly follows
   the WP's rule (independently verified by `test_write_zero_field_record_normalization` and
   `test_write_unquoted_empty`, both passing) — verified instead by round trip
   (`parse(write(records)) == records`), which holds.
2. **Representational limit (last record only, 55 cases).** When the lone-empty-field record is
   specifically the *last* record, `stark-csv`'s own write→parse round trip is genuinely lossy,
   independent of Python entirely: writing it contributes zero bytes, and "nothing" after the
   previous record's separator is indistinguishable from "no further record" (WP §12: `"" ->
   []`, "a trailing separator terminates the existing record but does not add another"). This is
   WP §17.5's own documented, accepted exception ("A zero-field record cannot be represented
   distinctly in CSV from a one-empty-field record") — not attempted as a round trip at all,
   since it is known by construction not to hold.

`tools/csv_oracle.py` (`has_lone_empty_field_record`, `is_trailing_lone_empty_field_record`) was
updated to detect and classify both shapes per case before generating STARK source, printing
`EXPLAINED <idx> lone-empty-field-quoting` or
`EXPLAINED <idx> trailing-lone-empty-field-representational-limit` instead of `FAIL`. See the
script's module docstring for the full analysis. The corrected, final run's full output:

```text
Python version: 3.14.4 (main, Apr  7 2026, 13:13:20) [Clang 21.0.0 (clang-2100.0.123.102)]
Seed: 20260724
Generated case count: 1000
Max dimensions: 100 records, 10 fields in a record
Max field bytes: 1025
Cases sent to STARK: 1000
STARK run exit code: 0
STARK run wall time: 304.8s
STARK-reported passed: 1000
STARK-reported failed: 0
FAIL lines: 0
EXPLAINED lines (known lone-empty-field quoting divergence, see docstring): 110
Overall result: PASS
```

## Known limitations

1. **Native execution is unavailable** (`WAITING_C6.3_NATIVE_EVIDENCE`, see BLOCKERS.md): the
   generated-Rust backend does not yet represent `Vec<Vec<String>>` or `str::bytes()`.
2. **Two documented CSV-format representational boundaries**, both inherent to minimal-quoting
   CSV and both explicitly authorized by the WP itself, not implementation defects: a record with
   exactly one empty field cannot be distinguished on the wire from a zero-field record (§17.5),
   and Python's own `csv` module resolves that same ambiguity differently (by quoting) than the
   frozen dialect requires (never quoting for emptiness) — see "Host-oracle differential" above.
3. **The workspace-confinement / package-placement finding** recorded in BLOCKERS.md's "Secondary
   finding" — worked around, not a blocker, but the WP's own illustrated `fixtures/consumer/`
   layout does not work as drawn against the current compiler.
4. **A pre-existing, previously-discovered engine limitation** (from the `std-time` session, WP
   §2's "known cross-file associated-function resolution limitation") shaped this package's API
   design directly: the frozen public API (§8) is free functions specifically because
   `Type::function()` paths do not resolve across `.stark` files. Not re-investigated here since
   it was already root-caused and the free-function design avoids it entirely — unlike
   `std-time`, this package's tests genuinely live in a separate `tests.stark` file (§23's cross-
   file proof), because there are no associated functions to trip the bug.

## Final status

PARTIAL — IMPLEMENTATION COMPLETE, WAITING_C6.3_NATIVE_EVIDENCE
