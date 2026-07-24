# stark-csv v0.1 blockers

## Primary classification

PARTIAL — IMPLEMENTATION COMPLETE, WAITING_C6.3_NATIVE_EVIDENCE

## Implementation baseline

8de4d41159083349133dc0696784a725a99aaacb

## Implementation commit

UNCOMMITTED

## Minimal reproducer

`fixtures/src/main.stark` -- the required external consumer -- calls `stark_csv::parse` (returns
`Vec<Vec<String>>`) then `stark_csv::write(&Vec<Vec<String>>)`, and is built with `stark build`
from `fixtures/`. A stripped-down reproducer without the package at all:

```stark
pub fn rows() -> Vec<Vec<String>> {
    let mut row = Vec::new();
    row.push(String::from("a"));
    let mut rows = Vec::new();
    rows.push(row);
    rows
}

pub fn count_rows(rows: &Vec<Vec<String>>) -> UInt64 {
    rows.len()
}
```
reproduces the same failure when built as a package's own `main`.

## Expected

A native generated-Rust binary that receives a nested `Vec<Vec<String>>` from `parse`, inspects
its fields, calls `write` on nested records, and returns a typed `CsvError` on malformed input --
i.e. real native execution of the package's actual data model (WP §10: `Vec<Vec<String>>`).

## Actual

Building the nested-`Vec<Vec<String>>` reproducer:
```text
error: native build does not yet support this program: linkage: body
`stark_csv::probe_count_rows@[]` param 0 is not C5-representable: MirTy Core(Vec,
[Core(Vec, [String])]) has no C5.3a generated-Rust representation yet -- enums land in
WP-C5.3b, Option/Result in WP-C5.3c, references/slices/String/Vec are outside the C5
subset; see WP-C5.1.md's MirTy matrix
```

Building the real consumer (`fixtures/`), which additionally calls `str::bytes()` internally via
`stark-csv`'s parser:
```text
error: native build does not yet support this program: method bytes on Str (a later C4.5e
sub-slice)
```

## Stage

emit (generated-Rust backend / MIR-to-Rust linkage, `starkc/src/backend/generated_rust`)

## Why package-local alternatives fail

1. `Vec<String>`, `Vec<Vec<String>>`, and `&str`/`String` byte-level operations are the package's
   actual data model (WP §10 "Data model": `Vec<Vec<String>>`) -- there is no alternative
   representation in Core v1 that avoids `String`/`Vec` entirely and still implements a CSV
   parser/writer as specified.
2. This is a native code-generation gap in `starkc/src/backend/generated_rust`, explicitly
   prohibited from modification by this work package (§6: "Prohibited: `starkc/**`").

## Prohibited upstream area required

`starkc/src/backend/generated_rust` -- native `String`/`Vec`/nested-`Vec<Vec<_>>` representation.
Tracked (per `COMPILER-STATE.md`'s current header) as unclosed WP-C6.3 ("runtime values and
collections incl. output, Track C").

## Work completed safely

- Full pure-STARK parser (four-state machine, §15) and deterministic writer (§17), 107 required
  tests (§19/§20/§21), all passing under `stark test` (HIR interpreter). See EVIDENCE.md's
  requirement ledger.
- Cross-file resolution proven: readiness Probe B, and the package's own `lib.stark`
  implementation / `tests.stark` corpus split (a genuine cross-file split, unlike `std-time`'s
  workaround of colocating tests with `impl` blocks in one file -- this package's public API is
  free functions specifically so that split works; see the note below).
- Cross-package resolution proven: readiness Probe C, and the required external consumer at
  `fixtures/` (parses two records, inspects exact fields, writes canonical CRLF output, triggers
  and matches one malformed-input error, using only the public API).
- Python host-oracle differential: see EVIDENCE.md and `tools/csv_oracle.py`.

## Unsupported completion claims

- Native execution is explicitly NOT claimed; `stark build` fails as shown above for both the
  package's own data model and the real consumer.
- `COMPLETE` status under WP §29 is NOT claimed; this package is classified PARTIAL.
- Target-platform support beyond what was actually built/run this session is NOT claimed.

## Minimum owner decision

Authorize/schedule WP-C6.3 (native `String`/`Vec`/nested-`Vec<Vec<_>>` representation in the
generated-Rust backend) as a prerequisite for any package needing collection types natively, or
accept this package at `PARTIAL — IMPLEMENTATION COMPLETE, WAITING_C6.3_NATIVE_EVIDENCE` until
that lands.

## Secondary finding (worked around, not a blocker)

`fixtures/` (the required external consumer) could not be placed at the WP §6-illustrated depth
`fixtures/consumer/` and still depend on the real `stark-csv/starkpkg.json`: the compiler's
workspace confinement (`starkc/src/package.rs::get_workspace_root`) is a fixed
"manifest's-directory's-parent" rule (not a nearest-common-ancestor search), and two directory
levels under `stark-csv/` cannot reach `stark-csv/starkpkg.json` itself, which sits one level
higher. Resolved by placing the consumer at `stark-csv/fixtures/` directly (one level, matching
the depth `starkc/tests/fixtures/c5-native-workspace/{app,logic}` uses for its own working
path-dependency siblings) rather than `stark-csv/fixtures/consumer/`. The same constraint applies
to `tools/csv_oracle.py`'s generated oracle-runner package, which is created and destroyed at
`stark-csv/.oracle_runner_scratch/` for the same reason. Not reported as a blocker because a
working layout exists; recorded because the WP's own illustrative layout does not work as drawn.
