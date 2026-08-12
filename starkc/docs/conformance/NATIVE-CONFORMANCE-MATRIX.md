# Native conformance matrix

**Generated. Do not edit.** Produced by `starkc/tests/native_conformance_matrix.rs` from a live
compiler run, and validated by that test on every CI run on all three Tier-1 platforms. Editing this
file by hand fails CI; so does changing the compiler without regenerating it.

```text
STARK_REGENERATE_CONFORMANCE_MATRIX=1 cargo test --test native_conformance_matrix
```

## What this answers

*Is this construct supported by the native compiler, and if not, what happens when I write it?* You
should not need `COMPILER-STATE.md`, the deviation ledger, or the compiler source to answer that.

Every cell below was measured on the run that generated this file. No cell is transcribed from a
report.

## Statuses

```text
SUPPORTED          every stage accepts it, and all four engine configurations -- HIR oracle,
                   MIR interpreter, native debug, native release -- produced the SAME
                   normative observation: stdout bytes, exit status, drop log, and any
                   trap's category and location

REFUSED-BY-DESIGN  STARK refuses it, deliberately, at a named stage. Writing it gives you a
                   STARK diagnostic at compile time -- never a rustc error, and never wrong
                   behaviour at run time

KNOWN-DEVIATION    the front end accepts it and MIR lowering refuses it. Valid STARK that this
                   compiler cannot build. The refusal is STARK's own and arrives before any
                   code is emitted, and the owning DEV is named

DEFERRED           scheduled, not yet implemented (no row currently carries this)

NOT-APPLICABLE     the stage has no meaning for this construct (no row currently carries this)
```

A `--` cell means the stage was never reached, because an earlier one refused the program. It does
**not** mean the stage accepted it.

## Summary

```text
constructs measured       20
SUPPORTED                 6   of which 6 were executed through all four engine
                               configurations and compared
REFUSED-BY-DESIGN         8
KNOWN-DEVIATION           6
```

**This is a boundary inventory, not a census of the language.** Every row is a construct chosen
because it sits at or near an edge of the supported subset -- that is what makes the boundary
legible. A construct absent from this table is not thereby unsupported; the core language beyond
these edges is covered by the conformance corpus and the differential suites. What this table is
authoritative about is the edges themselves.

## The matrix

| Construct | Parse | Resolve | Typecheck | HIR exec | MIR lower | MIR verify | MIR exec | Native debug | Native release | Status | Limitation / DEV |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A nested pattern in a match arm — `Some(Ok(n))` | pass | pass | pass | runs (exit 0) | pass | pass | runs (exit 0) | runs (exit 0) | runs (exit 0) | **SUPPORTED** | -- |
| A `match` over an integer with no arm covering the remaining values | pass | pass | REFUSES | -- | — | -- | -- | -- | -- | **REFUSED-BY-DESIGN** | `E0303` at type check |
| `Option::unwrap_or` where the payload owns a destructor — `Option<String>` | pass | pass | pass | runs (exit 0) | pass | pass | runs (exit 0) | runs (exit 0) | runs (exit 0) | **SUPPORTED** | -- |
| `Vec::insert` — and equally extend/truncate/sort/reverse/contains/dedup/split_off/drain/retain. `push`, `pop`, `len` and indexing are supported | pass | pass | pass | -- | REFUSES | -- | -- | -- | -- | **KNOWN-DEVIATION** | DEV-140 — refused by MIR lowering, before any code is emitted |
| `HashMap::entry` — reserved for the `std-full` profile this build does not carry | pass | pass | REFUSES | -- | — | -- | -- | -- | -- | **REFUSED-BY-DESIGN** | `E0304` at type check |
| `HashMap<K, V>` where `V` implements `Drop`. A `HashMap` of values without destructors is unaffected | pass | pass | pass | -- | REFUSES | -- | -- | -- | -- | **KNOWN-DEVIATION** | DEV-141 — refused by MIR lowering, before any code is emitted |
| `println` of a user struct that implements no `Display` | pass | pass | REFUSES | -- | — | -- | -- | -- | -- | **REFUSED-BY-DESIGN** | `E0500` at type check |
| `println` of a tuple of primitives — `(Int32, Bool)` | pass | pass | pass | runs (exit 0) | pass | pass | runs (exit 0) | runs (exit 0) | runs (exit 0) | **SUPPORTED** | -- |
| A composite mixing an owned droppable and a borrow — `(String, &str)`. Printing the parts separately works | pass | pass | pass | -- | REFUSES | -- | -- | -- | -- | **KNOWN-DEVIATION** | DEV-142 — refused by MIR lowering, before any code is emitted |
| `assert_eq` on a user type implementing `Eq`. `a == b` on the same type works in every engine | pass | pass | pass | -- | REFUSES | -- | -- | -- | -- | **KNOWN-DEVIATION** | DEV-143 — refused by MIR lowering, before any code is emitted |
| `for` over an iterator that is neither a range nor a `Vec` cursor — `HashMap::values()` | pass | pass | pass | -- | REFUSES | -- | -- | -- | -- | **KNOWN-DEVIATION** | DEV-144 — refused by MIR lowering, before any code is emitted |
| Moving out of an indexed place holding a droppable — `let m = a[0u64];` | pass | pass | REFUSES | -- | — | -- | -- | -- | -- | **REFUSED-BY-DESIGN** | `E0100` at type check |
| Tuple element access — `t.0` | pass | pass | pass | runs (exit 0) | pass | pass | runs (exit 0) | runs (exit 0) | runs (exit 0) | **SUPPORTED** | -- |
| `String::to_uppercase` — and equally to_lowercase/trim/replace/starts_with/ends_with/find/split_at/repeat. `len`, `as_str` and `push_str` are supported | pass | pass | pass | -- | REFUSES | -- | -- | -- | -- | **KNOWN-DEVIATION** | DEV-145 — refused by MIR lowering, before any code is emitted |
| Indexing a `String` — `s[0u64]` | pass | pass | REFUSES | -- | — | -- | -- | -- | -- | **REFUSED-BY-DESIGN** | `E0001` at type check |
| A fixed-size array with a literal length — `[Int32; 4]` | pass | pass | pass | runs (exit 0) | pass | pass | runs (exit 0) | runs (exit 0) | runs (exit 0) | **SUPPORTED** | -- |
| `HashSet::union` — reserved for the `std-full` profile this build does not carry | pass | pass | REFUSES | -- | — | -- | -- | -- | -- | **REFUSED-BY-DESIGN** | `E0304` at type check |
| Unary negation of an integer — `-x` | pass | pass | pass | runs (exit 0) | pass | pass | runs (exit 0) | runs (exit 0) | runs (exit 0) | **SUPPORTED** | -- |
| A function declared inside another function | REFUSES | — | — | -- | — | -- | -- | -- | -- | **REFUSED-BY-DESIGN** | refused by the parser |
| `Option::map_or` | pass | pass | REFUSES | -- | — | -- | -- | -- | -- | **REFUSED-BY-DESIGN** | `E0304` at type check |

## Tier-1 platforms

This matrix is platform-independent by construction, and the test that generates it runs in CI's
`fmt, clippy, test` job on **linux-x64, macos-arm64 and windows-x64**. A construct that behaved
differently on one of those would fail this test on that platform, rather than appearing here as a
footnote.

That is the entire Tier-1 claim. It is not evidence about any platform CI does not run.

## What this matrix does not tell you

- **It is not a completeness claim.** See the summary note above: these are boundary constructs,
  not the whole language.
- **A `KNOWN-DEVIATION` row is valid STARK.** It is refused by this compiler, not by the
  specification. Each names the DEV that owns the gap; the deviation ledger records why the repair
  is deferred, and what a working spelling is where one exists.
- **`SUPPORTED` means the four configurations agreed on this probe**, not that every program using
  the construct is correct. Agreement is checked against the shared comparator; the probe's own
  expectation is pinned separately by `layer_audit`, so a construct is not called supported merely
  because every engine is wrong about it in the same way.
