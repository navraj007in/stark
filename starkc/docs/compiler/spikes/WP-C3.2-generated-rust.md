# WP-C3.2 — Generated-Rust Backend Spike Report

Gate C3 spike. Prepared 2026-07-19. Records the generated-Rust candidate (Candidate A in
`STARKLANG/docs/compiler/proposals/NATIVE-CORE-ARCHITECTURE.md`) measured against the frozen
workload. **This is a disposable spike (charter §2.2), not a production backend.** Its artifact
is `starkc/tests/spike_genrust.rs`: an isolated integration test, not wired into `stark build`,
adding nothing to the library surface. It does not select a backend — that is WP-C3.4 (CE5).

## What was built and run

An isolated HIR→Rust lowerer that consumes programs already validated by the real front end
(parse, resolve, type/borrow check — no front-end check is bypassed, charter §2.2), walks the
typed HIR, and emits Rust source for a supported subset. The generated Rust is compiled with
`rustc`, run, and its stdout + exit status compared against the reference interpreter (the
semantic oracle) over the frozen `exec_snapshots` corpus (`corpus_version = 1.0.0`). Two tests:
`genrust_spike_matches_interpreter_on_frozen_corpus` (compile+run+diff; skips cleanly, class
MANUAL, if `rustc` is absent) and `genrust_spike_reports_unsupported_constructs_cleanly`.

## Result summary

- **Supported and matched the interpreter exactly: 4 / 17 frozen corpus cases.** Zero semantic
  mismatches on any supported case.
- **Trap-abort parity demonstrated**: `primitive__02` (`Int8` `120 + 100`) traps in the
  interpreter and the generated binary exits non-zero — the abort-without-unwind contract holds
  through generated Rust.
- **13 / 17 cases cleanly reported unsupported**, each with a specific reason — no silent
  mislowering.
- **Mean `rustc` compile time: 87 ms/case** (range 84–92 ms), default (debug) `rustc`, one file.

| Case (frozen corpus) | Result | Note |
|---|---|---|
| `expr_stmt__01_arithmetic_and_precedence` | ✅ match | workload item 1 (arithmetic/precedence) |
| `expr_stmt__03_loops_break_continue` | ✅ match | workload item 2 (loops, `for`/range, `break`/`continue`) |
| `primitive__01_integer_widths_and_overflow_traps` | ✅ match | multi-width integers |
| `primitive__02_integer_overflow_traps` | ✅ match/trap | workload item 8 (trap → abort) |
| `expr_stmt__02_if_else_and_block_tail` | — unsupported | `String` |
| `expr_stmt__04_match_and_patterns` | — unsupported | `match`, `String` |
| `primitive__03_float_arithmetic_and_casts` | — unsupported | `Float64` |
| `struct_enum_trait__01..04` | — unsupported | struct / enum / trait / generic |
| `ownership_drop__01/02` | — unsupported | struct / references |
| `option_result__01/02` | — unsupported | `Option`/`Result` (non-primitive) |
| `collection_iter__01/02` | — unsupported | `Vec` / `HashMap` |

## Measurement dimensions (WP-C3.2 required record)

**Unsupported constructs (recorded).** The subset is integer primitives (`i8`..`u64`) + `Bool`;
trap-checked arithmetic; comparisons/logic; `let`/`let mut`/assignment; `if`/`else`, `while`,
`loop`+bare `break`, `continue`, `for x in a..b`; block-tail values; non-generic
receiverless functions and their calls; `print`/`println`. Everything else is returned as
`Unsupported(reason)` and observed above: `String`/`Vec`/`HashMap`, floats, structs, enums
(incl. `Option`/`Result`), traits, generics, references, `match`, `?`, casts. These are the
constructs WP-C3.2's sibling breadth (structs/enums/generics/traits/references/Drop/function
values) would need, and they map onto the risk register in `NATIVE-CORE-ARCHITECTURE.md` §6.

**Source-to-generated-code traceability.** 1:1 at the item level: each STARK `fn` becomes one
Rust `fn` of the same name (STARK `main` → Rust `main`); locals and parameters keep their STARK
identifier text (valid Rust idents in-subset); the expression tree maps structurally, emitted
fully parenthesised so STARK precedence is preserved without re-parsing. Representative output
for `println(2 + 3 * 4)`:

```rust
stark_println((2i64).s_add((3i64).s_mul(4i64)));
```

The mapping is direct enough that a source-span → generated-line table would be mechanical; no
construct required restructuring that would break a line mapping. (A production version would
still need real debug-info mapping for trap file:line, see §"open" below.)

**Build-tool dependencies.** One: a host `rustc`. No `cargo`, no crates, no build script — the
generated program is a single self-contained `.rs` file (runtime prelude inlined). This is the
candidate's central liability at scale (a rustc toolchain is a heavy mandatory build dependency
for end users) and its central asset here (nothing else to integrate to get a working binary).

**Cross-platform behaviour.** Inherited from `rustc`: the generated source is
target-independent and uses only `std` (`println!`, `std::process::exit`, `checked_*`
arithmetic). No platform-specific code was generated. Tier-1 targets (linux-x64, macos-arm64)
come free from rustc's target support; not separately exercised in this spike (single host).

**Semantic mismatches.** Zero on supported cases. All four matched the interpreter's stdout
byte-for-byte, and the trap case matched the trap→non-zero-exit contract. The trap-checked
arithmetic helpers (`StarkChecked`) were necessary and sufficient to reproduce STARK's
always-trap-on-overflow/div-by-zero semantics *independent of build profile* — the generated
binary never relies on rustc's profile-dependent overflow behaviour.

**Amount of glue per language feature.** Low and localized:
- arithmetic with trap semantics: one `StarkChecked` trait + a macro impl over the 8 integer
  types (~20 lines of prelude), then every arithmetic op is a `.s_op()` call;
- `print`/`println`: two 1-line generic helpers over `Display`;
- control flow (`if`/`while`/`loop`/`for`/`break`/`continue`), `let`, assignment, calls,
  block-tail values: ~1:1 structural emission, effectively no glue.
The glue that *would* grow is exactly the unsupported set: aggregate layout, enum discriminants,
monomorphization (rustc absorbs this if we emit generic Rust), Drop elaboration, and the
provider/reference ABI — quantified only when the breadth is built.

**Feasibility of consuming verified MIR instead of typed HIR.** The spike lowers from typed HIR
and had to query `TypeTables::expr_types` at lowering time to (a) pick the Rust integer type for
literals and (b) confirm arithmetic operands are integers. A verified MIR (Gate C4) would carry
explicit typed locals/temporaries and explicit trap points, removing those side-table lookups
and making trap emission a direct translation of a MIR terminator rather than an inline helper
call. Nothing in the generated-Rust approach conflicts with consuming MIR — it is a strictly
cleaner input for this backend. This supports the mandatory production shape (typed HIR → **MIR**
→ backend).

## Mapping onto NATIVE-CORE-ARCHITECTURE.md §7 dimensions

| Dimension | Spike evidence |
|---|---|
| 1 implementation complexity | ~600-line isolated lowerer for the subset; arithmetic/control-flow glue low; breadth deferred |
| 2 compile time | 87 ms/case mean (single-file `rustc`, debug); scaling not yet measured |
| 3 executable size | not measured this spike (deferred to a breadth run) |
| 4 startup time | not measured (debug single binaries; trivial) |
| 5 runtime performance | not measured — subset too small to claim a ratio (charter caution) |
| 6 source mapping / trap file:line | traceability high at item/line level; real debug-info mapping is open work |
| 7 cross-platform effort | inherited from rustc; not separately exercised |
| 8 semantic parity risk | **zero mismatches on supported cases**, incl. trap parity |
| 9 external dependency / maintenance | single dep `rustc`; codegen is text (low maintenance) but toolchain weight is the liability |
| 10 MIR / ABI compatibility | MIR is a cleaner input than HIR; no conflict |
| 11 monomorphisation complexity | not exercised (generics unsupported in spike); rustc would absorb it if we emit generic Rust |
| 12 trait-dispatch complexity | not exercised (traits unsupported in spike) |
| 13 reference/slice/Drop ABI risk | not exercised (references/Drop unsupported in spike) |

## Reading of the evidence (not a decision)

For the subset it covers, generated Rust is a **low-glue, zero-mismatch** path: control flow and
arithmetic (including the load-bearing trap-abort contract) reproduce the interpreter exactly,
compile fast at single-file scale, and trace cleanly back to source. The candidate's open
questions are precisely the ones `NATIVE-CORE-ARCHITECTURE.md` §4 flagged as its liabilities and
this spike did **not** resolve: the rustc build-dependency weight, compile-time *scaling* (only
single-file measured), executable size, and real debug-info trap mapping — plus the whole
unsupported breadth (aggregates, generics, traits, references, Drop, function values). None of
these were falsified; none were cleared.

The direct-Cranelift spike (WP-C3.3) must be run before any comparison, and dimensions 3/5/11/12/
13 need a breadth run on both candidates. **Backend selection remains WP-C3.4 / CE5 (owner).**
This report is evidence for that decision, not the decision.

## Reproduce

```bash
cd starkc
cargo test --test spike_genrust -- --nocapture   # prints the frozen-corpus coverage table
```

Skips (MANUAL) if `rustc` is unavailable. The lowerer and harness are `tests/spike_genrust.rs`
and are deleted or rewritten when Gate C3 selects — they are not production code.
