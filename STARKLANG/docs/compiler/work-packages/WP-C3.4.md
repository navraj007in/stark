# WP-C3.4 — Backend and Runtime Architecture Selection

Gate: C3 (final WP). Scope from `COMPILER-ROADMAP.md` WP-C3.4. Selection is escalation **CE5 —
owner decision**.

## Scope

Compare reference interpreter / generated-Rust spike / direct-Cranelift spike and record one
outcome: `SELECT-GENERATED` / `SELECT-DIRECT` / `REVISE` / `BLOCKED`. A selected architecture must
specify its MIR consumption boundary, runtime ownership/ABI direction, target-platform plan,
debug/source-mapping approach, unsupported-MVP features + closure plan, and why the rejected
candidate is not the initial production path.

## Decision (owner, CE5, 2026-07-19)

**`SELECT-GENERATED`** — generated Rust as the initial production backend behind verified MIR,
with a **backend-neutral MIR contract keeping `SELECT-DIRECT` (Cranelift) open as a C7-gated
migration**. Recorded as CD-026 in `COMPILER-STATE.md`.

Basis (from the spikes + breadth run):
- generated-Rust reached 8/17 frozen-corpus breadth cheaply, zero mismatches, trap parity —
  shortest, lowest-risk path to correct broad native compilation (charter §1.6 rule 7);
- direct-Cranelift is correct and self-contained (no rustc dep) but owns monomorphization/layout/
  drop/runtime up front — the better *eventual* backend if the self-contained-compiler goal
  becomes primary (a C7 judgment on measured evidence);
- accepted trade: `stark build` permanently requires a rustc toolchain and is slower — acceptable
  for STARK-as-research-language, re-evaluated at C7.

Full three-way analysis and the required architecture commitments:
`starkc/docs/compiler/spikes/WP-C3.4-backend-selection-analysis.md` §4.

## Architecture commitments (summary; detail in the analysis §4)

- **MIR consumption:** emitter consumes verified MIR (Gate C4), not typed HIR.
- **Runtime/ABI:** small STARK runtime (print/panic/trap glue); Rust owns MVP value layout +
  calling convention; Native Provider ABI (C5.1) as `extern "C"` provider calls.
- **Targets:** rustc's matrix; Tier-1 (linux-x64, macos-arm64) first.
- **Debug/source mapping:** STARK span → generated-Rust line → rustc debug info; trap file:line
  via a span table (WP-C5.5).
- **Unsupported-MVP closure:** floats, `?`, tuple patterns, traits/Drop, references, Vec/HashMap,
  function values — tracked into C4.5/C5/C6.
- **Why direct is not the initial path:** higher up-front backend engineering inverts
  "correctness precedes optimisation"; retained as a C7-gated migration via backend-neutral MIR.

## Done when

- [x] Three-way comparison recorded with evidence (analysis doc).
- [x] One outcome selected by the owner (CE5) and recorded (CD-026, Native-backend-selection
      section = SELECTED / generated Rust/C).
- [x] The selected architecture's required commitments are specified.

## Next

**Gate C4 — WP-C4.1: MIR design review (CE3).** Define the backend-neutral MIR contract
(`STARKLANG/docs/compiler/mir.md`): functions/basic blocks, typed locals/temporaries, places/
projections, terminators, calls/returns, aggregates/discriminants, moves/copies, borrows, drop
flags/operations, trap/abort paths, spans/provenance, monomorphised-vs-generic representation,
validation invariants, textual dump + versioning. The generated-Rust emitter (and any future
direct backend) consumes this verified MIR.
