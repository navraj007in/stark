# WP-C3.2 — Generated Rust/C Backend Spike

Gate: C3. Scope from `COMPILER-ROADMAP.md` WP-C3.2.

## Scope

Implement an isolated prototype for the frozen workload subset using generated Rust. May use a
small runtime library and a host toolchain. Must not bypass type, ownership, or artifact checks
already completed by the front end (charter §2.2). Record: unsupported constructs,
source-to-generated traceability, build-tool dependencies, cross-platform behaviour, semantic
mismatches, glue per feature, feasibility of consuming verified MIR instead of typed HIR.

This WP does **not** select a backend (WP-C3.4, CE5) or merge the spike as production
architecture (charter §2.2 — spikes are disposable until the gate decision).

## Deliverables (done)

- `starkc/tests/spike_genrust.rs` — isolated HIR→Rust lowerer + compile/run/diff harness over
  the frozen `exec_snapshots` corpus (`corpus_version = 1.0.0`), interpreter as oracle.
- `starkc/docs/compiler/spikes/WP-C3.2-generated-rust.md` — the spike report with every required
  measurement dimension.

## Result

4/17 frozen corpus cases lowered and matched the interpreter exactly (arithmetic/precedence,
loops/`for`/`break`/`continue`, multi-width integers, and `Int8`-overflow trap→abort parity);
zero semantic mismatches on supported cases; 13/17 cleanly reported unsupported with reasons;
mean `rustc` compile 87 ms/case. Trap-abort parity demonstrated end to end. Full report and the
§7-dimension mapping: `starkc/docs/compiler/spikes/WP-C3.2-generated-rust.md`.

## Done when

- [x] Isolated generated-Rust prototype exists and does not bypass front-end checks.
- [x] Runs against the frozen workload with the interpreter as oracle; supported cases match.
- [x] Unsupported constructs recorded (not silently mislowered).
- [x] All WP-C3.2 measurement records captured in the spike report.
- [x] MIR-consumption feasibility assessed.

## Next

WP-C3.3 (direct Cranelift spike) — same frozen workload, same measurement record — then WP-C3.4
selects under CE5.
