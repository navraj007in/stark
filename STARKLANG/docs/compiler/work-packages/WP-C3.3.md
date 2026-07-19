# WP-C3.3 — Direct (Cranelift) Backend Spike

Gate: C3. Scope from `COMPILER-ROADMAP.md` WP-C3.3.

## Scope

Implement the same frozen workload subset as WP-C3.2 using the strongest plausible simple direct
backend (Cranelift). Do not implement advanced optimisation. Record the same measurements and
unsupported constructs. Must not bypass front-end checks (charter §2.2); disposable until Gate C3
selects (WP-C3.4, CE5).

## Deliverables (done)

- `starkc/tests/spike_cranelift.rs` — isolated HIR→Cranelift-IR lowerer, object emission, `cc`
  link to a standalone executable, run/diff harness over the frozen corpus (interpreter oracle).
- `starkc/docs/compiler/spikes/WP-C3.3-direct-cranelift.md` — spike report + head-to-head table
  vs WP-C3.2, with an explicit timing caveat (no general performance-multiple claim).
- Cranelift dev-dependencies (pinned 0.110 for rustc-1.93 compatibility) with a necessity note in
  `Cargo.toml`; dev-only, not part of the shipped compiler surface.

## Result

3/17 frozen corpus cases lowered and matched the interpreter exactly (arithmetic/precedence,
loops/`for`/break/continue, Int8-overflow trap→abort parity); 0 semantic mismatches on supported
cases; 14/17 cleanly reported unsupported (same families as C3.2 plus unsigned integers — spike
is signed-only). Cranelift codegen ~2 ms/case (phase-only), `cc` link ~47 ms/case; defensible
end-to-end ~49 ms vs rustc's ~87 ms ≈ 1.8× on this tiny workload — **not** a general multiple.
No rustc build dependency. Finding: Cranelift 0.133 needs rustc ≥1.94 (>1.93 here), forcing the
0.110 pin — a real MSRV-churn maintenance cost. Full report + comparison:
`starkc/docs/compiler/spikes/WP-C3.3-direct-cranelift.md`.

## Done when

- [x] Direct Cranelift prototype exists, does not bypass front-end checks, produces a standalone
      native executable.
- [x] Runs against the frozen workload with the interpreter oracle; supported cases match.
- [x] Same measurement record + unsupported constructs as WP-C3.2, with a head-to-head table.
- [x] MIR-consumption feasibility assessed (direct backend benefits most from the mandatory MIR).

## Next

WP-C3.4 — backend and runtime architecture selection (compares reference interpreter /
generated-Rust spike / direct spike; outcome SELECT-GENERATED / SELECT-DIRECT / REVISE / BLOCKED;
escalation CE5, owner decision).
