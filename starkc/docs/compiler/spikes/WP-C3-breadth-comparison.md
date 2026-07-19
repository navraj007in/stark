# Gate C3 Backend Breadth Comparison — generated-Rust vs direct-Cranelift

Prepared 2026-07-19, after the WP-C3.2/C3.3 breadth run. Consolidates the two spikes' coverage of
the frozen workload's *aggregate/generic breadth* — the dimension both initial spikes stopped
short of, and the one most likely to differentiate the candidates for WP-C3.4 (CE5). This is
evidence for the selection, **not the selection**.

## Scope of the breadth run (transparent)

- **Generated-Rust (WP-C3.2): extended broadly.** Added structs, `impl`/methods, struct literals,
  field/method access, generics + trait bounds, `Option`/`Result`, `match` + pattern lowering,
  and `String`/`&str`. Coverage 4/17 → **8/17**, all matching the interpreter.
- **Direct-Cranelift (WP-C3.3): breadth measured at the struct boundary, not fully implemented.**
  Extending the direct backend to the same breadth requires, per construct, a dedicated
  subsystem (detailed below). Implementing struct-by-value alone requires stack-slot layout,
  field-offset computation, load/store, **and an sret ABI transform for struct-returning
  functions** — a genuine ABI effort with real debugging risk. Rather than spend the C3 spike
  budget building a second real backend, the cost is **measured concretely from the Cranelift
  lowerer already built** (WP-C3.3's ~600-line integer/control-flow backend). Coverage remains
  **3/17**; the per-construct cost is analyzed below. (If an exact head-to-head number on structs
  is wanted, implementing Cranelift struct-by-value is a bounded, ~150–200-line follow-up.)

This asymmetry in *what it took to extend each* is itself the headline finding.

## Coverage

| Construct family | Generated-Rust (A) | Direct-Cranelift (B) |
|---|---|---|
| scalar arithmetic + control flow + traps | ✅ | ✅ |
| structs + methods (`struct_enum_trait__01`) | ✅ (~90 lines: struct/impl/literal/field/method text) | ✗ — needs stack-slot layout + field offsets + load/store + **sret ABI** |
| generics + trait bound (`struct_enum_trait__03`) | ✅ (rustc monomorphizes; ~20 lines) | ✗ — needs a **monomorphization engine** |
| `Option` + `match` (`option_result__01`) | ✅ (maps 1:1; match + pattern lowering) | ✗ — needs **tagged-union layout** + discriminant switch + pattern-match codegen |
| `String`/`&str` (`expr_stmt__02`, `__03`) | ✅ (maps to Rust `String`) | ✗ — needs a **string runtime** (alloc, len, compare) |
| **Frozen-corpus total** | **8/17** | **3/17** |

## Why the asymmetry (the decision-relevant core)

The generated-Rust backend climbs breadth **cheaply and mechanically** because every hard part of
lowering an aggregate/generic language is delegated to rustc:

- **Monomorphization** — rustc does it. The direct backend must implement generic instantiation,
  duplicate-instantiation control, and deterministic symbol naming itself (Gate C6.2 work anyway,
  but the backend owns it).
- **Aggregate layout & ABI** — rustc lays out structs/enums and handles struct-by-value
  passing/returning. The direct backend must compute field offsets, tagged-union representations,
  and struct-passing conventions (sret / register-packing) by hand.
- **Drop** — rustc runs `Drop` under `panic=abort`. The direct backend must elaborate drop flags
  and drop glue from MIR itself.
- **Runtime surface (String/Vec)** — rustc gives `String`/`Vec` for free. The direct backend
  needs a runtime library.

Conversely, the direct backend's advantages (measured in WP-C3.3) are real and orthogonal to
breadth: **no rustc build dependency**, **faster builds** (defensibly ~1.8× end-to-end on the
tiny workload — not a general multiple; see WP-C3.3's timing caveat), **direct ABI control**, and
it is the **bigger beneficiary of the mandatory MIR** (verified MIR ≈ Cranelift's own
block/terminator/`trapnz` model, and would supply the aggregate layout, monomorphization, and
drop elaboration the spike had to imagine writing by hand).

## What this does and does not establish

**Establishes:** for the aggregate/generic breadth the language actually needs, generated-Rust
reaches it with mechanical effort (proven: 8/17, zero mismatches), while the direct backend needs
a dedicated subsystem per construct family (measured from the built lowerer). The correctness of
both on their supported subsets is identical (zero mismatches, trap parity on both).

**Does not establish:** executable size, startup time, or steady-state runtime for either (still
unmeasured); the direct backend's *actual* breadth coverage once the MIR (Gate C4) supplies
layout/monomorphization/drop — which is the fairer comparison point and does not exist yet; and
whether the generated-Rust dependency weight or the direct-backend engineering cost is the
decisive factor. Those are WP-C3.4 / CE5 judgments.

## Bearing on WP-C3.4

The two candidates now have breadth evidence consistent with the §4 hypothesis in
`NATIVE-CORE-ARCHITECTURE.md`:

- **generated-Rust** = fastest path to *broad correctness* (reuses rustc's whole middle/back end),
  at the cost of a heavy mandatory toolchain dependency and slower builds;
- **direct-Cranelift** = fast builds + no rustc + ABI control + biggest MIR beneficiary, at the
  cost of owning monomorphization, aggregate layout, drop elaboration, and a runtime surface.

A key input WP-C3.4 should weigh: **most of the direct backend's breadth cost is MIR work that is
mandatory anyway** (Gate C4 builds monomorphization-ready, drop-elaborated, layout-bearing MIR).
Measured against *typed HIR* (as these spikes were), the direct backend looks far more expensive
than it will once it consumes verified MIR. The generated-Rust advantage measured here is
partly an artifact of comparing at the HIR level. WP-C3.4 should account for that.

Backend selection remains **WP-C3.4 / CE5 — owner decision.**
