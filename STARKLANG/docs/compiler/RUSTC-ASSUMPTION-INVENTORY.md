# rustc assumption inventory

**Packet:** `WP-ENGINE-INDEPENDENCE.md` EI3, approved 2026-08-09 (CD-392), executed as an AS8
prerequisite. **Vocabulary frozen at EI0.**

**Status: EI3 COMPLETE.**

The native backend compiles generated Rust. Everything rustc decides that STARK does not re-decide
is part of the native engine's **trusted base**, and is invisible to the HIR and MIR engines by
construction — they never see it. This inventory makes that base explicit.

---

## The shape of the finding

**The backend's posture is better than the packet assumed for the two assumptions that matter most,
and the reason is written into the build itself.** Arithmetic and panic behaviour are *not* delegated
to rustc:

```text
overflow    STARK lowers to explicit checked_* calls. `overflow-checks = true` is set in the
            release profile and the build comment states it is "RECORDED RATHER THAN RELIED UPON
            ... trapping does not depend on this -- but leaving it unstated would invite the
            reader to assume it does."
panic       panic = "abort" in BOTH profiles. The comment records it as defence-in-depth: the C7.0
            panic-site audit found no user-reachable Rust panic in the runtime, and the setting
            exists so the guarantee "rests on the build, not only on that audit staying true".
```

Both are `RUSTC_ASSUMPTION` entries that were deliberately **converted into STARK decisions**. That
is the right direction, and it narrows the trusted base rather than documenting it.

One case runs the other way and is the sharpest entry here: **`checked_shl` is not used**, because
Rust's `checked_shl` validates only the *shift count* and silently drops overflowed bits — which is
not STARK's rule. The backend emits its own check instead. A semantic difference between the two
languages, found and compensated.

## Inventory

| ID | Assumption class | STARK's position | Trusted base? | Residual |
| --- | --- | --- | --- | --- |
| `RA-OVERFLOW` | integer overflow configuration | **Not delegated.** Explicit `checked_*` per operation (`emit_checked_expr`); `overflow-checks = true` recorded as defence-in-depth | **No** — converted to a STARK decision | The generated `checked_*` calls are themselves a shared lowering; a defect there is invisible to hir/mir, which never execute it |
| `RA-SHIFT` | shift semantics | **Divergence, compensated.** Rust's `checked_shl` validates the shift count only and silently drops overflowed bits; STARK's rule differs, so the backend does not use it | **No** | Only the native engine exercises this path — hir and mir cannot corroborate it |
| `RA-PANIC` | panic strategy / unwinding | `panic = "abort"` in **both** dev and release profiles | **No** — pinned, not assumed | Abort is asserted by the profile; a toolchain that ignored the setting would be undetected in-tree |
| `RA-EDITION` | edition behaviour | `edition = "2021"`, pinned in the generated manifest | **Yes**, but pinned | Edition-specific behaviour changes silently on a future edition bump |
| `RA-LINTS` | compiler-version-specific acceptance | `#![allow(dead_code, unused, unconditional_panic, arithmetic_overflow)]` — the last two are **deny-by-default rustc lints** that generated code can legitimately trip | **Yes** | Suppressing `unconditional_panic`/`arithmetic_overflow` means rustc will no longer refuse code that STARK intends to trap. Correct for the lowering, and it removes a rustc-side safety net that would otherwise catch a lowering bug |
| `RA-FFI` | FFI calling convention | One `unsafe` block per provider call, scoped to **exactly the FFI call** — `emit_provider.rs:332` records the narrowness as deliberate | **Yes** | The provider ABI boundary is where rustc's guarantees stop; `ESF-PROV-001` already records that only two engines reach it |
| `RA-MAYBEUNINIT` | generated-reference validity | Provider out-parameters use `MaybeUninit` + `assume_init` | **Yes** | `assume_init` is a soundness obligation STARK discharges by construction; nothing in-tree proves the provider wrote the value |
| `RA-DROP` | drop order | STARK decides via `mir::drop_plan`; the backend **applies** that plan rather than relying on Rust's own drop order | **No** — converted | `ESF-DROP-002`: mir and native share the plan, so their agreement is inherited |
| `RA-LAYOUT` | layout and representation | Not relied upon for semantics; STARK's layout tables are its own | **Partially** | Not measured in this pass — recorded as a residual rather than claimed |
| `RA-SAFE-REJECT` | safe Rust rejection as a backend validity check | **This is the one to be careful about** — see below | — | See below |

## `RA-SAFE-REJECT` — rustc's rejection is a real check, and it is not ours

The pipeline emits **safe** Rust (one narrowly scoped `unsafe` per provider call). rustc therefore
rejects generated code that violates Rust's borrow and move rules, and that rejection has
historically caught real lowering defects — the DEV-150 shape the packet names:

```text
HIR accepted the program
MIR/native lowering emitted Rust
rustc rejected the generated Rust
```

*(DEV-150 itself is CLOSED — ruled and fixed at CD-357, 2026-08-02 — and is recorded here as the
shape, not as an open defect.)*

Two things follow, and they pull in opposite directions:

1. **It is genuine independent evidence.** rustc is an external checker that did not derive its
   rules from STARK's implementation. Under EI0's vocabulary this is the strongest kind of control
   the native lane has — it is `EXTERNALLY_DERIVED`.
2. **It only fires where Rust's rules and STARK's coincide.** A lowering defect that produces
   *valid* Rust with the *wrong* meaning passes rustc silently. And `RA-LINTS` deliberately
   suppresses two deny-by-default lints, narrowing what rustc will refuse.

So "the generated crate compiles" is admissible evidence about lowering validity and **not**
evidence about lowering *correctness*. Those are routinely conflated; recording the distinction is
this entry's purpose.

## Toolchain identity — registered, not implemented

EI3 proposes that AS5/C10 require public release artefacts to record:

```text
STARK compiler commit          rustc version and commit hash      panic strategy
STARK compiler version         Cargo version                      backend/runtime versions
Rust toolchain channel/version target triple / host triple        provider versions
                                                                  build profile
```

**Follow-up registered rather than implemented here**, as the packet directs: consider
`stark --version --verbose` emitting the above. This inventory does not implement it and does not
claim it exists.

**Current state, measured:** the C6.4/C6.5 qualification jobs already emit `host_triple` and
`commit_sha` in their result JSON, so part of this exists for CI artefacts but not for release
artefacts, and not as a single command.

## What EI3 hands to EI4 and EI5

```text
RA-OVERFLOW  RA-SHIFT  RA-DROP     converted to STARK decisions — mutate the STARK rule, not rustc
RA-LINTS     RA-SAFE-REJECT        the native lane's external control, with a stated limit
RA-FFI       RA-MAYBEUNINIT        soundness obligations at the provider boundary, two engines only
RA-LAYOUT                          NOT MEASURED — residual, per EI0 this is not "no assumption"
```

EI5's `RUSTC_ASSUMPTION` mutation class should target the **converted** rules — `checked_*`
emission, the shift check, the drop-plan application — because those are the places where STARK
took the decision away from rustc and is therefore solely responsible for it.
