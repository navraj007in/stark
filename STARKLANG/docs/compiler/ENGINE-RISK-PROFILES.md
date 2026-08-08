# Engine risk profiles

**Packet:** `WP-ENGINE-INDEPENDENCE.md` EI4, approved 2026-08-09 (CD-392), executed as an AS8
prerequisite. Inputs: EI1 register, EI2 evidence audit, EI3 rustc inventory.

**Status: EI4 COMPLETE.**

---

## Acceptance criterion 4, answered first: the claim is REPLACED, not qualified

> *"three independent engines" is justified with qualifications or replaced.*

**Replaced.** The evidence does not support it, and no qualification rescues the word *independent*:

```text
one front end          lexer, resolver, checker — shared by all three, no alternative
one reference engine   the differential machinery calls HIR "THE HIR ORACLE" (EI2)
six authorities        INVISIBLE to all three, because the front end decides once (EI1)
two engines, not three at the provider boundary — the interpreters have no host access (EI2)
```

The accurate claim is:

> **One front end and three execution strategies, differentially compared against a reference
> engine, over a shared semantic core.**

That is still valuable — it catches lowering and execution defects that a single engine cannot —
but it is a much narrower claim than three independent implementations, and every public statement
should use the narrower one. EI6 calibrates the public wording; this is the input.

## HIR interpreter — `src/interp.rs`, 13,227 lines

| | |
| --- | --- |
| **Purpose** | Reference execution semantics. The differential's oracle |
| **Input representation** | Typed HIR, straight from the checker |
| **Semantic authorities consulted** | `ESF-COPY-001` DIRECT, `ESF-DROP-001` DIRECT, `ESF-TRAP-001` DIRECT, `ESF-RES-001` DIRECT, `ESF-TYPE-001`/`ESF-NUM-001`/`ESF-TRAIT-001` INDIRECT |
| **Independently implemented** | Value model, expression evaluation, destruction walk (a different structure from `mir::drop_plan`), control flow |
| **Shared authorities inherited** | Every front-end decision. It re-derives *nothing* the checker settled |
| **Primary strengths** | Closest to the specification's operational model; the only engine that can be read against the spec chapter by chapter |
| **Known defect clusters** | Largest deviation surface in the tree — **40 `KNOWN-DEVIATIONS` entries name `interp.rs`**, more than any other engine file |
| **Known under-exercised areas** | **No host access at all**, so providers, TCP/TLS, filesystem and process paths are entirely unexercised here |
| **Can detect** | Wrong lowering (by disagreeing with MIR/native); execution-order and value-model defects in the other two |
| **Cannot detect** | Anything it inherits from the front end — which is every `INVISIBLE` authority; anything provider-shaped |
| **Release role** | **Oracle, not a shipping target.** Its answer defines "expected" for the other two |

## MIR interpreter — `src/mir/interp.rs` (2,870 lines) plus supporting modules (16,097 total)

| | |
| --- | --- |
| **Purpose** | Execute the lowered IR; prove the lowering is executable and means what HIR meant |
| **Input representation** | MIR, after `mir::lower` |
| **Semantic authorities consulted** | `ESF-COPY-002` DIRECT, `ESF-DROP-002` DIRECT, `ESF-PROV-001` DIRECT, `ESF-TRAP-001` DIRECT, `ESF-RES-001` DIRECT |
| **Independently implemented** | Slot model, MIR-level place and projection semantics, drop-flag execution |
| **Shared authorities inherited** | The front end's, **plus** — critically — `TypeContext::is_copy` and `mir::drop_plan`, which the native backend also consumes |
| **Primary strengths** | The only engine that exercises MIR verification (`mir/verify.rs`) as a live check; catches lowering defects HIR cannot see |
| **Known defect clusters** | 7 deviation mentions — a much smaller recorded surface than HIR's 40, which reflects both a smaller engine and less independent exposure |
| **Known under-exercised areas** | Shares its two most load-bearing authorities with native, so the pair's agreement is inherited rather than corroborating |
| **Can detect** | Lowering defects; slot/drop-flag defects; provider signature mismatches |
| **Cannot detect** | `ESF-COPY-002` and `ESF-DROP-002` defects — it and native compute them from the same source |
| **Release role** | **Lowering-verification gate.** Not shipped; gates the native path |

## Native backend — `src/backend/generated_rust/`, 9,012 lines

| | |
| --- | --- |
| **Purpose** | The shipping engine. Generates Rust, compiles it, runs it |
| **Input representation** | MIR, emitted as safe Rust |
| **Semantic authorities consulted** | `ESF-COPY-002` DIRECT (one-line delegate), `ESF-DROP-002` DIRECT (application of the canonical plan), `ESF-TRAP-001` DIRECT, `ESF-RES-001` DIRECT, `ESF-PROV-001` INDIRECT |
| **Independently implemented** | Rust emission, the `checked_*` arithmetic lowering, the shift rule that diverges from `checked_shl`, FFI thunks |
| **Shared authorities inherited** | Everything MIR inherits, and MIR's own two — it re-derives neither Copy nor drop order |
| **Primary strengths** | **rustc is a genuine external control** (`RA-SAFE-REJECT`): it rejects generated code violating Rust's borrow and move rules, and has historically caught real lowering defects (the DEV-150 shape) |
| **Known defect clusters** | **0 deviation mentions of `backend/generated_rust`** — which is a statement about where defects were *recorded*, not evidence of correctness |
| **Known under-exercised areas** | `RA-LAYOUT` unmeasured (EI3 residual); `RA-LINTS` suppresses two deny-by-default lints, narrowing what rustc refuses |
| **Can detect** | Anything rustc rejects; real end-to-end behaviour including providers and the pinned external sample suite |
| **Cannot detect** | Lowering defects that produce **valid Rust with the wrong meaning** — rustc is silent there; and the two authorities it shares with MIR |
| **Release role** | **The product.** Everything else is a check on it |

## What each engine is *for*, stated as a hierarchy rather than a set

```text
HIR      defines expected behaviour          oracle
MIR      proves the lowering executes        gate
native   is the thing users run              product
```

Describing them as peers is what produces the "three independent engines" error. They sit in a
line: each checks the next, and none checks the front end they all share.

## The blind spot they share, restated for EI5

**No engine can detect a defect in a front-end decision, because none of them re-derives one.** Six
registered authorities are in that class, two of them `critical` risk. This is the single most
important input to mutation-target selection: a mutation in `copy_eligible_types` or
`nominals_with_destructor` should be expected to **survive** every differential suite in the tree,
and if EI5's trials show otherwise, the trial harness is what needs checking first.

## Handover to EI5

Rank highest: `INVISIBLE` × `critical`/`high` risk × no independent control.

```text
ESF-COPY-001   critical   INVISIBLE all three   control is IMPLEMENTATION_GENERATED
ESF-DROP-001   critical   INVISIBLE all three   no control at all
ESF-TRAIT-001  high       INVISIBLE all three   control is IMPLEMENTATION_GENERATED
ESF-TRAP-001   high       INVISIBLE all three   fixtures assert occurrence, not category
ESF-COPY-002   high       INVISIBLE mir/native  hir is the only control
ESF-DROP-002   high       INVISIBLE mir/native  hir walks a different structure
ESF-PROV-001   high       INVISIBLE mir/native  two engines only; external loopback is the control
```
