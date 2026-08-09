# AS4 — the `is_copy` audit

**Packet:** AS4, Sprint 3. **Adopts:** `AS0-RB0-PREDICATE-INVENTORY.md` §1, which recorded `is_copy`
as *"already consolidated — the rule is already single"*.
**Branch:** `wp-arch-stability/sprint-3`. **Date:** 2026-08-08.
**Status:** COMPLETE for the `Ty` language. AS0's finding was right about MIR and incomplete overall.

---

## 1. AS0's scope, and what it missed

AS0 audited RB0's list, which is a **MIR** predicate inventory. Within that scope it was correct:
`TypeContext::is_copy` and `FnLowerer::is_copy` are thin wrappers over one
`mir::mir_ty_is_copy(ty, &eligibility)`, differing only in the eligibility set they pass, and
`emit_types::mir_ty_is_copy` is a one-line wrapper over `TypeContext::is_copy`. That rule is single.

The census this audit ran was not scoped to MIR, and found Copy answered in **four type languages**:

| Language | Implementations |
| --- | --- |
| `MirTy` | `mir::mir_ty_is_copy` (+3 wrappers) — single, correct |
| `Ty` | `typecheck::is_copy_with_impls` **and** `borrowck::is_copy_type` — **two** |
| `Value` | `interp::value_is_copy`, `interp::pointee_is_copy` — runtime shapes, different question |
| tensor | `TensorTy::is_copy` — consulted by the checker |

Different *languages* justify different code. Two implementations in **one** language do not, and
that is what `Ty` had — `borrowck`'s own comment said it existed to stay "aligned with the type
checker's `is_copy_with_impls`", an alignment maintained by hand.

## 2. Measured before merging

`borrowck::as4_copy_rule_inventory` compares both rules across 25 samples spanning every `Ty`
variant. They agreed on all but one:

```text
Never: checker=true borrowck=false
```

03-Type-System.md: *"reference values, function values, `Unit`, and `!` are `Copy`"*. The checker is
right, and `borrowck` was wrong **because of a wildcard** — `_ => false` swallowed `Ty::Never`. A
second divergence, `Ty::Extension` (the checker consults `tensor.is_copy()`, `borrowck` returned
false), could not be sampled without the extension machinery and is recorded here rather than
asserted.

Neither is a live defect: no value of type `Never` exists to move, and the tensor path is a deferred
research track. The point is that **a wildcard decided them, not a person**.

## 3. What changed

`borrowck::is_copy_type` delegates to `typecheck::is_copy_type_with`, the public entry point that
**already existed** — this duplicate never had to be written.

After the merge every Copy predicate is wildcard-free:

| Predicate | Property-bearing wildcard |
| --- | --- |
| `mir::mir_ty_is_copy` | none (exhaustive, with the `HostResource` arm explicit) |
| `typecheck::is_copy_with_impls` | none |
| `borrowck::is_copy_type` | **gone** — now a delegate |
| `interp::value_is_copy`, `pointee_is_copy` | none |

That satisfies item 3 for this property: a new `Ty` or `MirTy` variant breaks compilation at the
authority rather than acquiring an answer nobody chose.

## 4. Evidence

lib 537 (including the new inventory), `copy_canon_matrix`, `c61f_structural_copy`, `gate4_tensor`,
`operand_move_inventory`, `dev135_field_move_paths`, `mir_differential` 132,
`three_engine_differential` 109, `as4_property_adversaries` 12. No behavioural CD due: one
implementation deleted in favour of an existing, more correct one.

## 5. A process note worth more than the change

Chasing a suspected hang in `three_engine_differential`, I bisected the Copy change for a long time.
It was not the cause. Killing earlier test runs had left **48 stale per-case build directories** in
`/var/folders/**/T/stark_3eng_*`; the single "hanging" test passed in **2.34 s** once they were
cleared, and the full suite in 103 s. Two invocations competing for one target directory is what
made a healthy run look stuck in the first place.

Recorded because the failure mode is not the code: *before concluding a semantic edit caused a hang,
re-run the one test alone after cleaning, and count the stale directories first.*

## 6. AS4 status

```text
drop rule       decomposed, named, merged, adversaries   DONE
reference rule  merged + named apart                     DONE
is_copy         audited; Ty duplicate merged; wildcard-free  DONE
RB0 Q1          iterator drop-glue asymmetry             OPEN, untouched
```
