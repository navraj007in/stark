# AS4 — the reference rule: two implementations merged, one near neighbour named apart

**Packet:** AS4, Sprint 3. **Adopts:** `AS0-RB0-PREDICATE-INVENTORY.md` §2 and RB0's Q2.
**Branch:** `wp-arch-stability/sprint-3`. **Date:** 2026-08-07.
**Status:** COMPLETE. Behaviour-neutral; no decision record due.

---

## 1. AS0's summary hid the fact that decides it

AS0 measured three implementations across 33 samples spanning every `MirTy` variant and reported
*"they agree except on `FnPtr`"*. True, and it obscures the pairwise structure:

```text
lower::ty_carries_ref  ==  emit::ty_contains_ref        on EVERY sample, FnPtr included
emit::ty_carries_reference differs, and ONLY on FnPtr
```

This was never three implementations of one rule. It is **two implementations of one rule** plus
**one near neighbour asking a different question** — the same shape the drop rule turned out to
have, where `may_need_drop` was a genuine third question rather than a sloppy copy.

The pairwise identity is now asserted **per sample** inside AS0's matrix, so the claim the merge
rests on cannot quietly stop holding.

## 2. RB0's Q2, answered by consumers rather than by reading

> Determine first whether the two ask the same question.

| Predicate | Consumer | The question it actually asks |
| --- | --- | --- |
| `lower::ty_carries_ref` | Display guard for droppable composites | *would emitting this need a lifetime parameter the backend cannot generate (E0106)?* |
| `emit::ty_contains_ref` | `derives_for` | *does a structural derive apply?* |
| `emit::ty_carries_reference` | `emit_bodies`, `emit_places` local initialisation | *is a reference named anywhere here, including in a function's signature?* |

`FnPtr` separates them because a Rust `fn(&T)` is **higher-ranked** (`for<'a> fn(&'a T)`) and needs
no lifetime parameter. So the storage question answers *no* and the guard it feeds is right to; the
signature question answers *yes* and the initialisation logic it feeds is right to.

RB0's suspicion — "a function *value* representation could carry an environment, lifetime-bearing
metadata, or a bound receiver" — is not what is happening. The difference is simpler and it is
real.

## 3. The disagreement is LIVE, unlike `Core(File)`

Measured, because reachability is what settled DEV-196 and it had to be settled here too:

```stark
let g: fn(&Int32) -> Int32 = takes;     // constructible, lowers, runs in all three engines
let h: fn(&Int32) -> &Int32 = gives;    // same
```

So it could not be dismissed as an unreachable legacy shape. `tests/as4_reference_rule.rs` runs
both through the three-engine comparator, plus a reassigned function value (which forces the
initialisation path the differing predicate feeds) and a no-reference control.

One reachability limit recorded rather than left as an absence: a function value **nested in a
composite cannot be called through it** (`indirect callee expression (C4.5)`), so the disagreement
cannot be exercised one level deeper today. Pinned, and it fails when that limit lifts.

## 4. What changed

```text
lower::ty_carries_ref     ─┐
                           ├─→  mir::reference_rule::stores_a_reference     (merged, one authority)
emit::ty_contains_ref     ─┘

emit::ty_carries_reference  →  emit_types::mentions_a_reference             (renamed, kept)
```

`stores_a_reference` / `mentions_a_reference` cannot be substituted for one another by a reader
doing what the name suggests — which is what `carries_ref` vs `carries_reference` vs `contains_ref`
invited. Both matches stay exhaustive with no property-bearing wildcard, so a new `MirTy` variant
breaks compilation at the authority (item 3).

## 5. Evidence

| Check | Result |
| --- | --- |
| AS0 reference matrix | green, now asserting the pairwise identity per sample |
| `as4_reference_rule` | 5 cases, three engines |
| function-value and reference natives | `native_c5_4_function_values`, `native_c61f_aggregates`, `native_c61f_b3_stored_refs`, `native_c61f_ret_refs` green |
| engines | `mir_differential` 132, `three_engine_differential` 109 |
| external sample suite | 39/39 |
| accepted/rejected source programs | unchanged |

No behavioural CD is due: one predicate was deleted in favour of an identical one, and one was
renamed.

## 6. AS4 status

```text
drop rule       decomposed, named, merged, adversaries   DONE
reference rule  merged + named apart                      DONE
is_copy         consolidated behind mir_ty_is_copy        NOT AUDITED for wildcard-freedom
RB0 Q1          iterator drop-glue asymmetry              OPEN, untouched
RB0 Q2          FnPtr reference disagreement              ANSWERED: different questions, both kept
```
