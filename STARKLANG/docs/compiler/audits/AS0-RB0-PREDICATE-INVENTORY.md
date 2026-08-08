# AS0 item 7 — the RB0 predicate inventory, adopted

**Packet:** AS0 (Sprint 1, PARTIAL) item 7, executed as **AS4's opening inventory** per the
2026-08-07 scheduling decision.
**Adopts:** `WP-C7.8-RB0-MIR-Type-Property-Authority.md` (OPEN, not started).
**Branch:** `wp-arch-stability/sprint-2`. **Date:** 2026-08-07.
**Status:** COMPLETE. **AS0 now exits** — items 6, 7 done; item 10 deferred by decision.

AS0 item 7 says *"adopt the predicate inventory required by `WP-C7.8-RB0` **rather than creating a
second list**."* This document does that: it re-derives RB0's table against the current tree,
corrects the one row that has since gone stale, and supplies the equivalence evidence RB0's own
method demands before any consolidation. It creates no new authority and proposes no merge.

---

## 1. RB0's table, re-derived against the tree

| Rule | RB0 recorded | Verified 2026-08-07 |
| --- | --- | --- |
| is Copy | `TypeContext::is_copy` (`mir/mod.rs`), `FnLowerer::is_copy` (`mir/lower.rs`) | **STALE — not a duplication.** Both are three-line wrappers over one shared `mir::mir_ty_is_copy(ty, &eligibility)`; they differ only in which eligibility set they pass. The rule is already single |
| needs drop | `lower::ty_requires_drop_glue`, `lower::ty_has_user_destructor_guarded`, `verify::may_need_drop`, `verify::requires_drop_glue` | confirmed, all four present |
| carries a reference | `lower::ty_carries_ref`, `emit_types::ty_carries_reference`, `emit_types::ty_contains_ref` | confirmed, all three present |
| mentions a user nominal | `lower::ty_mentions_user_nominal` | confirmed, single implementation |

**The `is_copy` correction matters for scoping.** RB0's headline is "twelve implementations of four
rules". Two of the twelve are already consolidated — `mir_ty_is_copy` was extracted at some point
after RB0 was written, and the wrapper's own comment says so: *"Only the rule is shared, which is the
part that was drifting."* CD-240 (`HostResource` classified `Copy`, every resource leaked) is
therefore already structurally closed, not merely repaired.

That leaves **eight** implementations across **three** rules to consolidate, one of which
(`mentions a user nominal`) is a single implementation needing nothing.

---

## 2. The reference rule, measured

RB0's method is explicit: *"Each duplicate is removed only after a test demonstrates the survivor
agrees with it across the full `MirTy` variant set. A predicate that turns out to differ is kept,
renamed, and documented — deleting it would be the divergence, not the fix."*

That test now exists —
`mir::lower::as0_reference_predicate_inventory::the_three_reference_predicates_are_measured_against_each_other`
— over 33 samples spanning all 25 `MirTy` variants, with a second sample for every composite that can
carry a reference inside.

### Result

The three agree on **every variant except `FnPtr`**:

| Sample | `lower::ty_carries_ref` | `emit::ty_carries_reference` | `emit::ty_contains_ref` |
| --- | :---: | :---: | :---: |
| `FnPtr(ret &T)` | false | **true** | false |
| `FnPtr(param &T)` | false | **true** | false |
| every other variant, incl. `Struct<&T>`, `Enum<&T>`, `Tuple(&T)`, `Array(&T)`, `Slice(&T)`, `Core<&T>` | — agree — | | |

This is **RB0's Q2 reproduced from evidence** rather than from reading, and it narrows the packet
considerably. "Three implementations of one rule" is really *one rule with one disagreement, at one
variant*. Two of the three say a function pointer carries no reference; the third descends into its
parameters and return type.

### Not resolved here, deliberately

RB0 states the question that must be answered first:

> Determine first whether the two ask the same question. A plain function pointer does not borrow the
> values it will later receive, and a Rust `fn(&T)` is higher-ranked and needs no lifetime parameter
> — which is the only thing the lowering predicate guards (E0106). But a function *value*
> representation could carry an environment, lifetime-bearing metadata, or a bound receiver.

Forcing agreement would be a behavioural change without a CD, which RB0 exit criterion 5 forbids. The
disagreement is therefore **pinned in the test**, so it cannot change silently while AS4 decides.

---

## 3. The drop rule — four implementations, not yet measured

The four are not mechanically comparable the way the reference predicates are; their signatures
differ, and two need program context:

| Implementation | Signature | Needs |
| --- | --- | --- |
| `lower::ty_requires_drop_glue` | `(&self, &MirTy, Span) -> Result<bool, LowerError>` | a `FnLowerer` |
| `lower::ty_has_user_destructor_guarded` | `(&self, &MirTy, &mut BTreeSet<MirTy>) -> bool` | a `FnLowerer` + cycle set |
| `verify::requires_drop_glue` | `(&self, &MirTy) -> bool` | a verifier holding a program |
| `verify::may_need_drop` | `(&MirTy) -> bool` | free function |

**RB0 already records that they diverge, with the worked example**, so an equivalence harness is not
what unblocks this — a decision is:

- CD-287: `verify::requires_drop_glue` classified `HostResource` as needing no drop while
  `may_need_drop`, *in the same file answering the same question*, said the opposite.
- `HostResource` is the case that shows two of these are not duplicates at all: it **requires drop
  glue** (its provider close) while having **no user-defined destructor**, which is why
  `ty_has_user_destructor_guarded` answers `false` correctly and three others answer `true`.

So the drop rule splits into at least two genuinely different questions, exactly as RB0 anticipated:

```text
requires_drop_glue          vs   has_user_defined_destructor
```

Building the harness for these is AS4's work, because it needs a `FnLowerer` and a verified program —
i.e. it is a test over lowering, not over a pure function. Recorded here so AS4 opens with the shape
known rather than discovering it.

---

## 4. Q1 and Q2 — adopted, not answered

RB0 owns two open semantic questions and forbids resolving them by behavioural CD until its evidence
exists. Both are **adopted unchanged**:

| | Question | State after this inventory |
| --- | --- | --- |
| Q1 | iterator drop-glue asymmetry | **RESOLVED WITH EVIDENCE (2026-08-08)** — `AS4-RB0-Q1-ITERATORS.md`. Every constructible iterator is a borrowed cursor owning nothing, with no Rust `Drop` and a `Noop` drop plan, so the asymmetry is historical rather than semantic. Change deferred to its own CD |
| Q2 | `FnPtr` reference disagreement | **ANSWERED (2026-08-07)** — `AS4-REFERENCE-RULE.md`. The two ask different questions (storage vs signature); the identical pair merged, the near neighbour renamed |

---

## 5. What AS4 inherits

1. **Three rules, eight implementations** — not four rules and twelve. The `is_copy` rule is already
   consolidated; `mentions_user_nominal` is already single.
2. **The reference rule is one variant away from agreement.** Consolidating it is a decision about
   `FnPtr`, not a refactor across 25 variants.
3. **The drop rule is where the real work is** — four implementations answering at least two
   different questions, with a recorded instance of two of them contradicting each other inside one
   file.
4. **The equivalence-test pattern is established** and running in CI. AS4 extends it to the drop
   predicates rather than inventing an approach.

RB0's exit criterion 1 — *"inventory published: for each of the twelve predicates, the question it
actually answers"* — is **partially discharged** by this document: the reference rule's three are
measured and their one disagreement characterised; the drop rule's four are enumerated with their
known divergence but not yet measured. That remainder is AS4's opening work, which is the correct
place for it.

---

## 6. Method

```bash
# the predicate set, re-derived
grep -rn 'fn is_copy' starkc/src/mir/
grep -rn 'fn ty_requires_drop_glue\|fn ty_has_user_destructor_guarded\|fn may_need_drop\|fn requires_drop_glue' starkc/src/
grep -rn 'fn ty_carries_ref\|fn ty_carries_reference\|fn ty_contains_ref' starkc/src/
grep -rn 'fn ty_mentions_user_nominal' starkc/src/

# the equivalence matrix
cargo test --lib as0_reference_predicate
```

The harness lives in `src/mir/lower.rs` under `#[cfg(test)]`, with a test-only
`emit_types::contains_ref_for_inventory` window so a private predicate can be measured without
widening production visibility. It adds no caller and changes no behaviour — RB0's *"audit BEFORE
consolidation"* taken literally.

---

## 7. AS0 status after this item

| Item | State |
| --- | --- |
| 6 — callable execution-site inventory | **DONE** — `AS0-CALLABLE-EXECUTION-SITE-INVENTORY.md` |
| 7 — RB0 predicate inventory | **DONE** — this document |
| 10 — `WP-ENGINE-INDEPENDENCE.md` AS0 scope | **DEFERRED to AS8/C10** by owner decision, 2026-08-07 |

**AS0 EXITS.** All items are done or explicitly deferred by decision, which is the condition AS0's
own §7 sets. Campaign A's exit gate now requires only AS3 and AS4.
