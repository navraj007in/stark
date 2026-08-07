# AS4 opening — the drop rule, measured

**Packet:** AS4, Sprint 3, `WP-ARCHITECTURE-STABILIZATION.md`.
**Adopts:** `AS0-RB0-PREDICATE-INVENTORY.md` §3, which left the drop rule as AS4's opening work.
**Branch:** `wp-arch-stability/sprint-3`. **Date:** 2026-08-07.
**Status:** measurement + naming complete. **No predicate merged** — RB0 forbids consolidating
before the evidence exists, and the evidence says merging would be wrong.

---

## 1. The finding: three questions, not two

RB0 anticipated the drop rule splitting into `requires_drop_glue` vs `has_user_defined_destructor`.
The matrix shows a **third**, and it is not a wording difference:

| Question | Authority | `HostResource` | `Option<Int32>` |
| --- | --- | :---: | :---: |
| requires drop glue (precise) | `verify::requires_drop_glue`, `lower::ty_requires_drop_glue` | **true** | **false** |
| MAY require drop glue (conservative) | `verify::may_need_drop` | true | **true** |
| has a USER-written destructor | `lower::ty_has_user_destructor_guarded` | **false** | false |

Two variants separate the first two columns, measured rather than reasoned:

```text
AS4-DROP-DISAGREE ["Enum::CoreOption(Int32)", "Enum::CoreOrdering"]
```

`may_need_drop` answers `true` for every `Enum`/`Struct`/`Core` regardless of whether glue exists.
That is not a bug to fix. Substituting the precise rule for it would let the verifier **agree a
`Drop` was correctly absent** — the exact failure its own comment documents. Substituting it for the
precise rule would make the verifier reject valid lowerings.

`HostResource` separates the third column from both: its close is provider-driven (A11 §5), so it
requires glue while having no `impl Drop`. **CD-287 was two of these predicates disagreeing about
exactly this variant, inside one file, repaired one at a time.**

---

## 2. What changed (AS4 work items 1 and 2)

**Item 1 — one documented meaning and authority.** `verify::mir_needs_drop` was a method on
`BodyCx` reading only `self.program.types`, which made the rule unmeasurable without constructing a
per-body context. It is now the free function `verify::requires_drop_glue(types, ty)`; the method is
a one-line delegate. Same rule, one caller, now measurable — which is what let §1's matrix exist.

**Item 2 — named so they cannot be substituted.** The old names all led with "drop", the word all
three questions share:

| Before | After | Why |
| --- | --- | --- |
| `verify::mir_needs_drop` | `verify::requires_drop_glue` | names the question, pairs with lowering's |
| `lower::ty_needs_drop` | `lower::ty_requires_drop_glue` | read like any of the three |
| `lower::ty_has_user_drop_guarded` | `lower::ty_has_user_destructor_guarded` | buried the distinction ("user") mid-name |
| `verify::may_need_drop` | unchanged | already says "may" |

A reader substituting one for another was doing what the names invited.

---

## 3. Evidence

`mir::lower::as4_drop_predicate_inventory`, two tests:

- `the_conservative_and_precise_drop_rules_are_measured_against_each_other` — asserts the one
  direction that must always hold (**the conservative rule may never under-approximate the precise
  one**), and pins that they still differ somewhere. If they stop differing, one has silently become
  the other.
- `a_host_resource_answers_each_drop_question_differently` — pins the variant that makes three
  questions three.

Extends AS0's established equivalence-test pattern rather than inventing an approach.

---

## 4. What AS4 still owes

| Item | State |
| --- | --- |
| 1 — one meaning and authority per property | drop rule done; reference rule pinned by AS0; `is_copy` already consolidated |
| 2 — near-neighbours named apart | **done for the drop rule** |
| 3 — a new type variant forces every authority to be revisited | **not started** — needs an exhaustiveness obligation, not a test |
| 4 — resource/iterator/reference/generic-drop/partial-move adversaries across engines | **not started** |
| 5 — behavioural corrections get their own decision record | none required so far: this change is naming and extraction, no behaviour |

**RB0's Q1 (iterator drop-glue asymmetry) and Q2 (`FnPtr` reference disagreement) remain open and
untouched.** Q2 is pinned by AS0's matrix; neither may be resolved by a behavioural CD without its
own evidence.
