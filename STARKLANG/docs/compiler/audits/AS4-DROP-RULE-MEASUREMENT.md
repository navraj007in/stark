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

**Item 1 — the MEANINGS are separated; authority consolidation is still OPEN.** Reviewer correction, 2026-08-07: calling item 1 complete here was wrong, and weakened an otherwise disciplined evidence trail. The code still contains **two authorities for the same question** — see §5.

 `verify::mir_needs_drop` was a method on
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
| drop semantic decomposition | **DONE** |
| drop near-neighbour naming | **DONE** |
| precise-drop equivalence | **OPEN** — see §5 |
| precise-drop authority merge | **BLOCKED ON EQUIVALENCE** |
| 1 — one meaning and authority per property | reference rule pinned by AS0; `is_copy` already consolidated; **drop NOT consolidated** |
| 2 — near-neighbours named apart | **done for the drop rule** |
| 3 — a new type variant forces every authority to be revisited | **not started** — needs an exhaustiveness obligation, not a test |
| 4 — resource/iterator/reference/generic-drop/partial-move adversaries across engines | **not started** |
| 5 — behavioural corrections get their own decision record | none required so far: this change is naming and extraction, no behaviour |

**RB0's Q1 (iterator drop-glue asymmetry) and Q2 (`FnPtr` reference disagreement) remain open and
untouched.** Q2 is pinned by AS0's matrix; neither may be resolved by a behavioural CD without its
own evidence.


---

## 5. OPEN: the precise rule still has two authorities, and they disagree

Reviewer finding on `0257320`. The first row of §1's table names two implementations of **one**
question, and this packet's own measurement never compared them to each other:

```text
lower::ty_requires_drop_glue      vs      verify::requires_drop_glue
```

The harness compares `verify` conservative against `verify` precise. It establishes

```text
verify conservative != verify precise      (measured)
```

but not

```text
lower precise == verify precise            (NOT measured)
```

and the second is the equivalence AS4 actually needs before the property can be called consolidated.

### There is already a visible disagreement family

Lowering classifies these as **not** requiring glue:

```text
CharsIter  SplitIter  ValuesIter  MapIter  FilterIter  Random  IOError  File
Range  RangeInclusive  Ordering  Option  Result  String(CoreType)
```

while the verifier's precise authority opens with:

```rust
MirTy::String | MirTy::Core(..) => true,
```

so **every** `MirTy::Core` requires glue there. The two answers differ for every variant in that
list that is reachable as a `MirTy::Core`.

That matters more than the `may_need_drop` disagreement §1 records, because those two are *supposed*
to differ — these two claim to answer the same question.

### Three possible readings, and none may be assumed

1. Only the iterator/Core cases disagree and one side is demonstrably wrong → a behavioural DEV/CD
   **before** consolidation.
2. Producer and verifier deliberately ask subtly different questions → §1's "three questions"
   becomes four.
3. The verifier's `Core(..) => true` is an old conservative shortcut inside a predicate that is
   otherwise precise → characterise, then consolidate.

Reading 3 is what the surrounding comments suggest. **It is not adopted.** AS3's repeated lesson was
that promoting a plausible reading into a diagnosis before measuring is how two wrong causes reached
a permanent record; this document states the hypothesis and does not act on it.

### Next step, before items 3 and 4

```text
for every MirTy sample:
    lower::ty_requires_drop_glue
    verify::requires_drop_glue
record every disagreement
```

The asymmetry is itself evidence: lowering's predicate needs a `FnLowerer`, while the verifier's now
needs only a `TypeContext` — which already carries `struct_fields`, `enum_variants`, `drop_impls`
and `host_resource_closes`. A shared authority is plausible, but lowering is a **producer** of parts
of `TypeContext`, so producer and consumer cannot blindly share a table one is still building. That
is the same temporal split `Copy` already solved by sharing the structural recursion and
parameterising the fact that differs. Drop likely wants the same shape — **but that is a design to
reach after the matrix exists, not before.**

### Item 3 follows this, not the other way round

If AS4 ends with two exhaustive matches answering one property, item 3 succeeds while criterion 1
fails: adding `MirTy::NewThing` forces both to change, and the compiler still holds two authorities
free to choose different answers. The result worth having is one semantic drop-glue match failing to
compile, with all consumers inheriting the decision — while `may_need_drop` and
`has_user_destructor` fail separately because they genuinely answer different questions.
