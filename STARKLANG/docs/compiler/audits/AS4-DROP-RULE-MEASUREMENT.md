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


---

## 6. The lower-vs-verifier matrix, measured (2026-08-07)

§5's open item is closed as a **measurement**. `mir::lower::as4_drop_predicate_inventory::`
`the_two_precise_drop_rules_are_measured_against_each_other` compiles and lowers a real program —
both predicates need context, lowering's a `FnLowerer` and the verifier's a `TypeContext` — and
compares them across every `CoreType` plus the nominal shapes that exercise the recursion.

### 14 disagreements, every one in the same direction

```text
lower=false   verify=true
```

for `Core(String, Option, Result, Range, RangeInclusive, CharsIter, SplitIter, ValuesIter,
MapIter, FilterIter, Random, IOError, File, Ordering)`.

**Not one case of the reverse**, which is the direction that would be unsound: lowering claiming glue
is required where the verifier says it is not would mean lowering emits a `Drop` the verifier then
rejects. That direction is asserted against, permanently.

The measurement also **corrected the prediction in §5**, which listed 13 and missed `Core(String)`.

### 13 of the 14 are on shapes nothing constructs — and that is the useful half

`mir_ty`, the main typed-expression path, produces `MirTy::Core` for only:

```text
Box  CharsIter  HashMap  HashSet  Iter  KeysIter  Vec  VecIter
```

and `Ty::Core(Option/Result/Ordering)` lower to `MirTy::Enum(EnumRef::Core*)` instead, while
`String` lowers to `MirTy::String`. A disagreement on those rows is two authorities differing about
a shape no lowering emits. Real — two authorities free to diverge the day one becomes reachable —
but not currently reachable.

Cross-checking every `MirTy::Core(...)` construction in the tree adds `File` (built by resource
lowering, outside `mir_ty`). So the **reachable** disagreement is:

```text
Core(CharsIter)      lower=false   verify=true
Core(File)           lower=false   verify=true
```

### One consumer

The verifier's precise rule has exactly **one** production caller:

```rust
VecClear if self.requires_drop_glue(&t) => err("MIR-0016", ...)
```

So in the measured direction the verifier is strictly more restrictive: it would reject
`Vec<CharsIter>::clear()` / `Vec<File>::clear()` that lowering considers fine. Over-rejection, not
unsoundness — and only if such a `Vec` is constructible, which is **not established here**.

### Which reading holds

§5 offered three. The evidence supports **reading 3** — `Core(..) => true` is an old conservative
shortcut inside an otherwise-precise predicate — because the disagreement is one-directional,
confined to `Core`, and its only consumer is a rejection rule.

**It is recorded, not adopted.** Making the two agree changes which programs the verifier accepts,
so it is a behavioural correction and needs its own decision record (AS4 work item 5). What this
document establishes is that the decision is about **two variants and one rejection rule**, not
about fourteen disagreements across the type system.

### Status after this measurement

| | |
| --- | --- |
| drop semantic decomposition | DONE |
| drop near-neighbour naming | DONE |
| precise-drop equivalence | **MEASURED** — 14 disagreements, 2 reachable, all one-directional |
| precise-drop authority merge | **BLOCKED ON A DECISION**, no longer on evidence |


---

## 7. The disagreement is reachable, and it refuses a real program (DEV-195)

§6 established the reachable disagreement is `Core(CharsIter)` and `Core(File)`, and left open
whether such a `Vec` is constructible. **It is**, and the consequence is user-visible:

```stark
let mut v: Vec<CharsIter> = Vec::new();
v.push(s.chars());
v.clear();
```

checker accepts → interpreter prints `0` → lowering emits the fast `VecClear` → **verifier rejects,
MIR-0016**.

I had also suspected an inversion, because `Vec<String>::clear()` passes while `Vec<CharsIter>`
does not. Checking rather than asserting showed there is none: lowering emits `VecClear` **only**
when it believes the element needs no glue, so a droppable element takes a different path and never
reaches the guard. `Vec<String>` and `Vec<Vec<Int32>>` emit no `VecClear` at all.

That makes the mechanism exact: MIR-0016 guards the fast path, and the two rules put lowering on
one side of it and the verifier on the other, for precisely the types they answer differently
about.

Registered as **DEV-195 (OPEN, characterized)** with `tests/as4_vecclear_divergence.rs` pinning the
current refusal. The repair is behavioural and owes a decision record under work item 5.

**Method note.** The matrix said over-rejection was *possible*. Only running the compiler
established that a real, constructible program is refused today. AS3's lesson applied to AS4: a
measurement of predicates is not a measurement of programs.
