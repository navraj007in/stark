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


---

## 8. DEV-195 ruled; `File` becomes the blocker (2026-08-07)

**Owner ruling (CD-387): lowering is right about `CharsIter`.** A borrowed cursor requires no drop
glue, so `Vec<CharsIter>::clear()` may use the fast path and the verifier must not refuse it.
Implemented by replacing `verify::requires_drop_glue`'s `MirTy::Core(..) => true` blanket with an
**exhaustive** per-`CoreType` match: `CharsIter => false`, everything else unchanged.

Exhaustive rather than a producer census, per the same ruling — a future producer then cannot
enlarge the reachable set without using a variant whose semantics were already classified.

The matrix moved on its own, which is what a pinned measurement is for:

```text
disagreements   14 -> 13
reachable       [CharsIter, File] -> [File]
```

### `File` is not the same question wearing a different name

| | `CharsIter` | legacy Core `File` |
| --- | --- | --- |
| owns | nothing — a `&str` cursor | an `OwnedResourceHandle` |
| release | none | provider close, through MIR |
| `drop_plan::plan_for` | n/a | **`Noop`** |
| classification | hygiene | **safety-critical** |

So `verify`'s `File => true` may be an **accidental safety barrier**: lowering says no glue, the
drop plan does nothing, and only the verifier's refusal stops a fast `VecClear` from discarding open
handles. Registered as **DEV-196**.

**Reachability, measured:** `mir_ty` refuses `Core(File)` outright, so no ordinary program reaches
it; provider binding (`ResourceBinding::LegacyCore`) produces it in capability-declared builds. That
is where `Vec<File>` must be characterized — a `starkc run` probe cannot, and I established that by
running one rather than assuming it.

### The possible fourth question

`VecClear`'s guard actually asks *"can values of `T` be discarded by the fast clear without running
any language-required destruction?"* For ordinary types that is `!requires_drop_glue(T)`. Legacy
Core `File` may be the counterexample: outside ordinary type-driven drop glue, yet not
destruction-free. If the equivalence fails, AS4 has a fourth semantic question
(`is_trivially_discardable`), which would explain why `File` resists classification without either
existing predicate being wrong. **Not introduced now** — the `Vec<File>` experiment decides it.

### Status

```text
drop semantic decomposition       DONE
near-neighbour naming             DONE
precise-drop disagreement         MEASURED + REACHABLE
DEV-195 / CharsIter               DECIDED (CD-387): lowering wins
Core File classification          NEXT / SAFETY-CRITICAL (DEV-196)
precise authority merge           BLOCKED ON File
item 3 exhaustiveness             AFTER shared authority
item 4 adversaries                AFTER property decisions
```


---

## 9. The `Vec<File>` experiment (2026-08-07): the blocker dissolves

Run as §8 specified — capability-declared package, `stark build`, real `File::create`, moved into a
`Vec<File>`, cleared.

```text
Vec<File> push + clear        refused: type Core(File, []) (C4.5)
bare File bound by let        refused: same
File matched, never bound     refused: same
no File at all (control)      BUILT
```

**`Core(File)` is unlowerable from source**, with or without the capability. The `Ok(f)` binding
alone triggers it. So the second reachable row of the disagreement matrix is reachable only in the
sense that `MirTy::Core(File)` can be *constructed* — by the provider path's hand-built MIR — not in
the sense that any program produces it.

And there, destruction is **explicit**: WP-C7.8.4 closes the handle with `stark_file_close`
(`HandleConsumed`), never through drop planning. `drop_plan::plan_for(Core(File)) = Noop` is
therefore consistent with actual use rather than a gap.

### What this changes

| §8 said | §9 measured |
| --- | --- |
| `File => true` may be an accidental safety barrier | it guards a path nothing reaches — neither load-bearing nor harmful |
| a fourth predicate may be needed | no motivating case exists; **not introduced** |
| authority merge BLOCKED ON File | **not blocked by safety**; blocked only on `File` being untested |

The suspicion in §8 was the right one to hold — an owning handle with a `Noop` drop plan is exactly
the shape that should stop a consolidation — and measuring it is what showed the danger is not
currently reachable. Recorded that way round deliberately: the hypothesis was sound and the
measurement resolved it, which is different from the hypothesis having been wrong.

### Status

```text
drop semantic decomposition       DONE
near-neighbour naming             DONE
precise-drop disagreement         MEASURED + REACHABLE
DEV-195 / CharsIter               DECIDED (CD-387): lowering wins
DEV-196 / Core File               ANSWERED: unreachable from source, explicit close where used
fourth predicate                  NOT NEEDED — no motivating case
precise authority merge           UNBLOCKED on safety; needs a decision on File's classification
item 3 exhaustiveness             AFTER shared authority
item 4 adversaries                AFTER property decisions
```


---

## 10. AS4-DROP-AUTHORITY: the merge (2026-08-07)

Owner ruling: merge now, do not wait for the Core `File` → `HostResource` migration. Waiting would
make AS4 depend on an unrelated representation migration and keep alive exactly the defect class AS4
exists to remove, buying no safety now that DEV-196 has bounded the risk.

### The consolidation rule, not a winner-picking

> **If lowering cannot construct the representation, preserve the verifier's answer.**

```text
source-reachable cases     -> measured semantics decide      (CharsIter = false, CD-387)
unreachable legacy shapes  -> preserve verifier behaviour    (Core File = true)
HostResource               -> approved A11 semantics decide  (true)
```

`Core(File) = true` needs no fiat. Two independent reasons agree: it preserves current observable
verifier behaviour on a type DEV-196 proved unreachable from source, and A11's approved destination
sends `File` to `HostResource`, whose `Drop` invokes its validated close exactly once.

### Behaviour-neutral, and checkable

For every `CoreType` `mir_ty` can construct — `Box`, `CharsIter`, `HashMap`, `HashSet`, `Iter`,
`KeysIter`, `Vec`, `VecIter` — lowering and the verifier **already agreed after CD-387**. Adopting
the shared table changes lowering's answer only on representations it cannot produce, which have no
behavioural constituency. Verified before writing the code, not assumed after.

### Shape

`mir::drop_rule` owns the parts that drift; callers supply only what genuinely differs by phase,
because lowering is the **producer** of the table the verifier consumes:

```text
shared                      per-phase (DropFacts)
  MirTy recursion             has a user destructor?
  CoreType classification     instantiated struct fields
  HostResource = true         instantiated enum variants
  container recursion
```

The `CoreType` table is **shared, not a callback** — passing it out would have moved the
easiest-to-drift part into two adapters and called that consolidation.

`FnLowerer::ty_requires_drop_glue` keeps its `Result`: `nominal_instance_fields` can fail, and
swallowing that as "no fields" would answer `false` for a nominal whose shape could not be resolved
— the one answer this predicate must never give by accident. The adapter stashes the first failure
and the delegate turns it back into a `LowerError`.

Both structural copies are deleted.

### Evidence

| Check | Result |
| --- | --- |
| lower-vs-verifier matrix | **zero disagreements** — was 14 |
| DEV-195 acceptance | green |
| DEV-196 unlowerability | green |
| resource lifecycle | `a11_host_resource`, `c788_resource_lifecycle`, `c788_lifecycle_e2e`, `c784_file` green |
| engines | `mir_differential` 132, `three_engine_differential` 109 green |
| external sample suite | 39/39 |
| accepted/rejected source programs | unchanged |

No behavioural CD is due. The behavioural decision was CD-387; this is consolidation using it plus
DEV-196's evidence.

### DEV-196's pin changes meaning, deliberately

`dev196_a_vec_of_core_file_cannot_be_lowered_at_all` stays as the **migration tripwire**:

```text
today      Core(File) classified true, ordinary lowering cannot produce it
migration  source File -> HostResource, the pin FAILS, the migration explicitly
           replaces the legacy assumption, HostResource = true is inherited
```

Cleaner than leaving two drop algorithms around as an informal hedge.

### Item 3 is now largely structural

One exhaustive `MirTy` match plus one exhaustive nested `CoreType` classification, neither carrying
a property-bearing wildcard, means a new variant **breaks compilation at the semantic authority**
rather than at two copies that could answer differently. That is what "forces every applicable
authority to be revisited" should mean.

### Status

```text
drop semantic decomposition       DONE
near-neighbour naming             DONE
DEV-195 / CharsIter               DECIDED (CD-387)
DEV-196 / Core File               ANSWERED; pin kept as migration tripwire
fourth predicate                  NOT NEEDED
precise authority merge           DONE - one authority, zero disagreements
item 3 exhaustiveness             largely satisfied for the drop rule; other properties pending
item 4 adversaries                NEXT
```
