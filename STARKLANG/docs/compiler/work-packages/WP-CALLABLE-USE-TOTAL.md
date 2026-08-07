# WP-CALLABLE-USE-TOTAL — one record per callable use

**Status:** **CORRECTED 2026-08-07** after owner review returned the first draft (`f8c7e3e`) NO-GO.
AS3 work item 1 requires this document *before* implementation; §3 records what the first draft got
wrong and why the compiler contradicts it.
**Owning packet:** AS3, Sprint 3, `WP-ARCHITECTURE-STABILIZATION.md`.
**Opened:** 2026-08-07.
**Scope discipline:** this changes *where an answer comes from*, not *what the answer is*. Any
change to which body a call runs is a behavioural change needing its own CD.

---

## 1. The problem, measured

`AS0-CALLABLE-EXECUTION-SITE-INVENTORY.md` established the surface:

```text
21 production entry points into a user body, across 17 families
 3 selection algorithms:  typecheck (the authority)
                          Interpreter::find_method   6 call sites
                          FnLowerer::find_impl_fn    8 call sites
 2 entry points install the checker-selected generic environment
```

The type checker decides which body a call runs, and publishes part of that decision. Both engines
then decide again, independently, by scanning `hir.items` for an impl on the nominal declaring a
method with the right name.

They agree today. They agree because two independently written scans over the same data reach the
same answer, not because either is told — and the one recorded instance of them *not* agreeing
(DEV-BOUND-TRAIT-IDENTITY) was repaired by threading a hint into both scans rather than by removing
the second and third algorithms.

**`bound_trait_calls` is the shape of the problem, not the solution.** It is a narrowing hint,
applied at 1 of `find_impl_fn`'s 8 call sites. `eq`, `cmp`, `next` and all three `fmt` sites pass
`None`.

---

## 2. What is published today

| Table | Keyed by | Carries |
| --- | --- | --- |
| `callable_instantiations: HashMap<ExprId, CallableInstantiation>` | the call/use expression | `body: BlockId`, `bindings: Vec<(GenericBinder, Ty)>` |
| `bound_trait_calls: HashMap<ExprId, Res>` | the call expression | which trait a bound-dispatched call came from |
| `callable_types: HashMap<BlockId, CallableSigTy>` | the *body* | the signature (A3b; publication only) |

Recorded at **four** points in the checker: `instantiate_sig` (free calls and fn values),
`associated_fn_type` (associated functions), and `resolve_method` ×2 (methods).

### Why that is not enough

1. **It is keyed by expression, and carries a body — but not the callable's identity.** The
   interpreter still has to find *which* `ItemId` to run; `body` lets it check agreement after the
   fact, which `capture_function_value` does, but not select.
2. **Six families never reach a recording point at all**: equality, ordering, iteration, display,
   qualified calls, trait defaults. Their dispatch is decided by the checker — `bound_trait_calls`
   proves the checker knows — and then discarded.
3. **Receiver adjustment is not published.** TYPE-METHOD-002's auto-borrow/auto-deref decision is
   re-derived by each engine.

---

## 3. The record

**Corrected 2026-08-07 after owner review of the first draft.** That draft assumed three things the
compiler contradicts, each verified below. They are recorded rather than quietly fixed, because two
of them are the kind of assumption that would have been encoded into production tables during
Boundary 1 and only surfaced at Boundary 4.

| Draft assumed | The compiler says |
| --- | --- |
| methods and associated functions have their own `ItemId` | they do not. `ImplItem::Fn { vis, def: FnDef }` and `TraitItem::Method { sig, body }` carry no id; members are positional inside the impl's or trait's `Vec`. This is exactly why A3b chose `BlockId` as executable identity |
| one `ExprId` ↔ one callable use | false for `Display`. `display_deep` recurses through tuples, arrays, `Option`, `Result` and slots, invoking a nominal's `Display::fmt` at any depth — `println((a, b))` is one argument expression and two `fmt` bodies, and a vector is one expression and *n* |
| every use has a statically known body and environment | false for function values. DEV-178 established that the value carries the item and the bindings it was created with, because the call site's `Ty::Fn` cannot reconstruct which instantiation produced it |

### 3.1 Declaration identity — what the HIR actually has

```rust
/// WHICH declaration was selected, expressed in ids the HIR possesses.
pub enum CallableDeclId {
    /// A free function or a `const`-position item: it has its own `ItemId`.
    Item(ItemId),
    /// A member of an impl, by position in that impl's `items`.
    ImplMember { impl_item: ItemId, member: u32 },
    /// A member of a trait, by position — the declaration site of a default body.
    TraitMember { trait_item: ItemId, member: u32 },
}
```

`body: BlockId` remains the **executable** identity, as A3b established. `CallableDeclId` is the
*declaration* identity, which is what provenance and diagnostics need and what `BlockId` alone
cannot express. For a trait default reached through an impl, the trait member identifies the
declaration and the impl travels as provenance — not as a second identity.

No method `ItemId` is fabricated to make the model fit.

### 3.2 Keying — one record per static use site, not per expression

> **A `CallableUse` is one *static semantic callable-use site*. It is not one per runtime invocation,
> and not necessarily one per `ExprId`.**

A static use may execute zero, one, or thousands of times. `println(vec)` is one use site and *n*
invocations; `println((a, b))` is one expression with two distinct use sites.

```rust
pub struct CallableUseId(u32);

pub struct TypeTables {
    /// Every published use, indexed by `CallableUseId`.
    pub callable_uses: Vec<CallableUse>,
    /// The uses an expression gives rise to — zero, one, or many.
    pub callable_uses_by_expr: HashMap<ExprId, Vec<CallableUseId>>,
    // …
}
```

Equality has the same shape for a weaker reason: `language_equal` can dispatch `Eq::eq` from
collection lookup, reached with runtime values and a span rather than a unique invoking expression.

**The rule that keeps this honest**, since a recursive renderer must still pick *which* published use
applies at a nested static type:

> An engine may choose among checker-published records using runtime or static structure. It may not
> scan the HIR and re-run method selection.

Choosing from a published set is consumption. Deciding what the set contains is selection, and that
belongs to the checker alone.

**This keying model is Boundary 1's first deliverable**, proven before any consumer migrates —
because if it is wrong, every later boundary inherits it.

### 3.3 The record

```rust
pub struct CallableUse {
    /// WHAT runs. Static for an ordinary call; deferred to the value for a function value.
    pub selection: CalleeSelection,
    /// The generic environment, on the same footing.
    pub environment: GenericEnvironment,
    /// What the CALL SITE did to the receiver (TYPE-METHOD-002 auto-borrow/auto-deref).
    pub receiver_adjustment: ReceiverAdjustment,
    /// What the SELECTED CALLABLE binds. Normally correlated with the adjustment, but a different
    /// question and a different authority — AS3's contract names both, and AS4/concurrency will ask
    /// about the binding side specifically.
    pub receiver_binding: ReceiverBinding,
    /// This use's signature.
    ///
    /// **Inference-grounded, not fully concrete**: no surviving `Ty::Infer` or `Ty::Error`. A
    /// caller's own `Ty::Param` may remain and is concretised against the active caller
    /// environment — the same rule `CallableInstantiation` already documents, and the reason
    /// `callable_types` is body-parametric.
    pub signature: CallableSigTy,
    /// Why this callable and not another.
    pub provenance: DispatchProvenance,
}

pub enum CalleeSelection {
    /// The checker selected a specific declaration and body.
    Static { declaration: CallableDeclId, body: BlockId },
    /// A function value. The body comes from the value, not from this site (DEV-178).
    FunctionValue,
}

pub enum GenericEnvironment {
    /// Explicitly empty for a non-generic static call — an empty environment, never an absent one.
    Static(Vec<(GenericBinder, Ty)>),
    /// Fixed at coercion and carried by the value (DEV-178).
    FromFunctionValue,
}

pub enum ReceiverAdjustment {
    None,
    ByValue,
    Shared { derefs: u8 },
    Exclusive { derefs: u8 },
}

pub enum ReceiverBinding {
    None,
    ByValue,
    Shared,
    Exclusive,
}

pub enum DispatchProvenance {
    Direct,
    Inherent,
    TraitImpl { trait_item: ItemId },
    Qualified { trait_item: Option<ItemId> },
    /// A generic parameter's bound supplied the signature — what `bound_trait_calls` carries today.
    Bound { trait_item: ItemId },
    /// A compiler-known trait operation: `==`, `<`, `for`, `{}` formatting.
    CoreTrait { core: CoreTrait },
    FunctionValue,
}
```

`CoreTrait` provenance is the load-bearing addition. Equality, ordering, iteration and display are
dispatched by the *language* rather than by a written call, and they are exactly the four families
where both engines currently re-select with no filter at all.

### 3.4 The two tables must not become competing signature authorities

`callable_types[body]` is the body's *parametric* signature (A3b). `CallableUse::signature` is this
use's *instantiated* one. They are different views and must stay consistent:

> **Invariant.** For a `Static` selection, substituting `environment` into `callable_types[body]`
> yields `signature`. Enforced by an invariant test, not by convention.

Without it, two tables answer "what is this callable's signature" and the divergence hazard is back
in a new place — which is the entire pattern this programme exists to remove.

---

## 4. The rule

> **The checker publishes exactly one `CallableUse` per accepted static use site. Execution consumes
> it. Neither engine reconstructs selection.**

Three obligations:

1. **Totality.** Every accepted static use site has a record. An execution site that finds none is an
   internal compiler error, not a fallback to scanning.
2. **Uniqueness.** No use site has two. Duplicates fail an invariant test.
3. **No second algorithm.** `Interpreter::find_method` and `FnLowerer::find_impl_fn` are deleted, not
   filtered. While either exists, the hint-threading pattern remains available and will be used
   again.

---

## 5. Boundaries

The ten families group by what they share, and each group is a semantic boundary with its own
evidence. Within a group the work is the same shape; across groups it is not.

### Boundary 1 — the keying model, and the families that already consume selection
**free calls, function values.**

**First deliverable is the keying model itself**, not a migration: `CallableUseId`,
`callable_uses`, `callable_uses_by_expr`, and a test proving one expression can carry zero, one or
many uses. If that is wrong every later boundary inherits it, which is exactly the failure this
document was sent back to avoid.

Then the two families that already consume selection — the proof the mechanism works. Function values
are here deliberately rather than in a later group: they are the only case of
`CalleeSelection::FunctionValue` + `GenericEnvironment::FromFunctionValue`, so the dynamic half of the
model is exercised at the start rather than discovered at the end.

*Evidence: `cargo test --lib`, the totality/uniqueness invariant test, the signature-consistency
invariant of §3.4, and `three_engine_differential`.*

### Boundary 2 — named dispatch
**methods, associated functions, qualified calls, trait defaults.**
All reach a body by name through `find_method`/`find_impl_fn`. Work: publish `CalleeIdentity` at the
checker's four existing recording points plus the qualified paths, and make both engines consume it.
*Evidence: as above plus `c62b_*`, `cross_package_generics`, `dev175_dependency_alias_scope`.*

### Boundary 3 — operator dispatch
**equality, ordering.**
Dispatched by `==`/`<` rather than by a call. Neither engine passes a filter today. Work: publish
`CoreTrait` provenance for the operator paths and consume it.
*Evidence: as above plus `copy_canon_matrix`, `c62d_operator_coretrait`, `adversarial_*`.*

### Boundary 4 — protocol dispatch
**iteration, display.**
Same shape as boundary 3, driven by `for` and `{}`. `Display` has three separate `fmt` selection
sites in MIR alone. Work: publish and consume; then **delete `find_method` and `find_impl_fn`**,
which is only possible once nothing calls them.
*Evidence: as above plus `c63c_iterators`, the Display suites, `c6_metamorphic`, and the full
differential set.*

The deletion in boundary 4 is the packet's real exit. Until then the second algorithms exist and the
invariant is a convention.

---

## 6. What this does NOT do

- **It does not change which body any call runs.** If a boundary's evidence shows the checker and an
  engine disagreeing, that is a defect found — recorded under its own DEV number and repaired with a
  fails-first test, not absorbed.
- **It does not resolve RB0's Q1 or Q2.** Those are type-property questions (AS4), not callable
  selection.
- **It does not close DEV-121.** AS3 work item 6 requires a class-level evidence statement for the
  typed-mutation boundaries; that is separate and comes after.
- **It does not resume `WP-VALUE-REP-TOTAL` A4.** AS3 work item 5 gates that on callable-use
  exactness passing, which is this packet's exit, not its content.

---

## 7. Exit criteria

1. Every executable user-callable **static use site** has exactly one `CallableUse`; duplicates and
   omissions fail an invariant test over a real multi-package program. A recursive `Display` over a
   composite is many use sites and one expression, and the test asserts that shape rather than
   assuming `ExprId` is the key.
2. Substituting a use's `environment` into `callable_types[body]` yields its `signature`, so the two
   tables cannot become competing signature authorities (§3.4).
3. `Interpreter::find_method` and `FnLowerer::find_impl_fn` no longer exist.
4. Both engines install the checker-selected generic environment at every entry point that has one —
   measured against `AS0-CALLABLE-EXECUTION-SITE-INVENTORY.md`'s 21, not asserted.
5. The frozen corpus and all engine comparisons remain green.
6. Any checker/engine disagreement found on the way is recorded as its own DEV entry with a
   fails-first test, and not silently normalised.


---

## 8. Three binding times — the model corrected again (2026-08-07)

Boundary 4's Display characterization (`AS3-DISPLAY-CHARACTERIZATION.md`) found a category
`CalleeSelection` could not represent, and owner review established it is **not a Display corner**.

### The premise correction

The first reading was that Boundary 2 already handled bound dispatch and Display was an exception.
It does not. `resolve_method`'s bound branch records `bound_trait_calls`, calls
`check_bound_method_call`, and **returns** — before Boundary 2's publication is reached. Verified in
the source. So

```stark
fn f<T: Speak>(x: T) { x.speak(); }
```

publishes no `CallableUse` at all, and never did. AS3 found a **missing third category of callable
selection**, not an unusual formatting case.

### The three binding times

```text
Static          body known during typecheck
Bound           trait/member known during typecheck;
                body known when `Self` becomes concrete
FunctionValue   body and environment carried by the runtime value
```

Forcing the middle case into either end hides something true. Calling it `Static` because MIR
eventually monomorphises would assert the checker knew the body at the semantic use site; it did
not. Calling it `FunctionValue` would assert a value carries the target; none does.

### What `Bound` means, precisely

> **A checker-published dispatch obligation whose declaration identity is fixed, but whose
> executable target is resolved by the compiler's single bound-specialisation authority when its
> `Self` type becomes concrete.**

Not *"the engines may now find whichever impl matches"*. That reading would give the existing scans
a respectable name and leave the architecture where it was:

```text
                     Typecheck
                        │
                        ▼
               CallableUse::Bound
        trait + member + parametric Self
                        │
            ┌───────────┴───────────┐
      HIR generic frame       MIR monomorphisation
            └───────────┬───────────┘
                        ▼
             ONE bound specialiser
                        ▼
          concrete declaration / body / environment
```

Both engines consume that authority's result at different *times*; neither implements matching.

### Identity: `hir::BoundTrait`

`DispatchProvenance::Bound { trait_item: ItemId }` could not represent `T: Display`, because
`Display` is a `CoreTrait` with no trait `ItemId`. Both selection and provenance now use
`hir::BoundTrait { User(ItemId), Core(CoreTrait) }` — which **already existed**, because
`BoundMethod::{User, Core}` needed exactly this distinction. Selection, provenance and the rest of
the compiler now speak one identity language.

### `GenericEnvironment::FromBoundSelection`

A bound use's callee environment does not exist yet, because the callee is not selected yet. The
specialiser produces body and environment **atomically**; choosing the body in one place and
reconstructing the environment in another is how DEV-176 happened, and would happen again.

### §3.4's invariant is postponed, not weakened

For `Static` it stands unchanged. For `Bound`:

```text
specialize(use, caller_environment) → body + environment + signature
then  substitute(environment, callable_types[body]) == specialized(signature)
```

Worked example. At the declaration of `fn show<T: Display>(x: T)`:

```text
selection = Bound { trait_: Core(Display), member: "fmt", self_ty: T }
signature = (&T) -> String
```

At `show::<A>`: `T → A`, `Display + A → impl Display for A`, body `→ A::fmt`,
environment `→ Self = A`, signature `→ (&A) -> String`. `show::<W<Int32>>` specialises
independently.

### Boundary 2's status, narrowed

```text
Boundary 2 — concrete named dispatch    COMPLETE
Boundary 2 — generic bound dispatch     MOVED TO BOUNDARY 4
```

Not a reason to unwind Boundary 2. Bound dispatch is its own binding-time family and grouping it
under ordinary named dispatch was the error. The boundaries read better for it:

```text
Boundary 1  direct / value
Boundary 2  early-bound named
Boundary 3  operator
Boundary 4  protocol + LATE-BOUND trait dispatch
```

### Implementation order

1. **This commit** — model and record; no behaviour change.
2. `CalleeSelection::Bound` published for an ordinary explicit bound call
   (`fn f<T: Speak>(x: T) { x.speak(); }`), **before Display**, so the general mechanism is proved
   independently of recursive formatting.
3. The one bound-specialisation authority: user trait, core trait, generic impl, same-named methods
   on different traits, concrete core receiver.
4. Both engines onto it — HIR's bound path stops calling `find_method`, MIR's stops calling
   `find_impl_fn`.
5. `DisplayPath` publication: static nominal → `Static`, parametric `T: Display` → `Bound`, recurse
   composites, STOP at a nominal whose `fmt` owns rendering.
6. HIR and MIR Display, plus the remaining MIR Iterator site.
7. Delete `find_method` and `find_impl_fn`.

The crucial negative control at step 4: **HIR and MIR must obtain the same resolved
`body + environment` from the shared specialiser.**

---

## 9. Boundary 4 outcome (2026-08-07): the fallbacks are deleted, and what that cost

The exit condition for this packet is not "the checker publishes selections" — it is "no consumer
re-derives one". Those are different, and the gap between them is where the work turned out to be.

### The census

The MIR method fallback was instrumented rather than reasoned about, and the differential,
operator, iterator, bound-identity and Display suites run:

| Stage | `find_impl_fn` fires |
| --- | ---: |
| before consuming any selection | ~60 |
| after consuming `Static` (`static_selected_key`) | 2 |
| after DEV-190 published `self.m()` in trait defaults | 0 |

Only then was the arm deleted. Deleting it — rather than annotating it as unreached — immediately
failed `over_acceptance_audit`, which exercises an operator on a bounded parameter (DEV-191). That
suite was outside the two the earlier mutation evidence covered.

### Five defects the fallbacks were hiding

| | Defect | Effect |
| --- | --- | --- |
| DEV-188 | trait-method generics dropped at bound call sites | such methods **uncallable** through a bound |
| DEV-189 | MIR passed the bare nominal head | generic impls never used the shared authority |
| DEV-190 | `self.m()` in a trait default published nothing | both engines scanned by name |
| DEV-191 | operators on a bounded parameter published nothing | same |
| DEV-192 | `==` through an `Eq` bound fell through to structural equality | **wrong answers** from the reference engine |

DEV-192 is the one that justifies the whole approach. It is not a missing feature; it is the HIR
oracle printing `false` where the program means `true`, for any type whose `Eq` disagrees with
field-wise comparison. Every fixture in the suite had an `eq` that agreed with structural equality,
so a differential suite could not see it — two algorithms that coincide on all available inputs are
indistinguishable by differential testing, however many engines it compares.

### What a fallback actually costs

Each of these was live for as long as a fallback made it invisible. A fallback converts a missing
publication from a build failure into a silent second algorithm, and a second algorithm is exactly
what this packet exists to remove. "Verified unreached" is a statement about the suites that were
run, not about the program space — and it is the annotation that let DEV-191 sit behind a comment
claiming the opposite.

### Still open

`find_method` (interpreter) and `find_impl_fn` (MIR) survive on the **Display** and **Iterator**
paths, which select by name and need `DisplayPath` publication first. Those are Boundary 4's
remaining work; the method and operator paths are done.
