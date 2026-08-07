# WP-CALLABLE-USE-TOTAL — one record per callable use

**Status:** DRAFT for approval. AS3 work item 1 requires this document *before* implementation.
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

```rust
/// Everything the checker decided about one callable use. One per accepted invocation, keyed by
/// the expression that invokes it.
pub struct CallableUse {
    /// WHICH callable runs. Not a name to look up — the selection itself.
    pub callee: CalleeIdentity,
    /// The body, so a consumer can pair this with `callable_types`' signature (A3b).
    pub body: BlockId,
    /// The generic environment, explicitly EMPTY rather than absent for a non-generic use.
    pub environment: Vec<(GenericBinder, Ty)>,
    /// How the receiver was adjusted to make the call type-check (TYPE-METHOD-002).
    pub receiver: ReceiverAdjustment,
    /// Argument and result types, grounded.
    pub signature: CallableSigTy,
    /// Why this callable and not another — the audit trail, and what makes a wrong selection a
    /// diagnosable defect rather than a silent one.
    pub provenance: DispatchProvenance,
}

pub enum CalleeIdentity {
    /// A free function or associated function: the item runs directly.
    Item(ItemId),
    /// A method resolved to a specific impl's specific function item.
    ImplMethod { impl_item: ItemId, fn_item: ItemId },
    /// A trait default body, with the impl it was reached through.
    TraitDefault { trait_item: ItemId, fn_item: ItemId, impl_item: Option<ItemId> },
    /// A function value; the identity is in the value, not the call site.
    FunctionValue,
}

pub enum ReceiverAdjustment {
    None,
    ByValue,
    Shared { derefs: u8 },
    Exclusive { derefs: u8 },
}

pub enum DispatchProvenance {
    /// `f(x)` — a path resolved to an item.
    Direct,
    /// `x.m()` — inherent method resolution.
    Inherent,
    /// `x.m()` where `m` came from a trait impl.
    TraitImpl { trait_item: ItemId },
    /// `T::m()` / `<T as Tr>::m()`.
    Qualified { trait_item: Option<ItemId> },
    /// A generic parameter's bound supplied the signature. This is what `bound_trait_calls`
    /// carries today, and why it exists.
    Bound { trait_item: ItemId },
    /// A compiler-known trait operation: `==`, `<`, `for`, `{}` formatting.
    CoreTrait { core: CoreTrait },
    /// Calling a function value.
    FunctionValue,
}
```

`CoreTrait` provenance is the load-bearing addition. Equality, ordering, iteration and display are
dispatched by the *language*, not by a written call, and are precisely the four families where both
engines currently re-select with no filter at all.

---

## 4. The rule

> **The checker publishes exactly one `CallableUse` per accepted invocation. Execution consumes it.
> Neither engine reconstructs selection.**

Stated as three obligations:

1. **Totality.** Every accepted invocation has a record. An execution site that finds none is an
   internal compiler error, not a fallback to scanning.
2. **Uniqueness.** No invocation has two. Duplicates fail an invariant test.
3. **No second algorithm.** `Interpreter::find_method` and `FnLowerer::find_impl_fn` are deleted,
   not filtered. While either exists, the hint-threading pattern remains available and will be used
   again.

---

## 5. Boundaries

The ten families group by what they share, and each group is a semantic boundary with its own
evidence. Within a group the work is the same shape; across groups it is not.

### Boundary 1 — the families that already consume selection
**free calls, function values.**
These are the proof the mechanism works. Work: widen `CallableInstantiation` to `CallableUse` and
have these two consume the new fields. Nothing changes about which body runs.
*Evidence: `cargo test --lib`, the totality/uniqueness invariant test, `three_engine_differential`.*

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

1. Every executable user-callable use has exactly one `CallableUse`; duplicates and omissions fail an
   invariant test over a real multi-package program.
2. `Interpreter::find_method` and `FnLowerer::find_impl_fn` no longer exist.
3. Both engines install the checker-selected generic environment at every entry point that has one —
   measured against `AS0-CALLABLE-EXECUTION-SITE-INVENTORY.md`'s 21, not asserted.
4. The frozen corpus and all engine comparisons remain green.
5. Any checker/engine disagreement found on the way is recorded as its own DEV entry with a
   fails-first test, and not silently normalised.
