# AS7 — the approved decomposition and its dependency DAG

**Owner decision, 2026-08-08. Frozen for the packet.** Recorded here because AS7's exit criterion 2
requires the dependency direction to be *documented* and cycle-free, and because the rule below is
what stops a marathon from re-litigating its own design.

> **AMENDED 2026-08-09 (CD-393) — see "The corrected DAG" at the end.** The revised DAG below was
> declared but not enforced: the forcing test's ownership parser saw 36 of 234 methods, and the
> `traits -> convert` edge this document forbids was live in the shipped split. The amendment adds
> an **eleventh** module, `trait_contracts`, and moves three `state` edges. Everything above the
> amendment is preserved as decided on 2026-08-08.

## The module set

```text
starkc/src/typecheck/
├── mod.rs        public facade only — check/analyze entry points, re-exports, pass orchestration
├── types.rs      Ty, ExtensionTy, ModelTy, TypeVarId, TypeTables, pure structural helpers
├── state.rs      TypeChecker storage, scoped ambient contexts, snapshot/restoration machinery
├── infer.rs      substitution, resolve, unify, equality, inference vars, projection, grounding,
│                 tensor-unification bridge only
├── convert.rs    HIR TypeId -> Ty, generic argument conversion, generic parameter scopes,
│                 tensor written-type conversion bridge
├── traits.rs     trait/impl selection, bounds, associated types, Core trait contracts, coherence
├── patterns.rs   pattern typing/binding, match pattern type relations
├── body.rs       expression/statement/block checking, calls, operators, builtin call-form
│                 validation, control-flow typing
└── items.rs      item passes — fn/struct/enum/impl/trait checking, model-declaration Core side
```

## The dependency direction

```text
       types
         ^
       state
         ^
       infer
         ^
      convert
       ^    ^
  traits    patterns
       ^    ^
        body
         ^
       items
         ^
        mod
```

A module may depend on anything **reachable below it** and on nothing else.

`extensions::tensor::check` sits *beside* this hierarchy: it may depend on the lower type/service
representation, and `body`/`items` may call into it. **It must not regain control over expression
checking** — the AS6 boundary is frozen, and `as7_module_dependencies.rs` asserts `check_expr` does
not appear in it.

## Two constraints that make this a real cut rather than a cosmetic one

1. **`traits.rs` selects and validates traits; it does not check arbitrary expressions.** Any trait
   method invocation path that *evaluates arguments* stays in `body.rs`. Otherwise `traits <-> body`
   becomes the first cycle.
2. **The external surface is preserved.** `crate::typecheck::Ty`, `crate::typecheck::TypeTables`
   and `crate::typecheck::analyze(...)` keep working through `mod.rs` re-exports. AS7 is not a
   repository-wide import migration because `Ty` physically moved. The facade is part of the
   packet, not an afterthought.

## The rule for the marathon

> Once a semantic responsibility is assigned to one of these modules, **do not change the
> decomposition merely because moving a particular function is inconvenient.** Stop only if actual
> code proves the proposed dependency direction cannot express an established semantic interaction
> without a cycle.

A large function, many private helpers, or Rust visibility friction is **not** such evidence.

## Execution sequence

```text
Packet 1  architecture forcing   module DAG recorded; dependency-direction check and ambient
                                 restoration/nesting tests, both executable, BEFORE any move
Packet 2  ambient state          fix the two invariant-dependent fields, then convert all eight
                                 to scoped operations. No file splitting.
Packet 3  module shell           typecheck.rs -> typecheck/mod.rs as an IDENTITY move, facade preserved
Packet 4+ bottom-up extraction   types -> state -> infer -> convert -> traits/patterns -> body -> items
                                 After each: dependency checker green, and the old implementation
                                 GONE from its previous location.
Packet N  AS7 qualification      behaviour/diagnostic suites, dependency DAG evidence, stale-path
                                 census, public API comparison, compiler-map reconciliation,
                                 exact-head CI
```

## Exclusive ownership

AS7 runs in a dedicated worktree — `/Users/nexper/Documents/GitHub/stark-as7`, branch
`wp-arch-stability/as7-modularization`, forked from `3f18e49`. The worktree solves filesystem
isolation; this declaration solves semantic isolation. **No parallel session may change these paths
on another branch for the duration:**

```text
starkc/src/typecheck.rs
starkc/src/typecheck/**
starkc/tests/as7_*
starkc/docs/dev/compiler-map.md
AS7 work-package and audit records
starkc/src/lib.rs                        if the module transition requires it
starkc/src/extensions/tensor/check.rs    ONLY if import paths must change; its semantic
                                         boundary is frozen from AS6 and is not redesigned
```

The concern is not hypothetical: `git worktree list` shows a parallel `stark-c79` worktree live in
this repository at the time AS7 opened.

---

# CORRECTION — the stop condition fired in Packet 7 (owner ruling, 2026-08-08)

**The decomposition above was wrong at one edge, and the pre-move dependency check found it before
the split fossilised it.** The original module set is preserved exactly as frozen; this section
records what the evidence overturned and why. It was not always the plan.

## What the evidence showed

Extracting `convert.rs` produced a violation that is **not** extraction friction:

```text
convert_hir_type            (convert)  converting HashMap<K,V> must check K: Hash + Eq
   -> check_builtin_type_bounds
        -> satisfies_bound_parts       (traits)   mod.rs:9419
             -> convert_hir_type       (convert)  mod.rs:9538-9539
```

`satisfies_bound_parts` converts written types when it checks an **associated-type binding** —
`Iterator<Item = Foo>` requires converting the actual and the expected `Item` before comparing
them. Both directions are load-bearing:

```text
convert -> traits   what `HashMap<K,V>` MEANS depends on whether K satisfies a bound
traits -> convert   whether a bound holds depends on what the types written IN the bound mean
```

That is the real shape of a language with bounded generics and associated-type bindings.
`check_builtin_type_bounds` has exactly one caller, so there was nothing incidental to delete.

## The conceptual mistake

The original cut put **trait identity/selection** and **complete written-bound satisfaction** in one
`traits` module. They are different layers:

```text
traits    "Does this type satisfy this trait identity? Which impl makes that true?"
bounds    "Does it satisfy this complete written constraint, including `Item = Foo`?"
```

The modules do not need a cycle. **The missing layer was the orchestration of the two operations.**

## The revised DAG

```text
types <- state <- infer <- traits <- convert <- bounds <- patterns/body/items
```

```text
traits    may depend on  types, state, infer
          MUST NOT       convert, bounds, body, items
convert   may depend on  types, state, infer, traits
bounds    may depend on  types, state, infer, traits, convert
```

`traits` answers only whether the trait relation exists, returning a witness
(`BoundWitness::{No, Yes, Impl(ItemId)}`). **No HIR type conversion is permitted there.**
`bounds` owns the HIR-facing operation: ask `traits` for the witness; if there are associated
bindings and the witness is an impl, look up that impl's associated types, convert actual and
expected, compare.

`check_builtin_type_bounds` moves to **`convert.rs`** and stays at the same point during
conversion, so approach C's diagnostic-order and timing risk is not incurred.

## Constraints on the extraction

- **Preserve the asymmetries.** The generic-parameter branch may behave differently from the
  concrete-impl branch. **Do not improve it while extracting.** Anything that looks like a
  semantic deficiency is recorded separately unless it meets the live-defect pre-emption rule.
- **Do not touch the `assoc_projections` authority.** Its key is `(implementing nominal,
  associated-type name)`, whereas this path selects a particular trait impl *first*; those are not
  obviously equivalent when several traits expose the same associated name. Use the same selected
  impl and the same actual/expected conversion ordering the existing code uses. This decomposition
  repair is not an associated-type authority redesign.

## Rejected alternatives

```text
B  permit a convert <-> traits cycle    NO — criterion 2 becomes documentary, not enforced
C  move builtin bound validation out
   of conversion                        NO — unnecessary diagnostic/timing risk
D  merge convert + traits               NO for now — the SCC is caused by ONE higher-level
                                        associated-binding operation, so merging the whole
                                        responsibility is too coarse. D returns only if trait
                                        IDENTITY itself proves to need HIR conversion, which is
                                        not what the evidence says.
```

## The two trivial edges, ruled

```text
validate_generic_arity  -> convert.rs   validation of written generic application
dtype_from_primitive    -> delete or delegate. AS6 already established the extension-owned
                          authority `tensor_syntax::dtype_of_primitive`; do not create a second
                          mapping in the new Core modules.
```

---

## The corrected DAG (2026-08-09, CD-393)

**The `bounds` ruling was right and was applied to only half the code.** It split *complete written
bound satisfaction* out of `traits` by moving the explicit references. The **methods** stayed:
`validate_impl_rules`, `check_core_trait_impl`, `associated_fn_type`, `build_trait_impl_index`,
`contract_ty`, `declared_member_signature`, `trait_member_signature` and `assoc_binding_map` all
went on calling `convert_hir_type` from inside `traits`, so `convert <-> traits` never actually
closed. The forcing test could not contradict it because it did not parse `pub(super) fn`.

Those eight methods form a coherent layer, and it is the same distinction one level down:

```text
traits            "Does this type stand in this trait relation, and which impl says so?"
trait_contracts   "What does this WRITTEN type mean, in a trait position?"
```

`trait_contracts` sits **above** `convert` and may depend on `convert`, `traits`, `infer`, `state`,
`types`. `traits` keeps trait identity and impl selection, converts nothing, and **still holds
`core_method_signature`** — that arm table is trait identity, not conversion, so it did not move.

```text
types <- state <- infer <- traits <- convert <- {bounds, trait_contracts} <- patterns/body <- items <- mod
```

```text
traits           may depend on  types, state, infer
                 MUST NOT       convert, trait_contracts, bounds, body, items
convert          may depend on  types, state, infer, traits
bounds           may depend on  types, state, infer, traits, convert
trait_contracts  may depend on  types, state, infer, traits, convert
```

### Three `state` edges, and the classification error behind them

`state -> body`, `state -> infer` and `state -> traits` were live for a different reason, and it is
worth stating plainly because the rule was already written down and applied backwards.

Packets 9a/9b put the publication family in `state` on the reasoning that **publication writes
storage**. That classifies a function by its **effect**. The DAG classifies by **what a function
needs**: each `publish_*` resolves, instantiates or selects trait candidates *before* it writes, and
that is work `state` may not do. The decision of what to publish belongs with the caller, and those
functions are that decision.

```text
publish_* x5                  -> body
ty_to_string, format_nominal  -> infer     rendering resolves first
instantiate_sig               -> body      converts written types; its only caller is there
```

`state` fell from 896 lines to 569, which measures how much non-storage work had accumulated there.

### The enforcement rule this document now carries

Criterion 2 is satisfied by an executable check, so the check's **coverage** is part of the claim:

```text
methods owned by the map / methods in typecheck    234 / 234 as of CD-393
```

A green run over a partial map is not evidence. The forcing test prints this ratio, and any future
qualification citing it must cite the ratio with it.

