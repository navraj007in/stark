# AS7 — the approved decomposition and its dependency DAG

**Owner decision, 2026-08-08. Frozen for the packet.** Recorded here because AS7's exit criterion 2
requires the dependency direction to be *documented* and cycle-free, and because the rule below is
what stops a marathon from re-litigating its own design.

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
