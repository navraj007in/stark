# Gate 2 exit — Core semantic checker

Gate 2 closed on 2026-07-15. `starkc check` and the terminal IDE now run the
same pipeline:

```text
Source → Tokens → AST → resolved HIR → type tables → ownership/borrow checks
```

## Milestone evidence

### M2.1 — name resolution and HIR

- All semantic passes consume arena-backed HIR IDs rather than parser AST.
- Lexical scopes, parameters, `self`, generic parameters, shadowing, module
  paths, inline modules, `use` groups/globs/aliases, and item visibility are
  resolved before type checking.
- Core prelude names are represented explicitly (`Copy`, `Drop`, `Eq`, `Ord`,
  `Num`, collection/result types, constructors, and compiler builtins).
- Undefined, duplicate, ambiguous, and private names produce E02xx
  diagnostics with source spans.

### M2.2 — type and control-flow checking

- Local inference uses unification with occurs checks and `!` coercion.
- Calls, operators, casts, arrays/slices, tuples, structs, enums, patterns,
  field/index access, ranges, and `?` are checked.
- `if`, `match`, and value-producing `loop` expressions unify branch types;
  Boolean and enum matches are checked for exhaustiveness.
- Place mutability, assignment, immutable deferred initialization, and
  branch-sensitive definite assignment are enforced.
- Bare unsized values and reference fields are rejected.
- Missing return paths are errors and unreachable statements produce W0005.

### M2.3 — generics and traits

- Generic functions, structs, enums, and impl blocks retain and instantiate
  their type arguments, with arity and bound checking.
- Trait implementation completeness, associated types, associated equality
  bindings, `T::Item`/`Self::Item`, orphan rules, overlap, and Copy/Drop
  coherence are validated.
- Trait method signatures must match their declarations. Inherent and trait
  methods support receiver auto-borrow/auto-deref; ambiguous selections are
  rejected. Fully-qualified trait calls are supported.
- `Eq`, `Ord`, and `Num` operator requirements are enforced for generic code.

### M2.4 — ownership and borrows

- Non-Copy reads move values; repeated use, whole-value use after partial
  move, branch/loop moves, and invalid moves from Drop types are diagnosed.
- Field-sensitive partial moves and reinitialization restore the appropriate
  drop flags.
- Shared/mutable lexical borrows, mutation/read conflicts, method receiver
  borrows, dereference places, and statement-scoped temporary borrows are
  checked.
- Borrow taint propagates through tuples, arrays, generic nominal/core types,
  constructors, and user-function returns. References to stack locals cannot
  escape through a wrapper such as `Option<&T>`.

### M2.5 — conformance

- The manifest-driven suite covers all 121 extracted spec fixtures and
  enforces every recorded semantic E-code.
- `tests/gate2-valid/` contains 26 standalone valid programs (the gate asks
  for approximately 20), exercised end-to-end by `tests/gate2_valid.rs`.
- Unit tests cover the negative safety rules and diagnostic codes; robustness
  tests retain the no-panic/no-hang floor.
- Both `starkc check` and F9 compilation in `starkide` treat warnings as
  non-fatal and semantic errors as failures.

## Reproduce the exit check

```bash
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-targets --all-features
cargo build --release --all-targets
cargo run -- check tests/gate2-valid/25_associated_type_bindings.stark
```

## Intentional deferrals

These are the deferrals named by the Gate 2 plan, not incomplete exit work:

- match usefulness/redundant-arm warnings beyond exhaustiveness;
- full checking of trait default-method bodies;
- complex public re-export graphs beyond the module/import cases required by
  the conformance corpus.

Execution, runtime drop order, standard-library behavior, and `starkc run`
belong to Gate 3.
