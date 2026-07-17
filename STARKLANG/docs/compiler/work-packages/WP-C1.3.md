# WP-C1.3 — Types, Generics, Traits, and Operator Semantics

Gate: C1 (Core v1 Conformance Closure). Extracted verbatim in scope from
`STARKLANG/docs/compiler/COMPILER-ROADMAP.md`.

## Scope

Audit and test:

- local inference boundaries;
- explicit function return types;
- generic substitution;
- associated types and bindings;
- trait bounds;
- default methods;
- inherent versus trait method selection;
- orphan and overlap rules;
- method auto-borrow and one-level auto-deref;
- `From`/`Into`/`TryFrom` rules where normative;
- equality, ordering, arithmetic, indexing, and other operator-to-trait semantics;
- user-defined implementations versus compiler builtins;
- diagnostics identifying both sides of conflicting implementations.

The equality/trait-dispatch question must be closed here by either:

1. implementing the normative dispatch semantics consistently in checking and execution; or
2. correcting an unambiguous spec defect through the spec-bug protocol.

A hidden interpreter-only structural equality rule is not accepted as an undocumented third
behaviour.

## Inherited findings (owned by this WP per prior WPs' disposition)

- **DEV-008** — `==`/`!=` (`BinOp::Eq`/`Ne` in `interp.rs`) are pure structural equality on the
  interpreter's internal `Value` enum via Rust's derived `PartialEq` — there is no dispatch
  through a user's `Eq` trait implementation at runtime. `Eq` as a trait bound is currently a
  type-checker-only concept. This WP must close it: determine whether Core v1 actually permits
  hand-written `impl Eq for T`, then either implement real dispatch or correct an unambiguous
  spec defect. Do not leave a third, undocumented behavior in place.
- **DEV-013** — `STD-004` (standard traits: Clone/Hash/Default/Display/Error/Iterator)
  exhaustiveness is unresolved. `typecheck.rs:5598-5637` recognizes Clone/Hash/Display/Default/
  Iterator as compiler-known bounds; `grep -n '"Error"' typecheck.rs` found no matches during
  WP-C0.3/C0.4 — `Error` trait bound recognition may be genuinely absent. Also unresolved:
  whether users can hand-write `impl <StdTrait> for T` versus only trigger compiler-builtin
  behavior when the bound is satisfied structurally (the same question DEV-008 raises for `Eq`,
  likely generalizing to this whole trait family).

## Scope-control answers (Charter §2.6)

- **Exact compiler claim tested:** that `typecheck.rs` implements Core v1's type system, generic
  substitution, trait resolution/coherence, and operator-to-trait desugaring correctly and
  consistently with what `interp.rs` actually executes at runtime.
- **Later mechanism that would make the result impossible to attribute:** conflating a type-level
  bound-satisfaction check with runtime dispatch behavior — DEV-008 is exactly this trap, and any
  fix must keep the two stages honest about what each one actually verifies/executes.
- **Strongest existing comparator:** old Gate 2 (M2.2-M2.3) evidence plus existing generics/
  traits tests; strengthened, not replaced.
- **Negative result that would stop this WP/gate:** a type-checker bound accepted with no
  matching runtime behavior (the DEV-008 pattern generalized), or a coherence rule that lets two
  conflicting impls both compile silently.

## Input packet

`COMPILER-CHARTER.md` + `COMPILER-STATE.md` + this file + `03-Type-System.md`,
`04-Semantic-Analysis.md` (trait/coherence sections), `06-Standard-Library.md` (trait
definitions), `typecheck.rs`, `interp.rs` (operator/trait-dispatch code paths),
`tests/gate4a_prelude_traits.rs`.

## Execution log

**Closed 2026-07-17.** Full dated evidence, files touched, and decisions:
`COMPILER-STATE.md` session record `### WP-C1.3`.

Summary: both inherited findings closed with real fixes and regression tests, not just
documentation. DEV-008 (equality dispatch) resolved in favor of implementing normative `Eq::eq`
dispatch, per unambiguous spec text; a companion `Ty::Core` bound-checking gap found and fixed
along the way (`Option`/`Result`/`Vec`/`Box` equality was unconditionally rejected). DEV-013
(STD-004 exhaustiveness) closed after discovering `.clone()` was completely non-functional on
every compiler-builtin type and trait default method bodies were never used as a fallback —
both real, previously-unknown, now-fixed bugs. Two more bugs in the same trait-family
investigation (`Display`/`Hash` missing as builtin methods; `From::from` associated-function
resolution broken) were found and deliberately recorded rather than fixed, to keep this WP's
scope bounded after four substantial fixes already landed. `cargo test --workspace --all-targets
--all-features`: 418 passed / 0 failed / 2 ignored (up from 410/0/2). `cargo fmt --check` clean.
Clippy clean on all touched files.

Not exhaustively audited this WP (spot-checked only, against existing tests): local inference
boundaries, generic substitution, associated types, inherent-vs-trait selection, orphan/overlap
rules, auto-borrow/auto-deref, conflicting-impl diagnostics. Recorded as a known gap, not
silently claimed as thorough.

Next: WP-C1.4 (ownership, borrowing, lifetimes, and drop checking).
