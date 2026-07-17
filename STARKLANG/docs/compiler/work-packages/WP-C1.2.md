# WP-C1.2 — Name Resolution, Modules, and Visibility

Gate: C1 (Core v1 Conformance Closure). Extracted verbatim in scope from
`STARKLANG/docs/compiler/COMPILER-ROADMAP.md`.

## Scope

Build a matrix covering:

- lexical versus module scopes;
- `self`, `super`, and `crate` paths;
- inline and file modules;
- cross-file and cross-package imports;
- `pub use` and multi-level re-exports;
- ambiguous imports and duplicate definitions;
- private-item leakage through public signatures;
- undeclared dependency imports;
- source-file-correct diagnostics;
- cross-package coherence input collection.

The compiler track tests compiler semantics against the current package graph. It does not
redesign manifest or registry policy in this gate.

## Inherited findings (owned by this WP per WP-C1.1's disposition)

Three confirmed deviations from WP-C0.1/WP-C1.1 are explicitly assigned to this WP:

- **DEV-004** — `resolve_unqualified` (`resolve.rs:1854-1876`) resolves bare `min`/`max` to the
  tensor extension's builtin with no `options.tensor()` gate, unlike the correctly-gated
  `resolve_path_relative`. Fix: add the same gate.
- **DEV-006 (resolve half)** — `resolve.rs` has 20 diagnostic-construction sites and zero
  `.with_file()` calls, despite having full access to per-module file identity via
  `ModuleData::file`. Multi-file packages get diagnostics misattributed to the root file. Fix:
  extend resolve to back-fill `.with_file()` the way `typecheck.rs` already does per-item.
  (The `borrowck.rs`/`flow.rs` half of DEV-006 belongs to WP-C1.4, not this WP.)
- **DEV-007** — glob-import (`use mod::*`) expansion iterates an unsorted `HashMap`, making
  which of two colliding names wins nondeterministic across runs. Fix: sort before iterating at
  both call sites (`resolve.rs:475-479`, `:536-540`).

Charter §2.2 authorizes these as in-scope, spec-consistent fixes for this WP (not new syntax/
semantics, not weakening any check — DEV-004 and DEV-007 make behavior *more* correct/
deterministic, DEV-006 improves diagnostic accuracy without changing accept/reject outcomes).

## Scope-control answers (Charter §2.6)

- **Exact compiler claim tested:** that `resolve.rs` implements Core v1 name resolution,
  module/visibility rules, and import semantics correctly and deterministically, with source-
  accurate diagnostics, across single-file, multi-file, and multi-package compilation units.
- **Later mechanism that would make the result impossible to attribute:** conflating resolution
  correctness with type-checking correctness (WP-C1.3) — a program that resolves correctly but
  fails to typecheck is not a resolution bug, and vice versa.
- **Strongest existing comparator:** old Gate 2 (`gate2-exit.md`, M2.1) evidence plus the
  existing `gate2_valid.rs`/`gate2_package.rs`/`gate3_package_resolution.rs` test suites, which
  this WP strengthens rather than replaces.
- **Negative result that would stop this WP/gate:** a resolution rule accepting a program that
  should be rejected (e.g. private-item leakage) or rejecting one that should be accepted, or a
  diagnostic pointing at the wrong file/location in a way that would mislead a real user.

## Input packet

`COMPILER-CHARTER.md` + `COMPILER-STATE.md` + this file + `07-Modules-and-Packages.md`,
`04-Semantic-Analysis.md` (name resolution sections), `resolve.rs`, `package.rs`,
`tests/{gate2_valid,gate2_package,gate3_package_resolution}.rs`.

## Execution log

**Closed 2026-07-17.** Full dated evidence, files touched, and decisions:
`COMPILER-STATE.md` session record `### WP-C1.2`.

Summary: all three inherited bugs (DEV-004, DEV-006 resolve half, DEV-007) fixed with regression
tests. All 10 checklist items reviewed with cited evidence and strengthened. Biggest gap closed:
`pub use` re-exports had real, dedicated machinery (`reexport_vis`) with zero test coverage —
now has single-level, multi-level-chain, and leak-of-private-item tests. Two previously-
unverified cross-package mechanisms (coherence checking, diagnostic file attribution) confirmed
working via real two-package-workspace tests. One significant new finding recorded but
deliberately not fixed (DEV-019, diagnostic-code collisions — a public-contract change needing
its own bounded treatment). One design fact pinned down (STARK's stricter-than-Rust visibility
model). One feature gap identified and correctly left unimplemented per Charter rule 4
(DEV-022, private-item leakage — spec silent, needs a proposal). `cargo test --workspace
--all-targets --all-features`: 410 passed / 0 failed / 2 ignored (up from 395/0/2). `cargo fmt
--check` clean. Clippy clean on all touched files. `check-conformance.py` clean.

Next: WP-C1.3 (types, generics, traits, and operator semantics).
