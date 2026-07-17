# WP-C1.4 — Ownership, Borrowing, Lifetimes, and Drop Checking

Gate: C1 (Core v1 Conformance Closure). Extracted verbatim in scope from
`STARKLANG/docs/compiler/COMPILER-ROADMAP.md`.

## Scope

Construct positive and negative corpora for:

- moves on assignment, argument passing, returns, fields, tuples, arrays, and patterns;
- Copy propagation and all-fields-Copy requirements;
- Copy plus Drop prohibition;
- shared versus mutable borrow exclusivity;
- temporary versus lexical borrow duration;
- returned-reference provenance;
- shortest-input-lifetime rule;
- borrow-carrying generic values such as `Option<&T>`;
- nested generic wrappers;
- iterator and collection views;
- methods returning references;
- cross-module and cross-package APIs;
- partial moves and drop flags;
- prohibition on moving out of indexed places or Drop types;
- exactly-once destruction on normal execution paths;
- abort semantics for panic/trap.

Every soundness-relevant rule requires a negative test that would be dangerous if accepted.

## Inherited findings (owned by this WP per prior WPs' disposition)

- **DEV-006 (borrowck/flow half)** — `flow::check`'s file parameter is explicitly unused
  (`flow.rs:21-24`, named `_file`); `borrowck::check`/`check_fn`/`check_snippet` take a single
  `file: Arc<SourceFile>` for the whole crate, with no per-item `item_files` lookup the way
  `typecheck.rs` and the now-fixed `resolve.rs` (WP-C1.2) have. For multi-file packages, every
  flow-analysis and borrow-check diagnostic for a non-root-file item is misattributed to the
  root file. The resolve.rs half of DEV-006 was fixed in WP-C1.2 via a `push_diag`/
  `current_file_arc()` backfill pattern reused from typecheck.rs — the same pattern likely
  applies here, but borrowck.rs's/flow.rs's internal structure has not yet been read closely
  enough to confirm the fix shape is identical.

## Scope-control answers (Charter §2.6)

- **Exact compiler claim tested:** that `borrowck.rs`/`flow.rs` implement Core v1's ownership,
  borrowing, and drop rules correctly, soundly, and with source-accurate diagnostics — including
  in multi-file packages.
- **Later mechanism that would make the result impossible to attribute:** conflating a
  soundness-relevant rejection (e.g. use-after-move) with a merely stylistic/lint-like rejection;
  only the former requires the "would be dangerous if accepted" negative-test discipline the WP
  text calls for.
- **Strongest existing comparator:** old Gate 2 (M2.4) evidence plus existing borrow/ownership
  tests; strengthened, not replaced.
- **Negative result that would stop this WP/gate:** any soundness-relevant rule that can be
  bypassed by a real, compiling program (e.g. a double-free, a use-after-move, an aliased
  `&mut`) — this is the highest-severity class of finding this WP could produce, given borrow
  checking is Core v1's core safety guarantee.

## CE-escalation watch

Per the user's 2026-07-17 standing preference (`stark-ce-escalation-flagging` memory): flag any
CE1-CE9-shaped decision found in this WP *before* resolving it, not after. Ownership/borrowing/
drop semantics is exactly the territory where CE1 (normative Core semantic change), CE2
(spec-vs-implementation ambiguity), or CE4-adjacent (panic/trap/drop abort semantics, though CE4
proper is native-runtime-ABI-scoped) questions are most likely to surface. Watch especially for:
any finding that would require *weakening* a check to make code compile (Charter §2.2
explicitly forbids this without escalation) versus a diagnostics-only or provenance-only fix
(same class as DEV-006, safe to fix directly).

## Input packet

`COMPILER-CHARTER.md` + `COMPILER-STATE.md` + this file + `05-Memory-Model.md`,
`03-Type-System.md` (References and Lifetimes section), `borrowck.rs`, `flow.rs`, `interp.rs`
(drop execution), existing borrow/ownership test files.

## Execution log

See `COMPILER-STATE.md` session record `### WP-C1.4` for dated evidence, files touched, and
decisions.
