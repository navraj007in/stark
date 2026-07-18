# WP-C2.1 — Reference Interpreter Contract

Gate: C2 (Reference Execution Semantics and Compiler Service Foundation). Extracted verbatim in
scope from `STARKLANG/docs/compiler/COMPILER-ROADMAP.md`.

## Scope

Write `STARKLANG/docs/compiler/reference-execution.md` describing:

- evaluation order;
- function and method dispatch;
- place evaluation;
- moves, copies, borrows, and runtime representation;
- aggregate construction and destructuring;
- drop order and drop flags;
- panic and trap abort behaviour;
- numeric conversion and failure;
- standard-library builtin dispatch;
- deterministic output expectations;
- which properties are compile-time only and have no runtime representation.

Every rule must cite the normative Core specification rather than inventing interpreter-specific
semantics.

## Inherited findings (owned by this WP per prior WPs' disposition)

None directly assigned. WP-C1.7's exit report flagged three interpreter-behavior deviations
(DEV-009 `File` has no runtime representation, DEV-023 `Display`/`Hash` missing as callable
builtin methods, DEV-024 `From::from` fails to resolve) as natural WP-C2.2 candidates, not this
WP's — WP-C2.1 documents the reference execution model as it should be (cited to the normative
spec), it does not repair interpreter bugs. Where documenting the model surfaces a place the
*current* interpreter diverges from what the spec requires, that divergence is recorded as a new
or updated deviation for WP-C2.2 to resolve, not silently fixed here (this WP is
documentation-only; no source changes are in scope) and not silently smoothed over by describing
the interpreter's actual behavior as if it were the intended one.

## Scope-control answers (Charter §2.6)

- **Exact compiler claim tested:** that a single, written reference-execution contract can state,
  for every one of the listed topics, what the *correct* (spec-derived) behavior is — not what
  the interpreter happens to currently do — with every rule traceable to a normative spec
  citation.
- **Later mechanism that would make the result impossible to attribute:** describing the
  interpreter's current behavior as normative without checking it against the spec first (the
  exact anti-pattern the roadmap text's "cite the normative Core specification rather than
  inventing interpreter-specific semantics" warns against) — this would silently launder any
  existing interpreter bug into "intended behavior," making WP-C2.2 unable to tell repair targets
  from documented design.
- **Strongest existing comparator:** the normative spec itself (`03-Type-System.md`,
  `05-Memory-Model.md` primarily; `04-Semantic-Analysis.md`, `06-Standard-Library.md` for
  specific topics) is the source of truth; `interp.rs`'s actual behavior is evidence to be
  checked against it, not a comparator to defer to.
- **Negative result that would stop this WP/gate:** finding that the spec is silent or
  genuinely ambiguous on a topic the roadmap requires this document to cover, with no reasonable
  reading — that would be a CE1/CE2-shaped spec gap requiring escalation before this WP could
  responsibly assert a rule for that topic.

## CE-escalation watch

Per the user's 2026-07-17 standing preference (`stark-ce-escalation-flagging` memory): flag any
CE1-CE9-shaped decision found in this WP *before* resolving it, not after. This WP is
documentation-only (no source changes), which lowers most classes of risk, but watch for: any
point where the spec is genuinely silent/ambiguous on a required topic (CE1/CE2 — would need a
spec-bug-protocol answer, not an invented interpreter-specific rule); and any point where writing
the "correct" model reveals the current interpreter's behavior is wrong in a way not yet in the
deviation ledger — flag as a new finding for WP-C2.2, do not fix in this WP.

## Input packet

`COMPILER-CHARTER.md` + `COMPILER-STATE.md` + `starkc/docs/compiler/C1-exit-report.md` + this
file, `03-Type-System.md`, `05-Memory-Model.md`, `04-Semantic-Analysis.md`,
`06-Standard-Library.md`, `starkc/src/interp.rs` (the actual reference execution engine).

## Execution log

See `COMPILER-STATE.md` session record `### WP-C2.1` for dated evidence, files touched, and
decisions.
