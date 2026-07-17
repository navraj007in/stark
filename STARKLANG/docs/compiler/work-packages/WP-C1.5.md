# WP-C1.5 — Control Flow, Patterns, Constants, and Numeric Semantics

Gate: C1 (Core v1 Conformance Closure). Extracted verbatim in scope from
`STARKLANG/docs/compiler/COMPILER-ROADMAP.md`.

## Scope

Construct positive and negative corpora for:

- definite assignment;
- return-path completeness;
- never-type coercion;
- loop/break typing;
- match exhaustiveness;
- usefulness and unreachable arms;
- nested and range patterns;
- constant evaluation used by types and arrays;
- integer literal typing;
- overflow, division by zero, indexing, and cast traps;
- build-mode invariance of traps;
- `?`, `Result`, and `Option` propagation;
- unreachable-code warning policy.

## Inherited findings (owned by this WP per prior WPs' disposition)

- **DEV-015** — no pipeline stage checks a suffixed integer/float literal's magnitude against
  its suffix's representable range (`let x: UInt8 = 300u8;` compiles and `starkc check` reports
  clean; `typecheck.rs`'s `convert_int_suffix` only maps the suffix to a type tag, never
  inspects the literal's value). WP-C1.1 flagged it but left the owner ambiguous between
  WP-C1.3 (types) and WP-C1.5 (numeric semantics); WP-C1.3 closed without taking it, so it is
  now squarely this WP's — CLAUDE.md frames overflow as a numeric-semantics concern ("Integer
  overflow... always trap — in every build mode"), and this WP's own scope list names "integer
  literal typing" and "overflow... traps" explicitly. Where the check belongs (lexer-level
  immediate rejection vs. typecheck/const-eval-level check) is a design question, not resolved
  yet — this WP should settle it.

## Scope-control answers (Charter §2.6)

- **Exact compiler claim tested:** that the checker correctly implements Core v1's control-flow
  completeness rules (definite assignment, return-path completeness, exhaustiveness), constant
  evaluation, and numeric-trap semantics (overflow/division-by-zero/indexing/cast), including
  literal-range checking (DEV-015) and build-mode invariance of traps.
- **Later mechanism that would make the result impossible to attribute:** conflating a
  soundness-relevant rejection (e.g. a non-exhaustive match silently executing undefined
  behavior, or a trap that fails to fire in release mode) with a merely stylistic/lint-like
  rejection (e.g. unreachable-code warnings); only the former requires the "would be dangerous
  if accepted" negative-test discipline the WP text calls for.
- **Strongest existing comparator:** old Gate 2/M2 evidence plus existing flow/const-eval/trap
  tests in `flow.rs`, `typecheck.rs`'s const-eval paths, and `interp.rs`'s trap sites;
  strengthened, not replaced.
- **Negative result that would stop this WP/gate:** any soundness-relevant rule that can be
  bypassed by a real, compiling program — e.g. a trap that silently fails to fire in some build
  mode (violates CLAUDE.md's "always trap — in every build mode"), a non-exhaustive match that
  executes with an unhandled variant instead of trapping/rejecting, or a `let`-binding used
  before definite assignment producing garbage instead of being rejected at compile time.

## CE-escalation watch

Per the user's 2026-07-17 standing preference (`stark-ce-escalation-flagging` memory): flag any
CE1-CE9-shaped decision found in this WP *before* resolving it, not after. This WP's territory
overlaps CE1 (normative Core semantic change) and CE2 (spec-vs-implementation ambiguity)
particularly around: where DEV-015's literal-range check should live (a genuine, unresolved
design question inherited from WP-C1.1); whether build-mode invariance of traps actually holds
today (if it doesn't, deciding how to fix it may touch CE4-adjacent panic/trap semantics, though
CE4 proper is native-runtime-ABI-scoped); and match-exhaustiveness edge cases where the spec text
and a plausible implementation could diverge. As in WP-C1.4: watch for any finding that would
require *weakening* a check to make code compile (Charter §2.2 forbids this without escalation)
versus a diagnostics-only or provenance-only fix (safe to fix directly).

## Input packet

`COMPILER-CHARTER.md` + `COMPILER-STATE.md` + this file + `04-Semantic-Analysis.md` (exhaustiveness,
definite assignment, error codes), `03-Type-System.md` (numeric semantics, casts), `05-Memory-Model.md`
(trap/abort semantics), `flow.rs`, `typecheck.rs` (const-eval, cast/overflow checks), `interp.rs`
(trap sites), existing control-flow/pattern/const/numeric test files.

## Execution log

See `COMPILER-STATE.md` session record `### WP-C1.5` for dated evidence, files touched, and
decisions.
