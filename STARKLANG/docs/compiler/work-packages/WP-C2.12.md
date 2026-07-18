# WP-C2.12 — Differential interpreter corpus

Gate: C2 (Reference Execution Semantics and Compiler Service Foundation). Extracted verbatim in
scope from `STARKLANG/docs/compiler/COMPILER-ROADMAP.md`.

## Scope

Close DEV-036 by replacing filename-based missing-module suppression with an explicit
test-harness/conformance input mode and permanent real-project collision regressions.

Build generated and hand-written coverage for every expression and statement, primitive
operation, struct/enum/generic/trait/method, ownership/drop edge, `Option`/`Result`, collection
and iterator, and multi-file/package execution case with deterministic output/failure snapshots.
Metamorphic transformations include alpha-renaming, harmless scopes, equivalent explicit and
inferred generics, trait-qualified calls, field shorthand/explicit initialization, equivalent
pattern decompositions, equivalent non-overlapping match-arm order, and relocation of an entire
workspace without changing manifests, lock data, or logical package sources. The same snapshot
must later run against the HIR interpreter, MIR interpreter, and native debug/release builds.

## Inherited findings (owned by this WP per prior WPs' disposition)

- **DEV-036** — `parser.rs::load_submodules_recursive` suppresses "file not found for module"
  based on the compiled file's name/path containing `"spec-fixtures"`/`"STARKLANG"` or being
  named exactly `"test.stark"` — a heuristic that can silently swallow a genuinely missing
  module file for any real user project whose path happens to collide. Proposed disposition
  (from the deviation ledger): stop keying this off the file's name/path; route the one
  legitimate exemption (`07-Modules-and-Packages__01.stark`) through an explicit,
  harness-only opt-in instead.

## Scope-control answers (Charter §2.6)

- **Exact compiler claim tested:** that a single, well-defined snapshot of interpreter execution
  behavior (status/stdout/stderr on normal completion, or trap category/message) is stable,
  deterministic, and correctly identical across programs that differ syntactically but not
  semantically (the metamorphic-transformation classes named above) or across a relocated
  workspace. Separately, that DEV-036's residual real-project risk is closed with an explicit
  mechanism rather than a runtime heuristic.
- **Later mechanism that would make the result impossible to attribute:** building the corpus
  against a HIR interpreter this WP itself has just silently patched to make a metamorphic pair
  agree, rather than recording a genuine divergence as a new deviation. The corpus's value comes
  from being a faithful, unmodified oracle for later MIR/native cross-backend replay (Gate C3+);
  quietly reshaping test cases to dodge a real bug instead of reporting it would defeat that
  purpose.
- **Strongest existing comparator:** `tests/snapshots.rs`'s existing AST-snapshot
  (`UPDATE_SNAPSHOTS`-driven golden-file) convention, extended from parse-only output to
  full-pipeline execution output; `tests/gate3_execution.rs`'s existing parse/resolve/typecheck/
  run helper pattern, reused directly for the new harness's `render` function.
- **Negative result that would stop this WP/gate:** a metamorphic pair that genuinely diverges
  (two semantically-equivalent programs producing different execution output) would be a real,
  confirmed interpreter bug, not a reason to redesign the pair to hide it — it gets recorded as
  its own deviation, same as any other WP-C2.x finding.

## CE-escalation watch

Per the user's 2026-07-17 standing preference (`stark-ce-escalation-flagging` memory): flag any
CE1-CE9-shaped decision found in this WP before resolving it, not after. This WP found several
real, previously-unrecorded interpreter/type-checker gaps while building the corpus (see the
session record in `COMPILER-STATE.md` and DEV-051 through DEV-055 in
`starkc/docs/conformance/KNOWN-DEVIATIONS.md`). Corpus-building itself does not fix findings —
Charter §1.5 rule 4 ("no new Core syntax or semantics inside an implementation WP") plus the
general principle of not silently weakening checks apply — but DEV-053 (originally framed as
"tuple-pattern usefulness/exhaustiveness false positives") was investigated as a dedicated
follow-up in the same session, at the user's explicit request, once flagged. The investigation
found the original framing was itself imprecise: the real root cause is that a bare `None`
pattern never matched by value at all (misclassified as a fresh binding, since `resolve.rs`'s
`lower_pattern` checked module items for a known value but never checked `Res::Builtin`), and it
silently acted as an unconditional wildcard — confirmed to produce **wrong runtime output**
(`match Some(5) { None => 999, Some(a) => a }` printed `999`), not merely the spurious
diagnostics originally reported. This is exactly the dangerous-direction concern the
soundness-adjacent flag was warning about, just manifesting as silent wrong output rather than a
wrongly-accepted non-exhaustive match. Fixed in `resolve.rs`/`typecheck.rs` (also closing
DEV-054, the same root cause) with five new regression tests. The tuple-scrutinee
"non-exhaustive" half of the original DEV-053 report was, on closer reading, a deliberate,
already-documented, sound-by-construction design limitation (requiring an individually
irrefutable arm for non-enumerable scrutinee types) — not a bug, and not part of the fix. One
new, separate, unfixed finding (DEV-055: bare glob-imported unit enum variants don't resolve at
all, as either an expression or a pattern) was surfaced as a control case while scoping the fix
and remains open.

## Input packet

`COMPILER-CHARTER.md` + `COMPILER-STATE.md` + this file + `CORE-V1-ABSTRACT-MACHINE.md`
(`OBS-COMPARE-001`, the differential observation comparator this corpus implements),
`07-Modules-and-Packages.md` (multi-file layout rules, DEV-036's normative anchor), `parser.rs`
(`load_submodules_recursive`), `tests/snapshots.rs` (the AST-snapshot convention this WP mirrors
for execution), `tests/conformance.rs`/`tests/gate2_valid.rs` (existing missing-module-file
regressions), `tests/gate3_execution.rs` (existing parse/resolve/typecheck/run helper pattern),
`tests/gate2_package.rs`/`tests/gate3_package_resolution.rs` (existing multi-package workspace
construction patterns, reused for the workspace-relocation case).

## Execution log

See `COMPILER-STATE.md` session records `### WP-C2.12` and `### WP-C2.12 — DEV-053/DEV-054
investigation and fix` for dated evidence, files touched, and decisions. Summary of the first
pass: DEV-036 closed with a real code fix (not just documentation) plus six regression tests. A
new differential/execution snapshot harness (`starkc/tests/exec_snapshots.rs`) was built,
mirroring the existing AST-snapshot convention. An initial, representative (not exhaustive)
corpus was populated: 17 primary cases across all seven named coverage categories, one worked
pair for each of the seven named metamorphic transformation classes, and one
workspace-relocation execution-identity test. Four new, previously-unrecorded compiler gaps were
found and recorded (not fixed) while building the corpus: DEV-051 (trait default methods cannot
call another trait method on `self`), DEV-052 (`Trait::method(...)` qualified-call syntax fails
to resolve for compiler CoreTraits though it works for user-defined traits), DEV-053 (originally
framed as tuple-pattern usefulness/exhaustiveness false positives), DEV-054 (a tuple pattern
with a repeated by-value identifier rejected as a duplicate binding).

Summary of the follow-up investigation and fix (same session, user-requested): DEV-053's
original framing was corrected -- the real root cause is that a bare `None` pattern never
matched by value at all, silently acting as an unconditional wildcard and producing **wrong
runtime output** (not merely spurious diagnostics), because `resolve.rs`'s `lower_pattern` never
checked `Res::Builtin` when disambiguating a bare identifier pattern. Fixed in
`resolve.rs`/`typecheck.rs`, closing both DEV-053 and DEV-054 (same root cause) with five new
regression tests. One new, separate, unfixed finding, DEV-055 (bare glob-imported unit enum
variants don't resolve at all), was surfaced as a control case while scoping the fix.

**Remaining work, not done in this pass** (see COMPILER-STATE.md Follow-ups): the roadmap's
"generated" half of "generated and hand-written coverage" (a case generator, as opposed to this
pass's hand-written cases); deeper per-category breadth (this pass is representative, not
exhaustive, for any of the seven categories); cross-backend replay against a MIR interpreter and
native builds (explicitly named as future work by the roadmap text itself, blocked behind Gate
C3); DEV-051, DEV-052, and DEV-055 remain open and unscheduled.
