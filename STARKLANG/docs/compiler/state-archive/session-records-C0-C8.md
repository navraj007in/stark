# Session records C0–C8 — archived from COMPILER-STATE.md

**Archived 2026-08-09 under AS8 exit criterion 4.** Verbatim, in original order. The records for
WP-C0.0 through the Post-Gate-C2 correction brief were already archived once (CD-020) in
`C0-C2-closed-detail.md`; this file continues that practice for everything up to Sprint 4.

Sprint 4's own records are NOT here. They stay in `COMPILER-STATE.md`, because a compression
target is not a reason to archive a record that is still being worked against.

---

## Session records
Records for WP-C0.0 through the Post-Gate-C2 correction brief Issues 1-5 (2026-07-17 through
2026-07-18) are archived verbatim in
`STARKLANG/docs/compiler/state-archive/C0-C2-closed-detail.md` (CD-020). Gate-level evidence
remains in the C0/C1/C2 exit reports. Records below start at the most recent still-live
transition context.

### Post-Gate-C2 correction brief — Issues 6-8 (DEV-051, DEV-052, DEV-055) — 2026-07-19
DONE: user said "fix them" (Issues 6-8, previously deferred). Reproduced and closed all three
with real fixes, one working method throughout: reproduce on current head, isolate root cause by
reading the relevant resolver/checker code rather than guessing, fix, add regression tests, run
full verification, update docs.
- **DEV-055** (fixed first, most precisely diagnosed already): `use Color::*;`/`use
  Color::{Red, Blue};` silently expanded to nothing when the prefix names an enum rather than a
  real module. Root cause: `resolve_use_tree`'s `Glob`/`Group` arms (and their `_relative`
  counterparts) only ever consulted `submodule_map` (real modules); an enum's variants are
  resolved dynamically through `item_details`, never pre-populated into a module's `items` map.
  Fixed by adding `enum_variant_items`/`resolve_enum_variant_group_item`, wired into both arms in
  both functions. 5 new regression tests (`resolve.rs` x2, `interp.rs` x3), including one
  confirming a variant deliberately left out of a group import correctly stays undefined (rules
  out an overly-broad "import everything" fix).
- **DEV-051**: a trait default method body calling a sibling trait method through `self`
  (`self.name()` inside `fn greeting(&self) { self.name() }`) failed to type-check with `E0302`.
  Root cause: `resolve_method` already had a mechanism for an abstract `Ty::Param` receiver with
  no concrete `impl` to match (a bounded *generic function* type parameter), but it was scoped
  only to that case, never to `self` inside a trait's own default-method body (`current_self_ty
  == Ty::Param("Self")`, checked once, generically, at the trait declaration site). A first
  attempt placed the new check in the same spot as the existing one and it still failed, since
  `self`'s type at that point is `&Self` (a `Ty::Ref`), not bare `Ty::Param("Self")` — moved it
  to after the reference-deref loop, unlike the by-value generic-parameter case. Added
  `current_trait_id` (set alongside `current_self_ty` for trait default bodies) plus two shared
  helpers (`find_trait_method_sig`/`check_trait_member_call`) refactored out of the
  previously-inlined generic-parameter logic. 4 new regression tests, including a
  default-calling-another-default case and a wrong-arg-count case (confirms the fix doesn't
  silently swallow a genuine arity mismatch). **Side finding, NOT fixed** (confirmed pre-existing
  via `git stash`, not introduced by this fix): DEV-060 — calling the same un-overridden default
  method twice on one receiver wrongly raises `E0100 use of moved value` on the second call; two
  calls to an *overridden* trait method or an ordinary inherent method are both unaffected.
  Recorded as a new open deviation with its own regression tests documenting the current
  (defective) behavior and its exact scope, rather than silently worked around.
- **DEV-052**: `Eq::eq(&a, &b)` (fully-qualified call syntax) failed to resolve
  (`E0200 undefined variable 'Eq::eq'`) while the same syntax worked for a user-declared trait.
  Root cause: `resolve_path_relative`'s multi-segment loop only continued past a first segment
  resolving to `Res::Item` (a real trait declaration item, member indexed against
  `ItemDefDetail::Trait`); a `CoreTrait` (`Eq`, `Ord`, ...) has no such declaration item at all.
  Fixed by adding `Res::CoreTraitMember(CoreTrait, Span)`, resolved via a new
  `core_trait_method_name` table (one fixed callable method name per `CoreTrait`: `Eq`→"eq",
  `Ord`→"cmp", `Hash`→"hash", `Clone`→"clone", `Display`→"fmt", `Default`→"default"). Typecheck
  (`check_qualified_core_trait_call`) finds the matching impl's own method signature directly
  (no shared trait declaration to instantiate from, unlike the user-trait case), matching impls
  by trait-ref source text against a new `core_trait_source_name` table (mirroring
  `ty_satisfies_operator_bound`'s existing approach). The interpreter side needed no new
  impl-scanning logic at all: `call_qualified_core_trait` reuses the *exact* `find_method(...,
  Some(Res::CoreTrait(_)))` lookup the `==`/`<` operator sugar already calls for these traits — a
  qualified call is just an explicit spelling of the same dispatch. 4 new regression tests
  (`Eq` and `Ord`, an unimplemented-trait rejection, and a guard confirming the pre-existing
  user-trait qualified-call path is unaffected).
FILES: `starkc/src/resolve.rs` (DEV-055's `enum_variant_items`/`resolve_enum_variant_group_item`;
DEV-052's `core_trait_method_name` table and path-resolution wiring; both regression tests),
`starkc/src/typecheck.rs` (DEV-051's `current_trait_id` field and `find_trait_method_sig`/
`check_trait_member_call` helpers; DEV-052's `check_qualified_core_trait_call`/
`core_trait_source_name`; all three fixes' regression tests plus DEV-060's documentation test),
`starkc/src/interp.rs` (DEV-055/DEV-051 end-to-end regression tests; DEV-052's
`call_qualified_core_trait`; DEV-060's two scope-confirming companion tests),
`starkc/src/hir.rs` (new `Res::CoreTraitMember` variant), `starkc/src/analysis/query.rs`
(exhaustiveness update for the new `Res` variant), `starkc/docs/conformance/
KNOWN-DEVIATIONS.md` (DEV-051/052/055 marked resolved with full root-cause writeups; new
DEV-060 opened; count line updated to 58), this file.
RULES: none — three runtime/type-check-semantics corrections against already-normative rules
(trait default-method dispatch and fully-qualified trait-call syntax per `03-Type-System.md`;
glob-import name resolution per `07-Modules-and-Packages.md`); no conformance-database rule
citation or normative specification text changed.
DECISIONS: none new as CD/AD records. All three are spec-consistent corrections under Charter
§2.2 Sonnet-level autonomy — each makes a previously-rejected legal program accepted and correct,
none weakens an existing check or changes accepted behavior in a way that admits an unsound
program.
EVIDENCE: MANUAL + REG — every fix's original bug and every new regression scenario was run
against the actual compiler (not inferred from code reading alone); DEV-060's pre-existing,
unrelated-to-DEV-051 status was independently confirmed via `git stash` against the pre-fix head
before being recorded, not assumed. `cargo test --workspace --all-targets --all-features`:
**594 passed / 0 failed / 2 ignored** (up from 578/0/2 pre-this-pass, exactly the 16 new tests
across the three fixes — see each fix's own count above — zero regressions elsewhere). `cargo fmt --all -- --check` clean. `cargo clippy --workspace --all-targets
--all-features -- -D warnings` clean. `python3 scripts/check-conformance.py` re-run clean
(89.8%/53-of-59, unchanged -- none of these three fixes touch the conformance evidence database).
NEXT: no further work authorized this pass. DEV-060 (new, open) and DEV-009/DEV-022/DEV-023/
DEV-024 (long-open, C2.8/C2.9-owned) are the remaining known deviations without a fix.

### C3-entry governance-repair pass (CD-020) — 2026-07-19
DONE: full scope of CD-020 (see decision log): WP-C3-ENTRY.md created and wired into the
roadmap's C3-ENTRY section; WP-C4.4/C5.6/C6.5 amended to carry transferred WP-C2.12
obligations; CI widened to the C3-ENTRY baseline command forms plus new spec-regeneration
(`build-core-spec.py --check`) and named execution-snapshot steps; KNOWN-DEVIATIONS.md tail
summary corrected (DEV-009/022/023/024 were resolved by WP-C2.11, not open — the preceding
Issues 6-8 session record's own NEXT line repeats that stale claim and is corrected by this
note, left in place per append-only convention); state header head/fixture-census corrected
(`9e85396`, 113 entries/parse-pass 65); charter §5.3 dangling refs, commit-policy step, and
WP-C6.4 tier label fixed; SYSTEMS-ROADMAP.md gained the P1-relationship section; this file
compressed 3,145 → ~700 lines with all removed material verbatim in
`STARKLANG/docs/compiler/state-archive/C0-C2-closed-detail.md`; C2-exit-report open-deviation
table given a dated post-gate update note.
FILES: COMPILER-STATE.md, STARKLANG/docs/compiler/COMPILER-CHARTER.md,
STARKLANG/docs/compiler/COMPILER-ROADMAP.md,
STARKLANG/docs/compiler/work-packages/WP-C3-ENTRY.md (new),
STARKLANG/docs/compiler/state-archive/C0-C2-closed-detail.md (new),
STARKLANG/docs/ecosystem/SYSTEMS-ROADMAP.md, starkc/docs/conformance/KNOWN-DEVIATIONS.md,
starkc/docs/compiler/C2-exit-report.md, STARKLANG/tools/build-core-spec.py,
.github/workflows/ci.yml.
RULES: none — no normative rule, compiler, or interpreter change; governance surface only.
DECISIONS: CD-020.
EVIDENCE: `python3 STARKLANG/tools/build-core-spec.py --check` clean twice (deterministic);
`cargo fmt --all -- --check` clean; `cargo test --test exec_snapshots` 3 passed / 0 failed;
line-count arithmetic for the compression verified (588 kept + 2,557 archived = 3,145
original). Full `cargo test --workspace` not re-run this pass (no code changed); full CI run
of the updated workflow pending — tracked as the remaining CI blocker item in WP-C3-ENTRY.md.
FOLLOW-UP: owner decisions per WP-C3-ENTRY.md blockers 1-2 (six completeness rows, DEV-060);
corpus freeze after DEV-060 disposition; one demonstrated green CI run.
NEXT: WP-C3-ENTRY blocker closure; then C3-entry exit artifact; then WP-C3.1.

### CD-021 roadmap amendment — 2026-07-19
DONE: applied the owner-approved CD-021 amendment (see decision log): WP-C3.1 workload items
16-21 (existing function-value capability), C4.1/C4.3/C4.5 indirect-call ownership, C5.1
function-value ABI items, P1/S5 trap-abort operational report, WP-C10.7 release-blocking
deviation sweep.
FILES: STARKLANG/docs/compiler/COMPILER-ROADMAP.md,
STARKLANG/docs/ecosystem/SYSTEMS-ROADMAP.md, COMPILER-STATE.md.
RULES: none — no normative Core rule, compiler, or interpreter change; the workload items
reference already-frozen `fn(...)` semantics.
DECISIONS: CD-021 (owner-approved this session).
EVIDENCE: spec/implementation citations verified by direct grep before recording
(03-Type-System.md:198-200,999; 06-Standard-Library.md:243-244,260-262,663-666;
interp.rs:260). No count/enumeration references to the C3.1 workload existed to go stale
("at least:" phrasing confirmed).
FOLLOW-UP: draft the "Callable ABI and Future Closure Compatibility Spike" proposal before
WP-C5.1 (recommended during C3 spike work); WP-C3-ENTRY blockers unchanged and still open.
NEXT: WP-C3-ENTRY blocker closure (six completeness rows, DEV-060, corpus freeze, green CI
run); then C3-entry exit artifact; then WP-C3.1 with the 21-item workload [23 after CD-022].

### CD-022 follow-up amendment — 2026-07-19
DONE: applied the owner-approved CD-022 (see decision log): release-class claim-scope repair
(Compiler Stable vs General-Purpose Stable, CD-019 preserved), WP-C3.1 workload items 22-23
plus the pre-backend-selection Eq/Hash/monomorphised-identity resolution requirement,
state-header field renamed to "Amendment base commit".
FILES: STARKLANG/docs/compiler/COMPILER-ROADMAP.md, COMPILER-STATE.md.
RULES: none — no normative Core rule, compiler, or interpreter change. The two open
function-value properties are flagged for settlement, not settled here.
DECISIONS: CD-022 (owner-approved this session).
EVIDENCE: spec citation verified by direct read before recording (03-Type-System.md:748-749 —
function values are Copy); release-class contradiction verified against the roadmap text
(C7.7 P1 gate vs the vacuous conditional). Workload numbering re-verified contiguous 1-23.
FOLLOW-UP: push to origin and record one green run of the updated CI workflow (last
C3-entry CI blocker item); callable-ABI/closure-compatibility spike proposal still pending,
pre-C5.1.
NEXT: WP-C3-ENTRY blocker closure (six completeness rows, DEV-060, corpus freeze, green CI);
then C3-entry exit artifact; then WP-C3.1 with the 23-item workload.

### C3-ENTRY blockers 1-2 closure — 2026-07-19 (CD-023/CD-024)
DONE: applied both owner-approved decisions from this session. CD-023: six
`pending-owner-approval` completeness rows approved as-is, flipped to `settled` in
`CORE-V1-COMPLETENESS.md`, C2-exit-report.md given a dated post-gate note, WP-C3-ENTRY.md
blocker 1 marked closed. CD-024: DEV-060 root-caused and fixed in `borrowck.rs::method_receiver`
(missing trait-default-body fallback, mirroring typecheck.rs's own `default_fallback`); two new
regression tests plus one rewritten; KNOWN-DEVIATIONS.md, WP-C3-ENTRY.md blocker 2, and the
open-deviation index all updated to reflect closure.
FILES: starkc/src/borrowck.rs (fix), starkc/src/typecheck.rs (rewrote
`repeated_call_to_unoverridden_default_trait_method_is_wrongly_flagged_as_move` to
`_is_no_longer_flagged_as_move`; added `repeated_call_to_unoverridden_mut_default_trait_
method_is_no_longer_flagged_as_move`), starkc/src/interp.rs (added
`repeated_call_to_unoverridden_default_trait_method_executes_correctly`),
STARKLANG/docs/compiler/semantic-freeze/CORE-V1-COMPLETENESS.md,
starkc/docs/compiler/C2-exit-report.md, starkc/docs/conformance/KNOWN-DEVIATIONS.md,
STARKLANG/docs/compiler/work-packages/WP-C3-ENTRY.md, COMPILER-STATE.md.
RULES: none — no normative Core rule change; this closes a compiler defect where legal,
spec-conforming code was wrongly rejected (availability bug, not a soundness/acceptance bug).
DECISIONS: CD-023, CD-024 (both owner-approved this session).
EVIDENCE: `cargo build` clean; full `cargo test --workspace --all-targets --all-features`
596 passed / 0 failed / 2 ignored (up from 594); `cargo fmt --all -- --check` clean; `cargo
clippy --workspace --all-targets --all-features -- -D warnings` clean; `python3
starkc/scripts/check-conformance.py` re-run, unchanged (89.8%/53-of-59 — DEV-060 was a
runtime/borrowck defect, not a conformance-database entry). Root cause independently isolated
by direct code reading (borrowck.rs's `method_receiver` vs typecheck.rs's `resolve_method`),
not assumed from the ledger's prior "needs its own investigation" note.
FOLLOW-UP: corpus freeze is now unblocked (WP-C3-ENTRY.md required DEV-060 resolved first,
since a fix could legitimately change corpus output) — next actionable step; green CI run still
needs a push to origin.
NEXT: freeze the versioned execution corpus per WP-C3-ENTRY.md's procedure; then push and
obtain a green CI run; then write starkc/docs/compiler/C3-entry-exit.md; then WP-C3.1.

### C3-ENTRY blockers 3-4 closure + gate close — 2026-07-19 (CD-025)
DONE: froze the execution-snapshot corpus and closed the C3-ENTRY transition. corpus.lock
(v1.0.0, 48 files, base 3d12f45) + integrity test `corpus_lock_matches_frozen_snapshot`
(negatively verified). CI green on origin/main @ 3d12f45 (owner-confirmed). Wrote exit artifact
C3-entry-exit.md; flipped Position to Gate C3 / WP-C3.1 / Blocked: none; checked off all
WP-C3-ENTRY Done-when items. Gate C3 is open.
FILES: starkc/tests/exec_snapshots/corpus.lock (new), starkc/tests/exec_snapshots.rs (new
integrity test), starkc/docs/compiler/C3-entry-exit.md (new),
STARKLANG/docs/compiler/work-packages/WP-C3-ENTRY.md, COMPILER-STATE.md.
RULES: none — freeze/governance only, no Core behavior change.
DECISIONS: CD-025.
EVIDENCE: `cargo test --test exec_snapshots` 4 passed (incl. integrity test); tamper-then-
restore negative check confirms the integrity test fails on drift; `cargo fmt --all -- --check`
and `cargo clippy --test exec_snapshots --all-features -- -D warnings` clean; full workspace
596/0/2 from CD-024 unchanged (corpus freeze adds one test → next full run will read 597/0/2).
FOLLOW-UP: none blocking. Optional pre-C5.1: draft the "Callable ABI and Future Closure
Compatibility Spike" proposal during C3 spike work (CD-021).
NEXT: WP-C3.1 — freeze the 23-item representative workload, define the measurement set, write
STARKLANG/docs/compiler/proposals/NATIVE-CORE-ARCHITECTURE.md. Gate C3 selects backend
architecture (SELECT-GENERATED / SELECT-DIRECT / REVISE / BLOCKED), never interpreter-only.

### WP-C3.1 — Architecture hypothesis and workload freeze — 2026-07-19
DONE: authored the Gate C3 setup deliverables. Wrote
`STARKLANG/docs/compiler/proposals/NATIVE-CORE-ARCHITECTURE.md` (new proposals/ dir): the four
separated questions, pipeline context, the frozen-Core decisions that favor native lowering
(trap-abort-no-unwind, no trait objects, non-capturing fn values, borrow-check-before-codegen,
deterministic order), the architecture hypothesis (Candidate A generated Rust/C vs Candidate B
direct Cranelift; leading hypothesis SELECT-GENERATED with explicit falsifiers), the frozen
23-item workload mapped to corpus v1.0.0 (items 1-10) + specified reference programs (11-23),
the risk register (both candidates, per hard construct), the 13-dimension measurement framework,
and the WP-C3.4 decision preview. Created `work-packages/WP-C3.1.md`. Set Native-backend-
selection status to SPIKING; flipped Position Next to WP-C3.2/C3.3.
FILES: STARKLANG/docs/compiler/proposals/NATIVE-CORE-ARCHITECTURE.md (new),
STARKLANG/docs/compiler/work-packages/WP-C3.1.md (new), COMPILER-STATE.md.
RULES: none — non-normative proposal; no Core semantics, compiler, or interpreter change. States
a hypothesis, freezes a workload, defines measurements; selects nothing.
DECISIONS: none at CE level. Leading hypothesis (SELECT-GENERATED) is explicitly flagged as
falsifiable orientation for the spikes, not a decision — CE5 backend selection remains the
owner's at WP-C3.4. Flagged per the CE-escalation convention.
EVIDENCE: all 15 corpus-case references + the workspace-relocation test name + the two
metamorphic pair names verified to resolve against the real tree (no dangling pointers).
Interpreter support for the harder workload items confirmed by direct source read: function
values (`Value::Function`, interp.rs:2168 indirect call), file I/O (`Value::File` +
`read_to_string`/`write`, DEV-009 resolved), references/slices. No build/test run needed — no
code changed.
FOLLOW-UP: recommended (not approved) — draft the "Callable ABI and Future Closure Compatibility
Spike" memo during C3 spike work, before WP-C5.1 freezes the ABI (CD-021). The two open fn-value
properties (Eq/Ord/Hash participation, monomorphised-generic identity) must be settled before
WP-C3.4 selection (CD-022).
NEXT: WP-C3.2 (generated Rust/C spike) and WP-C3.3 (direct Cranelift spike) — each implements
the reachable workload subset and reports every measurement dimension + unsupported constructs;
then WP-C3.4 selects under CE5.

### WP-C3.2 — Generated-Rust backend spike — 2026-07-19
DONE: built and ran the generated-Rust backend spike (Candidate A). Isolated HIR→Rust lowerer +
compile/run/diff harness in `starkc/tests/spike_genrust.rs` (charter §2.2 — NOT wired into
`stark build`, adds nothing to the library surface, disposable). Lowers a supported subset
(integer primitives i8..u64 + Bool, trap-checked arithmetic, comparisons/logic, let/mut/assign,
if/while/loop/for/break/continue, block-tail values, non-generic fns + calls, print/println)
from typed HIR to Rust, compiles with rustc, runs, compares stdout+exit-status to the interpreter
oracle over the frozen exec_snapshots corpus v1.0.0. Wrote the spike report
`starkc/docs/compiler/spikes/WP-C3.2-generated-rust.md` (new spikes/ dir) with every WP-C3.2
measurement record + the NATIVE-CORE-ARCHITECTURE.md §7 dimension mapping. Created WP-C3.2.md.
RESULT: 4/17 corpus cases lowered and matched exactly (arithmetic/precedence,
loops/for/break/continue, multi-width ints, Int8-overflow trap→abort parity); 0 semantic
mismatches on supported cases; 13/17 cleanly reported unsupported with reasons; mean rustc
compile 87 ms/case. Candidate liabilities (rustc dep weight, compile-time scaling, exe size,
debug-info trap mapping, unsupported breadth) neither falsified nor cleared — that needs the
C3.3 spike + a breadth run.
FILES: starkc/tests/spike_genrust.rs (new), starkc/docs/compiler/spikes/WP-C3.2-generated-rust.md
(new), STARKLANG/docs/compiler/work-packages/WP-C3.2.md (new), COMPILER-STATE.md.
RULES: none — spike/evidence only. The spike does NOT bypass front-end checks (it consumes
already-validated typed HIR) and does NOT select a backend (WP-C3.4/CE5). No Core semantics,
compiler, or interpreter change.
DECISIONS: none at CE level. Native-backend-selection status stays SPIKING.
EVIDENCE: `cargo test --test spike_genrust` 2 passed; full workspace
`cargo test --workspace --all-targets --all-features` 599 passed / 0 failed / 2 ignored (597 +
the 2 spike tests); `cargo fmt --all -- --check` and `cargo clippy --test spike_genrust
--all-features -- -D warnings` clean. Coverage table reproduced via `-- --nocapture`.
FOLLOW-UP: WP-C3.3 direct-Cranelift spike must run before any candidate comparison; dimensions
3/5/11/12/13 (exe size, runtime perf, monomorphisation, trait dispatch, ref/slice/Drop ABI) need
a breadth run on both candidates. The two open fn-value properties (CD-022) still pending
pre-C3.4.
NEXT: WP-C3.3 — direct Cranelift spike over the same frozen workload with the same measurement
record; then WP-C3.4 selects under CE5.

### WP-C3.3 — Direct (Cranelift) backend spike — 2026-07-19
DONE: built and ran the direct Cranelift backend spike (Candidate B). Isolated HIR→Cranelift-IR
lowerer + object-emission + `cc`-link + run/diff harness in `starkc/tests/spike_cranelift.rs`
(charter §2.2 — NOT wired into `stark build`, disposable). Same frozen workload subset as C3.2.
Produces a real standalone native executable. Added Cranelift dev-dependencies (pinned 0.110 for
rustc-1.93 compat, with a necessity note in Cargo.toml; dev-only, not the shipped surface).
Object emission (not JIT) → no `unsafe` (crate forbids it). Wrote report
`starkc/docs/compiler/spikes/WP-C3.3-direct-cranelift.md` with the head-to-head table vs C3.2 and
an explicit timing caveat. Created WP-C3.3.md. Native-backend-selection section updated with both
spikes' results.
RESULT: 3/17 corpus cases matched the interpreter exactly (arithmetic, loops/for/break/continue,
Int8-overflow trap→abort parity); 0 semantic mismatches; 14/17 unsupported (same families as C3.2
plus unsigned ints). Timing: Cranelift codegen ~2 ms/case (phase-only, from built IR, no
parse/typecheck/link), `cc` link ~47 ms/case; end-to-end ~49 ms vs rustc ~87 ms ≈ 1.8× on 3
trivial programs — flagged as NOT a general multiple (charter caution). No rustc build dep.
MSRV-churn finding (0.133→rustc 1.94). Higher glue than generated-Rust; weaker debug-info;
biggest MIR beneficiary.
FILES: starkc/tests/spike_cranelift.rs (new), starkc/docs/compiler/spikes/
WP-C3.3-direct-cranelift.md (new), STARKLANG/docs/compiler/work-packages/WP-C3.3.md (new),
starkc/Cargo.toml (dev-deps), COMPILER-STATE.md.
RULES: none — spike/evidence only, no front-end bypass, no backend selection (WP-C3.4/CE5), no
Core/compiler/interpreter change. Cranelift is a dev-dependency only (charter §1.10 note in
Cargo.toml).
DECISIONS: none at CE level. Native-backend-selection stays SPIKING.
EVIDENCE: `cargo test --test spike_cranelift` 1 passed; full workspace 600 passed / 0 failed / 2
ignored (599 + the cranelift spike); `cargo fmt --all -- --check` + `cargo clippy --test
spike_cranelift --all-features -- -D warnings` clean. Coverage + timings via `-- --nocapture`.
FOLLOW-UP: WP-C3.4 needs a breadth run (aggregates/generics/traits/refs/Drop/fn-values) on both
candidates and exe-size/startup/runtime measurement before a confident selection; the two open
fn-value properties (CD-022) still pending pre-selection.
NEXT: WP-C3.4 — backend and runtime architecture selection under CE5 (owner decision):
SELECT-GENERATED / SELECT-DIRECT / REVISE / BLOCKED.

### WP-C3 breadth run (both spikes) — 2026-07-19
DONE: extended the generated-Rust spike across aggregate/generic breadth (structs, impl/methods,
struct literals, field/method access, generics + trait bounds, Option/Result, match + pattern
lowering, String/&str) → 8/17 frozen corpus cases, all matching the interpreter (was 4/17). ~250
lines of mechanical text emission; rustc absorbs monomorphization/layout/ABI/Drop. Cranelift
breadth measured at the struct boundary rather than fully implemented (would need stack-slot
layout + sret ABI for structs, tagged-union layout for enums, a monomorphization engine for
generics, a runtime for String/Vec — each a subsystem), grounded in the built ~600-line Cranelift
lowerer; Cranelift stays 3/17. Wrote WP-C3-breadth-comparison.md (the head-to-head + the caveat
that most direct-backend breadth cost is mandatory MIR work anyway, so the HIR-level comparison
overstates it). Updated WP-C3.2 and WP-C3.3 reports.
FILES: starkc/tests/spike_genrust.rs (breadth extension + updated unsupported-cases test),
starkc/docs/compiler/spikes/WP-C3.2-generated-rust.md, WP-C3.3-direct-cranelift.md,
WP-C3-breadth-comparison.md (new), COMPILER-STATE.md.
RULES: none — spike/evidence only; no front-end bypass, no backend selection, no Core/compiler/
interpreter change.
DECISIONS: none at CE level. Native-backend-selection stays SPIKING.
EVIDENCE: `cargo test --test spike_genrust` 2 passed (match-interpreter now 8/17 + updated
unsupported-cleanly test); full workspace `cargo test --workspace --all-targets --all-features`
600 passed / 0 failed / 2 ignored; `cargo fmt --all -- --check` + `cargo clippy --test
spike_genrust --all-features -- -D warnings` clean.
FOLLOW-UP: optional exact Cranelift struct head-to-head (~150-200-line sret impl); exe-size/
startup/runtime still unmeasured for both; the fair comparison is at the MIR level (Gate C4), not
HIR. The two open fn-value properties (CD-022) still pending pre-C3.4.
NEXT: WP-C3.4 — backend and runtime architecture selection under CE5 (owner): SELECT-GENERATED /
SELECT-DIRECT / REVISE / BLOCKED.

### WP-C3.4 — Backend selection (owner CE5 decision) — 2026-07-19
DONE: drafted the three-way backend-selection analysis
(`starkc/docs/compiler/spikes/WP-C3.4-backend-selection-analysis.md`) consolidating the WP-C3.1
framework + WP-C3.2/C3.3 spikes + breadth run, with a reasoned recommendation and the required
architecture commitments; presented the decision to the owner (CE5). **Owner selected
`SELECT-GENERATED`** — generated Rust as the initial production backend behind verified MIR,
backend-neutral MIR keeping direct-Cranelift open as a C7 migration. Recorded as CD-026;
Native-backend-selection section → SELECTED / generated Rust/C; created WP-C3.4.md; Position line
advanced to Gate C4 / WP-C4.1. Gate C3 is complete.
FILES: starkc/docs/compiler/spikes/WP-C3.4-backend-selection-analysis.md (new),
STARKLANG/docs/compiler/work-packages/WP-C3.4.md (new), COMPILER-STATE.md.
RULES: none — a strategy selection only; does not build MIR, define the MIR contract (C4/CE3), or
fix the runtime ABI (C5.1/CE4). No Core/compiler/interpreter change.
DECISIONS: CD-026 (owner CE5). Native-backend-selection = SELECTED.
EVIDENCE: decision presented and recorded; the supporting spike evidence (WP-C3.2/C3.3/breadth
reports) is unchanged and already committed. No new code; workspace baseline unchanged (600/0/2).
FOLLOW-UP: the disposable spikes (`tests/spike_genrust.rs`, `tests/spike_cranelift.rs`, Cranelift
dev-deps) are retained for now as C3 evidence; remove/rewrite them when the real MIR-consuming
generated-Rust backend lands (they are not production architecture, charter §2.2). The two open
fn-value properties (CD-022) must be settled during C4/C5. Optional: exe-size/startup measurement
and the Cranelift struct head-to-head remain available if C7 re-evaluation needs them.
NEXT: Gate C4 — WP-C4.1 (MIR design review, CE3): define the backend-neutral verified MIR contract
(`STARKLANG/docs/compiler/mir.md`) that the generated-Rust emitter consumes.

### Pre-C4.1 fn-value settlement and correction pass (CD-027) — 2026-07-19
DONE: settled both CD-022 carry-forward properties (TYPE-FN-001 non-participation in
Eq/Ord/Hash → identity unobservable; TYPE-FN-002 generic coercion = instantiate-at-coercion,
both owner-approved) as normative rules in 03-Type-System.md §Function Types; regenerated the
combined spec (fixtures unchanged — prose-only rules); added TYPE-FN-001/002 rows to the
completeness inventory (166 → 168). Discovered by first-ever execution of workload items 16-22
that the whole fn-value feature was a compile-time façade: recorded DEV-061/062/063, got owner
fix-now authorization, fixed all three (interp dispatch arm; Ty::Fn Copy classification in
borrowck+typecheck; Option/Result combinator signatures + consuming interp interception), and
recorded-but-deferred DEV-064 (undetermined-generic coercion, owner C4.5).
FILES: STARKLANG/docs/spec/03-Type-System.md (+ regenerated STARK-Core-v1.md/.html/.pdf),
STARKLANG/docs/compiler/semantic-freeze/CORE-V1-COMPLETENESS.md, starkc/src/interp.rs,
starkc/src/typecheck.rs, starkc/src/borrowck.rs, starkc/docs/conformance/KNOWN-DEVIATIONS.md,
COMPILER-STATE.md.
RULES: TYPE-FN-001, TYPE-FN-002 (new normative, owner-approved CE1/CE2 under CD-027).
DECISIONS: CD-027.
EVIDENCE: full workspace `cargo test --workspace --all-targets --all-features` **605 passed / 0
failed / 2 ignored** (600 → 605: 3 new interp tests, 2 new typecheck tests); `cargo fmt` +
`cargo clippy --workspace --all-targets --all-features -- -D warnings` clean;
`build-core-spec.py --check` in sync; fixture extraction in sync; check-conformance.py
unchanged (89.8%). All empirical claims verified by running the compiler on real programs
before recording (E0500 rejection, T1/T2/T3 failures pre-fix and outputs post-fix, combinator
outputs incl. pass-through sides, undetermined-generic acceptance).
FOLLOW-UP: DEV-064 owned by C4.5. Workload items 16-22 now have a working oracle; item 23
(Copy aggregate with fn-value field) untested — exercise during C4. The spike reports' "fn
values unsupported" rows are unaffected (spikes are frozen evidence).
NEXT: WP-C4.1 — MIR design review (CE3): draft the backend-neutral verified MIR contract
(STARKLANG/docs/compiler/mir.md) for owner review; the generated-Rust emitter consumes it.

### WP-C4.1 — MIR contract drafted (CE3 review pending) — 2026-07-19
DONE: drafted STARK MIR v0.1 (`STARKLANG/docs/compiler/mir.md`, status PROPOSED) covering every
roadmap-required element: monomorphised-only instances with deterministic injective symbol
naming; closed first-order MirTy set (no Param/Infer); return-place body model; places/
projections with CheckIndex-dominates-Index discipline; total (never-trapping) rvalue set with
every trapping operation as a Checked/Trap terminator carrying category + SourceInfo; no
unwinding/cleanup edges anywhere (abort semantics); Drop as a *statement* with ordinary Bool
drop-flag locals; direct/indirect/runtime callees (FnPtr constants per CD-021/CD-027, closed
versioned RuntimeFn surface); mandatory per-statement provenance with explicit FileId (DEV-006
lesson) and labeled synthetic origins; 13 verifier obligations mapped to WP-C4.3 with MIR-xxxx
safe-failure diagnostics; deterministic versioned textual dump. Five judgment calls flagged for
CE3 in §12. Created WP-C4.1.md.
FILES: STARKLANG/docs/compiler/mir.md (new, PROPOSED),
STARKLANG/docs/compiler/work-packages/WP-C4.1.md (new), COMPILER-STATE.md.
RULES: none — non-normative implementation contract, explicitly subordinate to
CORE-V1-ABSTRACT-MACHINE.md; binding only after CE3 approval.
DECISIONS: none yet — CE3 review is the owner's; WP-C4.2 does not open against an unapproved
contract.
EVIDENCE: design-only; no code changed; workspace baseline 605/0/2 unchanged.
FOLLOW-UP: on approval, record a CD entry flipping mir.md to APPROVED and open WP-C4.2 (scalar
HIR→MIR lowering). DEV-064 fix must land in typecheck before instance collection can rely on
full determination (C4.5 at latest).
NEXT: CE3 owner review of mir.md §12's five questions; then WP-C4.2.

### WP-C4.1 CE3 review outcome (CD-028) — 2026-07-19
DONE: owner CE3 review of the MIR v0.1 contract returned **APPROVE WITH REQUIRED CHANGES**;
all three required changes applied and the contract flipped to APPROVED. (1) Drop moved from
Statement to Terminator (`Drop { place, target }`, no unwind edge) — the review correctly
caught that the statement form violated the contract's own totality invariant, since
destructors are user code that may trap/diverge/mutate; the totality invariant is now stated
in full ("statements/rvalues never trap, never call user code, never diverge") and actually
holds. (2) Option/Result changed from opaque Core runtime types to **logical MIR enums**
(`EnumRef::CoreOption`/`CoreResult`, same aggregate/discriminant/match machinery as user
enums; physical layout stays C5.1/ABI; combinators may remain runtime calls) — the opaque form
had let the current interpreter's representation shape the IR. (3) CheckIndex/Index kept split
but the ordinary integer index local replaced with **opaque IndexProof tokens** binding
base+index+length, consumed only by Index projections on the same base (V-IDX-1/2); Vec
indexing stays on runtime ops in v0.1 (mutable length). Approved unchanged: trapping-ops-as-
terminators (with the one-normal-successor/implicit-abort refinement made explicit) and
monomorphised-only MIR (with three qualifications: mangling not a stable external ABI; named
resource limit; deduplicated discovery). Owner decision wordings recorded verbatim in mir.md
§12.
FILES: STARKLANG/docs/compiler/mir.md (APPROVED), STARKLANG/docs/compiler/work-packages/
WP-C4.1.md (closed), COMPILER-STATE.md.
RULES: none — implementation contract, subordinate to CORE-V1-ABSTRACT-MACHINE.md.
DECISIONS: CD-028 (owner CE3).
EVIDENCE: design review only; no code changed; workspace baseline 605/0/2 unchanged.
FOLLOW-UP: none blocking. DEV-064 (undetermined-generic coercion rejection) still owned by
C4.5; the CD-021 callable-ABI memo still recommended pre-C5.1.
NEXT: WP-C4.2 — typed HIR → MIR lowering, scalar core (literals/locals, unary/binary ops,
blocks/assignments, functions/calls, if/loops/break/continue/return, tuples/arrays/structs/
basic enums, pattern matching without advanced drop elaboration), with every MIR instruction
carrying real or labeled-synthetic SourceInfo.

### WP-C4.2 — Typed HIR → MIR lowering, scalar core — 2026-07-19
DONE: implemented the MIR v0.1 data model (`starkc/src/mir/mod.rs`) exactly per the approved
contract — Drop as terminator, logical Option/Result enums (EnumRef::CoreOption/CoreResult),
IndexProof local kind, Checked with one normal successor + TrapInfo, closed RuntimeFn surface,
interned FileId + SourceInfo on every statement/terminator, versioned deterministic dump — and
the scalar-core lowering (`src/mir/lower.rs`): monomorphised-only deterministic deduplicated
instance discovery from main; trapping ops as Checked terminators (int arith/neg, float
div/rem) with float add/sub/mul + comparisons as total rvalues; short-circuit &&/|| as CFG;
if/while/loop/for-range (labeled synthetic provenance)/break/continue/return; direct calls;
FnPtr constants + FnValue indirect calls (CD-021 items 16/17); tuples/arrays/structs
(written-order eval, decl-order aggregation); user enums incl. unit variants + struct-variant
literals; Option/Result construction as logical-enum aggregates and matching via
Discriminant+SwitchInt with VariantField binding; println/print via runtime surface with
uniform checked widening casts. Scalar-core drop restriction: Drop-impl types are Unsupported
(C4.5 owns elaboration). New `pub mod mir` in lib.rs.
FILES: starkc/src/mir/mod.rs (new), starkc/src/mir/lower.rs (new), starkc/src/lib.rs,
starkc/tests/mir_lowering.rs (new, 6 tests), STARKLANG/docs/compiler/work-packages/WP-C4.2.md
(new), COMPILER-STATE.md.
RULES: none — implementation of the approved contract; no Core semantics change; front-end
checks not bypassed (lowering consumes fully-checked typed HIR + TypeTables).
DECISIONS: none at CE level.
EVIDENCE: `cargo test --test mir_lowering` 6/6 (corpus scalar cases expr_stmt__01/__03,
primitive__01/__02, struct_enum_trait__02 lower with structural invariants — sealed
single-terminator blocks, in-bounds targets, valid FileId everywhere; dump deterministic +
versioned; golden mini-dump pinning Checked-Add/Cast/runtime-call/return-place shapes;
fn-value + indirect-call lowering incl. instance discovery of the target; Option lowers as
aggregate+discriminant with no runtime call; generics/strings/methods report clean Unsupported
naming C4.5). Full workspace 611 passed / 0 failed / 2 ignored (605 → 611). fmt + clippy
-D warnings clean.
FOLLOW-UP: golden documents that unsuffixed int literals infer Int32 and println's Int64
runtime signature forces an explicit (infallible, still Checked) widening cast — revisit cast
uniformity only via a contract version bump. Bool matches without a default arm and bitwise
int ops are recorded Unsupported (contract's non-trapping BinOp set lacks int bitwise ops —
flag for the C4.5-era contract addendum + version note).
NEXT: WP-C4.3 — MIR verifier (contract §10's 13 obligations, MIR-xxxx diagnostics, safe
failure); then WP-C4.4 MIR interpreter differential vs the HIR oracle.

### WP-C4.3 — MIR verifier — 2026-07-19
DONE: implemented `starkc/src/mir/verify.rs` — all 13 contract §10 obligations over MirProgram:
CFG/local/projection well-formedness with step-by-step place typing through a new
lowering-populated TypeContext (struct fields + user-enum variant payloads added to MirProgram
as an additive companion table; Option/Result payloads derived from type args); bidirectional
aggregate checking; call/checked/runtime signature checking; V-MOVE-1 as a conservative
whole-local any-path union-join fixpoint dataflow; drop-flag and index-proof (CE3 tokens)
discipline; TYPE-FN-001 enforcement at MIR level (no arithmetic/comparison on FnPtr); V-SRC-1
FileId validity. First MIR-xxxx namespace allocation recorded in the Diagnostic-codes section.
Safe-failure hardening: the negative test suite caught the move dataflow PANICKING on a broken
CFG edge (exactly the unsafe failure the contract forbids) — fixed to skip already-reported
edges; report-and-continue everywhere.
FILES: starkc/src/mir/verify.rs (new), starkc/src/mir/mod.rs (TypeContext + MirProgram.types),
starkc/src/mir/lower.rs (type-context population + hir_field_ty), starkc/tests/mir_verify.rs
(new, 14 tests), STARKLANG/docs/compiler/work-packages/WP-C4.3.md (new), COMPILER-STATE.md.
RULES: none — verifier implements the approved contract; no Core semantics change.
DECISIONS: none at CE level. MIR-0012 reserved rather than allocated (runtime-set violation is
structurally impossible while RuntimeFn is a closed Rust enum; becomes real with serialized
MIR).
EVIDENCE: `cargo test --test mir_verify` 14/14 — positive: all 5 lowerable corpus cases + 3
inline programs (fn-values, Option, structs) verify clean (lowering and verifier as two
independent contract readings agreeing); negative: 13 hand-crafted invalid bodies each
rejected with the specific MIR-xxxx code. Full workspace 625 passed / 0 failed / 2 ignored
(611 → 625: 14 verifier tests). fmt + clippy -D warnings clean.
FOLLOW-UP: V-MOVE-1 whole-local granularity documented as a refinement point (can reject
over-clever legal MIR, never accepts moved-from reads); field-precise tracking when C4.5's
partial moves need it. TypeContext addition noted as additive (no dump/shape change, no
version bump) — fold into the contract text at the next version bump.
NEXT: WP-C4.4 — MIR interpreter + differential harness vs the HIR oracle over corpus v1.0.0.

### WP-C4.4 — MIR interpreter + HIR/MIR differential — 2026-07-19
DONE: implemented `starkc/src/mir/interp.rs` (executes verified MIR: option-slot locals with
loud use-after-move detection via taking Moves; projection reads/writes; Checked terminators
with per-width trap semantics incl. MIN/-1 and CD-006 float div/rem-by-zero; checked numeric
casts; SwitchInt with the lowering's u128 key wrap; direct/indirect/runtime calls; 50M-step
fuel guard; TrapCategory outcomes distinct from internal errors) and the Gate C4 comparator
`tests/mir_differential.rs`: 7 tests running lower→verify→execute vs the HIR oracle — 5
lowerable frozen-corpus cases (byte-equal stdout+status; primitive__02 traps agree), fn-values
(CD-021 items 16/17/22 through MIR), Option/Result logical enums end-to-end, structs/tuples,
div-zero trap, mid-output trap, recursion+loops. `interp::canonical_float` exposed pub so the
MIR runtime formats floats with the oracle's own algorithm (single source, no drift).
RESULT: **zero semantic differences between HIR and MIR execution** across the supported
workload. One comparator-map bug caught by the harness itself (oracle "division by zero" vs
map's "divide by zero") — a harness fix, not an engine disagreement.
FILES: starkc/src/mir/interp.rs (new), starkc/src/mir/mod.rs (module reg),
starkc/src/interp.rs (canonical_float made pub with doc), starkc/tests/mir_differential.rs
(new, 7 tests), STARKLANG/docs/compiler/work-packages/WP-C4.4.md (new), COMPILER-STATE.md.
RULES: none — differential infrastructure; no Core semantics change. The MIR interpreter is
explicitly not a user-facing VM (charter §1.6 rule 11).
DECISIONS: none at CE level.
EVIDENCE: `cargo test --test mir_differential` 7/7; full workspace 632 passed / 0 failed /
2 ignored (625 → 632); fmt + clippy -D warnings clean. The C4.4 comparator condition — HIR
interpreter output/failure == MIR interpreter output/failure — holds for every workload the
scalar-core lowering supports.
FOLLOW-UP: the differential net must widen with every C4.5 construct as it lands (the roadmap's
"generated corpus" + full-corpus replay obligations, carried per CD-018/CD-020).
NEXT: WP-C4.5 — complete Core lowering (generics/monomorphisation, trait dispatch, patterns,
CheckIndex/indexing, strings/Vec/runtime surface, ownership/drop elaboration with real Drop
terminators, panic paths, multi-package linkage), differential-first.

### C4.5a + CD-029 correction pass — 2026-07-19
DONE: (1) WP-C4.5 split per charter §2.2 with the review-adopted increment order (WP-C4.5.md).
(2) C4.5a landed: FnKey instance identity (Top/ImplFn/TraitDefault-per-implementing-type),
method + associated-fn call lowering (receiver-before-arguments), trait dispatch with
inherent > trait-impl > default precedence, Self substitution; interim by-value reference
model documented in code (&self receivers Copy-passed; &mut self cleanly Unsupported until
C4.5b/d); corpus struct_enum_trait__01 now differential-green; 2 new differential tests
(methods/assoc fns incl. repeated &self + consuming self; trait default-vs-override).
(3) CD-029 corrections applied (see decision log): trap provenance end-to-end with exact-span
differential comparison; VerifiedMirProgram wrapper; TypeContext formalized in mir.md §2;
canonical_float spec tests (6, incl. boundary/subnormal/round-trip property).
FILES: starkc/src/mir/{lower,interp,verify}.rs, starkc/tests/{mir_differential,mir_lowering,
mir_verify,canonical_float}.rs (last new), STARKLANG/docs/compiler/mir.md (CD-029 amendments),
STARKLANG/docs/compiler/work-packages/WP-C4.5.md (new), COMPILER-STATE.md.
RULES: none — implementation + contract bookkeeping under the approved MIR contract.
DECISIONS: CD-029.
EVIDENCE: differential 9/9 with provenance comparison live (user-origin trap spans equal the
oracle's exactly in both trap tests); canonical_float 6/6; full workspace 640 passed / 0
failed / 2 ignored; fmt + clippy clean. Differential claim now stated in qualified form.
FOLLOW-UP: generated-Rust backend must consume VerifiedMirProgram when it lands (C5).
NEXT: WP-C4.5b — indexing and references (CheckIndex proof tokens, arrays/slices, real
reference places replacing the interim by-value model, &mut self).

### C4.5b-1 — array indexing with CheckIndex proof tokens — 2026-07-19
DONE: first real exercise of the CE3 proof-token discipline end to end. Lowering: `base[index]`
(reads, writes, loop-indexed access) emits `Checked { CheckIndex, args: [Copy(base_place),
index] }` defining an IndexProof local consumed by `Index(proof)` projections; base evaluated
before index (CD-007); non-place bases materialize a temp; Vec indexing stays runtime-surface,
slices deferred to C4.5b-2. Verifier: NEW same-base binding pass (`verify_index_bindings`) —
every Index(proof)'s place prefix must equal the base its CheckIndex bound (proof→base map;
place prefix equality; the exact rule CD-028's revision demanded beyond dominance), plus
CheckIndex arg typing (base must be Copy(place) of indexable type, index integer). Interp:
CheckIndex evaluates bounds and defines the proof as the checked index; place reads/writes
resolve proofs (writes pre-resolve before the mutable walk). ORACLE FIX (DEV-065, found by the
differential's category↔message mapping need): array OOB reported "use of moved or invalid
field" — now projection-kind-aware "index out of bounds"; diagnostics-only.
FILES: starkc/src/mir/{lower,verify,interp}.rs, starkc/src/mir/mod.rs (PartialEq on
Place/Projection), starkc/src/interp.rs (DEV-065), starkc/tests/{mir_differential,mir_verify}.rs,
starkc/docs/conformance/KNOWN-DEVIATIONS.md (DEV-065 closed; count 63), COMPILER-STATE.md.
RULES: none — implements the approved contract; DEV-065 is diagnostics-only (no
accepted/rejected or trap-behaviour change).
DECISIONS: none at CE level.
EVIDENCE: differential 11/11 (new: array reads/writes/loop-sum agree; OOB trap agrees in
category AND exact source span through the fixed oracle message); verifier 15/15 (new negative:
proof bound to base _1 used to index _2 → MIR-0010). Full workspace 643 passed / 0 failed / 2
ignored; fmt + clippy clean.
FOLLOW-UP: C4.5b-2 (references/slices/&mut self) needs the MIR-interp frame restructure
(cross-frame reference places) — the interim by-value reference model stays until then.
NEXT: WP-C4.5b-2, then C4.5c generics per WP-C4.5.md's increment order.

### C4.5b-2 — real references and the frame-stack MIR interpreter — 2026-07-19
DONE: the interim by-value reference model is gone. MIR interpreter restructured onto an
explicit frame stack; a reference value is a resolved (frame, local, concrete-projection-path);
`Deref` re-anchors place resolution; index proofs resolve in the evaluating frame before any
re-anchor; dangling-frame access is a loud Internal error (defense behind borrowck). Lowering:
`Ty::Ref` converts to real `MirTy::Ref` (peel removed); `&expr`/`&mut expr` lower to `RefOf`
(borrow of a place, never a value read); `*r` reads/writes via `Deref` projections; field
access and method dispatch auto-deref through reference-typed bases; `&self`/`&mut self`
receivers are real Ref-typed params (borrowed at call sites, forwarded when the receiver is
already a reference). The &mut-self Unsupported is gone — a &mut self method now genuinely
mutates the CALLER's local across the frame boundary (differential-verified). ORACLE FIX
(DEV-066, the differential's second front-end find after DEV-065): borrowck consumed a
reference on every deref-read (&mut T non-Copy → "use" became a move), rejecting the canonical
`*r = *r + 1`; both deref paths now availability-check without consuming; the
move-out-of-non-Copy-pointee rejection is unchanged.
FILES: starkc/src/mir/interp.rs (frame restructure, rewritten), starkc/src/mir/lower.rs,
starkc/src/borrowck.rs (DEV-066), starkc/tests/{mir_differential,mir_lowering}.rs,
starkc/docs/conformance/KNOWN-DEVIATIONS.md (DEV-066; count 64),
STARKLANG/docs/compiler/work-packages/WP-C4.5.md (b marked done; slices explicitly deferred to
C4.5e where their consumers live), COMPILER-STATE.md.
RULES: none — implements the approved contract's reference/Deref semantics; DEV-066 restores
spec-legal programs (rejection-of-legal fix, no new acceptance beyond the spec).
DECISIONS: none at CE level.
EVIDENCE: differential 14/14 — all prior tests pass unchanged under the REAL reference model,
plus 3 new: `&mut self` mutating the caller's local (read back both via method and direct
field), `&`/`&mut` arguments with cross-frame writes and derefs, `&mut` to a struct FIELD
(sibling field untouched). mir_lowering negative case swapped (mut-self now supported; `?`
takes its place). Full workspace 646 passed / 0 failed / 2 ignored; fmt + clippy clean.
FOLLOW-UP: none blocking. C4.5b complete.
NEXT: WP-C4.5c — generics and full static dispatch (real Instance.type_args monomorphisation,
deterministic dedup, named resource limit, operator dispatch on generic params, DEV-064's
typecheck rejection).

### WP-C4.7-1 — documentation/evidence reconciliation (coding-session remainder) — 2026-07-20
DONE: the three remaining C4.7-1 items from the plan (the doc half landed in the planning
commit). (1) **MIR amendment A3 recorded in `mir.md`** — the WP-C4.6 A5 arithmetic additions,
which CD-033 approved as a *class* but whose per-amendment recording the versioning policy
requires and which was missed at implementation time: `MirBinOp::BitAnd/BitOr/BitXor` as PURE
rvalues (same-width two's-complement results are always representable, so the §5 totality
invariant holds; `~x` lowers to `x ^ mask` rather than adding a `MirUnOp`), `CheckedOp::Pow`
(NUM-INT-ARITH-001, nonnegative exponent, checked intermediates), `CheckedOp::Shl`/`Shr`
activated (NUM-SHIFT-001 count bound, no masking), and `TrapCategory::InvalidShift` kept
DISTINCT from `IntegerOverflow` (a left shift still overflows on an unrepresentable result) with
the reference interpreter's `CheckedOutcome::Trap(Some(cat))` override documented as the rule a
backend must reproduce — it is the only category override in the evaluator. §5/§6 grammar blocks
updated to match. (2) **DEV-074** numbered: the A4-2e alignment of the oracle's three
slice-bound messages into the "out of bounds" family — an oracle *behavior* change that §0.5
says needs a ledger entry, previously recorded only in A1 rev. 10. CLOSED at creation (the code
is correct and spec-directed; the gap was governance). (3) A4's "complete" claim tightened to
"MIR runtime surface" in `WP-C4.6.md` and A1 rev. 10, with the front-end `core-min` holes
(`Box` deref, primitive `cmp`) pointed at WP-C4.7-6. (`Box` deref was later found
**misclassified** — spec-conformant to reject; see the WP-C4.7-6.1 record.)
FILES: STARKLANG/docs/compiler/mir.md (A3 amendment + grammar), mir-amendment-A1-strings-runtime.md
(rev. 10 wording + DEV-074 pointer), work-packages/WP-C4.6.md (A4 wording), work-packages/
WP-C4.7.md (tracker + A3→A4 renumber), starkc/docs/conformance/KNOWN-DEVIATIONS.md (DEV-074;
count 71 → 72; both enumerations), COMPILER-STATE.md (Position, open-deviation index refreshed
to post-Class-A reality with C4.7 owners, count 66 → 72, this record).
RULES: none — doc-only; no code, no test, no behavior change.
DECISIONS: **two items for the owner.** (a) Post-hoc **CE3 ratification of MIR amendment A3** —
the shape additions are already implemented and shipped; the ask is ratification of the record,
not of new code. (b) **Amendment renumbering**: because this increment names the A5 arithmetic
work "A3" (as WP-C4.7 §2 C4.7-1 directs), C4.7-3's layout amendment is renumbered **A4**
(`mir-amendment-A4-layout.md`) — the plan as written would have produced two A3s.
EVIDENCE: doc-only increment; full validation gate run anyway (workspace tests, fmt, clippy on
1.93 and 1.97) to keep the per-increment discipline honest.
FOLLOW-UP: none blocking.
NEXT: WP-C4.7-2 — verifier negatives (5–6 hand-built `MirBody` cases) + clean-Unsupported
fixtures for every recorded Class-A residual, each probed with `c46_probe` first.

### WP-C4.7-2 — evidence symmetry: verifier negatives + unsupported fixtures — 2026-07-20
DONE: CD-033's evidence rule says every Class-A class carries hand-built verifier negatives and
every recorded residual is pinned by a clean-Unsupported fixture. Both halves now hold.
**Verifier negatives (6, hand-built `MirBody`s in `tests/mir_verify.rs`)** — each checked to fail
for the *intended* reason, not incidentally (verified by temporarily asserting a bogus code and
reading the actual message): `rejects_bitwise_binop_on_floats` ("bitwise BinOp on Float64",
MIR-0004 — amendment A3's integer-only rule); `rejects_pow_on_non_integer_dest` (MIR-0004 — `Pow`
must not become a float power op with different trapping); `rejects_vec_get_ref_with_wrong_dest`
(MIR-0005 — the schematic-in-T signature must not degrade to "any Option of any reference");
`rejects_chars_iter_next_on_non_iterator` (MIR-0005, fixed table);
`rejects_runtime_call_arity_mismatch` (MIR-0005 — the plan's suggested
`rejects_call_arity_against_instance` did NOT exist, so the arity path is pinned here instead of
skipped); `rejects_switch_on_float` ("SwitchInt scrutinee is non-integer Float64", MIR-0004 —
pins that A2's Char-scrutinee widening did not over-widen).
**Unsupported fixtures (4, in `unsupported_constructs_report_cleanly`)**: droppable scrutinee +
nested pattern ("A2 residual"), droppable Iterator Item, `&mut base[range]`, `unwrap_or` on a
droppable payload. Every one probed with `c46_probe` (LOWER-UNSUPPORTED) *and* `oracle_run`
(ORACLE-OK) before being added, so each demonstrably pins a MIR gap rather than a front-end one;
`front_end_src` re-asserts typecheck-cleanliness on every run. A stale comment block above the
case table (describing a generic-comparison case that no longer exists) was removed.
FINDING (changes WP-C4.7-8's shape): the plan's fixtures for **method-own generic parameters**
and **non-bare impl heads** cannot live in this test because they are **front-end-blocked** —
`impl Holder { fn first<U>(&self, a: U, b: U) -> U }` + `h.first(7, 9)` fails E0001 "expected
'U', found 'Int32'" (method-own params are not substituted at the call site at all), and
`impl<T> Wrap for Holder<Vec<T>>` + `h.wrapped()` on `Holder<Vec<Int32>>` fails E0302 "method
'wrapped' not found" (method resolution does not structurally unify non-bare impl heads, though
DEV-073 records that it does handle bare-param heads). Neither reaches lowering, so by §1's rule
both are front-end work first. C4.7-8.4/8.5 annotated in the plan.
FILES: starkc/tests/mir_verify.rs (+6 tests), starkc/tests/mir_lowering.rs (+4 fixtures, stale
comment removed), STARKLANG/docs/compiler/work-packages/WP-C4.7.md (tracker + 8.4/8.5 notes),
COMPILER-STATE.md.
RULES: none — tests only; no compiler behavior changed.
DECISIONS: none at CE level. (CD-035 from C4.7-1 still awaits owner ratification.)
EVIDENCE: workspace 752 passed / 0 failed / 2 ignored (+6); fmt clean; clippy clean on 1.93 and
1.97.
FOLLOW-UP: none blocking.
NEXT: WP-C4.7-3 — research C2.9's target-layout decision, then DRAFT `mir-amendment-A4-layout.md`
(`Rvalue::LayoutQuery`) and STOP for owner CE3 approval before writing any code.

### WP-C4.7-3 — type-preserving layout queries (MIR amendment A4, CD-036) — 2026-07-20
DONE: research → CE3 draft → owner approval → implementation, in that order (the plan's
mandatory stop was honored; no code was written before approval).
RESEARCH: the plan asked what C2.9 actually decided about target results. Answer: **CD-015
approved only that `size_of`/`align_of` are the sole target-layout exposures and that Core
promises no ABI — it fixed no per-type values.** 07's LAYOUT-QUERY-001 requires positive,
compile-time/runtime-consistent values satisfying array/field placement; LAYOUT-ABI-001 says the
values may differ between named targets and compiler versions. So the numbers are C5.1's target
contract by design, and the C4 defect is purely representational: WP-C4.6 A4-1 lowered both
builtins to `Const 8` with `T` ERASED, and the HIR oracle returns `Value::Int(8)` for every type
— the differential passed only because both engines shared one placeholder.
IMPLEMENTED (amendment §6 scope, exactly): `Rvalue::LayoutQuery { kind: LayoutKind, ty: MirTy }`
+ dump `layout_size_of(<ty>)` / `layout_align_of(<ty>)` (`mod.rs`); the
`Res::Builtin(SizeOf|AlignOf)` arm now reads the call's turbofish type through `hir_field_ty`,
which applies the active `param_subst`, so a query inside a monomorphised generic body records
the INSTANTIATION's concrete type (`lower.rs`); one verifier typing rule — dest `UInt64`, else
MIR-0004, with the queried type deliberately unconstrained because `Sized`-ness is the checked
front end's property (`verify.rs`); one `eval_rvalue` arm delegating to a new
`reference_layout(ty) -> (u64, u64)` returning `(8, 8)` — the single override point a C5 backend
replaces (`interp.rs`). Rust's exhaustiveness checking usefully forced the new variant through
all four verifier operand/place analyses (move dataflow, drop-flag discipline, proof-token scan,
place collection); a layout query has no operands and no places, so each arm is empty by
construction rather than by assumption.
BEHAVIOR: unchanged, deliberately. The HIR oracle was NOT touched, and `size_of_align_of_agree`
passes **unmodified** — that it needed no edit is the evidence that A4 moved the representation
and not the semantics.
FILES: STARKLANG/docs/compiler/mir-amendment-A4-layout.md (new, APPROVED), mir.md (amendment
list + A4 paragraph + §5 rvalue grammar + §11 dump grammar), starkc/src/mir/{mod,lower,verify,
interp}.rs, starkc/tests/{mir_lowering,mir_verify}.rs, WP-C4.7.md, COMPILER-STATE.md.
RULES: LAYOUT-QUERY-001 and LAYOUT-ABI-001 (07), 06's "target-layout queries" classification.
No spec edit was needed — the normative documents already said what A4 implements.
DECISIONS: **CD-036** (above). CD-035 (amendment A3 record) ratified by the owner in the same
exchange.
EVIDENCE: 4 new tests — `layout_queries_preserve_the_queried_type` (dump golden: primitive and
nominal types survive; the old bare constant is gone),
`layout_query_inside_a_generic_body_records_the_instantiation` (Int32 and Bool instances each
record their own type), `rejects_layout_query_with_non_uint64_dest` (MIR-0004),
`accepts_layout_query_of_any_type_into_uint64` (an unsized queried type is a legal question).
Workspace 756 passed / 0 failed / 2 ignored; fmt clean; clippy clean on 1.93 and 1.97.
FOLLOW-UP: C5.1 replaces `reference_layout` with the named target's real layout algorithm. That
is the only place it must touch.
NEXT: WP-C4.7-4 — DEV-069, front-end multi-file span discipline (typecheck + borrowck, then the
oracle, as two commits; reproduce with throwaway two-file probes first).

### WP-C4.7-4 — DEV-069: multi-file span discipline in the front end and the oracle — 2026-07-20
DONE: **DEV-069 CLOSED**, discharging CD-033's C5 multi-file prerequisite.
ROOT CAUSE (one class, not four bugs): `typecheck.rs`, `borrowck.rs`, and `interp.rs` each hold a
single "current file" and read every `Span` against it. STARK parses each file of a `mod helper;`
program separately, so spans are FILE-RELATIVE. Reading a span against the current file is
correct for the item being CHECKED — `check_crate` already swapped `self.file` per item — and
silently wrong for every item being LOOKED UP, because the lookup scans (method resolution,
trait-default fallback, associated-fn search, `Drop` discovery, nominal name formatting) walk
ALL items in the program regardless of file. That single mistake produced all four documented
shapes: an out-of-bounds panic when the dependency file was longer, garbage method names,
unparseable literals, and wrong-field reads at runtime.
FIX, two mechanisms:
1. **`item_text(item, span)`** in all three modules, reading against the file that DECLARES
   `item` via `hir.item_files` — the map the resolver already populated and MIR's `ProgramMeta`
   already relies on, so the three engines now agree on one source of file identity. Applied to
   every cross-item read found by walking the scan loops: method resolution, trait defaults
   (which take the TRAIT's file, not the impl's), associated fns, `Drop` impls, `format_nominal`,
   `item_name`.
2. **Per-body file swap in the oracle**, which never swapped file at all. `Callable` now carries
   its declaring file, and all THREE body-execution funnels save/restore `self.file` around the
   body: `call_callable`, `call_user_method`, and the destructor path in `drop_value`. Restored
   on error paths too, and AFTER `cleanup_current_frame` on success, since a body's own
   destructors still belong to its file. Finding the second and third funnels took empirical
   probing — fixing only `call_callable` left cross-file methods broken, and fixing that left
   cross-file destructors broken.
`text()` is additionally non-panicking now in all three modules (`.get(..).unwrap_or("?")`): a
residual wrong-file read degrades to a visible `"?"` in a diagnostic instead of aborting the
compiler. That is a backstop, not the mechanism.
FILES: starkc/src/{typecheck,borrowck,interp}.rs, starkc/tests/multi_file_spans.rs (new),
starkc/tests/mir_differential.rs (widened), KNOWN-DEVIATIONS.md (DEV-069 closed + both
enumerations), WP-C4.7.md, COMPILER-STATE.md.
RULES: none normative — this is an implementation defect against 07-Modules-and-Packages'
multi-file model; no spec text changed and no accept/reject decision changed for single-file
programs (759 tests, all pre-existing ones unchanged).
DECISIONS: one deviation from the plan, recorded: the plan said do this in TWO commits
(typecheck+borrowck, then the oracle). Landed as ONE, because the regression tests exercise both
halves end-to-end — a typecheck-only commit would have pushed red tests, which the per-increment
green-CI rule forbids. The two halves are separable in review by module.
EVIDENCE: `tests/multi_file_spans.rs` — one test per failure shape, each checked AND executed:
cross-file methods/fields/literals (33/11/66/12345), a long-dependency-file panic guard, and
cross-file trait dispatch + `Drop` where destructor ORDER is the observable (40/1/4). The
multi-file differential test was WIDENED off the safe subset it had been pinned to — now a
cross-file struct with methods, a cross-file literal, a cross-file field read, and a cross-file
`Drop` impl — with the exact expected output asserted so two engines agreeing on nothing cannot
pass. Workspace 759 passed / 0 failed / 2 ignored (+3); fmt clean; clippy clean on 1.93 and 1.97.
FOLLOW-UP: none. C5 may now claim normal multi-file support.
NEXT: WP-C4.7-5 — DEV-072 (borrowck: move-out-of-shared-borrow via match bindings) and DEV-073
(typecheck: generic impls satisfying operator/iterable bounds).

### WP-C4.7-5 — DEV-072 + DEV-073 (front-end typecheck/borrowck) — 2026-07-20
DONE: both deviations CLOSED. They are opposite failures — one over-rejection, one
under-rejection — and both came down to the front end and MIR answering the same question with
different machinery.
**DEV-073 (over-rejection, typecheck).** The visible symptoms were `impl<T> Eq for W<T>` not
satisfying `W<Int32>: Eq` (E0500) and `impl<T> Iterator for Repeat<T>` not making `Repeat<Int32>`
iterable (E0001). The root cause sat one level below both checks:
`type_from_hir_without_diagnostics` **drops generic arguments** (`Ty::Struct(item, Vec::new())`).
That is invisible while the only consumers compare NON-generic nominals — `struct P` converts to
`Struct(id, [])` either way — but it meant an impl's written `W<T>` converted to `W<>`, whose
argument count could never equal `W<Int32>`'s, so the exact-match test failed for every generic
impl. Fix: a new `impl_self_ty_with_args(impl_item, ty)` that preserves the arguments and keeps
parameters as `Ty::Param`, with both checks unifying through **`match_impl_type`** — the same
one-way unification METHOD RESOLUTION already used for this exact question. That asymmetry is why
method calls on generic nominals had always worked while operators and `for` loops on the same
types did not; the two paths now agree by construction. The iterable half additionally applies
the resulting substitution to the associated type, so `type Item = T` on `Repeat<Int32>` yields
`Int32` instead of a dangling parameter.
**MIR needed no change at all** — WP-C4.6 A1 had already made dispatch instantiation-ready, and
both programs lowered and ran correctly the moment the checker admitted them. The plan predicted
this and flagged that a lowering break would be a real finding; there was none.
**DEV-072 (under-rejection, borrowck).** `borrowck.rs`'s `match` handling inspected no patterns
whatsoever, so binding a non-`Copy` payload out of a scrutinee read through a reference — a move
out of a borrow — passed the front end while MIR refused it. The two engines disagreed about
whether the program was legal, and the oracle's legacy clone semantics hid the unsoundness at
runtime by consuming the clone rather than the referent. Fix: borrowck now classifies the
scrutinee with `scrutinee_reads_through_ref`, a deliberate mirror of MIR lowering's function of
the same name (so the classification cannot drift again), and walks each arm's pattern
recursively — nested tuple/array/struct patterns and shorthand struct-field bindings included —
reporting E0101 for any non-`Copy` binding. Shared and mutable derefs both count.
What stays LEGAL mattered as much as what does not: wildcards, literals, and unit-variant paths
bind nothing, and `Copy` bindings copy rather than move. A fix that rejected all by-reference
matching would have been "correct" against the repro while breaking far more than it repaired, so
both positives are pinned by tests. The MIR guard is KEPT as defense in depth, with its message
updated to say it is unreachable for checked programs — the charter's rule is that nothing
unsupported reaches a backend silently, and an unreachable guard costs nothing.
FILES: starkc/src/typecheck.rs (`impl_self_ty_with_args`, operator-bound + iterable checks),
starkc/src/borrowck.rs (`scrutinee_reads_through_ref`, `reject_moves_out_of_borrow`),
starkc/src/mir/lower.rs (guard comment only), starkc/tests/{mir_differential,gate2_valid}.rs,
KNOWN-DEVIATIONS.md (both closed, both enumerations), WP-C4.7.md (tracker + the now-stale §1
quirk notes struck), COMPILER-STATE.md.
RULES: 03-Type-System operator traits and the `Iterator` for-protocol (DEV-073); the ownership
rule that a borrow never transfers ownership (DEV-072). No spec text changed.
DECISIONS: none at CE level.
EVIDENCE: `mir_differential.rs::generic_impl_eq_dispatch_agrees` and
`::generic_user_iterator_for_loop_agrees` — the two tests DEV-073 had blocked, added back per the
plan; `gate2_valid.rs::binding_a_non_copy_payload_through_a_reference_is_rejected` (E0101) and
`::matching_through_a_reference_without_taking_ownership_is_accepted` (wildcard + Copy positives);
`match_deref_self_noncopy_wildcard_agree` still green unchanged. Workspace 763 passed / 0 failed /
2 ignored (+4); fmt clean; clippy clean on 1.93 and 1.97.
FOLLOW-UP: none.
NEXT: WP-C4.7-6 — front-end `core-min` completions. 6.1 `Box<T>` deref (report before implementing
the MIR half if it needs a new MirTy/runtime op — §0.5 stop), 6.2 primitive `cmp`/`Ordering`,
6.3 the integer-literal-typing question, which is an **OWNER DECISION** (check 03's literal
inference rules first; if 03 forbids the coercion, record as spec-conformant and close as
"not a bug").

### WP-C4.7-6 — front-end `core-min` completions: 6.2 done, 6.1 and 6.3 to the owner — 2026-07-20
DONE (6.2): **primitive `Ord::cmp`.** 06-Standard-Library specifies `impl Ord for Int32 { fn cmp
(&self, other: &Int32) -> Ordering }` and "similar for other types"; `Ordering` is `core-min`
prelude. `3.cmp(&5)` nevertheless failed E0304 "method call on non-struct/enum type" — primitives
had no `cmp` entry at all, so the only way to obtain an `Ordering` value was a user-defined `Ord`
impl. Implemented in all three engines: (a) checker — a `cmp` entry in the core-method surface
returning `Core(Ordering)` with an `&Self` parameter; (b) oracle — evaluated through the existing
`Ord for Value`, i.e. the SAME comparison the `<` operator path and sorted-collection iteration
already use; (c) MIR — `lower_primitive_cmp` computes the comparisons `<`/`==` already lower
(routing `String`/`str` through the existing `StrCmp`) and CONSTRUCTS the `CoreOrdering` variant
from them. That is the exact inverse of `lower_user_ord`, which calls a user `cmp` and switches
on the resulting discriminant. **No new MIR shape, no new `RuntimeFn`, no surface bump** — the
dispatch is placed before the String/Vec/HashMap runtime dispatches, since `String` is a
primitive receiver for this purpose. Both operands are read into temps before branching, so each
is evaluated exactly once, receiver before argument (EXEC-ONCE-001).
FOUND WHILE SCOPING 6.2 — **DEV-075**, pre-existing and unrelated to this change: the checker
accepts ordered comparison on `Bool` and `Char`, but `false < true` fails in BOTH engines
("invalid binary operation" / `BinOp Lt on Bool`) — an accept-then-fail — and `'a' < 'b'`
**succeeds in MIR while the oracle rejects it**, an engine divergence of exactly the kind the
differential exists to catch, missed only because no test compares an ordered operator on `Char`.
`cmp` was therefore scoped to integers + `String`/`str` rather than built on this gap; enabling
`Bool`/`Char` belongs in the change that closes DEV-075. Fixing it needs a spec reading — 03
gives primitives "built-in meaning (Numeric Semantics below)", which addresses numeric types and
does not settle `Bool`/`Char` ordering — so it is not a pure code fix. Ledger count 72 → 73.
TO THE OWNER — both remaining items contradict the plan's framing of them:
**6.1 `Box<T>`.** The plan (and the WP-C4.6 audit) called "`Box` deref" a `core-min` hole. The
spec says otherwise: 06 defines `Box<T>` with exactly `new` and `into_inner`; there is **no
`Deref` trait in Core v1** (not among core-min's essential traits); TYPE-METHOD-002's
auto-dereference "repeatedly removes one leading `&`/`&mut`" — references only; and the abstract
machine's Dereference operates on "the reference". So `*Box::new(5)` failing E0001 is
**spec-conformant**, and the audit's classification was wrong. The REAL gap is one level over:
`Box::new(v).into_inner()` is typecheck-clean and oracle-supported but **MIR-unsupported**
("type Core(Box, [...]) (C4.5)"). Closing it is a §0.5-class decision either way — an honest
representation needs `BoxNew`/`BoxIntoInner` runtime ops plus a surface bump, while the tempting
alternative (lower `Box<T>` transparently as `T`, since Core v1 makes addresses unobservable) is
a semantic claim that recursive types through `Box` would break; the front end already accepts
`struct Node { next: Box<Node> }`.
**6.3 integer-literal typing.** The plan hedged that 03 might FORBID a literal adopting an
expected `UInt64`, in which case the item closes as "not a bug". 03 says the opposite, and says
it twice: expected types "flow inward from ... **function parameters** ...", and defaulting
applies only to "an **unconstrained** integer literal". A literal in a `UInt64` parameter
position is constrained, so defaulting to `Int32` must not apply — this is expected-type
propagation, not a coercion (step 4 limits coercions to explicit sites), so it does not collide
with the no-implicit-coercion rule either. `v.get(0)` failing "expected 'UInt64', found 'Int32'"
is therefore a **real conformance bug (over-rejection)**, not spec-conformant behavior.
FILES: starkc/src/{typecheck,interp}.rs, starkc/src/mir/lower.rs, starkc/tests/mir_differential.rs,
KNOWN-DEVIATIONS.md (DEV-075; count 72 → 73; both enumerations), COMPILER-STATE.md.
RULES: 06's `Ord` impls for primitives and `core-min` prelude `Ordering`; CD-015 (floats are not
`Ord`). No spec text changed.
DECISIONS: none taken at CE level; two put TO the owner (6.1, 6.3, above).
EVIDENCE: `mir_differential.rs::primitive_cmp_agrees` (Less/Equal/Greater over integers and
`String`, plus a local receiver) and `::primitive_cmp_and_ordered_operators_agree`, which states
the consistency property as a test rather than assuming it: for the same pair, the variant `cmp`
reports and the answer `<`/`==` give must never disagree. Workspace 765 passed / 0 failed /
2 ignored (+2); fmt clean; clippy clean on 1.93 and 1.97.
FOLLOW-UP: DEV-075; owner decisions on 6.1 and 6.3.
NEXT: blocked on the two owner decisions; C4.7-7 (DEV-067 + DEV-071) is independent and can
proceed meanwhile.

### WP-C4.7-7 — DEV-067 + DEV-071 (bounded generics; Ordering exhaustiveness) — 2026-07-20
DONE: both CLOSED. **With this increment every front-end deviation the C4 track owned is closed.**
The remaining open ledger entries are the long-standing unscheduled ones (DEV-005/010/011/012/017)
and DEV-075, which C4.7-6.2 opened the same day.
**DEV-071 (exhaustiveness).** The prelude `Ordering` is `Ty::Core(CoreType::Ordering)` whose
variants resolve to `Res::Builtin`, which makes it structurally identical to `Option`/`Result` —
and invisible to the `Ty::Enum`/`matched_variants` machinery for exactly the same reason those
two were, before WP-C1.5 gave them explicit arms. `Ordering` never got one, so it fell through to
the same WP-C1.5 default that requires a wildcard for any domain the checker cannot enumerate.
The check now tracks `Less`/`Equal`/`Greater` and treats all three as exhaustive. The enumeration
is exact, and that matters: an over-generous domain would silently accept genuinely non-exhaustive
matches, so a two-variant match staying E0303 is pinned by its own test.
**DEV-067 (bounded generics).** One ledger entry, two independent causes:
- **(b) behind `&T`.** The bounded-parameter method lookup tested the UNPEELED receiver type, so
  it matched `t: T` but never `t: &T`. TYPE-METHOD-002 requires auto-dereference to peel leading
  `&`/`&mut` before receiver matching — and the concrete-type path immediately below already
  computed exactly such a peeled `receiver_ty`. The peel was simply performed *after* the
  parameter check instead of before it; moving it above makes both paths obey one rule.
- **(a) at intra-generic call sites.** `satisfies_bound` had **no `Ty::Param` arm at all** and
  fell through to `_ => false`, so a caller's own `T: Ord` could never discharge a callee's
  (TYPE-GENERIC-001). Adding the arm alone did not fix it — the probe still failed — because
  trait-bound obligations are collected during body checking and verified in a **deferred pass**
  that runs after every body, by which time `current_fn_generics` belongs to whatever was checked
  last. Each obligation now carries the generic environment it arose in, and the deferred pass
  restores it. The new arm mirrors the one `ty_satisfies_operator_bound` already had, so the two
  bound checks finally agree about what a parameter satisfies.
SOUNDNESS: over-rejection removed, nothing newly accepted. An obligation is discharged only by a
bound the enclosing function actually declared — both a concrete type lacking the impl and an
UNBOUNDED parameter forwarded into a bounded position are still E0500, each pinned by a test,
because "relax a check" is exactly the kind of change that silently over-accepts.
FILES: starkc/src/typecheck.rs (exhaustiveness arms; receiver peel order; `Ty::Param` bound arm;
`bounds_checks` carries its generic environment), starkc/tests/{mir_differential,gate2_valid}.rs,
KNOWN-DEVIATIONS.md (both closed, both enumerations), WP-C4.7.md (tracker + the DEV-071 §1 quirk
note struck), COMPILER-STATE.md.
RULES: TYPE-METHOD-002 (auto-dereference before receiver matching), TYPE-GENERIC-001 (the caller's
bound discharges the callee's obligation), 04-Semantic-Analysis exhaustiveness. No spec change.
DECISIONS: none at CE level.
EVIDENCE: `bounded_generic_method_through_reference_agrees` (instantiated at TWO types, so
monomorphised dispatch is exercised and not merely the check), `bounded_generic_call_chain_agrees`
(three-deep bounded chain), `unsatisfied_trait_bounds_are_still_rejected` (both negatives),
`ordering_match_exhaustiveness_counts_all_three_variants` (both directions), and
`ordering_value_round_trips_through_match_agree` **rewritten to three explicit arms** — dropping
the `_` workaround it carried for DEV-071 is what makes it exercise the exhaustiveness path.
Workspace 769 passed / 0 failed / 2 ignored (+4); fmt clean; clippy clean on 1.93 and 1.97.
FOLLOW-UP: none.
NEXT: the two owner decisions (6.1 `Box`, 6.3 literal typing), then C4.7-8 (MIR residuals; 8.6
mutable slices is itself an owner decision) and C4.7-9 (fresh audit + exit report).

### WP-C4.7-6.1 — `Box<T>` on the MIR runtime surface (`0.1-A7`), owner option (a) — 2026-07-20
DONE: `Box<T>` reaches MIR. Implemented exactly to the owner's decision: an **opaque owning**
runtime type, `RuntimeFn::BoxNew` + `BoxIntoInner` activated through the dated A1-amendment
mechanism (rev. 11), surface **`0.1-A6` → `0.1-A7`**, representation stays
`MirTy::Core(Box, [T])` with **no new `MirTy`**, and explicitly NOT lowered transparently as `T`.
AUDIT CORRECTION (owner-directed): the WP-C4.6 gate audit listed "`Box` deref" as a `core-min`
hole. It is not one. Core v1 has **no `Deref` trait** (absent from `core-min`'s essential-trait
list), TYPE-METHOD-002's auto-dereference removes only leading `&`/`&mut`, the abstract machine's
dereference operates on *the reference*, and 06 gives `Box<T>` exactly `new` and `into_inner`.
`*Box::new(5)` failing is therefore **specification-conformant** and is now pinned by a negative
front-end test so a later session cannot "fix" conformant behaviour. The real gap was the
construction/extraction pair — typecheck-clean and oracle-supported, but with no MIR lowering at
all — which is what this increment closes.
SEMANTICS: `BoxNew(T) -> Box<T>` consumes its argument exactly once. `BoxIntoInner(Box<T>) -> T`
consumes the box and transfers the value out **without dropping it** (ownership moves to the
caller), releasing the allocation. There is **no public box-drop operation**: ordinary
destruction goes through the existing `Drop` terminator's structural glue, which drops the
contained `T` exactly once and then releases the allocation. A box consumed by `into_inner` holds
nothing, so nothing drops twice. Allocation failure stays a classified host/resource failure, not
a language trap (the reference interpreter cannot fail to allocate and raises none). Interpreter
representation is a one-element aggregate — addresses are unobservable (LAYOUT-QUERY-001), so the
reference engine models only the observable fact that the box OWNS its value.
THREE PRE-EXISTING DEFECTS surfaced while implementing this; none was in the plan:
1. **Drop-instance discovery never descended into `Core` container type arguments.** A
   `Box<Tag>`'s `Drop` terminator was emitted correctly and then silently found no destructor
   registered — the box dropped nothing at all. The walk now descends into every `Core`
   container's arguments (which also makes the Vec path robust rather than incidentally correct).
2. **That walk had no cycle guard**, which only mattered once `Box` made types recursive:
   `Node -> Option<Box<Node>> -> Box<Node> -> Node` overflowed the stack (observed, not
   theorised). Guarded by a visited-type set — right regardless, since a type's dtor instances
   need discovering once.
3. **DEV-077** (opened and CLOSED here): the oracle's `Box::into_inner` read its receiver through
   the *borrowing* method path, which operates on a CLONE. `.take()` emptied the clone while the
   original box kept the value and destroyed it again at scope end — an observable double drop
   with a `Drop` payload, and a divergence from MIR, which was correct. It now consumes the real
   place via `take_place`, exactly like the pre-existing `File::close` case beside it. The
   differential could not agree until the oracle was right, which is how it was caught.
FILES: starkc/src/mir/{mod,lower,verify,interp}.rs, starkc/src/interp.rs (DEV-077),
starkc/tests/{mir_differential,mir_verify,mir_lowering,gate2_valid}.rs (incl. the two
surface-string goldens the plan's §1 warns about), mir-amendment-A1-strings-runtime.md (rev. 11),
KNOWN-DEVIATIONS.md (DEV-077 closed; count 74 → 75), COMPILER-STATE.md.
RULES: 06's `core-min` `Box<T>`; TYPE-METHOD-002; LAYOUT-QUERY-001 (addresses unobservable);
EXEC-ONCE-001 (the DEV-077 double drop). No spec text changed.
DECISIONS: implements the owner's 6.1 decision (option (a)); no new CE-level decision taken.
EVIDENCE: `box_new_and_into_inner_agree`; `box_drop_timing_agrees` (exact destructor interleaving
— printed ORDER is the assertion, not a multiset); `box_recursive_type_agrees` (a finite value of
a recursive type, which is the whole reason Box stays opaque, and which also pins the cycle
guard); `rejects_box_into_inner_on_non_box` and `rejects_box_new_with_mismatched_dest` (verifier);
`box_deref_is_rejected` (front-end negative). Workspace 775 passed / 0 failed / 2 ignored (+6);
fmt clean; clippy clean on 1.93 and 1.97.
FOLLOW-UP: none for Box.
NEXT: WP-C4.7-6.3 (integer-literal expected typing — owner-decided to fix), then DEV-075
(Char/Bool ordering + the normative primitive trait/operator matrix, which requires spec edits and
regenerating the compiled spec).

### WP-C4.7-6.3 — expected typing of integer literals (DEV-078) — 2026-07-20
DONE, per the owner's decision that this is a real Core conformance defect rather than
spec-conformant behaviour. The evidence for that reading is 03-Type-System stating it twice:
expected types "flow inward from explicit annotations, **function parameters**, return types,
assignment destinations, aggregate fields, …", and solving step 5 defaults only "an
**unconstrained** integer literal". A literal in a `UInt64` parameter position is constrained, so
defaulting must not apply.
PREVIOUS BEHAVIOUR: the checker assigned `Int32`/`Int64` **at the literal**, before any
expectation could reach it. `takes_u64(0)`, `v.get(0)`, `let a: UInt64 = 9`, and a `UInt64`
struct-field initializer were all `E0001 expected 'UInt64', found 'Int32'`. It had been recorded
as a "`Vec::get` literal-typing quirk", which understated it — nothing about it was specific to
`Vec::get`, and the `0 as UInt64` workaround had been trained into the corpus and into WP-C4.7
§1's guidance for test authors.
IMPLEMENTED as general expected-type inference: an unsuffixed literal takes a fresh
**integer-kinded** inference variable; ordinary unification carries the expected type in; and
03's step 5 becomes a real pass (`default_unconstrained_int_literals`) that runs after every body
is checked and before the deferred bound checks. Binding a literal variable **range-checks** the
value (`takes_u8(300)` → E0008 at compile time, not truncation). The kind restriction is what
keeps this from being an implicit-conversion hole: the variable unifies only with primitive
integer types (plus `!` for the never-coercion rule and error-recovery types), so an integer
literal cannot satisfy a `Bool` parameter. And because this is propagation rather than coercion —
03's step 4 confines coercions to explicit coercion sites — a SUFFIXED literal (`0i32`) and a
TYPED value (`x: Int32`) both still fail against `UInt64`, which is the whole point.
TWO PLACES MUST SETTLE A LITERAL EAGERLY, because they branch on a concrete type and have no
later constraint to wait for: method-call receivers (`3.cmp(&5)` — otherwise "method call on
non-struct/enum type '_infer_N'") and cast operands (`5 as UInt8` — otherwise "casts are permitted
only between numeric types").
SUBTLETY WORTH RECORDING: a literal variable is frequently unified with ANOTHER variable rather
than a concrete type — `MyOpt::Some2(7)` unifies it with the enum's element variable. Defaulting
only variables absent from the substitution therefore left such chains unbound while they LOOKED
constrained, and they surfaced as `type Infer(N)` at MIR lowering. Defaulting resolves first and
defaults the end of the chain.
FILES: starkc/src/typecheck.rs (literal site, integer-kinded binding, defaulting pass, eager
settle at receivers/casts, array-repeat count), starkc/src/literal.rs
(`primitive_int_range_contains`), starkc/tests/{gate2_valid,mir_differential}.rs,
KNOWN-DEVIATIONS.md (DEV-078 closed; count 75 → 76), COMPILER-STATE.md.
RULES: 03-Type-System's inference algorithm (inward expected types; step 5 defaulting; step 4
coercion confinement). No spec text changed — the spec already required this.
DECISIONS: implements the owner's 6.3 decision; no new CE-level decision.
EVIDENCE: `unsuffixed_integer_literals_adopt_the_expected_integer_type` (parameter, annotation,
struct field, and the TYPE-INFER-001 later-use case `let index = 0; v.get(index)`);
`integer_literal_typing_negatives_still_fail` (range, suffix, typed value, non-integer kind — four
different reasons, all of which must keep failing); `expected_typed_integer_literals_agree`
(differential — adopted widths are observable at runtime through `UInt64` arithmetic and indexing,
so checker-side agreement alone would not be evidence). Unnecessary `as UInt64` casts removed from
the differential corpus; casts of genuinely typed values retained. Workspace 778 passed / 0 failed
/ 2 ignored (+3); fmt clean; clippy clean on 1.93 and 1.97.
FOLLOW-UP: WP-C4.7 §1's "integer literals don't coerce to `UInt64`" guidance is now obsolete and
has been struck.
NEXT: DEV-075 (Char/Bool ordering + the normative primitive trait/operator matrix — requires spec
source edits and regenerating the compiled spec).

### WP-C4.7 — DEV-075: Char/Bool ordering and the normative primitive matrix — 2026-07-20
DONE: DEV-075 CLOSED under an **owner specification decision**. This is the first change to
normative spec text in WP-C4.7.
THE DECISION (owner, 2026-07-20) split the two types rather than treating DEV-075 as one gap:
- **`Char` is totally ordered by Unicode scalar value** — implements `Eq`, `Ord`, `Hash`; all four
  ordered operators compare scalar values; `Char::cmp` returns the corresponding `Ordering`.
  Explicitly NOT locale-sensitive or linguistic collation, and Core v1 offers no collation
  facility.
- **`Bool` implements `Eq` and `Hash` but NOT `Ord`** — `<`, `<=`, `>`, `>=` and `Bool::cmp` are
  compile-time errors; `==`/`!=` remain valid. An ordering is definable, but Core v1 has no use
  for ordering truth values, and rejecting is clearer than fixing an arbitrary one.
IMPLEMENTED: the divergence ran in `Char`'s favour — MIR executed `'a' < 'b'` correctly while the
oracle rejected it — so the ORACLE was aligned to MIR (a `(Char, Char)` arm in `eval_binary`,
matching Rust's scalar-value `char: Ord`), and `Char` joined the primitive `cmp` surface in both
the checker and lowering. `Bool` was removed from the `Ord` operator gate, which is what turns
`false < true` from an accept-then-fail into a diagnostic.
SPEC CHANGE: **`PRIM-TRAIT-001`**, a normative "Primitive Trait and Operator Matrix" in
06-Standard-Library, replacing the illustrative `impl Eq for Int32` plus `// ... similar for other
types` — which the owner correctly identified as not being a specification at all. 03-Type-System's
operator table now cross-references it. `STARK-Core-v1.md`/`.html`/`.pdf` regenerated via
`build-core-spec.py`; the spec-fixture corpus re-extracted with `extract-spec-examples.sh` (one
fixture changed, 112 blocks, manifest in sync).
THE DISTINCTION THE MATRIX FORCED: for primitives, operators have built-in meaning and do **not**
dispatch through the traits, so the operator question and the trait question are separate. The
float row is where they differ: `Float64` admits `<` and `==` as built-in IEEE operations (CD-006)
while implementing neither `Eq` nor `Ord`, because IEEE comparison is neither an equivalence
relation nor a total order — NaN is unordered and unequal to itself — so `Float64` cannot satisfy
a `T: Ord` bound or key a `HashMap`. Conflating the two gates silently broke ordinary float
comparison during implementation (`1.5 < 2.5` started failing E0500); the operator gate
(`ty_satisfies_operator_bound`) and the trait gate (`satisfies_bound`) now carry the matrix
separately, and both directions are pinned by a test.
FILES: STARKLANG/docs/spec/{06-Standard-Library,03-Type-System}.md (+ regenerated
STARK-Core-v1.{md,html,pdf}), STARKLANG/tests/spec-fixtures/06-Standard-Library__18.stark,
starkc/src/{interp,typecheck}.rs, starkc/src/mir/lower.rs,
starkc/tests/{mir_differential,gate2_valid}.rs, KNOWN-DEVIATIONS.md (DEV-075 closed; both
enumerations), COMPILER-STATE.md.
RULES: new **PRIM-TRAIT-001**; consistent with CD-015 (floats are not `Eq`/`Ord`/`Hash`) and
CD-006 (IEEE float operations).
DECISIONS: owner specification decision, recorded above; no CE-level decision taken by the session.
EVIDENCE: `char_ordering_agrees` (all four operators + `cmp`, both engines) and
`char_ordering_is_scalar_value_not_collation_agrees` — the second deliberately uses `'Z' < 'a'`
and `'0' < 'A'`, comparisons a COLLATION order would get wrong, so it distinguishes the specified
rule from a plausible alternative rather than merely re-testing that comparison works;
`bool_is_not_ordered` (four operators + `Bool::cmp` rejected, `==` still accepted);
`floats_compare_but_do_not_satisfy_ord_bounds` (both sides of the operator/trait distinction).
OBSERVABLE NARROWING (intended, and worth stating plainly): because primitive floats no longer
satisfy `T: Ord`, a bounded generic can no longer be INSTANTIATED at a float —
`fn largest<T: Ord>(..)` called as `largest(2.5, 1.5)` was legal before and is now E0500. One
existing differential test did exactly that; it was updated to instantiate `largest` at `Int32`
and `Char` (both `Ord`) while `twice<T: Num>` keeps the float instantiation, since `Num` does
include floats. That preserves the test's real subject — multiple primitive instantiations of a
bounded generic — and adds positive `Char`-as-`Ord` coverage. This failure only surfaced in a
COMPLETE workspace run; several partial runs never reached `mir_differential`.
FOLLOW-UP: none.
NEXT: C4.7-8. **8.1 is blocked on DEV-076** (the oracle's `unwrap_or` double-drop must be fixed
before MIR is built to match it); 8.4/8.5 were reclassified front-end-first by C4.7-2; 8.6
(mutable slices) is an owner decision.

### WP-C4.7-8.1a — DEV-076: the oracle's `unwrap_or` drop semantics — 2026-07-20
DONE: DEV-076 CLOSED. This is the oracle half of C4.7-8.1, split out and landed on its own
because it is a SOUNDNESS fix that is independently valuable and is a hard prerequisite for the
MIR half — §0.6 makes the oracle the semantics authority MIR must match, and an oracle that
double-drops is not an authority, it is a bug that would have been faithfully copied into MIR.
THE DEFECT: with a `Drop`-carrying payload, `Option::unwrap_or` destroyed the payload **twice**
and the discarded default **never**. Root cause identical to DEV-077: `unwrap_or` was handled on
the *borrowing* method path, which operates on a CLONE of the receiver, so taking the payload
emptied the clone while the original `Option` kept it and destroyed it again at end of scope. The
default fared worse — nothing consumed it, so its destructor never ran at all. (Core has no
laziness, so the default is always *evaluated*, which is exactly why it always owes a
destruction.) Both halves violate EXEC-ONCE-001.
FIX: `unwrap_or` now consumes the receiver from the real place (`take_place`), joining
`into_inner`/`close` at the same interception point, and explicitly drops whichever value it
discards — on `Some`/`Ok` it yields the payload and drops the default; on `None` it yields the
default; on `Err` it yields the default and drops the displaced error payload.
PINNED TIMING (the point of doing this first, and NOT the obvious answer): the discarded default
is destroyed **at the `unwrap_or` call**, not at end of scope. For
`let t = Some(Tag{1}).unwrap_or(Tag{2})` the observable order is `2` then `1`. Before the fix it
was `1`, `1` — the payload twice, the default never. Any MIR lowering written against the old
behaviour would have encoded a double drop into the backend contract.
MIR HALF: still open, still a CLEAN `Unsupported` ("unwrap_or on a droppable payload type"). A
first attempt at the lowering is deliberately NOT in this commit: moving a payload out of a
**drop-tracked** local through a `VariantField` projection is refused by the C4.5d guard ("move
through a non-field projection of a drop-tracked local"), so the consuming path needs the
drop-flag machinery `lower_enum_match` uses (`consume_variant_payload`/`consume_field`). That is
real work rather than a small extension, and landing a half-built lowering — which regressed the
Unsupported message from the precise one to a confusing internal one — would have been worse than
leaving the construct cleanly refused. It is now writable against a correct oracle.
FILES: starkc/src/interp.rs, KNOWN-DEVIATIONS.md (DEV-076 closed; both enumerations),
COMPILER-STATE.md, WP-C4.7.md.
RULES: EXEC-ONCE-001 / DROP-ORDER-001 (every value's destructor runs exactly once).
DECISIONS: none at CE level — §0.5 permits an oracle behaviour change that a DEV entry documents,
and DEV-076 is that entry, written before the fix.
EVIDENCE: probe programs with printing destructors, run through `oracle_run`, covering all three
paths — `Some` with a discarded default (`100 2 200 1 300 1`), `None` (`100 200 2 300 2`), and the
minimal ordering case (`100 2 999 1`). MIR continues to refuse the construct cleanly, so the
differential is unchanged and no test needed rewriting.
FOLLOW-UP: the MIR half of C4.7-8.1.
NEXT: droppable `unwrap_or` lowering via the drop-flag machinery, then 8.2 (droppable Iterator
Item) and 8.3 (droppable scrutinee + nested patterns, the hardest piece).

### WP-C4.7-8.1 — droppable `unwrap_or` lowering (MIR half) — 2026-07-20
DONE: C4.7-8.1 complete. The oracle half landed as 8.1a (DEV-076); this is the lowering, written
against the corrected oracle rather than against the double drop it used to exhibit.
SEMANTICS IMPLEMENTED (pinned empirically first, per §0.6): `unwrap_or` discards exactly one of
two values and the discarded one owes a destructor — Core has no laziness, so the default is
evaluated whether or not it is used, which is exactly why it always owes one. The discarded value
is destroyed **at the call**, not at end of scope. On `Some`/`Ok`: yield the payload, drop the
default there. On `None`: yield the default. On `Err`: yield the default and drop the displaced
error payload — the case with no `Option` analogue, and the one most likely to be missed.
THE BLOCKER AND ITS RESOLUTION: a first attempt (reverted in 8.1a rather than shipped half-built)
died on the C4.5d guard "move through a non-field projection of a drop-tracked local" — consuming
a payload out of a drop-tracked local via `VariantField` is refused outright. `lower_match` had
already solved exactly this: it materializes the scrutinee into a fresh temp, whose initial move
clears the SOURCE local's drop flags, and a temp is never auto-dropped, so ownership transfers
exactly once with no double-drop possible. Reusing that discipline — rather than inventing a
second one for `unwrap_or` — is what turned this from a redesign into a few lines, and it keeps
one drop-elaboration story in the lowering instead of two.
SCOPE DISCIPLINE: the temp materialization and the default temp are introduced ONLY when a
droppable type is actually involved; the non-droppable path lowers byte-for-byte as before, so
no existing golden or corpus expectation moved.
FILES: starkc/src/mir/lower.rs, starkc/tests/mir_differential.rs (+3),
starkc/tests/mir_lowering.rs (the now-stale `unwrap_or` Unsupported fixture REMOVED — a residual
fixture that no longer describes a residual is worse than none), COMPILER-STATE.md, WP-C4.7.md.
RULES: EXEC-ONCE-001 / DROP-ORDER-001. No spec change; no MIR shape or runtime-surface change —
this is lowering only, using `Drop` terminators that already exist.
DECISIONS: none at CE level.
EVIDENCE: `droppable_unwrap_or_drop_timing_agrees` (both `Some` and `None` paths, with the
printed ORDER as the assertion — `100 2 200 1 300` then `400 3 500`, so the default's destruction
at the call is what is being checked, not merely that it happens);
`droppable_result_unwrap_or_drops_the_error_payload_agrees` (both type arguments carry
destructors, so neither can hide; pins `9` dropping at the call and reverse-order scope exit);
`droppable_unwrap_or_with_runtime_type_agrees` (`String` payload — the runtime-type drop path
rather than a user `Drop` impl). Workspace 785 passed / 0 failed / 2 ignored (+3); fmt clean;
clippy clean on 1.93 and 1.97.
FOLLOW-UP: none for 8.1.
NEXT: C4.7-8.2 — droppable `Iterator` Item (per-iteration scope around the loop-variable binding;
oracle-pin first), then 8.3.

### WP-C4.7-8.2 — droppable `Iterator` Item (per-iteration drop scope) — 2026-07-20
DONE: a user `Iterator` whose `Item` needs dropping now lowers.
PINNED FIRST (§0.6), and it is the non-obvious part: each yielded value is destroyed at the
**end of its own iteration**, not accumulated and destroyed at loop exit. A three-element loop
over a printing-destructor `Item` observes `body, value, DROP, body, value, DROP, …`. `break`
destroys the current iteration's value before leaving; `continue` destroys it before looping back.
All four shapes were confirmed against the oracle before the lowering existed.
IMPLEMENTATION: a per-iteration scope around the loop-variable binding — `scopes.push`, register
the binding as droppable with flags FALSE then set true (the binding is initialized by the move
out of the `Option`, and the flag must not be live before that point), lower the body, then
`emit_scope_drops_from` at the latch and pop.
THE ORDERING DECISION THAT DID THE WORK: the loop's `scope_depth` is captured **before** the
per-iteration scope is pushed. `break`/`continue` already drop every scope from `scope_depth`
onward, so both early-exit paths destroy the current iteration's value with **no special casing
at all** — the existing machinery covers them. Pushing the scope before capturing the depth would
have left the value alive on `break`, which is exactly the kind of leak that only shows up in a
test that bothers to break out of the loop. Both early-exit paths are pinned by a test for that
reason.
SCOPE DISCIPLINE: the scope is pushed unconditionally (harmless and keeps one code path) but the
binding is only registered when the `Item` actually needs dropping, so non-droppable iteration
lowers as before.
FILES: starkc/src/mir/lower.rs (`lower_for_over_user_iter`), starkc/tests/mir_differential.rs
(+2 tests, 3 programs), starkc/tests/mir_lowering.rs (stale Unsupported fixture removed),
COMPILER-STATE.md, WP-C4.7.md.
RULES: EXEC-ONCE-001 / DROP-ORDER-001 / EXEC-FOR-001. No spec, MIR-shape, or runtime-surface
change — lowering only.
DECISIONS: none at CE level.
EVIDENCE: `droppable_iterator_item_drop_timing_agrees` (printed ORDER is the assertion, so what
is checked is per-iteration destruction rather than merely that destruction happens) and
`droppable_iterator_item_break_and_continue_agree` (both early-exit paths, which is where a
per-iteration scope is easiest to get wrong). The pre-existing `String`-Item probe also agrees.
FOLLOW-UP: none.
NEXT: C4.7-8.3 — droppable scrutinee + nested patterns, the last MIR residual and the hardest
piece in the plan.

### WP-C4.7-8.3a — DEV-079 + DEV-080: two hidden defects in the flat match path — 2026-07-20
DONE: both CLOSED. Neither was in the plan. Both were found by pinning oracle drop behaviour
before writing 8.3's lowering — the §0.6 discipline paying for itself — and both sit in the FLAT
enum-match path that WP-C4.6 A2 ("general pattern engine") and C4.5d (match-drop elaboration) had
already signed off.
**DEV-079 — the verifier rejected valid MIR.** V-MOVE-1 keyed moved places as `(local, pure-Field
path)` and collapsed ANY non-`Field` projection to the whole local. `VariantField` is such a
projection, so moving two different payload fields out of one enum local looked like two moves of
the same whole place, and the second was reported `MIR-0007 move from possibly-moved place _N[]`.
Consequence: **every enum variant with two or more droppable payload fields** — with or without a
wildcard, user-`Drop` or `String` — produced MIR that **lowering accepted and verification
rejected**. That is worse than a clean `Unsupported`: the two components are supposed to be
independent readings of the same contract, and here they disagreed silently until someone wrote
the program. Fix: `moved_key` gives `VariantField(v, i)` two path components (variant, then
field), making siblings distinguishable. No collision with struct `Field` paths is possible — a
local has exactly one type, so its projections are either struct/tuple fields or variant fields.
`Deref`/`Index` still collapse to the whole local: conservative and correct, since neither denotes
a statically-known disjoint sub-place.
**DEV-080 — the drop order the verifier bug had been hiding.** With the verifier fixed, such
programs ran for the first time and immediately disagreed with the oracle. For a payload mixing
bound and wildcard fields, MIR destroyed leaves in plain reverse-FIELD order; the oracle destroys
**all bound bindings first, in reverse binding order, then the discarded leaves**. Fix:
`consume_variant_payload` consumes unbound fields FIRST and bound fields second — arm-end drops
run in reverse registration order, so registering the discarded leaves first makes the bindings
drop first and the discards after, which is the oracle's order.
WHY THIS PAIR IS WORTH NOTING: the second defect was strictly unobservable while the first
existed, because no such program could verify. A conservative rejection is not a safe place to
stop — it can hide a real semantic divergence behind itself indefinitely, and the corpus will
look green the whole time.
FILES: starkc/src/mir/verify.rs (`moved_key` + the honest-limitations note),
starkc/src/mir/lower.rs (`consume_variant_payload`), starkc/tests/mir_differential.rs (+2 tests,
4 programs), KNOWN-DEVIATIONS.md (DEV-079/080; count 76 → 78), COMPILER-STATE.md, WP-C4.7.md.
RULES: V-MOVE-1 (refined, not weakened); DROP-ORDER-001 / PAT-DROP-001. No spec, MIR-shape, or
runtime-surface change.
DECISIONS: none at CE level.
EVIDENCE: `enum_variant_with_two_droppable_fields_agrees` (user-`Drop` and `String` payload
forms) and `variant_payload_drop_order_with_wildcards_agrees`. The three-field `(a, _, c)` case is
the discriminating one: its expected order — `c`, `a`, then the wildcard — matches neither plain
reverse-field order nor declaration order, so it pins the actual rule instead of a coincidence.
Workspace 789 passed / 0 failed / 2 ignored (+2); fmt clean; clippy clean on 1.93 and 1.97.
FOLLOW-UP: none for these two.
NEXT: C4.7-8.3b — the original 8.3 target, a droppable scrutinee under NESTED patterns
(`Some((s, n))`), still a clean `Unsupported` ("A2 residual").

### WP-C4.7-8.3b — droppable scrutinee under nested patterns (+ DEV-081) — 2026-07-20
DONE: the last recorded MIR residual of the WP-C4.6 Class-A campaign is closed.
IMPLEMENTED: `consume_unbound_leaves` — a recursive walk that moves every droppable sub-place the
pattern does NOT bind into an arm-scoped registered temp. A consuming match decomposes the
scrutinee completely, so whatever the pattern discards still owes a destructor: wildcards,
unmentioned struct fields, and nested tuple/variant sub-places all covered. Bindings themselves
now register as droppable in the general engine, matching what the flat path's `bind_field_local`
already did.
ORDER: the unbound walk runs BEFORE the binding walk. Arm-end drops run in reverse registration
order, so registering the discarded leaves first makes the bindings drop first — in reverse
binding order — and the discards after them, which is what the oracle does (the rule established
by DEV-080). The three-element `Some((a, _, c))` case is the discriminating evidence: expected
order `c`, `a`, wildcard, which matches neither plain reverse-field order nor declaration order.
**DEV-081 — a third pre-existing defect, found here.** `bind_shorthand` (the lowering for
`P { a, b }` rather than `P { a: a, b: b }`) moved the field value into the binding local but
**never registered that local as droppable, in any mode**. The value left the scrutinee and
nothing destroyed it. This is a **leak, not a double drop**, which is exactly why it survived: no
verifier rule is violated, no assertion trips, and a program whose destructor does not print looks
correct. It affected the FLAT path as well — `enum E { V { a: Tag, b: Tag } }` matched by
`E::V { a, b }` leaked before 8.3b existed — so it is genuinely pre-existing rather than exposed
by the new code. The named and shorthand binding paths differed in exactly this one respect, which
is what made it easy to miss.
THREE DEFECTS IN ONE INCREMENT, all in already-signed-off code (DEV-079/080 in 8.3a, DEV-081
here), all found by pinning oracle behaviour before writing lowering. Two of the three were
invisible to the existing corpus: one because a conservative verifier rejection hid it, one
because a leak has no loud failure mode.
RESIDUALS NOW: the clean-`Unsupported` list is down to `HashMap::values` (std-full, explicitly
reserved by CD-033 — not an exit blocker) and mutable slice views (WP-C4.7-8.6, an owner
decision). Every other Class-A residual recorded by WP-C4.6 is closed.
FILES: starkc/src/mir/lower.rs (`consume_unbound_leaves`, `bind_pattern` binding registration,
`bind_shorthand`, guard removed), starkc/tests/mir_differential.rs (+3 tests, 8 programs),
starkc/tests/mir_lowering.rs (last stale residual fixture removed),
KNOWN-DEVIATIONS.md (DEV-081; count 78 → 79), COMPILER-STATE.md, WP-C4.7.md.
RULES: PAT-DROP-001 / DROP-ORDER-001 / EXEC-ONCE-001. No spec, MIR-shape, or runtime-surface
change — lowering only.
DECISIONS: none at CE level.
EVIDENCE: `droppable_nested_pattern_drop_order_agrees` (four shapes incl. the discriminating
three-field case and a whole-payload wildcard), `droppable_nested_pattern_depth_and_mixed_payloads_agree`
(two-level nesting; `String`+user-`Drop` mixed payload), `struct_shorthand_bindings_drop_agrees`
(both the struct-nominal and struct-shaped-enum-variant forms). Workspace 792 passed / 0 failed /
2 ignored (+3); fmt clean; clippy clean on 1.93 and 1.97.
FOLLOW-UP: none.
NEXT: **C4.7-9** — re-run the unsupported-site sweep over all `unsupported(` sites, re-verify the
frozen corpus, classify 8.4/8.5, and write the exit report for the owner's decision.

### WP-C4.7-8.6 — exclusive slice views (surface 0.1-A8) + DEV-082 — 2026-07-20
DONE, under the owner's decision to implement 8.6/8.5/8.4 before auditing rather than defer any
of them. The evidence for that decision, recorded because it settles a question the plan had left
open: **REF-SLICE-001** states outright that "writes through an exclusive slice reference update
the original object", 03-Type-System §107 gives `&mut expr[r]` the type `&mut [T]`, and §547 lists
`&mut [T; N] -> &mut [T]` among the permitted coercions. Mutable slice views are therefore
normative Core, and rev. 10's deferral of them would have exited C4 with a gap in a rule the
abstract machine states directly.
IMPLEMENTED: `RuntimeFn::SliceNewMut` (A1 amendment rev. 12, surface `0.1-A7` → `0.1-A8`),
`&mut [T]` destination, exclusive receiver borrow. The shared and exclusive constructors compute
the SAME window and share one interpreter arm — they differ only in the reference they yield, and
write permission is a static property the verifier enforces rather than something the runtime
value carries.
WRITE-THROUGH: the interpreter's WRITE path now composes a `Slice { start, len }` window with a
following `Index(i)` into the absolute element `start + i` — precisely the composition its READ
path already performed. That composition IS the write-through semantics; without it a write
through a view could not reach the base. A bare window with no following index is not a writable
place (it denotes the sub-view as a value) and is rejected loudly.
**DEV-082, found here and closed.** `borrowck.rs`'s `method_receiver` had no arm for slice or
array receivers, so a method call on one returned `None` and the caller's fallback CONSUMED the
receiver. For `&[T]` that is harmless — shared references are `Copy`, so the "move" is a copy —
which is exactly why shared slices shipped in A4-2e without anyone noticing. For `&mut [T]` it is
a real move, so `let s = &mut a[1..4]; s.len(); s[0]` failed E0100. The defect was **structurally
invisible until exclusive views existed to expose it**: no program could hold a non-`Copy` slice
reference before today. MIR had the same shape — lowering passed the receiver by MOVE — and now
reads it by `Copy`, the MIR-level equivalent of a shared reborrow, since `len`/`is_empty` only
read.
FILES: starkc/src/mir/{mod,lower,verify,interp}.rs, starkc/src/borrowck.rs (DEV-082),
starkc/tests/{mir_differential,mir_lowering}.rs (incl. both surface-string goldens and the last
`mutslice` Unsupported fixture removed), mir-amendment-A1-strings-runtime.md (rev. 12),
KNOWN-DEVIATIONS.md (DEV-082; count 79 → 80), COMPILER-STATE.md, WP-C4.7.md.
RULES: REF-SLICE-001 (write-through), 03 §107/§547. No spec text changed — the spec already
required this. MIR shape unchanged; runtime surface bumped by the pre-authorized dated-enumeration
mechanism.
DECISIONS: implements the owner's 8.6 decision; no new CE-level decision taken.
EVIDENCE: `mutable_slice_views_agree` — write-through observed at the BASE object (array and
`Vec`, the latter at a non-zero view-relative index), a view passed to a function that mutates it
through the parameter, and repeated use of a `&mut [T]` local (the DEV-082 case).
FOLLOW-UP: none.
NEXT: WP-C4.7-8.5 — non-bare impl heads (`impl<T> Wrap for Holder<Vec<T>>`), front-end-first per
C4.7-2's finding, then 8.4 (method-own generics), then C4.7-9.

### WP-C4.7-8.5 — non-bare impl heads — 2026-07-20
DONE. `02:117` (`Impl ::= 'impl' GenericParams? Type …`) admits any `Type` as an impl self type,
so a non-bare head is normative Core; C4.7-2 had already found this front-end-blocked rather than
a MIR gap.
ROOT CAUSE: `match_impl_type` bound an impl parameter only when it stood ALONE as a type argument
and otherwise fell back to `types_equal`. So `Option<T>` versus `Option<Int32>` compared unequal
and the impl was invisible to method resolution — E0302 "method not found for type
`Holder<Option<Int32>>`".
FIX: `unify_impl_ty`, one-way structural unification over nominals, `Core` containers, tuples,
references, arrays and slices. One-way matters: parameters bind from the IMPLEMENTATION side only.
A `Ty::Param` on the RECEIVER side is an ordinary type to match against, never a hole to fill —
otherwise an impl for a concrete type would spuriously match a generic receiver. A parameter that
recurs (`Pair<T, T>`) must see the same type at each occurrence, so bindings are checked for
consistency rather than overwritten.
BOTH ENGINES, DELIBERATELY: lowering's `impl_generic_subst` had the same bare-parameter
restriction and gained the matching `bind_written_impl_arg`. The checker decides WHICH impls
apply; lowering recovers the substitution that decision implies. Had only the checker been
generalized, the front end would have admitted programs that lowering then refused — exactly the
DEV-079 failure shape, where lowering and verification disagreed about the same contract.
DEV-083 RECORDED, NOT FIXED: a CONCRETE position in an impl head cannot match a receiver argument
that is still an unresolved inference variable at resolution time (`impl<T> Pair<Option<T>, Int32>`
against `Pair<Option<_infer>, _infer>`). Fixing it requires committing inference variables during
candidate search, which can select the wrong impl — a semantics change needing its own design and
evidence under TYPE-METHOD-001, not a bug fix to fold into this increment. It is a narrow
over-rejection (needs a generic impl AND a concrete head position AND an unresolved receiver
argument), both engines reject identically, and annotating the receiver is a working workaround.
FILES: starkc/src/typecheck.rs (`unify_impl_ty`), starkc/src/mir/lower.rs
(`bind_written_impl_arg`), starkc/tests/mir_differential.rs (+1 test, 3 programs),
KNOWN-DEVIATIONS.md (DEV-083; count 80 → 81), COMPILER-STATE.md, WP-C4.7.md.
RULES: 02:117 (impl grammar), TYPE-METHOD-001. No spec, MIR-shape, or runtime-surface change.
DECISIONS: none at CE level.
EVIDENCE: `non_bare_impl_heads_agree` — a trait impl and an inherent impl on `Holder<Option<T>>`,
the latter at TWO instantiations so monomorphised dispatch through a non-bare head is exercised
rather than merely the checker's acceptance, plus a concrete head position with a known receiver
type. Workspace 794 passed / 0 failed / 2 ignored (+1); fmt clean; clippy clean on 1.93 and 1.97.
FOLLOW-UP: DEV-083.
NEXT: WP-C4.7-8.4 — method-own generic parameters, the last implementation item before the audit.

### WP-C4.7-8.4 — method-own generic parameters — 2026-07-20
DONE. This completes every implementation item in WP-C4.7; only the audit and exit report remain.
NORMATIVE BASIS: `02:64` puts `GenericParams?` on every `FunctionSig` and `02:120` makes an impl
item a `Function`, so a method may declare its own generic parameters. C4.7-2 had found this
front-end-blocked (E0001 "expected 'U', found …") rather than a MIR gap, which is why it moved out
of the MIR column and needed both engines fixed.
TWO HALVES:
- **Checker.** The selected candidate's substitution map carried only the IMPL's parameters, so a
  method's own `U` stayed a rigid `Ty::Param` and no argument could unify against it. It now gets
  a fresh inference variable per call site (or the turbofish types when given) — exactly what the
  ASSOCIATED-FUNCTION path already did. Only the method path lacked it.
- **MIR.** `FnKey::ImplFn` gains `method_args` beside the impl's `type_args`; `lower_body` binds
  the method's parameters from it, `key_symbol` renders it in a second bracket, and the call site
  fills it from a new per-call-site record keyed by the call expression — the method equivalent of
  C4.5c's `generic_insts` for top-level generic fns. Impl-level and method-level substitutions
  stay SEPARATE, because a method on a generic nominal is generic in both and conflating them
  would monomorphise at the wrong arguments.
CE3 QUESTION THE PLAN ASKED ME TO SETTLE: **`FnKey` appears zero times in `mir.md`.** It is purely
lowering-internal, so extending it is not a contract change and needs no CE3. The rendered
`Instance.symbol` does change for generic methods, but §2 states symbols are "deterministic and
injective for identical inputs; NOT a stable external ABI", and a method with no own generics
renders exactly as before, so no existing symbol moved.
FILES: starkc/src/typecheck.rs (method-level instantiation + per-call-site recording),
starkc/src/mir/lower.rs (`FnKey::ImplFn::method_args`, symbol rendering, body substitution, call
site), starkc/tests/mir_differential.rs (+1 test, 3 programs), COMPILER-STATE.md, WP-C4.7.md.
RULES: 02:64/02:120 (grammar), TYPE-GENERIC-001. No spec, MIR-shape, or runtime-surface change.
DECISIONS: none at CE level — see the `FnKey` conclusion above.
EVIDENCE: `method_own_generics_agree` — two instantiations at different primitives; two
method-own parameters in one signature with a droppable (`String`) instantiation; and a GENERIC
METHOD ON A GENERIC NOMINAL at two different `U`s plus a second nominal instantiation, which is
the case that would fail if the two substitution levels were conflated. Every case uses multiple
instantiations, so what is exercised is one lowered body per instantiation rather than the
checker's acceptance alone. Workspace 795 passed / 0 failed / 2 ignored (+1); fmt clean; clippy
clean on 1.93 and 1.97.
FOLLOW-UP: none.
NEXT: **C4.7-9** — re-run the unsupported-site sweep over every `unsupported(` site, re-verify the
frozen corpus, and write the exit report for the owner's decision.

### WP-C4.7-9 (audit sweep) — six further findings; four fixed, two recorded — 2026-07-20
DONE: the sweep. Every `unsupported(` site in `lower.rs` enumerated, partitioned
defensive-vs-construct, and each construct candidate probed with `c46_probe` AND `oracle_run`.
The forecast that the audit would find more was correct.
FIXED UNDER THE OWNER'S DIRECTION ("fix 1, 2, 4 and the checker rejection for 3"):
- **DEV-084 — `print`/`println` accepted any type.** They typed their argument as a fresh
  inference variable, so a `Display`-less user struct was accepted. Three engines gave three
  answers for a program 06 says is invalid: the checker accepted it, the oracle rendered an
  unspecified `{x: 1}`, MIR refused. The CHECKER was the wrong one, and the fix is a rejection,
  not an implementation — deferred to the same pass as the bound checks so an argument still
  under inference is not judged early. One interpreter test depended on the over-acceptance and
  now asserts the rejection; its real subject (`Float32` digits nested in an aggregate) was
  already covered by its `Option`/`Result` and tuple siblings.
- **DEV-085 — `for` over a fixed-length array.** Checker accepted, oracle ran, MIR alone refused:
  an internal inconsistency, not a language boundary. Lowered as a counting loop reading one
  element per iteration through the ordinary `CheckIndex` proof discipline. **Its own
  implementation had a bug the test caught:** `continue` first targeted the loop header directly,
  skipping the increment and spinning until the interpreter's fuel ran out. The continue target
  is now a latch that increments first — and the control-flow test that exposed it was written
  before the fix, not retrofitted after.
- **Trait-default methods with own generic parameters.** WP-C4.7-8.4 fixed the selected-impl path
  and left this one: the checker's default-fallback did not instantiate the method's own
  parameters, and `FnKey::TraitDefault` had no `method_args`. Both now match the `ImplFn`
  treatment.
RECORDED, NOT FIXED:
- **DEV-086 — droppable elements in array patterns, and by-value array iteration.** An array
  element place needs `Projection::Index(ProofLocal)`, and the only way to mint a proof is a
  `CheckIndex` that READS the array. Moving one element out poisons the whole local for V-MOVE-1
  (`Index` must collapse to the whole local — a dynamic proof names no statically-known
  sub-place), so the next element's check reads a possibly-moved place. The fix is a
  **constant-index projection form**, a MIR shape change requiring CE3 (§0.5), so it is recorded
  rather than invented. The contract already points that way — §6 says the proof discipline
  "covers fixed-length `Array` (verifier may validate against the compile-time length)" — but it
  is the owner's call. Non-droppable array patterns and `Copy`-element iteration are unaffected.
- **DEV-083** (from 8.5) remains open on the same footing.
CORRECTLY RESERVED, not blockers: `HashMap::values`, `Vec::contains`, `String::insert` — std-full,
explicitly reserved by CD-033. Or-patterns (`A(n) | B(n)`) are **not in 02's Pattern grammar**
(`02:284-291`), so the parse error is correct behaviour, not a gap.
FILES: starkc/src/typecheck.rs (Display check + trait-default method generics),
starkc/src/mir/lower.rs (`lower_for_over_array`, `FnKey::TraitDefault::method_args`, array-pattern
residual), starkc/src/interp.rs (the repurposed test + a `type_diagnostics` helper),
starkc/tests/{mir_differential,gate2_valid}.rs, KNOWN-DEVIATIONS.md (DEV-084/085 closed, DEV-086
opened; count 81 → 84), COMPILER-STATE.md, WP-C4.7.md.
RULES: 06 (`Display` is not a syntax hook), EXEC-FOR-001, 02:284-291 (Pattern grammar),
02:64/02:120 (generic method signatures). No spec text changed.
DECISIONS: none at CE level; DEV-086 is flagged AS a CE3 question rather than resolved.
EVIDENCE: `for_over_array_agrees` (values, running total, `break`/`continue`, single-element
array), `trait_default_method_own_generics_agree` (two instantiations),
`printing_requires_display` (rejection plus the standard displayable types still printing),
`printing_a_struct_without_a_display_impl_is_rejected`. Frozen corpus green. Workspace 798 passed
/ 0 failed / 2 ignored (+4); fmt clean; clippy clean on 1.93 and 1.97.
FOLLOW-UP: DEV-083, DEV-086 — both over-rejections, both consistent across engines, both needing
an owner decision rather than more implementation.
NEXT: write the C4.7-9 exit report as a new final section of `WP-C4.6.md` and present it. The gate
decision is the owner's; this session does not close it.

### WP-C4.7-9 — the Gate C4 exit report — 2026-07-20
DONE: the report is written as the final section of `WP-C4.6.md`, superseding that document's
2026-07-19 Verdict. Presented to the owner; **the gate is not closed by this session**.
VERDICT AS WRITTEN: conditions 1 (corpus equivalence) and 3 (nothing carried silently) are
SATISFIED outright. Condition 2 (every normative Core construct lowers) is satisfied EXCEPT for
DEV-086 and DEV-083 — both over-rejections, both consistent across engines, neither closable by
more implementation of the same kind: one needs a CE3 constant-index projection form, the other a
method-resolution design decision under TYPE-METHOD-001.
RECOMMENDATION: close C4 **conditional on the owner disposing of those two by explicit dated
decision** (implement in C5.x, or defer with the deferral recorded) rather than leaving them
undisposed. Recording them WITH a disposition is what makes carrying them forward honest rather
than silent — which is exactly what CD-033's condition 3 asks for.
THE COUNTER-ARGUMENT, STATED IN THE REPORT RATHER THAN OMITTED: today's sweep found six items
after four increments had already "finished" the residual list, and 11 of this package's 13
defects were in signed-off code. The defect-discovery rate has **not visibly plateaued**. Two
things argue against another round now — the sweep was systematic rather than opportunistic
(every `unsupported(` site, both engines), and the two survivors are analysed and decision-blocked
rather than effort-blocked — but the risk statement belongs in front of the owner, not buried.
WHAT THE REPORT CLASSIFIES: every remaining rejection, in four buckets — spec-conformant (with the
authority cited, including the corrected "Box deref" audit error), CD-033-reserved std-full,
defensive guards (incl. the two deliberately-retained unreachable ones), and the two open
deviations. Plus the ledger state (84 numbered; 16 closed by this package; the three SOUNDNESS
defects called out separately) and the contract/spec changes (amendments A3/A4, surface
`0.1-A6` → `0.1-A8`, and the new normative `PRIM-TRAIT-001`).
FILES: STARKLANG/docs/compiler/work-packages/WP-C4.6.md (the report),
starkc/docs/conformance/KNOWN-DEVIATIONS.md (one stale line about 8.1's MIR half corrected),
COMPILER-STATE.md, WP-C4.7.md.
EVIDENCE CITED: workspace 798/0/2, 114 differential tests, frozen corpus green, fmt + clippy clean
on 1.93 and 1.97.
NEXT: **the owner's decision.** Report §6 is the decision table: DEV-086, DEV-083, post-hoc
ratification of surface revs 11/12, frozen-corpus growth, and gate closure.

### WP-C4.7 close-out — CD-038/039/040 executed; C4 NOT closed (DEV-089) — 2026-07-20
DONE: the owner's close-out directive, in full, except the closure itself.
**1. DEV-086 IMPLEMENTED (CD-038, CE3).** `Projection::ConstIndex(u64)` — statically known array
element, valid only on `Array<T, N>`, verifier bounds-checks it directly, no `CheckIndex` and no
`IndexProof`, invalid on `Vec`/slice, dynamic indexing unchanged. Consuming array patterns over
droppable elements now lower and agree with the oracle including drop order. The same decision's
**typed internal paths** were adopted: move-dataflow and drop-unit paths are typed components
(field / variant field / constant index) instead of raw `u32` sequences, and fixed-length arrays
decompose into per-element drop units — without which moving one element out and then dropping the
array would destroy it twice. Recorded in `mir.md` as amendment A5.
**NARROWED, not closed:** by-value iteration over a NON-`Copy` array element. The loop index is a
runtime counter, so no `ConstIndex` names the consumed element and V-MOVE-1 has nothing precise to
track. Reading by copy instead would be UNSOUND — the array still owns the element and destroys it
again, a double free for a `String` in a real backend — so it is refused cleanly with that reason.
Closing it needs unrolling or runtime-indexed drop flags: a separate design question, not an
extension of A5. This is recorded rather than approximated, deliberately.
**2. DEV-083 DEFERRED (CD-040b)** to `WP-C6.x Method Resolution Completion`, with the owner's
disposition text recorded verbatim in the ledger (candidate-local inference snapshots;
declaration-order-independent evaluation; no mutation of global inference state while probing).
**3. RUNTIME SURFACE RATIFIED (CD-040a):** A1 revs 11 and 12 (`0.1-A7`, `0.1-A8`). Documentation
and the active constant agree, so no implementation change was needed.
**4. CORPUS 1.2.0 (CD-039).** Completes the compact refresh to the six specified workloads: adds a
MULTI-FILE case (cross-file structs, methods, trait default + override, cross-file `Drop`,
provenance) and folds DEV-086's array pattern into the array/slice case. A bump rather than an
amendment of 1.1.0 because the array case's bytes changed. **All 48 hashes from 1.0.0 verified
byte-identical**, so the original baseline survives inside 1.2.0.
**5. GATE NOT CLOSED — DEV-089.** The bounded validation surfaced a new ENGINE DIVERGENCE, and §6
of the directive says to stop and report on exactly that. `println(p)` where `P` HAS a `Display`
impl: checker accepts, oracle runs it but prints its own debug form ignoring the user's
`Display::fmt`, MIR refuses to lower it. Not a soundness defect and not invalid MIR — nothing
mislowers — but the stopping rule's clause "no known … engine divergence remains" is not satisfied,
so closing would require asserting something untrue. It surfaced only because DEV-084 narrowed the
checker: before that, `println` accepted any type, so "has an impl" and "has no impl" were
indistinguishable.
ALSO FOUND AND PARTLY FIXED: **DEV-088** — cross-file `const` initializers were evaluated against
the entry file (the fourth per-item-file site DEV-069 missed). Declaration-time evaluation fixed;
the USE site remains open in both engines (a clean over-rejection). The multi-file corpus case was
reduced to its subject rather than chasing it, per the scope-discipline instruction.
BOUNDED VALIDATION: workspace **802 passed / 0 failed / 2 ignored**, exit 0; fmt clean; clippy
clean on 1.93 and 1.97; corpus 1.2.0 lock integrity green; `entire_frozen_corpus_agrees` green over
all 23 cases; DEV-076…086 regressions green; unsupported-site classification re-run (171 sites).
FILES: starkc/src/mir/{mod,lower,verify,interp}.rs, starkc/src/interp.rs (DEV-088),
starkc/tests/{mir_differential,mir_verify,exec_snapshots}.rs, the corpus (+3 files, 1 modified) and
its lock, STARKLANG/docs/compiler/mir.md (amendment A5), KNOWN-DEVIATIONS.md (DEV-086 closed/
narrowed, DEV-083 deferred, DEV-088/089 opened; count 85 → 87), COMPILER-STATE.md.
NEXT: **owner decision on DEV-089**, then closure. Everything else in the directive is done.

### WP-COPY-CANON — Phases 0–4 done, Phase 5 partial — 2026-08-01

**Reconciliation gap, stated first because it is the largest fact here.** Before this entry the
highest CD recorded in this file was **CD-294**. CD-295 through CD-306 remain unrecorded — they are
other work (package-track fixes, the HTTP substrate packages, `stark test` defects, DevOps) and
some of it belongs to parallel sessions. This entry covers **CD-307..CD-316 only** and does not
close that gap.

**The packet.** WP-COPY-CANON governs one law: *after expression typing, Copy/move behaviour — and
the runtime representation that carries it — is determined exclusively by the normalized semantic
type, never by the expression that produced the value.* It binds the checker, MIR lowering, the
native backend and each interpreter equally. Registered under CD-307 before any investigation, per
the packet's ordering rule.

**Phase 1 — the sentinel matrix (CD-308).** Six producers of a reference-typed value against six
use modes, checked on three axes: MirTy copy-eligibility, the emitted MIR call operand (asserted
from the dump, so a wrong operand fails even when runtime behaviour is green), and the runtime value
kind. A per-producer matrix rather than a regression for `bytes()`, because DEV-121 was per-producer:
`bytes()` and `as_slice()` share a normalized type, were built by different code paths, and only one
was wrong.

On its first run it failed on **CD-305's own fix**, not on DEV-121. CD-305 promoted `bytes()`'s
materialised storage into the *current* frame; correct locally, dangling the moment the view is
returned. `fn borrow_of(s: &String) -> &[UInt8] { s.bytes() }` is valid Core v1 and produced
"dangling reference". CD-305's regression tests had no escaping-view case; the matrix's
ordinary-language producer controls do.

**Phase 2 — the escape fix (CD-308) and DEV-126 (CD-313).** `promote_to_temp_place_in` takes the
owning frame explicitly. That was not sufficient: CI failed `stark-json` 9/10 on all three platforms
with "dangling reference", because `as_str` returned `Value::Str(string.clone())` — a detached copy
with no link to its origin — so the chained `c.input.as_str().bytes()` had nothing to anchor to.
`as_str` now returns `Value::Ref(receiver_place)`. Consequence: `s.as_str()` reaches builtins as a
`Value::Ref`, and `flatten_string_refs` derefs a reference argument **when its referent is a
string** — keyed on the referent's kind, not the callee's name, unlike the pre-existing
`remove`/`contains_key`/`contains` special case which only ever covered the three reported.

**Phase 3 — INV-MOVE-001 / MIR-0036 (CD-311..CD-315).** Unconditional, no exemption mechanism. It
found the same defect at seven sites across four DEVs, each invisible until a workload of the right
shape ran:

| DEV | Sites | What reached it |
| --- | --- | --- |
| 124 | for-loop desugar, both forms | any `for` loop — 12 unit tests |
| 125 | provider status→`Result`; out-slot tuple; `?`'s `Err` payload | the REST workload and C7.8 only |
| 127 | `borrow_set_receiver` | the DEV-116 HashSet corpus only |

In every case the correct idiom sat next to the defect: `assign_provider_ok` read its slots through
`read_place` then hand-built the `Move` wrapping them; `borrow_map_receiver` used `read_place` while
`borrow_set_receiver` three lines away did not. **The fix is never "write `copy`"** — a non-`Copy`
payload must still move; the defect is that the site had an opinion at all.

Two structural consequences (CD-315, DEV-128): the `Copy` rule now exists **once**, in
`mir::mir_ty_is_copy`, with the nominal case passed as a predicate — it had been two byte-identical
matches differing in one lookup, and the comment naming
`lowered_copy_classification_matches_the_type_context` as the test keeping them in step referred to
**a test that does not exist**. And `operand_move_inventory` pins all eleven `Operand::Move`
occurrences in `lower.rs` with a reason each, so a new one fails at authoring time.

**Phase 4 — `diag::resolve_span` (CD-309).** The one checked path from a span to a location; never
panics, never falls back to another source. An interim guard, not the architecture: filed as
`WP-SPAN-SOURCEID.md`, which CD-309 committed to and did not do until now.

**Two test fixtures retyped (CD-314), recorded because the distinction matters.** `mir_verify`'s
`partial_move_of_one_field_leaves_sibling_readable` and `dev117_...` hand-build MIR moving `Int32`
locals; `Int32` was incidental filler, and under INV-MOVE-001 an `Int32` move is invalid MIR on its
own account, so both failed for a reason neither test concerns. Retyped to `&mut Int32` with every
assertion unchanged. The weakening NOT done: exempting `Copy` moves in the invariant.

**Method finding, recorded because it cost the most.** Four instances of one defect reached CI one
round at a time, because each local run covered a different slice — lib suite, then four iterator
tests, then the provider workloads, then the C6 corpus. INV-MOVE-001 was correct every time; the
local evidence was too narrow for a change that constrains every lowering site in the compiler. The
compensating measures are CD-315's authoring-time inventory and CD-316's matrix chaining axis.

**Phase 5 — PARTIAL.** This reconciliation and `WP-SPAN-SOURCEID.md` are done. Not done:
qualification evidence, and the frozen-corpus question. On the latter: the new matrix and chaining
cases are plain `#[test]`s in `copy_canon_matrix.rs`, not corpus cases, so **no corpus bump may be
owed at all** — an earlier claim in this session that the corpus was "locked at 1.2.0" was wrong
(CD-069 re-pinned it to 1.3.0, and `exec_snapshots` and the generated corpus carry their own
versions). Establishing which corpus, if any, is affected is the remaining Phase 5 work.

**Still open from the packet:** INV-VALUE-REP-001, the actual class-closer for DEV-121. Not
attempted. It needs the normalized type available at interpreter binding sites, and the HIR
interpreter is largely untyped at runtime.

EVIDENCE: lib 495/495; mir_verify 51/51; copy_canon_matrix 7/7; operand_move_inventory 1/1;
c6_generated_corpus 7/7 over 170 cases; c788_lifecycle_e2e 9/9; stark-json 10/10; C7 P1 REST
workload 24/24 byte-exact HTTP cases on all three platforms. CI on develop is the outstanding judge
for CD-313..CD-316.
FILES: starkc/src/mir/{mod,lower,verify}.rs, starkc/src/{interp,diag}.rs,
starkc/tests/{copy_canon_matrix,operand_move_inventory,mir_verify}.rs, KNOWN-DEVIATIONS.md
(DEV-121..DEV-128), STARKLANG/docs/compiler/work-packages/WP-SPAN-SOURCEID.md, COMPILER-STATE.md.

### WP-COPY-CANON — CLOSED 2026-08-01 (Phase 5)

**Verdict: the packet's law is enforced in one direction each for behaviour and representation,
with the remainder filed rather than claimed.**

**Phase 5 disposition.**
- **Corpus: NO BUMP OWED, established rather than assumed.** `git diff 0bd4d54..HEAD` over
  `tests/c6-corpus/` and `exec_snapshots` is EMPTY: no corpus case was added, changed or removed.
  The packet's new tests are three plain `#[test]` files (`copy_canon_matrix.rs`,
  `operand_move_inventory.rs`, and edits to `mir_verify.rs`). The generated corpus stays at
  `EXPECTED_CORPUS_VERSION = "1.5.0"`.
  **Two version claims made earlier in this session were wrong and are corrected here**: "locked at
  1.2.0" was stale memory, and 1.3.0 is the FROZEN EXEC corpus re-pinned by CD-069 — a different
  corpus from the C6 generated one. Three corpora with independent versions is the trap; naming
  which one is meant is the fix.
- **Qualification.** `qualify-first-party-packages.py` — the exact script CI runs — passed locally
  at exit 0 over JSON, URL, Base64, Hex and UUID plus their consumers, including native builds.
  Package suites: json 10/10, percent 3/3, ascii 4/4, and mime/query/form 10/11/11 where all three
  previously had ZERO (CD-320).
- **Reconciliation.** CD-307..CD-316 recorded under CD-317; CD-317..CD-322 recorded here.
  **CD-295..CD-306 remain unrecorded** — other work, some from parallel sessions. That gap is
  restated rather than quietly closed.

**What the packet actually established.**

| Half of the law | Invariant | Status |
| --- | --- | --- |
| Copy/move behaviour follows the type | INV-MOVE-001 (MIR-0036) | ENFORCED, unconditional |
| The representation carrying it follows the type | INV-VALUE-REP-001 | NARROW — one direction, one pairing |

INV-MOVE-001 found four latent defects on its first runs: DEV-124 (for-loop desugar, both forms),
DEV-125 (provider status→`Result`, out-slot tuple, `?`'s `Err` payload), DEV-127
(`borrow_set_receiver`). In every case the correct idiom sat beside the defect — `assign_provider_ok`
read its slots through `read_place` then hand-built the `Move` wrapping them; `borrow_map_receiver`
used `read_place` while its sibling three lines away did not.

INV-VALUE-REP-001 is narrow deliberately and DEV-121 is recorded **NARROWED, not class-closed**,
with residual exposure named: `&T` for scalar `T`, and the `Str`/`String` duality. Deferred by owner
direction to `WP-VALUE-REP-TOTAL.md`.

**Structural changes that outlive the packet.**
- The `Copy` rule exists once (`mir::mir_ty_is_copy`), not twice. The comment claiming a test kept
  the two copies in step named a test that **does not exist** (DEV-128).
- Structural equality exists once (`values_equal`), not in four places with the `Str`/`String`
  pairing in only one of them (DEV-130).
- `operand_move_inventory` pins all eleven `Operand::Move` sites in `lower.rs` with a reason each,
  so the next one fails at authoring time rather than when a workload of the right shape runs.
- The matrix crosses producers with each other, not only with use modes (CD-316) — the axis whose
  absence let DEV-126 reach CI.

**Method finding, recorded because it cost the most.** Four instances of one defect reached CI one
round at a time, because each local run covered a different slice: lib suite, then four iterator
tests, then the provider workloads, then the C6 corpus, then `gate4a_prelude_traits`. The invariants
were correct every time; the local evidence was too narrow for changes that constrain every lowering
site and every binding site in the compiler.

**Two defects were introduced by this packet's own fixes and are recorded as such**: CD-305's
escaping-view flaw (found by the matrix on its first run) and DEV-131's over-broad string flattening
(which broke `take`, one commit after its own DEV entry criticised name-keyed derefs).

FOLLOW-UPS FILED, NOT PENDING: `WP-VALUE-REP-TOTAL.md` (owner-deferred),
`WP-SPAN-SOURCEID.md` (CD-317). Neither blocks other work.

### CD-295..CD-306 — backfill of the gap restated by CD-317 and CD-323 — 2026-08-01

Recorded from the commits themselves, not reconstructed. These sit between the last entry the
ledger carried (CD-294) and WP-COPY-CANON's registration (CD-307). They are not one work package:
they are a Windows-encoding fix, a package-tooling batch, a DevOps change, and two compiler defects
that the packages exposed. Grouped by what they were, in commit order.

**Windows encoding — CD-295, CD-296.**
- **CD-295** — `qualify-first-party-packages.py` decoded UTF-8 and then re-encoded to cp1252.
  `13c4eb0` had fixed the READ; the WRITE failed one line later, so a STARK program printing an
  emoji killed the script REPORTING its result while the program itself had already emitted correct
  bytes and passed. `sys.stdout`/`stderr` reconfigured to UTF-8 with `errors="replace"` — a
  reporting path must not fail a qualification run over a byte it cannot render, and the comparison
  happens on decoded text so substitution cannot mask a real mismatch. Verified by forcing
  `PYTHONIOENCODING=cp1252`.
- **CD-296** — the §9.5 output-contract test used `héllo wörld`, every character of which is present
  in cp1252, so a host round-tripping stdout through the console codepage would still have passed.
  Replaced with `😀` (4-byte UTF-8, no cp1252 representation). **The compiler was right and its test
  was incomplete** — same shape as CD-276.

**Package tooling — CD-297, CD-297a, CD-298, CD-300, CD-302.**
- **CD-297** — `stark-random` plus an EXECUTION test, which is the point: three compiler-side tests
  and four native crate tests all passed while the package's STARK code could not lex, had no
  imports, had never compiled `fill_bytes`, and trapped in `next_u64` on its second call. The last
  is a **language-level finding worth carrying**: STARK traps on integer overflow in every build
  mode, and a shift discarding set bits IS an overflow — so every wrapping-arithmetic algorithm
  (hashes, PRNGs, checksums, bit mixers) needs explicit masking, and the failure mode is a runtime
  trap rather than a compile error. Also corrected `c63c_iterators`: **CD-293's E0106 was
  redundant** — E0100 had always refused moving a non-`Copy` value out of an indexed place, and
  E0106 was reasoned from a MIR message without checking a source program could reach it.
- **CD-297a** — `assert_eq!(x, false)` is `clippy::bool_assert_comparison` under `-D warnings`. It
  failed TWO jobs: the lint job and the C6.4 qualification, whose gate runs clippy — **a single lint
  failure invalidates qualification evidence, not just the lint step.**
- **CD-298** — `stark-io` docs four commits out of date. Established that the recorded "library
  packages cannot test themselves" blocker is **narrower than written**: `stark test` already works
  on a library package (parse, resolve, type-check, run through the interpreter, no `main`); what a
  library cannot do is be NATIVELY qualified without an artificial entrypoint. Docs only.
- **CD-300** — `stark test` never synthesized `provider_api`, so every generated `*_raw` was E0200
  and a provider-bound package failed before discovering one test (`stark-io`: 18 undefined
  variables). Added, with `target::host_triple_of_this_build()` derived from `std::env::consts`
  rather than probing `rustc` — testing runs through the interpreter and compiles nothing. Also:
  `stark-random/stark.lock` was malformed, and `stark-random-native` depended on `getrandom` from
  crates.io, which broke every runner under `cargo generate-lockfile --offline`.
- **CD-302** — `stark test` PANICKED on any package with a dependency: `item_text` sliced the root
  file with a dependency's span (`byte index 2147483648` — 2^31, a synthetic span). Every package
  depending on another was untestable, which is most of them; this is why reviewing the package
  batch was impossible before it. `item_text` returns `Option` and callers skip an item whose span
  does not fit its file — not clamped, not guessed. Also added `# Safety` to all 37 unsafe extern
  fns across four provider crates; **CI lints only the `starkc` workspace, so none was checked.**

**DevOps — CD-301.** `develop` branch flow, so a red run cannot land on `main`. The reasoning worth
keeping: `ci.yml` has eleven jobs, three of them matrices, so real check names are generated —
naming them in a protection rule **fails OPEN**, because a renamed matrix entry is simply not found
and GitHub reports the rule satisfied. One `ci-complete` aggregator with `if: always()` and explicit
`needs.*.result` checks is the only name protection needs.

**Compiler defects the packages exposed — CD-303, CD-304, CD-305, CD-306.**
- **CD-303** — **PAT-BIND-001 was never enforced.** `Ty::Ref` fell through every classifier, giving
  the worst combination: exhaustiveness demanded a wildcard on a match that already covered every
  variant (E0303 pointing at the wrong problem), and the `_` arm added to satisfy it then ABSORBED
  EVERY CASE at run time. So the obvious response to a misleading error produced a function silently
  returning the wildcard's answer for every input. Now rejected with help naming `match *r`.
  **Deliberately not done:** making `match r` work by peeling the scrutinee type — that is Rust
  match ergonomics, contradicts PAT-BIND-001, and is a language-design proposal requiring
  coordinated checker/MIR/interpreter change. Caught only when opening the spec to document it.
- **CD-304** — landed Gemini's five HTTP-substrate packages with what does and does not work
  recorded: ascii 4/4 and percent 3/3 passing; mime, query and form with **zero tests and failing
  consumers**, characterised but not fixed. (Their tests were written later, under CD-320.)
- **CD-305** — `String::bytes()` returned an owned `Value::Vec` for a declared `&[UInt8]`, so
  passing the view consumed it. **Which engine was wrong was established, not assumed**: emitted MIR
  was `copy` on both calls, so the checker and MIR were right and the HIR interpreter alone was
  wrong. Predates the session, verified by A/B against a compiler built at `77d763e`. This became
  DEV-121 and the whole of WP-COPY-CANON; its own fix later proved incomplete for escaping views
  (CD-308) and for chained producers (CD-313).
- **CD-306** — a dependency's runtime span was rendered against the root consumer's source: a fault
  inside `stark-mime` reported at line 31 of a 21-LINE consumer. Two causes, one per layer —
  `cmd_run` never read `error.file`, and `SourceFile::line_col` CLAMPS an out-of-range offset so a
  foreign span produces a plausible WRONG location rather than a failure. **It cost real
  investigation time**: the wrong file sent the first characterisation of CD-305 to the wrong shape
  entirely. This became DEV-122, later given a checked resolution path (CD-309) and a filed
  correction (`WP-SPAN-SOURCEID.md`).

**Why this gap existed.** These twelve were pushed across a single long session in which the ledger
was not updated once. The lesson is the same one CD-317 recorded for CD-307..CD-316 and is now
stated for both: `COMPILER-STATE.md` is the live status source, and a session that pushes twelve CDs
without touching it leaves the ledger describing a compiler that no longer exists.

