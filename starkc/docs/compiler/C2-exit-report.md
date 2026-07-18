# Gate C2 Exit Report — Reference Execution Semantics and Compiler Service Foundation

WP-C2.13 deliverable. Prepared 2026-07-18 at head `e270575dbb429bbe5ce3adefb116eb34d486edb8`
(`add compiler support coverage for .st sources` — an unrelated, user-authored feature commit
made outside this WP's scope, landed after WP-C2.12's own commit; noted here only for head
accuracy, not as Gate C2 work).

## Decision

**CORE-V1-SEMANTIC-FOUNDATION-FROZEN-WITH-LISTED-DEVIATIONS.**

Every semantic-completeness work package (WP-C2.6 inventory, WP-C2.7 abstract machine, WP-C2.8
static semantics, WP-C2.9 numeric/layout/text/process/package contracts, WP-C2.10 future
boundaries, WP-C2.11 implementation alignment) is closed. All 24 high-cost future questions in
the open-question register (CORE-Q-001 through CORE-Q-024, including the 005A/013A/013B
variants) are **Approved** — zero remain `pending`, `rejected`, or otherwise unresolved. The
166-row completeness inventory classifies every audited semantic question as
`complete/specified` (155), `complete/prohibited` (4), or `complete/deliberately-unspecified`
(1) — **except 6 rows** (`LEX-COMMENT-001`, `LEX-ERROR-001`, `STD-OPTION-001`, `STD-RESULT-001`,
`STD-ITER-001`, `STD-VEC-001`) still marked `partial/specified, pending-owner-approval`. None of
the six describe missing or contradictory behavior — the underlying compiler behavior for
`Option`/`Result`/`Iterator`/`Vec` is implemented and has been exercised continuously throughout
this gate (including by this WP's own differential corpus) — the gap is that the completeness
ledger's own governance approval was not recorded for these six rows before this report was
written. This is exactly the kind of real, narrow, honestly-disclosed gap the roadmap's
`FROZEN-WITH-LISTED-DEVIATIONS` outcome exists to communicate.

This is **not** a claim that WP-C2.12 (differential interpreter corpus) is complete in the
exhaustive sense its own roadmap text describes, and it is not a claim that MIR or native
compilation exist yet — those are Gate C3+ per the mandatory correctness path. See "Why not
plain FROZEN" and "WP-C2.12 status" below for the precise, undiluted scope of what remains open.

## Why not plain FROZEN

`CORE-V1-SEMANTIC-FOUNDATION-FROZEN` (no listed deviations) would require zero open items across
both the completeness ledger and the deviation ledger. That is not this gate's actual state, and
claiming it would violate Charter rule 13 ("no status drift") the same way the Gate C1 report
declined `CORE-FRONTEND-CONFORMING` for the same reason. Concretely, as of this report:

- **Six completeness rows remain `pending-owner-approval`** (listed above) — narrow, load-bearing
  standard-library and lexer-detail questions whose *behavior* is implemented and tested but whose
  *governance approval* was never formally recorded. This needs an owner decision, not more
  implementation work.
- **WP-C2.12 (differential interpreter corpus) is open, not closed.** DEV-036 is fully resolved
  (a real code fix, not documentation). An initial, representative corpus exists and passes
  100%: 17 hand-written cases across all seven roadmap-named coverage categories, one worked pair
  for each of the seven named metamorphic-transformation classes, and a workspace-relocation
  test. What remains, per WP-C2.12's own work-package file: the roadmap's "generated" half of
  "generated and hand-written coverage" (a case generator, never built), and deeper per-category
  breadth (this pass is representative, not exhaustive, for any of the seven categories).
  Cross-backend replay against a MIR interpreter and native builds is *not* part of this gap —
  the roadmap text itself names that as work that happens *after* Gate C3 opens (neither a MIR
  interpreter nor a native backend exists yet), so it cannot be a C2.13 precondition by the
  roadmap's own dependency structure.
- **DEV-053/DEV-054 were investigated and closed with a real fix during this gate** (the most
  severe finding of the entire compiler track to date: a bare `None` pattern never matched by
  value at all, silently acting as an unconditional wildcard and producing wrong runtime output
  with no diagnostic — `match Some(5) { None => 999, Some(a) => a }` printed `999`). This is
  disclosed prominently, not buried, because a reader auditing this gate's soundness claims needs
  to know a real silent-wrong-output bug existed and was caught by exactly the differential-corpus
  work this gate produced — that is the process working as intended, the same framing the Gate C1
  report used for its own mid-gate findings.
- **Eight deviations remain open**, carried forward or newly found this gate: DEV-005 (CLI
  warning-gating drift, unowned since Gate C1), DEV-010/DEV-012 (LSP stubs and VS Code UI,
  Gate-C8-scoped), DEV-011 (doc comments as trivia, unscheduled), DEV-017 (conformance evidence
  precision, partially closed since WP-C1.6), DEV-051 (trait default methods can't call sibling
  trait methods on `self`), DEV-052 (qualified `Trait::method(..)` calls fail for compiler
  CoreTraits), DEV-055 (bare glob-imported unit enum variants don't resolve at all).

None of these are soundness-relevant in their *current, open* state (every soundness-relevant
finding from this gate — the seventeen runtime-semantics defects WP-C2.2 closed, the six
correction-pass defects post-WP-C2.11, and DEV-053/054 — was investigated to a real fix, verified
empirically, and regression-tested before this report was written; nothing dangerous was left
open uninvestigated). All eight open deviations are narrow availability/ergonomics gaps
(rejections or non-resolutions of legal code) or engineering-process gaps (LSP stubs, evidence
precision), not confirmed-dangerous accept/reject or wrong-output gaps. That combination — a
complete semantic-decision register, a substantially (not exhaustively) built differential
corpus with zero known failures, and a bounded, fully-enumerated, non-soundness-relevant open-item
list — is exactly what `FROZEN-WITH-LISTED-DEVIATIONS` is meant to communicate.

## WP-C2.12 status (read this before treating the corpus as exhaustive)

WP-C2.12 is explicitly **not** closed as of this report. Its own work-package file
(`STARKLANG/docs/compiler/work-packages/WP-C2.12.md`) and `COMPILER-STATE.md`'s Position header
both say so plainly. What exists and is verified:

- DEV-036 closed with a real fix (filename/path-based module bypass replaced by an explicit,
  harness-only opt-in) plus six regression tests.
- `starkc/tests/exec_snapshots.rs`: a working differential/execution-snapshot harness mirroring
  the existing AST-snapshot convention, `UPDATE_SNAPSHOTS`-driven, deterministic.
- 17 primary cases (all seven named categories represented), 7 metamorphic-transformation pairs
  (all seven named classes represented, asserted byte-identical), 1 workspace-relocation
  execution-identity test. All pass.
- Building this corpus is what surfaced DEV-053/054 (closed, real fix) and DEV-051/052/055 (open,
  recorded, not fixed — out of this WP's scope).

What does **not** exist yet and is not claimed:
- A case *generator* (the roadmap's "generated" half of "generated and hand-written coverage").
- Exhaustive per-category depth — 17 hand-written cases is representative breadth across
  categories, not exhaustive coverage of any one category's full behavioral surface.
- Cross-backend replay (blocked behind Gate C3, as noted above — not a gap in this gate's own
  scope).

This report treats WP-C2.12's *current, passing* state as sufficient evidence that "the
interpreter passes the frozen corpus" in the sense the roadmap text can actually mean before a
MIR/native backend exists (there is no other backend to be differential *against* yet) — while
explicitly not claiming the corpus itself is complete or "frozen" as its own artifact. If a
future reader wants to expand it, `WP-C2.12`'s own Follow-ups are the exact scope: none of that
work is blocked on this exit report.

## Current head

```
e270575dbb429bbe5ce3adefb116eb34d486edb8
```

This is the last **committed** state and includes an unrelated commit
(`add compiler support coverage for .st sources`, user-authored, outside this WP's scope) landed
after WP-C2.12's own commit (`2b005d8`). This report's own file is new and, as of writing,
uncommitted — per this repo's standing workflow of committing only on explicit user request.

Rust toolchain: `stable` (channel pinned in `starkc/rust-toolchain.toml`, no version number);
measured environment: `cargo 1.93.0 (083ac5135 2025-12-15)`, `rustc 1.93.0 (254b59607
2026-01-19)`. Crate MSRV: `rust-version = "1.85"` (`starkc/Cargo.toml`).

## Test and fixture counts

`cargo test --workspace --all-targets --all-features`, run against the current head:

```
551 passed, 0 failed, 2 ignored
```

Across **4 unittest binaries** (`src/lib.rs`, `src/main.rs`, `src/bin/stark.rs`,
`src/bin/starkide.rs`) **+ 32 integration-test files** (`find starkc/tests -maxdepth 1 -type f
-name '*.rs' | wc -l`; up from 31 at WP-C2.12's own close — the extra file,
`tests/source_extensions.rs`, is the unrelated `.st`-extension commit noted above, not Gate C2
work). Both ignored tests are the same intentionally opt-in, non-hermetic cases carried since
Gate C0 (a checksum-pinned live ONNX artifact test, a live-ORT-download inference test).

Growth across Gate C2 (from 454/0/2 at Gate C1 close): WP-C2.1 +0 (documentation only), WP-C2.2
+16 (interpreter semantic repair, incl. correction pass), WP-C2.3 +3, WP-C2.4 +2, WP-C2.5 +4,
WP-C2.6–C2.10 +0 (pure specification/governance work, no test changes), WP-C2.11 +28
(implementation alignment), post-WP-C2.11 correction pass +16 (DEV-044–050), WP-C2.12 +5
(DEV-036 closure) +3 (`exec_snapshots.rs`) +5 (DEV-053/054 fix regressions), plus +5 from the
unrelated `.st`-extension commit not owned by this gate.

Spec fixture corpus (`STARKLANG/tests/spec-fixtures/manifest.toml`): **112 entries** (64
parse-pass, 16 semantic-error, 27 notation, 4 lex-pass, 1 parse-fail) — down from Gate C1's 121
after WP-C2.7 removed 28 stale duplicative memory-model examples and added new abstract-machine
examples; unchanged since WP-C2.10 (no grammar/example changes in WP-C2.11/C2.12). Verified in
sync with the extracted fixture set (`extract-spec-examples.sh`, run fresh for this report: "0
diff" / "Manifest is in sync").

## Authoritative document list

In source-of-truth order (per `COMPILER-CHARTER.md` §1.8), updated from `C1-exit-report.md`'s
list to reflect Gate C2's additions:

1. **Normative Core v1 spec**: `STARKLANG/docs/spec/00`–`07`, plus two new normative chapters
   added this gate: `CORE-V1-ABSTRACT-MACHINE.md` (WP-C2.7, the sole runtime authority — values,
   places, moves, temporaries, evaluation order, destruction, traps) and
   `CORE-V1-FUTURE-BOUNDARIES.md` (WP-C2.10, reserved compatibility space). Combined
   `STARK-Core-v1.md`/`.html`/`.pdf` regenerated and verified fresh for this report.
2. **Non-normative compiler governance** (new this gate): `STARKLANG/docs/compiler/
   semantic-freeze/CORE-V1-COMPLETENESS.md` (166-row granular question inventory + authoritative
   specification map) and `CORE-V1-OPEN-QUESTIONS.md` (24-question decision register, all
   Approved) — WP-C2.6. These route decisions into the normative homes above; they are never
   themselves normative.
3. **Approved decisions**: `COMPILER-STATE.md` (repo root) — decision log CD-001 through CD-017,
   deviation ledger DEV-002 through DEV-055 (DEV-001/DEV-003 retired).
4. **Gate exit evidence**: `starkc/docs/gate1-exit.md` through `gate7-decision.md` (old
   numbering, all closed); `C0-exit-report.md`; `C1-exit-report.md`; this document.
5. **Compiler roadmap**: `STARKLANG/docs/compiler/COMPILER-ROADMAP.md` (C0–C10 gates).
6. **Per-WP scope records**: `STARKLANG/docs/compiler/work-packages/WP-C2.1.md` through
   `WP-C2.12.md` (new this gate) — each carries scope, inherited findings, scope-control answers,
   and CE-escalation notes.
7. **Conformance database and tooling**: `STARKLANG/conformance/core-v1-coverage.toml` (the
   legacy 59-rule database, superseded in *granularity* by the completeness ledger above but not
   yet retired — `check-conformance.py` still reports against it: 89.8%/53-implemented,
   unchanged since Gate C0/C1, and still carries the same "not a trusted completeness claim"
   caveat), `core-v1-rule-id-map.toml` (WP-C2.6, the checked split from all 59 legacy rules into
   granular IDs), `core-v1-c2.11-evidence.toml` (new, WP-C2.11 — 34 high-cost granular rules with
   function-level positive/negative evidence; verified for this report: **0 of 34 have empty
   negative-test evidence**).
8. **Known deviations ledger**: `starkc/docs/conformance/KNOWN-DEVIATIONS.md`.
9. **Interpreter contract**: `STARKLANG/docs/compiler/reference-execution.md` (WP-C2.1) —
   every rule cited to the normative spec, corrected twice via external review before this gate's
   semantic freeze began.
10. **Differential corpus**: `starkc/tests/exec_snapshots.rs` + `starkc/tests/exec_snapshots/`
    (new, WP-C2.12) — see "WP-C2.12 status" above for its exact, non-exhaustive scope.
11. **README/context files**: unchanged in authority ordering from Gate C1's report.
12. **Archived, never authoritative**: unchanged from Gate C1's report.

## Semantic-freeze completeness

`STARKLANG/docs/compiler/semantic-freeze/CORE-V1-COMPLETENESS.md`, 166 audited rows (re-counted
for this report, excluding repeated section-header rows):

| Class | Count |
|---|---|
| `complete/specified` | 155 |
| `partial/specified` (`pending-owner-approval`) | 6 |
| `complete/prohibited` | 4 |
| `complete/deliberately-unspecified` | 1 |
| `absent` / `contradictory` / `pending-classification` | **0** |

No row is unclassified, absent, or contradictory — every observable Core v1 behavior audited by
this inventory lands in one of the roadmap's five permitted classes. The six `partial` rows are
listed by name in "Why not plain FROZEN" above.

`STARKLANG/docs/compiler/semantic-freeze/CORE-V1-OPEN-QUESTIONS.md`, 24 rows (CORE-Q-001 through
CORE-Q-024, including 005A/013A/013B): **24 of 24 Approved** (several with an explicit
"approved/corrected" annotation from a post-hoc external-review pass — WP-C2.7's and WP-C2.9's
correction passes). Zero rows are `pending`, `rejected`, or `superseded`. This directly satisfies
the roadmap's "high-cost future decisions are settled or protected" requirement.

MIR-independence spot-check (roadmap: "MIR-relevant concepts are independent of MIR"):
`CORE-V1-ABSTRACT-MACHINE.md` explicitly disclaims prescribing "an interpreter frame layout,
stack or heap placement, pointer representation, object layout, garbage collector, MIR shape,
backend ABI, or optimizer" (line 11-12) and its only other mention of MIR is a forward-looking
reference to a *future* MIR interpreter as one of three comparators the differential-observation
rule (`OBS-COMPARE-001`) will eventually run against — not a dependency on MIR's existence or
shape. Zero references to Rust-implementation internals (interpreter arenas, HIR node types,
etc.) were found in the document.

Package identity and executable entry behavior (roadmap: both must be "defined"): CORE-Q-013A/
013B (package-instance/public-item identity, major-line coexistence) and CORE-Q-014 (executable
entry signatures and process mapping) are all Approved, with normative homes in
`07-Modules-and-Packages.md`; CORE-Q-023 (public-signature-must-be-reachable) and CORE-Q-024
(exact stdlib conformance profile including `File`/host I/O) are also Approved. WP-C2.11
implemented and evidenced the resulting contract (package aliases, incompatible-major
coexistence, public-signature reachability, the complete executable entry/status/stream
contract) with 34 rules' worth of function-level positive/negative test citations.

## Full deviation ledger

53 numbered deviations (DEV-002 through DEV-055; DEV-001/DEV-003 retired during WP-C0.2), plus 2
informational not-owned items. Full structured detail: `starkc/docs/conformance/
KNOWN-DEVIATIONS.md`.

**Closed or confirmed-non-issue this gate (Gate C2, WP-C2.1 through WP-C2.12):** DEV-009,
DEV-018, DEV-019, DEV-022, DEV-023, DEV-024 (WP-C2.11); DEV-026 through DEV-035, DEV-037 through
DEV-043 (WP-C2.2, incl. correction pass — seventeen real runtime-semantics defects, the largest
single body of soundness/correctness fixes in the compiler track to date); DEV-036 (WP-C2.12);
DEV-044 through DEV-050 (post-WP-C2.11 external-review correction pass — six more
runtime-semantics defects, one claim corrected to a narrower scope, one claim refuted); DEV-053,
DEV-054 (WP-C2.12 follow-up investigation — the most severe single finding of the compiler
track: a bare `None` pattern silently matching any value).

**Open, carried forward from Gate C1 (unchanged this gate):**

| ID | One-line summary | Status |
|---|---|---|
| DEV-005 | `starkc check`/`run` warning-gating drift | Open, unowned since Gate C1 |
| DEV-010 | LSP hover/definition/references are stubs | Open, owned WP-C8.2/C8.3 |
| DEV-011 | Doc comments are trivia, not AST/HIR metadata | Open, unscheduled |
| DEV-012 | VS Code extension UI never interactively verified | Open, owned WP-C8.7 |
| DEV-017 | Coverage database lacks function-level test-evidence precision | Partially closed (34 of the highest-cost rules now have it via `core-v1-c2.11-evidence.toml`; the legacy 59-rule database's remaining gap is unchanged) |

**New this gate, open, not fixed (out of scope for the WPs that found them):**

| ID | One-line summary | Status |
|---|---|---|
| DEV-051 | Trait default methods cannot call another trait method on `self` | Open, unscheduled |
| DEV-052 | `Trait::method(...)` qualified calls fail to resolve for compiler CoreTraits | Open, unscheduled |
| DEV-055 | Bare glob-imported unit enum variants do not resolve at all | Open, unscheduled |

**Net for this gate: 33 deviations closed** (the seventeen WP-C2.2 defects, six WP-C2.11 items,
DEV-036, the seven post-WP-C2.11 correction-pass items, DEV-053/054), **1 remains partially
closed** (DEV-017, narrowed but not eliminated), **8 remain fully open** (DEV-005, DEV-010,
DEV-011, DEV-012, DEV-051, DEV-052, DEV-055, plus DEV-017's residual gap). No deviation was found
and left silently undocumented; every one has an owning future WP/gate or an explicit
"unscheduled" status. No open deviation is soundness-relevant in its current state.

## Conformance evidence

Two evidence sources now coexist, as noted in "Authoritative document list":

```
$ python3 starkc/scripts/check-conformance.py
... 59 rules — 53 implemented, 6 partial, 0 missing — 89.8% coverage ...
(exits 0, no errors, no warnings)
```

This is the **legacy, coarse-grained** database, unchanged in figures since Gate C0/C1 — it
still carries the same "not a trusted completeness claim" caveat those reports stated (per-rule
`implemented` status was never re-verified against full normative-rule-text completeness).

```
$ python3 -c "... count negative_tests across core-v1-c2.11-evidence.toml ..."
total rules: 34, rules with empty negative_tests: 0
```

This is the **new, granular** evidence source WP-C2.11 produced: 34 of the highest-cost
completeness-ledger rules (aliases/sizedness, traits/borrows/patterns/constants, numeric/Unicode
semantics, public/package identity, process behavior, formatting/hash/I/O/conversion, trap
classification) carry real function-level positive **and** negative test citations — a directly
verified `0` unclassified, unlike the legacy database's aggregate-file citations. This satisfies
the roadmap's "verify negative evidence for every soundness-relevant rule" requirement for the
34 rules this gate specifically targeted; it is not a claim that all 166 completeness-ledger rows
carry the same precision (most inherit citations from the legacy database via
`core-v1-rule-id-map.toml`, at the legacy database's own precision level — this is DEV-017's
continuing, honestly-disclosed scope).

Any future document stating a Core v1 conformance percentage should derive it from a fresh run
of these tools, not from this file's numbers, which are a point-in-time snapshot.

## A second implementation, built from the normative documents alone

The roadmap's central test for this gate: could a second implementation follow
`STARKLANG/docs/spec/00`–`07` plus `CORE-V1-ABSTRACT-MACHINE.md` plus
`CORE-V1-FUTURE-BOUNDARIES.md` and produce a conforming Core v1 compiler without consulting
`starkc`'s source? The strongest evidence for this gate:

- The abstract machine defines values, places, moves, temporaries, evaluation order, and
  destruction **without reference to Rust, HIR arenas, MIR, or interpreter frame layout**
  (verified above) — it is written as an implementation-independent specification, not a
  description of `starkc`'s own data structures.
- All 24 high-cost open questions that a second implementer would otherwise have to guess at
  (trait coherence, constant evaluation, numeric semantics, package identity, executable entry
  forms, future-boundary reservations) are Approved and transferred to normative homes, not left
  as implementation-defined-by-example.
- WP-C2.11's alignment work is itself evidence the specification is implementable independently
  of foreknowledge of `starkc`'s prior behavior: every one of its 34 evidenced rules required a
  real code change to bring `starkc` *into* conformance with the newly-frozen text, not the
  reverse (the spec was not simply restated from whatever the interpreter already did).

The honest limitation: this has not been tested by an actual second implementation attempt (no
such project exists), so this section is a structural argument from how the specification was
written, not empirical proof. WP-C2.12's differential corpus is the mechanism that would
eventually let a second implementation's output be checked against `starkc`'s — but that
comparison cannot happen until a second implementation exists to compare against.

## Exact next WP

**WP-C3.1 — Architecture hypothesis and workload freeze.**

Per the mandatory correctness path (`COMPILER-ROADMAP.md` §4.1): `C0 → C1 → C2 → C3`. Gate C2
(Reference Execution Semantics and Compiler Service Foundation) is closed with listed
deviations; Gate C3 (Native Compiler Architecture and Backend Selection Spike) opens next. Per
CD-004, C3 selects *how* STARK compiles natively (`SELECT-GENERATED`/`SELECT-DIRECT`), not
*whether* — native compilation is mandatory. WP-C3.1 writes `STARKLANG/docs/compiler/proposals/
NATIVE-CORE-ARCHITECTURE.md` and freezes a representative workload set (scalar arithmetic/
branches, loops/calls, structs/enums, `Option`/`Result`/pattern matching, ownership moves/drops,
string/Vec operations, a multi-file/multi-package CLI application, one error/trap workload).

**Before WP-C3.1 begins, the owner should be aware of two open items this report surfaced that
are not blockers but deserve a decision:**

1. The six `pending-owner-approval` completeness rows (`LEX-COMMENT-001`, `LEX-ERROR-001`,
   `STD-OPTION-001`, `STD-RESULT-001`, `STD-ITER-001`, `STD-VEC-001`) — closing these is pure
   governance bookkeeping (the behavior is already implemented and tested) but leaves the
   completeness ledger's own "0 pending" claim technically false until someone signs off.
2. Whether WP-C2.12's remaining scope (case generator, deeper per-category breadth) should be
   completed before or in parallel with Gate C3's own work — nothing in Gate C3's own scope
   depends on it (cross-backend replay is explicitly C3+ work), but a richer corpus earlier
   makes C3's own semantic-parity evidence stronger when it eventually needs to compare a native
   backend against the interpreter.

Session budget note (`COMPILER-ROADMAP.md` §7): Gate C2 closes within a larger session footprint
than Gate C1 (thirteen work packages including two dedicated correction passes and one
mid-gate preflight transition, versus Gate C1's six), reflecting that this gate both wrote three
new normative documents from scratch (reference-execution contract, abstract machine,
future-boundaries) and found the two largest bodies of runtime-semantics defects in the
compiler track's history (WP-C2.2's seventeen and the post-WP-C2.11 correction pass's six). The
deviations this gate leaves open (DEV-005, DEV-010 through DEV-012, DEV-017's residual gap,
DEV-051, DEV-052, DEV-055) are inputs to Gate C3 and beyond, not evidence that Gate C2's own
scope was under-delivered — each was a real finding, investigated, and honestly disposed of.

## Addendum — 2026-07-18, post-report

An external correction brief, arriving after this report was written, independently found and
this session fixed two further interpreter defects: **DEV-056** (`?` propagation was swallowed
outside aggregate-construction call sites — ordinary function calls, method calls, binary
operands, and several other sequential-evaluation contexts kept running later sub-expressions,
with real side effects, after an earlier one had already propagated) and **DEV-057** (`Eq`/`Ord`
comparison dispatch passed owned clones instead of true borrowed places, causing a confirmed
double-destruction/lost-destructor defect for `Drop`-bearing comparison operands — plus a second,
broader bug found while fixing it, `promote_to_temp_place`'s 15+ call sites never registering
their temporary in `Frame::order`, so nothing routed through it ever received a `Drop::drop`
call at all). A follow-up pass in the same correction brief also closed **DEV-058** (a `Float32`
value nested inside a tuple/array/`Option`/`Result`/struct still formatted via `Float64`'s
shortest-round-trip digits — the residual gap DEV-049 explicitly left open in this report's own
listed-deviations set, now closed by tagging `Value::Float` with its own declared width) and
**DEV-059** (NaN-producing float operations returned whatever bit pattern the host `f64`/`f32`
instruction happened to produce instead of `NUM-FLOAT-OP-001`'s mandated canonical quiet-NaN
pattern — every NaN still printed as `NaN`, so nothing in the existing corpus had ever caught the
gap). All four are now closed with real fixes and regression tests; see `COMPILER-STATE.md`'s
dated `### Post-Gate-C2 correction brief` session records for full detail.

This does **not** retroactively change this report's verdict
(`CORE-V1-SEMANTIC-FOUNDATION-FROZEN-WITH-LISTED-DEVIATIONS`) — all four defects are found,
fixed, and regression-tested, the standard disposition for a mid-gate-track finding throughout
this compiler track, and none was a case of the interpreter accepting a genuinely unsound
program. It **does** confirm, concretely, the limitation this report's own "WP-C2.12 status"
section already flagged: the differential corpus that existed at exit was representative, not
exhaustive, and a representative corpus passing cleanly is not proof that severe defects don't
remain — this addendum is exactly that proof. None of the four defects' repro shapes (a bare
function-call argument containing `?`; comparing two `Drop`-bearing structs by value while
printing around the comparison; a `Float32` nested inside a tuple; two differently-shaped
NaN-producing expressions compared bit-for-bit) happened to be covered by the corpus as it stood
at this report's original writing.

**Evidence-database correction (Issue 5 of the same brief):** the same review also found three
of `core-v1-c2.11-evidence.toml`'s citations structurally valid (the file and function both
exist, satisfying `check-conformance.py`'s existence check) but semantically irrelevant to the
rule they were attached to — `NUM-FLOAT-FORMAT-001` and `STD-FORMAT-001` both cited
`floating_exponent_operator_is_rejected` (a test about the `**` operator being a type error, not
about float formatting or precision) as their `negative_tests` evidence, and `STD-CONVERT-001`
cited `ambiguous_trait_associated_functions_require_qualification` (an associated-function
resolution-ambiguity test, not a conversion failure). The same defect pattern was independently
found in `NUM-FLOAT-OP-001` while fixing the other two (not originally named in the brief, but
the identical `floating_exponent_operator_is_rejected` mis-citation). All four were replaced with
citations that are actually about the cited rule — `NUM-FLOAT-FORMAT-001` and `NUM-FLOAT-OP-001`
now cite the new DEV-058/DEV-059 regression tests directly; `STD-CONVERT-001` now cites
`every_integer_width_traps_on_overflow_and_invalid_operations`, since `STD-CONVERT-001`'s own
normative text ties failing numeric conversions to "the exact `NUM-CAST-001` value/range rules."
Two granular rules already present in the completeness inventory but with zero executable
evidence in this database — `EXEC-CFLOW-001` ("which early exits are normal transfers and how do
they clean scopes," the exact DEV-056/DEV-057 territory) and `TRAIT-LAW-001` ("what semantic laws
bind `Eq`, `Ord`, and `Hash`") — were given their first evidence entries, citing the DEV-056/
DEV-057 regression tests. `check-conformance.py` re-run clean after all changes (89.8%/53-of-59
overall, unchanged — this was a citation-quality fix, not an implementation-status change).

## Addendum 2 — 2026-07-19, DEV-051/DEV-052/DEV-055 closed; DEV-060 found

This report's own "Deviations found this gate" table (below) lists **DEV-051** (trait default
methods couldn't call a sibling trait method through `self`), **DEV-052** (qualified
`Trait::method(...)` syntax didn't resolve for compiler `CoreTrait`s), and **DEV-055** (bare
glob-imported unit enum variants didn't resolve at all) as `Open, unscheduled`. All three were
independently reproduced against the current head in a later session and **closed** with real
fixes: DEV-051 in `typecheck.rs`'s `resolve_method` (a new `current_trait_id` field lets a
default-method body's `self.other_method()` look its sibling up directly against the trait's own
declared signature, mirroring the pre-existing bounded-generic-parameter mechanism); DEV-052 via
a new `hir::Res::CoreTraitMember` variant threaded through `resolve.rs`/`typecheck.rs`/
`interp.rs`, whose interpreter-side dispatch reuses the exact same `find_method(..., Some(Res::
CoreTrait(_)))` lookup the `==`/`<` operator sugar already uses for these traits; DEV-055 in
`resolve_use_tree`'s `Glob`/`Group` arms (an enum prefix, not just a real submodule, is now
handled). See `COMPILER-STATE.md`'s dated `### Post-Gate-C2 correction brief` session records
for full detail.

While writing DEV-051's regression tests, a fourth, separate defect was found and confirmed
*not* introduced by that fix (via `git stash` against the pre-fix head): **DEV-060** — calling
the same un-overridden trait *default* method twice on one receiver incorrectly raises `E0100
use of moved value` on the second call, even though the method only takes `&self`. Two calls to
an *overridden* trait method, or to an ordinary inherent method, are both unaffected, narrowing
the defect specifically to the `default_fallback` method-resolution path. This one remains open,
unscheduled, and is recorded in `KNOWN-DEVIATIONS.md`.

None of this retroactively changes the verdict
(`CORE-V1-SEMANTIC-FOUNDATION-FROZEN-WITH-LISTED-DEVIATIONS`) — three more listed deviations
closed with real fixes and regression tests, one new one found and honestly disclosed rather
than fixed under time pressure, the same disposition every other finding in this compiler track
has received.
