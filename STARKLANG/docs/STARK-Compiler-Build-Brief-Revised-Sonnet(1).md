# STARK Compiler Build Brief — Revised Sonnet Edition (Native Compiler Required)

<!-- MASTER BRIEF: use this to bootstrap and govern the compiler track. -->
<!-- Do NOT feed this entire roadmap at the start of every later session. -->
<!-- After bootstrap, feed only COMPILER-CHARTER.md + COMPILER-STATE.md + the active WP. -->

**You are:** Claude Sonnet, acting as the implementation engineer for the STARK compiler track.
You work in bounded work packages, preserve the normative language contract, distinguish implementation repair from language design, keep a shared state file current, and escalate the small number of architectural or semantic decisions listed in this brief.

This document is the **master compiler roadmap**, not the routine session payload.
On the first session, execute Gate C0 bootstrap and split the durable guidance into:

```text
STARKLANG/docs/compiler/COMPILER-CHARTER.md
    source-of-truth rules, track boundaries, standing constraints,
    escalation rules, not-yet list

STARKLANG/docs/compiler/COMPILER-ROADMAP.md
    gates, work packages, dependencies, exit criteria, mandatory compiler path, optional tracks

COMPILER-STATE.md
    current head, position, decisions, conformance state, deviations,
    architecture choices, evidence, follow-ups, gate exits

STARKLANG/docs/compiler/work-packages/WP-C<gate>.<n>.md
    the exact active work package and its acceptance checks
```

For every later work session, the intended input packet is:

```text
COMPILER-CHARTER.md
+ COMPILER-STATE.md
+ the active WP file
+ only the specifications, source files, tests, and prior decisions
  directly referenced by that WP
```

Load the full compiler roadmap again only when closing a gate, selecting an optional track or backend tier, resolving a roadmap-level contradiction, or reviewing a proposed change to Core semantics.
This reduces the tendency to pull later compiler mechanisms into earlier correctness work.

---

# 1. PROJECT CONTEXT — FIXED, DO NOT RE-DERIVE

**STARK** is an ownership-safe, Rust-inspired research programming language with a normative Core v1 specification and an optional tensor/model extension.
The repository already contains a Rust lexer, parser, resolver, typed HIR, type checker, borrow checker, interpreter, multi-file/module support, package graph support, tensor/ONNX import and verification, bounded native ONNX deployment, a formatter, test runner, documentation generator, LSP foundation, and editor integration.

The latest known repository head when this master brief was prepared was:

```text
6fa8c15b94bd1376a847132498d31dd356524180
```

Treat that as a **starting reference, not a permanent truth**.
Gate C0 must read the current head and establish what is implemented, partial, stale, stubbed, inconsistent, or only claimed in documentation.
Do not rebuild a subsystem merely because an older roadmap says it is missing.
Do not mark a subsystem complete merely because a commit message says it is complete.

## 1.1 Compiler-track responsibility

The compiler track owns:

- lexical and syntactic conformance;
- AST, HIR, source maps, symbol identity, and compiler query infrastructure;
- name resolution, type checking, trait checking, coherence, ownership, borrowing, drop analysis, exhaustiveness, and other Core semantics;
- the reference interpreter and executable semantics;
- a mandatory backend-independent MIR that preserves Core execution semantics;
- mandatory general native Core compilation that produces standalone executables;
- compiler diagnostics and machine-readable diagnostic protocols;
- compiler-side semantic services required by LSP clients;
- extension isolation and compiler-side extension-provider infrastructure;
- conformance, fuzzing, differential testing, compatibility, and release qualification.

The compiler track does **not** own:

- package ecosystem product design, public registry infrastructure, package-author workflows, or third-party package policy;
- standard network/TLS/HTTP package design;
- cloud-provider syntax or infrastructure-as-code features;
- the OpenAPI product experiment itself;
- VS Code UI behaviour beyond the compiler/LSP protocol it consumes;
- new language features merely because a tool or package would find them convenient.

The ecosystem and compiler tracks may depend on one another, but neither may silently absorb the other's scope.

## 1.2 Guaranteed compiler completion state

This roadmap is complete only when STARK has a general native compiler for Core programs.
The required end-to-end path is:

```text
STARK source
  -> parse and resolve
  -> type, trait, ownership, and borrow checking
  -> typed HIR
  -> verified MIR
  -> selected native backend
  -> standalone executable
```

The following are mandatory outcomes, not optional research branches:

- MIR with explicit control flow, moves, drops, calls, aggregates, and trap paths;
- a MIR verifier and MIR interpreter used for differential validation;
- a native backend capable of compiling ordinary multi-file, multi-package Core applications;
- interpreter/MIR/native semantic parity for the supported Core v1 surface;
- debug and release build profiles;
- documented target support and reproducible build metadata;
- a release statement that explicitly includes the native Core backend.

Gate C3 may select generated Rust/C, Cranelift, or another approved backend strategy.
It may reject a candidate, request one bounded revision spike, or escalate an architectural blocker.
It may not conclude that an interpreter-only implementation is an acceptable completed compiler track.
If no credible native path can be selected, the compiler roadmap remains blocked or the owner records a project-level stop decision; it does not silently redefine completion.

## 1.3 North-star technical thesis

> **STARK investigates artifact-bound programming: application code and the external artifacts it depends on are verified as one typed system rather than through disconnected tools.**

The existing ONNX work is one artifact implementation.
The compiler track must preserve that implementation, but it must not generalise it prematurely into public Core syntax.
A generic artifact-provider abstraction is considered only after a second independent artifact implementation exists and exposes materially shared compiler needs.

## 1.4 Keep four questions separate

Never allow one successful experiment to answer a different question by implication:

1. **Core correctness:** Does the compiler implement the normative Core v1 language accurately and consistently?
2. **Native compiler architecture:** Which MIR, runtime ABI, and native backend strategy can deliver standalone Core executables with acceptable correctness, portability, and maintenance cost?
3. **Artifact-binding generality:** Does the source/artifact verification mechanism generalise beyond ONNX?
4. **AI-development methodology:** Does AI materially improve language-development velocity or quality under controlled governance?

The compiler roadmap must complete question 1 and deliver a concrete implementation answer to question 2.
Question 3 requires the ecosystem's second-artifact experiment.
Question 4 requires its own development-process evidence and is not a substitute for semantic correctness.

## 1.5 Existing evidence and strategic constraint

The tensor track has already produced a bounded positive result in a computer-vision deployment workload.
It demonstrated useful symbolic-shape and artifact-drift guarantees against a strong typed-Rust comparator.
The recorded project decision is to retain STARK as a research language while external demand remains unproven.

Therefore:

- maintain the working tensor/ONNX path;
- do not use it to justify unrestricted language expansion;
- keep `stark verify` external-user validation independent of compiler expansion;
- do not build a full tensor runtime, VM, GPU compiler, or broad domain system without a new bounded proposal and evidence gate.

## 1.6 Governing rules — violating any is a wrong implementation

1. **The normative specification defines language behaviour.** The compiler conforms to the spec; the spec is not rewritten merely to excuse an implementation shortcut.
2. **Spec defects are real and may be fixed.** A spec fix must include the normative source change, regenerated combined documents, fixture re-triage where relevant, compiler change, and executable evidence in one bounded change.
3. **Implementation repair is not language design.** If the compiler disagrees with the spec, first determine whether the compiler or specification is wrong; do not invent a third behaviour.
4. **No new Core syntax or semantics inside an implementation WP.** New features require a separate proposal, concrete blocked use case, alternatives, compatibility analysis, and owner approval.
5. **Core remains extension-neutral.** Tensor/model rules must not leak into Core-only parsing, resolution, typing, diagnostics, formatting, tests, or LSP sessions.
6. **The interpreter is the semantic reference until a later gate explicitly changes that decision.** Native execution must match it or expose a documented spec defect; native behaviour does not silently redefine the language.
7. **Correctness precedes optimisation.** No optimisation pass is accepted until the unoptimised path passes the differential semantics suite.
8. **Stable Rust only.** Nightly compiler features are prohibited unless separately approved with a portability and contributor-cost analysis.
9. **Native compilation is mandatory; backend lock-in is not.** Cranelift, generated Rust, generated C, LLVM, or another backend is selected through Gate C3 evidence, but the roadmap may not choose an interpreter-only completion state.
10. **No LLVM by prestige.** LLVM enters the roadmap only if measured workloads show a material need that the selected simpler backend cannot meet.
11. **No custom VM by default.** A bytecode VM requires a separate evidence-backed proposal; MIR and a MIR interpreter do not automatically imply a production VM.
12. **No silent semantic stubs.** A handler or compiler path returning placeholder output must be labelled `stub` in docs and tests.
13. **No status drift.** When implementation status changes, update the authoritative state/coverage records and user-facing summaries in the same work package.
14. **Conformance claims require executable evidence.** Parsing, type-checking, or a commit message alone never proves a normative rule complete.
15. **Positive and negative evidence travel together.** Every semantic rule needs valid and invalid cases where rejection is meaningful.
16. **Diagnostics are part of behaviour.** Error code, primary span, related spans, notes, help text, and machine-readable form must remain testable and deterministic.
17. **Source identity must survive the pipeline.** AST/HIR/MIR/query results and diagnostics must retain the correct file, module, package, and artifact provenance.
18. **Cross-tool compiler behaviour must converge.** CLI, test runner, doc example validator, LSP, deployment tool, and future native builder must use shared compiler entry points rather than subtly different pipelines.
19. **Do not generalise from one extension too early.** Internal artifact-provider infrastructure is promoted only after two independent artifact implementations demonstrate the same need.
20. **Negative candidate results are valid.** A backend candidate may be rejected, an optimisation tier or generic abstraction may be deferred, and a bounded architecture revision may be required. General native Core compilation itself is a mandatory completion requirement.

## 1.7 Explicit not-Core list

The following are not authorised by this compiler roadmap:

```text
async / await
closures or lambdas
macros
unsafe blocks
raw pointers
general C FFI
trait objects / dyn
lifetime annotation syntax
actors
garbage collection
reflection
compiler plugins
cloud annotations or provider syntax
capability or effect syntax
information-flow label syntax
public Artifact<...> syntax
self-hosting
```

Any one of these requires a separate language proposal and evidence gate.
Do not implement it as an incidental solution to a compiler, tooling, standard-library, package, or native-backend problem.

## 1.8 Terminology

- **normative specification** — the individual Core v1 documents under `STARKLANG/docs/spec/` and approved extension specifications under `STARKLANG/docs/extensions/`.
- **generated specification** — combined Markdown/HTML/PDF outputs; regenerated artifacts, never the editing surface.
- **AST** — syntax-preserving arena representation produced by the parser.
- **HIR** — resolved/desugared semantic representation used by type checking, borrow checking, interpretation, and compiler queries.
- **typed HIR** — HIR plus type, resolution, ownership, and related side tables.
- **MIR** — mandatory backend-independent control-flow IR introduced after Gate C3 selects the native compiler architecture.
- **semantic oracle** — the normative spec plus the reference interpreter and conformance tests used to judge later backends.
- **compiler query** — a stable read-only semantic operation such as “symbol at position,” “type of node,” “definition of symbol,” or “references to symbol.”
- **extension provider** — compiler-internal integration for optional syntax/semantics; extensions remain explicitly enabled and isolated from Core.
- **artifact provider** — a compiler/tooling contract that reads an artifact, extracts a typed contract, records identity/provenance, verifies drift, and contributes diagnostics.
- **conformance deviation** — a known mismatch between implementation and normative behaviour, explicitly recorded with impact and planned disposition.
- **backend capability** — a declared statement of what a selected execution backend can actually lower and execute.

## 1.9 Source-of-truth hierarchy

When documents disagree, use this order:

1. approved normative Core or extension specification;
2. approved decision record in `COMPILER-STATE.md` or a gate proposal;
3. gate exit evidence and executable tests;
4. compiler roadmap;
5. engineering plan/work-package documentation;
6. README, CLAUDE context, implementation notes, and commit messages;
7. archived pre-pivot documents — never authoritative.

Gate C0 must identify stale documents and either update them or label them historical.

## 1.10 Standing implementation constraints

- Implementation language: stable Rust.
- Existing `starkc/` Rust implementation is the active compiler; archived Python compiler code is not extended.
- Arena IDs and explicit spans remain the default representation unless an approved architecture decision replaces them.
- The reference execution path remains the interpreter until a later gate records otherwise.
- New external Rust dependencies require a short necessity, maintenance, licence, and security note.
- Hand-written parsers/serialisers are not a virtue by themselves; use a mature dependency when it materially lowers correctness or maintenance risk and is compatible with project constraints.
- Avoid broad refactors that are not required by the active WP.
- Public compiler APIs must not expose unstable internal arena indexes as persistent external identities.

---

# 2. OPERATING PROTOCOL

## 2.1 Session flow

1. Load `COMPILER-CHARTER.md`, `COMPILER-STATE.md`, and the active WP file.
2. Confirm the repository head and current WP scope.
3. Read only the normative rules, source modules, tests, and prior decisions required by the WP.
4. Record any newly discovered adjacent problem as `FOLLOW-UP:` unless it blocks the WP's correctness.
5. Implement the WP without importing later-gate mechanisms.
6. Run every `Done when` check.
7. Run deterministic commands twice when deterministic output is claimed.
8. Update conformance records for every normative rule whose status changed.
9. Update `COMPILER-STATE.md` with files, decisions, deviations, evidence, follow-ups, and the next WP.
10. Commit with message `[WP-Cx.y] <summary>`.

At a gate exit:

1. reload the full compiler roadmap;
2. evaluate every criterion as PASS, FAIL, DEFERRED-BY-DECISION, or NOT-APPLICABLE;
3. write the gate exit report with command-level evidence;
4. obtain owner review for semantic, architecture, or scope conclusions;
5. only then open the next mandatory gate or an explicitly optional track.

## 2.2 Sonnet-level autonomy

You may:

- implement spec-consistent fixes in lexer, parser, resolver, type checker, borrow checker, interpreter, diagnostics, compiler query infrastructure, and tests when the active WP authorises those areas;
- refactor files already in scope when the refactor directly enables the WP and behaviour remains covered;
- add positive, negative, regression, fuzz, and property tests;
- split a WP before implementation when its acceptance surface is too large;
- improve diagnostics without changing the accepted/rejected program set, provided diagnostic tests and codes remain compatible or an approved correction is recorded;
- create temporary backend spikes inside Gate C3 that are not merged as product architecture until the gate decision.

You may not, without escalation:

- change the normative accepted/rejected program set;
- add syntax, keywords, attributes, operators, type forms, effects, or ownership rules;
- weaken ownership, borrowing, coherence, exhaustiveness, or artifact verification to make code compile;
- alter panic/trap/drop semantics;
- choose the MIR contract, runtime ABI, data layout, unwind model, or production backend;
- add nightly Rust requirements;
- introduce a new extension or artifact class;
- claim Core v1 or tensor v0.1 conformance;
- convert a known stub into a documented feature without implementing semantic behaviour;
- begin anything on the not-yet list.

## 2.3 Escalations

| ID | Decision | Why escalated |
|---|---|---|
| CE1 | Normative Core or tensor semantic change | changes the language contract and compatibility surface |
| CE2 | Resolution of a spec-versus-implementation ambiguity with multiple plausible behaviours | requires owner judgment, not implementation convenience |
| CE3 | MIR design and verifier contract | foundational architecture; mistakes propagate into every backend |
| CE4 | Runtime ABI, value layout, drop glue, panic/trap, and native resource model | cross-platform semantic and safety foundation |
| CE5 | Production backend selection after Gate C3 | large long-term maintenance and portability consequence |
| CE6 | Optimisation tier or LLVM adoption | must be justified by measured workloads |
| CE7 | Generic extension/artifact-provider abstraction | allowed only after two independent implementations expose common semantics |
| CE8 | Any Core conformance or release claim | public technical claim requiring an evidence audit |
| CE9 | Security-sensitive compiler behaviour | archive extraction, process execution, code generation, native linking, trust boundaries |

A spec bug whose correction is unambiguous may be handled under the established spec-bug protocol, but record it in `COMPILER-STATE.md` and the gate evidence ledger.

## 2.4 `COMPILER-STATE.md` shared memory

Bootstrap template:

```markdown
# STARK Compiler STATE
Updated: <date> after <WP-id>

## Position
Gate: C<n>  Next: WP-C<n>.<m>  Blocked: <none|reason>
Mandatory compiler path: Core=<open|done>  MIR=<blocked|open|done>  Native=<blocked|open|done>
Optional tracks: ArtifactInfra=<blocked|open|done>  TensorExpansion=<blocked|open|done>

## Repository baseline
- Head: <sha>
- Rust toolchain: <version>
- Test count / suites: <summary>
- Core spec revision: <paths + hashes or commit>
- Tensor spec revision: <path + hash or commit>

## Current compiler pipeline
- Source -> lexer -> parser -> AST -> HIR -> type/borrow -> interpreter
- Additional entry points: <CLI/LSP/doc/deploy/etc>

## Decision log — append-only
- CD-001 [WP-C0.1] <decision and rationale>

## Conformance summary
- Lexical: <implemented/partial/deviations>
- Syntax: ...
- Types: ...
- Semantics: ...
- Memory: ...
- Modules/packages compiler surface: ...
- Tensor extension: ...

## Known deviations — open until closed explicitly
- DEV-001 <normative rule> — <implementation behaviour> — <impact> — <owner>

## Architecture decisions
- AD-001 <decision>

## Native backend selection
- Status: not evaluated | SPIKING | SELECTED | REVISE | BLOCKED
- Selected strategy: <generated Rust/C | Cranelift | other approved option | none yet>
- Evidence: <links>

## Diagnostic codes allocated or changed
- <code> — <meaning>

## Evidence inventory
- <path> — <what it proves>

## File inventory for current gate
- <path> — <purpose>

## Follow-ups
- [ ] <bounded future item>

## Gate exit summaries
- C<n>: <decision and evidence>
```

Rules:

- decision log is append-only;
- deviations are never deleted without a closing note and evidence link;
- keep the state file under approximately 700 lines by compressing closed gate detail into summaries;
- every session ends with:

```markdown
### WP-Cx.y — <date>
DONE: <1–3 sentences>
FILES: <paths>
RULES: <normative rule IDs affected | none>
DECISIONS: <CD/AD/DEV ids | none>
EVIDENCE: <commands/tests/results>
FOLLOW-UP: <items | none>
NEXT: WP-Cx.z
```

## 2.5 Definition-of-done defaults

When a WP does not state stricter checks:

- `cargo fmt --check` passes;
- `cargo clippy --all-targets -- -D warnings` passes;
- the full Rust test suite passes;
- Core fixture conformance passes;
- tensor/extension tests pass when extension code is touched;
- every changed normative rule has positive evidence and negative evidence where applicable;
- every regression fix has a test that failed before the fix;
- diagnostics are snapshot- or structurally tested;
- no new undocumented `unwrap`, panic, global mutable state, nondeterministic iteration, or process execution is introduced in compiler paths;
- generated output is deterministic across two runs;
- docs distinguish complete, partial, stub, and deferred behaviour;
- `COMPILER-STATE.md` and coverage data are updated;
- nothing merges with failing checks.

## 2.6 Scope-control test before coding

Before each WP, answer internally:

```text
What exact compiler claim is this WP testing or completing?
What later mechanism would make the result impossible to attribute?
What is the strongest existing implementation path or comparator?
What negative result would cause this WP or gate to stop?
```

If those answers are unclear, narrow the WP before implementation.

---

# 3. GATES AND WORK PACKAGES

Gates close in order within each dependency chain.
Parallel tracks are allowed only where explicitly shown in §4.
A gate status is determined by evidence, not session count.

---

## GATE C0 — Bootstrap, Current-State Audit, and Authority Repair

*Outcome: one trusted account of what the compiler implements, what the specifications require, and which documents or status claims are stale.*

*This gate is mandatory even though substantial implementation already exists.*

### WP-C0.0 — Bootstrap compiler governance

Create:

```text
STARKLANG/docs/compiler/COMPILER-CHARTER.md
STARKLANG/docs/compiler/COMPILER-ROADMAP.md
COMPILER-STATE.md
STARKLANG/docs/compiler/work-packages/
```

Extract this brief into the first two files without changing meaning.
Record current head, toolchain, test commands, active normative specs, and all existing roadmap/plan/status documents in `COMPILER-STATE.md`.

**Done when:** a later session can operate using only the charter, state, and one active WP.

### WP-C0.1 — Repository and pipeline map

Deliver `starkc/docs/dev/compiler-map.md` containing:

- every active compiler module and purpose;
- input/output representation for lexer, parser, resolver, type checker, borrow checker, interpreter, ONNX tooling, formatter, docs, LSP, and package CLI;
- shared and duplicated compiler entry points;
- global state, filesystem access, process execution, and nondeterministic collections;
- source-file and module provenance flow;
- test files that cover each subsystem;
- stubs and deliberately incomplete handlers;
- archived code that must not be modified.

The map must identify where the same source is parsed or analysed through different pipelines.

**Done when:** every active compiler/tool entry point can be traced to the normative stage and tests that support it.

### WP-C0.2 — Documentation and status reconciliation

Audit at minimum:

```text
CLAUDE.md
README.md
starkc/README.md
STARKLANG/docs/ROADMAP.md
STARKLANG/docs/PLAN.md
stark-spec-parity-roadmap.md
STARKLANG/conformance/core-v1-coverage.toml
starkc/docs/gate*-exit.md
starkc/docs/*IMPLEMENTATION*.md
```

For every material claim, classify:

```text
current and evidenced
current but incompletely evidenced
stale
contradictory
historical
```

Update or label stale documents.
Do not rewrite historical gate evidence to match later implementation; add dated updates instead.

**Done when:** no active document says a completed gate is “next,” no README advertises an LSP stub as implemented semantic behaviour, and status summaries agree with `COMPILER-STATE.md`.

### WP-C0.3 — Conformance database integrity audit

The existing coverage database must be inspected, not trusted.

Deliver:

- a count of normative rules in each individual spec chapter;
- a stable rule-ID mapping;
- detection of duplicate, missing, orphaned, and stale rules;
- validation that every `implemented` entry points to real source and tests;
- validation that every `missing` or `partial` entry is still accurate;
- a machine-generated chapter report;
- a list of rules the current database cannot represent precisely.

Upgrade the conformance checker so it can detect more than missing files.
At minimum it must reject:

- duplicate rule IDs;
- references to nonexistent spec sections;
- `implemented` entries with no positive test;
- semantic rejection rules with no negative test;
- stale status where a linked completion marker or approved test contradicts `missing`.

The last check may use explicit implementation markers rather than guessing from source text.

### WP-C0.4 — Known-deviation and stub ledger

Create `starkc/docs/conformance/KNOWN-DEVIATIONS.md` and record every known issue found by audit.
Seed the review with, but independently verify:

- runtime operator equality versus user-visible trait semantics;
- extension-name and builtin isolation paths;
- hardcoded standard-library names versus a real `std` module/package surface;
- first-class resource/file-handle limitations in the interpreter value model;
- doc comments existing as re-associated trivia rather than AST/HIR metadata;
- LSP hover/definition/reference/diagnostic publication status;
- per-file versus project-wide analysis differences in formatter/doc tooling;
- any incomplete tensor backend capability;
- any claimed conformance rule lacking executable evidence.

Each entry must state:

```text
normative expectation
current behaviour
user impact
security/soundness impact
workaround
proposed disposition
owning future gate
```

### WP-C0.5 — Gate exit

Write `starkc/docs/compiler/C0-exit-report.md` with:

- current head;
- test and fixture counts;
- authoritative document list;
- subsystem status matrix;
- open deviation ledger;
- exact next WP;
- explicit statement that no conformance percentage is trusted unless generated from the repaired database.

**C0 closes only when the project's current state can be stated without relying on commit-message optimism or stale roadmap text.**

---

## GATE C1 — Core v1 Conformance Closure

*Outcome: an evidence-backed Core v1 implementation statement for the front end and semantic checker, with every remaining deviation explicit.*

*This gate adds no language features.*

### WP-C1.1 — Lexical and syntax requalification

Re-run and strengthen:

- every classified specification fixture;
- reserved-token handling;
- literal boundary and overflow lexing;
- nested comments and raw strings;
- generic-closing-token splitting;
- parser recovery progress guarantees;
- recursion/depth limits;
- AST span integrity;
- multi-file module parsing;
- extension syntax rejected in Core-only mode and accepted only when explicitly enabled.

Add deterministic property/fuzz tests for no panic, no hang, bounded resource use, and stable diagnostics on malformed input.

**Done when:** every lexical/syntax rule has a coverage entry and executable evidence; no known syntax deviation remains unrecorded.

### WP-C1.2 — Name resolution, modules, and visibility

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

The compiler track tests compiler semantics against the current package graph.
It does not redesign manifest or registry policy in this gate.

### WP-C1.3 — Types, generics, traits, and operator semantics

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

A hidden interpreter-only structural equality rule is not accepted as an undocumented third behaviour.

### WP-C1.4 — Ownership, borrowing, lifetimes, and drop checking

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

### WP-C1.5 — Control flow, patterns, constants, and numeric semantics

Cover:

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

### WP-C1.6 — Conformance evidence generator

Implement one command or CI job that emits:

```text
rule id
spec chapter/section
status
source implementation
positive tests
negative tests
deviation id if any
last verified commit
```

The report must be generated from machine-readable data, deterministic, and checked into CI as an artefact or summary.

### WP-C1.7 — Gate exit

Write `starkc/docs/compiler/C1-exit-report.md`.
The conclusion must be one of:

```text
CORE-FRONTEND-CONFORMING
CORE-FRONTEND-CONFORMING-WITH-LISTED-DEVIATIONS
CORE-FRONTEND-NOT-YET-CONFORMING
```

Do not use “complete” without the rule-level report.
Native compilation is not required for this gate.
Interpreter runtime parity is completed in Gate C2.

---

## GATE C2 — Reference Execution Semantics and Compiler Service Foundation

*Outcome: one trustworthy reference execution model and one shared project-analysis API used by compiler-facing tools.*

### WP-C2.1 — Reference interpreter contract

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

Every rule must cite the normative Core specification rather than inventing interpreter-specific semantics.

### WP-C2.2 — Interpreter semantic repair

Resolve all C0/C1 deviations that affect executed behaviour, including any confirmed issues in:

- operator/trait dispatch;
- default trait methods;
- method receiver mutation;
- structural versus semantic equality;
- enum and pattern representation;
- drop order and partial moves;
- panic/trap paths;
- iterator state and aliasing;
- collection mutation;
- standard builtin resolution;
- extension builtin gating.

No new public standard-library API belongs here unless required by the normative spec and approved under the ecosystem compatibility process.

### WP-C2.3 — Shared project-analysis entry point

Create a compiler API conceptually equivalent to:

```text
analyze_project(project_input, language_options)
    -> ProjectAnalysis
```

It should own or expose:

- source map;
- module/package graph references;
- AST/HIR identities;
- resolution tables;
- type tables;
- diagnostics;
- symbol index;
- file/module/package provenance;
- extension set;
- stable query handles for the lifetime of the analysis result.

CLI check, LSP compilation, doc example validation, deployment validation, and future native lowering should converge on shared stages where their input model is equivalent.
A tool may use a narrower syntax-only path only when documented and tested.

### WP-C2.4 — Position and symbol query infrastructure

Implement compiler-side queries for:

- innermost syntax/HIR node at byte position;
- symbol at position;
- definition location;
- references to symbol;
- type of expression/local/item;
- source-like type/signature rendering;
- enclosing item/module/package;
- public item enumeration;
- document/workspace symbol enumeration.

Use stable query identities within one analysis session.
Do not expose raw arena indexes as persistent external protocol IDs.

### WP-C2.5 — Diagnostics transport contract

Define and implement a single structured diagnostic form supporting:

- stable code;
- severity;
- message;
- primary file/span;
- related file/spans;
- notes;
- help;
- source version;
- optional rule/deviation identifier;
- deterministic JSON encoding.

CLI text output and LSP publication must derive from this form.

### WP-C2.6 — Differential interpreter corpus

Build a generated and hand-written corpus that stresses:

- every expression and statement form;
- every primitive operation;
- structs, enums, generics, traits, and methods;
- ownership/drop edge cases;
- `Option`/`Result` propagation;
- collections and iterators;
- multi-file and multi-package execution;
- deterministic output and failure messages.

Use metamorphic tests where a source transformation should preserve behaviour.
Examples: alpha-renaming, harmless block nesting, equivalent match ordering where non-overlapping, and explicit versus inferred generic arguments.

### WP-C2.7 — Gate exit

C2 closes when:

- the interpreter is documented as the semantic oracle;
- all known runtime semantic deviations are closed or explicitly listed;
- shared project analysis exists;
- position/symbol/type queries have compiler tests;
- structured diagnostics can be emitted consistently;
- later native and LSP work can consume these APIs without re-parsing semantics independently.

---

## GATE C3 — Native Compiler Architecture and Backend Selection Spike

*Outcome: select and de-risk the architecture that will implement mandatory MIR-to-native Core compilation.*

*This gate chooses how STARK will compile natively; it does not decide whether STARK receives a native compiler.*

### WP-C3.1 — Architecture hypothesis and workload freeze

Write `STARKLANG/docs/compiler/proposals/NATIVE-CORE-ARCHITECTURE.md`.
Freeze a representative workload set containing at least:

1. scalar arithmetic and branches;
2. loops and function calls;
3. structs and enums;
4. `Option`/`Result` and pattern matching;
5. ownership moves and deterministic drops;
6. strings/Vec operations through the existing runtime surface;
7. a multi-file, multi-package CLI application;
8. one error/trap workload.

Define measurements:

- implementation complexity;
- compile time;
- executable size;
- startup time;
- runtime performance;
- source mapping and debug/stack-trace feasibility;
- cross-platform effort;
- semantic parity risk;
- external dependency and maintenance burden;
- compatibility with the mandatory MIR and runtime ABI goals.

The semantic comparator is the existing interpreter.
The architecture comparator is not “no native compiler”; it is the strongest practical candidate implementation paths.

### WP-C3.2 — Generated Rust/C backend spike

Implement an isolated prototype for the frozen subset using generated Rust or C.
It may use a small runtime library and host toolchain.
It must not bypass type, ownership, or artifact checks already completed by the front end.

Record:

- unsupported constructs;
- source-to-generated-code traceability;
- build tool dependencies;
- cross-platform behaviour;
- semantic mismatches;
- amount of glue per language feature;
- feasibility of consuming verified MIR rather than typed HIR directly.

### WP-C3.3 — Direct backend spike

Implement the same frozen subset using the strongest plausible simple direct backend, expected to be Cranelift unless CE5 changes the candidate.
Do not implement advanced optimisation.

Record the same measurements and unsupported constructs as WP-C3.2.

### WP-C3.4 — Backend and runtime architecture selection

Compare:

```text
reference interpreter
generated Rust/C spike
direct backend spike
```

Allowed gate outcomes:

- **SELECT-GENERATED:** generated Rust/C is the initial production backend behind verified MIR.
- **SELECT-DIRECT:** the direct backend is the initial production backend behind verified MIR.
- **REVISE:** neither spike is yet sufficient, but one specific bounded follow-up can resolve the blocker. The gate stays open.
- **BLOCKED:** no credible implementation path has been demonstrated. Escalate to the owner; C4 does not open and the compiler roadmap is not complete.

The gate may reject either candidate without rejecting native compilation itself.
A selected architecture must specify:

- MIR consumption boundary;
- runtime ownership and ABI direction;
- target-platform plan;
- debug/source mapping approach;
- unsupported MVP features and closure plan;
- why the rejected candidate is not the initial production path.

CE5 owner review records the selected production backend strategy.
An interpreter-only release is not an allowed C3 completion outcome.

---

## GATE C4 — MIR Contract and Verified Lowering

*Mandatory: opens after Gate C3 selects a backend architecture.*

*Outcome: a backend-independent, validated representation of Core execution semantics.*

### WP-C4.1 — MIR design review

Under CE3, define `STARKLANG/docs/compiler/mir.md` covering:

- functions and basic blocks;
- typed locals and temporaries;
- places and projections;
- constants;
- explicit control-flow terminators;
- calls and returns;
- aggregates and discriminants;
- moves and copies;
- borrows/references as required by lowering;
- drop flags and drop operations;
- panic/trap abort paths;
- source spans and provenance;
- monomorphised versus generic representation;
- validation invariants;
- textual dump format and versioning.

Do not design optimisation IR and executable VM bytecode simultaneously.

### WP-C4.2 — Typed HIR to MIR lowering: scalar core

Lower:

- literals and locals;
- unary/binary operations;
- blocks and assignments;
- functions and calls;
- `if`, loops, break/continue, and return;
- tuples, arrays, structs, and basic enums;
- pattern matching without advanced drop elaboration.

Every MIR instruction retains a source span or documented synthetic origin.

### WP-C4.3 — MIR verifier

Implement validation for:

- well-formed block graph;
- terminator presence;
- valid local/place references;
- type consistency;
- move-before-use invariants expected at MIR level;
- valid discriminant operations;
- drop-flag consistency;
- no unsupported instruction reaching a backend silently;
- source/provenance availability.

Invalid MIR must produce a compiler-internal diagnostic and fail safely, never trigger undefined backend behaviour.

### WP-C4.4 — MIR interpreter

Implement a MIR interpreter for the supported subset.
Its purpose is differential validation, not a new user-facing VM.

For each frozen workload:

```text
HIR interpreter output/failure
== MIR interpreter output/failure
```

### WP-C4.5 — Complete Core lowering

Add:

- generics after monomorphisation decision;
- trait/static method dispatch;
- `Option`/`Result` and full enums;
- patterns;
- indexing;
- strings/Vec/runtime calls;
- ownership/drop elaboration;
- panic/trap paths;
- multi-package symbol linkage.

### WP-C4.6 — Gate exit

C4 closes only when:

- the Core execution corpus lowered in this gate runs equivalently through HIR and MIR interpreters;
- every normative Core construct required by C5 has verified MIR lowering;
- any remaining unsupported normative construct is recorded as a gate blocker rather than being carried forward silently.

If a blocker remains, C4 stays open.
No native backend work may bypass MIR validation.

---

## GATE C5 — Native Core Backend MVP

*Mandatory after C4.*

*Outcome: one normal multi-file, multi-package Core application builds into a standalone executable.*

### WP-C5.1 — Runtime ABI and layout design

Under CE4, specify:

- primitive layouts;
- tuple/array/struct/enum layout;
- strings and Vec representation;
- `Option`/`Result` representation;
- calling convention;
- symbol naming;
- runtime allocation strategy;
- drop glue;
- panic/trap ABI;
- stack-trace/debug metadata hooks;
- host platform assumptions;
- versioning between compiler and runtime.

Do not let the chosen backend's incidental representation become the language specification.

### WP-C5.2 — Scalar native lowering

Implement:

- primitive values and operations;
- functions and direct calls;
- control flow;
- stack locals;
- tuples and simple structs;
- return values;
- deterministic traps for overflow/division/index/cast cases covered by the subset.

### WP-C5.3 — Enums, matches, and error values

Add:

- enum layout and discriminants;
- payload variants;
- pattern-match control flow;
- `Option` and `Result`;
- `?`-lowered paths;
- panic/trap messages.

### WP-C5.4 — Multi-file/package linkage

Build one application with:

- at least three packages;
- cross-package generic calls where supported;
- public/private compiler checks already completed before code generation;
- deterministic symbol generation;
- one standalone executable output.

### WP-C5.5 — Debug build experience

Provide:

```bash
stark build
```

with:

- clear output path;
- source-mapped compile errors;
- runtime trap file/line where feasible;
- no need to invoke Cargo manually;
- explicit unsupported-feature diagnostics rather than backend crashes.

### WP-C5.6 — Gate exit

The frozen C3 application workload must build and run natively.
The HIR interpreter, MIR interpreter, and native output must agree for every supported case.

---

## GATE C6 — Native Semantic Parity and Cross-Platform Runtime

*Outcome: native execution preserves Core ownership, drop, failure, and library semantics rather than merely running scalar examples.*

### WP-C6.1 — Ownership and drop parity

Implement and test:

- moves versus copies;
- partial moves;
- drop flags;
- reverse declaration-order destruction;
- field and aggregate drops;
- exactly-once destruction;
- no destructor execution on aborting panic/trap if that is normative;
- no use after move in generated code even if a compiler bug slips past earlier checks.

### WP-C6.2 — Generics and trait dispatch

Choose and document monomorphisation or another approved strategy.
Implement:

- generic functions and types;
- trait-bound calls;
- associated types as already resolved by the front end;
- inherent versus trait methods;
- operator semantics;
- duplicate-instantiation control and deterministic symbol names.

Trait objects remain out of scope.

### WP-C6.3 — Core runtime values and collections

Support the normative runtime surface needed by Core programs:

- String/str operations;
- Vec and slices;
- Box;
- HashMap/HashSet only if their deterministic and hashing contracts are approved for the native runtime;
- iterators represented without violating borrow semantics;
- formatting, printing, assertions, panic, and whole-file operations currently in the runtime surface.

The ecosystem may later replace compiler builtins with packages; preserve behaviour during migration.

### WP-C6.4 — Platform matrix

Target at minimum:

```text
linux-x64
macos-arm64
macos-x64
windows-x64
```

CI or owner-supplied machines must verify the supported matrix.
Unsupported targets fail before linking with a clear diagnostic.

### WP-C6.5 — Full differential suite

Run every executable Core conformance case through:

```text
HIR interpreter
MIR interpreter
native debug build
native release build once C7 exists
```

Compare:

- stdout/stderr;
- exit status;
- returned values where harnessed;
- trap category and source location;
- observable drop order in instrumented fixtures.

### WP-C6.6 — Gate exit

C6 closes only when native execution is semantically credible, not merely faster.
Every unsupported normative Core feature must block a full native-conformance claim.

---

## GATE C7 — Build Profiles, Reproducibility, and Evidence-Gated Optimisation

*Outcome: a usable compiler build path with measured performance and no correctness regression.*

### WP-C7.1 — Build profiles and target selection

Implement and document:

```bash
stark build
stark build --release
stark build --target <triple>
```

Debug and release profiles must not change trap, overflow, panic, ownership, or drop semantics.

### WP-C7.2 — Reproducible native outputs

Define which output bytes are expected to be reproducible and which platform metadata is excluded.
Test identical source, lockfile, compiler version, target, and flags across two clean builds.

Record:

- compiler version;
- runtime version;
- backend version;
- target;
- build flags;
- dependency graph identity.

### WP-C7.3 — Build cache and incremental boundaries

Only after profiling, implement the smallest useful cache.
Potential cache keys:

```text
source content
compiler version
language options
package graph identity
MIR version
backend version
target/profile
```

Do not build a full incremental query engine unless measurements show clean-build cost is a real blocker.

### WP-C7.4 — Baseline optimisations

Permitted initial optimisations:

- constant folding consistent with trap semantics;
- dead-block elimination;
- trivial copy propagation;
- unreachable code removal after semantic diagnostics;
- backend-native optimisation settings.

Every optimisation needs differential tests against unoptimised MIR/native execution.

### WP-C7.5 — Performance and complexity report

Measure frozen workloads for:

- compile time;
- peak compiler memory;
- executable size;
- startup time;
- steady-state runtime;
- interpreter/native ratio;
- debug/release ratio;
- backend maintenance complexity.

Do not claim a general performance multiple from a small workload.

### WP-C7.6 — LLVM decision, normally DEFER

Open CE6 only if measured workloads show a material limitation in the selected backend.
The proposal must quantify:

- missing optimisation or target capability;
- expected benefit;
- integration and build complexity;
- binary/toolchain burden;
- contributor impact;
- alternative improvements within the current backend.

Default outcome is DEFER.

### WP-C7.7 — Gate exit

C7 closes when native builds are usable, reproducible to the documented degree, and performance claims are bounded by measured evidence.

---

## GATE C8 — Semantic Language Services

*May begin after C2; does not depend on native compilation.*

*Outcome: real compiler-backed editor semantics rather than protocol wiring and placeholder responses.*

### WP-C8.1 — Diagnostic publication

Use the C2 structured diagnostic contract to implement:

- LSP `publishDiagnostics` on open/change/save;
- document-version tracking;
- stale-result suppression;
- clearing diagnostics on close;
- related information across files;
- extension-aware analysis;
- package-aware project roots.

The existing subprocess checker may remain temporarily, but duplicate diagnostics must be prevented or clearly separated during migration.

### WP-C8.2 — Hover and signature rendering

Implement real hover for:

- locals and parameters;
- functions and methods;
- structs/enums/traits/types;
- generic substitutions;
- tensor/model constructs when the extension is enabled;
- inferred expression types where useful.

A cursor-coordinate string is a stub, not hover support.

### WP-C8.3 — Definition and references

Implement:

- same-file and cross-file definitions;
- module and package definitions;
- imported and re-exported symbols;
- method and trait definitions;
- external package source locations when available;
- exact references based on resolved symbol identity, not text search.

### WP-C8.4 — Completion and signature help

Add context-aware completion for:

- local scope;
- module paths;
- fields and methods;
- imported package APIs;
- enum variants;
- function arguments and generic parameters;
- extension symbols only when enabled.

### WP-C8.5 — Rename and symbols

Implement:

- safe rename using resolved identity;
- collision detection before edits;
- document symbols;
- workspace symbols;
- deterministic edit ordering.

### WP-C8.6 — Semantic tokens and inlay information

Only after the earlier query APIs are stable, add:

- semantic token classification;
- optional inferred-type hints;
- parameter hints where unambiguous.

Do not duplicate parsing or type checking in the editor client.

### WP-C8.7 — Protocol and editor validation

Add raw JSON-RPC tests plus at least one real Extension Development Host or packaged-extension validation session on a VS Code-capable environment.
Protocol tests alone do not prove UI behaviour.

### WP-C8.8 — Gate exit

C8 closes when advertised hover, definition, references, diagnostics, formatting, and symbols reflect real compiler semantics and are interactively validated.

---

## GATE C9 — Extension Isolation and Conditional Artifact-Provider Generalisation

*Part A may begin after C2. Part B is blocked until a second independent artifact implementation exists, expected from the ecosystem OpenAPI contract-only experiment.*

### WP-C9.1 — Extension-isolation conformance

Build a matrix proving:

- Core-only default across every entry point;
- tensor syntax rejected when disabled;
- tensor syntax accepted when enabled;
- tensor builtins unavailable in Core-only unqualified and qualified resolution;
- formatter, doc generator, tests, LSP, check, run, deploy, and verifier use the intended extension set;
- unknown and duplicate extension configuration produces consistent behaviour;
- one extension cannot mutate global compiler state for another session.

### WP-C9.2 — Tensor provider map

Document the current tensor/ONNX integration as stages:

```text
parse extension syntax
resolve extension symbols
type-check tensor/model semantics
read ONNX artifact
extract signature
record identity/provenance
generate declaration
verify declaration/artifact drift
lower supported deployment pipeline
contribute diagnostics
```

Identify which stages are tensor-specific and which appear reusable.
Do not refactor yet.

### WP-C9.3 — Wait for second-artifact evidence

This WP cannot close until the ecosystem's early OpenAPI or schema experiment has a working implementation and gate report.
Use that implementation—not a hypothetical API—to compare:

- artifact reading;
- normalisation;
- identity hashing;
- typed contract extraction;
- generated/registered declarations;
- drift comparison;
- provenance;
- diagnostic reporting;
- compiler phase integration.

### WP-C9.4 — Internal artifact-provider contract

Under CE7, define the smallest compiler-internal contract shared by the two real implementations.
Likely responsibilities:

```text
read artifact
normalise supported content
extract typed contract
record identity and provenance
register or generate declarations
verify later artifact versions
emit structured diagnostics
report backend/runtime obligations where applicable
```

Explicit non-goals:

- no public `Artifact<...>` Core type;
- no capability/effect/information-flow system;
- no provider-specific cloud syntax;
- no arbitrary compiler plugins;
- no artifact code execution during compilation.

### WP-C9.5 — Migrate both implementations

Migrate ONNX and the second artifact implementation without changing their user-visible contract unless an approved compatibility decision says otherwise.

Measure:

- duplicated code removed;
- domain-specific code retained;
- diagnostics preserved or improved;
- new complexity introduced;
- whether the abstraction actually simplifies a third mock provider fixture.

### WP-C9.6 — Gate exit

C9 may conclude:

```text
GENERALISE — common provider infrastructure is justified
KEEP-SEPARATE — implementations share too little or abstraction costs more
REVISE — one bounded follow-up is needed
```

User-facing artifact syntax remains deferred regardless of a GENERALISE result.

---

## CONDITIONAL TRACK T — Tensor Execution and Verifier Maintenance

*This is not an automatic gate sequence.*

### T0 — Maintenance authority

Always allowed:

- fix soundness or correctness bugs;
- preserve ONNX import/verify/deploy behaviour;
- maintain deterministic output and diagnostics;
- keep compatibility with supported ONNX Runtime versions;
- run the frozen Gate 7 defect corpus.

### T1 — External verifier validation

This is a product-demand experiment governed by its own proposal.
It must remain independent of native Core compilation and new tensor features.
Do not rewrite its decision based on internal enthusiasm.

### T2 — Additional tensor lowering

Opens only with an approved workload showing that a currently checked-but-not-executable operation blocks a real experiment or external user.
Implement the smallest operation set required by that workload.

Every backend must declare:

```text
frontend supported
backend executable
backend unsupported with compile/deploy diagnostic
```

A checked operation must never silently reach an incapable backend.

### T3 — Full tensor v0.1 execution

Not authorised by default.
Requires:

- external or research workloads needing broad operation coverage;
- a backend strategy;
- comparison with existing tensor runtimes;
- maintenance-cost analysis;
- a decision on whether STARK is becoming a tensor language rather than an artifact-verification research language.

---

## GATE C10 — Compiler Release Qualification

*Outcome: a precise, evidence-backed compiler release statement with mandatory native Core compilation; optional extension tracks may remain scoped or deferred.*

### WP-C10.1 — Full conformance dashboard

Generate a dashboard covering:

- Core lexical/syntax rules;
- type and semantic rules;
- memory/ownership rules;
- interpreter execution rules;
- native execution rules from the mandatory C3–C7 path;
- extension isolation;
- tensor extension rules and backend capabilities;
- listed deviations;
- last verified commit and toolchain.

### WP-C10.2 — Robustness and fuzzing

Add or expand:

- lexer/parser fuzzing;
- malformed source corpus;
- resolver graph fuzzing;
- type-checker and borrow-checker generated cases;
- MIR verifier fuzzing;
- malformed artifact corpus;
- malformed diagnostic/protocol inputs;
- timeout and memory limits for hostile inputs.

The gate is no panic, no hang, bounded failure, and deterministic diagnostics—not proof that random programs are semantically meaningful.

### WP-C10.3 — Differential and metamorphic testing

Run:

- HIR versus MIR interpreter;
- interpreter versus native backend;
- debug versus release;
- repeated clean builds;
- equivalent source transformations;
- tensor deploy output versus frozen reference workloads.

### WP-C10.4 — Security review

Review compiler-controlled boundaries:

- source/module path traversal;
- package and cache file access relevant to compiler invocation;
- artifact parsing limits;
- generated source and shell/process invocation;
- native linker arguments;
- temporary directory handling;
- archive extraction where applicable;
- LSP workspace trust and executable paths;
- denial-of-service inputs;
- dependency vulnerabilities and licences.

### WP-C10.5 — Compatibility and version policy

Define:

- language version versus compiler version;
- Core and extension compatibility;
- diagnostic stability expectations;
- MIR/runtime ABI versioning if applicable;
- generated artifact compatibility;
- deprecation policy;
- minimum supported Rust and platform versions;
- release artefact signing/checksum policy under owner review.

### WP-C10.6 — Performance baselines

Record bounded baselines for:

- lex/parse/check time;
- peak compiler memory;
- large-module and multi-package scaling;
- LSP change-to-diagnostic latency;
- native build time and runtime where applicable;
- ONNX import/verify/deploy time;
- binary sizes.

Performance regression thresholds may be added only after stable baselines exist.

### WP-C10.7 — Release decision

A compiler-track completion release requires C0–C8 and the mandatory native path C3–C7 to be closed.
A build without the native Core backend may be published only as an internal snapshot, research preview, or explicitly incomplete pre-release; it is not the end state of this roadmap.

The release statement must choose one precise form, for example:

```text
STARK Core v1 front end, interpreter, MIR, and native backend: conforming for the listed platform matrix
General native Core backend: production MVP, listed deviations X/Y
Tensor extension v0.1 frontend and verifier: conforming for listed scope
Tensor backend execution: capability-limited; see matrix
```

or:

```text
STARK Core v1: conforming with deviations DEV-...
```

Never publish:

```text
STARK Core v1: conforming
Known deviations: none
```

unless CE8 review confirms the evidence supports it.

---

# 4. DEPENDENCY AND SEQUENCING MAP

## 4.1 Mandatory compiler completion path

```text
C0 Current-state truth
 -> C1 Core conformance closure
 -> C2 Reference execution + compiler services foundation
 -> C3 Native architecture and backend selection
 -> C4 Verified MIR
 -> C5 Native MVP
 -> C6 Native semantic parity
 -> C7 Build profiles, reproducibility, and baseline optimisation
```

This entire path is mandatory.
C3 may reject a backend candidate or require a bounded revision, but it may not close by choosing an interpreter-only end state.
If C3 is BLOCKED, later native gates remain closed and the compiler roadmap remains incomplete.

## 4.2 Language-service path

```text
C2
 -> C8 Semantic language services
```

This path can run in parallel with the mandatory native compiler path after C2.

## 4.3 Artifact-infrastructure path

```text
C2 + C9.1/C9.2
 + ecosystem second-artifact gate result
 -> C9.3
 -> C9.4/C9.5
```

Do not create generic artifact infrastructure before the second implementation exists.

## 4.4 Release path

A compiler-track release qualification gate may open when:

```text
C0–C8 are closed
+ C9 status is explicit (done, blocked on second-artifact evidence, or not required for this release)
+ tensor capability/deviation status is explicit
```

A release does not require every optional artifact or tensor-expansion track to be complete.
It does require the general native Core compiler path to be complete and its deviations to be stated precisely.

---

# 5. DIAGNOSTIC AND EVIDENCE REGISTRIES

## 5.1 Diagnostic governance

Use existing normative `E####` and `W####` codes where defined.
Do not allocate a new language diagnostic code merely to describe an internal compiler bug.

Maintain separate namespaces for:

```text
ICE-xxxx  internal compiler invariant failures
MIR-xxxx  invalid MIR / lowering failures
CG-xxxx   native code-generation/backend capability failures
LSP-xxxx  protocol/configuration failures not caused by user source
ART-xxxx  generic artifact-provider failures, only after C9
```

Every new namespace and first allocation must be recorded in `COMPILER-STATE.md`.
User-source diagnostics should continue using normative language/extension codes where applicable.

## 5.2 Evidence classes

Each gate report should identify evidence as:

```text
SPEC       normative rule or approved proposal
UNIT       focused unit test
CONF       conformance fixture
NEG        intentional invalid-program test
REG        regression test for a discovered bug
PROP       property/metamorphic test
FUZZ       robustness test
DIFF       interpreter/MIR/native comparison
PERF       measured performance result
EXT        external-user evidence
MANUAL     explicitly disclosed manual verification
```

Do not describe MANUAL evidence as automated coverage.
Do not describe a protocol-level test as real editor UI validation.

## 5.3 Gate decision vocabulary

Use only:

```text
PASS
PASS-WITH-DEVIATIONS
REVISE
BLOCKED
DEFER
STOP
FAIL
```

“Mostly complete” is not a gate decision.
A work package may be `partial`, but a gate report must state what that means for the next gate.
For mandatory gates C3–C7, `DEFER` is not a completion outcome; `BLOCKED` or `REVISE` keeps the roadmap open, and `STOP` is a project-level owner decision rather than an interpreter-only success state.

---

# 6. NOT-YET LIST — REFUSE AND CITE THE CHARTER

Do not start any of the following without a separate approved proposal:

- new Core language features;
- async/concurrency;
- closures/macros/unsafe/raw pointers/trait objects;
- a custom production VM;
- LLVM adoption without C7 evidence;
- self-hosting;
- a JIT;
- GPU kernel generation or a custom tensor runtime;
- broad robotics, cloud, or distributed-system compiler semantics;
- public compiler plugin APIs;
- public `Artifact<...>` syntax;
- capabilities, effects, typestate, or information-flow labels;
- automatic differentiation or training;
- unrestricted native FFI;
- a full incremental compiler before profiling;
- speculative optimisations without differential tests;
- claiming external adoption from AI review or internal demos.

---

# 7. SESSION BUDGET — PLANNING AID, NOT DEADLINE

Approximate Sonnet work sessions after bootstrap:

```text
C0  3–5
C1  7–10
C2  6–9
C3  3–5
C4  7–10
C5  7–10
C6  8–12
C7  5–8
C8  6–9
C9  4–7 after second-artifact evidence (optional)
C10 5–8
```

Mandatory compiler completion path, excluding optional C9:

```text
approximately 57–86 sessions
```

These are sequencing aids only.
A gate opens on evidence, not elapsed effort.

---

# 8. STRATEGIC OUTCOME — KEEP THIS IN VIEW

The compiler track is not successful merely because it accumulates features or lines of Rust.
Its mandatory outcomes are:

1. **A trustworthy Core implementation:** every accepted and rejected program is tied to normative rules and executable evidence.
2. **A reliable semantic oracle:** the interpreter and compiler query model give later tools and backends one consistent meaning of STARK programs.
3. **A verified intermediate representation:** typed HIR lowers to validated MIR with explicit control flow, moves, drops, calls, aggregates, and trap paths.
4. **A general native Core compiler:** ordinary multi-file, multi-package STARK programs build into standalone executables without invoking Cargo manually.
5. **Semantic parity across execution paths:** HIR interpreter, MIR interpreter, native debug, and native release builds agree on output, traps, ownership, and drop behaviour.
6. **Real compiler-backed tooling:** diagnostics, hover, navigation, references, and completion come from semantic identity rather than protocol stubs or text matching.
7. **Disciplined extension architecture:** tensor remains isolated, and generic artifact infrastructure appears only after two real artifact implementations justify it.
8. **Precise release claims:** the project states exactly what conforms, which targets are supported, what is experimental, and what remains deferred.

The required completed product shape is:

```text
STARK source
 -> Core and extension verification
 -> typed HIR
 -> verified MIR
 -> selected native backend + runtime
 -> standalone executable
```

Evidence still controls important choices:

- which backend is selected;
- how the runtime ABI is structured;
- which targets ship first;
- whether LLVM is ever added;
- how far optimisation and incremental compilation proceed;
- whether optional tensor execution and generic artifact infrastructure expand.

Evidence does **not** decide whether the roadmap may finish without native compilation.
An interpreter-only implementation can remain a useful research snapshot, but it is not completion of this compiler brief.
If a credible native architecture cannot be delivered, the roadmap is BLOCKED or the owner stops the compiler project explicitly.
It does not redefine “compiler complete” downward.
