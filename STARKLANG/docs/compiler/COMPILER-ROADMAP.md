# STARK Compiler Roadmap

Extracted from `STARKLANG/docs/STARK-Compiler-Build-Brief-Revised-Sonnet.md` (source of truth
for meaning). Gates and work packages, dependencies, exit criteria, and conditional tracks.
Standing rules, escalation IDs, and the not-yet list live in `COMPILER-CHARTER.md`. Current
position and evidence live in `COMPILER-STATE.md` (repo root).

Load this full file only when closing a gate, selecting a conditional track, resolving a
roadmap-level contradiction, or reviewing a proposed change to Core semantics. Routine sessions
should load `COMPILER-CHARTER.md` + `COMPILER-STATE.md` + the active WP file only.

> **Relationship to the pre-existing (non-"C") gate track.** This repository already closed a
> different, unrelated gate sequence — `starkc/docs/gate1-exit.md` through `gate7-decision.md`
> — covering lexer/parser (old Gate 1), semantic checker (old Gate 2), interpreter (old Gate 3),
> tensor/ONNX front end (old Gate 4), native ONNX-Runtime deployment demonstrator (old Gate 5),
> and two decision checkpoints (old Gate 6: REVISE; old Gate 7: RETAIN AS RESEARCH LANGUAGE,
> authorizing only a `stark verify` validation track). See `COMPILER-STATE.md` CD-002. The gates
> below (`C0`-`C10`) are a **new, independent numbering** introduced by this roadmap; they are
> not a renumbering of the old track and do not re-open old Gate 1-5 work from zero. Gate C0's
> job is precisely to establish how the old track's evidence maps onto the new gates.

---

# 3. Gates and work packages

Gates close in order within each dependency chain.
Parallel tracks are allowed only where explicitly shown in §4.
A gate status is determined by evidence, not session count.

---

## GATE C0 — Bootstrap, Current-State Audit, and Authority Repair

*Outcome: one trusted account of what the compiler implements, what the specifications require,
and which documents or status claims are stale.*

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
Record current head, toolchain, test commands, active normative specs, and all existing
roadmap/plan/status documents in `COMPILER-STATE.md`.

**Done when:** a later session can operate using only the charter, state, and one active WP.

### WP-C0.1 — Repository and pipeline map

Deliver `starkc/docs/dev/compiler-map.md` containing:

- every active compiler module and purpose;
- input/output representation for lexer, parser, resolver, type checker, borrow checker,
  interpreter, ONNX tooling, formatter, docs, LSP, and package CLI;
- shared and duplicated compiler entry points;
- global state, filesystem access, process execution, and nondeterministic collections;
- source-file and module provenance flow;
- test files that cover each subsystem;
- stubs and deliberately incomplete handlers;
- archived code that must not be modified.

The map must identify where the same source is parsed or analysed through different pipelines.

**Done when:** every active compiler/tool entry point can be traced to the normative stage and
tests that support it.

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
Do not rewrite historical gate evidence to match later implementation; add dated updates
instead.

**Done when:** no active document says a completed gate is "next," no README advertises an LSP
stub as implemented semantic behaviour, and status summaries agree with `COMPILER-STATE.md`.

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

Create `starkc/docs/conformance/KNOWN-DEVIATIONS.md` and record every known issue found by
audit.
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
- explicit statement that no conformance percentage is trusted unless generated from the
  repaired database.

**C0 closes only when the project's current state can be stated without relying on
commit-message optimism or stale roadmap text.**

---

## GATE C1 — Core v1 Conformance Closure

*Outcome: an evidence-backed Core v1 implementation statement for the front end and semantic
checker, with every remaining deviation explicit.*

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

Add deterministic property/fuzz tests for no panic, no hang, bounded resource use, and stable
diagnostics on malformed input.

**Done when:** every lexical/syntax rule has a coverage entry and executable evidence; no known
syntax deviation remains unrecorded.

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

A hidden interpreter-only structural equality rule is not accepted as an undocumented third
behaviour.

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

The report must be generated from machine-readable data, deterministic, and checked into CI as
an artefact or summary.

### WP-C1.7 — Gate exit

Write `starkc/docs/compiler/C1-exit-report.md`.
The conclusion must be one of:

```text
CORE-FRONTEND-CONFORMING
CORE-FRONTEND-CONFORMING-WITH-LISTED-DEVIATIONS
CORE-FRONTEND-NOT-YET-CONFORMING
```

Do not use "complete" without the rule-level report.
Native compilation is not required for this gate.
Interpreter runtime parity is completed in Gate C2.

---

## GATE C2 — Reference Execution Semantics and Compiler Service Foundation

*Outcome: one trustworthy reference execution model and one shared project-analysis API used by
compiler-facing tools.*

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

Every rule must cite the normative Core specification rather than inventing
interpreter-specific semantics.

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

No new public standard-library API belongs here unless required by the normative spec and
approved under the ecosystem compatibility process.

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

CLI check, LSP compilation, doc example validation, deployment validation, and future native
lowering should converge on shared stages where their input model is equivalent.
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
Examples: alpha-renaming, harmless block nesting, equivalent match ordering where
non-overlapping, and explicit versus inferred generic arguments.

### WP-C2.7 — Gate exit

C2 closes when:

- the interpreter is documented as the semantic oracle;
- all known runtime semantic deviations are closed or explicitly listed;
- shared project analysis exists;
- position/symbol/type queries have compiler tests;
- structured diagnostics can be emitted consistently;
- later native and LSP work can consume these APIs without re-parsing semantics
  independently.

---

## GATE C3 — Compiled-Language Decision Spike

*Outcome: an evidence-backed GO, REVISE, DEFER, or STOP decision on general native Core
compilation before committing to MIR and a production backend.*

*This gate is intentionally small relative to a real backend.*

### WP-C3.1 — Decision hypothesis and workload freeze

Write `STARKLANG/docs/compiler/proposals/NATIVE-CORE-DECISION.md`.
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
- debug/stack-trace feasibility;
- cross-platform effort;
- semantic parity risk;
- external dependency and maintenance burden.

The comparator is the existing interpreter plus any existing generated-host deployment path, not
"no execution."

### WP-C3.2 — Generated Rust/C bootstrap spike

Implement a throwaway or isolated prototype for the frozen subset using generated Rust or C.
It may use a small runtime library.
It must not become the production architecture by default.

Record:

- unsupported constructs;
- source-to-generated-code traceability;
- build tool dependency;
- cross-platform behaviour;
- semantic mismatches;
- amount of glue per language feature.

### WP-C3.3 — Direct backend spike

Implement the same frozen subset using the strongest plausible simple direct backend, expected
to be Cranelift unless CE5 changes the candidate.
Do not implement advanced optimisation.

Record the same measurements and unsupported constructs.

### WP-C3.4 — Architecture comparison and decision

Compare:

```text
reference interpreter
generated Rust/C spike
direct backend spike
```

Decision rules:

- **GO:** one path shows a credible route to standalone Core programs with manageable semantic
  and maintenance cost.
- **REVISE:** value exists, but the workload or architecture needs one bounded follow-up spike.
- **DEFER:** native compilation is technically viable but not currently worth the effort relative
  to research priorities.
- **STOP:** the full-language native path provides insufficient value or unacceptable
  complexity; retain interpreter plus specialised deployment backends.

A GO decision must select the production backend strategy under CE5 owner review.
A DEFER or STOP decision is not failure and blocks Gates C4-C7 until explicitly revisited.

---

## GATE C4 — MIR Contract and Verified Lowering

*Conditional: opens only after Gate C3 records GO.*

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

Invalid MIR must produce a compiler-internal diagnostic and fail safely, never trigger undefined
backend behaviour.

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

C4 closes when the full Core conformance execution corpus either:

- runs equivalently through HIR and MIR interpreters; or
- lists bounded unsupported features that block opening C5.

No native backend work may bypass MIR validation.

---

## GATE C5 — Native Core Backend MVP

*Conditional on C4.*

*Outcome: one normal multi-file, multi-package Core application builds into a standalone
executable.*

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

*Outcome: native execution preserves Core ownership, drop, failure, and library semantics
rather than merely running scalar examples.*

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
- HashMap/HashSet only if their deterministic and hashing contracts are approved for the native
  runtime;
- iterators represented without violating borrow semantics;
- formatting, printing, assertions, panic, and whole-file operations currently in the runtime
  surface.

The ecosystem may later replace compiler builtins with packages; preserve behaviour during
migration.

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

*Outcome: a usable compiler build path with measured performance and no correctness
regression.*

### WP-C7.1 — Build profiles and target selection

Implement and document:

```bash
stark build
stark build --release
stark build --target <triple>
```

Debug and release profiles must not change trap, overflow, panic, ownership, or drop semantics.

### WP-C7.2 — Reproducible native outputs

Define which output bytes are expected to be reproducible and which platform metadata is
excluded.
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

Do not build a full incremental query engine unless measurements show clean-build cost is a real
blocker.

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

C7 closes when native builds are usable, reproducible to the documented degree, and performance
claims are bounded by measured evidence.

---

## GATE C8 — Semantic Language Services

*May begin after C2; does not depend on native compilation.*

*Outcome: real compiler-backed editor semantics rather than protocol wiring and placeholder
responses.*

### WP-C8.1 — Diagnostic publication

Use the C2 structured diagnostic contract to implement:

- LSP `publishDiagnostics` on open/change/save;
- document-version tracking;
- stale-result suppression;
- clearing diagnostics on close;
- related information across files;
- extension-aware analysis;
- package-aware project roots.

The existing subprocess checker may remain temporarily, but duplicate diagnostics must be
prevented or clearly separated during migration.

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

Add raw JSON-RPC tests plus at least one real Extension Development Host or packaged-extension
validation session on a VS Code-capable environment.
Protocol tests alone do not prove UI behaviour.

### WP-C8.8 — Gate exit

C8 closes when advertised hover, definition, references, diagnostics, formatting, and symbols
reflect real compiler semantics and are interactively validated.

---

## GATE C9 — Extension Isolation and Conditional Artifact-Provider Generalisation

*Part A may begin after C2. Part B is blocked until a second independent artifact
implementation exists, expected from the ecosystem OpenAPI contract-only experiment.*

### WP-C9.1 — Extension-isolation conformance

Build a matrix proving:

- Core-only default across every entry point;
- tensor syntax rejected when disabled;
- tensor syntax accepted when enabled;
- tensor builtins unavailable in Core-only unqualified and qualified resolution;
- formatter, doc generator, tests, LSP, check, run, deploy, and verifier use the intended
  extension set;
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

This WP cannot close until the ecosystem's early OpenAPI or schema experiment has a working
implementation and gate report.
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

Under CE7, define the smallest compiler-internal contract shared by the two real
implementations.
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

Migrate ONNX and the second artifact implementation without changing their user-visible
contract unless an approved compatibility decision says otherwise.

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

Opens only with an approved workload showing that a currently checked-but-not-executable
operation blocks a real experiment or external user.
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
- a decision on whether STARK is becoming a tensor language rather than an
  artifact-verification research language.

---

## GATE C10 — Compiler Release Qualification

*Outcome: a precise, evidence-backed release statement—not necessarily a claim that every
optional track is complete.*

### WP-C10.1 — Full conformance dashboard

Generate a dashboard covering:

- Core lexical/syntax rules;
- type and semantic rules;
- memory/ownership rules;
- interpreter execution rules;
- native execution rules if C3-C7 completed;
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
- MIR verifier fuzzing if MIR exists;
- malformed artifact corpus;
- malformed diagnostic/protocol inputs;
- timeout and memory limits for hostile inputs.

The gate is no panic, no hang, bounded failure, and deterministic diagnostics—not proof that
random programs are semantically meaningful.

### WP-C10.3 — Differential and metamorphic testing

Run:

- HIR versus MIR interpreter if MIR exists;
- interpreter versus native backend if native exists;
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

The release statement must choose one precise form, for example:

```text
STARK Core v1 front end and interpreter: conforming
General native Core backend: experimental, listed deviations X/Y
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

# 4. Dependency and sequencing map

## 4.1 Mandatory correctness path

```text
C0 Current-state truth
 -> C1 Core conformance closure
 -> C2 Reference execution + compiler services foundation
```

This path is mandatory regardless of whether native compilation proceeds.

## 4.2 Native compiler path

```text
C2
 -> C3 Compiled-language decision spike
 -> [GO only] C4 MIR
 -> C5 Native MVP
 -> C6 Native parity
 -> C7 Build profiles and optimisation
```

A DEFER or STOP at C3 freezes C4-C7 without blocking C8, C9, tensor maintenance, or an
interpreter-based research release.

## 4.3 Language-service path

```text
C2
 -> C8 Semantic language services
```

This path can run in parallel with the native track after C2.

## 4.4 Artifact-infrastructure path

```text
C2 + C9.1/C9.2
 + ecosystem second-artifact gate result
 -> C9.3
 -> C9.4/C9.5
```

Do not create generic artifact infrastructure before the second implementation exists.

## 4.5 Release path

A release qualification gate may open when:

```text
C1 and C2 are closed
+ C8 status is explicit
+ C9 status is explicit
+ native track is either completed or formally DEFERRED/STOPPED
+ tensor capability/deviation status is explicit
```

A release does not require every conditional track to be GO.
It requires precise claims.

---

# 7. Session budget — planning aid, not deadline

Approximate Sonnet work sessions after bootstrap:

```text
C0  3-5
C1  7-10
C2  6-9
C3  3-5
C4  7-10   conditional
C5  7-10   conditional
C6  8-12   conditional
C7  5-8    conditional
C8  6-9
C9  4-7 after second-artifact evidence
C10 5-8
```

Interpreter/research release path without native GO:

```text
approximately 31-48 sessions
```

Full native path:

```text
approximately 58-88 sessions
```

These are sequencing aids only.
A gate opens on evidence, not elapsed effort.
