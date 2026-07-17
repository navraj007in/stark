# STARK Compiler Charter

Extracted from `STARKLANG/docs/STARK-Compiler-Build-Brief-Revised-Sonnet.md` (source of truth
for meaning; this file must not diverge from it without a recorded decision in
`COMPILER-STATE.md`). This is the durable governance surface: source-of-truth rules, track
boundaries, standing constraints, escalation rules, and the not-yet list.

For the intended session input packet and gate/work-package structure, see
`COMPILER-ROADMAP.md`. For current position, decisions, deviations, and evidence, see
`COMPILER-STATE.md` (repo root).

---

## 1. Project context — fixed, do not re-derive

**STARK** is an ownership-safe, Rust-inspired research programming language with a normative
Core v1 specification and an optional tensor/model extension.
The repository already contains a Rust lexer, parser, resolver, typed HIR, type checker, borrow
checker, interpreter, multi-file/module support, package graph support, tensor/ONNX import and
verification, bounded native ONNX deployment, a formatter, test runner, documentation generator,
LSP foundation, and editor integration.

The latest known repository head when this master brief was prepared was:

```text
6fa8c15b94bd1376a847132498d31dd356524180
```

Treat that as a **starting reference, not a permanent truth**.
Gate C0 must read the current head and establish what is implemented, partial, stale, stubbed,
inconsistent, or only claimed in documentation.
Do not rebuild a subsystem merely because an older roadmap says it is missing.
Do not mark a subsystem complete merely because a commit message says it is complete.

### 1.1 Compiler-track responsibility

The compiler track owns:

- lexical and syntactic conformance;
- AST, HIR, source maps, symbol identity, and compiler query infrastructure;
- name resolution, type checking, trait checking, coherence, ownership, borrowing, drop
  analysis, exhaustiveness, and other Core semantics;
- the reference interpreter and executable semantics;
- backend-independent IR, if evidence authorises it;
- native Core compilation, if the compiled-language decision gate authorises it;
- compiler diagnostics and machine-readable diagnostic protocols;
- compiler-side semantic services required by LSP clients;
- extension isolation and compiler-side extension-provider infrastructure;
- conformance, fuzzing, differential testing, compatibility, and release qualification.

The compiler track does **not** own:

- package ecosystem product design, public registry infrastructure, package-author workflows,
  or third-party package policy;
- standard network/TLS/HTTP package design;
- cloud-provider syntax or infrastructure-as-code features;
- the OpenAPI product experiment itself;
- VS Code UI behaviour beyond the compiler/LSP protocol it consumes;
- new language features merely because a tool or package would find them convenient.

The ecosystem and compiler tracks may depend on one another, but neither may silently absorb the
other's scope.

### 1.2 North-star technical thesis

> **STARK investigates artifact-bound programming: application code and the external artifacts
> it depends on are verified as one typed system rather than through disconnected tools.**

The existing ONNX work is one artifact implementation.
The compiler track must preserve that implementation, but it must not generalise it prematurely
into public Core syntax.
A generic artifact-provider abstraction is considered only after a second independent artifact
implementation exists and exposes materially shared compiler needs.

### 1.3 Keep four questions separate

Never allow one successful experiment to answer a different question by implication:

1. **Core correctness:** Does the compiler implement the normative Core v1 language accurately
   and consistently?
2. **Compiled-language value:** Is a general native backend worth its complexity relative to the
   interpreter and generated-host deployment paths?
3. **Artifact-binding generality:** Does the source/artifact verification mechanism generalise
   beyond ONNX?
4. **AI-development methodology:** Does AI materially improve language-development velocity or
   quality under controlled governance?

The compiler roadmap primarily tests questions 1 and 2.
Question 3 requires the ecosystem's second-artifact experiment.
Question 4 requires its own development-process evidence and is not a substitute for semantic
correctness.

### 1.4 Existing evidence and strategic constraint

The tensor track has already produced a bounded positive result in a computer-vision deployment
workload.
It demonstrated useful symbolic-shape and artifact-drift guarantees against a strong typed-Rust
comparator.
The recorded project decision is to retain STARK as a research language while external demand
remains unproven.

Therefore:

- maintain the working tensor/ONNX path;
- do not use it to justify unrestricted language expansion;
- keep `stark verify` external-user validation independent of compiler expansion;
- do not build a full tensor runtime, VM, GPU compiler, or broad domain system without a new
  bounded proposal and evidence gate.

> **C0 note (see COMPILER-STATE.md CD-002):** the "existing evidence and strategic constraint"
> above is not hypothetical — it is the old-numbering Gate 6/Gate 7 track
> (`starkc/docs/gate6-memo.md`, `starkc/docs/gate7-decision.md`), already closed with verdicts
> REVISE and RETAIN AS RESEARCH LANGUAGE respectively. Gate C3 (compiled-language decision spike)
> must treat that prior decision as directly relevant precedent, not reopen it from zero.

### 1.5 Governing rules — violating any is a wrong implementation

1. **The normative specification defines language behaviour.** The compiler conforms to the
   spec; the spec is not rewritten merely to excuse an implementation shortcut.
2. **Spec defects are real and may be fixed.** A spec fix must include the normative source
   change, regenerated combined documents, fixture re-triage where relevant, compiler change,
   and executable evidence in one bounded change.
3. **Implementation repair is not language design.** If the compiler disagrees with the spec,
   first determine whether the compiler or specification is wrong; do not invent a third
   behaviour.
4. **No new Core syntax or semantics inside an implementation WP.** New features require a
   separate proposal, concrete blocked use case, alternatives, compatibility analysis, and owner
   approval.
5. **Core remains extension-neutral.** Tensor/model rules must not leak into Core-only parsing,
   resolution, typing, diagnostics, formatting, tests, or LSP sessions.
6. **The interpreter is the semantic reference until a later gate explicitly changes that
   decision.** Native execution must match it or expose a documented spec defect; native
   behaviour does not silently redefine the language.
7. **Correctness precedes optimisation.** No optimisation pass is accepted until the unoptimised
   path passes the differential semantics suite.
8. **Stable Rust only.** Nightly compiler features are prohibited unless separately approved with
   a portability and contributor-cost analysis.
9. **No backend lock-in before evidence.** Cranelift, generated Rust, generated C, LLVM, or
   another backend is selected only through the compiled-language decision gate.
10. **No LLVM by prestige.** LLVM enters the roadmap only if measured workloads show a material
    need that the selected simpler backend cannot meet.
11. **No custom VM by default.** A bytecode VM requires a separate evidence-backed proposal; MIR
    and a MIR interpreter do not automatically imply a production VM.
12. **No silent semantic stubs.** A handler or compiler path returning placeholder output must be
    labelled `stub` in docs and tests.
13. **No status drift.** When implementation status changes, update the authoritative
    state/coverage records and user-facing summaries in the same work package.
14. **Conformance claims require executable evidence.** Parsing, type-checking, or a commit
    message alone never proves a normative rule complete.
15. **Positive and negative evidence travel together.** Every semantic rule needs valid and
    invalid cases where rejection is meaningful.
16. **Diagnostics are part of behaviour.** Error code, primary span, related spans, notes, help
    text, and machine-readable form must remain testable and deterministic.
17. **Source identity must survive the pipeline.** AST/HIR/MIR/query results and diagnostics must
    retain the correct file, module, package, and artifact provenance.
18. **Cross-tool compiler behaviour must converge.** CLI, test runner, doc example validator,
    LSP, deployment tool, and future native builder must use shared compiler entry points rather
    than subtly different pipelines.
19. **Do not generalise from one extension too early.** Internal artifact-provider infrastructure
    is promoted only after two independent artifact implementations demonstrate the same need.
20. **Negative gate results are valid.** A decision to defer native compilation, an optimisation
    tier, or a generic abstraction is a successful research result when supported by evidence.

### 1.6 Explicit not-Core list

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
Do not implement it as an incidental solution to a compiler, tooling, standard-library, package,
or native-backend problem.

### 1.7 Terminology

- **normative specification** — the individual Core v1 documents under `STARKLANG/docs/spec/`
  and approved extension specifications under `STARKLANG/docs/extensions/`.
- **generated specification** — combined Markdown/HTML/PDF outputs; regenerated artifacts, never
  the editing surface.
- **AST** — syntax-preserving arena representation produced by the parser.
- **HIR** — resolved/desugared semantic representation used by type checking, borrow checking,
  interpretation, and compiler queries.
- **typed HIR** — HIR plus type, resolution, ownership, and related side tables.
- **MIR** — optional backend-independent control-flow IR introduced only after Gate C3
  authorises the compiled-language track.
- **semantic oracle** — the normative spec plus the reference interpreter and conformance tests
  used to judge later backends.
- **compiler query** — a stable read-only semantic operation such as "symbol at position," "type
  of node," "definition of symbol," or "references to symbol."
- **extension provider** — compiler-internal integration for optional syntax/semantics;
  extensions remain explicitly enabled and isolated from Core.
- **artifact provider** — a compiler/tooling contract that reads an artifact, extracts a typed
  contract, records identity/provenance, verifies drift, and contributes diagnostics.
- **conformance deviation** — a known mismatch between implementation and normative behaviour,
  explicitly recorded with impact and planned disposition.
- **backend capability** — a declared statement of what a selected execution backend can
  actually lower and execute.

### 1.8 Source-of-truth hierarchy

When documents disagree, use this order:

1. approved normative Core or extension specification;
2. approved decision record in `COMPILER-STATE.md` or a gate proposal;
3. gate exit evidence and executable tests;
4. compiler roadmap;
5. engineering plan/work-package documentation;
6. README, CLAUDE context, implementation notes, and commit messages;
7. archived pre-pivot documents — never authoritative.

Gate C0 must identify stale documents and either update them or label them historical.

### 1.9 Standing implementation constraints

- Implementation language: stable Rust.
- Existing `starkc/` Rust implementation is the active compiler; archived Python compiler code
  is not extended.
- Arena IDs and explicit spans remain the default representation unless an approved architecture
  decision replaces them.
- The reference execution path remains the interpreter until a later gate records otherwise.
- New external Rust dependencies require a short necessity, maintenance, licence, and security
  note.
- Hand-written parsers/serialisers are not a virtue by themselves; use a mature dependency when
  it materially lowers correctness or maintenance risk and is compatible with project
  constraints.
- Avoid broad refactors that are not required by the active WP.
- Public compiler APIs must not expose unstable internal arena indexes as persistent external
  identities.

---

## 2. Operating protocol

### 2.1 Session flow

1. Load `COMPILER-CHARTER.md`, `COMPILER-STATE.md`, and the active WP file.
2. Confirm the repository head and current WP scope.
3. Read only the normative rules, source modules, tests, and prior decisions required by the WP.
4. Record any newly discovered adjacent problem as `FOLLOW-UP:` unless it blocks the WP's
   correctness.
5. Implement the WP without importing later-gate mechanisms.
6. Run every `Done when` check.
7. Run deterministic commands twice when deterministic output is claimed.
8. Update conformance records for every normative rule whose status changed.
9. Update `COMPILER-STATE.md` with files, decisions, deviations, evidence, follow-ups, and the
   next WP.
10. Commit with message `[WP-Cx.y] <summary>`.

At a gate exit:

1. reload the full compiler roadmap;
2. evaluate every criterion as PASS, FAIL, DEFERRED-BY-DECISION, or NOT-APPLICABLE;
3. write the gate exit report with command-level evidence;
4. obtain owner review for semantic, architecture, or scope conclusions;
5. only then open the next gate or conditional track.

### 2.2 Sonnet-level autonomy

You may:

- implement spec-consistent fixes in lexer, parser, resolver, type checker, borrow checker,
  interpreter, diagnostics, compiler query infrastructure, and tests when the active WP
  authorises those areas;
- refactor files already in scope when the refactor directly enables the WP and behaviour
  remains covered;
- add positive, negative, regression, fuzz, and property tests;
- split a WP before implementation when its acceptance surface is too large;
- improve diagnostics without changing the accepted/rejected program set, provided diagnostic
  tests and codes remain compatible or an approved correction is recorded;
- create temporary backend spikes inside Gate C3 that are not merged as product architecture
  until the gate decision.

You may not, without escalation:

- change the normative accepted/rejected program set;
- add syntax, keywords, attributes, operators, type forms, effects, or ownership rules;
- weaken ownership, borrowing, coherence, exhaustiveness, or artifact verification to make code
  compile;
- alter panic/trap/drop semantics;
- choose the MIR contract, runtime ABI, data layout, unwind model, or production backend;
- add nightly Rust requirements;
- introduce a new extension or artifact class;
- claim Core v1 or tensor v0.1 conformance;
- convert a known stub into a documented feature without implementing semantic behaviour;
- begin anything on the not-yet list.

### 2.3 Escalations

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

A spec bug whose correction is unambiguous may be handled under the established spec-bug
protocol, but record it in `COMPILER-STATE.md` and the gate evidence ledger.

### 2.4 `COMPILER-STATE.md` shared memory

See `COMPILER-STATE.md` at the repository root. Rules:

- decision log is append-only;
- deviations are never deleted without a closing note and evidence link;
- keep the state file under approximately 700 lines by compressing closed gate detail into
  summaries;
- every session ends with a dated `### WP-Cx.y` entry recording DONE/FILES/RULES/DECISIONS/
  EVIDENCE/FOLLOW-UP/NEXT.

### 2.5 Definition-of-done defaults

When a WP does not state stricter checks:

- `cargo fmt --check` passes;
- `cargo clippy --all-targets -- -D warnings` passes;
- the full Rust test suite passes;
- Core fixture conformance passes;
- tensor/extension tests pass when extension code is touched;
- every changed normative rule has positive evidence and negative evidence where applicable;
- every regression fix has a test that failed before the fix;
- diagnostics are snapshot- or structurally tested;
- no new undocumented `unwrap`, panic, global mutable state, nondeterministic iteration, or
  process execution is introduced in compiler paths;
- generated output is deterministic across two runs;
- docs distinguish complete, partial, stub, and deferred behaviour;
- `COMPILER-STATE.md` and coverage data are updated;
- nothing merges with failing checks.

### 2.6 Scope-control test before coding

Before each WP, answer internally:

```text
What exact compiler claim is this WP testing or completing?
What later mechanism would make the result impossible to attribute?
What is the strongest existing implementation path or comparator?
What negative result would cause this WP or gate to stop?
```

If those answers are unclear, narrow the WP before implementation.

---

## 5. Diagnostic and evidence registries

### 5.1 Diagnostic governance

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
User-source diagnostics should continue using normative language/extension codes where
applicable.

### 5.2 Evidence classes

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

### 5.3 Gate decision vocabulary

Use only:

```text
PASS
PASS-WITH-DEVIATIONS
REVISE
DEFER
STOP
FAIL
```

"Mostly complete" is not a gate decision.
A work package may be `partial`, but a gate report must state what that means for the next gate.

---

## 6. Not-yet list — refuse and cite the charter

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

## 8. Strategic outcome — keep this in view

The compiler track is not successful merely because it accumulates features or lines of Rust.
Its outcomes are:

1. **A trustworthy Core implementation:** every accepted and rejected program is tied to
   normative rules and executable evidence.
2. **A reliable semantic oracle:** the interpreter and compiler query model give later tools and
   backends one consistent meaning of STARK programs.
3. **An evidence-based native decision:** STARK becomes a general native compiler only if the
   value justifies MIR, runtime, backend, platform, and maintenance complexity.
4. **Real compiler-backed tooling:** diagnostics, hover, navigation, references, and completion
   come from semantic identity rather than protocol stubs or text matching.
5. **Disciplined extension architecture:** tensor remains isolated, and generic artifact
   infrastructure appears only after two real artifact implementations justify it.
6. **Precise release claims:** the project states exactly what conforms, what is experimental,
   and what remains deferred.

The strongest possible compiler result may be any of:

```text
A conforming Core front end + interpreter research language
A conforming Core language with a practical native backend
A compiler framework specialised for artifact-bound verification
A negative native-backend decision that saves years of unnecessary work
```

The roadmap must discover which result the evidence supports.
It must not assume the answer in advance.
