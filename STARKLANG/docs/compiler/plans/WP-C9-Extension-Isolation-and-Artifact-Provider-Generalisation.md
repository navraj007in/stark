# WP-C9 — Extension Isolation and Conditional Artifact-Provider Generalisation

**Execution brief for Codex / Claude**  
**Repository:** `navraj007in/stark`  
**Prepared:** 2026-07-31  
**Starting position:** Gate C8 was `C8-CANDIDATE-COMPLETE` at CD-281, which is what permitted C9 to
begin. C8 has since been **CLOSED** (CD-385, 2026-08-06); this line records the position at the time
and is not the current status.  
**Authority:** `STARKLANG/docs/compiler/COMPILER-ROADMAP.md`, Gate C9.

---

## 0. Executive directive

Execute Gate C9 in two deliberately separated parts:

```text
PART A — may begin immediately
  C9.0  baseline and governance
  C9.1  extension-isolation conformance
  C9.2  tensor/ONNX provider map

PART B — BLOCKED
  C9.3  second-artifact comparison
  C9.4  internal artifact-provider contract
  C9.5  migrate both implementations
  C9.6  gate decision
```

Do **not** begin provider generalisation from ONNX alone.

Part B may start only after a second independent artifact implementation exists and has a working gate report. The expected candidate is a bounded OpenAPI or schema contract-only experiment, but this brief does not authorise building a large OpenAPI product merely to unblock C9.

The permitted C9 outcomes are:

```text
GENERALISE
KEEP-SEPARATE
REVISE
```

`GENERALISE` is not predetermined.

---

## 1. Purpose

Gate C9 has two goals:

1. Prove that optional compiler extensions are isolated from Core and from each other across all compiler and editor entry points.
2. Decide, from two real implementations, whether STARK needs an internal reusable artifact-provider contract.

C9 is **not** a language-feature gate. It must not introduce:

- public generic artifact syntax;
- arbitrary compiler plugins;
- artifact code execution during compilation;
- cloud-provider-specific language syntax;
- a capability/effect/information-flow system;
- broad OpenAPI client/server generation;
- a public `Artifact<T>` Core type.

---

## 2. Starting facts

The current compiler already has:

- Core-only compiler modes;
- an optional tensor/ONNX extension;
- compiler-backed LSP analysis;
- shared `ProjectAnalysis` semantic queries;
- package-aware LSP snapshots with open-buffer overlays;
- CLI and editor extension configuration paths;
- ONNX artifact reading, signature extraction, identity/provenance recording, declaration generation, drift verification, deployment lowering, and diagnostics.

Gate C8 is **closed** (CD-385, 2026-08-06; candidate-complete when this plan was written). C9 work must not reopen C8 scope except where C9.1 tests extension isolation through the LSP and exposes a real defect.

The C8 validation gap, `DEV-012`, remains post-C8 interactive editor validation and is not owned by C9.

---

## 3. Non-negotiable controls

### 3.1 Core remains the default

Every entry point must behave as Core-only unless an extension is explicitly enabled.

### 3.2 No hidden global extension state

One analysis session must not affect another.

This must be proven for:

- sequential compiler invocations in the same process;
- repeated LSP initialisations where supported;
- tests running different extension sets;
- package and single-file paths;
- formatter/doc/check/run/deploy/verifier paths.

### 3.3 No abstraction before evidence

C9.2 maps the ONNX implementation but does not refactor it.

C9.4 is prohibited until C9.3 has compared ONNX with a second working implementation.

### 3.4 Preserve user-visible contracts

Migration under C9.5 must preserve:

- accepted source syntax;
- diagnostics and diagnostic categories;
- declaration shapes;
- identity/provenance behaviour;
- drift-detection behaviour;
- deployment/runtime obligations;

unless a separate approved compatibility decision explicitly changes them.

### 3.5 Tests must assert absence as well as presence

For each extension-enabled positive case, add a corresponding Core-only negative case where meaningful.

The test suite must be able to detect accidental extension leakage.

---

# PART A — IMMEDIATE WORK

## 4. WP-C9.0 — Baseline and governance

### 4.1 Objective

Record the exact current extension architecture before changing anything.

### 4.2 Deliverables

Create:

```text
STARKLANG/docs/compiler/work-packages/WP-C9.0-BASELINE.md
STARKLANG/docs/compiler/work-packages/WP-C9.1-EXTENSION-ISOLATION.md
STARKLANG/docs/compiler/work-packages/WP-C9.2-TENSOR-PROVIDER-MAP.md
starkc/docs/compiler/evidence/c9/README.md
```

Update `COMPILER-STATE.md` additively with:

```text
Gate C9: OPEN
Active WP: C9.0
Part A: AUTHORISED
Part B: BLOCKED pending second-artifact evidence
```

### 4.3 Baseline inventory

Identify every active extension-selection path:

- CLI flags;
- compiler options types;
- package manifest configuration;
- LSP `initializationOptions`;
- test helpers;
- formatter;
- doc generator;
- verifier;
- run/check/build/deploy commands;
- any tensor-specific binary or subcommand;
- any environment-derived or global configuration;
- cached compiler/LSP state.

For each path record:

```text
entry point
default extension set
configuration source
validation behaviour
unknown-name behaviour
duplicate-name behaviour
session lifetime
shared/global state
tests
```

### 4.4 Baseline commands

At minimum run and record:

```bash
cargo check --workspace
cargo test --workspace
cargo test -p starkc --lib lsp:: -- --nocapture
npm run compile --prefix editors/vscode
```

If the full workspace suite is too expensive for every packet, run it at C9.0 baseline and at Part A close, while using targeted suites during implementation.

### 4.5 Exit criteria

C9.0 closes when:

- every extension entry point is inventoried;
- current default behaviour is measured rather than assumed;
- all known global or process-wide state is listed;
- Part B remains explicitly blocked;
- the exact next implementation packet is named.

---

## 5. WP-C9.1 — Extension-isolation conformance

### 5.1 Objective

Prove that Core and optional extensions are isolated across every supported entry point and session boundary.

### 5.2 Required behaviour matrix

Build a permanent machine-readable or table-driven matrix covering at least:

| Entry point | Core default | Tensor disabled rejects | Tensor enabled accepts | Builtins isolated | Unknown extension | Duplicate extension | Session isolation |
|---|---:|---:|---:|---:|---:|---:|---:|
| `starkc check` | Required | Required | Required | Required | Required | Required | N/A |
| `starkc run` | Required | Required | Required | Required | Required | Required | N/A |
| `stark check` | Required | Required | Required | Required | Required | Required | N/A |
| `stark build` | Required | Required | Required | Required | Required | Required | N/A |
| formatter | Required | Required | Required | Required | Required | Required | Required |
| doc generator | Required | Required | Required | Required | Required | Required | Required |
| verifier | Required | Required | Required | Required | Required | Required | Required |
| deploy path | Required | Required | Required | Required | Required | Required | Required |
| LSP single-file | Required | Required | Required | Required | Required | Required | Required |
| LSP package mode | Required | Required | Required | Required | Required | Required | Required |
| test harness helpers | Required | Required | Required | Required | Required | Required | Required |

If an entry point does not support an extension configuration surface, record and test the intended refusal or fixed mode. Do not silently omit it.

### 5.3 Required scenarios

#### A. Core-only default

A source file containing tensor syntax or tensor-only symbols must be rejected when no extension is enabled.

Test:

- unqualified tensor builtin use;
- qualified tensor builtin use;
- tensor syntax;
- tensor model declarations;
- artifact declarations or annotations if present;
- imported module attempting to use tensor-only symbols.

#### B. Explicit enablement

The same source forms must be accepted when the tensor extension is explicitly enabled through each supported configuration path.

#### C. Namespace and resolution isolation

Tensor-only names must not enter Core symbol tables.

Prove both:

```text
unqualified lookup fails
qualified lookup fails
```

where Core mode does not admit the tensor namespace.

#### D. Unknown extensions

Unknown extension names must produce one consistent, documented outcome.

Allowed outcomes:

```text
reject with structured diagnostic
ignore with explicit documented policy
```

The policy must be consistent across CLI, package, LSP, and internal options conversion where those surfaces apply.

If current behaviour differs, escalate the policy decision before normalising it.

#### E. Duplicate extensions

Duplicate declarations must be deterministic and consistent.

Allowed outcomes:

```text
reject duplicate declaration
deduplicate idempotently
```

Again, do not choose policy accidentally in implementation.

#### F. Sequential-session isolation

In one process:

```text
session 1: tensor enabled
session 2: Core only
```

The second session must not retain:

- extension syntax;
- extension builtins;
- extension diagnostics;
- extension symbol registrations;
- artifact-provider state;
- cached declarations;
- package graph extension configuration.

Repeat in the reverse order.

#### G. LSP isolation

Test at least:

- Core-only LSP initialisation;
- tensor-enabled LSP initialisation;
- separate server instances with different options;
- package and standalone documents;
- close/reopen behaviour;
- configuration restart behaviour supported by the VS Code extension.

Do not rely only on advertised capabilities. Drive actual semantic requests or diagnostics.

#### H. Parallel safety

Where compiler analysis may be run concurrently, add a test using two extension configurations in parallel. The test must prove no shared mutable extension state crosses requests.

### 5.4 Implementation constraints

Prefer immutable per-analysis compiler options.

Do not solve isolation by:

- resetting globals before every test;
- serialising all analyses;
- relying on test process isolation;
- cloning a global registry that can still be mutated;
- making tensor symbols universally visible and filtering diagnostics later.

### 5.5 Diagnostics

Extension-disabled errors must:

- use stable structured diagnostic codes;
- identify the disabled extension where useful;
- distinguish unsupported extension syntax from ordinary unknown symbols;
- preserve source provenance;
- behave consistently in CLI and LSP publication.

### 5.6 Evidence

Add permanent tests with names that identify the property, not the implementation.

Suggested groups:

```text
c91_core_default.rs
c91_tensor_enablement.rs
c91_builtin_isolation.rs
c91_unknown_duplicate_config.rs
c91_session_isolation.rs
c91_entry_point_matrix.rs
c91_lsp_isolation.rs
```

Update:

```text
starkc/docs/compiler/evidence/c9/README.md
WP-C9.1-EXTENSION-ISOLATION.md
COMPILER-STATE.md
```

### 5.7 Exit criteria

C9.1 closes only when:

- every active entry point is classified and tested;
- Core-only default is proven;
- tensor syntax and builtins do not leak;
- unknown and duplicate configuration behaviour is unified or explicitly dispositioned;
- sequential and parallel session isolation are proven;
- LSP isolation is covered through real semantic behaviour;
- no global mutable extension state remains unclassified;
- all targeted and full qualification suites pass.

---

## 6. WP-C9.2 — Tensor/ONNX provider map

### 6.1 Objective

Document the existing ONNX integration as a stage pipeline and identify candidate reusable responsibilities without refactoring.

### 6.2 Required stage map

Map these stages to concrete modules, functions, types, tests, and diagnostics:

```text
1. parse extension syntax
2. resolve extension symbols
3. type-check tensor/model semantics
4. read ONNX artifact
5. normalise supported artifact content
6. extract typed signature
7. record identity and provenance
8. generate/register declarations
9. verify declaration/artifact drift
10. lower supported deployment pipeline
11. report backend/runtime obligations
12. contribute structured diagnostics
```

For every stage record:

```text
owner module
input representation
output representation
tensor-specific logic
apparently reusable logic
state/cache used
filesystem access
hashing/identity method
diagnostic codes
tests
failure modes
```

### 6.3 Required data-flow diagrams

Include at least:

```text
source -> extension parser -> semantic model
artifact path -> ONNX reader -> normalised model -> typed contract
typed contract -> declaration registration
artifact identity -> provenance record
later artifact -> drift comparator
verified model -> deployment lowering
```

### 6.4 Candidate abstraction ledger

Classify each responsibility as:

```text
TENSOR-SPECIFIC
POSSIBLY-REUSABLE
EVIDENCE-INSUFFICIENT
MUST-REMAIN-SEPARATE
```

Do not mark anything `REUSABLE` yet. That conclusion belongs to C9.4 after C9.3.

### 6.5 Questions C9.2 must answer

- What exactly is the artifact identity?
- Which bytes or normalised structures are hashed?
- What provenance is retained?
- How is declaration/artifact drift defined?
- What source locations are attached to artifact diagnostics?
- Does artifact reading occur during check, verify, deploy, or all three?
- What is cached?
- Can a malformed artifact panic the compiler?
- Which limits bound artifact size, graph depth, tensor rank, and recursion?
- Which stages are deterministic?
- Which deployment obligations are ONNX-runtime-specific?
- Which compiler phases know they are processing ONNX?

### 6.6 Prohibited work

Do not:

- introduce a provider trait;
- move ONNX code behind a generic interface;
- rename domain types to generic artifact terminology;
- add mock providers as evidence of reuse;
- alter ONNX user-visible behaviour.

### 6.7 Exit criteria

C9.2 closes when:

- the complete ONNX path is traceable;
- stage boundaries and ownership are explicit;
- candidate reusable and tensor-specific portions are classified;
- no refactor has been performed;
- the second-artifact evidence requirements are concrete.

---

# PART B — BLOCKED UNTIL SECOND IMPLEMENTATION EXISTS

## 7. WP-C9.3 — Second-artifact evidence gate

### 7.1 Entry condition

Do not start until there is a working second artifact implementation with:

- source syntax or configuration;
- artifact reading;
- typed contract extraction;
- identity/provenance;
- declarations or registration;
- drift verification;
- structured diagnostics;
- executable tests;
- a gate or closure report;
- bounded, explicit non-goals.

A documentation-only design does not qualify.

### 7.2 Preferred second implementation

The preferred candidate is a narrow OpenAPI or schema contract-only experiment.

Recommended scope:

```text
read a pinned OpenAPI document
normalise the supported subset
extract endpoint/request/response type contracts
register declarations
record identity/provenance
detect artifact drift
produce structured diagnostics
```

Explicitly exclude unless separately authorised:

```text
HTTP execution
client generation
server generation
authentication
code generation for every schema feature
remote URL fetching
runtime networking
cloud deployment
```

### 7.3 Comparison matrix

Compare ONNX and the second implementation across:

- artifact acquisition;
- parsing;
- normalisation;
- identity hashing;
- provenance;
- typed contract extraction;
- declaration registration/generation;
- drift detection;
- diagnostics;
- compiler phase integration;
- caching;
- security limits;
- backend/runtime obligations;
- deterministic output;
- test strategy.

### 7.4 Required conclusion

For each candidate common responsibility, classify:

```text
same semantics
same shape, different semantics
superficially similar only
provider-specific
```

### 7.5 Exit criteria

C9.3 closes only when the comparison is based on running code and committed evidence.

---

## 8. WP-C9.4 — Internal artifact-provider contract

### 8.1 Objective

Under CE7, define the smallest internal contract justified by C9.3.

### 8.2 Candidate responsibilities

The contract may cover only responsibilities proven common:

```text
read artifact
normalise supported content
extract typed contract
record identity and provenance
register or generate declarations
verify later artifact versions
emit structured diagnostics
report backend/runtime obligations
```

### 8.3 Design requirements

The contract must specify:

- lifecycle;
- ownership;
- error taxonomy;
- deterministic ordering;
- artifact identity;
- provenance schema;
- caching boundaries;
- compiler phase entry points;
- cancellation/resource limits where relevant;
- diagnostic conversion;
- no artifact code execution.

### 8.4 Internal only

The contract is compiler-internal.

No public language syntax or stable plugin ABI is authorised.

### 8.5 Acceptance test

Before migration, implement a deliberately tiny third mock fixture only to test the abstraction’s shape. It must not be counted as the second real implementation and must not drive the design.

The mock should demonstrate whether the proposed contract forces unnecessary domain concepts.

### 8.6 Decision gate

If the contract requires many optional hooks, provider downcasts, domain switches, or leaky enums, prefer `KEEP-SEPARATE`.

---

## 9. WP-C9.5 — Migrate both implementations

### 9.1 Objective

Move ONNX and the second implementation onto the approved internal contract without changing user-visible behaviour.

### 9.2 Required evidence

For both implementations prove:

- existing positive cases still pass;
- existing negative diagnostics remain stable or improve;
- artifact identity remains compatible;
- drift detection is unchanged;
- provenance remains available;
- compiler phase integration remains deterministic;
- no new global state is introduced;
- unsupported features remain refused.

### 9.3 Measurements

Record:

```text
duplicated code removed
domain-specific code retained
new indirection introduced
diagnostic churn
test churn
compile-time change
maintenance complexity
```

Do not claim success merely from line-count reduction.

### 9.4 Rollback condition

If migration makes either provider harder to understand, weakens diagnostics, introduces generic escape hatches, or couples backend obligations, stop and recommend `KEEP-SEPARATE`.

---

## 10. WP-C9.6 — Gate exit

### 10.1 Permitted outcomes

#### GENERALISE

Use only when:

- two real implementations share a stable internal lifecycle;
- duplication is materially reduced;
- diagnostics remain precise;
- domain code remains separate;
- a third fixture fits without special cases;
- the contract is smaller than the duplicated machinery it replaces.

#### KEEP-SEPARATE

Use when:

- similarity is superficial;
- domain semantics dominate;
- the abstraction adds hooks/switches/downcasts;
- migration weakens clarity;
- duplicate code is small or intentionally different.

This is a successful evidence-based outcome, not a failure.

#### REVISE

Use only for one bounded follow-up with a concrete unanswered question.

### 10.2 Gate report

Create:

```text
STARKLANG/docs/compiler/GATE-C9-CLOSURE.md
```

Include:

- qualifying commit;
- Part A evidence;
- second-artifact identity and gate report;
- comparison matrix;
- CE7 decision;
- migration evidence;
- final outcome;
- explicitly deferred user-facing syntax;
- open deviations and owners.

---

## 11. Parallel execution plan

### Track A — Codex

Recommended ownership:

```text
C9.0 baseline inventory
C9.1 entry-point matrix
CLI/package/test harness isolation tests
global-state audit
```

### Track B — Claude

Recommended ownership:

```text
C9.2 ONNX stage map
diagnostic/provenance/identity analysis
LSP extension-isolation tests
governance and evidence documents
```

### Shared files requiring coordination

Expect overlap in:

```text
COMPILER-STATE.md
STARKLANG/docs/compiler/COMPILER-ROADMAP.md
starkc/src/options.rs
starkc/src/analysis.rs
starkc/src/package.rs
starkc/src/lsp/**
compiler test registries
```

Use separate branches/worktrees.

Do not allow both agents to edit the same shared file concurrently without an explicit ownership split.

Recommended split:

```text
Codex owns implementation and non-LSP tests.
Claude owns docs, ONNX map, and LSP-specific tests.
One integrator owns COMPILER-STATE.md and final reconciliation.
```

---

## 12. Commit discipline

Use one commit per coherent packet where possible.

Suggested sequence:

```text
CD-C9-01  C9.0 baseline and state admission
CD-C9-02  Core default and tensor enablement matrix
CD-C9-03  builtin and namespace isolation
CD-C9-04  unknown/duplicate configuration policy
CD-C9-05  sequential and parallel session isolation
CD-C9-06  LSP/package/editor extension isolation
CD-C9-07  C9.1 qualification and closure
CD-C9-08  ONNX provider map
CD-C9-09  C9 Part A closeout
```

Do not assign final CD numbers in advance if the repository uses owner-issued numbering. Use descriptive commit messages until integrated.

Each commit message should state:

```text
problem
finding
change
evidence
claim boundary
carried limitations
```

---

## 13. Required qualification at Part A close

Run:

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace
cargo test -p starkc --lib lsp:: -- --nocapture
npm run compile --prefix editors/vscode
```

Also run all new C9 isolation matrices explicitly and record their case counts.

Cross-platform qualification is required if any extension-selection path contains platform-specific file, URI, process, or environment behaviour.

At minimum:

```text
linux-x64
macos-arm64
windows-x64
```

for the relevant CI suite.

---

## 14. Definition of done for the immediate assignment

The immediate Codex/Claude assignment is complete when:

- C9.0 is closed;
- C9.1 is closed;
- C9.2 is closed;
- Gate C9 is recorded as:

```text
PART-A-COMPLETE
PART-B-BLOCKED-PENDING-SECOND-ARTIFACT
```

- no provider abstraction has been introduced;
- the second-artifact entry criteria are explicit;
- all tests and documentation are committed;
- C8 remains untouched except for genuine extension-isolation fixes in shared LSP/options code.

---

## 15. Final instruction to the implementation agent

Do not optimise for declaring C9 complete.

Optimise for making two claims trustworthy:

1. Core and optional extensions are isolated across every compiler and editor path.
2. Any future artifact-provider abstraction will be justified by two real implementations rather than by resemblance observed in one.

If Part A finds architectural leakage, fix and qualify it before proceeding.

If Part B later finds that ONNX and the second implementation do not share enough semantics, record `KEEP-SEPARATE` and close the gate honestly.
