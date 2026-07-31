# WP-C7 — Build Profiles, Reproducibility, and Evidence-Gated Optimisation

**Assigned implementer:** Claude
**Parallel track:** Gate C8 may proceed concurrently under a separate owner.
**Starting point:** exact Gate C6 closure commit agreed by both tracks.
**Dependencies:** Gate C6 closed.
**External dependency for final closure:** Systems Roadmap P1 completed sufficiently to provide the practical systems workloads required by WP-C7.5.
**Primary outcome:** a usable, measurable and reproducible native build path whose release and optimisation modes preserve STARK semantics.

---

## 0. Directive

Implement Gate C7 without reopening language semantics or weakening the differential guarantees established by C6.

C7 is not permission to optimise first and test later.

The order of authority is:

```text
STARK specification
    ↓
HIR semantic authority
    ↓
unoptimised MIR execution
    ↓
optimised MIR/native execution
    ↓
measured performance
```

An optimisation that changes observable semantics is invalid even when it improves performance.

C7 must deliver:

* documented debug and release build profiles;
* target selection;
* identical language semantics across profiles;
* a precise reproducibility contract;
* measured evidence for output reproducibility;
* profiling before caching;
* the smallest justified build cache;
* a bounded optimisation pipeline;
* differential validation of every optimisation;
* performance and complexity measurements;
* an evidence-based LLVM decision, normally `DEFER`;
* a final C7 exit report.

Do not:

* add language features;
* broaden the standard-library API;
* implement C8 editor services;
* add File, Random, networking or concurrency as compiler features;
* use performance results to retroactively redefine semantics;
* introduce a full incremental query engine without evidence;
* select LLVM because it appears more mature in general;
* claim broad performance superiority from microbenchmarks.

---

## 1. Parallel-work isolation

C7 and C8 are active concurrently.

### 1.1 C7-owned areas

Claude owns, subject to the repository lease protocol:

```text
build command and CLI profile handling
target selection
native build orchestration
generated-Rust profile configuration
backend optimisation configuration
reproducibility metadata
build cache implementation
optimisation passes
benchmark harnesses
performance reports
C7 work-package documents
C7 evidence
C7 exit report
```

Likely affected paths include, but must be verified before editing:

```text
starkc/src/cli/**
starkc/src/backend/**
starkc/src/mir/**
starkc/src/build/**
starkc/src/package/**
starkc/tests/*build*
starkc/tests/*native*
starkc/tests/*optim*
starkc/benches/**
.github/workflows/**
STARKLANG/docs/compiler/work-packages/WP-C7*.md
starkc/docs/compiler/evidence/c7/**
starkc/docs/compiler/C7-exit-report.md
```

Do not assume these exact paths exist. Map the current tree first.

### 1.2 Shared areas

Potentially shared with C8:

```text
COMPILER-STATE.md
COMPILER-ROADMAP.md
COMPILER-CHARTER.md
Cargo.toml
Cargo.lock
CI workflows
compiler entry points
project/package loading
structured diagnostics
test-support utilities
```

Before editing a shared file:

1. take a lease in the integration ledger;
2. state the bounded reason;
3. avoid unrelated formatting;
4. commit the smallest coherent change;
5. release the lease;
6. reconcile against the C8 branch before final qualification.

### 1.3 Forbidden interference

Do not:

* modify LSP handlers or VS Code extension files;
* rewrite the shared project-analysis API for build convenience;
* change source spans or diagnostic contracts without C8 coordination;
* overwrite C8 state updates;
* perform repository-wide dependency or formatting churn.

---

## 2. WP-C7.0 — Baseline and measurement integrity

Before implementing profiles or optimisations, establish the actual current build path.

### 2.1 Build-path inventory

Document:

* current `stark build` behaviour, if any;
* current `stark run` native path;
* generated Rust stages;
* temporary directories;
* `rustc` invocation;
* linker invocation;
* runtime crate selection;
* target selection;
* debug/release defaults;
* environment variables affecting output;
* timestamps and absolute paths embedded in outputs;
* package graph resolution;
* lockfile use;
* build artefact locations;
* cleanup behaviour;
* any existing cache;
* any backend optimisation flags;
* CI build targets.

### 2.2 Frozen baseline workloads

Select and check in a bounded initial workload set:

1. minimal program;
2. arithmetic/control-flow program;
3. generic/trait program;
4. String/Vec allocation workload;
5. HashMap/HashSet workload;
6. multi-package program;
7. Drop/ownership-sensitive program;
8. parser or structured-data workload when available.

Later add P1 workloads.

For each workload, record:

```text
source identity
package graph identity
compiler commit
toolchain
target
profile
expected output
expected trap/drop behaviour where relevant
```

### 2.3 Measurement harness

Create a deterministic harness capable of recording:

* front-end time;
* MIR construction and verification time;
* generated-source emission time;
* host compiler time;
* link time;
* total clean-build time;
* warm-build time;
* peak compiler memory;
* generated source size;
* executable size;
* startup time;
* steady-state runtime.

Separate STARK compiler time from host compiler time.

Do not report only the combined total.

### 2.4 Baseline report

Create:

```text
STARKLANG/docs/compiler/work-packages/WP-C7.0-BASELINE.md
```

Include:

* exact head;
* machine and OS;
* Rust toolchain;
* build commands;
* workload identities;
* raw measurements;
* known measurement noise;
* current profile behaviour;
* current reproducibility observations;
* candidate cache bottlenecks;
* candidate optimisation opportunities.

C7.0 closes only when later performance claims can be compared against a fixed baseline.

---

## 3. WP-C7.1 — Build profiles and target selection

Implement and document:

```bash
stark build
stark build --release
stark build --target <triple>
```

### 3.1 Profile semantics

Define at minimum:

#### Debug

Purpose:

* rapid development;
* useful generated/native diagnostics;
* no semantic shortcuts;
* limited or no optimisation;
* debug information where practical.

#### Release

Purpose:

* optimised native output;
* same STARK-observable semantics;
* deterministic and documented optimisation pipeline;
* no unchecked arithmetic substitution;
* no removal or reordering of observable Drop or traps.

### 3.2 Semantic invariance

Debug and release must agree on:

* stdout;
* stderr where specified;
* return value;
* exit status;
* trap category;
* trap source location to the documented precision;
* integer overflow;
* division by zero;
* indexing failures;
* ownership;
* borrow validity;
* Drop order;
* assertion behaviour;
* panic/abort behaviour;
* trait and method dispatch;
* collection iteration order;
* package identity.

Release must not inherit host-language profile-dependent semantics that contradict STARK.

Examples requiring explicit protection:

* Rust overflow checks differing by profile;
* debug assertions present only in one profile;
* backend optimisations eliminating trapping expressions;
* host compiler reordering Drop-observable actions;
* integer casts changing behaviour under optimisation.

### 3.3 Target selection

Support the target tiers already established:

```text
Tier 1:
- x86_64-unknown-linux-gnu
- aarch64-apple-darwin

Tier 2:
- x86_64-pc-windows-msvc

Tier 3 when justified:
- x86_64-apple-darwin
```

Use the repository’s canonical target names.

Required behaviour:

* validate target before expensive compilation;
* reject unknown targets with a structured diagnostic;
* distinguish unsupported target from missing local toolchain;
* do not silently compile for the host when a target was requested;
* include target in artefact paths and cache keys.

### 3.4 Output layout

Define deterministic output locations, for example:

```text
target/stark/<target>/debug/<package>
target/stark/<target>/release/<package>
```

The exact layout may differ, but it must prevent:

* profile collisions;
* target collisions;
* package collisions;
* stale artefact reuse across incompatible settings.

### 3.5 CLI tests

Test:

* default debug build;
* release build;
* explicit host target;
* supported cross-target request;
* unsupported target;
* malformed target;
* profile-specific output path;
* package workspace build;
* binary name collision;
* build after source change;
* build after lockfile change;
* build after compiler-version change.

### 3.6 Differential profile suite

Run every admitted executable C6 corpus case through:

```text
HIR
MIR
native debug
native release
```

Compare the same observation fields used by C6.

Any debug/release divergence is blocking unless the difference is explicitly non-semantic metadata excluded by contract.

### 3.7 Done when

`stark build`, `--release`, and `--target` are functional, documented, tested and semantically equivalent for the admitted executable subset.

---

## 4. WP-C7.2 — Reproducible native outputs

Define reproducibility precisely before claiming it.

### 4.1 Reproducibility classes

Classify artefacts separately:

```text
generated Rust source
intermediate metadata
object files
linked executable
debug symbols
package manifest
dependency graph record
build evidence record
```

For each, state one of:

```text
BYTE-REPRODUCIBLE
SEMANTICALLY-REPRODUCIBLE
PLATFORM-METADATA-EXCLUDED
NOT-YET-REPRODUCIBLE
```

Do not use one global “reproducible” label.

### 4.2 Reproducibility input identity

A reproducibility claim must bind:

```text
STARK source contents
package manifests
lockfile
dependency graph identity
compiler version/commit
runtime version
backend version
MIR version
language/extension options
target triple
profile
build flags
host compiler version
linker version where relevant
environment exclusions
```

### 4.3 Clean-build experiment

For every frozen workload:

1. remove all build outputs;
2. build in clean directory A;
3. build in clean directory B;
4. use identical logical source and lockfile;
5. use different absolute checkout paths;
6. compare each artefact class;
7. record exact differences.

At minimum test:

* same machine, same path;
* same machine, different path;
* two CI runners of the same target where available;
* repeated build at different times;
* debug and release separately.

### 4.4 Sources of nondeterminism

Audit and remove or classify:

* timestamps;
* random temporary names;
* absolute paths;
* hash-map iteration order;
* package traversal order;
* symbol numbering;
* object member ordering;
* linker build IDs;
* UUIDs;
* environment-dependent flags;
* locale;
* timezone;
* host username;
* workspace root;
* dependency discovery order.

Do not strip useful provenance unless the reproducibility contract specifies how it is represented separately.

### 4.5 Generated-source determinism

Generated Rust must be byte-identical for identical logical inputs, target and profile, except for explicitly documented non-semantic headers if any.

Test:

* function ordering;
* symbol naming;
* package/module ordering;
* generic instantiation ordering;
* runtime import ordering;
* metadata ordering;
* diagnostic output ordering.

### 4.6 Build manifest

Emit a machine-readable build manifest containing:

```text
compiler identity
runtime identity
backend identity
target
profile
source hashes
manifest and lockfile hashes
dependency graph hash
generated-source hash
binary hash
reproducibility classification
excluded metadata
```

The manifest itself must be deterministic.

### 4.7 Done when

The repository can state exactly which C7 artefacts reproduce byte-for-byte and which are only semantically reproducible, with clean-build evidence.

---

## 5. WP-C7.3 — Build cache and incremental boundaries

Do not begin implementation until C7.0/C7.1 profiling identifies actual repeated work.

### 5.1 Decision gate

Before building a cache, produce:

```text
clean build stage timings
repeat build stage timings
percentage spent in STARK front end
percentage spent in MIR
percentage spent in generated source
percentage spent in rustc/linker
expected benefit of each cache boundary
correctness invalidation risks
```

If host compilation dominates, do not pretend a front-end-only cache solves total build latency.

### 5.2 Permitted initial cache scope

Implement the smallest useful cache, likely one or more of:

* parsed/resolved/type-checked package snapshot;
* MIR per package or function;
* generated Rust per package;
* host build artefact reuse;
* dependency package output reuse.

Choose based on evidence.

Do not build a full demand-driven incremental query engine unless clean-build or edit-build measurements prove it necessary.

### 5.3 Cache keys

Include every semantic input affecting the cached artefact:

```text
source content hash
compiler version
language version
extension options
package graph identity
dependency public-interface identity
MIR version
runtime/backend version
target
profile
feature/build flags
relevant environment contract
```

A cache hit based only on file modification time is unacceptable.

### 5.4 Dependency invalidation

Test:

* private implementation change;
* public function signature change;
* exported type change;
* trait implementation change;
* package manifest change;
* lockfile change;
* dependency version change;
* compiler upgrade;
* target change;
* debug-to-release change;
* extension configuration change.

Define whether private changes may avoid rebuilding downstream packages. Do not implement interface hashing unless justified.

### 5.5 Cache safety

A stale or corrupt cache must never silently change program semantics.

Required behaviour:

* validate cache format/version;
* reject incompatible entries;
* use content-addressed identities;
* write atomically;
* tolerate interrupted writes;
* avoid following unsafe paths outside the cache root;
* provide cache clear command or documented removal;
* allow cache disabling for qualification.

### 5.6 Differential validation

For every workload compare:

```text
clean uncached build
warm cached build
cache-disabled rebuild
```

All observations and generated artefact identities covered by the contract must agree.

### 5.7 Measurements

Report:

* cache hit rates;
* cold build;
* no-change rebuild;
* one-file private edit;
* public-interface edit;
* dependency edit;
* cache size;
* peak memory;
* invalidation breadth.

### 5.8 Done when

The cache has measured benefit, complete keys for its chosen boundary, and adversarial invalidation tests.

---

## 6. WP-C7.4 — Baseline optimisations

Only these initial optimisation classes are authorised unless a separate owner decision expands scope:

* constant folding consistent with STARK trap semantics;
* dead-block elimination;
* trivial copy propagation;
* unreachable-code removal after semantic diagnostics;
* backend-native optimisation settings.

### 6.1 Optimisation pipeline contract

Document:

```text
input MIR version
pass order
pass preconditions
pass invariants
verification between or after passes
debug profile pass set
release profile pass set
dump/debug mechanism
```

Retain unoptimised MIR as an executable comparison authority.

### 6.2 Constant folding

May fold only when compile-time evaluation exactly matches runtime semantics.

Cover:

* integer arithmetic;
* checked overflow;
* division and remainder;
* divide by zero;
* shifts;
* casts;
* comparisons;
* boolean operations;
* branches;
* string/aggregate constants only where representation is principled.

Do not fold:

* operations whose trap source location would be lost incorrectly;
* host-profile-dependent arithmetic;
* user trait calls unless the specification and implementation make them pure and statically resolvable;
* Drop-bearing constructions where elimination changes observable destruction;
* calls with possible traps or effects.

A constant expression that would trap must preserve the specified trap category and source location. Decide whether it becomes an explicit MIR trap or remains unfolded.

### 6.3 Dead-block elimination

Remove only control-flow blocks proven unreachable after semantic analysis.

Preserve:

* required diagnostics already emitted;
* Drop semantics of reachable paths;
* trap paths;
* source maps needed for surviving instructions;
* verifier invariants.

Do not use optimiser reachability to suppress front-end diagnostics required by the language.

### 6.4 Copy propagation

Limit to trivial cases proven safe under:

* move semantics;
* Copy classification;
* borrows;
* aliasing;
* partial moves;
* Drop;
* reinitialisation;
* projections;
* iterator/reference lifetimes.

Never propagate a Move value as though it were Copy.

Never eliminate a local whose lifetime or Drop is observable.

### 6.5 Unreachable-code removal

Run only after:

* parser/resolver/type-check diagnostics;
* unreachable-code diagnostics;
* ownership checks;
* MIR verification where required.

Optimisation must not become the mechanism that decides whether source code is valid.

### 6.6 Backend-native optimisation settings

Map STARK release mode to explicit host compiler settings.

Record:

* optimisation level;
* overflow-check policy;
* debug assertion policy;
* panic strategy;
* LTO setting;
* codegen units;
* strip/debug information policy;
* target CPU/features policy.

Do not inherit defaults without recording them.

Any host compiler option that changes STARK semantics must be overridden.

### 6.7 Per-pass testing

Every pass requires:

1. unit tests on MIR transformation;
2. verifier run after transformation;
3. unoptimised versus optimised MIR comparison;
4. native debug versus native release comparison;
5. trap fixtures;
6. Drop-order fixtures;
7. borrow/ownership fixtures;
8. package and generic fixtures where relevant.

### 6.8 Metamorphic optimisation controls

Include transformations such as:

* equivalent constant expression forms;
* redundant assignments;
* dead branches;
* reordered independent declarations where semantics allow;
* nested blocks;
* unused Copy locals;
* unreachable code after return/trap.

The optimiser must not introduce new observable differences.

### 6.9 Optimisation fallback

Provide a build flag or internal qualification mode that disables C7 optimisations.

This is required for:

* differential diagnosis;
* benchmark comparison;
* bug reduction;
* release qualification.

### 6.10 Done when

Every enabled optimisation has a written semantic contract, permanent tests and measured effect.

---

## 7. Usage-shape and lifecycle regressions carried from C6

C7 must retain and run the C6 lifecycle-sensitive cases, especially DEV-119.

Release optimisation and host compiler settings must not regress:

```text
iterator dead after loop → mutation succeeds
yielded reference still live → mutation rejected
```

Include at minimum:

* `Vec::iter`;
* `HashMap::keys`;
* `HashSet::iter`;
* exhaustion;
* `break`;
* `continue`;
* nested loops;
* held yielded reference;
* mutation inside loop;
* post-loop mutation;
* Drop-bearing iterator/source where supported.

Also preserve:

* DEV-117 reinitialisation;
* partial moves;
* drop flags;
* user-defined equality/hash authority;
* package-symbol determinism;
* trap provenance.

These cases must run in debug and release.

---

## 8. WP-C7.5 — Performance and complexity report

C7.5 may begin once C7.1–C7.4 produce measurable builds.

It cannot close until P1 provides the practical systems workloads required by the roadmap.

### 8.1 Workload classes

Measure the frozen compiler workloads plus, when P1 is ready:

* file-processing CLI;
* JSON parser workload;
* sequential HTTP server;
* request-routing benchmark;
* allocation-heavy String/Vec workload.

If a P1 workload cannot run because a platform feature remains absent, record that as a dependency failure rather than substituting an unrelated toy benchmark.

### 8.2 Required measurements

For each workload record:

```text
source lines and package count
clean debug compile time
clean release compile time
warm build time
cache-disabled rebuild
peak compiler memory
generated Rust size
executable size
startup time
steady-state runtime
HIR interpreter runtime
MIR interpreter runtime
native debug runtime
native release runtime
debug/release ratio
interpreter/native ratio
```

Separate:

* front-end;
* MIR;
* optimisation;
* generation;
* rustc;
* linker.

### 8.3 Measurement discipline

Use:

* pinned compiler commit;
* pinned Rust toolchain;
* fixed target;
* fixed workload input;
* warm-up runs;
* multiple measured runs;
* median and spread;
* machine specification;
* idle/load conditions;
* checked-in harness.

Do not publish only the best run.

### 8.4 Complexity report

Measure and discuss:

* backend code size;
* optimisation-pass code size;
* generated runtime coupling;
* number of backend-specific tests;
* target-specific branches;
* maintenance burden;
* common failure modes;
* diagnostic quality;
* contributor setup complexity;
* cache complexity.

Performance is not the only C7 output.

### 8.5 Bounded claims

Permitted statements:

> On workload X, release mode reduced steady-state runtime by Y under environment Z.

Not permitted:

> STARK is N times faster than interpreted languages.

Do not extrapolate from a small program to general workloads.

### 8.6 Regression thresholds

Define thresholds for CI or scheduled benchmarking, such as:

* executable count must not decrease;
* debug/release semantic observations must remain identical;
* clean compile time warning threshold;
* generated-source size warning threshold;
* binary-size warning threshold;
* benchmark regression threshold accounting for noise.

Avoid flaky hard failures for noisy microbenchmarks. Use controlled or repeated evidence.

### 8.7 Deliverable

Create:

```text
starkc/docs/compiler/C7-performance-and-complexity-report.md
```

Include raw data or machine-readable attachments.

---

## 9. WP-C7.6 — LLVM decision

The default decision is `DEFER`.

Do not open a broad LLVM implementation project merely because direct code generation may eventually be desirable.

### 9.1 CE6 trigger

Open CE6 only if measured C7/P1 workloads demonstrate a material limitation in the selected generated-Rust backend.

A valid trigger must identify at least one of:

* unacceptable compile latency;
* unacceptable binary size;
* missing target capability;
* inability to preserve a required semantic feature;
* runtime performance materially below a defined requirement;
* host-toolchain deployment burden incompatible with the product;
* backend debugging or maintenance cost that exceeds the alternative.

### 9.2 Required comparison

Compare:

```text
current generated-Rust backend
improvements available within current backend
direct Cranelift route if still relevant
LLVM route
```

Quantify:

* expected benefit;
* implementation scope;
* semantic risk;
* MIR changes;
* runtime/ABI changes;
* debug-info requirements;
* linker/toolchain burden;
* cross-platform support;
* build distribution impact;
* contributor experience;
* maintenance cost;
* qualification cost;
* coexistence or migration strategy.

### 9.3 Allowed outcomes

```text
DEFER
SPIKE-LLVM
SELECT-LLVM-AS-SECOND-BACKEND
REPLACE-BACKEND
REVISE
```

`REPLACE-BACKEND` requires unusually strong evidence and an explicit owner decision.

The existence of a one-second host compilation step alone is not sufficient unless measured workloads show it materially blocks the intended developer experience and simpler mitigations fail.

### 9.4 Spike boundary

If `SPIKE-LLVM` is selected:

* use one frozen representative workload;
* implement the smallest end-to-end path;
* do not port the full language;
* compare semantic fidelity first;
* record unsupported MIR operations;
* delete or archive the spike if it does not justify continuation.

---

## 10. Reproducibility and optimisation evidence registry

Create a machine-readable C7 evidence registry.

For each build profile, target, cache boundary and optimisation record:

```text
feature/pass identity
semantic contract
source implementation
positive tests
negative/adversarial tests
differential tests
targets qualified
workloads measured
last verified commit
known limitations
```

The evidence checker must fail when:

* an enabled optimisation has no differential evidence;
* a named test does not exist;
* a build profile is advertised but not qualified;
* a target is claimed without a target record;
* reproducibility classification lacks a clean-build comparison;
* cached and uncached paths lack agreement evidence;
* benchmark results reference an unknown workload version.

---

## 11. CI requirements

Add or strengthen jobs for:

### 11.1 Profile agreement

```text
HIR
MIR unoptimised
MIR optimised where executable
native debug
native release
```

Compare the frozen differential corpus.

### 11.2 Reproducibility

Perform two clean builds in distinct paths and compare contracted artefacts.

### 11.3 Cache correctness

Compare clean, warm and disabled-cache outputs.

### 11.4 Target matrix

Qualify:

* Linux x64;
* macOS arm64;
* Windows x64 before Core v1 Compiler Stable.

### 11.5 Optimisation tests

Run:

* per-pass unit tests;
* MIR verifier;
* semantic differential suite;
* trap and Drop regressions.

### 11.6 Benchmark recording

Benchmarks may be scheduled or manually qualified if ordinary CI noise is excessive.

Do not let a benchmark job claim correctness.

---

## 12. Robustness and security review

Review the build path for:

* target argument injection;
* linker argument injection;
* generated source path escaping;
* temporary directory races;
* symlink traversal;
* unsafe cache paths;
* corrupt cache entries;
* environment-variable poisoning;
* untrusted package names becoming filenames or symbols;
* absolute path leakage;
* command-line length limits;
* concurrent builds targeting the same artefact;
* interrupted builds;
* partial output publication;
* stale binary execution after failed rebuild.

Host process invocation must use argument arrays, not shell-concatenated command strings.

---

## 13. Commit discipline

Use bounded commits corresponding to one C7 capability.

Every material commit message must state:

```text
problem
measurement or evidence motivating it
semantic invariants
implementation
tests
performance effect
known limitations
shared files touched
C8 reconciliation requirement
```

Do not combine:

* optimisation with unrelated refactoring;
* cache implementation with package-policy redesign;
* profile work with language changes;
* benchmark changes with test expectation weakening;
* dependency upgrades not required by the work package.

If an optimisation discovers a semantic defect:

1. reduce the case;
2. assign a DEV identity;
3. determine whether unoptimised execution is correct;
4. fix the shared semantic layer if necessary;
5. retain the case permanently;
6. do not hide it by disabling the test only in release.

---

## 14. Work-package order

Execute in this order:

```text
C7.0  baseline and measurement integrity
C7.1  build profiles and target selection
C7.2  reproducible native outputs
C7.3  measured cache boundary
C7.4  bounded baseline optimisations
C7.5  performance and complexity report
C7.6  LLVM decision, normally DEFER
C7.7  gate exit
```

Permitted overlap:

* C7.2 reproducibility analysis may start while C7.1 is finalised.
* C7.4 pass prototypes may begin after the unoptimised baseline is frozen.
* C7.5 measurement tooling may be built early.
* C7.5 cannot close before P1 workloads exist.
* C7.6 cannot begin without C7.5 evidence showing a material backend limitation.

---

## 15. Gate exit — WP-C7.7

Create:

```text
starkc/docs/compiler/C7-exit-report.md
```

### 15.1 Required contents

Include:

* exact qualified commit;
* compiler and Rust toolchains;
* build-profile definitions;
* target matrix;
* semantic profile-agreement results;
* reproducibility contract;
* clean-build comparison results;
* cache architecture and invalidation evidence;
* optimisation-pass inventory;
* per-pass semantic evidence;
* frozen workload identities;
* performance measurements;
* complexity assessment;
* P1 practical-workload results;
* LLVM decision;
* known limitations;
* carried DEVs;
* C7/C8 shared-file reconciliation;
* release claim.

### 15.2 C7 closure conditions

C7 may close only when:

1. `stark build`, `--release` and `--target` are usable;
2. debug and release preserve the admitted STARK semantics;
3. supported targets are explicitly qualified;
4. reproducibility is defined per artefact and evidenced;
5. cache behaviour is measured and semantically safe;
6. every enabled optimisation has differential evidence;
7. unoptimised mode remains available for qualification;
8. performance claims are bounded by frozen measurements;
9. P1 practical systems workloads are included;
10. the LLVM decision is recorded;
11. no active C7 claim depends on nonexistent tests or unverified evidence;
12. C8 parallel changes are reconciled on the exact qualified commit.

### 15.3 Permitted exit conclusions

```text
C7-CLOSED
C7-CANDIDATE-COMPLETE-BLOCKED-BY-P1
C7-BLOCKED
```

Before P1 is ready, the highest defensible state is:

```text
C7-CANDIDATE-COMPLETE-BLOCKED-BY-P1
```

provided C7.1–C7.4 are otherwise complete.

### 15.4 Exact closure claim

Use wording no stronger than:

> Gate C7 provides documented debug and release native build profiles for the qualified STARK executable subset, with target-aware outputs, a stated reproducibility contract, measured cache behaviour and evidence-gated baseline optimisations. Debug, release, cached and uncached paths agree on the recorded semantic corpus. Performance conclusions are limited to the frozen workloads and environments listed in the report.

Do not claim:

* full Core native conformance;
* universal reproducible binaries across all platforms;
* general performance superiority;
* complete incremental compilation;
* production-grade optimisation comparable with mature native toolchains;
* LLVM is unnecessary forever;
* all targets are supported.

---

## 16. Immediate first task

Begin with WP-C7.0.

Before implementing release optimisation:

1. map the current build and host-compiler pipeline;
2. freeze the workload set;
3. build the stage-timing harness;
4. record current debug/release behaviour;
5. identify current host compiler flags;
6. run two clean builds in different checkout paths;
7. identify sources of output nondeterminism;
8. record the C7/C8 file lease map.

Commit the baseline separately.

Do not start by adding optimisation flags.
