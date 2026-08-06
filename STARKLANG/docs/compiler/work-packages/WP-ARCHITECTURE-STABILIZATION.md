# WP-ARCHITECTURE-STABILIZATION — Compiler architecture consolidation programme

**Status:** PROPOSED — approval is requested only for AS0 + AS1a; later packets require the AS0
report and a second owner decision.
**Date:** 2026-08-06.
**Owning track:** compiler, under `COMPILER-CHARTER.md` and `COMPILER-ROADMAP.md`.
**Roadmap relationship:** this is a proposed compiler work-package programme, not a second live
project roadmap. `ROADMAP.md` remains the only live platform plan. If the later campaigns are
approved after AS0, the integration gates in §4 must be added to that roadmap and the current
compiler position must be recorded in `COMPILER-STATE.md`.

---

## 1. Decision requested

Approve **AS0 + AS1a only** as the first bounded packet:

1. establish the baseline and exact inventories;
2. close the reproduced package source-identity/provenance defect;
3. report what the evidence says about the size, order and value of the remaining packets.

Do **not** approve Campaigns A and B wholesale yet. Their current scope and ordering are a proposal
to be resized after AS0. In particular, the proposed Campaign A gate would block the project
roadmap's structured-concurrency compiler/runtime work until the correctness foundations pass. That
platform impact requires an explicit second owner decision; it is not implied by approving AS0 +
AS1a.

The compiler's large-scale pipeline is retained:

```text
source/package
  -> AST
  -> resolved HIR
  -> type / flow / borrow / constant analysis
  -> typed-HIR execution
  -> monomorphised MIR
  -> optimisation
  -> MIR verification
  -> MIR execution or generated Rust
  -> native executable + runtime/providers
```

This programme is not a rewrite. It consolidates the contracts connecting those stages so that one
semantic fact has one authoritative representation and every compiler entry point uses the same
pipeline.

### Approval boundary

| Scope | Decision now | Later integration gate if approved |
| --- | --- | --- |
| AS0 + AS1a | approve or reject now | none; this is a defect/inventory packet |
| remainder of Campaign A | reserve until the AS0 report | before structured-concurrency compiler/runtime work |
| Campaign B | reserve until Campaign A and C8 positions are known | before C10 release qualification |

No calendar estimate is attached before the inventories exist. Planning is expressed in bounded
packets, and each packet exits only on its evidence. `ROADMAP.md` §2.2's work-in-progress limit
remains binding: only one major compiler/runtime packet is active at once.

---

## 2. Why this programme exists

The architecture has a strong backbone: typed arena IRs, an executable HIR oracle, explicit MIR,
a verifier-gated backend boundary, generated-Rust native compilation, a separate runtime/provider
ABI, and multi-engine differential evidence.

The recurrent risk is **semantic-authority fragmentation**:

- a byte span does not carry the identity of its source;
- package roots can acquire both absolute and logical source names;
- explicit and implicit calls do not publish the same callable-selection metadata;
- type-to-runtime-value conformance is executable but cannot yet be enforced at every boundary;
- Copy, drop, reference-containment and related MIR properties have multiple implementations;
- CLI, package, test and tool entry points still assemble overlapping compiler pipelines;
- Core/tensor isolation is policy-tested but tensor knowledge remains embedded in central phases;
- JSON and version-surface rules are implemented in several partial authorities;
- the largest passes combine many responsibilities through large mutable contexts.

Fixing individual symptoms without consolidating their authority would preserve the defect class.

---

## 3. Programme rules

1. **No language redesign.** The accepted/rejected Core program set, ownership model, trap model,
   MIR semantics, ABI and backend remain unchanged unless separately approved through the charter.
2. **Behaviour before modularity.** Establish executable invariants and authority boundaries before
   splitting large files. Moving fragmented logic into smaller files does not consolidate it.
3. **One authority, independently challenged.** Producers and verifiers may remain independent
   consumers, but they must be tested against a declared semantic model rather than drifting local
   approximations.
4. **No generic extension framework.** C9's second-artifact evidence requirement remains binding.
   Tensor code may be quarantined behind existing internal boundaries; no public plugin/provider
   abstraction is authorised here.
5. **No full incremental compiler.** LSP work is limited to measured debounce, cancellation and
   cache ownership improvements. `COMPILER-CHARTER.md` §6 continues to defer full incrementality.
6. **No broad cleanup packet.** Each packet owns a named correctness or maintainability claim,
   adds evidence before deleting old mechanisms, and records adjacent findings as follow-ups.
7. **The live correctness stream continues.** Audit- or application-discovered soundness,
   double-destruction, over-acceptance and release-blocking defects are not queued behind this
   programme. A bounded defect repair may interrupt the active architecture packet. If the defect
   expands into a major compiler/runtime campaign, pause the architecture packet so the WIP limit
   still holds. At proposal time CD-383 is evidence that this lane is active, not historical.
8. **Shared-checkout discipline applies.** Each implementation packet declares its file ownership
   set and uses explicit-path staging if the owner requests a commit.

### Live-defect pre-emption rule

Before starting or resuming any packet, read the current `COMPILER-STATE.md` position and newest
audit entries. A newly reproduced memory-safety analogue, double destruction, ownership violation,
wrong-code result or accepted-invalid program with incorrect execution takes priority over roadmap
sequencing. The focused repair keeps its own DEV/CD evidence and does not get absorbed into a broad
architecture commit. Non-blocking adjacent findings remain follow-ups under the charter.

---

## 4. Dependency and integration map

```text
AS0 Baseline and reproductions
 |
 +--> AS1a Canonical package source identity and provenance
 |      |
 |      +--> AS2 One compiler driver
 |                |
 |                +--> AS1b SourceId-bearing spans
 |                          |
 |                          +--> AS3 Callable-use totality -> resume value-representation enforcement
 |
 +--> AS4 Semantic type-property authority

                  [CAMPAIGN A EXIT]
        required before structured-concurrency compiler/runtime work

C8 explicit gate exit/owner decision
 |
 +--> AS5 Protocol and version contracts
 |
 +--> AS8 post-C8 tooling scale work

Campaign A exit
 |
 +--> AS6 Core/extension quarantine
 |
 +--> AS7 Pass modularisation and compiler API boundary
 |
 +--> AS8 Engine independence, tooling scale and governance closure

                  [CAMPAIGN B EXIT]
                 required before C10
```

Only AS0 + AS1a are proposed for approval now. AS1a and AS4 may later be designed in parallel, but
not implemented concurrently. AS1b follows AS2 so SourceId is threaded through one pipeline rather
than several assemblies that AS2 would immediately delete. AS3 follows both; otherwise callable
metadata would be integrated into several pipelines independently. C8 must receive an explicit gate
decision before AS5 or AS8 takes on overlapping protocol/editor work.

---

## 5. Campaign A — correctness foundations

### AS0 — Baseline, reproduction and authority inventory

#### Claim

Every later packet begins from a reproducible defect or a complete authority inventory, not from a
module-size impression.

#### Work

- Reproduce package-root source duplication with a package whose entry is outside the invoking
  process's current directory.
- Reproduce the provenance half separately: after logical package naming, verify whether
  `build_source_map` classifies real package files as `Module { package: None }` while the phantom
  absolute entry is the only `Root` record.
- Build the same package in two absolute checkout locations and compare source maps, MIR dumps and
  native build keys.
- Inventory every parse/resolve/typecheck pipeline assembly by entry point.
- Inventory every explicit and implicit user-callable execution site.
- Adopt the predicate inventory required by
  `WP-C7.8-RB0-MIR-Type-Property-Authority.md` rather than creating a second list.
- Inventory JSON parsers, serializers and accepted deviations from RFC 8259.
- Record baselines for check time, native build time, LSP change latency, compiler binary size and
  dependency count.
- Execute the approved AS0 scope of `WP-ENGINE-INDEPENDENCE.md` rather than inventing a second
  shared-fate vocabulary. Its register, evidence audit and engine-risk profiles are AS0 outputs;
  its rustc inventory and mutation recommendations feed AS5/C10 and AS8 respectively.
- Record and run the external `stark-samples` qualification suite, when available, pinned by commit
  hash and expected manifest. Treat it as independent application evidence, not as a normative
  source or an unversioned dependency on `~/Code/stark-samples`.

#### Exit criteria

1. The duplicate-identity and wrong-provenance halves are each reproduced or closed with contrary
   executable evidence.
2. The driver, callable, predicate and JSON inventories are exact-set checked where practical.
3. Performance commands and raw results are recorded and repeatable.
4. The pinned samples-suite result is recorded, or its absence is explicit rather than silently
   reducing the independent evidence set.
5. Each later packet has a bounded ownership set and an identified rollback point.

#### Stop condition

If relocation already produces one logical root identity and an invariant build key, close that
finding and do not manufacture an AS1 root fix. SourceId work remains independently justified by
`WP-SPAN-SOURCEID.md`.

---

### AS1a — Canonical package source identity and provenance

#### Dependencies

- AS0 reproduction of the duplicate-identity and wrong-provenance halves.

#### Work

- Give each physical source exactly one logical compiler identity.
- Keep canonical disk paths as loading metadata, never as source names.
- Make root/module/package provenance explicit rather than inferring it by comparing a logical
  source name with an absolute package-entry parent.
- Remove absolute checkout paths from MIR/build-key identity unless deliberately included in
  non-reproducible debug metadata.
- Use one helper for package entry `SourceFile` construction at every current call site; AS2 later
  makes the whole pipeline singular.

#### Exit criteria

1. One physical package root produces one `SourceRecord`.
2. The logical entry is the sole `Root`; every package module carries the correct non-empty package
   provenance.
3. Relocating identical source/package graphs preserves logical source maps, MIR dumps and build
   keys in two consecutive runs.
4. No canonical absolute checkout path participates in reproducible source identity.
5. Package, package-with-overlay and native-build paths share the same logical-entry helper and
   focused regression.

#### Risks and escalation

AS1a is deliberately narrower than SourceId-bearing spans. Any proposed MIR debug-contract change
beyond removal of accidental absolute identity is a CE3 decision and is excluded from AS1a.

---

### AS2 — One compiler session and one pipeline

#### Claim

All tools observe the same package loading, overlays, language options, resolution, checking,
diagnostic and source-identity behaviour.

#### Work

- Define one internal `CompilerSession`/driver facade with explicit operations such as:

  ```text
  analyze
  check
  execute_hir
  lower_mir
  execute_mir
  build_native
  query
  ```

- Make package loading, provider overlays, language options, source maps and diagnostic collection
  session-owned inputs.
- Migrate `starkc check/run`, `stark check/run/test/build`, documentation example validation,
  deployment analysis and LSP package analysis.
- Keep command-line parsing and presentation outside the driver.
- Remove the superseded manual parse -> resolve -> typecheck assemblies only after an exact-set
  entry-point test proves migration completeness.

#### Exit criteria

1. A repository search finds no production entry point independently assembling the semantic
   pipeline outside the driver.
2. The same invalid package produces the same ordered diagnostic structures through compiler CLI,
   package CLI, test runner and LSP analysis.
3. Core/tensor language options remain per-session under sequential and parallel analysis.
4. Provider-backed packages use the same analysis result for checking and native building.
5. Existing unit, integration, fixture, package and differential suites remain green.

#### Non-goal

This is not an incremental query engine and does not introduce persistent compiler state between
commands.

---

### AS1b — SourceId-bearing spans

#### Dependencies

- AS2 shared compiler session/driver.
- Existing `WP-SPAN-SOURCEID.md`, which remains the normative implementation packet.

#### Work

- Execute `WP-SPAN-SOURCEID.md` through the single AS2 pipeline.
- Route compile-time diagnostics and runtime trap locations through `SourceMap`.
- Remove ambient-file guesses and the interim wrong-source detector after total resolution exists.
- Prove that CLI, package, test, documentation and LSP consumers obtain the same SourceId-bearing
  diagnostics from the shared analysis result rather than adding per-entry-point plumbing.

#### Exit criteria

1. Dependency diagnostics and runtime traps resolve against the dependency's file and line table.
2. No AST/HIR/MIR/query diagnostic path accepts a bare byte range without source identity.
3. Span-to-location resolution is total through `SourceMap` in compile-time and runtime paths.
4. Existing diagnostic JSON remains deterministic.
5. Superseded diagnostic ambient-file guessing is removed only after exact-set migration evidence
   exists; item-to-file metadata used for separate module semantics is retained or removed on its
   own demonstrated purpose.

#### Risks and escalation

Span representation is foundational but does not change language semantics. Any proposed MIR
debug-contract change beyond source identity is a CE3 decision and is excluded from AS1b.

---

### AS3 — Total callable-use metadata and oracle representation enforcement

#### Dependencies

- AS2 shared session/driver and AS1b source-aware semantic metadata path.
- Existing `WP-VALUE-REP-TOTAL.md` A0–A3c work.

#### Work

1. Author and approve `WP-CALLABLE-USE-TOTAL` before implementation.
2. Publish exactly one checker-selected `CallableUse` for every accepted explicit or implicit
   user-callable invocation, including:
   - selected callable identity;
   - explicit empty or populated generic environment;
   - receiver adjustment and binding mode;
   - argument and result types;
   - dispatch provenance, including compiler-known trait operations.
3. Make HIR execution and MIR lowering consume `CallableUse`; neither may reconstruct selection.
4. Add exact-set coverage across free calls, methods, associated functions, function values,
   trait defaults, qualified calls, equality, ordering, iteration and display.
5. Resume A4 of `WP-VALUE-REP-TOTAL` only after callable-use exactness passes.
6. Inventory and close the separately identified typed-mutation boundaries before closing the
   DEV-121 defect class.

#### Exit criteria

1. Every executable user-callable use has exactly one record; duplicates and omissions fail an
   invariant test.
2. Implicit and explicit dispatch install the checker-selected generic environment in the HIR
   oracle.
3. The total type-to-`Value` relation is enforced at parameters, returns, receiver boundaries,
   bindings and typed mutation without exemptions.
4. The frozen corpus and all engine comparisons remain green.
5. DEV-121 closes only with a class-level evidence statement, not one regression case.

#### Risks and escalation

Callable metadata is a semantic compiler contract. If the work changes overload selection, trait
semantics or the accepted/rejected program set, stop and use CE1/CE2 rather than folding the change
into this packet.

---

### AS4 — One authority for semantic type properties

#### Dependency

Execute the existing `WP-C7.8-RB0-MIR-Type-Property-Authority.md`; do not replace it with a fresh
cleanup design.

#### Work

- Complete the required inventory for:
  - Copy classification;
  - runtime drop glue;
  - user-defined destruction;
  - stored-reference containment;
  - borrow-lifetime carrying;
  - user-nominal containment;
  - runtime representation.
- Distinguish differently worded semantic questions before consolidating implementations.
- Add equivalence/adversarial tests over the full type-variant set before deleting duplicates.
- Give lowering and backends one semantic authority surface.
- Preserve verifier challenge value: either use an independently implemented verifier predicate
  checked against the same declared matrix, or justify direct consumption where independence adds
  no evidence.
- Resolve or explicitly carry the iterator drop and function-pointer reference questions named by
  the existing packet.

#### Exit criteria

1. Every type property has one documented meaning and authority.
2. Near-neighbour predicates with different meanings are named so they cannot be substituted
   accidentally.
3. Adding a type/representation variant forces every applicable authority and evidence matrix to
   be updated.
4. Resource, iterator, reference, generic-drop and partial-move adversaries pass across HIR, MIR
   and native engines.
5. Any behavioural correction receives its own decision record; AS4 itself does not disguise one
   as refactoring.

#### Campaign A exit gate

Campaign A passes only when AS0, AS1a, AS2, AS1b, AS3 and AS4 are complete and owner-reviewed. The
exit report must classify each criterion PASS, FAIL, DEFERRED-BY-DECISION or NOT-APPLICABLE and
include command-level evidence.

**Reserved project-roadmap decision, not approved by AS0 + AS1a:** after the AS0 report, the owner
must decide whether to amend the project roadmap so structured-concurrency compiler/runtime
implementation may not begin until Campaign A passes. Package work not dependent on new compiler
semantics may continue under `ROADMAP.md`'s WIP limits.

---

## 6. Campaign B — maintainability and release readiness

### AS5 — Protocol, manifest and version-surface contracts

#### Dependencies

- C8 is CANDIDATE-COMPLETE, not closed, at proposal time; it receives an explicit owner gate
  decision before AS5 begins.
- EI3 of `WP-ENGINE-INDEPENDENCE.md` supplies the rustc/toolchain assumption inventory; AS5 decides
  how that proposal integrates with version and build-provenance contracts.
- If C8 remains open, malformed/trailing JSON-RPC handling and real editor validation belong to
  WP-C8.7/C8 exit first. AS5 consumes that evidence; it does not silently reopen or duplicate C8.
- If C8 closes first, AS5 preserves its protocol/interactive baseline while consolidating the
  shared JSON implementation used by non-LSP surfaces.

#### Work

- Choose one strict JSON authority for package manifests, JSON-RPC/LSP, install manifests and
  compiler-generated JSON:
  - preferably a vetted library with dependency/risk review; or
  - one internal parser/serializer with RFC 8259 conformance tests.
- Reject trailing input and malformed escapes deterministically.
- Handle Unicode escapes and surrogate pairs correctly where JSON strings are accepted.
- Escape every required control character in every generated JSON surface.
- Preserve protocol-specific data models above the shared JSON layer.
- Replace manually remembered MIR/runtime-surface version bumps with a deterministic schema
  fingerprint or an exact-set test tied to the canonical surface.
- Add compatibility fixtures for old/new manifests and machine-readable diagnostics.

#### Exit criteria

1. Production code contains one JSON parser and one escaping authority.
2. A standard JSON test corpus and project-specific malformed cases pass.
3. C8's LSP protocol baseline proves rejection of trailing garbage and valid JSON for every
   diagnostic string; AS5's shared authority keeps that evidence green.
4. A runtime/MIR surface change cannot compile or pass tests without updating its compatibility
   identity.
5. Security-sensitive parsing decisions receive CE9 review where applicable.

---

### AS6 — Quarantine extension-specific compiler knowledge

#### Dependencies

- Preserve the closed Part A behaviour of `WP-C9.1-EXTENSION-ISOLATION.md`.
- C9's second-artifact evidence gate remains closed unless independent evidence appears.

#### Work

- Inventory tensor/model/dtype/device branches in lexer/parser, resolver, checker, formatter,
  diagnostics and LSP.
- Move extension-owned names, type rules, methods and diagnostics behind sealed internal tensor
  modules/interfaces selected by the existing per-session `LanguageOptions`.
- Keep Core pass data structures extension-neutral where this can be done without a generic public
  abstraction.
- Add dependency/lint tests preventing new tensor imports in designated Core-only modules.
- Retain explicit frontend enablement and all C9.1 session-isolation tests.

#### Exit criteria

1. Core-only sessions load no tensor-owned name or semantic rule.
2. Central Core modules do not contain open-ended tensor spelling tables or method catalogues.
3. Tensor-enabled behaviour and ONNX verification remain unchanged for their documented scope.
4. No public extension/plugin/provider API is introduced.
5. Part B generic artifact-provider work remains blocked unless C9.3's independent evidence exists.

---

### AS7 — Pass modularisation and compiler API boundary

#### Dependencies

Campaign A plus AS5 and AS6 must establish the boundaries first. AS7 does not invent them while
moving code.

#### Work

- Split the type checker by semantic ownership: inference, traits/method selection, patterns,
  ownership/borrowing, callable publication and extension checking.
- Split MIR lowering by calls, patterns, drop planning, intrinsics and metadata construction.
- Split the HIR interpreter into value model, executor, callable dispatch and Core-library
  operations.
- Replace ambient current-file/module/impl/generic state with scoped context objects where a
  missing restore can alter later work.
- Define a narrow supported compiler facade; make implementation modules `pub(crate)` unless an
  actual external consumer requires them.
- Move obsolete backend spikes and Cranelift-only development dependencies out of default compiler
  builds, preserving historical evidence in documentation or a non-default spike crate if needed.

#### Exit criteria

1. No semantic behaviour or diagnostic structure changes in modularisation commits.
2. Dependency direction between submodules is documented and cycle-free.
3. Internal modules are not accidentally part of the supported public API.
4. Default dependency/build surfaces contain only active compiler architecture.
5. File-size reduction is reported as an outcome, not used as the acceptance criterion.

---

### AS8 — Independent evidence, tooling scale and governance closure

#### Dependencies

- Explicit C8 gate exit. AS8 is post-C8 performance/ownership work, not a substitute for C8's
  protocol and interactive semantic validation.

#### Work

- Consume the shared-fate register, evidence audit, engine-risk profiles and ranked mutation
  targets produced by `WP-ENGINE-INDEPENDENCE.md`; do not repeat its inventory under a second
  taxonomy.
- Add real compiler-source mutation trials for selected ownership, trap, drop, resolver and MIR
  verifier rules; observation/comparator mutation alone is insufficient.
- Establish line/branch coverage baselines for compiler crates and report uncovered semantic
  arms—without imposing an arbitrary percentage as a conformance claim.
- Run the external `stark-samples` suite as pinned independent application evidence. Record the
  suite commit and expectation manifest with the result; if it becomes a required CI gate, vendor
  or fetch an explicitly versioned artifact rather than depending on a developer home path.
- Profile LSP package analysis on representative multi-file projects.
- If evidence warrants it, add bounded debounce, cancellation and one-analysis-per-package cache
  ownership. Do not build full incrementality.
- Replace whole-package `ProjectAnalysis` duplication per open URI where measurement shows material
  cost.
- Compress `COMPILER-STATE.md` back toward the charter's current-state contract while preserving
  append-only history in an archive/ledger, and reconcile deviation statuses with executable
  evidence.
- Update `compiler-map.md`, `lib.rs` crate documentation and the canonical roadmaps at the campaign
  exit.

#### Exit criteria

1. Each differential claim names shared phases and at least one independent evidence source.
2. Selected source mutations are killed by the claimed suites; survivors are recorded as test
   gaps.
3. LSP changes are justified by before/after measurements and cancellation correctness tests.
4. Current compiler position is discoverable from the beginning of `COMPILER-STATE.md` without
   reconstructing chronology.
5. Architecture documentation matches production entry points and module ownership.

#### Campaign B exit gate

Campaign B passes only when AS5–AS8 are complete or explicitly deferred with owner-approved
evidence. Its report is a prerequisite for C10 release qualification, but it does not itself make a
stability or conformance claim.

---

## 7. Required evidence for every packet

Unless a packet states stricter requirements, its closeout includes:

- `cargo fmt --check`;
- `cargo clippy --all-targets -- -D warnings`;
- scoped local Rust tests selected from the packet's ownership and risk surface;
- the full Rust suite through CI, or from an isolated clean worktree when explicitly required—not
  by running a broad shared-checkout command as an unrecorded substitute for CI;
- Core positive and negative fixture conformance;
- HIR/MIR/native debug/native release differential rows for affected semantics;
- tensor/extension tests when extension code is touched;
- deterministic outputs executed twice when identity, ordering or generated output is claimed;
- package/provider qualification when package loading, capabilities, runtime or build metadata is
  touched;
- the pinned external samples suite for packets affecting accepted programs, ownership, execution,
  packages or engine agreement, when that suite is available;
- focused tests demonstrated to fail before the repair when the packet closes a defect;
- updated deviations, coverage records, `COMPILER-STATE.md` and architecture documentation.

Provider-backed packages are built, not run through the interpreter. Provider crates receive their
own `--manifest-path` build/test rows where their sources are touched.

---

## 8. Programme success measures

The architecture-stabilisation programme succeeds when all of the following are demonstrable:

| Property | Measure |
| --- | --- |
| Source identity | one physical source has one logical identity; all spans resolve through `SourceId` |
| Relocation stability | identical package graphs at different roots produce identical logical MIR/build identities |
| Entry-point convergence | every tool reaches semantic analysis through one driver |
| Callable authority | every executable call has exactly one checker-published `CallableUse` |
| Runtime representation | every typed HIR boundary enforces the total `Ty`→`Value` relation |
| Type properties | Copy/drop/reference questions have documented, exact-set-tested authorities |
| Extension isolation | Core modules do not embed tensor-owned catalogues; no premature public framework exists |
| Protocol correctness | one strict JSON authority and mechanically checked compatibility surfaces |
| Maintainability | major passes have explicit ownership boundaries and a narrow public facade |
| Evidence independence | differential results state shared fate and are challenged by source mutation or independent fixtures |
| Tooling scale | LSP latency/cancellation/cache behaviour is measured and bounded without premature incrementality |
| Governance | current status, deviations and architecture documents agree with executable evidence |

No single metric, test count or green differential run is sufficient. The exit claim is that the
existing compiler architecture has stable, authoritative contracts—not that the language or public
toolchain has reached v1 stability.

---

## 9. Explicit non-goals

This programme does not authorise:

- new Core syntax or semantics;
- async/await, concurrency semantics or an HTTP server;
- a VM, JIT, LLVM migration or direct Cranelift backend;
- a new MIR, runtime ABI or value layout;
- a public compiler-plugin or generic artifact-provider framework;
- tensor productisation or broader tensor execution;
- full incremental compilation;
- compiler self-hosting;
- a Core conformance, stable compiler or public release claim.

Those remain governed by the specification, charter, C9/C10 gates and the consolidated project
roadmap.
