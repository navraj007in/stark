# STARK Ecosystem Build Brief — Revised Sonnet Edition

<!-- MASTER BRIEF: use this to bootstrap and govern the ecosystem track. -->
<!-- Do NOT feed this entire roadmap at the start of every later session. -->
<!-- After bootstrap, feed only ECOSYSTEM-CHARTER.md + STATE.md + the active WP. -->

**You are:** Claude Sonnet, acting as the implementation engineer for the STARK ecosystem track.
You work in bounded work packages, preserve compatibility deliberately, keep a shared state file current, and escalate the small number of decisions listed in this brief.

This document is the **master roadmap**, not the routine session payload.
On the first session, execute Gate E0 bootstrap and split the durable guidance into:

```text
docs/ecosystem/ECOSYSTEM-CHARTER.md
    north star, layers, governing rules, escalation rules, not-yet list

docs/ecosystem/ECOSYSTEM-ROADMAP.md
    gates, work packages, exit criteria, dependencies

STATE.md
    current position, decisions, compatibility choices, follow-ups, gate exits

docs/ecosystem/work-packages/WP-<id>.md
    the exact active work package and its acceptance checks
```

For every later work session, the intended input packet is:

```text
ECOSYSTEM-CHARTER.md
+ STATE.md
+ the active WP file
+ only the specs/source files directly referenced by that WP
```

Load the full roadmap again only when closing a gate, selecting the next gate, or resolving a roadmap-level contradiction.
This reduces scope leakage from later, more interesting features into earlier experiments.

---

# 1. PROJECT CONTEXT — FIXED, DO NOT RE-DERIVE

**STARK** is an ownership-safe, Rust-inspired research programming language.
The repository already contains a Core v1 specification, a Rust compiler/interpreter, tensor/ONNX tooling, a prototype package system, a formatter, a test runner, an LSP foundation, and a VS Code extension.

Treat that list as a **starting hypothesis, not a conformance claim**.
Gate E0 must inspect the current repository and establish what is implemented, partial, stale, incompatible, or only documented.
Do not rebuild an existing subsystem merely because this roadmap describes it.

The language-core track and ecosystem track are separate:

- The **Core track** owns syntax, semantics, ownership, borrowing, traits, effects, and type-system rules.
- The **ecosystem track** owns packages, modules, standard packages, native providers, tooling, artifact importers, distribution, and eventually a registry.
- This ecosystem roadmap must not quietly add Core syntax or semantics.

## 1.1 North-star technical thesis

> **STARK investigates artifact-bound programming: application code and the external artifacts it depends on are verified as one typed system rather than through disconnected tools.**

The current ONNX work is the first artifact class.
The ecosystem track must test a second artifact class early and independently, before making that test depend on a complete networking stack.

The public-facing formulation is:

> **STARK is a research language in which external artifacts—models, schemas, APIs, and deployment plans—can participate in whole-program verification, allowing contracts to survive from source code through deployment.**

## 1.2 Keep three research claims separate

Never present evidence for one claim as proof of another:

1. **Artifact-binding claim:** external artifacts can participate usefully in whole-program verification.
2. **AI-development claim:** AI materially changes the economics and process of language design and implementation.
3. **Agent-safety claim:** strong types, capabilities, effects, and artifact contracts can make AI-generated code safer to accept.

This ecosystem track primarily tests claim 1.
Claims 2 and 3 require their own evidence and are not acceptance criteria for ordinary package work.

## 1.3 The five layers — never blur them

| Layer | Contents | Constraints |
|---|---|---|
| L0 Core + Prelude | primitives, `Option`/`Result`, `String`/`Vec`/`Box`, ownership operations, foundational traits and compiler-coupled contracts | small; no networking or OS policy; ecosystem gaps do not become language features |
| L1 Pure standard modules | collections, conversion, formatting, iteration, memory helpers, result/option utilities | written in STARK where the language can express them; shipped with the toolchain; no privileged native access |
| L2 Versioned standard packages | I/O, filesystem, paths, environment, time, logging, testing, JSON, URL, networking, TLS, HTTP | ordinary packages; may use approved native providers internally; public APIs remain STARK-shaped |
| L3 Optional first-party packages | CSV, Base64, UUID, regex, OpenAPI, image, ONNX and similar domain packages | independent releases; prove the same authoring model expected of the community |
| L4 Third-party packages | database clients, cloud SDKs, JWT, Redis, application frameworks and other community work | require zero compiler modification, private hooks, or privileged install behaviour |

## 1.4 Governing rules — violating any is a wrong implementation

> **Package-system invariants:** Package acquisition never executes code. A build resolves
> one visible identity per `(source, name, major)` slot. Every package byte is
> content-verified. For pure STARK packages, host capabilities are derived from the resolved
> graph rather than trusted from declarations. Every native component is a disclosed,
> consented trust tier.

1. **Do not enlarge Core to compensate for missing ecosystem functionality.**
2. **Do not introduce new syntax inside an ecosystem WP.** Any syntax proposal is a separate Core-track proposal with evidence, alternatives, migration impact, and owner approval.
3. **Use existing STARK language constructs first.** Missing convenience is not proof that new syntax is necessary.
4. **Keep high-level logic in STARK.** Reuse mature native Rust libraries only for OS, sockets, TLS, cryptography, clocks, and other host capabilities that STARK cannot safely implement itself.
5. **Keep public APIs STARK-shaped.** Underlying Rust crate names, Rust error types, lifetimes, traits, or implementation terminology must not leak into STARK identifiers, diagnostics, or user documentation.
6. **Use blocking I/O only in this roadmap.** Async/concurrency remains Gate P11: a future, evidence-gated Core proposal.
7. **No package build scripts, install scripts, compiler plugins, network access during compilation, or code execution during dependency resolution.**
8. **Every dependency source must be lockable and reproducible.** Path, git revision, and future registry sources must map to stable lockfile identity and content verification.
9. **Native code is a separate trust tier and must be disclosed in package metadata.** A native-backed package discloses its provider identity, artifacts, hashes, targets, and declared capabilities, and requires explicit root-application consent. Disclosure is metadata, not a proof of runtime behaviour.
10. **Preserve compatibility deliberately.** Existing manifest, lockfile, package-name, CLI, and cache formats must be audited before being changed. Breaking changes require explicit new format versions and migration fixtures.
11. **Isolate experimental hypotheses.** A gate testing artifact identity must not quietly include capabilities, taint tracking, effects, idempotency, or other later mechanisms.
12. **Do not let later mechanisms rescue a failed earlier experiment.** If a narrow artifact-binding gate cannot beat its comparator, record the negative result before adding richer semantics.
13. **Registry infrastructure comes only after external package authors demonstrate repeatable value.**
14. **Contract-aware packages are the differentiator.** Plain protocol wrappers are necessary ecosystem infrastructure, not the research payoff.
15. **External usability evidence begins early.** Small outside walkthroughs occur after major authoring gates; do not postpone all human feedback until the registry phase.
16. **Deterministic tie-breaking applies only to non-semantic choices.** Alphabetical or numerical tie-breaking is acceptable for ordering, test fixtures, and filenames—not for public API design, security, compatibility, language semantics, or user-facing defaults.
17. **Documentation must distinguish implemented behaviour from wiring, stubs, plans, and known limitations.**
18. **Canonical package identity is `source + name + version + content hash`.** A package name alone never identifies a package. Resolution selects at most one version per `(source, name, major-version line)` slot; semver-incompatible majors may coexist but must be visible in the lockfile, dependency tree, update summaries, and diagnostics. Applications may deny multi-major coexistence globally or per package name. Silent duplication, silent source substitution, and silent registry fallback are all wrong implementations.
19. **For pure STARK packages, host capabilities are derived, not trusted.** The toolchain computes each package's capability set from references to host-backed interfaces in the resolved graph (reference-level, conservative). A manifest may declare a capability envelope; the build fails if `derived ⊄ declared`. The root application approves the derived transitive closure. Capability metadata is never merely informational once the derivation mechanism (WP-P1.6) lands.
20. **Specify durable formats early; defer registry operations.** Anything encoded into package identity, lockfiles, archives, or capability metadata is decided in P0–P1 under escalations E1/E2/E7. Operational registry machinery (publisher identity, MFA, typosquat defense, advisories, quarantine) is designed only at Gate P10.

## 1.5 Explicit Core-scope prohibitions

The ecosystem track must not silently add any of the following:

```text
pub(package)
@test or #[test]
native resource syntax
async / await
effect syntax
capability syntax
information-flow labels
macros or compiler plugins
```

Current ecosystem conventions must use existing language surface:

- package-visible implementation details remain private;
- externally consumable items use the existing `pub` mechanism;
- tests use the existing `test_*` and `test_ignored_*` naming convention;
- native handles are represented through the approved provider ABI and ordinary/opaque STARK types;
- capabilities, effects, labels, and typestate remain separate research proposals until multiple independent domains demonstrate the same need. Package-level capability *metadata and derivation* (rule 19, WP-P1.6) are package-tooling analysis over existing imports and add no language surface; the prohibition here is on Core *syntax and semantics* for capabilities, which remain a separate research proposal.

If an existing construct cannot support a required property, write an escalation note showing:

1. the exact blocked use case;
2. two non-Core alternatives attempted;
3. why those alternatives fail;
4. the minimum Core mechanism proposed;
5. compatibility and implementation cost.

Do not implement the proposal inside the ecosystem WP.

## 1.6 Terminology

- **module** — a namespace/source unit inside one package; resolved at compile time; not independently versioned.
- **package** — a versioned compilation and distribution unit described by a manifest; may contain many modules and depend on other packages.
- **workspace** — a group of packages developed and locked together under one workspace root.
- **native provider** — an approved Rust implementation behind a typed STARK package boundary for host capabilities such as filesystem, TLS, clocks, or sockets.
- **artifact provider** — compiler/tooling infrastructure that reads an external artifact, normalises it, extracts a typed contract, records identity/provenance, generates or registers declarations, verifies drift, and contributes diagnostics.
- **registry** — a future package index and immutable archive service. The current filesystem registry is a resolver prototype, not a public registry.
- **contract-only importer** — an artifact tool that imports and verifies a contract without requiring runtime execution against the external system.
- **executable client** — a generated package that uses the imported contract at runtime through standard networking/serialization packages.

## 1.7 Repository layout — target, subject to Gate E0 compatibility audit

```text
stark/            compiler, Core spec, package resolver, tools, editor integrations
stark-std/        first-party standard packages, one ordinary package per directory
stark-packages/   optional first-party packages and the external-author template
```

Monorepos may coordinate changes, but each package must remain independently manifest-described, versioned, testable, documentable, and consumable through normal package resolution.

## 1.8 Conventions

- JSON: 2-space indentation and deterministic key ordering where the format permits it.
- Normative specifications: Markdown, ATX headings, one normative sentence per line where practical, and stable rule identifiers.
- Diagnostics: allocated from the central registry in this brief and recorded in `STATE.md`.
- Generated artifacts: stable ordering, stable line endings, deterministic headers, no wall-clock timestamps unless explicitly excluded from identity.
- Claims: use `implemented`, `partial`, `stub`, `planned`, or `deferred` precisely.

---

# 2. OPERATING PROTOCOL

## 2.1 Session flow

1. Load `ECOSYSTEM-CHARTER.md`, `STATE.md`, and the active WP file.
2. Read only the source/spec files directly needed by the active WP.
3. Confirm the WP scope against `STATE.md`.
4. Execute only that WP.
5. Record adjacent problems as `FOLLOW-UP:` entries rather than solving them opportunistically.
6. Verify every `Done when` check.
7. Run deterministic operations twice when determinism is claimed.
8. Update `STATE.md` with decisions, evidence, files, follow-ups, and the next WP.
9. Commit with message `[WP-<id>] <summary>`.

At a gate exit, reload `ECOSYSTEM-ROADMAP.md`, evaluate every criterion, obtain owner review, and only then select the next gate.

## 2.2 Scope and Sonnet-level autonomy

You may:

- complete a bounded WP spanning specification, implementation, tests, CLI documentation, and migration fixtures;
- split a WP when it is too large, provided the split is recorded before implementation and each part has independent acceptance checks;
- refactor files already in scope when the refactor directly supports the WP;
- make medium-complexity resolver, package-manager, tooling, and standard-package changes with test coverage;
- choose deterministic non-semantic implementation details and record them.

You may not, without escalation:

- change Core syntax;
- change ownership or borrowing semantics;
- change trait coherence or type-system rules;
- design the native-provider ABI;
- make security-sensitive decisions about hashes, package signing, TLS, cryptography, or trust roots;
- redefine an existing manifest/lockfile/cache format without an explicit compatibility decision;
- reverse a decision already recorded in `STATE.md`;
- begin any item in the not-yet list;
- merge a WP whose acceptance checks or external comparator are incomplete.

## 2.3 Escalations

| ID | Decision | Reason |
|---|---|---|
| E1 | Existing manifest/lockfile/cache compatibility and versioning decision | prevents the roadmap from retroactively redefining already-shipped prototype formats |
| E2 | Lockfile content hash, canonicalisation, and cache identity | reproducibility and supply-chain foundation |
| E3 | Native-provider boundary ABI | wrong design would force every native package to be rewritten |
| E4 | TLS implementation and public configuration surface | security-critical; must rely on an audited stack |
| E5 | Any proposed Core syntax/semantic mechanism | ecosystem work cannot approve its own language expansion |
| E6 | Registry trust model and minimal design | publishing identity, immutability, yank behaviour, provenance, and index signing |
| E7 | Capability vocabulary, derivation contract, and enforcement semantics | becomes a durable serialized format in manifests and lockfiles; enforcement changes what builds are rejected |

The early OpenAPI contract-only experiment requires owner review of its hypothesis and comparator, but it does not require a new Core mechanism.

## 2.4 `STATE.md` shared memory

Bootstrap template:

```markdown
# STARK Ecosystem STATE
Updated: <date> after <WP-id>

## Position
Gate: <E0|P0|P1|...>  Next: WP-<id>  Blocked: <none|reason>

## Current repository baseline
- Compiler head: <sha>
- Existing package manifest version/shape: <summary>
- Existing lockfile/cache shape: <summary>
- Existing CLI/tooling relevant to current gate: <summary>

## Decision log — append-only
- D-001 [WP-E0.1] <decision and rationale>

## Compatibility decisions
- C-001 <format/API> — preserve | additive evolution | new version + migration

## Diagnostic codes allocated
- PKG-01xx ...

## File inventory
- <path> — <purpose>

## External feedback checkpoints
- <gate> — <participant/context/result or pending>

## Follow-ups
- [ ] <bounded future item>

## Gate exit summaries
- <gate>: <result and evidence>
```

Every session appends:

```markdown
### WP-<id> — <date>
DONE: <1–3 sentences>
EVIDENCE: <tests/commands/comparator outputs>
FILES: <paths>
DECISIONS: <D/C identifiers or none>
FOLLOW-UP: <items or none>
NEXT: WP-<id>
```

Rules:

- The decision log is append-only.
- Compatibility reversals require a new decision referencing the superseded one; never rewrite history.
- Keep current details concise and compress closed gates into one evidence-backed paragraph.
- Do not use `STATE.md` as a substitute for normative specifications.

## 2.5 Definition-of-done defaults

Unless a WP states stricter conditions:

- Code compiles with zero warnings.
- Existing tests remain green.
- New public functions have at least one success and one error-path test where an error path exists.
- Deterministic outputs are generated twice and compared byte-for-byte.
- Normative rules have stable identifiers and valid/invalid examples.
- CLI docs include synopsis, one runnable example, output, and exit codes.
- Compatibility-affecting changes include migration fixtures or a documented no-migration decision approved through E1.
- Security-sensitive code has no bespoke cryptography or unaudited protocol implementation.
- Generated code re-parses and type-checks.
- Claims in README/changelog/status documents match the actual implementation.
- `STATE.md` is updated.
- Nothing is merged with failing checks.

## 2.6 Experimental discipline

Every research gate must define before implementation:

1. **Hypothesis** — one primary mechanism being tested.
2. **Excluded mechanisms** — attractive later ideas deliberately out of scope.
3. **Strong comparator** — the realistic established stack, not a weak baseline.
4. **Defect corpus** — valid and invalid cases selected before results are known.
5. **Metrics** — defects caught, detection stage, duplicated declarations, glue/configuration, diagnostics, false positives, and maintenance under change.
6. **Decision rule** — continue, revise, narrow, or stop.

Do not add a later mechanism because the first result is weak.
A negative result is valid evidence.

## 2.7 External usability checkpoints

Small external walkthroughs are required before the full P9 pilot:

- after P1: consume and inspect a multi-package workspace;
- after P2: create, test, document, and package a small library;
- after A1: run contract import and drift verification on a real API specification;
- after P3: consume a git dependency from a clean checkout;
- after P4: attempt a minimal native provider from the authoring guide.

One external participant is enough for these checkpoints.
Record friction and blockers in `STATE.md`.
Do not call them adoption evidence; they are early usability evidence.

---

# 3. GATES AND WORK PACKAGES

Gates close in order unless this roadmap explicitly allows parallel documentation work.
A failed exit criterion blocks progression or produces an explicit `REVISE` decision.

---

# GATE E0 — CURRENT-STATE AUDIT AND ROADMAP BOOTSTRAP

**Outcome:** the roadmap is grounded in the repository that exists today, and later work evolves rather than accidentally replaces it.

## WP-E0.0 — Bootstrap the session system

Create:

```text
docs/ecosystem/ECOSYSTEM-CHARTER.md
docs/ecosystem/ECOSYSTEM-ROADMAP.md
docs/ecosystem/work-packages/
STATE.md
```

The charter contains Sections 1, 2, and the not-yet list.
The roadmap contains the gate structure and dependencies.
Create the active `WP-E0.1.md` packet.

**Done when:** a fresh Sonnet session given only charter + state + active WP can state the current scope, prohibitions, deliverables, and exit checks without the master brief.

## WP-E0.1 — Repository implementation inventory

Audit the current `main` branch and create `docs/ecosystem/current-state-audit.md` covering:

- manifest parser and current schema;
- lockfile format and cache identity;
- path, registry-prototype, version-range, locked, and offline behaviour;
- multi-file modules and package resolution;
- `stark` CLI commands;
- formatter and test runner;
- native/FFI mechanisms, if any;
- standard-library built-ins versus ordinary packages;
- ONNX import/verify/deploy artifact flow;
- LSP and VS Code features, explicitly separating real semantics from stubs;
- conformance/status documents that disagree with implementation.

For each area classify:

```text
implemented | partial | stub | documented-only | stale-status | absent
```

Record the reviewed commit SHA.

**Done when:** every P0–P4 assumption in this roadmap maps to current code evidence or is marked absent.

## WP-E0.2 — Compatibility matrix and format decisions

Create `docs/ecosystem/compatibility-matrix.md`:

| Area | Existing form | Proposed target | Decision | Migration/test |
|---|---|---|---|---|
| manifest | current `starkpkg.json` fields | target package contract | preserve/additive/new version | fixture |
| lockfile | current format | target deterministic format | preserve/additive/new version | fixture |
| package naming | current examples and resolver rules | target naming rules | preserve/transition | fixture |
| version requirements | current exact/range support | target policy | retain/restrict with reason | tests |
| entry files | current `entry` behaviour | app/library conventions | compatible mapping | tests |
| registry prototype | current filesystem layout | future registry/archive identity | compatibility boundary | tests |
| cache | current path/hash layout | content-addressed target | migration/invalidated | tests |

Rules:

- Never relabel an existing format as “v1” if its shape is being replaced.
- A breaking manifest or lockfile design becomes version 2 or another explicit new version.
- Existing range support must not be removed merely because an early draft preferred exact versions.
- Package names and aliases must account for current underscore/hyphen behaviour and generated identifiers.

This WP requires escalation E1.

## WP-E0.3 — Reconcile status and known defects

Update or file follow-ups for stale conformance and status claims.
At minimum verify whether the repository still has known issues in:

- extension gating of unqualified built-ins;
- runtime equality versus trait semantics;
- package-level extension configuration;
- LSP diagnostic publication and semantic navigation;
- README claims versus server stubs;
- standard-library entries marked missing after implementation.

Do not solve unrelated Core defects in this WP.
Record them with owner/routing labels.

## WP-E0.4 — Tailor and close E0

Create `docs/ecosystem/E0-exit-report.md` with:

1. repository baseline established;
2. compatibility decisions approved;
3. roadmap WPs marked `already satisfied`, `partially satisfied`, or `new work`;
4. current known blockers routed;
5. session packet process verified.

All pass → Gate P0.

---

# GATE P0 — PACKAGE AND MODULE CONTRACT

**Outcome:** an implementation-independent package contract aligned with existing Core syntax and explicit compatibility decisions.

## WP-P0.1 — Naming, visibility, and entrypoints

Deliver:

```text
spec/packages/naming.md
spec/packages/visibility.md
spec/packages/entrypoints.md
```

Requirements:

- Naming rules must be selected through the E0 compatibility matrix.
- Module paths use existing Core identifiers and existing module syntax.
- Maximum depth/path rules are package-tooling constraints, not new language grammar.
- Visibility uses only existing constructs:
  - private by default;
  - `pub` for the externally reachable package surface;
  - no `pub(package)` syntax.
- Re-export rules use only syntax already implemented and specified.
- A public API must not expose a private type in its externally visible signature; implement this as package/API validation only after confirming where it belongs in the existing compiler architecture.
- Entrypoint forms must match the current Core specification and interpreter/compiler capabilities.
- Do not invent a new `main` return convention if the existing Core does not support it.
- Applications cannot be dependencies unless the compatibility decision explicitly supports that behaviour.

Allocate `PKG-01xx` visibility/API-surface diagnostics.

**Done when:** every normative rule has a stable ID; valid and invalid examples compile or fail as documented against the current language.

## WP-P0.2 — Sources and manifest contract

Deliver:

```text
spec/packages/sources.md
spec/packages/manifest.md
spec/packages/schema/<manifest-version>.schema.json
```

Source kinds:

- relative path;
- git HTTPS URL plus immutable full commit revision;
- registry identity reserved for the later registry, following E0 compatibility decisions. Registry/source identity is a component of canonical package identity (rule 18) and is designed into the lockfile format at E2 time, before any registry exists.

Every source kind must map to lockfile identity and content verification.
No absolute path may appear in a portable package manifest or lockfile.

Manifest design must follow the approved E0 decision:

- preserve existing fields where compatible;
- add fields only with defined defaults and older-reader behaviour;
- use a new manifest version for breaking shape changes;
- reject unknown fields only if the selected version contract intentionally requires closed-world parsing;
- native disclosure, targets, and capabilities are metadata fields, not language features;
- capability metadata follows the envelope/derived contract of rule 19 and WP-P1.6 (E7); until WP-P1.6 lands, manifests may carry declared envelopes, and no build enforcement occurs;
- an optional `declared-capabilities` field (the envelope, capability-vocabulary-versioned) is part of the manifest design;
- explicit alias syntax for intentional multi-source or multi-major coexistence is part of the manifest design (exact field shape settled under the E1 process); aliases must be visible in both the manifest and the lockfile.

Include complete application and library examples plus migration examples from the current manifest shape.

## WP-P0.3 — Restrictions and native disclosure

Deliver:

```text
spec/packages/restrictions.md
spec/packages/native-disclosure.md
```

Normative restrictions:

- no package build scripts;
- no install scripts;
- no dependency-time execution;
- no compiler plugins;
- no network access during compilation;
- no mutable branch/tag dependency identity in a locked build;
- no undeclared native code;
- no implied safety claim from “reviewed” metadata;
- no silent dependency substitution (alias, fork, override, or source change invisible to manifest + lockfile);
- no capability entering the graph without appearing in the derived/declared record;
- a pure→native transition in any dependency release is a security-significant change and must surface in `stark lock diff` regardless of semver compatibility.

Each restriction includes rationale, diagnostic stage, and one invalid fixture.

## WP-P0.4 — Lockfile contract

Requires E2.

Deliver `spec/packages/lockfile.md` aligned with the E0 compatibility decision.
Define:

- format version;
- canonical package identity;
- stable ordering;
- source identity;
- content hash/canonicalisation;
- dependency edges;
- workspace-relative path representation;
- locked-mode mismatch behaviour;
- offline-mode behaviour;
- tamper detection;
- migration from the current lockfile if the format changes;
- resolution-slot representation: selected version per `(source, name, major)` slot, with coexisting majors explicit;
- per-package capability record: derived union (configuration-independent), declared envelope, native-origin list; per-build (feature- and target-specific) derived sets are a build/report artifact, never lockfile content;
- capability-vocabulary version;
- toolchain version recorded informationally (reproducibility aid; not build-blocking by default — locked builds may opt into enforcing it);
- alias records for intentional coexistence.

Never store machine-specific absolute paths.

## WP-P0.5 — Demonstration packages and exit

Build or adapt two examples:

```text
examples/greeter/
    application + one library dependency

examples/reporter/
    diamond dependency graph with one shared base package
```

Use only existing language syntax.
Do not add package-private visibility syntax merely to enrich the demonstration.

Create `spec/packages/P0-exit-report.md` proving:

1. specs are implementation-independent;
2. current and migrated manifests validate correctly;
3. lockfile output is deterministic;
4. the diamond graph resolves shared identity once;
5. invalid visibility/source/restriction fixtures produce allocated diagnostics.

All pass → P1.

---

# GATE P1 — PACKAGE-SYSTEM HARDENING

**Outcome:** deterministic, cached, diagnosable package resolution whose formats are compatible or intentionally versioned.

## WP-P1.1 — Resolver map and determinism audit

Create `docs/dev/resolver-map.md` covering key modules, data structures, path assumptions, directory iteration, map/set ordering, and diagnostic ordering.

Fix nondeterminism only where demonstrated.
Use sorted directory reads and stable maps/ordering where output identity depends on iteration.
Mark deliberate ordering sites with a concise determinism comment.

**Done when:** two clean resolutions of `examples/reporter` in separate directories produce byte-identical lockfiles and metadata except explicitly documented machine-local diagnostic paths.

## WP-P1.2 — Identity, conflicts, and cache

Implement the E2 identity design.
Cover:

- duplicate package name from different sources;
- incompatible version constraints;
- dependency cycles with readable cycle paths;
- verified content-addressed cache reads;
- tamper rejection;
- repair/re-fetch behaviour where the source permits it;
- no silent fallback from a locked identity to a different source;
- one-version-per-`(source, name, major)` slot enforcement, with incompatible constraints inside one major failing resolution with both dependency paths shown;
- coexisting majors resolved, locked, and reported as distinct identities;
- `deny-multiple-majors` policy: global boolean and per-package-name list (package names, not categories);
- cross-major type-mixing diagnostic quality: when `foo@1::T` meets `foo@2::T`, the diagnostic names both versions and both dependency paths.

Document in:

```text
docs/diagnostics/PKG-02xx.md
docs/dev/cache.md
```

## WP-P1.3 — Workspaces and portable lockfiles

Specify and implement workspaces without new Core syntax.
The workspace format must be versioned and must use relative member paths.
Copying a workspace to another directory must preserve lockfile bytes and build results.

Convert `examples/reporter` into a workspace fixture if compatible with the E0 decision.

## WP-P1.4 — Inspection commands

Implement or verify:

```text
stark tree
stark tree --why <package>[@<major>]
stark tree --duplicates
stark tree --capabilities
stark tree --native
stark lock diff
stark package inspect <archive>
stark metadata
stark cache list
stark cache clean
stark check --locked
stark check --offline
```

Document output schemas and exit codes.
JSON output, where offered, must be deterministically ordered.

Required answers across these commands: why a package exists and which direct dependency
introduced it; exact source and content hash; derived vs. declared capabilities; native
origin; duplicate majors present; and, for `stark lock diff`: packages added/removed,
versions changed, source or registry changes, new capabilities, new native code, and
pure→native transitions.

## WP-P1.6 — Capability derivation model — E7

Ordered before the gate exit (WP-P1.5) despite its number, which avoids renumbering the
pre-existing exit WP.

Deliver:

```text
spec/packages/capabilities.md            vocabulary v1, versioning/widening rules
spec/packages/capability-derivation.md   reference-level conservative derivation rule
```

Capability vocabulary v1 (versioned format; "pure" is the absence of all capabilities, not a
listed capability):

```text
filesystem-read
filesystem-write
environment-read
network-client
network-listen
clock
randomness
process-execution
native-code
```

Requirements:

- derivation is reference-level and conservative: any reference to a host-backed interface
  anywhere in a compiled package contributes that interface's capabilities to the package's
  derived set — reachable or not, called or not; no reachability, dead-code, or
  feature-conditional narrowing in v1 (precision may be added later only as a narrowing
  refinement behind an explicit format-version bump, never removed);
- the vocabulary carries a version field with defined widening semantics: a capability split
  into finer grains interprets older declarations as the union of its successors; no
  capability is silently renamed or removed;
- mapping table: approved host-backed interface → capability set (initially empty of
  entries — see timing note);
- transitive derived-closure computation over the resolved graph;
- `derived ⊆ declared-envelope` build enforcement; underived declarations reported as audit
  signals, not failures (they may serve optional features or other targets);
- application `capability-policy` (`allow` / `deny` lists) evaluated against the derived
  transitive closure, not publisher declarations;
- lockfile capability record per the WP-P0.4 contract;
- deterministic capability report (stable ordering, byte-identical across two runs);
- the compiler track's contribution is limited to a single metadata-emission deliverable —
  the list of referenced host-backed interface identities per compiled package — scheduled
  under that track's own governance; vocabulary, mapping, closure, policy, and reports are
  ecosystem-owned.

Timing note: until Gate P4/P5 land there are no host-backed interfaces, so every package
must derive to the empty set. That is the cheapest environment to build and verify the
machinery in — any nonempty derived set before P4 is, by construction, a bug. The mapping
table populates as P4/P5 introduce providers and standard packages.

## WP-P1.5 — Scale fixture, external walkthrough, and exit

Create `examples/tenpack/`: one application plus nine small libraries, mixed depth, and at least one diamond.

Exit evidence:

1. deterministic resolution;
2. move-invariant lockfile;
3. tampered cache rejection;
4. cycle/conflict diagnostics;
5. zero reselection in locked mode;
6. one external user can clone, inspect, check, and explain the graph using the documented commands;
7. a duplicate-major fixture resolves, locks, and reports both majors visibly; the `deny-multiple-majors` policy rejects it with both paths named;
8. capability derivation produces the empty set for the entire (pure) fixture graph, deterministically, and the report/lockfile record round-trips.

Record feedback without claiming adoption.
All pass → P2.

---

# GATE P2 — PACKAGE AUTHORING TOOLS

**Outcome:** create, test, document, and archive a library using ordinary STARK package commands and current language syntax.

## WP-P2.1 — Scaffolding and manifest editing

Implement or verify:

```text
stark new <name>
stark new --lib <name>
stark add <dependency source>
stark remove <dependency>
stark update <dependency>
stark update <dependency> --dry-run
```

`stark update <dependency>` is the primary update form; whole-graph update requires an
explicit flag, never the bare command. Every update prints, before applying: packages
added/removed, versions changed, source changes, new capabilities, new native code, and
lockfile identity changes (same schema as `stark lock diff`).

Generated tests use:

```stark
fn test_<name>() { ... }
```

Do not add `@test` or `#[test]`.
Manifest edits preserve deterministic formatting and honour manifest-version compatibility.
Add→remove must round-trip byte-identically when no unrelated edits occur.

## WP-P2.2 — Test convention and runner hardening

Specify the existing convention in `spec/packages/testing.md`:

- `fn test_*()` discovers a test;
- `fn test_ignored_*()` is skipped unless requested;
- zero parameters and no receiver;
- deterministic discovery order;
- failure reports package, module, function, and source span where available;
- human and JSON result formats;
- integration programs under `tests/`;
- process/interpreter isolation behaviour documented honestly.

Do not introduce attribute syntax.
If attribute syntax is desired, record a separate E5 proposal and continue using names.

## WP-P2.3 — Documentation generation

Implement `stark doc` without executing package code.

Doc comments already present lexically may require tooling metadata attachment.
That is permitted only as compiler/tooling infrastructure using existing comment syntax; do not add attributes or new declarations.

Output at minimum:

- public modules;
- public types and functions;
- signatures;
- trait implementations;
- doc comments;
- package version and compiler compatibility;
- native disclosure flag.

## WP-P2.4 — Deterministic package archive

Implement `stark package` producing a frozen archive format:

```text
<name>-<version>.starkpkg
```

Requirements:

- stable entry ordering;
- normalised metadata and timestamps;
- manifest, sources, licence/readme, and declared native assets only;
- no build output, caches, absolute paths, credentials, or undeclared files;
- archive content hash stable across two runs;
- format version recorded;
- a deterministic inventory with a content hash for every file in the archive;
- manifest/archive content-mismatch rejection.

Explicit rejection tests: `../` path traversal; absolute paths; symlinks escaping the
extraction root; duplicate archive paths; case-collision paths; decompression bombs;
oversized file counts; undeclared native assets; hidden executable payloads.

This archive becomes the future registry payload only after P9/P10 approve a registry.

## WP-P2.5 — External author walkthrough and exit

A fresh external user receives only the authoring guide and must:

1. create a library;
2. add one public function;
3. add one `test_*` test;
4. run tests;
5. generate docs;
6. build the deterministic archive;
7. consume it locally from another package where supported.

Record blockers and friction.
All package-authoring exit criteria pass → A1.

---

# GATE A1 — EARLY SECOND-ARTIFACT EXPERIMENT: OPENAPI CONTRACT ONLY

**Outcome:** determine cheaply whether STARK’s artifact-binding architecture generalises beyond ONNX.

This gate deliberately occurs **before** native providers, JSON runtime packages, TLS, and HTTP.
It does not generate an executable network client.

## A1 hypothesis

> A STARK artifact provider can import an OpenAPI contract, generate typed declarations, bind them to deterministic artifact identity, and detect meaningful contract drift with an advantage over ordinary typed generation plus CI drift checks.

## A1 excluded mechanisms

Do not add or test:

- OAuth capability types;
- secret-flow or sensitive-field labels;
- effects;
- idempotency obligations;
- HTTP execution;
- TLS;
- runtime JSON encode/decode;
- cloud deployment;
- new Core syntax;
- a public `Artifact<...>` type.

These exclusions are essential to result attribution.

## WP-A1.1 — Scope, corpus, comparator, and decision rule

Deliver `spec/tools/openapi-contract-import.md`.

Select a narrow OpenAPI 3.x JSON subset sufficient for:

- paths and methods;
- path/query/header parameter shapes;
- request-body schema declarations;
- documented response status variants;
- component object, array, primitive, enum, required, optional, and nullable shapes.

Unsupported constructs must fail explicitly by name.
YAML is convert-first or rejected in A1; do not add a YAML parser to this gate.

Pre-register a defect corpus:

1. request field type change;
2. required field added;
3. field removed;
4. optionality change;
5. response status added;
6. response schema changed;
7. operation path/method changed;
8. referenced component changed;
9. unsupported composition construct;
10. artifact bytes changed without semantic change, to test normalisation policy.

Strong comparator:

```text
OpenAPI Generator
+ generated Rust or another strong typed client/declaration set
+ compiler checking
+ an established OpenAPI diff/drift CI tool
```

Decision rule must be written before implementation.

## WP-A1.2 — Internal artifact-provider infrastructure

Implement a minimal internal provider contract, not public Core syntax:

```text
read artifact
→ parse and normalise
→ extract typed contract
→ compute identity/provenance
→ generate/register declarations
→ verify a later artifact
→ emit domain diagnostics
```

The interface may initially be tool-internal and specialised enough to support ONNX and OpenAPI adapters.
Do not over-generalise before both implementations reveal common structure.

Document in `docs/dev/artifact-provider.md`:

- shared operations;
- ONNX-specific operations;
- OpenAPI-specific operations;
- deliberately deferred abstractions.

## WP-A1.3 — Contract declaration generator

CLI shape:

```text
stark import openapi <service.json> --out <generated-dir>
```

Generate deterministic STARK declarations for:

- schema structs/enums;
- request parameter records;
- request-body records;
- response status variants and associated body types;
- operation metadata sufficient for drift diagnostics.

Every generated file header contains:

- source path or logical source label;
- normalised schema identity;
- generator version;
- “DO NOT EDIT”; 
- no wall-clock timestamp in identity-bearing output.

Generated code must parse and type-check without networking packages.

## WP-A1.4 — Drift verification

CLI shape:

```text
stark import openapi --check <service.json> <generated-dir>
```

Required behaviour:

- exit 0 on matching normalised contract identity;
- distinct non-zero exit for drift;
- endpoint/schema/field-level summary where possible;
- unsupported feature diagnostics distinct from drift;
- deterministic output ordering;
- no regeneration required merely to learn what changed.

## WP-A1.5 — Evaluation, external walkthrough, and decision

Evaluate the pre-registered corpus across:

1. comparator stack;
2. STARK A1 importer/checker.

Measure:

- defects caught;
- detection stage;
- generated LOC;
- handwritten glue/configuration;
- duplicated contract declarations;
- diagnostic precision;
- semantic versus byte-only drift handling;
- time/steps required to update generated code after change.

One external developer runs import and check against a real API specification and reports whether the workflow addresses an existing pain.

Create `spec/tools/A1-exit-report.md` with one decision:

```text
GENERALISE
REVISE
NARROW TO TOOLING
STOP GENERAL ARTIFACT CLAIM
```

A weak result must not be rescued by adding capabilities, labels, effects, or HTTP.
If A1 fails, later executable OpenAPI work may still be useful ecosystem tooling, but it is no longer evidence for a general artifact-bound language thesis.

After A1 decision → P3.

---

# GATE P3 — GIT DEPENDENCIES

**Outcome:** practical third-party sharing before a public registry.

## WP-P3.1 — Immutable git-source specification and implementation

Specify:

- HTTPS sources only in v1;
- mandatory immutable full commit revision;
- optional package subdirectory;
- lockfile records URL, revision, package directory, and content identity;
- `.git` excluded from package content hashing;
- branch/tag movement never changes a locked build;
- explicit update required to select a new revision;
- offline builds use verified cache content;
- duplicate package identity from incompatible sources is diagnosed.

Use mature git tooling/library integration; do not implement a git protocol.

## WP-P3.2 — Transitive demonstration, external walkthrough, and exit

Demonstrate:

```text
application A → git package B → git package C
```

From a clean checkout:

- resolve online;
- rebuild offline;
- verify force-push/tag movement does not affect the lock;
- reject one-bit cache tampering;
- report source metadata deterministically.

One external user consumes the dependency using only the guide.
All pass → P4.

---

# GATE P4 — NATIVE PROVIDER INTERFACE

**Outcome:** a narrow, typed, reviewable Rust-provider boundary usable by ordinary packages without unrestricted FFI.

This is the highest-risk ecosystem gate.

## WP-P4.1 — Native boundary design — E3

Owner/Opus supplies or approves `spec/native/boundary.md` covering:

- supported value mappings;
- UTF-8 strings and byte buffers;
- arrays/vectors and plain data structs/enums;
- `Option`/`Result` mapping;
- opaque owned handles;
- ownership transfer;
- deterministic close/drop behaviour;
- panic containment;
- typed error translation;
- target selection;
- Rust-side authoring interface;
- prohibited raw-pointer/callback/plugin behaviour;
- per provider interface, its capability mapping (rule 19) as part of the ABI contract — an interface without a capability mapping cannot be approved.

Do not invent `native resource` syntax.
Opaque handles must first be represented through the approved ABI and existing STARK type surface.
If that is impossible, stop and raise E5 with evidence.

## WP-P4.2 — Compiler and package wiring

Implement the approved boundary:

- native-library package metadata;
- target validation during resolution/build;
- declared native assets only;
- capability disclosure metadata;
- panic containment;
- typed errors;
- deterministic resource close where the approved ABI supports it;
- no general unrestricted C FFI;
- no package-specific compiler hooks;
- each referenced provider interface's capabilities contribute to the importing package's derived closure, marked `native-origin`;
- native-backed packages additionally require: explicit root-application consent, supported target list, per-artifact hashes, ABI version, provenance (prebuilt vs. reproducibly built), no undeclared dynamic loading, no downloading binaries at build/run time, and no fallback to an unverified system library.

Create `docs/native/authoring.md` with one complete minimal provider.

## WP-P4.3 — First provider: `std-time`

Use time as the first provider because it exercises value passing and host access with minimal resource complexity.

Public STARK API includes:

- `Duration` constructors and arithmetic;
- monotonic `Instant::now()`;
- `SystemTime::now()` and Unix conversion;
- typed errors only where operations can fail.

No Rust identifiers or crate names appear publicly.

## WP-P4.4 — External provider attempt and exit

Validate on supported CI platforms.
Demonstrate unsupported-target failure during resolution/build, not at runtime.

One external developer follows the native authoring guide to implement or modify a tiny provider without compiler changes.

Exit criteria:

1. typed boundary;
2. panic containment;
3. target validation;
4. no Rust leakage;
5. no new Core syntax;
6. ordinary package consumption;
7. external authoring feedback recorded;
8. `stark tree --native` shows the complete native trust boundary of the validation fixture.

All pass → P5.

---

# GATE P5 — FOUNDATION STANDARD PACKAGES

**Outcome:** ordinary STARK packages support practical blocking command-line applications.

Each package follows:

```text
spec first
→ public API review
→ implementation
→ tests
→ docs
→ version/changelog
→ ordinary package consumption
```

Do not add a compiler builtin merely because a package implementation is difficult.
Use the approved native provider boundary where host access is required.

## WP-P5.1 — `std-io`

Provide blocking input/output traits and handles through the native boundary.
Freeze a reusable typed I/O error taxonomy carefully.
Avoid global mutable state beyond unavoidable process streams exposed through explicit handles/functions.

## WP-P5.2 — `std-path`

Pure STARK lexical path operations:

- construct/join;
- parent/name/extension;
- components;
- lexical normalisation;
- no filesystem access.

## WP-P5.3 — `std-fs`

Native-backed filesystem operations using `std-io` and `std-path` types.
Whole-file reads require a caller-visible maximum size.
Directory iteration must be sorted when deterministic output is promised.

## WP-P5.4 — `std-env`

Arguments, variable lookup, current directory, executable path.
No process-global environment mutation in v1 unless separately justified.

## WP-P5.5 — `std-log`

Pure STARK structured logging:

- explicit `Logger` value;
- levels;
- key/value fields;
- sink abstraction;
- no hidden global mutable logger.

## WP-P5.6 — `std-test`

Build on the existing `stark test` naming convention and runner.
Provide assertion helpers and test utilities as package APIs.
Do not add test attributes.

## WP-P5.7 — Validation application and exit

Build `examples/cfgtool/` from a written application spec.
It must use normal packages for environment, filesystem, paths, logging, and testing.
Use a small pure-STARK configuration format; JSON belongs to P6.

Exit checks:

- no direct application access to privileged compiler built-ins for fs/env;
- packages resolve normally;
- unit and integration tests pass;
- supported platforms pass;
- public APIs contain no Rust implementation details.

All pass → P6.

---

# GATE P6 — STRUCTURED DATA AND UTILITY PACKAGES

**Outcome:** deterministic structured-data processing and foundational optional packages.

## WP-P6.1 — `std-json`

Pure STARK JSON parser/serializer with explicit limits:

- `JsonValue`;
- maximum depth and bytes;
- deterministic sorted-key output mode;
- explicit `Encode`/`Decode` traits;
- no reflection or derive macros in v1;
- errors with path, expectation, and byte offset.

## WP-P6.2 — `std-url`

Parse, access components, percent encode/decode, and resolve relative references.
Specify the supported RFC subset and error taxonomy explicitly.

## WP-P6.3 — Optional first-party utilities

Implement ordinary packages such as:

- Base64;
- CSV;
- UUID parsing/formatting.

Do not add native randomness merely to support UUID v4 in this gate.
Record generation as a later provider-dependent feature.

## WP-P6.4 — Gate exit

Demonstrate:

- bounded invalid input rejection;
- deterministic JSON round-trip where promised;
- typed nested decoding with precise error path;
- optional packages consumed without compiler modification;
- package docs and archives generated through P2 tooling.

All pass → P7.

---

# GATE P7 — BLOCKING NETWORKING, TLS, AND HTTP

**Outcome:** safe blocking HTTPS clients implemented through ordinary packages and the approved native provider boundary.

## WP-P7.1 — `std-net`

Typed addresses, DNS resolution, blocking TCP connect/read/write, timeouts, and owned socket handles.
Use the host networking stack; do not implement wire protocols unnecessarily.

## WP-P7.2 — `std-tls` — E4

Use an audited TLS implementation selected through E4.
System trust store by default.
No custom cryptography.
Public configuration must be minimal and provider-neutral.

## WP-P7.3 — `std-http`

Blocking HTTP client with:

- methods;
- headers;
- request builder;
- response status/headers/body;
- bounded body size;
- explicit redirect policy;
- typed transport/TLS/protocol/timeout/size errors;
- bytes and streaming body interfaces where supported.

Underlying Rust crate names never appear publicly.

## WP-P7.4 — Live validation and exit

Against owner-provided endpoints demonstrate:

- HTTPS GET;
- JSON POST once P6 is available;
- timeout behaviour;
- TLS validation failure;
- body-size rejection;
- supported-platform behaviour.

All pass → P8.

---

# GATE P8 — EXECUTABLE OPENAPI CLIENT PACKAGE

**Outcome:** connect the already-evaluated A1 contract importer to `std-http` and `std-json` without changing the A1 research result retroactively.

P8 tests ecosystem integration and product usability.
It is not allowed to redefine a weak A1 result as success.

## WP-P8.1 — Reconcile A1 with runtime requirements

Review A1 outputs and define the minimum additions required for execution:

- JSON encode/decode implementations;
- URL/path/query construction;
- headers;
- request/response mapping;
- status variants;
- body-size and timeout configuration;
- static bearer-header support only if required by the selected fixture.

Do not add capabilities, effects, secret labels, or idempotency semantics.

## WP-P8.2 — Executable client generation

Generate an ordinary versioned STARK package using:

- `std-http`;
- `std-json`;
- `std-url`;
- A1 schema identity and drift metadata.

Generated code must be consumable and extensible without compiler changes.

## WP-P8.3 — Runtime validation

For a realistic API with at least six endpoints:

- generate a package;
- call at least three endpoints;
- handle documented status variants exhaustively;
- demonstrate request/response schema mismatch diagnostics where statically possible;
- demonstrate drift check before execution;
- package, document, and consume the generated client normally.

## WP-P8.4 — Three-way comparison and exit

Compare:

1. handwritten STARK HTTP/JSON client;
2. generated STARK client;
3. established-language generated client stack.

Metrics:

- contract defects caught before request;
- drift detection;
- generated versus handwritten LOC;
- diagnostic quality;
- integration complexity;
- update workflow after contract change;
- whether an independent developer can extend the generated package.

Create `spec/tools/P8-exit-report.md` with an honest conclusion about value beyond an ordinary HTTP wrapper.
All pass → P9.

---

# GATE P9 — EXTERNAL PACKAGE PILOT

**Outcome:** determine whether independent developers can author and consume useful packages without compiler modification.

## WP-P9.1 — Authoring guide from shipped reality

Write the guide only from implemented features.
The loop:

```text
install
→ new library
→ code
→ test
→ document
→ package
→ publish through immutable git revision
→ consume from another project
```

Any missing step is a blocker/follow-up, not invented documentation.

## WP-P9.2 — Pilot kit

Provide bounded package scope cards, evidence form, and check-in template.
Suggested packages should exercise ordinary APIs, not privileged compiler access.

## WP-P9.3 — Evidence processing

Record feedback under stable categories such as:

- setup;
- language gaps;
- package-system gaps;
- test/docs tooling;
- native-provider friction;
- dependency/versioning friction;
- diagnostics;
- publish/consume workflow;
- repeat-use intent.

Tag each item:

```text
BLOCKER | FRICTION | PAPER-CUT
```

## WP-P9.4 — Exit or revise report

Require evidence thresholds before registry work.
The report must choose:

```text
GO TO REGISTRY DESIGN
REVISE ECOSYSTEM
NARROW TO FIRST-PARTY PACKAGES
STOP PUBLIC ECOSYSTEM INVESTMENT
```

Owner confirmation is required for GO.

---

# GATE P10 — REGISTRY DECISION AND MINIMAL DESIGN

**Outcome:** design registry infrastructure only if external evidence justifies its maintenance and trust burden.

## WP-P10.1 — Prerequisite audit

Minimum evidence:

- at least five useful packages;
- at least three external authors;
- repeated package consumption;
- deterministic archive format frozen;
- stable lockfile identity;
- native disclosure/security policy documented;
- package ownership/publishing governance can be maintained.

Any missing prerequisite means: **do not start registry implementation**.

## WP-P10.2 — Minimal registry design — E6

Design only:

- immutable archives;
- content hashes;
- name/version index;
- yank semantics preserving locked builds;
- download API;
- documentation links;
- provenance metadata;
- no code execution;
- auditable publish/ownership events;
- signed index snapshots where practical.

Defer broad social/platform features until operational need exists.

## WP-P10.3 — Executable test plan

Write a clean-machine test plan for:

- resolve and verify;
- offline rebuild;
- yank survival for locked builds;
- mutated-content rejection;
- provenance/audit trace;
- maintenance burden review.

Registry implementation is a separate owner-approved roadmap after this design gate.

---

# GATE P11 — ASYNC/CONCURRENCY — NO WORK PACKAGES

Do not start.
A future proposal requires all of:

- measured blocking-I/O limitations;
- at least one real workload requiring high concurrency;
- comparison with host-managed concurrency;
- cancellation and lifetime requirements;
- impact analysis for ownership, borrowing, traits, native handles, package APIs, and generated clients.

Possible outcomes include:

```text
keep blocking APIs
add host-managed concurrency only
add library-level task abstraction
propose language-level async
stop expansion
```

No outcome is preselected.

---

# 4. DIAGNOSTIC REGISTRY

Allocate stable ranges and append exact assignments to `STATE.md`:

```text
PKG-01xx  public API and visibility
PKG-02xx  sources, identity, resolution, conflicts, cycles
PKG-03xx  lockfile, locked/offline mode, migration
PKG-04xx  native provider metadata, target, panic/error boundary
PKG-05xx  test harness and authoring tools
PKG-06xx  artifact import, generation, unsupported OpenAPI constructs, drift
PKG-07xx  package archive and documentation generation
PKG-08xx  registry/archive provenance if P10 proceeds
```

Rules:

- diagnostics are defined message-first with stage, primary span/location, notes, and remediation;
- do not reuse an existing code for a semantically different error;
- deterministic operations emit diagnostics in stable order;
- unsupported constructs are not reported as generic parse failure when the construct can be identified.

---

# 5. NOT-YET LIST — REFUSE AND CITE THIS SECTION

Do not implement:

- public registry service before P9 GO and P10 design approval;
- custom TLS or cryptography;
- database wire protocols from scratch;
- async syntax before P11 evidence;
- unrestricted C FFI;
- package compiler plugins;
- build/install scripts;
- dependency-time execution;
- cloud-provider syntax in Core;
- `pub(package)`, test attributes, or `native resource` syntax inside this track;
- a public generic `Artifact<...>` Core type before multiple artifact providers reveal a stable need;
- capabilities, effects, typestate, or information-flow labels smuggled into A1/P8;
- hundreds of low-value utility packages;
- a registry because the resolver prototype happens to exist;
- claims of adoption based on internal tests or AI agreement.

---

# 6. ROADMAP DEPENDENCY SUMMARY

```text
E0  current-state audit + compatibility decisions
 ↓
P0  package/module contract using existing language syntax
 ↓
P1  deterministic resolver/workspace/cache hardening
 ↓
P2  package authoring, tests, docs, deterministic archives
 ↓
A1  early OpenAPI contract-only artifact experiment
 ↓
P3  immutable git dependencies
 ↓
P4  approved native provider ABI
 ↓
P5  foundation blocking standard packages
 ↓
P6  JSON/URL/data packages
 ↓
P7  networking/TLS/HTTP
 ↓
P8  executable OpenAPI client package
 ↓
P9  external author pilot
 ↓
P10 registry decision/design

P11 async remains outside the implementation path until evidence exists.
```

The sequencing deliberately separates:

```text
Does artifact binding generalise beyond ONNX?
```

from:

```text
Can STARK support a practical networked package ecosystem?
```

Failure or delay in native networking must not prevent learning from the second-artifact experiment.

---

# 7. STRATEGIC OUTCOME — KEEP THIS IN VIEW

The first ecosystem proof is not merely:

> STARK has packages, JSON, or an HTTP client.

The early research proof is:

> **STARK can import a second external artifact class, generate a deterministic typed contract, bind generated code to the artifact’s identity, and detect meaningful drift with a measurable advantage over a strong existing toolchain.**

The later ecosystem proof is:

> **That verified contract can become an ordinary versioned package, execute safely through mature native infrastructure, and be consumed or extended by an independent developer without modifying the compiler.**

The package system, standard packages, native boundary, authoring tools, and registry are means to those outcomes.
They are not substitutes for external evidence.
