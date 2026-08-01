# Gemini STARK Repository Context

**Purpose:** Stable repository and engineering context for Gemini when making code changes in the STARK project  
**Use with:** A separate task-specific work package  
**Branch model:** Work from `develop`; do not commit directly to `main`  
**Authority:** This file governs how to work in the repository. The assigned work package governs what to build.

---

## 1. Operating instruction

Read this file before editing the STARK repository.

Then read the assigned task-specific work package.

Apply this precedence:

```text
Repository context governs how to work.
Task work package governs what to build.
Active specifications and decisions govern semantics.
```

Do not broaden scope, introduce architectural changes, or modify unrelated subsystems unless the work package explicitly authorises it.

---

## 2. Repository purpose

STARK is a general-purpose programming language and compiler project with:

- a reference interpreter;
- MIR-based execution;
- generated native compilation;
- ownership and resource-lifecycle rules;
- first-party packages;
- host-capability providers;
- differential semantic testing across execution engines;
- qualification evidence for supported platforms and profiles.

This is not a Rust library project with a thin custom syntax layer. STARK has its own:

- language specification;
- ownership model;
- MIR contract;
- package system;
- provider ABI;
- diagnostics;
- execution semantics;
- qualification gates.

Do not infer STARK behaviour from Rust, TypeScript, Go, C#, Python, or another language.

---

## 3. Repository orientation

Important repository areas include:

```text
STARKLANG/
    docs/
        compiler/
        archive/
        specifications/
    packages/

starkc/
    src/
    tests/
    docs/
        compiler/

.github/
    workflows/
```

Depending on the active repository layout, first-party packages may appear under a package root such as:

```text
STARKLANG/packages/
packages/
```

Provider-backed packages may contain a native implementation area such as:

```text
<package>/
    native/
```

Before editing, inspect the repository instead of assuming exact paths from this document.

### Important document categories

```text
Active specifications
Active compiler-state and decision records
Frozen work packages
Exit reports and qualification records
Package-local specifications
Roadmaps
Archived design documents
```

Archived documents are historical context, not automatically active requirements.

---

## 4. Branch and integration workflow

Current intended branch model:

```text
main
    Protected integration/release branch.

develop
    Normal implementation integration branch.

feature/*
    Isolated work for one package, defect, or work package.
```

Rules:

1. update local references;
2. start from the latest `develop`;
3. create a focused feature branch where practical;
4. keep one package or one coherent defect per commit;
5. do not commit directly to `main`;
6. do not merge a red branch;
7. preserve the aggregate required CI check named `ci-complete`;
8. do not bypass branch protection;
9. report unrun checks explicitly.

Typical flow:

```bash
git fetch origin
git checkout develop
git pull --ff-only origin develop
git checkout -b feature/<short-task-name>
```

Do not force-push shared branches unless explicitly directed.

---

## 5. Decision hierarchy

Use this authority order when sources disagree:

```text
1. Current approved language or ABI specification
2. Active COMPILER-STATE and approved decision records
3. Frozen work package for the assigned task
4. Current implementation plus executable conformance tests
5. Current package-local SPEC.md
6. Roadmaps and planning documents
7. Archived documents
```

Rules:

- do not silently resolve contradictions;
- record the conflict;
- identify both sources;
- ask the owning decision-maker or escalate through the work package;
- do not create an architectural decision inside implementation code;
- do not treat archived documents as current merely because they are detailed.

Executable behaviour is evidence, not automatic authority. A passing implementation can still be wrong if it conflicts with an approved specification.

---

## 6. STARK language assumptions that are unsafe

Do not assume a language feature exists because it exists in Rust or another language.

Verify uncertain syntax with a minimal probe before using it in package code.

Known examples of unsafe assumptions include:

- `#[test]` attributes;
- implicit parent-module name lookup;
- blanket `.into()`;
- arbitrary `to_string()` support;
- generic derived traits;
- reference-bearing structs;
- string concatenation with `+`;
- arbitrary iterator methods;
- moving non-Copy values out of indexed collection positions;
- Rust-style dereference behaviour;
- array method surfaces;
- implicit wrapping arithmetic;
- automatic provider calls in the reference interpreter.

### Test discovery

STARK tests are currently discovered by function-name convention:

```stark
fn test_example() {
    // ...
}
```

Do not use:

```stark
#[test]
```

The `#` character is not part of the admitted test syntax and can cause lexing failure.

### Module lookup

Unqualified names do not implicitly search the parent module.

Test modules and submodules must import or qualify names explicitly.

### Integer semantics

Integer overflow traps.

Algorithms that rely on wrapping arithmetic must express wrapping behaviour explicitly and safely, for example with masks or checked transformations.

This matters for:

- PRNGs;
- hashes;
- checksums;
- bit mixers;
- binary parsers.

Do not assume a shift silently discards overflowing bits.

### Borrow-sensitive loops

A loop condition such as:

```stark
while i < output.len() {
    output[i] = value;
}
```

may hold a borrow across mutation.

A faithful form may require reading the length once before the loop:

```stark
let length = output.len();

while i < length {
    output[i] = value;
}
```

Do not apply this mechanically. Use it only when it preserves semantics.

### Indexed non-Copy values

Do not assume a non-Copy value can be moved out through indexing.

Use supported borrowing or in-place operations where required.

---

## 7. Probe-first rule

Before relying on an uncertain language feature:

1. create the smallest valid source file;
2. run the actual STARK checker;
3. run the relevant interpreter or native path;
4. record the result;
5. only then use the spelling in package code.

Good probe subjects include:

- reference-bearing structs;
- method resolution;
- generic bounds;
- enum payload syntax;
- string concatenation;
- trait-associated functions;
- `JsonValue::to_string`;
- `Duration::seconds`;
- builder chaining;
- borrowed return values;
- slice operations.

Do not design a public package API around an unproven spelling.

---

## 8. Package layout and conventions

A typical pure STARK package should contain repository-equivalent forms of:

```text
package-name/
    stark.toml
    stark.lock
    README.md
    SPEC.md
    TEST-MATRIX.md
    BLOCKERS.md
    src/
        lib.stark
        tests.stark
    examples/
```

Use the actual repository convention where names or paths differ.

### Package requirements

Each package should define:

- exact purpose;
- public API;
- semantic rules;
- bounds and limits;
- error model;
- deterministic behaviour;
- dependencies;
- exclusions;
- test evidence;
- unresolved blockers.

### Lockfile shape

Do not invent a lockfile structure.

A dependency-free package uses the repository-supported empty package collection shape, such as:

```json
{
  "packages": []
}
```

Inspect an existing valid package before creating or editing `stark.lock`.

### Package status vocabulary

Use precise status labels.

Typical distinctions:

```text
PLANNED
PARTIAL
IMPLEMENTED_LOCAL
QUALIFIED
TIER1_QUALIFIED
RELEASED
BLOCKED
```

Do not claim `QUALIFIED`, `TIER1_QUALIFIED`, or `RELEASED` without matching evidence.

---

## 9. Package test behaviour

Run the package, not only the compiler’s Rust tests.

A package can fail even when compiler-side tests pass because of:

- invalid STARK syntax;
- missing imports;
- wrong module paths;
- malformed lockfile;
- unsupported language features;
- runtime traps;
- consumer-resolution failures;
- interpreter/native divergence.

### Provider-bound packages

`stark test` now synthesizes declared `provider_api` functions so provider-bound packages can compile and run pure package tests.

However:

- package tests run through the reference interpreter;
- the interpreter cannot execute real provider calls;
- tests that avoid provider invocation can run;
- tests that require actual host effects need native consumer or qualification evidence.

### Native library qualification

A library-only package may still require an artificial executable entry point or consumer fixture for native qualification.

Do not confuse:

```text
library package can run stark test
```

with:

```text
library package can be natively qualified without a consumer entry point
```

These are separate capabilities.

---

## 10. Build and test commands

Use the exact repository commands and CI flags where available.

Common commands include:

```bash
stark check
stark test
stark build
```

Compiler workspace checks commonly include:

```bash
cargo fmt --all -- --check

cargo clippy \
  --workspace \
  --all-targets \
  --all-features \
  -- \
  -D warnings

cargo test \
  --workspace \
  --all-targets
```

### Testing discipline

- run formatting after the final edit;
- run clippy after the final edit;
- run the full workspace test command after the final edit;
- targeted tests do not replace workspace-wide tests;
- do not carry an earlier green result past later edits;
- do not state that CI is green unless CI actually completed;
- report exact command, result, and relevant counts.

Bad report:

```text
Tests pass.
```

Good report:

```text
cargo test --workspace --all-targets
Result: PASS
495 unit tests and all integration targets passed.
```

If a command was not run, say so.

---

## 11. Three-engine evidence

STARK uses multiple execution paths to detect semantic divergence.

Depending on the admitted feature, evidence may include:

- HIR/reference interpreter;
- MIR interpreter;
- generated native execution.

Rules:

- do not assume one engine is authoritative;
- compare results where the work package requires it;
- a disagreement is a defect;
- preserve exact output and error behaviour;
- do not hide divergence by weakening tests;
- do not mark three-engine qualification when only one path ran.

Record evidence separately:

```text
HIR: PASS
MIR: PASS
Native: PASS
Tier-1 CI: PENDING
```

---

## 12. Ownership and resource rules

Provider-backed resources are not ordinary copyable values.

Core principles include:

- resource handles are affine/non-Copy;
- application source calls package APIs, not raw ABI symbols;
- ownership transitions are represented and checked through compiler/MIR machinery;
- consumed resources become dead at the defined call boundary;
- close paths must be exactly once;
- successful resource wrapping must not leave the original handle live;
- failure cleanup must be explicit in the contract;
- package code must not duplicate compiler-managed cleanup;
- ownership semantics must never be weakened to make code compile.

For a resource transition:

```text
Input resource --consumed--> provider operation --output--> new resource
```

The contract must state:

- when input ownership ends;
- who closes on failure;
- who owns the underlying resource on success;
- whether retry requires reacquisition;
- which close operation owns final cleanup.

### Pure package rule

Pure packages must not introduce:

- providers;
- raw ABI functions;
- native crates;
- host authority;
- resource handles;
- platform-specific behaviour.

If a supposedly pure package appears to require any of these, stop and escalate.

---

## 13. Provider architecture

The provider boundary is conceptually:

```text
Public STARK package API
        ↓
provider_api declaration
        ↓
compiler-synthesized raw binding
        ↓
provider ABI contract
        ↓
qualified native provider
```

Rules:

- raw ABI symbols are not application APIs;
- application packages must not call raw symbols directly;
- provider status codes are closed contracts;
- undeclared status codes are contract violations;
- provider metadata must pass the repository validator;
- provider identity and version are build properties;
- do not invent provider priority rules;
- do not modify ABI contracts without explicit authority.

### Offline generated builds

Generated native builds use offline Cargo resolution.

First-party provider crates must not accidentally rely on a warmed local Cargo cache.

Prefer:

- standard library;
- operating-system APIs;
- path-only first-party dependencies.

A crates.io dependency in a runtime provider can break clean CI or user builds even when it passes locally.

Dev-dependencies used only for provider-crate tests are a separate case and must not enter generated application graphs.

---

## 14. Error discipline

Prefer closed, package-specific typed error enums.

Requirements:

- preserve exact malformed-input offsets where relevant;
- preserve boundedness failures;
- preserve provider status distinctions according to the approved mapping;
- avoid generic context-wrapping frameworks;
- avoid stringly typed errors as the primary API;
- do not collapse unrelated failures into one catch-all unless the package specification permits it;
- ensure malformed input fails deterministically;
- ensure overflow is explicit and non-ambiguous.

Example pattern:

```stark
enum ParseError {
    InvalidByte,
    UnexpectedEnd,
    LimitExceeded,
}
```

Do not add a generic `ContextError<E>` abstraction unless separately approved.

---

## 15. Bounds and parser safety

All parsers and serializers must define limits.

Typical limits include:

- maximum total input bytes;
- maximum output bytes;
- maximum item count;
- maximum nesting depth;
- maximum header count;
- maximum token length;
- maximum parameter count.

Rules:

- validate lengths before allocation;
- use checked arithmetic;
- do not allocate from untrusted declared sizes;
- preserve exact framing;
- reject malformed continuation forms;
- consume required framing sections even when discarding their semantic content;
- write negative tests before relying on happy-path tests.

Do not create a generic bounds framework prospectively. Apply the rule of three: extract a shared abstraction only after at least three real packages demonstrate the same stable shape.

---

## 16. Documentation and evidence files

### `SPEC.md`

Record:

- purpose;
- scope;
- public API;
- semantics;
- errors;
- bounds;
- determinism;
- dependencies;
- exclusions;
- unsupported behaviour.

### `TEST-MATRIX.md`

Distinguish:

```text
implemented
test written
test executed
HIR qualified
MIR qualified
native qualified
Tier-1 qualified
pending
blocked
```

Do not mark a test covered merely because source exists.

### `BLOCKERS.md`

Each blocker should include:

```text
ID
summary
minimal reproducer
expected behaviour
actual behaviour
failing layer
impact
temporary workaround, if any
closure requirement
```

A workaround must be identified as a workaround. It must not silently become package semantics.

### `README.md`

Keep it aligned with current implementation.

Do not leave stale “not implemented” sections after functionality lands.

Do not advertise future or partial functionality as complete.

---

## 17. General project law

> **Do not modify package semantics to hide a compiler defect.**

When a valid package design exposes a compiler, MIR, ABI, provider, package-manager, test-runner, or backend defect:

1. preserve the intended package semantics;
2. reduce the problem to a minimal reproducer;
3. classify the failing layer;
4. record the blocker;
5. implement a temporary workaround only if explicitly allowed;
6. keep the workaround visible;
7. close the underlying defect independently;
8. remove the workaround after closure.

Do not:

- weaken ownership;
- loosen validation;
- change error meanings;
- duplicate APIs;
- remove bounds;
- change protocol semantics;
- avoid a correct type merely because the compiler currently rejects it.

---

## 18. Scope discipline

Implement the smallest frozen surface.

Do not add:

- “helpful” convenience APIs not requested;
- generic abstractions for one use;
- additional protocol modes;
- alternate spellings;
- speculative future support;
- unrelated refactors;
- new dependencies;
- new roadmap priorities.

A task-specific work package may define explicit extension points. Otherwise, defer them.

Examples of scope creep:

- configurable percent-encoding tables when named sets are sufficient;
- generic result-extension frameworks;
- full logging frameworks for a bounded logging package;
- full MIME processing for media-type parsing;
- Unicode normalization for ASCII protocol parsing;
- async networking for a synchronous HTTP milestone.

---

## 19. Coding behaviour

Follow this sequence:

```text
Inspect before editing.
Read active decisions.
Read the package specification.
Probe uncertain syntax.
Write negative tests for parser rules.
Implement the smallest admitted API.
Run the actual package.
Run a consumer fixture.
Run full repository checks.
Update evidence documents.
Report exact results.
```

Additional rules:

- keep code readable;
- prefer explicit logic over clever generic abstractions;
- preserve deterministic output;
- avoid hidden platform dependencies;
- do not edit generated files unless the repository explicitly requires it;
- do not make unrelated formatting changes across large areas;
- do not rewrite active decisions in implementation commits.

---

## 20. Commit discipline

Each commit should represent one coherent change.

A package implementation commit should state:

- what was implemented;
- exact files or package affected;
- tests written;
- tests actually run;
- results;
- qualification status;
- blockers;
- whether compiler/provider code changed.

Example structure:

```text
Implement stark-ascii byte primitives

- Add byte classification and ASCII case conversion
- Add all-256-byte test table
- Add case-insensitive slice comparison
- Add SPEC, TEST-MATRIX, and BLOCKERS
- stark test: 24 passed
- native consumer: passed
- Tier-1 qualification: pending
- Status: IMPLEMENTED_LOCAL
- No compiler, MIR, or provider changes
```

Do not claim more than the evidence proves.

---

## 21. Escalation triggers

Stop and report before continuing if the task requires:

- compiler modification;
- MIR modification;
- provider ABI change;
- ownership-rule change;
- package-manager change;
- new generic mechanism;
- new Unicode rule;
- new string representation rule;
- new cross-package visibility behaviour;
- reinterpretation of a frozen protocol rule;
- scope expansion;
- conflict between active authoritative documents.

Also escalate when:

- required syntax is not expressible;
- HIR, MIR, and native disagree;
- a package compiles only after semantic weakening;
- tests cannot execute due to tooling;
- consumer resolution fails unexpectedly;
- a pure package appears to require host authority;
- a status code or error mapping is undeclared;
- a dependency cannot build offline;
- a library cannot be qualified without an artificial entry point and the task requires native evidence.

---

## 22. Reporting template

Use this structure at completion:

```text
Task:
Branch:
Commit(s):

Implemented:
- ...

Not implemented:
- ...

Public API:
- ...

Tests written:
- ...

Commands run:
- command
  result

Execution evidence:
- HIR:
- MIR:
- Native:
- Tier-1 CI:

Blockers:
- ...

Deviations from work package:
- none / details

Status:
- IMPLEMENTED_LOCAL / PARTIAL / BLOCKED
```

Do not omit failures.

---

## 23. Current known repository facts

At the time this context was prepared:

- normal implementation work should proceed through `develop`;
- `main` is protected by an aggregate `ci-complete` check;
- `stark test` synthesizes `provider_api`;
- provider-bound packages can run pure interpreter tests;
- actual provider effects still require native execution;
- library-only native qualification may still require a consumer entry point;
- integer overflow traps;
- tests use the `test_` name prefix;
- `#[test]` is not admitted;
- parent-module names are not implicitly visible;
- generated provider builds must work offline;
- first-party runtime providers should avoid accidental crates.io dependencies;
- provider-crate unsafe extern documentation and provider-wide lint coverage may still need separate cleanup.

These facts can change. Verify current repository state before relying on them.

---

## 24. Reusable instruction for Gemini

Use this prompt with a task-specific work package:

```text
Read STARKLANG/docs/compiler/work-packages/GEMINI-STARK-REPOSITORY-CONTEXT.md first.

Then read the assigned task work package.

Repository context governs how to work.
The work package governs what to build.
Active specifications and approved decisions govern semantics.

Work from the latest develop branch.
Do not broaden scope.
Do not modify compiler, MIR, provider ABI, ownership rules, or roadmap priorities unless explicitly authorised.
Probe uncertain STARK syntax before designing around it.
Run the actual package and exact repository checks.
Record blockers instead of weakening semantics.
Commit coherent changes separately.
Report exact commands, results, evidence, and remaining limitations.
```

---

## 25. Final rule

The goal is not merely to produce plausible code.

The goal is to produce code that:

- belongs in the STARK repository;
- obeys current STARK semantics;
- uses admitted language features;
- survives actual package execution;
- preserves ownership and provider boundaries;
- passes exact repository checks;
- carries honest evidence;
- does not create hidden architectural decisions.
