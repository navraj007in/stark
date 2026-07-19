# STARK Core Packages and Ecosystem Roadmap

**Status:** Proposed
**Date:** 17 July 2026
**Scope:** Core modules, standard packages, native-backed packages, third-party dependencies and eventual registry infrastructure

## 1. Objective

Make STARK capable of supporting useful command-line, data-processing and networked applications without turning operating-system services, HTTP, JSON or database functionality into compiler built-ins.

The track should establish:

- A clear boundary between the language Core and libraries.
- A usable official standard-package set.
- A controlled method for reusing mature native libraries.
- A practical third-party package-authoring workflow.
- Reproducible Git and registry dependencies.
- Evidence of independent package creation before building public registry infrastructure.

The goal is not to reproduce the package counts of npm, NuGet or crates.io. The goal is to make STARK useful enough that external developers can build and share real packages without modifying the compiler.

---

# 2. Architectural Model

STARK should use five distinct layers.

## Layer 0 — Language Core and Prelude

Compiler-coupled functionality required by nearly every program:

```text
Primitive types
Option and Result
String, Vec and Box
Iterator foundations
Core ownership operations
Fundamental traits
Basic formatting contracts
```

This layer must remain small, portable and independent of operating-system or network services.

A missing library feature must not automatically become a new language feature.

## Layer 1 — Standard Modules

Portable modules distributed with the compiler or standard library:

```text
std::collections
std::convert
std::fmt
std::iter
std::mem
std::result
std::option
```

These contain STARK code built around Core types and traits. They should not require native platform access.

## Layer 2 — Official Standard Packages

Versioned first-party packages maintained by the STARK project:

```text
std-io
std-fs
std-path
std-env
std-time
std-log
std-test
std-json
std-url
std-net
std-tls
std-http
```

These packages may use native providers internally, but their public API must remain defined in STARK terms.

## Layer 3 — Official Optional Packages

Useful first-party packages that are not part of the minimum standard environment:

```text
stark-csv
stark-base64
stark-uuid
stark-regex
stark-openapi
stark-image
stark-onnx
```

These can release independently and should prove the same package-authoring model intended for the community.

## Layer 4 — Third-Party Packages

Community-controlled packages such as:

```text
postgres
mongodb
redis
websocket
grpc
jwt
azure
aws
kafka
graphql
```

Third-party packages must not require compiler modifications or privileged access unavailable to ordinary package authors.

---

# 3. Terminology

## Module

A namespace and source-organisation unit inside a package.

Examples:

```text
std::http::client
std::http::headers
std::json::decoder
```

Modules are resolved during compilation and are not independently versioned.

## Package

A versioned compilation and distribution unit described by `starkpkg.json`.

A package can contain multiple modules and can depend on other packages.

## Native provider

A platform implementation used behind a typed STARK package interface.

Examples include:

- Rust filesystem implementation
- Native TLS implementation
- ONNX Runtime adapter
- Operating-system clock implementation

## Registry

An index and artifact-distribution service. The existing filesystem registry is a resolver prototype, not yet a public package service.

---

# 4. Governing Principles

1. **Do not enlarge Core to compensate for missing packages.**

2. **Prefer pure STARK implementations for high-level logic.**

3. **Reuse mature native implementations for sockets, TLS, cryptography and operating-system integration.**

4. **Expose STARK-defined APIs, not Rust-library APIs.**

5. **Begin with blocking I/O.** Async and concurrency require a separate language proposal.

6. **No unrestricted package build scripts initially.**

7. **All dependency sources must be reproducible and lockable.**

8. **Native code must be visible in package metadata.**

9. **Build a public registry only after external package creation is demonstrated.**

10. **Contract-aware packages should become a differentiator, not merely ordinary protocol wrappers.**

---

# 5. Delivery Sequence

## Gate P0 — Package and Module Contract

### Objective

Define the stable conceptual boundary between modules, packages, native providers and dependency sources.

### Deliverables

- Normative module-path and package-name rules.
- Package visibility and re-export rules.
- Package entry-point rules for libraries and applications.
- Dependency alias rules.
- Package source model:
  - path;
  - Git;
  - registry.

- Package type model:
  - pure STARK;
  - native-backed;
  - application.

- Target and platform metadata.
- Compiler/package compatibility field.
- Lockfile schema version.
- Package feature policy.
- Native-code disclosure rules.

### Initial restrictions

- No arbitrary build scripts.
- No implicit network access during compilation.
- No package-installed compiler plugins.
- No dependency source that cannot be represented in the lockfile.
- No automatic execution of downloaded code during resolution.

### Exit criteria

- Two multi-package applications compile without compiler-specific path assumptions.
- The same dependency graph resolves deterministically on two machines.
- Public and private cross-package symbols behave according to the specification.
- The package contract is documented independently of its Rust implementation.

---

## Gate P1 — Package-System Hardening

### Objective

Turn the existing package resolver into reliable infrastructure suitable for official packages.

### Deliverables

- Workspace manifests for related packages.
- Dependency aliases.
- Better cross-package diagnostics.
- Duplicate package and version conflict diagnostics.
- Deterministic dependency traversal.
- Canonical package identity.
- Lockfile validation independent of local absolute paths.
- Content-addressed package cache.
- Cache cleanup and inspection commands.
- Dependency-tree command.

### Commands

```text
stark tree
stark metadata
stark cache list
stark cache clean
stark check --locked
stark check --offline
```

### Exit criteria

- A workspace containing at least ten packages resolves and checks deterministically.
- Moving the workspace to a different directory does not change the lockfile.
- Tampered cached content is rejected.
- Cycles and conflicting package identities produce actionable diagnostics.
- Locked builds perform no dependency reselection.

---

## Gate P2 — Package Authoring Tools

### Objective

Allow a developer to create, test and document a package without manually constructing repository files.

### Deliverables

```text
stark new app-name
stark new --lib package-name
stark add dependency
stark remove dependency
stark update
stark test
stark doc
stark package
```

`stark test` must become an actual test runner rather than an alias for successful compilation.

### Test model

Introduce a bounded test convention:

```stark
@test
fn parses_valid_user() {
    assert(...);
}
```

Required capabilities:

- Test discovery.
- Per-test isolation.
- Pass/fail reporting.
- Runtime-error reporting.
- Package and integration tests.
- Deterministic exit codes.
- Optional machine-readable output.

### Documentation model

Generate:

- Public modules.
- Public types and functions.
- Trait implementations.
- Examples.
- Package version and compatibility.
- Native-code status.

### Exit criteria

- A new library can be created, tested, documented and packaged using only `stark` commands.
- Package documentation can be generated without executing package code.
- A failing test identifies its package, module, function and source span.

---

## Gate P3 — Git Dependencies

### Objective

Enable an early third-party ecosystem without first operating a public registry.

### Manifest form

```json
{
  "dependencies": {
    "stark-json": {
      "git": "https://github.com/starklang/stark-json",
      "rev": "exact-commit"
    }
  }
}
```

Later support may include tags and branches, but the lockfile must always record the exact commit.

### Deliverables

- Git repository dependency source.
- Exact revision locking.
- Content hash verification.
- Subdirectory package support.
- Offline cache support.
- Duplicate-source detection.
- Clear handling of moved or deleted tags.
- Source provenance in `stark metadata`.

### Exit criteria

- A Git dependency can be resolved, cached and rebuilt offline.
- A moved branch or tag does not alter a locked build.
- Content differing from the locked revision or hash is rejected.
- Transitive Git dependencies resolve without local path assumptions.

---

## Gate P4 — Native Provider Interface

### Objective

Allow standard and third-party packages to reuse mature native implementations through a narrow, typed boundary.

This is not unrestricted general FFI. The first version should support Rust providers only.

### Package structure

```text
std-http/
├── starkpkg.json
├── src/
│   ├── lib.stark
│   ├── request.stark
│   └── response.stark
└── native/
    ├── Cargo.toml
    └── src/lib.rs
```

### Required boundary types

- Integers and floating-point values.
- Boolean and character values.
- UTF-8 strings.
- Byte buffers.
- Arrays and vectors.
- Structs.
- Enums.
- `Option`.
- `Result`.
- Opaque resource handles.

### Resource model

Native resources such as files, sockets and HTTP clients must be represented as owned handles with deterministic cleanup.

```stark
pub struct FileHandle {
    native resource;
}
```

Moving the handle transfers ownership. Dropping the handle closes the native resource.

### Safety rules

- Native packages are clearly marked.
- Native packages declare supported targets.
- Provider panics do not cross into STARK.
- Native errors become typed STARK errors.
- No raw pointer exposure in the first version.
- No callback from native code into arbitrary STARK code initially.
- No package-defined compiler hooks.
- No automatic post-install scripts.

### First validation package

Implement `std-time` or `std-fs` as the first native-backed package.

### Exit criteria

- One native-backed package works on macOS, Linux and Windows.
- Strings, byte buffers, errors and resource handles cross the boundary correctly.
- Resource cleanup is deterministic.
- Unsupported targets fail during package resolution or build, not at runtime.
- Native implementation details do not appear in the public STARK API.

---

## Gate P5 — Foundation Standard Packages

### Objective

Make STARK suitable for real command-line and local data-processing applications.

### Initial package set

#### `std-io`

- `Read`
- `Write`
- `BufRead`
- standard input/output/error
- buffered readers and writers
- typed I/O errors

#### `std-fs`

- read and write files
- create and remove files/directories
- metadata
- directory iteration
- canonical paths
- bounded file-size operations

#### `std-path`

- path construction
- joining
- parent and filename access
- extension access
- normalization without unsafe assumptions

#### `std-env`

- arguments
- environment variables
- current directory
- executable path

#### `std-time`

- duration
- monotonic time
- wall-clock time
- basic timestamp conversion

#### `std-log`

- levels
- structured fields
- configurable sink
- no global mutable logger requirement

#### `std-test`

- assertions
- fixtures
- test-result formatting
- temporary directories

### Validation application

Build a configuration-driven CLI program that:

1. Reads command-line arguments.
2. Loads a file.
3. Parses structured configuration.
4. Performs a transformation.
5. Writes output.
6. Emits structured logs.
7. Includes unit and integration tests.

### Exit criteria

- The validation application contains no direct compiler built-ins for filesystem or environment access.
- Every standard package is consumed through normal package resolution.
- The application runs on all supported platforms.
- Official packages can be versioned and tested independently.

---

## Gate P6 — Structured Data Packages

### Objective

Support real data interchange before introducing networking.

### Packages

#### `std-json`

Initial capabilities:

- Parser and serializer.
- JSON value tree.
- Explicit `Encode` and `Decode` traits.
- Configurable depth and size limits.
- Path-aware diagnostics.
- Deterministic output mode.

Do not depend on runtime reflection in the first version.

Typed conversion can initially use explicit implementations:

```stark
impl Decode for User {
    fn decode(value: JsonValue) -> Result<User, JsonError> {
        ...
    }
}
```

A later generator can produce these implementations.

#### `std-url`

- URL parsing.
- Scheme, host, port, path and query handling.
- Percent encoding.
- Relative URL resolution.
- Validation errors.

#### Optional first-party packages

- `stark-base64`
- `stark-csv`
- `stark-uuid`

### Exit criteria

- A nontrivial JSON document can be decoded into typed STARK structures.
- Invalid data reports a field path and expected type.
- Resource limits reject deeply nested or oversized input.
- The JSON package is usable without compiler-specific knowledge.

---

## Gate P7 — Blocking Networking and HTTP

### Objective

Support practical REST clients without first changing the language for async execution.

### Packages

#### `std-net`

- IP addresses.
- Socket addresses.
- Blocking TCP streams.
- Connection timeouts.
- Read/write timeouts.
- DNS resolution through a controlled provider.

#### `std-tls`

- Client TLS configuration.
- System trust store.
- Certificate-validation errors.
- Optional custom trust roots.
- No custom cryptographic implementation.

#### `std-http`

- HTTP methods.
- Headers.
- Request builder.
- Response type.
- Byte and streaming bodies.
- Redirect policy.
- Timeouts.
- Body-size limits.
- Typed transport, TLS and protocol errors.

### Public API principle

The API must not reveal the Rust HTTP library selected underneath it.

```stark
let response = http::Client::new()
    .timeout(Duration::seconds(10))
    .get(url)?
    .send()?;
```

### Exit criteria

- A STARK program calls a real HTTPS endpoint.
- It sends and receives headers and JSON.
- Timeouts and TLS failures produce typed errors.
- The same public API works on macOS, Linux and Windows.
- The application does not depend on async language features.
- The implementation uses an established TLS and HTTP stack rather than rewriting either protocol.

---

## Gate P8 — Contract-Aware API Packages

### Objective

Connect general-purpose package usability to STARK’s actual differentiator: carrying external contracts through compilation and execution.

### OpenAPI importer

```text
stark import openapi service.yaml --out generated/service
```

Generate:

- Request structures.
- Response structures.
- Path and query parameters.
- Required headers.
- Status-code variants.
- JSON encoders and decoders.
- Typed client methods.
- Schema identity and artifact hash.
- Drift-verification command.

### Example

```stark
let response: GetUserResponse =
    client.get_user(UserId::new(42))?;
```

### Validation

Compare:

1. Handwritten HTTP and JSON calls.
2. Generated STARK client.
3. A strong typed client in an established language.

Measure:

- Contract defects detected before the request.
- Schema drift detection.
- Generated and handwritten code size.
- Diagnostic quality.
- Integration complexity.
- External-developer comprehension.

### Exit criteria

- One realistic multi-endpoint API is imported and used.
- Required-field, type, path and status-code mismatches are caught before incorrect application logic executes.
- Schema drift is detected deterministically.
- Generated code remains inspectable and replaceable.
- The result demonstrates value beyond an ordinary HTTP wrapper.

---

## Gate P9 — External Package Pilot

### Objective

Determine whether people who did not build the compiler can create and maintain STARK packages.

### Required pilot

At least three external developers create at least five packages:

- Three pure STARK packages.
- One native-backed utility package.
- One package that depends transitively on other packages.

Suggested pilot packages:

```text
slug
dotenv
csv
uuid
jwt
markdown
retry
```

### Evidence collected

- Setup time.
- Time to first passing test.
- Compiler changes required.
- Package-manager failures.
- Documentation gaps.
- Cross-platform problems.
- Native-boundary problems.
- Dependency-resolution problems.
- API stability complaints.

### Exit criteria

- At least three developers publish usable packages without direct compiler modification.
- At least one package is reused by another developer.
- Locked and offline builds work outside the original repository.
- The package-authoring documentation is sufficient for independent completion.
- No critical workflow depends on undocumented owner knowledge.

### Stop or revise criteria

Revise the ecosystem strategy if:

- Every useful package requires compiler changes.
- Native providers are too difficult to author safely.
- Package-resolution failures dominate the pilot.
- External developers cannot understand ownership across package APIs.
- Existing languages plus generators solve the same use cases more simply.

---

## Gate P10 — Registry Decision

### Objective

Decide whether public registry infrastructure is justified by demonstrated package demand.

Do not start this gate merely because the resolver supports registry-shaped dependencies.

### Minimum prerequisite

- At least five independently useful packages.
- At least three external package authors.
- Repeat package consumption by external users.
- Stable package archive format.
- Stable lockfile source identity.
- Documented package security policy.

### Initial registry design

Begin with the smallest viable model:

- Immutable package archives.
- Content hashes.
- Package name and version index.
- Yank support.
- Download API.
- Static documentation links.
- Provenance metadata.
- No arbitrary code execution.
- Signed index snapshots where practical.

Publishing, authentication, name ownership, moderation and security advisories can be added incrementally.

### Exit criteria

- A clean machine can resolve and verify registry packages.
- A locked build remains reproducible after a version is yanked.
- Mutated package content is rejected.
- Package ownership and publishing events are auditable.
- Registry operation does not become the project’s dominant maintenance burden.

---

## Gate P11 — Async and Concurrency Decision

### Objective

Determine whether networked STARK applications require new language-level concurrency features.

This is a separate proposal, not an automatic continuation of the HTTP package.

### Evidence required

- Measured blocking-I/O limitations.
- At least one workload requiring high concurrency.
- Comparison with native host-managed concurrency.
- Cancellation and resource-lifetime requirements.
- Impact on ownership, borrowing, traits and package APIs.

Possible outcomes:

- Keep blocking APIs only.
- Add host-managed task APIs.
- Add library-level futures.
- Add language-level `async`/`await`.
- Stop expansion if the complexity is not justified.

---

# 6. Recommended Repository Structure

During the pre-registry phase:

```text
stark/
├── starkc/
├── STARKLANG/
└── tools/

stark-std/
├── std-io/
├── std-fs/
├── std-path/
├── std-env/
├── std-time/
├── std-log/
├── std-test/
├── std-json/
├── std-url/
├── std-net/
├── std-tls/
└── std-http/

stark-packages/
├── stark-base64/
├── stark-csv/
├── stark-uuid/
├── stark-openapi/
└── package-template/
```

Keep official packages in one monorepo initially for coordinated changes, but ensure each remains an ordinary package with its own manifest, tests and version.

---

# 7. Versioning Policy

Before STARK 1.0:

- The compiler and official standard packages use a compatibility matrix.
- Official packages may share a coordinated release train.
- Breaking package API changes require changelog entries.
- Lockfiles record exact package and source identities.
- Third-party packages use semantic versions.
- Native providers declare supported compiler ABI or provider-interface version.

After a stable package interface exists:

- Core language stability and package API stability are treated separately.
- Standard packages can evolve without requiring grammar changes.
- Package formats should change only through explicit schema versions.

---

# 8. Security and Trust Model

Every package should disclose:

```text
Pure STARK
Native-backed
Network access required
Filesystem access required
Process execution required
Supported platforms
Source provenance
Content hash
```

Initial security rules:

- No install-time scripts.
- No implicit compiler plugins.
- No hidden native code.
- No unsigned substitution of locked content.
- No execution during dependency resolution.
- Native package review status is metadata, not an automatic claim of safety.
- TLS and cryptographic packages must use established audited implementations.

A future capability model may allow applications to grant explicit filesystem, network or process access, but this should begin as a package/runtime policy rather than a Core syntax change.

---

# 9. Priority Order

## Critical path

```text
P0 Package contract
    ↓
P1 Resolver hardening
    ↓
P2 Authoring and real test tools
    ↓
P3 Git dependencies
    ↓
P4 Native provider interface
    ↓
P5 Foundation standard packages
    ↓
P6 JSON and URL
    ↓
P7 Blocking HTTPS
    ↓
P8 OpenAPI contract-aware clients
    ↓
P9 External package pilot
    ↓
P10 Public registry decision
```

P11 async/concurrency remains independent and evidence-gated.

---

# 10. Work That Should Not Begin Yet

Do not prioritise:

- A large public registry service.
- Hundreds of low-value utility packages.
- A custom TLS implementation.
- A custom cryptographic library.
- Database wire protocols written from scratch.
- Async syntax before blocking HTTP is validated.
- General unrestricted C FFI.
- Package compiler plugins.
- Arbitrary build scripts.
- Distributed package execution.
- Cloud-provider syntax in the language Core.
- Moving every standard-library operation into compiler built-ins.

---

# 11. Alpha Readiness Criteria

STARK can claim a credible package-capable public alpha when:

- A developer can create, test, document and consume a library.
- Git and local/registry dependencies are reproducible.
- Filesystem, time, JSON and HTTPS packages work on supported platforms.
- Native-backed packages have a controlled interface.
- At least three external developers create packages.
- At least one external package is reused by another project.
- A real application can be built without changing the compiler.
- The package and standard-library compatibility policy is documented.

---

# 12. Strategic Outcome

The package strategy should make STARK useful without diluting its reason to exist.

Ordinary infrastructure should be reused:

```text
Sockets → mature native implementation
TLS → audited native implementation
HTTP → established protocol library
Filesystem → operating-system provider
```

STARK should add value above that infrastructure:

```text
Typed public APIs
Ownership-safe resources
Reproducible package identity
Explicit capabilities
Generated external contracts
Schema and artifact drift detection
Domain meaning carried through execution
```

The first ecosystem proof should therefore not be “STARK has an HTTP client.”

It should be:

> STARK can import an external API contract, generate a typed package, verify the contract’s identity, safely execute it through mature native infrastructure, and let an independent developer consume or extend that package without modifying the compiler.
