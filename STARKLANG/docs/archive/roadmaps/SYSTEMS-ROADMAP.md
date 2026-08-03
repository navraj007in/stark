# STARK Systems and Ecosystem Roadmap

This companion roadmap proves practical native systems capability without moving package
ecosystem product design, networking/TLS/HTTP package design, or concurrency semantics into the
compiler roadmap. It depends on the compiler track for native executables, runtime/provider ABI,
resource semantics, and Core language parity.

Before S0 implementation begins, expand this strategic outline into executable work-package
documents with entry criteria, exit criteria, owning track, mandatory tests, target platforms,
security limits, error contracts, and provider-versus-pure-STARK boundaries. Initial S0 work
packages should include:

```text
WP-S0.1 Provider manifest and identity
WP-S0.2 Provider linker integration
WP-S0.3 Capability validation
WP-S0.4 Opaque resource ownership and Drop
WP-S0.5 Negative and portability corpus
```

Use the same work-package pattern later for networking and HTTP.

## Relationship to the compiler roadmap's P1 checkpoint

The compiler roadmap (`STARKLANG/docs/compiler/COMPILER-ROADMAP.md` §4.2) defines **P1 — Native
Systems Baseline**, the mandatory post-C6 checkpoint that gates WP-C7.5/C7.7 closure and the
Native Systems Preview / STARK v1 General-Purpose Stable release classes. P1's exit criteria
are delivered by this roadmap: S0–S4 supply the provider contract, process/filesystem/time/
network capability, and the pure-STARK JSON/HTTP packages, and **S5's standalone server is the
work that completes P1** — P1 is evaluated against the compiler roadmap's §4.2 exit list, S5 is
the stage that produces the evidence. S6/S7 are beyond P1's scope.

---

## S0 — Native Provider Contract Integration

- provider manifest;
- provider loading/linking;
- capability declaration;
- ABI compatibility validation;
- safe wrapper package;
- negative tests for undeclared and mismatched providers.

## S1 — Process Baseline

- command-line arguments;
- environment variables;
- stdin/stdout/stderr;
- exit status;
- working directory.

## S2 — Filesystem and Time

- paths;
- file and directory operations;
- wall clock;
- monotonic clock;
- sleep;
- deterministic resource cleanup.

## S3 — Blocking Networking

- IP/socket address;
- TCP listener;
- TCP stream;
- DNS;
- timeouts;
- bounded reads and writes.

## S4 — Application Packages

- byte buffer utilities;
- URL parsing;
- JSON values/parser/encoder;
- HTTP request/response;
- HTTP/1.1 parser;
- static routing;
- logging/configuration.

## S5 — Practical Systems Application

Build a standalone server with:

```text
GET  /health
GET  /users/:id
POST /echo
```

Requirements:

- Linux native executable;
- no interpreter fallback;
- no Rust/C application logic;
- bounded requests;
- explicit JSON;
- sequential blocking execution;
- correct Drop and shutdown;
- reproducible package/provider metadata;
- a documented trap-abort operational report (deliberately trap one handler; record the effect
  on in-flight connections, open resources, buffered output, and process state — mirrors the
  P1 exit requirement in `COMPILER-ROADMAP.md` §4.2, CD-021).

This is the practical systems capability milestone.

## S6 — Concurrency v0.1

Opens only after the sequential HTTP server is complete and its throughput/blocking limitations
are documented.

The systems roadmap is the strategic owner for this milestone, but concurrency is joint
compiler, runtime/provider, and ecosystem work. It requires a compiler/extension work package
because `Send`/`Share`, ownership transfer, guards, failure semantics, MIR operations, native
lowering, and diagnostics are language implementation concerns rather than ordinary package
implementation.

### S6A — Concurrency language proposal

- `Send`/`Share` laws;
- thread ownership;
- shared-reference rules;
- cross-thread Drop;
- failure/shutdown semantics.

### S6B — Compiler implementation

- type checking;
- borrow checking;
- MIR operations and verifier;
- native lowering;
- diagnostics.

### S6C — Runtime/provider

- OS threads;
- joins;
- channels;
- mutexes;
- worker groups.

### S6D — Ecosystem validation

- worker-pool HTTP server;
- shared application state;
- shutdown and load tests.

Excluded:

- async/await;
- futures;
- actors;
- green threads;
- lock-free structures;
- async Drop.

## S7 — Production Package Expansion

Expand provider-backed and pure-STARK packages only after the S5 application and S6 concurrency
evidence identify concrete needs. Package policy, public registry design, and package-author
workflows require separate owner-approved proposals.
