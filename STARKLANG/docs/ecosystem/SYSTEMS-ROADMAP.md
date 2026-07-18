# STARK Systems and Ecosystem Roadmap

This companion roadmap proves practical native systems capability without moving package
ecosystem product design, networking/TLS/HTTP package design, or concurrency semantics into the
compiler roadmap. It depends on the compiler track for native executables, runtime/provider ABI,
resource semantics, and Core language parity.

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
- reproducible package/provider metadata.

This is the practical systems capability milestone.

## S6 — Concurrency v0.1

Opens only after the sequential HTTP server is complete and its throughput/blocking limitations
are documented.

Initial scope:

- compiler-known `Send` and `Share`;
- named-function `spawn`;
- explicit owned argument;
- `JoinHandle`;
- bounded channels;
- `Shared<T>`;
- `Mutex<T>` and guard;
- structured worker group;
- trap and shutdown rules.

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
