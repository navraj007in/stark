# WP-C7.P1 — Native HTTP/JSON REST Workload

**Status:** **IMPLEMENTED — TIER-1 QUALIFIED** (CD-273). Six execution rows green at `d735b35`; see
`../work-packages/WP-C7-P1-REST-REPORT.md`. This document is retained as the frozen commission, not
as open work — P1 is frozen at 24 exchanges and is not to be modified (`COMPILER-STATE.md`).  
**Implementation-era scope:** built against MIR **0.2** and the pre-A11 provider model, whose package
source neither declares nor calls a close. MIR is now **0.3** (amendment A12, CD-265). The evidence
below stands at its commit; when TCP migrates fully onto the A11 resource-nominal path, P1 is the
natural re-qualification workload — the two provider generations are not interchangeable.  
**Primary implementer:** Codex  
**Gate:** C7 — Native Build and Realistic Workload Qualification  
**Repository:** `navraj007in/stark`  
**Date:** 2026-07-30  
**Authority:** This work package is subordinate to the current C7 roadmap, `WP-C7.7-GATE-EXIT.md`, the closed C7.8 decisions, MIR 0.2/A11, the Core v1 language specification, and existing provider ABI contracts.

---

## 0. Purpose

Implement P1: a small but realistic HTTP/1.1 REST service written predominantly in STARK and built through the ordinary command-line native pipeline.

P1 is not a new compiler feature and is not a provider-design exercise. Its purpose is to prove that the current language, package system, native backend, generated-Rust path, runtime, and first-party host capabilities can support an integrated systems workload.

The required path is:

```text
STARK package source
→ starkc build
→ ordinary parse / resolve / typecheck
→ MIR 0.2
→ generated Rust
→ native provider link
→ executable REST server
→ real TCP requests and responses
```

P1 then becomes the workload used for the final C7 functional, build-performance, runtime-performance, optimisation, and backend-deficit reassessment.

---

## 1. Parallel-work boundary

Codex owns the P1 application, its application-level tests, reusable pure-STARK HTTP/JSON modules created for P1, and the P1 measurement harness.

Codex must not redesign or modify C7.8 host-resource architecture while this work runs in parallel.

### 1.1 Files and subsystems Codex must not modify without escalation

Do not modify the following merely to make P1 compile:

- `MirTy::HostResource` or `HostResourceNominal`;
- A11 resource identity rules;
- provider close selection or close-arena semantics;
- host-resource Drop planning or native Drop emission;
- `MIR-0026`, `MIR-0027`, `MIR-0028`–`MIR-0033` semantics;
- the zero-variant resource nominal mechanism;
- `OWN-COPY-001` resource-affinity decisions;
- provider ABI ownership/channel definitions;
- Core `File`'s SELECT-C legacy representation;
- `stark-net-native`, `stark-file-native`, or other provider behaviour, except for a proven provider defect with an isolated reproduction;
- direct-backend/Cranelift design;
- unrelated language syntax or type-system expansion.

When P1 encounters a compiler or provider failure, first reduce it to the smallest independent reproduction. Record it as a blocker with:

1. source program;
2. expected behaviour;
3. actual diagnostic or runtime result;
4. failing compiler stage;
5. whether an existing specification already requires the expected behaviour.

Do not patch architecture from inside the workload branch.

### 1.2 Allowed compiler changes

Small compiler fixes are allowed only when all of the following hold:

- the behaviour is already required by the current language specification;
- the defect is general, not P1-specific;
- a minimal non-P1 regression test is added;
- the fix does not change an approved C7.8/A11 decision;
- the commit clearly separates compiler correction from P1 application progress.

Examples include a general parser, type-checker, definite-assignment, move-checker, lowering, or backend defect exposed by ordinary legal STARK code.

---

## 2. Definition of P1

P1 is a single-process, single-threaded, bounded HTTP/1.1 REST service using the first-party TCP provider.

It must:

- bind a loopback TCP listener;
- accept client connections;
- read one complete request per connection;
- parse a bounded HTTP/1.1 request;
- route the request to one of three endpoints;
- parse JSON for the write endpoint;
- serialize JSON responses in STARK;
- handle partial socket writes correctly;
- return deterministic HTTP status codes and bodies;
- close every accepted stream exactly once;
- preserve the listener across accepted connections;
- build through ordinary `starkc build`;
- execute as a native binary without hand-built MIR or handwritten Rust application logic.

P1 is deliberately not a production web server. It is a bounded conformance and performance workload.

---

## 3. Required application behaviour

Implement these three endpoints.

### 3.1 `GET /health`

Response:

```http
HTTP/1.1 200 OK
Content-Type: application/json
Connection: close
Content-Length: 15

{"status":"ok"}
```

Acceptance:

- exact status `200`;
- body parses as JSON;
- field `status` equals `ok`;
- repeated requests are deterministic.

### 3.2 `GET /items/{id}`

Use an in-process deterministic fixture store. Persistence is not required.

Minimum fixture:

```text
1 → {"id":1,"name":"alpha"}
2 → {"id":2,"name":"beta"}
```

Successful example:

```http
GET /items/1 HTTP/1.1
Host: localhost
Connection: close
```

Response body:

```json
{"id":1,"name":"alpha"}
```

Unknown IDs return:

```http
HTTP/1.1 404 Not Found
Content-Type: application/json
Connection: close

{"error":"not_found"}
```

Acceptance:

- parse the path segment as an unsigned integer;
- reject empty, signed, non-numeric, or overflowing IDs as `400`;
- return `404` for a valid but absent ID;
- serialize the successful item body in STARK.

### 3.3 `POST /items`

Accepted request body:

```json
{"name":"gamma"}
```

Response:

```http
HTTP/1.1 201 Created
Content-Type: application/json
Connection: close

{"id":3,"name":"gamma"}
```

The implementation may use a deterministic fixed ID for the bounded workload. Durable mutation is not required unless trivial with existing features.

Invalid JSON or a missing/invalid `name` returns:

```http
HTTP/1.1 400 Bad Request
Content-Type: application/json
Connection: close

{"error":"bad_request"}
```

Acceptance:

- require `Content-Length`;
- read exactly the declared body length within the configured bound;
- reject malformed JSON;
- reject absent, duplicate, non-string, or empty `name` fields;
- correctly escape the returned JSON string.

---

## 4. HTTP subset

Implement only the following frozen subset.

### 4.1 Accepted request form

- HTTP version: `HTTP/1.1` only;
- methods: `GET` and `POST` only;
- request target: origin-form path, no absolute URI;
- line ending: `\r\n`;
- one request per connection;
- `Host` header required;
- `Content-Length` supported;
- `Connection: close` accepted and emitted;
- header names compared ASCII-case-insensitively if existing language facilities make this practical; otherwise define and test a canonical-case restriction explicitly;
- no chunked transfer encoding;
- no compression;
- no keep-alive reuse;
- no trailers;
- no TLS;
- no HTTP/2;
- no WebSocket upgrade;
- no multipart parsing;
- no query-string semantics required.

### 4.2 Bounds

Use explicit fixed bounds so malformed clients cannot cause unbounded memory growth.

Recommended initial limits:

| Input | Bound |
|---|---:|
| request line | 2 KiB |
| total header bytes | 8 KiB |
| header count | 32 |
| body | 64 KiB |
| path | 1 KiB |
| JSON nesting depth | 32 |
| JSON string length | 16 KiB |

If STARK APIs make another bound materially simpler, document the substituted value. Bounds must be enforced and tested.

### 4.3 Response requirements

Every response must include:

- status line;
- `Content-Type: application/json`;
- correct byte-count `Content-Length`;
- `Connection: close`;
- one blank line;
- exact body bytes.

Implement `write_all` in STARK so partial provider writes cannot truncate a response.

---

## 5. JSON subset

The JSON implementation must be pure STARK application/library code. Do not delegate JSON parsing or serialization to Rust, the native provider, shell utilities, Python, or test-only host code.

### 5.1 Parser requirements

The parser must support enough JSON for P1 and be structured as reusable code.

Required values:

- object;
- array;
- string;
- integer number;
- `true`;
- `false`;
- `null`.

Floating-point numbers may be explicitly rejected for P1 unless the existing language APIs make their correct parsing straightforward.

Required string escapes:

- `\"`;
- `\\`;
- `\/`;
- `\b`;
- `\f`;
- `\n`;
- `\r`;
- `\t`.

Unicode `\uXXXX` may be:

- implemented correctly; or
- rejected with a documented JSON limitation for P1.

It must never silently reinterpret malformed escapes.

### 5.2 Serializer requirements

The serializer must:

- produce valid JSON;
- quote and escape string content;
- emit deterministic object field order for P1 responses;
- avoid locale-sensitive formatting;
- calculate HTTP `Content-Length` from encoded bytes, not character count assumptions.

### 5.3 Suggested representation

Use existing language features only. A likely shape is:

```stark
pub enum JsonValue {
    Null,
    Bool(Bool),
    Int(Int64),
    String(String),
    Array(Vec<JsonValue>),
    Object(Vec<JsonMember>),
}

pub struct JsonMember {
    pub key: String,
    pub value: JsonValue,
}
```

Adapt this representation if recursive enums, generic collections, or ownership constraints require a different legal Core-v1 form. Do not add language features merely to preserve this sketch.

---

## 6. Proposed package layout

Use a dedicated package under the compiler test/workload area. Follow existing repository naming conventions discovered during implementation.

Suggested layout:

```text
starkc/tests/workloads/c7-p1-rest/
├── stark.json                 # or the repository's canonical package manifest name
├── README.md
└── src/
    ├── main.stark
    ├── server.stark
    ├── http.stark
    ├── router.stark
    ├── json.stark
    ├── model.stark
    └── bytes.stark
```

Suggested responsibilities:

- `main.stark`: configuration, bind, bounded accept loop;
- `server.stark`: connection lifecycle and request/response orchestration;
- `http.stark`: HTTP request parser and response encoder;
- `router.stark`: endpoint matching and status selection;
- `json.stark`: pure-STARK parser and serializer;
- `model.stark`: request/response and item structures;
- `bytes.stark`: bounded byte cursor, ASCII helpers, decimal parsing, `write_all`.

If the current package system cannot yet compile this many source modules, use the smallest supported package structure and record that limitation. Do not patch the package system unless the current specification already requires the missing behaviour.

---

## 7. Native capability use

P1 may use host providers only for operations that inherently require the host.

### 7.1 Allowed host operations

- TCP bind;
- TCP accept;
- TCP read;
- TCP write;
- TCP close through normal resource Drop;
- optional process arguments or environment variables for bind address and bounded request count;
- optional monotonic time only for the external measurement harness, not application correctness.

### 7.2 Forbidden host shortcuts

Do not use host/native code for:

- HTTP parsing;
- routing;
- JSON parsing;
- JSON serialization;
- item lookup logic;
- response construction;
- `Content-Length` calculation;
- validation of request bodies.

The generated Rust backend is an implementation mechanism, not permission to insert handwritten Rust application code.

---

## 8. Server lifecycle and termination

Tests require deterministic termination.

Support a bounded request count using one of these mechanisms, in preference order:

1. command-line argument such as `--max-requests 10` if current args APIs are stable;
2. environment variable such as `STARK_P1_MAX_REQUESTS=10`;
3. a test-build constant in the P1 package;
4. external process termination only as a last resort.

The server must:

- bind to loopback only by default;
- allow port `0` if the provider exposes the selected ephemeral port, otherwise accept a test-supplied free port;
- handle exactly the configured number of accepted connections;
- drop each accepted stream after its response;
- drop the listener on normal process completion;
- return a non-zero process result or deterministic failure diagnostic for unrecoverable bind/accept errors.

Do not introduce concurrency in P1.

---

## 9. Implementation phases

Each phase should be separately reviewable and should not overclaim later phases.

### Phase P1.0 — Freeze and inventory

Deliver:

- confirm exact `stark-net` source API names and signatures;
- confirm current package/module layout;
- confirm available byte/string/Vec operations;
- confirm how test code obtains the bound port;
- list any missing language operation before implementation;
- add `README.md` containing the frozen endpoint and protocol subset.

Acceptance:

- no compiler changes;
- no server implementation yet;
- a short compatibility table maps this specification to existing APIs.

### Phase P1.1 — Pure-STARK byte and JSON core

Deliver:

- byte cursor/helpers;
- bounded decimal parser;
- JSON tokenizer/parser;
- JSON string escaping and serializer;
- unit tests independent of TCP.

Acceptance:

- positive and negative JSON corpus passes through STARK-native execution where supported;
- malformed input never traps unexpectedly;
- bounds are enforced;
- no host JSON implementation.

### Phase P1.2 — Pure-STARK HTTP parser and encoder

Deliver:

- request-line parser;
- bounded header parser;
- `Content-Length` handling;
- response encoder;
- `write_all` logic tested against a controllable short-write seam if available.

Acceptance:

- parser tests cover fragmented input assembly at the application-buffer level;
- malformed CRLF, version, method, header, length, and oversized requests are rejected deterministically;
- encoded responses have exact `Content-Length`.

### Phase P1.3 — Router and handlers

Deliver:

- `/health`;
- `/items/{id}`;
- `POST /items`;
- deterministic error mapping.

Acceptance:

- handler tests run without TCP;
- exact status/body matrix passes;
- route precedence is explicit and tested.

### Phase P1.4 — Native TCP integration

Deliver:

- listener bind;
- bounded accept loop;
- request read loop;
- stream lifecycle;
- partial-write-safe response output;
- ordinary `starkc build` integration.

Acceptance:

- no hand-built MIR;
- no handwritten Rust server logic;
- executable serves all endpoints to an external test client;
- every accepted stream is closed exactly once through existing resource semantics.

### Phase P1.5 — Cross-platform functional qualification

Deliver:

- Linux, macOS, and Windows functional runs where supported by the current C7 Tier-1 matrix;
- CI job or script invoking the same test vectors;
- captured outputs and failure diagnostics.

Acceptance:

- all mandatory endpoint and malformed-request cases pass on each required platform;
- platform-specific exclusions are explicit gate findings, not silent skips.

### Phase P1.6 — Measurement harness

Deliver reproducible commands and machine-readable results for:

- source lines and generated Rust lines;
- front-end/MIR/emission time;
- cold native build time;
- no-change rebuild time;
- binary size;
- server startup time;
- sequential request latency;
- bounded throughput;
- debug vs release/profile comparison;
- optimiser-on vs optimiser-off comparison where C7.4 requires it.

Acceptance:

- workload output is verified for correctness during measurement;
- warm-up and sample count are stated;
- raw observations are retained;
- no performance conclusion is inferred from a single run.

---

## 10. Test plan

### 10.1 Pure JSON tests

Positive:

- empty object and array;
- nested objects/arrays within depth limit;
- strings with every supported escape;
- integers at accepted boundaries;
- booleans and null;
- whitespace around values;
- P1 request body.

Negative:

- unterminated string;
- illegal escape;
- malformed number;
- trailing garbage;
- duplicate `name` field for POST body;
- depth overflow;
- string/body size overflow;
- missing delimiter/comma/colon.

### 10.2 HTTP parser tests

Positive:

- health GET;
- item GET;
- item POST with body;
- headers in accepted order variations;
- body received over multiple reads;
- request line and headers split across reads.

Negative:

- LF without CR where strict CRLF is required;
- unsupported method;
- unsupported HTTP version;
- missing Host;
- malformed header;
- invalid Content-Length;
- conflicting duplicate Content-Length;
- unsupported Transfer-Encoding;
- body shorter than declared;
- oversized line, headers, or body;
- non-numeric/overflowing item ID.

### 10.3 Endpoint tests

| Request | Expected |
|---|---|
| `GET /health` | `200`, `{"status":"ok"}` |
| `GET /items/1` | `200`, alpha item |
| `GET /items/2` | `200`, beta item |
| `GET /items/999` | `404` |
| `GET /items/nope` | `400` |
| `POST /items` valid body | `201` |
| `POST /items` malformed JSON | `400` |
| `POST /items` missing name | `400` |
| unknown path | `404` |
| unsupported method | `405` or frozen `400`; choose once and test consistently |

### 10.4 Resource/lifecycle integration tests

P1 should reuse existing C7.8 lifecycle guarantees rather than reimplement their verifier tests, but add application-level proofs that:

- listener survives multiple accepts;
- accepted streams do not alias;
- failure while parsing still drops the stream;
- failure while writing still drops the stream;
- normal completion drops each stream once;
- bounded server shutdown drops the listener;
- no package code directly calls provider close functions.

Where exact close counts require instrumentation not exposed by the production provider, use the existing lifecycle test provider or a test-only provider fixture without changing the production ABI contract.

### 10.5 End-to-end client

The external test client may be written in Rust or Python because it is test infrastructure, not application logic.

It must:

- launch the built P1 binary;
- wait for readiness deterministically;
- send raw TCP HTTP requests, not rely solely on a forgiving high-level HTTP library;
- validate exact status, headers, length, and body;
- exercise split writes to the server;
- exercise malformed requests;
- shut down cleanly after the bounded request count;
- capture stdout/stderr and exit status.

---

## 11. Diagnostics and error policy

P1 must not convert every failure into a trap.

Use ordinary recoverable results for:

- malformed HTTP;
- invalid routes or IDs;
- invalid JSON;
- request limit violations;
- recoverable socket read/write failures where the API exposes them.

Reserve panic/trap for internal invariants such as impossible parser states.

Client-caused malformed input must produce a deterministic HTTP response when the connection remains usable. If a malformed framing condition makes a safe response impossible, close the stream without process-wide failure and record the case in tests.

Compiler or provider contract violations must remain hard failures under existing C7.8 rules.

---

## 12. Performance rules

P1 is a correctness workload first and a benchmark second.

Do not optimise by weakening correctness, bounds, or ownership.

Record at minimum:

- compiler commit SHA;
- Rust toolchain identity;
- target triple;
- build profile;
- CPU/OS information;
- request corpus;
- request count;
- concurrency level, fixed at one unless a later work package explicitly changes it;
- median and dispersion, not only minimum;
- whether generated crate preservation/cache reuse was enabled.

No comparison to mature production web frameworks is required for gate closure. The relevant comparison is current STARK generated-Rust behaviour across profiles and against C7's existing baseline claims.

---

## 13. Commit and branch discipline

Recommended branch:

```text
c7-p1-rest-workload
```

Recommended commit sequence:

1. `P1.0: freeze REST workload and API inventory`
2. `P1.1: pure-STARK JSON parser and serializer`
3. `P1.2: bounded HTTP/1.1 parser and response encoder`
4. `P1.3: three-route application handlers`
5. `P1.4: TCP server through starkc build`
6. `P1.5: cross-platform functional harness`
7. `P1.6: C7 measurement harness and raw results`
8. `P1: closure evidence and gate handoff`

Compiler defect fixes discovered during P1 must be separate commits with independent tests and decision IDs according to repository convention.

Do not combine speculative refactors with the workload.

---

## 14. Required deliverables

The completed branch must contain:

1. the P1 STARK package;
2. pure-STARK JSON implementation;
3. pure-STARK HTTP subset implementation;
4. router and three handlers;
5. deterministic external e2e harness;
6. positive and negative test corpus;
7. cross-platform CI integration or documented platform evidence;
8. performance/measurement script;
9. raw machine-readable measurements;
10. a P1 report containing:
    - implementation status;
    - STARK vs generated-Rust line counts;
    - capability usage;
    - correctness evidence;
    - build/runtime measurements;
    - discovered compiler/provider defects;
    - explicit deferrals;
    - whether P1 satisfies the C7 gate workload requirement.

Suggested report path:

```text
STARKLANG/docs/compiler/work-packages/WP-C7-P1-REST-REPORT.md
```

---

## 15. Exit criteria

P1 is complete only when all mandatory criteria hold.

### Functional

- [ ] ordinary `starkc build` produces the server executable;
- [ ] `GET /health` passes;
- [ ] `GET /items/{id}` success, bad-ID, and not-found cases pass;
- [ ] `POST /items` success and invalid-body cases pass;
- [ ] response `Content-Length` is byte-exact;
- [ ] partial reads and writes are handled;
- [ ] malformed requests do not cause unsoundness or process-wide failure;
- [ ] server terminates deterministically in tests.

### Language/application ownership

- [ ] HTTP parsing is STARK code;
- [ ] JSON parsing and serialization are STARK code;
- [ ] routing and handlers are STARK code;
- [ ] no hand-built MIR is used;
- [ ] no handwritten Rust application implementation is used;
- [ ] host operations are limited to admitted providers.

### Resource safety

- [ ] accepted streams close exactly once;
- [ ] listener remains valid across accepts and closes at shutdown;
- [ ] failed paths do not leak live resources;
- [ ] no direct A11 provider-close call appears in source-generated MIR;
- [ ] existing C7.8 verifier and lifecycle suites remain green.

### Qualification

- [ ] mandatory Tier-1 platforms pass or a gate-level blocker is recorded;
- [ ] full relevant test suite, fmt, and clippy are green;
- [ ] bounded C6 non-regression remains green;
- [ ] measurements are reproducible and retained;
- [ ] P1 report is complete and does not overclaim.

---

## 16. Stop and escalate conditions

Stop the affected phase and report rather than redesign when any of these occurs:

- P1 requires changing A11 resource semantics;
- a resource must become Copy/Clone or directly constructible;
- provider close must be called manually from STARK source;
- the only apparent solution changes Core `File`'s SELECT-C representation;
- generated Rust needs a fabricated/default host handle;
- package resource identity depends on build configuration in a new way;
- a required HTTP/JSON operation is impossible because the current language genuinely lacks a specified feature;
- a compiler fix would alter a closed language rule rather than implement it;
- cross-platform behaviour conflicts at the provider ABI level;
- passing the workload would require removing a verifier refusal instead of producing valid MIR.

The report must distinguish:

```text
WORKLOAD DEFECT
COMPILER DEFECT AGAINST EXISTING SPEC
PROVIDER DEFECT AGAINST EXISTING ABI
MISSING SPECIFIED FEATURE
NEW DESIGN REQUIREMENT — ESCALATE
```

---

## 17. Non-goals and explicit deferrals

P1 does not require:

- asynchronous I/O;
- threads or actors;
- concurrent connections;
- TLS;
- HTTP keep-alive;
- chunked transfer encoding;
- HTTP/2 or HTTP/3;
- WebSockets;
- production-grade security hardening;
- persistent database storage;
- filesystem-backed content;
- a general package release of HTTP or JSON libraries;
- direct native code generation;
- competitive framework-level throughput;
- deployment packaging, containers, or cloud orchestration.

Reusable modules are encouraged, but P1 must not expand into the full standard-library roadmap.

---

## 18. Handoff to final C7 work

After P1 closes, its package and harness become inputs to:

- C7.4 optimisation effectiveness rerun;
- C7.5 native build/runtime performance report;
- C7.6 generated-Rust backend deficit reassessment;
- C7.7 final gate-exit decision.

Codex should not itself declare Gate C7 closed. It should provide the evidence package and an explicit recommendation:

```text
P1 SATISFIED
P1 PARTIAL — named gaps
P1 BLOCKED — named authority/design issue
```

The C7.7 owner decision remains separate.
