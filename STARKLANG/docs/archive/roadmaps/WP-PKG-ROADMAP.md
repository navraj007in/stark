> **ARCHIVED — SUPERSEDED 2026-08-03.** This roadmap was consolidated into the single
> forward plan at `ROADMAP.md` (repository root). It is retained for provenance and for
> the citations that point at it; **do not schedule work from it**. Where it disagrees
> with `ROADMAP.md`, `ROADMAP.md` wins.
>
> Former path: `STARKLANG/docs/compiler/work-packages/WP-PKG-ROADMAP.md`

---

# WP-PKG-ROADMAP — Remaining STARK First-Party Packages

**Status:** Proposed execution roadmap  
**Prepared:** 2026-07-31  
**Repository:** `navraj007in/stark`  
**Scope:** Remaining first-party package implementation and qualification after Gate C7  
**Ordering basis:** dependency leverage, compiler eligibility, implementation complexity, host/provider risk, and application value

---

## 1. Objective

Complete the remaining STARK first-party package ecosystem in an order that:

1. closes already-implemented packages before widening scope;
2. prioritises packages that unlock CLI, data-processing, networking, and HTTP applications;
3. separates pure-STARK libraries from host-backed packages;
4. avoids beginning packages whose required language/runtime semantics are not ready;
5. requires native artefact execution rather than accepting build-only evidence;
6. keeps package work from silently turning into unrelated compiler redesign.

This roadmap covers package implementation, qualification, and package-specific provider work. It does not replace compiler work packages such as `WP-C7.8-RB0`.

---

## 2. Status vocabulary

| Status | Meaning |
|---|---|
| `IMPLEMENTED_LOCAL` | Package implementation exists and local package tests pass. |
| `QUALIFICATION_PENDING` | Implementation exists, but native/Tier-1 CI evidence is incomplete. |
| `READY` | Current language/compiler/provider surface appears sufficient to implement the package. |
| `READY_BOUNDED` | A deliberately limited v0.1 can be implemented now; later features need additional capabilities. |
| `PROVIDER_EXPANSION` | Package requires new host-provider symbols but no major language feature. |
| `LANGUAGE_BLOCKED` | Package depends on compiler/runtime semantics not yet admitted or qualified. |
| `DEFER` | Valuable, but should not compete with higher-leverage foundational packages. |

Complexity:

- **S** — small, bounded package or qualification job;
- **M** — moderate parser/model/provider package;
- **L** — substantial package with multiple semantic surfaces;
- **XL** — major platform/runtime programme, not a normal package task.

---

## 3. Current baseline

### 3.1 Implemented packages needing closure or qualification

| Package | Current state | Remaining work |
|---|---|---|
| `stark-json` | Parser and compact encoder implemented; native consumer executes locally | Add Tier-1 CI qualification and record Linux/macOS/Windows evidence |
| `stark-url` | Origin-form target parsing, percent encoding/decoding, ordered query handling implemented | Add Tier-1 CI qualification |
| `stark-base64` | Encoder/decoder and strong HIR test corpus implemented | Add cross-package consumer, MIR/native execution, and Tier-1 CI |
| `stark-hex` | v0.1 package exists | Re-audit current evidence, run native consumer, and add Tier-1 CI |
| `stark-io` | Large synchronous file/filesystem surface executes from STARK source | Finish edge-policy hardening and Tier-1 qualification; formally close v0.1 |
| `stark-time` | Pure time types and provider foundations substantially implemented | Consolidate package surface and qualify remaining public operations |

### 3.2 Provider foundations already present

- environment/argument native provider;
- time provider;
- file provider;
- TCP provider;
- provider capability selection;
- normal `stark build` provider integration;
- scalar and buffer provider calls;
- provider-backed resource ownership and MIR-managed close;
- installed-toolchain provider discovery.

These make several packages eligible now that were previously blocked.

---

# PART A — Immediate closure track

## 4. WP-PKG-Q1 — Unified Tier-1 package qualification

**Priority:** P0  
**Complexity:** M  
**Type:** Qualification infrastructure  
**Dependencies:** Current compiler and package directories  
**Blocks:** Formal closure claims for JSON, URL, Base64, Hex, and later packages

### Purpose

Create one CI matrix for first-party pure packages and their native consumers.

### Required platforms

- Linux x64;
- macOS arm64;
- Windows x64.

### Per-package required actions

```text
stark check
stark test
stark fmt --check
consumer check
consumer interpreter run
consumer native build --no-build-cache
execute the produced native artefact
compare exact expected stdout/exit status
```

A successful native build without executing the artefact is not acceptance evidence.

### First admitted packages

1. `stark-json`
2. `stark-url`
3. `stark-base64`
4. `stark-hex`

### Exit criteria

- all matrix rows green on one exact commit;
- no unclassified `#[ignore]`;
- evidence documents updated with commit SHA and platform rows;
- package status changed from local/partial to `QUALIFIED`;
- failure output identifies package, engine, platform, and command.

---

## 5. WP-PKG-Q2 — Close `stark-io` v0.1

**Priority:** P0  
**Complexity:** S–M  
**Type:** Existing implementation hardening  
**Dependencies:** Current `stark-io` implementation  
**Blocks:** Useful CLI milestone

### Required fixes and decisions

1. Resolve the `append + truncate` option policy and test it.
2. Replace unchecked timestamp narrowing with checked conversion.
3. Narrow `sync_all` documentation to an operating-system durability request, not an unconditional power-loss guarantee.
4. Document rejection of non-UTF-8 directory-entry names.
5. Confirm the expanded provider manifest and source consumer on all Tier-1 platforms.
6. Preserve deliberate exclusions:
   - no recursive directory creation;
   - no recursive deletion;
   - no asynchronous file APIs.

### Exit criteria

- minimal and expanded source consumers execute on all Tier-1 platforms;
- A11 resource-lifecycle tests remain green;
- Core `File` legacy tests remain green;
- no named verifier exemptions;
- package README states exact guarantees and exclusions;
- status becomes `stark-io v0.1 QUALIFIED`.

---

## 6. WP-PKG-Q3 — Complete `stark-time` package qualification

**Priority:** P1  
**Complexity:** M  
**Type:** Consolidation and qualification  
**Dependencies:** Existing time provider and pure types

### v0.1 target

- `Duration`;
- `UnixTimestamp`;
- `Instant`;
- checked arithmetic;
- comparisons and conversions;
- monotonic time;
- Unix wall-clock time;
- bounded ISO-8601 parse/format if already supportable without widening the compiler.

### Deferred

- time zones;
- locale formatting;
- full calendar database;
- cron;
- async timers.

### Exit criteria

- deterministic pure operations agree across engines;
- provider-backed operations execute natively on all Tier-1 platforms;
- overflow and unavailable-clock behaviours are pinned;
- no sentinel value is used for absence where all scalar values are valid.

---

# PART B — Ready for implementation now

## 7. WP-PKG-ENV1 — `stark-env` v0.1

**Priority:** P0  
**Complexity:** S–M  
**Status:** `READY`  
**Type:** Host-backed package over an existing provider  
**Dependencies:** Existing environment provider and provider API build path

### Why now

The provider was created before the normal package/provider build path was complete. That compiler work is now available. The P1 workload already proves environment calls can execute from ordinary STARK source.

### Recommended v0.1 surface

```stark
pub enum EnvError {
    InvalidName,
    InvalidEncoding,
    NotPresent,
    LimitExceeded,
    Other,
}

pub struct EnvLimits {
    pub max_args: UInt64,
    pub max_total_arg_bytes: UInt64,
    pub max_value_bytes: UInt64,
}

pub fn default_limits() -> EnvLimits;

pub fn args() -> Result<Vec<String>, EnvError>;

pub fn args_with_limits(
    limits: &EnvLimits,
) -> Result<Vec<String>, EnvError>;

pub fn get(name: &str) -> Result<Option<String>, EnvError>;

pub fn get_with_limit(
    name: &str,
    max_bytes: UInt64,
) -> Result<Option<String>, EnvError>;

pub fn get_required(name: &str) -> Result<String, EnvError>;
```

### Required semantics

- argument zero is included;
- argument order is preserved;
- empty arguments survive NUL-separated decoding;
- absent and present-empty environment values remain distinct;
- invalid UTF-8 is reported, never replaced lossily;
- invalid names are rejected;
- no environment mutation;
- explicit output limits prevent unbounded allocation.

### Exit criteria

- package-local tests;
- native consumer reading known test variables and arguments;
- Tier-1 CI execution;
- provider metadata and package API documented;
- no new MIR/resource work.

---

## 8. WP-PKG-UUID1 — `stark-uuid` v0.1 parsing and formatting

**Priority:** P1  
**Complexity:** S–M  
**Status:** `READY_BOUNDED`  
**Type:** Pure STARK  
**Dependencies:** Byte arrays/vector operations and hex utilities

### Scope

Implement now:

- UUID value type backed by 16 bytes;
- canonical hyphenated parsing;
- canonical lowercase formatting;
- nil UUID;
- `from_bytes`;
- `as_bytes`;
- version inspection;
- variant inspection;
- `is_nil`;
- equality and ordering if supported cleanly.

### Required decisions

- accept uppercase input: recommended **yes**;
- emit lowercase output: **yes**;
- reject braces, URNs, and compact 32-character form in v0.1 unless explicitly admitted;
- report exact invalid character/position;
- reject invalid hyphen placement;
- do not infer validity solely from length.

### Explicitly excluded

- UUID v4 generation;
- UUID v7 generation;
- weak pseudo-random generation;
- MAC/time-based v1;
- namespace v3/v5 until hash packages exist.

### Exit criteria

- fixed RFC vectors;
- all byte positions and boundaries exercised;
- parse/format/parse round trip;
- native consumer and Tier-1 CI;
- zero provider dependencies.

---

## 9. WP-PKG-SEMVER1 — `stark-semver` v0.1

**Priority:** P1  
**Complexity:** M  
**Status:** `READY`  
**Type:** Pure STARK  
**Dependencies:** String parsing, vectors, comparison

### Strategic purpose

The package manager and package compatibility policy need one authoritative semantic-version implementation.

### Scope

- strict SemVer version parsing;
- major/minor/patch;
- prerelease identifiers;
- build metadata;
- precedence comparison;
- canonical formatting;
- exact parse errors;
- basic compatibility requirements.

### Recommended range floor

Start with:

- exact versions;
- caret requirements;
- tilde requirements;
- comparison sets such as `>=1.2.0,<2.0.0`.

Defer npm/Cargo-complete range grammar until a concrete package resolver requires it.

### Exit criteria

- SemVer 2.0.0 official precedence vectors;
- malformed leading-zero cases;
- deterministic comparator;
- package-manager consumer test;
- Tier-1 CI.

---

## 10. WP-PKG-CHECKSUM1 — `stark-checksum` v0.1

**Priority:** P1  
**Complexity:** S  
**Status:** `READY`  
**Type:** Pure STARK  
**Dependencies:** Integer and byte operations

### Scope

- CRC32;
- Adler-32;
- one-shot byte input;
- incremental state objects if expressible without unnecessary API complexity;
- lowercase hex formatting through `stark-hex`.

### Value

Creates an immediate useful application:

```text
stark checksum <file>
```

This integrates:

- `stark-env`;
- `stark-io`;
- `stark-checksum`;
- `stark-hex`.

### Exit criteria

- published standard vectors;
- chunked versus one-shot equivalence;
- full native CLI consumer;
- Tier-1 CI.

---

## 11. WP-PKG-PATH1 — `stark-path` v0.1

**Priority:** P1  
**Complexity:** M  
**Status:** `READY_BOUNDED`  
**Type:** Primarily pure STARK  
**Dependencies:** String operations

### Scope

Lexical operations:

- `join`;
- `components`;
- `parent`;
- `file_name`;
- `stem`;
- `extension`;
- `with_extension`;
- `is_absolute`;
- separator-aware lexical normalisation.

### Design boundary

Keep lexical operations pure. Host filesystem canonicalisation belongs in `stark-io`, because it can access the filesystem, resolve symlinks, fail with OS errors, and depend on the current directory.

### Required platform policy

The package must either:

1. expose explicit `PathStyle::{Unix, Windows}`, or
2. compile with a documented target-specific style.

Prefer explicit style for deterministic tests and cross-platform processing.

### Exit criteria

- Unix and Windows lexical vectors on every host;
- no filesystem access for lexical operations;
- `stark-io` integration consumer;
- Tier-1 CI.

---

# PART C — Provider expansion packages

## 12. WP-PKG-RANDOM1 — `stark-random` v0.1

**Priority:** P2  
**Complexity:** M–L  
**Status:** `PROVIDER_EXPANSION`  
**Type:** Pure deterministic PRNG plus secure host-backed source  
**Dependencies:** New secure-random provider; stable integer semantics

### Two deliberately separate APIs

#### Deterministic PRNG

- exact specified algorithm;
- seeded constructor;
- `next_u64`;
- bounded integer generation without modulo bias;
- byte generation;
- sequence compatibility frozen by version.

#### Secure randomness

```stark
pub fn secure_bytes(
    count: UInt64,
) -> Result<Vec<UInt8>, RandomError>;
```

### Rules

- deterministic PRNG must never be labelled secure;
- secure API must use OS entropy;
- no fallback from secure to deterministic;
- output size must be bounded;
- partial or unavailable OS randomness must fail.

### Exit criteria

- same seed, same sequence on every engine/platform;
- secure provider executes on Tier-1 platforms;
- provider failure tests;
- no hidden global RNG state.

---

## 13. WP-PKG-UUID2 — UUID generation

**Priority:** P2  
**Complexity:** S–M after dependencies  
**Status:** Blocked on `stark-random`; v7 also needs qualified wall-clock time

### Scope

- UUID v4 generation using secure randomness;
- UUID v7 generation after time semantics are frozen;
- bit layout validation;
- monotonicity policy for same-millisecond v7 generation must be explicitly decided.

### Exit criteria

- correct version/variant bits;
- injected deterministic providers for tests;
- no weak fallback;
- native Tier-1 execution.

---

## 14. WP-PKG-ENV2 — Expanded environment/process information

**Priority:** P2  
**Complexity:** M  
**Status:** `PROVIDER_EXPANSION`

Potential additions:

- current directory;
- change current directory;
- executable path;
- platform identifier;
- architecture identifier;
- process ID.

Environment mutation should remain deferred unless a real application requires it, because process-global mutation has concurrency and test-isolation consequences.

---

## 15. WP-PKG-PROCESS1 — `stark-process` v0.1

**Priority:** P3  
**Complexity:** L  
**Status:** `PROVIDER_EXPANSION`  
**Dependencies:** `stark-env`, `stark-io`, resource lifecycle, pipe model

### Initial bounded scope

- process exit;
- spawn executable with argument vector;
- wait/status;
- optional captured stdout/stderr with explicit limits;
- no shell interpretation;
- no implicit PATH search unless explicitly specified.

### Risks

- child process resource lifecycle;
- pipe handles;
- deadlock when capturing multiple streams;
- platform-specific executable lookup;
- cancellation and cleanup;
- environment inheritance policy.

A separate detailed work package is mandatory before implementation.

---

# PART D — Networking and web packages

## 16. WP-PKG-NET1 — Complete `stark-net` v0.1

**Priority:** P2  
**Complexity:** M–L  
**Status:** Partially implemented through TCP/P1 foundations  
**Dependencies:** Existing TCP provider and lifecycle support

### Scope

- `TcpListener`;
- `TcpStream`;
- bind/listen/accept/connect;
- read/write/write-all;
- shutdown;
- peer/local address where provider-supported;
- explicit partial-read/partial-write semantics;
- EOF;
- timeouts if provider support exists.

### Missing companion work

- structured address types;
- DNS resolver;
- UDP remains separate;
- no async.

### Exit criteria

- source-level echo client/server;
- bounded multi-request lifecycle test;
- all close paths observed;
- Tier-1 CI.

---

## 17. WP-PKG-DNS1 — Resolver API

**Priority:** P3  
**Complexity:** M  
**Status:** `PROVIDER_EXPANSION`  
**Dependencies:** Networking provider expansion

### Scope

- hostname to address resolution;
- ordered results;
- address-family classification;
- explicit result limits;
- deterministic error mapping.

Do not implement the DNS wire protocol in STARK v0.1. Use the host resolver behind the provider boundary.

---

## 18. WP-PKG-HTTP1 — `stark-http-core`

**Priority:** P2  
**Complexity:** M  
**Status:** `READY` as a pure package  
**Dependencies:** `stark-url`; JSON optional

### Scope

- HTTP method;
- status code;
- version;
- headers preserving insertion/order policy;
- request and response models;
- body as bounded bytes;
- validation of token/header names and values.

No socket access in this package.

---

## 19. WP-PKG-HTTP2 — `stark-http-parser` v0.1

**Priority:** P2  
**Complexity:** L  
**Status:** `READY_BOUNDED`  
**Dependencies:** `stark-http-core`, string/byte parsing

### Scope

HTTP/1.1 only:

- request line;
- status line;
- headers;
- content length;
- chunked transfer;
- strict byte/header/body limits;
- deterministic malformed-input errors;
- incremental parser if current ownership/borrowing allows it cleanly.

### Required security rules

- reject conflicting `Content-Length`;
- reject ambiguous transfer framing;
- reject invalid line endings according to the chosen strict profile;
- prevent request-smuggling ambiguity;
- bound header count and total bytes.

---

## 20. WP-PKG-HTTP3 — Synchronous HTTP server/client

**Priority:** P3  
**Complexity:** L  
**Dependencies:** `stark-net`, `stark-http-core`, `stark-http-parser`, `stark-url`

### Initial server

- one connection at a time;
- bounded keep-alive;
- request parsing;
- response writing;
- fixed-length and chunked bodies where qualified;
- clean resource closure.

### Initial client

- HTTP only initially;
- GET/POST;
- headers;
- fixed/chunked response bodies;
- redirects deferred;
- TLS integration later.

---

## 21. WP-PKG-TLS1 — `stark-tls`

**Priority:** P4  
**Complexity:** L–XL  
**Status:** `PROVIDER_EXPANSION`  
**Dependencies:** networking, certificate/error model

Use a proven host TLS implementation. Do not implement cryptography or TLS protocol logic from scratch.

Required governance:

- provider implementation and version recorded;
- protocol-version policy;
- hostname verification;
- certificate trust source;
- client/server configuration;
- secure stream ownership and close semantics.

---

# PART E — Additional pure data packages

## 22. WP-PKG-CSV1 — `stark-csv` v0.1

**Priority:** P2  
**Complexity:** M  
**Status:** `READY`  
**Dependencies:** String/Vec

Scope:

- configurable one-byte delimiter;
- quoted fields;
- escaped quotes;
- CRLF/LF handling;
- row/column error positions;
- in-memory parser and encoder;
- streaming deferred.

---

## 23. WP-PKG-TOML1 — `stark-toml-lite`

**Priority:** P3  
**Complexity:** M–L  
**Status:** `READY_BOUNDED`

Bounded subset:

- strings;
- integers;
- booleans;
- arrays;
- tables;
- deterministic duplicate-key rejection.

Use only if the package manager or tooling needs TOML. Do not compete with `stark-semver` and HTTP foundations without a concrete consumer.

---

## 24. WP-PKG-GLOB1 — `stark-glob`

**Priority:** P3  
**Complexity:** S–M  
**Status:** `READY`

Scope:

- `*`;
- `?`;
- character classes if bounded;
- explicit path-separator behaviour;
- no filesystem traversal in the pure matcher.

Prefer this before attempting a full regex engine.

---

## 25. WP-PKG-REGEX1 — `stark-regex-lite`

**Priority:** P4  
**Complexity:** L  
**Status:** `DEFER`

A regex engine should not be started until a concrete grammar, complexity bound, and denial-of-service policy are specified. Parser combinators may provide better immediate leverage.

---

# PART F — Language/runtime-gated ecosystem work

## 26. WP-PLATFORM-THREADS — Structured threads and synchronisation

**Priority:** P4  
**Complexity:** XL  
**Status:** `LANGUAGE_BLOCKED`

Requires language/runtime rules for:

- values crossing thread boundaries;
- shared ownership;
- scoped thread borrows;
- thread-safe drop;
- join/trap propagation;
- mutexes, channels, atomics.

This must be a compiler/platform gate, not a package-only implementation.

---

## 27. WP-PLATFORM-ASYNC — Async/await runtime

**Priority:** P5  
**Complexity:** XL  
**Status:** `LANGUAGE_BLOCKED`

Requires:

- `async fn`;
- `await`;
- future state layout;
- suspension-point ownership;
- cancellation;
- drop of suspended futures;
- executor/reactor interfaces.

Do not begin as a package task.

---

## 28. WP-PKG-WEB — Router and REST framework

**Priority:** P5  
**Complexity:** L  
**Dependencies:** HTTP server, JSON, URL; concurrency optional for first version

Packages:

- `stark-router`;
- `stark-web`;
- middleware;
- JSON request/response helpers;
- error mapping;
- body limits;
- logging hooks.

A single-threaded first version is acceptable after HTTP foundations close.

---

## 29. WP-PKG-SERDE — Typed encoding/decoding

**Priority:** P5  
**Complexity:** L–XL  
**Status:** Partly language/tooling blocked

Potential surface:

- `Encode<T>`;
- `Decode<T>`;
- JSON codecs;
- query/form codecs;
- generated or derived implementations.

Do not add reflection implicitly. Derive/code generation requires a separate language/tooling decision.

---

## 30. WP-PKG-DATABASE — Database connectivity

**Priority:** P5  
**Complexity:** L per driver  
**Dependencies:** provider ABI, networking/TLS where needed

Recommended order:

1. SQLite through a provider-backed native library;
2. PostgreSQL through a provider-backed driver;
3. Redis later.

Do not implement every wire protocol from scratch initially.

---

# PART G — Recommended execution order

## 31. Strict priority order

### Wave 0 — Close what already exists

1. `WP-PKG-Q1` — unified Tier-1 qualification for JSON, URL, Base64, Hex
2. `WP-PKG-Q2` — close `stark-io`
3. `WP-PKG-Q3` — consolidate/qualify `stark-time`

### Wave 1 — Immediate useful CLI foundation

4. `stark-env v0.1`
5. `stark-uuid v0.1` parse/format
6. `stark-semver v0.1`
7. `stark-checksum v0.1`
8. `stark-path v0.1`

### Wave 2 — Connected application foundation

9. `stark-random v0.1`
10. UUID v4/v7 generation
11. complete `stark-net v0.1`
12. `stark-http-core`
13. `stark-http-parser`
14. `stark-csv`

### Wave 3 — Platform breadth

15. DNS resolver
16. synchronous HTTP server/client
17. expanded environment/process information
18. `stark-process`
19. `stark-toml-lite`
20. `stark-glob`

### Wave 4 — Security and ecosystem expansion

21. TLS
22. regex-lite or parser combinators
23. SQLite provider
24. PostgreSQL provider

### Wave 5 — New language/runtime gates

25. structured threads
26. async/await
27. router/web framework
28. serde/derive
29. broader database ecosystem

---

## 32. Complexity and dependency summary

| Order | Package/WP | Priority | Complexity | Current state | Main dependency |
|---:|---|---|---|---|---|
| 1 | Package Tier-1 CI matrix | P0 | M | Needed | Existing packages |
| 2 | `stark-io` closure | P0 | S–M | Implemented | CI and edge fixes |
| 3 | `stark-time` closure | P1 | M | Partial/substantial | Existing provider |
| 4 | `stark-env` | P0 | S–M | Ready | Existing provider |
| 5 | `stark-uuid` parse/format | P1 | S–M | Ready | Pure byte/hex operations |
| 6 | `stark-semver` | P1 | M | Ready | Parser/comparison |
| 7 | `stark-checksum` | P1 | S | Ready | Bytes/integers |
| 8 | `stark-path` | P1 | M | Ready bounded | String operations |
| 9 | `stark-random` | P2 | M–L | Provider expansion | OS entropy |
| 10 | UUID generation | P2 | S–M | Dependency blocked | Random/time |
| 11 | `stark-net` completion | P2 | M–L | Partial | TCP provider |
| 12 | `stark-http-core` | P2 | M | Ready | URL |
| 13 | `stark-http-parser` | P2 | L | Ready bounded | HTTP core |
| 14 | `stark-csv` | P2 | M | Ready | String/Vec |
| 15 | DNS | P3 | M | Provider expansion | Networking |
| 16 | HTTP client/server | P3 | L | Dependency blocked | Net/parser |
| 17 | `stark-process` | P3 | L | Provider expansion | Pipes/resources |
| 18 | `stark-toml-lite` | P3 | M–L | Ready bounded | Parser primitives |
| 19 | `stark-glob` | P3 | S–M | Ready | Pure |
| 20 | TLS | P4 | L–XL | Provider expansion | Net/security policy |
| 21 | Threads | P4 | XL | Language blocked | Ownership/concurrency |
| 22 | Async | P5 | XL | Language blocked | Compiler/runtime |
| 23 | Web framework | P5 | L | Dependency blocked | HTTP |
| 24 | Serde/derive | P5 | L–XL | Tooling decision | Reflection/codegen |
| 25 | Databases | P5 | L each | Provider expansion | Platform maturity |

---

## 33. Parallel execution model

### Track A — Qualification and closure

- package Tier-1 CI;
- JSON/URL/Base64/Hex evidence;
- `stark-io` closure;
- `stark-time` qualification.

### Track B — Pure packages

- UUID parse/format;
- semver;
- checksum;
- path;
- CSV.

### Track C — Host-backed packages

- environment package;
- secure random provider;
- network completion;
- DNS;
- process package.

### Track D — Compiler/platform prerequisites

- `WP-C7.8-RB0`;
- vector/non-Copy collection ergonomics;
- future concurrency/async design.

Tracks may run in parallel only when they do not edit the same compiler-owned files. Shared-checkout commits must stage explicit owned paths.

---

## 34. Package acceptance template

Every package must record:

```text
Package name and version
Frozen public API
Language surface required
Host providers required
Determinism guarantees
Resource and allocation limits
Error model
Platform support
HIR qualification
MIR qualification
Native build qualification
Native artefact execution
Cross-package consumer
Known exclusions
Exact qualifying commit
```

A package is not `QUALIFIED` when:

- only `stark check` passes;
- only HIR tests pass;
- native build succeeds but the artefact is not executed;
- only one host platform was run while Tier-1 support is claimed;
- ignored tests are unclassified;
- the public API advertises types or functions with no executing path.

---

## 35. First useful application milestones

### Milestone A — STARK CLI Foundation

Requires:

- `stark-env`;
- `stark-io`;
- `stark-path`;
- `stark-checksum`;
- `stark-hex`;
- `stark-json`.

Deliverables:

- checksum CLI;
- JSON formatter/validator;
- log/file analyser;
- directory inventory tool.

### Milestone B — Connected STARK

Requires:

- `stark-net`;
- `stark-url`;
- `stark-http-core`;
- `stark-http-parser`;
- JSON.

Deliverables:

- simple HTTP server;
- HTTP client;
- JSON REST service;
- static file server with explicit path controls.

### Milestone C — Platform STARK

Requires:

- time;
- randomness;
- UUID;
- process;
- TLS.

Deliverables:

- production-shaped CLI utilities;
- secure identifiers;
- HTTPS client/server foundations;
- child-process tooling.

---

## 36. Immediate owner decisions

The following can begin without further architecture decisions:

```text
GO:
- unified package Tier-1 CI
- stark-env v0.1
- stark-uuid v0.1 parsing/formatting
- stark-semver v0.1
- stark-checksum v0.1
- stark-path v0.1 design and implementation
```

The following require a bounded work package before coding:

```text
SPEC FIRST:
- secure randomness provider
- process spawning
- DNS
- HTTP parser security profile
- TLS provider
```

The following must wait for language/runtime gates:

```text
WAIT:
- structured concurrency
- async/await
- derive/reflection-based serde
```

---

## 37. Recommended next action

Execute in this exact order:

1. add JSON, URL, Base64, and Hex to Tier-1 package CI;
2. close the remaining `stark-io` edge items;
3. implement `stark-env v0.1`;
4. implement `stark-uuid v0.1` parse/format;
5. implement `stark-semver`;
6. implement `stark-checksum`;
7. implement `stark-path`;
8. then start secure randomness and the HTTP foundation.

This gives STARK a coherent, demonstrable package baseline without waiting for concurrency or async.
