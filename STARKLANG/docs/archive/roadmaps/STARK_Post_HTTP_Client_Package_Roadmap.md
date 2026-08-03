> **ARCHIVED — SUPERSEDED 2026-08-03.** This roadmap was consolidated into the single
> forward plan at `ROADMAP.md` (repository root). It is retained for provenance and for
> the citations that point at it; **do not schedule work from it**. Where it disagrees
> with `ROADMAP.md`, `ROADMAP.md` wins.
>
> Former path: `starkc/docs/compiler/STARK_Post_HTTP_Client_Package_Roadmap.md`

---

# STARK Post-HTTP-Client Package Roadmap

**Status:** Proposed authoritative rebaseline  
**Assumption:** HC0–HC13 are complete, including TLS, HTTPS integration, JSON convenience, redirect policy, and Tier-1 qualification  
**Starting point:** The synchronous HTTP client stack is shipped and qualified; concurrency remains out of scope  
**Purpose:** Replace the stale pre-HTTP package ordering with an executable programme for the next useful application milestones

---

## 1. Rebaseline

The original package roadmap was written before the following existed:

- the unified first-party package qualification gate;
- executed-public-surface coverage;
- native resource lifecycle evidence;
- qualified DNS, TCP, HTTP core, serialization, parser, and plain HTTP client;
- cross-provider resource transfer;
- manifest-backed external-provider discovery;
- `stark-tls` over `rustls` + `aws-lc-rs`;
- a qualified synchronous HTTPS client.

Those items must no longer appear as future work.

### Assumed complete at entry

The post-HTTP baseline includes:

- `stark-json`
- `stark-url`
- `stark-base64`
- `stark-hex`
- `stark-uuid` parsing and formatting
- `stark-ascii`
- `stark-percent`
- `stark-mime`
- `stark-query`
- `stark-form`
- `stark-net`
- DNS resolution
- `stark-http-core`
- `stark-http-serialize`
- `stark-http-parser`
- `stark-http-client`
- `stark-tls`
- HTTPS scheme dispatch
- JSON request/response helpers
- redirect policy
- Tier-1 qualification on Linux x64, macOS arm64, and Windows x64

Any item above that has not actually reached the assumed exit state remains in the HTTP programme and must not be silently carried into this roadmap as complete.

---

## 2. Governing rules

Every package or provider must satisfy the existing package gate:

1. package checks;
2. package tests run;
3. format check;
4. consumer checks;
5. interpreter execution where the surface is pure;
6. native build;
7. produced native artifact executes;
8. every public callable is exercised;
9. every host resource has acquire/use/explicit-close/drop-close evidence;
10. exact qualifying commit and Tier-1 rows are recorded.

Additional rules:

- A successful build without executing the produced artifact is not qualification.
- A provider package must meet the same evidence bar whether first-party or external.
- No package task may quietly become a compiler redesign.
- A newly discovered compiler defect must be registered separately and fixed at the owning abstraction.
- External standards provide algorithm or format authority where one exists.
- Public APIs, error models, limits, and exclusions freeze before implementation.
- Package work must be driven by a named consumer or milestone.

---

## 3. Status vocabulary

| Status | Meaning |
|---|---|
| `CLOSE` | Implementation exists; evidence or surface closure remains |
| `READY` | Can be implemented against the current language and provider surface |
| `SPEC_FIRST` | Requires a bounded decision packet before code |
| `PROVIDER` | Requires native provider work but no new language feature |
| `PLATFORM` | Requires a compiler/runtime programme, not a normal package task |
| `DEFER` | Valuable but not on the immediate application path |

---

# Wave 0 — Reconcile and close existing foundations

**Goal:** Enter new package development with no ambiguous “implemented but not qualified” foundations.

## W0.1 Rebaseline the qualification matrix

Update the qualification script and package state to include every completed HTTP/TLS package and all previously existing host packages.

Required rows:

- `stark-env`
- `stark-io`
- `stark-time`
- `stark-random`
- `stark-tls`
- the full HTTP client stack

Exit:

- one authoritative list of first-party packages;
- zero package appearing as complete in prose but absent from the gate;
- zero stale blocked-surface records;
- all-target compiler and provider builds included.

## W0.2 Close `stark-env` v0.1 — `CLOSE`

Freeze and qualify:

- process arguments;
- environment lookup;
- absent versus present-empty values;
- UTF-8 failure;
- explicit output limits;
- Tier-1 native consumer.

Do not add mutation yet.

## W0.3 Close `stark-io` v0.1 — `CLOSE`

Finish:

- option-policy edge cases;
- checked timestamp conversions;
- exact durability wording;
- non-UTF-8 filename policy;
- expanded native surface evidence;
- Tier-1 lifecycle qualification.

## W0.4 Close `stark-time` v0.1 — `CLOSE`

Freeze:

- `Duration`;
- `Instant`;
- `UnixTimestamp`;
- checked arithmetic;
- wall and monotonic clocks;
- exact unavailable/overflow errors.

Time zones, locale formatting, cron, and async timers remain excluded.

## W0.5 Close `stark-random` v0.1 — `CLOSE`

Freeze the split:

- deterministic versioned PRNG;
- secure OS entropy;
- no secure-to-deterministic fallback;
- bounded output;
- Tier-1 provider failure evidence.

### Wave 0 exit

- Existing host foundations are fully represented in the gate.
- No later work depends on an unqualified package.
- The package state document is regenerated from executable evidence rather than manually inferred.

---

# Wave 1 — CLI and artifact foundation

**Goal:** Make STARK capable of building useful pipeline-friendly command-line tools and computing its own artifact identities.

This wave is the highest immediate leverage after the HTTP client.

## W1.1 `stark-sha2` v0.1 — `READY`

Scope:

- SHA-256 one-shot;
- incremental `update` / `finalize`;
- fixed 32-byte digest;
- rendering through `stark-hex`;
- file-hashing native consumer.

Required evidence:

- FIPS/NIST vectors;
- empty input;
- padding boundaries;
- block boundaries;
- at least 1 MiB input;
- one-shot versus incremental equivalence;
- HIR/MIR/native agreement.

SHA-512 remains v0.2 unless a consumer requires it.

## W1.2 Standard streams in `stark-io` v0.2 — `SPEC_FIRST`, then `PROVIDER`

Freeze before code:

- stdin/stdout/stderr are ambient capability functions, not closeable resources;
- byte-oriented provider boundary;
- partial-write semantics;
- `write_all`;
- flush;
- Core `print`/`println` interleaving rule;
- binary-mode behaviour on Windows.

Required consumer:

```text
stdin bytes -> transform -> stdout bytes
                         -> diagnostics on stderr
```

CI must pipe real bytes into the native artifact.

## W1.3 `stark-args` v0.1 — `READY`

Scope:

- long flags;
- long options with separate or `=` values;
- short flags without clustering;
- positional arguments;
- `--` terminator;
- unknown and missing-value errors;
- generated usage text.

Required consumers:

- checksum CLI;
- one HTTP diagnostic CLI.

## W1.4 `stark-checksum` v0.1 — `READY`

Scope:

- CRC32;
- Adler-32;
- incremental and one-shot forms;
- lowercase rendering via `stark-hex`.

Consumer:

```text
stark checksum <file>
stark checksum --sha256 <file>
cat file | stark checksum --sha256
```

The SHA-256 mode is supplied by `stark-sha2`, not reimplemented.

## W1.5 `stark-path` v0.1 — `READY`

Pure lexical operations:

- join;
- components;
- parent;
- filename;
- stem;
- extension;
- absolute check;
- lexical normalization;
- explicit Unix and Windows styles.

Filesystem canonicalization stays in `stark-io`.

## W1.6 `stark-semver` v0.1 — `READY`

Scope:

- SemVer 2.0.0 parsing and precedence;
- canonical formatting;
- exact, caret, tilde, and comparison-set requirements;
- exact parse diagnostics.

Required consumer:

- package/provider compatibility check.

## W1.7 `stark-bufio` v0.1 — `SPEC_FIRST`

First prove the exact abstraction the language can express.

Minimum acceptable v0.1:

- buffered file/stdin reader;
- `read_line`;
- lines longer than the buffer;
- LF and optional CRLF stripping;
- explicit writer flush;
- documented loss of unflushed data on Drop.

Do not force a generic iterator API if it conflicts with current ownership rules.

## W1.8 `stark-log` v0.1 — `READY` after stdio

Scope:

- `Error`, `Warn`, `Info`, `Debug`;
- explicit `Logger` value;
- stderr sink;
- deterministic no-timestamp mode;
- optional timestamp from `stark-time`;
- frozen line format.

No global logger, macros, rotation, or structured logging in v0.1.

### Milestone A — STARK CLI Foundation

Deliver:

- SHA-256/checksum CLI supporting file and stdin;
- JSON formatter/validator;
- log analyser using `BufReader`;
- directory inventory tool;
- shared argument parser and logger.

Acceptance:

- every tool works in a real shell pipeline;
- Linux/macOS/Windows artifacts execute;
- output is byte-compared;
- no tool hand-rolls argument parsing.

---

# Wave 2 — Identity, integrity, and package self-verification

**Goal:** Provide secure identifiers and content identities needed by package management and artifact-bound verification.

## W2.1 UUID v4 generation — `READY` after random closure

- secure randomness only;
- correct version/variant bits;
- no weak fallback;
- deterministic injected source for tests.

## W2.2 UUID v7 generation — `SPEC_FIRST`

Freeze:

- same-millisecond monotonicity policy;
- clock rollback behaviour;
- random-field source;
- ordering guarantee;
- overflow behaviour.

Do not bundle v7 into v4 merely because both produce UUIDs.

## W2.3 `stark-hmac` v0.1 — `READY` after SHA-2

Scope:

- HMAC-SHA256;
- RFC 4231 vectors;
- long-key handling;
- one-shot and incremental message input.

Security boundary:

- pure-STARK tag computation is allowed;
- timing-sensitive verification must not claim constant time unless implemented provider-side.

## W2.4 `stark-digest` v0.1 — `READY`

Naming ruling:

> Use `stark-digest` for content addressing. Keep `stark-checksum` for CRC/Adler classes.

Scope:

- digest bytes;
- canonical lowercase SHA-256 rendering;
- streaming `digest_file`;
- `digest_bytes`;
- `digest_string`;
- frozen interchange spelling.

Consumer:

- package-lock/artifact digest tool.

### Milestone B — Self-verifying artifacts

Deliver:

- canonical package and file digests;
- exact provider/package checksum recording;
- lockfile integrity validation;
- first self-hosted input to `stark verify`.

---

# Wave 3 — Operable server foundation

**Goal:** Complement the completed HTTP client with a bounded, single-threaded HTTP server that can shut down cleanly.

## W3.1 Signal/shutdown provider — `SPEC_FIRST`

Freeze:

- polling API: `shutdown_requested() -> Bool`;
- SIGINT/SIGTERM and Windows console control mapping;
- second-interrupt behaviour;
- fatal-signal limitation on resource Drop;
- capability declaration;
- CI mechanism that sends real signals.

No callbacks and no STARK code executing inside native signal handlers.

## W3.2 `stark-http-server` v0.1 — `SPEC_FIRST`

Initial scope:

- one connection at a time;
- bounded request size;
- bounded response size;
- HTTP/1.1;
- fixed-length request and response bodies;
- optional bounded keep-alive;
- explicit connection-close policy;
- handler returns a response value;
- clean listener and stream Drop;
- shutdown polling at accept/request boundaries.

Must freeze before implementation:

- public server API;
- handler representation;
- request body policy;
- keep-alive count and timeout;
- error mapping;
- malformed request response policy;
- graceful shutdown boundary.

Do not add concurrency, async, middleware, or routing in this package.

## W3.3 `stark-router` v0.1 — `READY` after server API freezes

Pure package:

- method + path matching;
- static segments;
- named parameters;
- bounded route count;
- deterministic precedence;
- no regex routes;
- no hidden global registry.

### Milestone C — Connected STARK

Deliver:

- HTTPS client;
- single-threaded HTTP server;
- JSON REST service;
- static file server with explicit path controls;
- clean Ctrl-C shutdown observed in Tier-1 CI.

---

# Wave 4 — Database foundation

**Goal:** Build the first provider ecosystem proof beyond first-party networking.

External provider discovery is an entry requirement. No database provider may require compiler-source registration.

## W4.1 `stark-db-core` v0.1 — `SPEC_FIRST`, pure STARK

Freeze the relational contract:

```text
DbValue
DbError
ConnectionConfig
Query
Parameters
Row
Rows/Cursor
Transaction
```

Required decisions:

- null representation;
- integer and floating conversion rules;
- text and byte values;
- timestamp boundary;
- parameter binding;
- row lookup by index/name;
- cursor ownership;
- transaction commit/rollback/drop;
- timeout and cancellation exclusions;
- pooling exclusion.

MongoDB/document APIs are separate and must not be forced into this model.

## W4.2 SQLite provider — `SPEC_FIRST`, then `PROVIDER`

Purpose:

- prove an external provider can be discovered, validated, linked, and qualified without compiler edits;
- provide a local database useful for tools and applications.

Initial scope:

- open database;
- prepare/execute/query;
- positional parameters;
- rows cursor;
- transaction;
- structured errors;
- exactly-once close.

No ORM, migrations framework, pooling, or async.

## W4.3 PostgreSQL provider — `SPEC_FIRST`, then `PROVIDER`

Initial scope:

- connection string/config;
- TLS policy;
- parameterized queries;
- rows cursor;
- transactions;
- bounded result handling;
- structured server and transport errors.

No wire-protocol implementation in STARK for v0.1; use a mature native driver behind the provider boundary.

### Milestone D — Data-backed STARK

Deliver:

- SQLite-backed CLI application;
- PostgreSQL-backed JSON service;
- transactions and parameterized queries;
- provider package installed outside the compiler repository;
- no compiler-source changes for either driver.

---

# Wave 5 — Native cryptography packages

**Goal:** Extend the CD-361 `rustls` + `aws-lc-rs` foundation into deliberate application cryptography.

This wave does not block the CLI, server, or database milestones.

## W5.1 `stark-aead` v0.1 — `SPEC_FIRST`

Freeze:

- algorithm/profile;
- affine key resource;
- nonce generation and limits;
- ciphertext representation;
- associated data;
- authentication failure shape;
- zeroization claim;
- Profile N/Profile F separation.

## W5.2 `stark-signature` v0.1 — `SPEC_FIRST`

Freeze:

- selected algorithm per profile;
- signing-key resource;
- verification-key value/resource boundary;
- signature encoding;
- deterministic/random signing policy;
- malformed versus invalid collapse;
- zeroization and export rules.

## W5.3 `stark-kdf` v0.1 — `SPEC_FIRST`

Initial scope:

- HKDF-SHA256 for high-entropy input only;
- explicit context/info;
- bounded output;
- no passwords.

Password hashing and password storage are separate work and remain excluded.

## W5.4 `stark-key-agreement` v0.1 — `DEFER` unless a consumer requires it

Only begin for a named protocol or application.

### Milestone E — Secure application primitives

Deliver:

- provider-backed AEAD;
- signing and verification;
- key derivation;
- Profile N qualification;
- separately reported Profile F evidence where supported.

---

# Wave 6 — Process and data tooling

These items may run in parallel with Waves 4–5 when ownership is separate.

## W6.1 Expanded environment/process information — `SPEC_FIRST`

Potential v0.2 surface:

- current directory;
- executable path;
- platform;
- architecture;
- process ID.

Changing current directory and environment mutation remain deferred unless a concrete consumer requires them.

## W6.2 `stark-process` v0.1 — `SPEC_FIRST`

Freeze:

- child resource;
- spawn without shell interpretation;
- argument vector;
- environment inheritance;
- PATH-search policy;
- wait/status;
- bounded stdout/stderr capture;
- pipe deadlock avoidance;
- cancellation and Drop.

## W6.3 `stark-csv` v0.1 — `READY`

- one-byte delimiter;
- quoting and escaped quotes;
- CRLF/LF;
- row/column diagnostics;
- parser and encoder;
- streaming v0.2 after bufio proves the shape.

## W6.4 `stark-toml-lite` v0.1 — `READY_BOUNDED`

Only implement if package tooling needs it.

- strings;
- integers;
- booleans;
- arrays;
- tables;
- duplicate-key rejection.

## W6.5 `stark-glob` v0.1 — `READY`

Pure matching only:

- `*`;
- `?`;
- bounded character classes;
- explicit separator policy.

Filesystem traversal belongs elsewhere.

## W6.6 Parser combinators before regex — `DEFER`

Prefer a bounded parser-combinator package before a regex engine unless a concrete consumer requires regex.

A regex package requires:

- grammar;
- complexity bound;
- denial-of-service policy;
- matching semantics;
- Unicode policy.

---

# Wave 7 — Application ergonomics

## W7.1 `stark-web` v0.1 — `SPEC_FIRST`

Build only after server and router qualification.

Scope:

- request extraction;
- JSON request/response;
- error mapping;
- body limits;
- middleware as explicit composition;
- logging hooks.

No async requirement for v0.1.

## W7.2 Typed serialization/code generation — `PLATFORM`

Do not add implicit reflection.

Requires a separate tooling decision for:

- generated `Encode` / `Decode`;
- JSON codecs;
- field naming;
- schema evolution;
- derive/code-generation mechanism;
- reproducible generated artifacts.

---

# Wave 8 — New language/runtime gates

These are not package tasks.

## W8.1 Structured threads — `PLATFORM`

Requires:

- values crossing thread boundaries;
- shared ownership;
- scoped borrows;
- thread-safe Drop;
- join/trap propagation;
- mutexes/channels/atomics.

## W8.2 Async/await — `PLATFORM`

Requires:

- future layout;
- suspension ownership;
- cancellation;
- suspended Drop;
- executor/reactor model.

Do not block the post-HTTP package programme on either gate.

---

## 4. Recommended strict order

```text
0. Rebaseline qualification and close env/io/time/random
1. SHA-2
2. Standard streams
3. Args + checksum + path + semver
4. Bufio + log
5. UUID v4, then v7 decision
6. HMAC + content addressing
7. Signal/shutdown
8. HTTP server
9. Router
10. DB core contract
11. SQLite external provider
12. PostgreSQL external provider
13. AEAD / signatures / KDF
14. Process package
15. CSV / TOML-lite / glob
16. Web framework
17. Serialization tooling
18. Threads and async only as separate platform programmes
```

---

## 5. Parallel execution lanes

### Lane A — Pure packages

- SHA-2
- args
- checksum
- path
- semver
- UUID
- HMAC
- digest
- CSV
- glob

### Lane B — Existing provider closure

- env
- io
- time
- random
- standard streams
- signals

### Lane C — Applications

- HTTP server
- router
- CLI milestone applications
- database consumers

### Lane D — External providers

- SQLite
- PostgreSQL
- later native crypto providers

### Lane E — Design packets

- bufio abstraction
- UUID v7 policy
- signals
- HTTP server API
- database core
- process
- AEAD/signature/KDF
- web/serialization

Lanes may run in parallel only when they do not edit the same compiler-owned files or qualification machinery.

---

## 6. Required execution packet template

The roadmap chooses and sequences work. Each implementation must still receive a bounded packet containing:

1. verified current state;
2. exact package name and version;
3. frozen public API;
4. exact error types;
5. limits and deterministic rules;
6. dependencies;
7. provider ABI, if any;
8. ownership and Drop semantics;
9. required files/manifests;
10. positive tests;
11. negative tests;
12. native consumer;
13. Tier-1 qualification commands;
14. forbidden compiler/platform changes;
15. exit evidence and exact qualifying commit.

No `SPEC_FIRST` item may begin coding before this packet is approved.

---

## 7. Programme checkpoints

### Checkpoint 1 — CLI-capable language

Complete through Wave 1.

STARK can build portable pipeline-friendly tools and hash its own artifacts.

### Checkpoint 2 — Operable connected language

Complete through Wave 3.

STARK can act as both HTTPS client and bounded HTTP server with graceful shutdown.

### Checkpoint 3 — Application platform

Complete through Wave 4.

STARK can build database-backed services using external native providers without compiler changes.

### Checkpoint 4 — Security platform

Complete through Wave 5.

STARK exposes deliberate provider-backed application cryptography with separate Profile N/F evidence.

### Checkpoint 5 — Ecosystem ergonomics

Complete through Wave 7.

STARK has process tooling, common data formats, routing, web helpers, and an explicit serialization strategy.

---

## 8. Items explicitly not on the critical path

- HTTP/2 and HTTP/3
- WebSockets
- connection pooling
- async networking
- ORM
- full MongoDB/document API
- automatic database migrations
- full regex engine
- dynamic provider loading
- native-provider sandboxing
- client certificates
- global logger
- environment mutation
- package-manager registry service
- threads
- async/await

These require named consumers or separate platform decisions.

---

## 9. Immediate next packets after HC13

Produce these in order:

1. `WP-PKG-REBASELINE-CLOSURE`
2. `WP-PKG-SHA2-V01`
3. `WP-PKG-STDIO-V02`
4. `WP-PKG-ARGS-V01`
5. `WP-PKG-CHECKSUM-V01`
6. `WP-PKG-PATH-V01`
7. `WP-PKG-SEMVER-V01`
8. `WP-PKG-BUFIO-V01`
9. `WP-PKG-LOG-V01`
10. `WP-PKG-SIGNAL-V01`

SHA-2, args, checksum, path, and semver may be implemented in parallel after their packets freeze. Standard streams and signals require provider-owned lanes.
