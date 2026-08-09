# STARK Consolidated Roadmap

## August 2026 – February 2027

**Status:** ACTIVE — this is the single forward-looking roadmap for the project.
**Adopted:** 2026-08-03
**Supersedes:** all prior package, ecosystem and application roadmaps (see §0.2).

---

# 0. Scope boundary and document authority

## 0.1 What this document governs

This roadmap is the **only** live forward plan for STARK's package, application and
platform work. Where any other document proposes future package ordering, ecosystem
phases or application milestones, this document wins.

Two governance tracks remain **live alongside** it and are **not** superseded:

| Track | Document | Why it is still live |
| --- | --- | --- |
| Compiler gates C0–C10 | `STARKLANG/docs/compiler/COMPILER-ROADMAP.md` (+ `COMPILER-CHARTER.md`, `COMPILER-STATE.md`) | **Updated 2026-08-09 (CD-395).** Gate **C10 (release qualification) is OPEN**; C9 Part A is closed and **Part B is DEFERRED** pending second-artifact evidence, which does **not** block C10. This roadmap does not define C9/C10 exit criteria, and §11 below assumes a functioning compiler-correctness lane. |
| ~~Architecture stabilization~~ | `STARKLANG/docs/compiler/work-packages/WP-ARCHITECTURE-STABILIZATION.md` | **COMPLETE 2026-08-09 — all four sprints CLOSED; Campaign A PASS, Campaign B EXITED PASS.** The row previously read *"Sprints 1 and 2 are complete; AS3 and AS4 remain"*, which was true when written and is now stale — corrected under CD-395 (OD-6). **§6.0's binding entry gate on Phase 4 is therefore SATISFIED**; the gate text itself stands unrewritten, with the satisfaction recorded at §6.0. The programme is retained as a historical record and schedules no further work. |
| ~~HTTP client HC0–HC13~~ | `STARKLANG/docs/compiler/work-packages/WP-HTTP-CLIENT-ROADMAP.md` | **CLOSED 2026-08-03 (CD-375, corrected CD-376).** HC13 delivered the Tier-1 qualification this row was waiting on, so §1's "completed platform capability" claim is now paid for. The track is no longer live. What it did NOT settle is release readiness — DEV-165 (`connect_timeout` accepted and ignored) still blocks a public release, and does not belong to this track. The installer blocker has narrowed rather than cleared — see §1. |

Current compiler position always comes from `COMPILER-STATE.md` (repo root), never from
this file.

`STARKLANG/docs/ROADMAP.md` and `STARKLANG/docs/PLAN.md` are the **historical record** of
the pre-existing Gate 1–7 sequence (all closed). They are retained for their citations and
non-goals, not as forward plans.

Gate 7's project-wide **RETAIN AS RESEARCH LANGUAGE** policy — which those documents cite —
was **superseded on 2026-08-04** by `starkc/docs/gate7-superseded.md`; the adoption of this
roadmap is one of the things that retired it. Gate 7's *tensor-track* verdicts are untouched
and still govern that track: technical POSITIVE, **productisation DEFER** pending
external-developer evidence that has not been gathered. Nothing in this roadmap authorises
tensor productisation, and §12's non-goals continue to apply.

## 0.2 Superseded documents

The following were consolidated into this roadmap and moved to
`STARKLANG/docs/archive/roadmaps/`:

| Archived document | What it covered |
| --- | --- |
| `STARK_Post_HTTP_Client_Package_Roadmap.md` | Direct predecessor draft of this document |
| `WP-PKG-ROADMAP.md` | Remaining first-party package waves after Gate C7 |
| `WP-PKG-OPS-ROADMAP.md` | Identity and operability layer companion |
| `STARK-Standard-Package-Roadmap.md` | Standard-package phases P0–Pn |
| `CORE_PACKAGES_ECOSYSTEM_ROADMAP.md` | Core modules, registry and third-party proposal |
| `SYSTEMS-ROADMAP.md` | Systems/ecosystem S0 outline |

Their content is preserved for provenance. Do not schedule work from them.

---

# 1. Strategic position

STARK has completed its initial architecture-validation phase.

The language has demonstrated that its central architecture can support:

* native compilation through generated Rust;
* ownership and deterministic destruction;
* place-sensitive move semantics;
* manifest-driven native providers;
* affine host resources;
* cross-provider ownership transfer;
* DNS and TCP networking;
* verified TLS;
* HTTP/1.1 client functionality;
* JSON request and response handling;
* bounded redirects and protocol hardening;
* multi-platform package qualification.

The HTTP client track is a completed platform capability. The HC13 qualification debt §0.1
recorded is **paid**: 42 executed cases against controlled adversarial peers on three Tier-1
platforms, five evidence documents, and four defects found and fixed in the closing packets
(DEV-163, DEV-164, SEC-HTTP-001, SEC-HTTP-002).

Completed is not the same as releasable. `HC13-RELEASE-CHECKLIST.md` §0 lists what still blocks a
public release, and none of it is HTTP work:

```text
Installer Phase I / compiler distribution   IMPLEMENTED   tar.gz + pkg/deb/zip, versioned
                                                          install, uninstall, `stark doctor`
Standalone first-party toolchain            PARTIAL       the payload carries the compiler,
                                                          runtime and provider ABI -- not the
                                                          first-party package/provider set
Offline package/provider build              NOT PROVEN    a clean machine cannot yet build an
                                                          HTTP/TLS program without obtaining the
                                                          packages separately
Public signed distribution                  NOT PROVEN    the manifest establishes INTEGRITY,
                                                          not AUTHENTICITY -- see below
DEV-165                                     OPEN          connect_timeout accepted and ignored
```

**Integrity is not authenticity.** `stark doctor` re-hashes every payload file against
`manifest.json`, which detects corruption and a partial extraction. It does not establish that the
manifest came from a STARK release: anyone who can replace the payload can replace the manifest and
the sidecar with it. A public distribution needs a signed manifest, a trusted release key,
signature verification before installation, and platform notarisation.

The next phase is not another continuous compiler sprint. It is a measured programme to turn
STARK into a usable application platform with:

* command-line tooling;
* secure identity and artifact packages;
* a REST API server;
* structured concurrency — **gated on Campaign A, see §6.0**;
* persistent storage;
* distribution and documentation.

---

# 2. Development model

## 2.1 Sustainable cadence

The project should move from daily high-intensity development to monthly outcomes.

Recommended monthly rhythm:

```text
Week 1     design, probes and scope freeze
Week 2–3   implementation and native consumers
Week 4     qualification, review, documentation and examples
```

A consolidation-only month is acceptable after a large compiler or runtime programme.

## 2.2 Work-in-progress limits

At any point, limit active work to:

```text
1 major compiler/runtime programme
2–3 supporting packages
1 proving application
```

Do not run several cross-cutting compiler campaigns simultaneously.

## 2.3 Application-led packages

New packages should be justified by one of:

* a real proving application;
* a concrete user workflow;
* an existing package dependency;
* a missing platform capability;
* a compiler/runtime proof requirement.

Avoid implementing packages merely to imitate another language's standard library.

## 2.4 Closure standard

A package or runtime feature is not complete merely because it type-checks.

Depending on scope, closure should require:

* package tests;
* malformed-input tests;
* HIR interpreter execution;
* MIR interpreter execution;
* native debug execution;
* native release execution;
* exact stdout/stderr checks;
* resource-lifecycle evidence;
* supported Tier-1 platform rows;
* written exclusions and limits.

---

# 3. Phase 1 — Operability Foundation

## August–September 2026

## Objective

Make STARK practical for command-line applications and shell workflows.

The language can communicate over verified HTTPS. It now needs the ordinary capabilities
surrounding a complete application.

---

## 3.1 Environment and arguments

### `stark-env`

Complete and qualify:

* command-line arguments;
* environment-variable lookup;
* current working directory where appropriate;
* explicit missing-variable result;
* bounded allocation;
* strict UTF-8 behaviour.

### `stark-args`

Implement a pure-STARK argument parser above `stark-env`.

Initial scope:

* positional arguments;
* short flags;
* long flags;
* required option values;
* optional values;
* defaults;
* repeated options where explicitly admitted;
* `--` terminator;
* generated help text;
* deterministic errors.

Avoid derive macros or reflection-based argument parsing in the first version.

---

## 3.2 Standard streams

Extend the I/O packages with:

* stdin;
* stdout;
* stderr;
* byte reads and writes;
* strict text helpers;
* explicit flush;
* EOF distinction;
* broken-pipe handling;
* bounded read-to-end.

Target shell workflows:

```bash
cat input.json | stark-json format
stark-hash archive.bin
stark-get https://example.test/data
```

---

## 3.3 Buffered I/O

### `stark-bufio`

Provide:

* buffered reader;
* buffered writer;
* read line;
* read until delimiter;
* bounded internal buffers;
* explicit flush;
* deterministic EOF behaviour.

No helper should silently read an unbounded stream into memory.

---

## 3.4 Paths and filesystem ergonomics

### `stark-path`

Start with pure lexical operations:

* construct;
* join;
* parent;
* filename;
* extension;
* absolute/relative classification;
* platform separator handling;
* lexical normalization.

Filesystem canonicalisation should remain a provider-backed operation because it depends on
actual host state, symlinks and permissions.

---

## 3.5 Time consolidation

Complete the time package around:

* checked `Duration`;
* monotonic `Instant`;
* wall-clock timestamp;
* elapsed time;
* deadline construction;
* Unix timestamp conversion;
* comparison;
* checked arithmetic;
* bounded formatting.

Calendar, locale and timezone databases should remain outside the first package unless
required by a real application.

---

## 3.6 Hashing and digest utilities

### `stark-sha2`

Initial scope:

* SHA-256;
* one-shot hashing;
* incremental hashing;
* byte-slice hashing;
* file hashing;
* official fixed vectors;
* identical results across engines.

### `stark-digest`

Provide:

* fixed-length digest values;
* hexadecimal encoding and parsing;
* digest comparison;
* algorithm-tagged values;
* content identifier helpers.

---

## 3.7 Structured logging

### `stark-log`

Initial scope:

* levels;
* timestamp;
* target/module;
* message;
* key-value fields;
* stdout and stderr sinks;
* deterministic text output;
* optional JSON-lines output;
* explicit flush.

Prefer passed logger/configuration values over unrestricted global mutable state.

---

## 3.8 Proving applications

Build and qualify:

### `stark-json`

* validate JSON;
* pretty-print;
* compact;
* read stdin or file;
* write stdout;
* bounded input.

### `stark-hash`

* hash file or stdin;
* output canonical hex;
* verify an expected digest;
* meaningful exit status.

### `stark-get`

* HTTP/HTTPS GET;
* request headers;
* response body to stdout;
* timeout controls;
* response-size controls;
* optional strict JSON parsing.

---

## Phase 1 outcome

STARK can build useful shell-native programs without custom application-specific providers.

---

# 4. Phase 2 — Security, Identity and Artifact Integrity

## October 2026

## Objective

Provide the cryptographic and identity primitives needed by applications, package tooling and
reproducible artefacts.

---

## 4.1 Secure randomness

### `stark-random`

Provider-backed secure randomness:

* fill a caller-provided buffer;
* bounded output;
* explicit failure;
* no partial-success ambiguity;
* no fallback to deterministic randomness;
* no weak-random mode.

A deterministic pseudorandom generator should use a separate package, type and API.

---

## 4.2 UUID completion

Extend the existing UUID package with:

* UUID v4 generation;
* UUID v7 generation;
* canonical parsing;
* canonical formatting;
* version inspection;
* variant inspection;
* malformed-input refusal;
* v7 ordering tests.

UUID v4 must depend on secure randomness.

UUID v7 must define its behaviour when several identifiers are generated during the same
timestamp interval.

---

## 4.3 HMAC

### `stark-hmac`

Initial scope:

* HMAC-SHA256;
* one-shot API;
* incremental API where useful;
* constant-time verification;
* fixed vectors;
* explicit key ownership.

Ordinary equality must not be the recommended authentication-tag verification path.

---

## 4.4 Content addressing

### `stark-content-id`

Provide:

* algorithm-tagged identifiers;
* byte hashing;
* file hashing;
* canonical text encoding;
* parsing;
* deterministic serialization;
* optional metadata envelopes.

This becomes a foundation for build caches, package artifacts and evidence records.

---

## 4.5 Native build provenance

Record in native build metadata:

* compiler version;
* MIR version;
* runtime version;
* backend version;
* source-content hash;
* package graph hash;
* target triple;
* build profile;
* provider identities;
* provider versions;
* provider hashes.

Add an inspection command:

```bash
stark inspect-build ./program
```

---

## 4.6 Proving application

### `stark-artifact`

Capabilities:

* hash a file;
* create a JSON evidence record;
* verify an existing record;
* download an artifact over verified HTTPS;
* verify expected size and digest;
* return meaningful exit status.

---

## Phase 2 outcome

STARK can securely identify, retrieve and verify artifacts using its own ecosystem.

---

# 5. Phase 3 — REST API Server v0.1

## November 2026

## Objective

Deliver a complete synchronous REST API server before introducing concurrency.

The first version may process one connection at a time. This is an intentional semantic and
package milestone, not a scalability claim.

---

## 5.1 TCP listener

Extend `stark-net` with:

* `TcpListener`;
* bind;
* listen;
* accept;
* local address;
* peer address;
* accept timeout or polling interval;
* shutdown;
* exactly-once listener release;
* exactly-once accepted-stream release.

Resource ownership must remain affine and explicit.

---

## 5.2 Shutdown capability

Introduce minimal shutdown handling.

Recommended first model:

```stark
while !shutdown_requested() {
    match listener.accept() {
        Ok(stream) => handle_connection(stream),
        Err(error) => handle_accept_error(error),
    }
}
```

Requirements:

* polling-based;
* no arbitrary host-thread callbacks into STARK;
* ordinary control flow performs cleanup;
* Ctrl-C/termination where supported;
* explicit unsupported-platform result;
* no detached background handler.

---

## 5.3 HTTP server core

### `stark-http-server`

Initial scope:

* HTTP/1.1;
* request line parsing;
* request headers;
* fixed-length request bodies;
* bounded chunked request bodies where safely supported;
* configurable header count and size limits;
* configurable body-size limit;
* read/write/request timeouts;
* response serialization;
* structured parse errors;
* connection-close policy;
* bounded or explicitly excluded keep-alive.

Explicitly defer:

* HTTP/2;
* WebSockets;
* unrestricted streaming bodies;
* async handlers;
* middleware frameworks;
* automatic unbounded buffering.

---

## 5.4 Router

### `stark-router`

Initial scope:

* GET;
* POST;
* PUT;
* PATCH;
* DELETE;
* static paths;
* path parameters;
* query access;
* deterministic route precedence;
* 404;
* 405;
* structured JSON errors.

Target API shape:

```stark
let mut router = Router::new();

router.get("/health", health);
router.get("/users/:id", get_user);
router.post("/users", create_user);

serve(listener, router);
```

Use callable forms that STARK already supports. Do not introduce closures, trait objects or
reflection solely to copy another framework's syntax.

---

## 5.5 JSON REST helpers

Add:

* parse bounded JSON request body;
* JSON response constructor;
* content-type handling;
* error response envelope;
* status-code helpers;
* request ID support where practical.

---

## 5.6 Static-file support

Optional bounded first version:

* configured root;
* lexical path containment;
* traversal refusal;
* explicit index-file policy;
* MIME lookup;
* maximum file size;
* no directory listing by default.

---

## 5.7 Proving application

### `stark-notes`

A complete local REST service:

* `GET /health`;
* `POST /notes`;
* `GET /notes`;
* `GET /notes/:id`;
* JSON request and response bodies;
* in-memory or simple file persistence;
* request limits;
* timeouts;
* structured logs;
* clean shutdown.

---

## Phase 3 outcome

STARK can host a bounded, useful REST API with deterministic ownership and cleanup.

---

# 6. Phase 4 — Structured Concurrency

## December 2026

## 6.0 Entry gate — Campaign A must be green (BINDING, 2026-08-07)

> **Structured-concurrency compiler/runtime work may not begin until Campaign A exits green:
> AS0, AS1a, AS2, AS1b, AS3 and AS4 closed and owner-reviewed.**

This is a **binding platform gate**, not a proposal. It was approved on 2026-08-07 after two sprints
of `WP-ARCHITECTURE-STABILIZATION.md` produced the evidence for it.

> ### GATE SATISFIED — 2026-08-09 (CD-395, OD-6)
>
> **Campaign A exited PASS**, and Campaign B exited PASS after it; all four sprints of
> `WP-ARCHITECTURE-STABILIZATION.md` are CLOSED. AS0, AS1a, AS2, AS1b, AS3 and AS4 are closed and
> owner-reviewed, which is exactly what this gate asked for.
>
> **The requirement above is not removed and not rewritten.** It stands as the decision that was
> made, and this note records that it has since been met — two records, not one edited record. A
> reader arriving at Phase 4 needs both: what was demanded, and that it was delivered.
>
> Evidence: `audits/CAMPAIGN-A-EXIT-REPORT.md`, `audits/CAMPAIGN-B-EXIT-REPORT.md`,
> `audits/AS-SPRINT4-CLOSEOUT.md`. Phase 4 is unblocked by this gate; it remains subject to every
> other constraint in §6 and to `ROADMAP.md` §2.2's work-in-progress limits.

### Why the evidence justifies a gate

AS0 inventoried **six** pipeline assemblies bypassing the shared driver, several parallel provenance
authorities, and semantic authorities still uninventoried. Sprint 2 then found **four** defects while
consolidating just two authorities — including `TRAIT-COHERENCE-001`'s cross-package clause, a
normative language rule that had **effectively never worked as specified** (DEV-183). The recurrent
architectural risk this programme names is demonstrated, not theoretical.

### Why AS3 and AS4 specifically

They are the two foundations concurrency would amplify.

- **AS3** closes the callable-use / generic-instantiation gap that currently prevents total
  value-representation enforcement. That work is *already paused* because implicit callable dispatch
  does not publish enough semantic information.
- **AS4** establishes one authority for `Copy`, drop, borrow/reference containment and related type
  properties.

Structured concurrency depends immediately on exactly those facts:

```text
Can this value cross into a task?
Is it moved or borrowed?
Does a generic callable capture the instantiated type correctly?
Who owns an affine resource after spawn?
When does its Drop run?
What survives cancellation?
What must join restore?
Can a reference outlive the spawning scope?
```

Building task semantics while those answers still have several authorities would make concurrency
another **producer** of compensating mechanisms — the opposite of what the stabilization programme
is for.

### What the gate does NOT block

Ordinary package and platform work continues. This roadmap already places a **synchronous** REST
server before concurrency (§5), deliberately. The gate blocks the compiler/runtime concurrency
campaign and nothing else: packages, tooling, documentation and the synchronous server are
unaffected.

### Sequencing

```text
Sprint 2 green
    ↓
Sprint 3  ── AS3 callable authority
              ↓ semantic-complete checkpoint
           ── AS4 type-property authority
              ↓
           ── remaining Campaign-A AS0 inventories
    ↓
CAMPAIGN A GREEN
    ↓
structured-concurrency compiler/runtime work permitted
```

### Where Campaign A's status lives

`WP-ARCHITECTURE-STABILIZATION.md`'s approval-and-status table, together with the per-sprint
closeout reports under `STARKLANG/docs/compiler/audits/`. **On `develop` those documents lag the
execution**: the programme runs on `wp-arch-stability/sprint-N` branches which are deliberately not
merged, so `develop`'s copy of the work package is the original proposal. Read the status from the
branch carrying the current sprint, the same way current compiler position always comes from
`COMPILER-STATE.md` rather than from this file.

This gate is discharged by that table recording AS0, AS1a, AS2, AS1b, AS3 **and** AS4 all closed and
owner-reviewed — not by any single sprint closing, and not by Campaign A's implementation being
finished without the review.

---

## Objective

Introduce bounded concurrency without immediately committing the language to async/await.

The recommended first model is structured native threads.

---

## 6.1 Concurrency principles

The first concurrency system should guarantee:

* every spawned task belongs to a visible scope;
* no detached tasks;
* explicit join;
* bounded task/thread creation;
* owned values may move into tasks;
* borrows cannot outlive their scope;
* affine resources transfer exactly once;
* traps and failures propagate visibly;
* cleanup occurs on success and failure;
* shutdown can wait for active work.

---

## 6.2 Scoped threads

### `stark-thread`

Conceptual target:

```stark
scope(|scope| {
    let task = scope.spawn(move || {
        process(value)
    });

    let result = task.join()?;
});
```

The exact syntax should follow language capabilities, but semantics must include:

* lexical scope;
* join handle;
* move capture;
* borrowed capture only when statically scoped;
* task result;
* task failure;
* bounded spawn failure;
* no detached thread in v0.1.

---

## 6.3 Join and failure semantics

Define explicitly:

* successful task result;
* STARK trap inside a child;
* provider failure inside a child;
* host-thread panic containment;
* parent abandoning a scope;
* multiple child failures;
* join order;
* cleanup ordering.

A child failure must not silently terminate the entire process without a classified outcome.

---

## 6.4 Channels

### `stark-channel`

Start with bounded channels:

* sender;
* receiver;
* send;
* receive;
* try-send;
* try-receive;
* close;
* capacity;
* blocked-operation shutdown behaviour.

Requirements:

* affine endpoints where appropriate;
* non-Copy values transferable;
* no duplicate ownership;
* sender/receiver Drop behaviour;
* bounded memory.

Unbounded channels should not be part of v0.1.

---

## 6.5 Synchronisation

### `stark-sync`

Initial primitives:

* mutex;
* scoped guard;
* condition variable only if required;
* atomic boolean/counter;
* cancellation token.

Read/write locks can be deferred until a proving workload needs them.

The API must prevent a lock guard escaping the protected scope.

---

## 6.6 Worker pool

Build a fixed worker-pool abstraction:

* configured worker count;
* bounded job queue;
* backpressure;
* shutdown;
* drain active jobs;
* reject new jobs after shutdown;
* visible failed jobs.

This will become the concurrency foundation for the REST server.

---

## 6.7 Required compiler/runtime proofs

Before closure, prove:

* owned non-Copy values cross thread boundaries safely;
* references cannot outlive their spawning scope;
* affine host resources cannot be duplicated;
* socket ownership may move to a worker exactly once;
* child trap/failure does not double-drop resources;
* channel shutdown releases queued values;
* worker-pool shutdown drains or explicitly cancels work;
* generated native code agrees with STARK ownership rules.

---

## Phase 4 outcome

STARK has a bounded structured-concurrency model suitable for practical applications without
requiring async/await.

---

# 7. Phase 5 — Concurrent REST API Server

## January 2027

## Objective

Upgrade the synchronous REST server into a bounded concurrent service.

The preferred first architecture is a fixed worker pool rather than unbounded
thread-per-connection spawning.

---

## 7.1 Server architecture

```text
TcpListener
    ↓
accept loop
    ↓
bounded connection queue
    ↓
fixed worker pool
    ↓
request parsing and routing
    ↓
response
```

Properties:

* configurable worker count;
* bounded pending-connection queue;
* bounded per-connection resources;
* overload refusal;
* no unbounded spawning;
* graceful shutdown;
* active-request draining;
* exactly-once socket release.

---

## 7.2 Overload policy

Define explicit behaviour when capacity is exhausted:

* refuse immediately;
* bounded wait;
* return service-unavailable response where possible;
* close connection cleanly;
* record structured overload log.

No hidden unbounded queue.

---

## 7.3 Graceful shutdown

Required sequence:

```text
stop accepting
→ reject or drain queued connections
→ signal workers
→ finish bounded active requests
→ close remaining resources
→ join workers
→ exit
```

Define timeout behaviour when requests do not complete during shutdown.

---

## 7.4 Request isolation

A failed request handler must not corrupt:

* listener state;
* worker pool;
* unrelated requests;
* shared application state;
* connection accounting.

Document whether a STARK trap ends:

* the request;
* the worker;
* the server process.

The initial design should prefer containing failure to the request/worker where safely
possible.

---

## 7.5 REST middleware primitives

Add bounded, explicit middleware support for:

* request IDs;
* access logging;
* CORS;
* authentication extraction;
* body-size enforcement;
* response headers;
* error mapping;
* timing.

Avoid an overly generic middleware abstraction if static composition is sufficient.

---

## 7.6 Authentication helpers

Initial packages may include:

* bearer-token extraction;
* basic authentication parsing;
* constant-time token comparison;
* signed request verification where HMAC is available.

Do not build a full identity platform during this phase.

---

## 7.7 Proving application

Upgrade `stark-notes` or build a project/evidence API with:

* concurrent clients;
* bounded worker pool;
* JSON REST endpoints;
* authentication;
* structured logs;
* graceful shutdown;
* persistent storage;
* artifact attachments;
* hash verification.

---

## Phase 5 outcome

STARK supports a bounded concurrent REST API server suitable for local services and moderate
production workloads.

---

# 8. Phase 6 — Data and Persistence

## January–February 2027

This phase may overlap with the concurrent server only after structured concurrency is stable.

## Objective

Support meaningful stateful applications.

---

## 8.1 SQLite

Introduce SQLite through a provider-backed package.

Initial scope:

* open database;
* prepare statement;
* bind scalar values;
* execute;
* query rows;
* transactions;
* explicit close;
* bounded result collection;
* structured errors.

Resource model:

```text
Database
Statement
Transaction
RowCursor, if retained
```

Every resource must have explicit affine ownership and exactly-once cleanup.

---

## 8.2 Connection access under concurrency

Choose and document one initial model:

### Single database worker

All database operations flow through one owned database thread and bounded channel.

Advantages:

* simple ownership;
* no cross-thread SQLite handle sharing;
* deterministic sequencing.

### Bounded connection pool

Only if the provider and ownership model can prove:

* one owner per connection;
* bounded pool size;
* checkout/check-in safety;
* transaction confinement;
* shutdown cleanup.

The single-database-worker design is likely the safer first version.

---

## 8.3 CSV

### `stark-csv`

* configurable delimiter;
* quoted fields;
* escaped quotes;
* CRLF/LF;
* row limits;
* field limits;
* buffered streaming reader;
* deterministic writer.

---

## 8.4 Glob

### `stark-glob`

Initial scope:

* `*`;
* `?`;
* path-segment awareness;
* bounded matching complexity;
* explicit recursive-glob policy.

Do not combine this with a full regex engine.

---

## 8.5 Process execution

### `stark-process`

* executable path;
* arguments;
* environment overrides;
* working directory;
* exit status;
* bounded stdout/stderr capture;
* timeout;
* no shell interpretation by default.

---

## 8.6 SemVer

### `stark-semver`

* parse;
* compare;
* canonical formatting;
* prerelease ordering;
* version-range matching;
* strict malformed-input handling.

---

## Phase 6 outcome

STARK can support persistent concurrent services and practical local applications.

---

# 9. Phase 7 — Ecosystem Consolidation

## February 2027

## Objective

Make STARK installable, documented and usable by developers who do not know its compiler
internals.

---

## 9.1 Package documentation

Every first-party package should document:

* purpose;
* supported platforms;
* API examples;
* limits;
* error model;
* resource ownership;
* exclusions;
* qualification status.

Generate a package index.

---

## 9.2 Examples repository

Maintain CI-qualified examples:

```text
01-basics
02-files-and-stdio
03-json
04-http-client
05-hashing
06-cli
07-rest-server
08-structured-concurrency
09-concurrent-rest-server
10-sqlite
11-artifact-verifier
12-project-tracker
```

Every example must compile and run in CI.

---

## 9.3 General package qualification

A package should declare:

* pure tests;
* interpreter consumer;
* native consumer;
* providers;
* expected output;
* supported targets;
* debug/release requirements;
* negative cases;
* lifecycle checks.

Adding a package should be primarily data/fixture-driven rather than requiring several
hardcoded script edits.

---

## 9.4 Distribution

Define and implement:

* compiler archives;
* standard-package bundle;
* provider crates;
* pinned dependencies;
* platform assets;
* cache location;
* offline-build policy;
* version inspection;
* update strategy.

Produce repeatable Tier-1 release archives.

---

## 9.5 Diagnostics

Audit errors encountered by the proving applications:

* package resolution;
* provider selection;
* native build;
* ownership;
* thread/task safety;
* missing capability;
* malformed manifest;
* unsupported platform;
* REST request parsing;
* database errors.

Prioritise high-frequency confusing diagnostics over broad wording rewrites.

---

## 9.6 Formatter and editor tooling

Consolidate:

* formatter stability;
* syntax highlighting;
* VS Code extension;
* diagnostics;
* hover documentation;
* go-to-definition;
* package import completion.

Only implement rename/refactor features where semantic correctness can be guaranteed.

---

# 10. Async and Event-Loop Decision Gate

## After February 2027

Async should not be assumed to be the next step merely because Node.js, Rust or other
ecosystems use it.

The concurrent worker-pool REST server should first provide real workload evidence.

## Candidate outcomes

### Continue structured threads

Suitable if bounded worker pools meet application needs.

### Add non-blocking event loop

Potentially useful for many mostly idle connections.

Requires:

* non-blocking sockets;
* readiness polling;
* connection state machines;
* timers;
* cancellation;
* bounded registration.

### Add async/await

Requires:

* suspension-aware ownership;
* generated state machines;
* borrowing across suspension;
* task cancellation;
* executor;
* reactor;
* task-local cleanup;
* structured task groups.

### Remain synchronous/threaded longer

Valid if CLI, local server and moderate production workloads are already well served.

## Decision criteria

Measure:

* concurrent connection needs;
* memory per connection;
* thread overhead;
* database bottlenecks;
* shutdown complexity;
* provider compatibility;
* ownership complexity;
* implementation cost.

Do not start async implementation until the decision record identifies a workload that
structured concurrency cannot reasonably satisfy.

---

# 11. Continuous maintenance lane

## Compiler correctness

When an application exposes a defect:

```text
reduce
→ register
→ inspect all engines
→ inspect generated Rust
→ determine roadmap impact
→ fix in a focused change
```

Do not automatically turn every adjacent possibility into a new immediate compiler campaign.

Defects are registered as `DEV-nnn` in `COMPILER-STATE.md` under the compiler track's existing
governance (§0.1); this roadmap does not introduce a second defect register.

## Documentation

Update architecture, packages and decisions during each consolidation period.

## Testing

Continue adding:

* differential tests;
* malformed-input cases;
* generated-source assertions;
* ownership adversaries;
* concurrency lifecycle cases;
* native resource evidence.

## Performance

Track:

* check time;
* native build time;
* cache-hit time;
* binary size;
* HTTP client latency;
* REST throughput;
* worker-pool saturation;
* memory per active connection;
* SQLite latency.

Optimise only when real workloads demonstrate a problem.

---

# 12. Explicit non-goals for this roadmap

Unless demanded by a proving application, defer:

* async/await implementation;
* HTTP/2;
* WebSockets;
* distributed actors;
* JIT compilation;
* direct Cranelift backend;
* garbage collection;
* reflection;
* procedural macros;
* trait objects;
* full regex engine;
* general GUI framework;
* package registry service;
* compiler self-hosting;
* a large web framework.

---

# 13. Consolidated timeline

```text
August–September 2026
    Operability foundation
    env, args, stdio, bufio, path, time, SHA-256, logging
    tools: stark-json, stark-hash, stark-get

October 2026
    Security and artifact integrity
    secure random, UUID generation, HMAC, content IDs, build provenance
    tool: stark-artifact

November 2026
    REST API server v0.1
    TcpListener, shutdown, HTTP server, router, JSON REST helpers
    app: stark-notes

December 2026
    Structured concurrency
    scoped threads, join, bounded channels, synchronization, worker pool

January 2027
    Concurrent REST API server
    bounded connection queue, worker pool, overload policy,
    graceful shutdown, middleware and authentication helpers

January–February 2027
    Persistence and application packages
    SQLite, CSV, glob, process, SemVer
    app: project/evidence API

February 2027
    Ecosystem consolidation
    documentation, examples, qualification framework,
    distribution, diagnostics and editor tooling

After February 2027
    Evidence-based async/event-loop decision
```

---

# 14. Measures of success

By the end of this roadmap, STARK should demonstrate:

## Command-line applications

* shell pipelines;
* files and standard streams;
* arguments;
* JSON and CSV;
* logging;
* hashing and verification.

## Secure applications

* secure randomness;
* UUID generation;
* HMAC;
* artifact identities;
* build provenance;
* verified HTTPS.

## REST services

* synchronous REST API;
* structured routing;
* request limits;
* JSON bodies;
* clean shutdown;
* concurrent bounded worker-pool server;
* overload control;
* request isolation.

## Concurrency

* scoped thread creation;
* owned value transfer;
* bounded channels;
* affine resource transfer;
* explicit joins;
* predictable failure and cleanup;
* no detached background tasks.

## Data applications

* SQLite persistence;
* transactional operations;
* concurrent server integration;
* artifact-backed attachments.

## Ecosystem

* repeatable installation;
* documented packages;
* executable examples;
* Tier-1 qualification;
* maintainable release artifacts.

---

# 15. Strategic endpoint

At the end of this roadmap, STARK should be credibly described as:

> A coherent native programming language with secure host capabilities, practical command-line
> tools, a bounded concurrent REST API stack, persistent local storage, reproducible artifacts
> and a disciplined path toward asynchronous execution.

The architecture-proving phase is complete.

The next stage is to develop that architecture into a durable application platform at a
sustainable pace.
