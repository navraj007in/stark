# Recommended STARK standard-package roadmap

The roadmap should separate three things that are often mixed together:

1. **Pure libraries** written entirely in STARK.
2. **Host-backed standard packages** that require operating-system providers.
3. **Language/runtime capabilities** such as concurrency and async.

Do not make networking or concurrency a prerequisite for starting the package ecosystem.

---

## Phase P0 — Freeze the package foundation

Complete this immediately after C6.

### Goals

* Publish the exact executable Core and standard-library method inventory.
* Record every unsupported method by name.
* Stabilise package manifests, dependency identity and versioning.
* Define compatibility expectations for package releases.
* Establish the native-provider boundary.

### Required decisions

Define:

* package naming rules;
* semantic versioning policy;
* dependency resolution and lockfile rules;
* one package identity per name/version;
* feature flags, or explicitly defer them;
* public API visibility;
* documentation format;
* test layout;
* supported platforms;
* package qualification levels.

Suggested qualification labels:

```text
PURE
HOST-BACKED
EXPERIMENTAL
QUALIFIED
PLATFORM-SPECIFIC
```

### Core gaps worth closing first

Prioritise methods that unblock many packages:

* `String::into_bytes`
* `String::substring`
* `String::starts_with`
* `String::ends_with`
* `String::find`
* `String::trim`
* `String::split`
* `Vec::capacity`
* `Vec::insert`
* `Vec::append`
* `HashMap::remove`
* `HashMap::clear`
* `HashMap::values`
* `HashMap::iter`
* proper `Hash + Eq` bound enforcement

Not every one must block packages, but they reduce unnecessary workarounds.

---

## Phase P1 — Encoding and utility packages

These are low-risk, deterministic and ideal for proving the package model.

### First package set

#### `stark-hex`

Features:

* encode bytes to lowercase and uppercase hex;
* decode strict hexadecimal;
* exact invalid-character offsets;
* odd-length handling;
* fixed test vectors.

#### `stark-base64`

Features:

* standard alphabet;
* URL-safe alphabet;
* padded and unpadded forms;
* strict decoding;
* malformed input errors;
* deterministic test vectors.

#### `stark-uuid`

Initially:

* UUID parsing;
* formatting;
* nil UUID;
* version and variant inspection.

Generation can wait for the randomness provider.

#### `stark-semver`

Features:

* semantic-version parsing;
* comparison;
* compatibility ranges;
* package-version validation.

This is strategically useful because the package manager itself will need it.

#### `stark-checksum`

Start with non-cryptographic checksums:

* CRC32;
* Adler-32.

Then add cryptographic hashes through audited implementations or a provider strategy.

### Exit condition

Each package should have:

* public API documentation;
* malformed-input tests;
* cross-package consumer test;
* HIR/MIR/native qualification;
* exact package version;
* no hidden host dependency.

---

## Phase P2 — Text and structured-data packages

This phase makes STARK useful for real data-processing programs.

### `stark-url`

Complete the current package:

* percent encoding and decoding;
* UTF-8 reconstruction;
* query parsing;
* origin-form request targets;
* URI components;
* normalisation rules;
* explicit limits.

Do not immediately attempt the full browser WHATWG URL standard. Start with a clearly scoped RFC-oriented API.

### `stark-json`

Suggested API:

```stark
enum JsonValue {
    Null,
    Bool(Bool),
    Number(JsonNumber),
    String(String),
    Array(Vec<JsonValue>),
    Object(HashMap<String, JsonValue>),
}
```

Features:

* parser;
* serializer;
* exact error location;
* depth limits;
* string escape handling;
* Unicode;
* integer versus floating-point policy;
* deterministic object output policy.

Later:

* typed encode/decode traits;
* derive or generated codecs.

### `stark-csv`

Features:

* configurable delimiter;
* quoted fields;
* escaped quotes;
* line and column errors;
* streaming API later;
* in-memory parser first.

### `stark-toml-lite`

Useful for package manifests:

* strings;
* integers;
* booleans;
* arrays;
* tables;
* deterministic parsing.

A deliberately bounded TOML subset is acceptable initially.

### `stark-regex-lite` or parser combinators

Do not build a full regex engine immediately.

Start with either:

* glob and wildcard matching; or
* parser-combinator primitives.

Parser combinators would benefit JSON, URL, HTTP and configuration packages.

---

## Phase P3 — Time, identity and deterministic services

### `stark-time`

Split pure and host-backed functionality.

#### Pure layer

* `Duration`;
* `UnixTimestamp`;
* checked arithmetic;
* comparison;
* conversion;
* calendar calculations;
* ISO-8601 parsing and formatting.

#### Provider-backed layer

* monotonic `Instant`;
* wall-clock `SystemTime`;
* sleep;
* timezone information later.

Keep the provider API explicit:

```stark
trait ClockProvider {
    fn monotonic_now() -> Instant;
    fn unix_now() -> Result<UnixTimestamp, TimeError>;
}
```

### `stark-random`

Separate deterministic and secure randomness.

#### Deterministic PRNG

Specify one exact algorithm, such as a stable PCG or Xoshiro variant.

Requirements:

* same seed gives the same sequence on every engine and platform;
* algorithm is part of the compatibility contract;
* integer and float generation are precisely defined.

#### Secure randomness

Use a host provider:

```stark
fn secure_bytes(count: UInt64) -> Result<Vec<UInt8>, RandomError>;
```

Do not use the deterministic `Random` type for cryptographic purposes.

### `stark-uuid` completion

Once randomness exists:

* UUID v4 generation;
* possibly v7 generation once time and randomness are available.

---

## Phase P4 — Filesystem, process and environment

This is the first major host-backed standard-library phase.

### Provider architecture

Define a small host ABI rather than exposing Rust or operating-system details directly.

Suggested layers:

```text
STARK public package API
        ↓
stable host-provider ABI
        ↓
native provider implementation
        ↓
operating system
```

### `std.fs`

Initial API:

* read whole file;
* write whole file;
* append;
* metadata;
* exists;
* create/remove directory;
* remove file;
* rename;
* directory listing.

Example:

```stark
let text = fs::read_to_string("config.json")?;
fs::write("output.txt", text.bytes())?;
```

Use resource-owning types where appropriate:

```stark
struct File {
    handle: HostFileHandle,
}
```

`Drop` should close handles deterministically.

### `std.path`

Pure package:

* path joining;
* components;
* filename;
* extension;
* parent;
* normalisation;
* platform-aware path rules.

Avoid treating paths as plain strings throughout APIs.

### `std.io`

Define:

* `Read`;
* `Write`;
* `Seek`;
* `BufReader`;
* `BufWriter`;
* byte streams;
* text decoding boundaries.

Start synchronously.

### `std.env`

* environment-variable lookup;
* current directory;
* command-line arguments;
* executable path;
* platform identifiers.

### `std.process`

Later in the phase:

* process exit;
* spawn;
* status;
* captured output;
* stdin/stdout/stderr pipes.

### Exit condition

Build useful CLI applications such as:

* JSON formatter;
* recursive file scanner;
* checksum tool;
* log analyser;
* static-site generator.

This should be the first major "STARK can build useful programs" milestone.

---

## Phase P5 — Synchronous networking

Do this before async.

### `std.net`

Initial types:

* `IpAddress`;
* `Ipv4Address`;
* `Ipv6Address`;
* `SocketAddress`;
* `TcpListener`;
* `TcpStream`;
* `UdpSocket`;
* DNS resolver.

Example:

```stark
let listener = TcpListener::bind("127.0.0.1:8080")?;

loop {
    let stream = listener.accept()?;
    handle(stream)?;
}
```

Initial implementation can be single-threaded and blocking.

### Required semantics

Specify:

* ownership of sockets;
* close-on-Drop;
* partial reads and writes;
* timeouts;
* connection shutdown;
* end-of-stream;
* address parsing;
* platform error mapping;
* maximum buffer sizes.

### `stark-tls`

Do not implement cryptography from scratch initially.

Use a provider-backed package over a proven TLS implementation.

Expose:

* client configuration;
* server configuration;
* certificates;
* secure stream wrapper;
* hostname verification;
* protocol version policy.

Keep provider identity and version in evidence records.

---

## Phase P6 — HTTP foundation

Once synchronous networking works, build HTTP as ordinary STARK packages.

### `stark-http-core`

Pure package:

* request method;
* status code;
* headers;
* URI/request target;
* HTTP version;
* request and response models.

### `stark-http-parser`

Start with HTTP/1.1:

* request line;
* response line;
* headers;
* content length;
* chunked transfer;
* strict limits;
* malformed input errors;
* incremental parser.

The parser should be testable without sockets.

### `stark-http-client`

Initial synchronous client:

* GET/POST;
* request headers;
* fixed and chunked bodies;
* redirects later;
* TLS integration;
* timeout handling.

### `stark-http-server`

Initial server:

* blocking listener;
* request parsing;
* response writing;
* connection limits;
* keep-alive;
* basic error responses.

This server may handle one connection at a time initially. That is still valuable for qualification and demos.

---

## Phase P7 — Concurrency foundation

Only start this after synchronous I/O is working and validated.

### First concurrency model

I recommend structured native threads before async/await.

Core types:

* `Thread`;
* `JoinHandle<T>`;
* `Mutex<T>`;
* `RwLock<T>`;
* `Arc<T>` or STARK equivalent;
* atomics;
* channels.

Required type-system rules:

* values allowed to cross thread boundaries;
* values allowed to be shared;
* thread-safe Drop;
* trap propagation through `join`;
* prevention of reference escape;
* scoped threads.

Prefer:

```stark
thread::scope(|scope| {
    scope.spawn(move || handle(connection));
});
```

over unrestricted detached threads.

### Why threads first

* easier to specify than async suspension;
* maps directly to existing native backends;
* validates ownership across concurrency;
* enough for an initial multi-client HTTP server;
* provides primitives needed by an async executor later.

---

## Phase P8 — Async and scalable networking

Treat this as a major compiler gate, not just a package.

### Language features

* `async fn`;
* `await`;
* `Future<T>`;
* task cancellation;
* suspension-point ownership;
* Drop of suspended futures;
* scoped tasks;
* executor interface.

### Runtime packages

* event reactor;
* timers;
* async TCP;
* async file APIs where supported;
* task scheduler;
* async channels.

### Validation requirements

The three engines must agree on:

* completion;
* cancellation;
* destruction order;
* trap propagation;
* task joins;
* timeout behaviour;
* resource cleanup.

Async should not be added until these semantics are written first.

---

## Phase P9 — REST framework and application packages

Once HTTP and concurrency are stable:

### `stark-router`

* static routes;
* path parameters;
* query extraction;
* method matching;
* nested routers;
* deterministic route precedence.

### `stark-web`

* handlers;
* middleware;
* request context;
* body limits;
* JSON request/response;
* errors;
* logging;
* authentication hooks.

Example:

```stark
let app = Router::new()
    .get("/health", health)
    .post("/users", create_user);

Server::bind("0.0.0.0:8080")?
    .serve(app)?;
```

### `stark-serde`

Eventually provide:

* `Encode<T>`;
* `Decode<T>`;
* JSON codecs;
* form codecs;
* query extraction;
* generated or derived implementations.

### Database packages

Begin with provider-backed drivers:

* PostgreSQL;
* SQLite;
* Redis later.

Do not implement wire protocols for all databases initially. A stable native-provider boundary will accelerate adoption.

---

## Suggested order of actual execution

```text
P0  Package and provider contracts
P1  hex, base64, semver, checksum
P2  URL, JSON, CSV, config parsing
P3  time, deterministic random, UUID
P4  path, fs, io, env, process
P5  synchronous TCP, DNS, TLS
P6  HTTP parser, client, single-threaded server
P7  structured threads and channels
P8  async/await and event runtime
P9  router, REST framework, database packages
```

## First twelve concrete packages

1. `stark-hex`
2. `stark-base64`
3. `stark-semver`
4. `stark-url`
5. `stark-json`
6. `stark-csv`
7. `stark-time`
8. `stark-random`
9. `stark-uuid`
10. `stark-path`
11. `stark-fs`
12. `stark-net`

## Release milestones

### STARK Packages 0.1 — Pure Foundation

* hex;
* Base64;
* URL;
* semver;
* JSON;
* CSV;
* time arithmetic.

### STARK Platform 0.2 — Useful CLI

* filesystem;
* paths;
* process/environment;
* clocks;
* randomness;
* executable CLI tools.

### STARK Platform 0.3 — Connected

* TCP;
* DNS;
* TLS;
* HTTP client;
* simple HTTP server.

### STARK Platform 0.4 — Concurrent

* threads;
* synchronization;
* channels;
* multi-client HTTP server.

### STARK Platform 0.5 — Web

* async runtime;
* router;
* middleware;
* REST framework;
* JSON codecs;
* database connectivity.

## Governance rule

Every package should declare:

```text
Language surface required
Host providers required
Determinism guarantees
Platform support
Resource limits
Error model
Three-engine qualification status
Known exclusions
```

This prevents packages from quietly depending on unsupported methods or backend-specific behaviour.

The next immediate move should be **P0 plus P1 in parallel**: freeze package/provider contracts while completing `stark-hex`, `stark-base64`, `stark-url`, `stark-semver` and `stark-json`.
