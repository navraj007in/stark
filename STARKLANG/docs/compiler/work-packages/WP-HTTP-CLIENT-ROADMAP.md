# STARK HTTP Client Roadmap

**Document type:** Frozen implementation roadmap and work-package plan  
**Audience:** Claude Code, Codex, compiler/package maintainers  
**Status:** FROZEN FOR EXECUTION
**Target:** A synchronous, blocking, cross-platform HTTPS client written primarily in STARK  
**Concurrency:** Explicitly out of scope  
**Primary milestone:** A STARK application can perform bounded `GET` and `POST` requests to ordinary hostname-based HTTPS REST APIs and safely decode JSON responses.

> **Freeze rule:** This document defines one initial request-construction spelling only: build a request independently, call `.build()`, then pass the resulting request to `Client::send`. A builder must not retain a reference to `Client` unless HC0 first proves reference-bearing structs and their lifetime behaviour across HIR, MIR, and native execution.

---

## 1. Goal

Deliver a qualified HTTP/1.1 client stack with this public shape:

```stark
use stark_http::{Client, ClientConfig, Request};
use stark_json::JsonValue;
use stark_time::Duration;

fn call_api(token: String) -> Result<JsonValue, ApiError> {
    let client = Client::new(ClientConfig {
        connect_timeout: Duration::seconds(10),
        read_timeout: Duration::seconds(30),
        write_timeout: Duration::seconds(30),
        max_response_bytes: 4u64 * 1024u64 * 1024u64,
        follow_redirects: false,
    });

    let request = Request::post(
        "https://api.example.com/v1/items",
    )?
        .header("Authorization", token)?
        .header("Content-Type", "application/json")?
        .json(
            JsonValue::object()
                .insert("name", JsonValue::string("example")),
        )?
        .build()?;

    let response = client.send(request)?;

    if response.status() < 200u16 || response.status() >= 300u16 {
        return Err(ApiError::HttpStatus(response.status()));
    }

    return response.json().map_err(ApiError::InvalidJson);
}
```

The completed stack must support:

- hostname-based connections;
- IPv4 and IPv6 resolution;
- blocking TCP;
- HTTPS with certificate and hostname verification;
- HTTP/1.1 request generation;
- incremental response parsing;
- `Content-Length`;
- chunked transfer decoding;
- connection-close-delimited bodies;
- bounded headers and bodies;
- connect, read, write, and TLS-handshake timeouts;
- structured errors;
- deterministic cleanup on every exit path;
- Linux, macOS, and Windows qualification.

---

## 2. Non-goals

The first release must not include:

- concurrency;
- async/await;
- HTTP/2;
- HTTP/3;
- WebSockets;
- connection pooling;
- proxy support;
- cookie jars;
- automatic compression;
- automatic retries;
- automatic redirects by default;
- client certificates;
- OAuth framework abstractions;
- generated typed JSON codecs;
- browser-style WHATWG URL behaviour;
- arbitrary native FFI from application code.

These may be added after the synchronous HTTP/1.1 client is complete and qualified.

---

## 3. Architectural rule

The stack must preserve the existing STARK provider architecture:

```text
Application STARK source
        ↓
Public STARK package APIs
        ↓
Pure STARK protocol logic
        ↓
Declared provider API
        ↓
Qualified native provider
        ↓
Operating system / TLS implementation
```

Application code must never call raw ABI symbols.

All host authority must be visible through package manifests and provider bindings.

The implementation must keep these boundaries:

| Layer | Responsibility |
|---|---|
| `stark-url` | URL parsing and request-target construction |
| `stark-net` | IP types, DNS, TCP, timeouts, socket errors |
| `stark-tls` | TLS configuration, handshake, certificate validation, secure stream |
| `stark-http-core` | HTTP types and shared rules |
| `stark-http-parser` | Incremental HTTP/1.1 parsing and body framing |
| `stark-http-client` | Request orchestration and public client API |
| `stark-json` | JSON parsing and serialization |
| native providers | OS networking, resolver, TLS engine, clocks, secure randomness |

---

## 4. Delivery sequence

```text
HC0  Freeze contracts and inventory
HC1  Complete URL package
HC2  Complete TCP client surface
HC3  Add DNS provider and package API
HC4  Add socket timeout support
HC5  Implement HTTP core types
HC6  Implement request serializer
HC7  Implement incremental response parser
HC8  Deliver plain HTTP client
HC9  Add TLS provider and secure stream          CLOSED (CD-365)
HC10 Deliver HTTPS client                       CLOSED (CD-366)
HC11 Add JSON convenience API                   CLOSED (CD-367)
HC12 Add redirects policy                       CLOSED (CD-368)
HC13 Cross-platform qualification and release
```

The sequence is intentionally split so that plain HTTP can prove the protocol stack before TLS is introduced.

---

# HC0 — Freeze Contracts and Current-State Inventory

## Objective

Establish the authoritative starting point before implementation.

## Tasks

1. Inventory the current public and provider-backed networking surface.
2. Record exact status of:
   - `TcpStream`;
   - TCP connect;
   - read;
   - write;
   - close/drop;
   - shutdown;
   - provider synthesis;
   - provider-bound package tests;
   - native qualification.
3. Inventory current:
   - `stark-url`;
   - `stark-json`;
   - `stark-time`;
   - `stark-random`.
4. Record all compiler blockers that prevent package APIs from compiling or testing.
5. Freeze package names and dependency direction.
6. Freeze the initial HTTP client scope.
7. Define qualification labels and evidence locations.
8. Compile every goal-state syntax fragment used by this roadmap, including:
   - `Request::post`;
   - builder `.build()`;
   - `Client::send`;
   - `Duration::seconds`;
   - header value construction;
   - string concatenation suitable for values such as `"Bearer " + token`;
   - `JsonValue` rendering, including whether `JsonValue::to_string()` exists and is admitted;
   - JSON object construction.
9. Probe whether provider ABI calls can combine:
   - `HandleConsumed(TcpStream)`;
   - `HandleOut(TlsStream)`;
   in one call.
10. Confirm whether reference-bearing structs are admitted and qualified. If not, builder-held client references are forbidden.
11. Confirm the authoritative DNS provider specification and record the dependency on WP-PKG-HOST-CAPABILITIES Part E.

## Required outputs

```text
docs/http-client/HC0-CURRENT-STATE.md
docs/http-client/HC0-DECISIONS.md
docs/http-client/HC0-BLOCKERS.md
```

## Acceptance criteria

- No roadmap task depends on an assumed API.
- Every existing symbol is confirmed from source.
- Every package dependency is acyclic.
- Compiler work and package work are separately identified.
- No implementation begins before provider-resource ownership rules are confirmed.
- Every example in this roadmap either compiles against the admitted language surface or is explicitly marked aspirational.
- The initial request API is frozen as `builder.build()?` followed by `client.send(request)?`.
- Combined `HandleConsumed` plus `HandleOut` support is proven or a fallback transition protocol is designed before TLS work starts.

---

# HC1 — Complete `stark-url`

## Objective

Provide the minimum URL model required by an HTTP client.

## Package

```text
packages/stark-url
```

## Required API

```stark
enum UrlScheme {
    Http,
    Https,
}

struct Authority {
    host: String,
    port: Option<UInt16>,
}

struct Url {
    scheme: UrlScheme,
    authority: Authority,
    path: String,
    query: Option<String>,
    fragment: Option<String>,
}

impl Url {
    fn parse(input: &String) -> Result<Url, UrlError>;
    fn effective_port(&self) -> UInt16;
    fn origin_form_target(&self) -> String;
    fn host_header_value(&self) -> String;
}
```

## Required behaviour

- support only `http` and `https`;
- parse hostname, IPv4 literal, and bracketed IPv6 literal;
- apply default ports:
  - HTTP: `80`;
  - HTTPS: `443`;
- preserve path and query;
- exclude fragments from HTTP request targets;
- emit `/` when path is empty;
- include explicit non-default port in the `Host` header;
- reject user-info initially;
- reject malformed ports;
- reject missing hosts;
- enforce bounded URL length.

## Tests

- hostname with path;
- hostname with query;
- default ports;
- explicit ports;
- IPv4;
- IPv6;
- empty path;
- fragment exclusion;
- invalid scheme;
- malformed authority;
- invalid port;
- oversized URL.

## Qualification

- pure package;
- HIR/MIR/native parity;
- fixed vectors;
- no provider dependency.

## Exit criteria

A URL can be converted deterministically into:

```text
scheme
host
port
origin-form request target
Host header
```

---

# HC2 — Complete `stark-net` TCP Client Surface

## Objective

Expose a stable, application-safe TCP client API over the existing provider machinery.

## Package

```text
packages/stark-net
```

## Required public types

```stark
enum IpAddress {
    V4(Ipv4Address),
    V6(Ipv6Address),
}

struct SocketAddress {
    address: IpAddress,
    port: UInt16,
}

struct TcpStream {
    // Provider-bound affine resource.
}

enum NetworkError {
    InvalidAddress,
    AddressNotAvailable,
    ConnectionRefused,
    ConnectionReset,
    TimedOut,
    EndOfStream,
    Interrupted,
    PermissionDenied,
    Unsupported,
    ResourceExhausted,
    ProviderFailure(UInt32),
}
```

## Required API

```stark
impl TcpStream {
    fn connect(
        address: SocketAddress,
        timeout: Duration,
    ) -> Result<TcpStream, NetworkError>;

    fn read(
        &mut self,
        output: &mut [UInt8],
    ) -> Result<UInt64, NetworkError>;

    fn write(
        &mut self,
        input: &[UInt8],
    ) -> Result<UInt64, NetworkError>;

    fn write_all(
        &mut self,
        input: &[UInt8],
    ) -> Result<(), NetworkError>;

    fn shutdown_write(
        &mut self,
    ) -> Result<(), NetworkError>;
}
```

## Required semantics

- `TcpStream` is non-Copy;
- successful construction creates one owned resource;
- cleanup is compiler-managed;
- ownership cannot be duplicated;
- `write()` may be partial;
- `write_all()` loops until complete or error;
- `read()` returns `0` only for end-of-stream if that convention is selected;
- interrupted operations surface `NetworkError::Interrupted`; no internal retry is permitted;
- every provider status code is mapped explicitly;
- undeclared provider status codes are contract violations;
- all lengths are bounded before ABI conversion.

## Tests

- loopback connect;
- partial write simulation;
- partial read simulation;
- remote close;
- refused connection;
- invalid address;
- timeout;
- cleanup after success;
- cleanup after read error;
- cleanup after write error;
- cleanup after early return;
- no duplicate close;
- no use after move.

## Exit criteria

A STARK program can connect to a known IP address and exchange arbitrary bytes safely.

---

# HC3 — Add DNS Resolution

## Objective

Resolve ordinary API hostnames to connectable socket addresses.

## Packages

```text
packages/stark-net
providers/stark-net-native
```

## Specification authority

**WP-PKG-HOST-CAPABILITIES Part E is authoritative for the DNS provider ABI.**

Part E owns:

- resolver record encoding;
- fixed record width;
- provider status codes;
- native resolver semantics;
- ABI bounds;
- platform-normalisation rules.

HC3 owns only:

- the public STARK package API;
- conversion from Part E records into STARK values;
- package-level limits;
- package-level error mapping;
- address-attempt policy.

This roadmap must not independently redefine DNS provider records or native status codes.

## Provider API

Use the exact bounded output ABI frozen by Part E. Do not return native pointers.

## Public API

```stark
struct ResolvedAddress {
    address: IpAddress,
    port: UInt16,
}

fn resolve(
    host: &String,
    port: UInt16,
    limits: ResolveLimits,
) -> Result<Vec<ResolvedAddress>, DnsError>;
```

## Required semantics

- use the operating-system resolver;
- return IPv4 and IPv6;
- preserve a specified ordering;
- bound hostname size;
- bound address count;
- reject embedded NUL where relevant;
- define behaviour for no results;
- no implicit indefinite cache;
- no raw resolver structures cross the ABI;
- provider panic must not unwind into STARK;
- every failure maps to a declared error.

## Required errors

```stark
enum DnsError {
    InvalidHost,
    NotFound,
    TemporaryFailure,
    TooManyResults,
    UnsupportedAddressFamily,
    Unsupported,
    Other(UInt32),
}
```

## Tests

- localhost;
- invalid hostname;
- no-result hostname;
- IPv4 result;
- IPv6 result where available;
- output truncation refusal;
- maximum result count;
- deterministic fake provider cases;
- Linux/macOS/Windows native runs.

## Exit criteria

A STARK program can resolve a hostname and attempt TCP connection to each returned address.

---

# HC4 — Add Socket Timeout Support

## Objective

Prevent DNS, connect, reads, and writes from blocking forever.

## Required API

```stark
struct SocketTimeouts {
    connect: Duration,
    read: Duration,
    write: Duration,
}

impl TcpStream {
    fn set_read_timeout(
        &mut self,
        timeout: Option<Duration>,
    ) -> Result<(), NetworkError>;

    fn set_write_timeout(
        &mut self,
        timeout: Option<Duration>,
    ) -> Result<(), NetworkError>;
}
```

## Required rules

- zero-duration semantics must be specified;
- overflow converting duration to OS units must fail cleanly;
- timeout errors must be distinguishable from EOF;
- timeout configuration must survive TLS wrapping or be deliberately transferred;
- no concurrency is required;
- connect timeout must not depend on async runtime support;
- socket read and write timeouts are per-blocking-operation idle bounds, not total request-duration guarantees;
- a peer that continues delivering small amounts of data can remain below a per-read timeout;
- total response work is bounded by a monotonic-clock phase budget plus `max_response_bytes`;
- **every timeout is terminal for the current connection**;
- after timeout, the stream must be dropped and must not be resumed;
- `NetworkError::Interrupted` is terminal for the current request and connection;
- the HTTP client must clean up and return `Interrupted` without retrying internally.

## Tests

- connect timeout;
- read timeout against silent peer;
- write timeout where reproducibly simulatable;
- clear timeout;
- duration overflow;
- resource cleanup after timeout.

## Exit criteria

No operation required by the HTTP client can block indefinitely when configured with finite limits.

---

# HC5 — Implement `stark-http-core`

## Objective

Define protocol-independent HTTP data types and invariants.

## Package

```text
packages/stark-http-core
```

## Required types

```stark
enum HttpMethod {
    Get,
    Post,
    Put,
    Patch,
    Delete,
    Head,
    Options,
}

enum HttpVersion {
    Http10,
    Http11,
}

struct Header {
    name: String,
    value: String,
}

struct HeaderMap {
    entries: Vec<Header>,
}

struct HttpRequest {
    method: HttpMethod,
    target: String,
    version: HttpVersion,
    headers: HeaderMap,
    body: Vec<UInt8>,
}

struct HttpResponse {
    version: HttpVersion,
    status: UInt16,
    reason: String,
    headers: HeaderMap,
    body: Vec<UInt8>,
}
```

## Required rules

- header names are validated;
- header values reject CR and LF;
- lookup is ASCII case-insensitive;
- duplicate headers are preserved;
- singleton-header helpers reject ambiguous duplicates where necessary;
- status codes are range-checked;
- request target cannot contain spaces or CR/LF;
- `Host` is required for HTTP/1.1 request serialization;
- user-set `Content-Length` conflicts are rejected;
- `Transfer-Encoding` policy is explicit.

## Limits

```stark
struct HttpLimits {
    max_status_line_bytes: UInt64,
    max_header_line_bytes: UInt64,
    max_header_count: UInt64,
    max_header_bytes: UInt64,
    max_body_bytes: UInt64,
    max_chunk_line_bytes: UInt64,
}
```

## Tests

- header validation;
- case-insensitive lookup;
- duplicate preservation;
- CRLF injection rejection;
- invalid target;
- invalid status;
- body and header limit validation.

## Exit criteria

HTTP requests and responses can be represented without sockets or providers.

---

# HC6 — Implement HTTP Request Serialization

## Objective

Serialize a structured request into exact HTTP/1.1 bytes.

## Package

```text
packages/stark-http-client
```

or a pure submodule under `stark-http-core`.

## Required output

Example:

```text
POST /v1/items?active=true HTTP/1.1\r\n
Host: api.example.com\r\n
Content-Type: application/json\r\n
Content-Length: 18\r\n
Connection: close\r\n
\r\n
{"name":"example"}
```

## Required behaviour

- exact CRLF line endings;
- deterministic header ordering policy;
- automatic `Host`;
- automatic `Content-Length` for fixed bodies;
- default `Connection: close` in first release;
- reject caller-supplied conflicting framing headers;
- no chunked request bodies initially;
- no implicit compression;
- no invalid UTF-8 assumptions for body bytes;
- exact maximum request size.

## Tests

- GET with no body;
- POST with body;
- query target;
- explicit non-default port in `Host`;
- IPv6 `Host`;
- custom headers;
- duplicate allowed headers;
- CRLF injection refusal;
- conflicting `Content-Length`;
- exact byte snapshots.

## Exit criteria

Request bytes are independently testable and identical across all engines.

---

# HC7 — Implement Incremental HTTP/1.1 Response Parser

## Objective

Parse arbitrarily fragmented HTTP responses safely.

## Package

```text
packages/stark-http-parser
```

## Public shape

```stark
enum ParseProgress<T> {
    NeedMore,
    Complete(T, UInt64),
}

struct HttpResponseParser {
    // Explicit parser state.
}

impl HttpResponseParser {
    fn new(limits: HttpLimits) -> HttpResponseParser;

    fn push(
        &mut self,
        bytes: &[UInt8],
    ) -> Result<ParseProgress<HttpResponse>, HttpParseError>;

    fn finish(
        self,
    ) -> Result<HttpResponse, HttpParseError>;
}
```

## Parser states

At minimum:

```text
StatusLine
Headers
FixedBody
ChunkSize
ChunkData
ChunkDataCrlf
Trailers
CloseDelimitedBody
Complete
Failed
```

## Required framing support

- HTTP/1.0 and HTTP/1.1 response line;
- `Content-Length`;
- `Transfer-Encoding: chunked`;
- no-body responses;
- connection-close-delimited bodies;
- fragmented lines;
- fragmented chunks;
- informational `1xx` handling;
- chunked trailer consumption;
- parse-and-discard trailer policy for v0.1.

## Required security rules

- reject conflicting `Content-Length`;
- reject unsupported transfer codings;
- define policy for `Content-Length` plus `Transfer-Encoding`;
- reject invalid chunk sizes;
- reject overflow in decimal and hexadecimal lengths;
- reject oversized lines, headers, body, and chunk metadata;
- reject premature EOF;
- reject invalid status syntax;
- reject bare LF unless explicitly allowed;
- reject obs-fold: any continuation line beginning with SP or HTAB;
- always consume the trailer section of a chunked message;
- parse and validate trailers under the ordinary trailer/header limits, then discard them in v0.1;
- never allocate based on unvalidated lengths.

## No-body rules

Handle correctly:

- `HEAD` responses;
- `1xx`;
- `204`;
- `304`.

The parser may need request-method context to classify `HEAD`.

## Tests

### Positive

- one-buffer response;
- one-byte-at-a-time response;
- every split position;
- fixed body;
- chunked body;
- chunk extensions if supported;
- close-delimited body;
- `204`;
- `304`;
- multiple `1xx` before final response;
- duplicate ordinary headers.

### Negative

- malformed status;
- invalid header name;
- invalid line endings;
- header overflow;
- body overflow;
- chunk overflow;
- bad chunk terminator;
- truncated fixed body;
- truncated chunked body;
- conflicting framing;
- unsupported transfer encoding.

## Qualification

- pure STARK;
- HIR/MIR/native differential corpus;
- generated fragmentation tests;
- frozen malformed-input corpus.

## Exit criteria

The parser can consume any legal fragmentation pattern without socket knowledge.

---

# HC8 — Deliver Plain HTTP Client

## Objective

Compose URL, DNS, TCP, request serialization, and response parsing for `http://`.

## Package

```text
packages/stark-http-client
```

## Public API

```stark
struct ClientConfig {
    connect_timeout: Duration,
    read_timeout: Duration,
    write_timeout: Duration,
    max_response_bytes: UInt64,
    follow_redirects: Bool,
}

struct Client {
    config: ClientConfig,
}

struct RequestBuilder {
    // Request under construction.
}

impl Client {
    fn new(config: ClientConfig) -> Client;
    fn send(&self, request: HttpRequest) -> Result<HttpResponse, HttpError>;

    fn get(&self, url: &String) -> Result<RequestBuilder, HttpError>;
    fn post(&self, url: &String) -> Result<RequestBuilder, HttpError>;
}
```

## Request flow

```text
parse URL
→ require HTTP scheme
→ resolve hostname
→ attempt addresses
→ connect
→ set timeouts
→ serialize request
→ write_all
→ read incrementally
→ parse response
→ enforce body limit
→ close stream
→ return response
```

## Address-attempt policy

Specify:

- address order;
- whether all results are attempted;
- how the final error is selected;
- whether timeout budget is per address or total;
- no Happy Eyeballs requirement in first release.

## Tests

Use a controlled local server:

- GET;
- POST;
- request headers;
- fixed response;
- chunked response;
- fragmented response;
- slow response timeout;
- early close;
- malformed response;
- oversized response;
- DNS failure;
- connection refusal;
- cleanup on every failure.

## Exit criteria

A STARK application can call a local or explicitly insecure HTTP API by hostname.

---

# HC9 — Implement `stark-tls`

**STATUS: CLOSED 2026-08-03 (CD-365).** Evidence and exact boundary in
`STARKLANG/docs/http-client/HC9-TLS-EVIDENCE.md`. Delivered: `stark-tls`, `stark-tls-native`,
`stark-tls-consumer`, and the certificate fixtures in `stark-tls/fixtures`. `SystemRoots` and
`BundledRoots` are declared and REFUSED — they are HC10's. Profile F is not qualified.

## Objective

Provide a safe, provider-backed TLS client stream.

## Packages

```text
packages/stark-tls
providers/stark-tls-native
```

## Native implementation — DECIDED (CD-361)

Use a mature TLS implementation. Do not implement TLS or cryptographic primitives in STARK.

**The backend decision is CLOSED. Backend selection is no longer part of the HC9 estimate.** See
`WP-CRYPTO0-TLS-BACKEND.md` for the joint HC9/CRYPTO0 record.

```text
TLS engine                rustls
default crypto provider   aws-lc-rs, non-FIPS          (Profile N)
Profile F provider        aws-lc-rs FIPS               (qualified separately)
native-tls                REJECTED as first-party; permitted later as an external provider
root-store policy         SEPARATE from engine selection — HC9's fixture uses ExplicitRoots
versioning                exact versions and checksums pinned at qualification, never "latest"
```

Two consequences HC9 must plan for rather than discover:

- `aws-lc-rs` needs a **C/C++ compiler** for Profile N, and providers link statically into the
  generated workspace — so this becomes a requirement for every user building a TLS program, not
  only for the provider's authors. Profile F additionally needs CMake and Go.
- Profile F is **not** a Cargo feature: it requires installing the FIPS provider and verifying
  `ClientConfig::fips()` at runtime. Both belong in Profile F's qualification criteria.

Preferred properties:

- memory-safe implementation;
- TLS 1.2 and TLS 1.3;
- certificate-chain validation;
- hostname verification;
- SNI;
- well-maintained certificate-root integration;
- cross-platform support;
- explicit provider version evidence.

## Public API

```stark
struct TlsClientConfig {
    min_version: TlsVersion,
    max_version: TlsVersion,
    root_store: RootStorePolicy,
    handshake_timeout: Duration,
}

struct TlsStream {
    // Provider-bound affine resource.
}

impl TlsStream {
    fn connect(
        tcp: TcpStream,
        server_name: String,
        config: TlsClientConfig,
    ) -> Result<TlsStream, TlsError>;

    fn read(
        &mut self,
        output: &mut [UInt8],
    ) -> Result<UInt64, TlsError>;

    fn write(
        &mut self,
        input: &[UInt8],
    ) -> Result<UInt64, TlsError>;

    fn write_all(
        &mut self,
        input: &[UInt8],
    ) -> Result<(), TlsError>;
}
```

## Ownership requirements

The handshake consumes the TCP stream unconditionally at provider-call entry:

```text
TcpStream --HandleConsumed--> TLS provider
```

### Success path

```text
TcpStream slot becomes dead
        ↓
provider emits TlsStream through HandleOut
        ↓
one live TlsStream resource remains
```

Normative rules:

- the consumed TCP handle's MIR close must never run;
- the original TCP handle cannot be used after the call;
- `TlsStream` owns both TLS state and the underlying socket;
- one `TlsStream` close performs both TLS shutdown and socket close;
- there must be exactly one close path for both effects.

### Failure path

Normative rules:

- ownership does not return to the caller;
- the TLS provider closes the consumed socket;
- the caller receives no reusable TCP stream;
- retry requires full DNS resolution and a new TCP connection;
- no manually duplicated close path is permitted.

### ABI proof obligation

HC0 must prove whether one provider call can combine:

```text
HandleConsumed(TcpStream)
HandleOut(TlsStream)
```

If supported, use that atomic resource transition.

If unsupported, freeze an explicit fallback before HC9 implementation. A two-call consume-then-create design must specify provider-internal handoff state, cancellation, failure cleanup, and exactly-once close semantics.

## TLS configuration

Root acquisition is POLICY, separate from engine selection (CD-361): `SystemRoots`,
`BundledRoots`, `ExplicitRoots`. **HC9's controlled fixture uses `ExplicitRoots`** containing the
test CA; `SystemRoots` is HC10's concern and does not require handing the protocol to a platform
TLS stack.

First release:

- system or explicitly bundled trusted roots;
- hostname verification mandatory;
- SNI mandatory for hostnames;
- no “accept invalid certificate” production API;
- no TLS downgrade below policy;
- no client certificates;
- no custom verification callbacks.

## Provider dependencies

TLS may require declared:

- network;
- secure randomness;
- wall-clock time.

These requirements must be visible in package/provider metadata.

## Tests

- trusted local certificate chain;
- untrusted certificate;
- expired certificate;
- not-yet-valid certificate;
- hostname mismatch;
- missing intermediate;
- successful TLS 1.2 if supported;
- successful TLS 1.3;
- handshake timeout;
- peer closes during handshake;
- fragmented encrypted reads;
- cleanup on success;
- cleanup on every handshake failure;
- Windows/macOS/Linux qualification.

## Exit criteria

A STARK program can establish a verified TLS stream to a hostname without accessing raw provider symbols.

**MET.** `stark-tls-consumer` does exactly this natively against three controlled peers, on both
release paths, and is the 16th case in the first-party package gate.

---

# HC10 — Deliver HTTPS Client

**STATUS: CLOSED 2026-08-03 (CD-366).** Evidence and exact boundary in
`STARKLANG/docs/http-client/HC10-HTTPS-EVIDENCE.md`. `SystemRoots` is implemented and is the
default; `BundledRoots` remains refused. Redirects stay HC12 and JSON convenience stays HC11.

## Objective

Extend the HTTP client to support ordinary HTTPS APIs.

## Behaviour

```text
parse URL
→ HTTPS scheme
→ DNS
→ TCP connect
→ timeout configuration
→ TLS handshake with hostname verification
→ write HTTP request
→ parse HTTP response
→ close secure stream
```

## Required API behaviour

The public `Client::send` must select HTTP or HTTPS from the URL scheme.

No application-level distinction should be required beyond the URL.

## Errors

```stark
enum HttpError {
    InvalidUrl(UrlError),
    Dns(DnsError),
    Network(NetworkError),
    Tls(TlsError),
    InvalidRequest(HttpRequestError),
    InvalidResponse(HttpParseError),
    RequestTooLarge,
    ResponseTooLarge,
    Timeout(HttpTimeoutPhase),
    UnsupportedScheme,
}
```

## Required timeout phases

```stark
enum HttpTimeoutPhase {
    Resolve,
    Connect,
    TlsHandshake,
    WriteRequest,
    ReadResponse,
}
```

## Tests

- HTTPS GET;
- HTTPS POST;
- custom authorization header;
- JSON body;
- trusted local CA;
- certificate rejection;
- hostname rejection;
- TLS timeout;
- response parser failure after TLS success;
- resource cleanup across all error phases.

## Exit criteria

A STARK program can call a normal hostname-based HTTPS endpoint with verified certificates.

**MET.** `stark-http-client-consumer` does so natively under the package gate, over eleven cases —
including three refusals (untrusted chain, hostname mismatch, cleartext peer on the secure path),
because a gate that observed only the happy path would pass against a client that skipped
verification entirely.

---

# HC11 — Add JSON Convenience Integration

**STATUS: CLOSED 2026-08-03 (CD-367).** Evidence in
`STARKLANG/docs/http-client/HC11-JSON-EVIDENCE.md`. `body_text` and the strict UTF-8 decoder landed
in `stark-http-core`; the JSON half is in `stark-http-client`, because core must not depend on
`stark-json`. Typed codecs remain out of scope.

## Objective

Make common REST calls concise without coupling HTTP core to JSON internals.

## Packages

```text
packages/stark-http-client
packages/stark-json
```

## Required API

```stark
impl RequestBuilder {
    fn json(
        self,
        value: JsonValue,
    ) -> Result<RequestBuilder, HttpError>;
}

impl HttpResponse {
    fn body_text(
        &self,
    ) -> Result<String, TextDecodeError>;

    fn json(
        &self,
    ) -> Result<JsonValue, JsonError>;
}
```

## Required rules

- `RequestBuilder::json` sets `Content-Type: application/json`;
- serialize deterministically according to `stark-json` policy;
- body bytes are UTF-8;
- response JSON decoding is explicit;
- content type may be validated optionally;
- body limits are enforced before parsing;
- JSON depth and allocation limits remain active.

## Tests

- JSON object POST;
- Unicode JSON;
- invalid JSON response;
- oversized JSON;
- empty response body;
- content-type mismatch policy;
- deterministic serialized request snapshots.

## Exit criteria

Common JSON REST calls no longer require manual byte conversion or header construction.

**MET.** `post(url, empty).json(&value)?` then `response.json_checked()?`, exercised natively over a
verified TLS session as the consumer's twelfth case.

---

# HC12 — Add Safe Redirect Policy

**STATUS: CLOSED 2026-08-03 (CD-368).** Evidence in
`STARKLANG/docs/http-client/HC12-REDIRECT-EVIDENCE.md`. Following stays OFF by default; the policy
below is what "on" means.

## Objective

Provide bounded redirect handling without leaking credentials.

## Initial default

```text
follow_redirects = false
```

## Later opt-in policy

```stark
struct RedirectPolicy {
    max_redirects: UInt8,
    allow_https_to_http: Bool,
    preserve_authorization_same_origin_only: Bool,
}
```

## Required rules

- maximum redirect count;
- loop detection;
- resolve relative `Location`;
- strip `Authorization`, cookies, and sensitive custom headers when origin changes;
- reject HTTPS-to-HTTP downgrade by default;
- method-rewrite policy for `301`, `302`, `303`, `307`, `308`;
- body replay only when safe and buffered;
- redirect target must pass URL limits.

## Tests

- relative redirect;
- absolute redirect;
- loop;
- maximum count;
- cross-origin credential stripping;
- HTTPS downgrade refusal;
- `303` method conversion;
- `307` method preservation.

## Exit criteria

Redirect support is opt-in, bounded, and cannot silently forward credentials to another origin.

**MET.** Ten native consumer cases against live peers, including cross-origin credential stripping
asserted by reading what the peer actually received, and a downgrade refused before anything is
dialled.

---

# HC13 — Qualification, Documentation, and Release

**Status: CLOSED 2026-08-03 (CD-375).** Evidence in `STARKLANG/docs/http-client/HC13-*.md`.
One acceptance criterion is **partial** and reported as partial: two of five timeout phases have no
stalling peer. Found and fixed on the way: DEV-163.

## Objective

Produce defensible evidence for the complete client stack.

## Qualification layers

### Pure packages

Run through:

- HIR interpreter;
- MIR interpreter;
- native backend;
- Linux;
- macOS;
- Windows;
- frozen corpus;
- generated fragmentation corpus.

Packages:

- `stark-url`;
- `stark-http-core`;
- `stark-http-parser`;
- pure portions of `stark-http-client`;
- `stark-json`.

### Provider-backed packages

Run native qualification for:

- DNS;
- TCP;
- timeout handling;
- TLS;
- full HTTP;
- full HTTPS.

## Controlled test infrastructure

Build local fixtures for:

- plain HTTP server;
- TLS server with test CA;
- invalid-certificate server;
- hostname mismatch;
- delayed server;
- fragmented-response server;
- chunked-response server;
- malformed-response server;
- oversized-response server;
- premature-close server.

Do not make qualification depend exclusively on public internet services.

## Required evidence

```text
docs/http-client/HC13-QUALIFICATION-REPORT.md
docs/http-client/HC13-PLATFORM-MATRIX.md
docs/http-client/HC13-THREAT-MODEL.md
docs/http-client/HC13-KNOWN-LIMITATIONS.md
docs/http-client/HC13-RELEASE-CHECKLIST.md
```

## Release acceptance criteria

- GET and POST over HTTPS pass on all Tier-1 platforms;
- hostname verification is proven by negative tests;
- untrusted certificates are rejected;
- DNS works on all Tier-1 platforms;
- fixed and chunked responses pass;
- all documented malformed responses are rejected;
- body/header limits are enforced;
- timeouts are phase-specific;
- no resource leak or duplicate close is observed;
- provider APIs are unreachable without manifest declarations;
- application code cannot call raw ABI symbols;
- package tests and native qualification are automated;
- exact provider identities and versions are recorded;
- all public exclusions are documented.

---

## 5. Package dependency graph

```text
stark-time ─────────────┐
                       ├── stark-net ────────┐
stark-random ──────────┤                    │
                       └── stark-tls ───────┤
stark-url ──────────────────────────────────┤
stark-http-core ────────────────────────────┤
stark-http-parser ──────────────────────────┤
stark-json ─────────────────────────────────┤
                                            ↓
                                  stark-http-client
```

Rules:

- `stark-http-parser` must not depend on `stark-net`;
- `stark-http-core` must not depend on `stark-tls`;
- `stark-url` must remain pure;
- `stark-json` must remain usable without HTTP;
- `stark-tls` may depend on `stark-net`, `stark-time`, and `stark-random`;
- `stark-http-client` is the composition package.

---

## 6. Compiler work versus package work

### Expected package/provider work

- URL completion;
- DNS package;
- DNS native provider;
- TCP API completion;
- timeout provider operations;
- HTTP core;
- request serializer;
- response parser;
- HTTP client;
- TLS package;
- TLS provider;
- JSON convenience API;
- qualification harnesses.

### Compiler/tooling work only when encountered

The package effort must not silently absorb compiler defects. Record and isolate them.

Likely compiler/tooling dependencies:

- provider API synthesis under `stark test`;
- native qualification for library-only packages;
- correct provider-bound resource typing;
- affine ownership through TLS resource wrapping;
- output-slot discipline for resolver results;
- package visibility and cross-module public APIs;
- reliable slice and byte-buffer operations;
- diagnostics for unsupported source patterns;
- deterministic package test discovery;
- provider dependency propagation into manifests;
- exact resource cleanup on parser/TLS error paths.

Each compiler blocker must receive:

```text
reproducer
expected behaviour
actual behaviour
error layer
temporary workaround, if any
blocking packages
owner
closure evidence
```

Do not modify package semantics merely to hide a compiler defect.

---

## 7. Parallel execution plan

## Track A — Pure protocol packages

Suitable for Codex or Claude in parallel:

1. `stark-url`;
2. `stark-http-core`;
3. HTTP request serializer;
4. response parser skeleton;
5. fixed-body parser;
6. chunked parser;
7. malformed-response corpus;
8. JSON integration.

This track must use fake byte streams and must not depend on DNS, TCP, or TLS completion.

## Track B — Network provider work

1. public TCP package cleanup;
2. DNS ABI;
3. DNS native provider;
4. deterministic fake resolver;
5. connect timeout;
6. read/write timeout;
7. platform error normalization;
8. native qualification.

## Track C — TLS provider work

May begin after the TCP ownership contract is frozen:

1. provider selection decision;
2. TLS ABI design;
3. root-store policy;
4. secure stream resource type;
5. TCP-to-TLS ownership transfer;
6. native implementation;
7. failure-path cleanup;
8. certificate test fixtures;
9. cross-platform qualification.

## Track D — Tooling and compiler blockers

1. provider synthesis in `stark test`;
2. library-only native qualification;
3. package consumer fixtures;
4. manifest capability dependency propagation;
5. resource-affinity regression corpus.

## Integration order

```text
A URL + HTTP core
B TCP + DNS
A request serializer + parser
A+B plain HTTP client
C TLS
A+B+C HTTPS client
A JSON integration
D full package qualification
```

---

## 8. Suggested Claude/Codex assignment boundaries

### Codex-friendly tasks

- implement pure data types;
- parser state machines;
- serializer functions;
- test vectors;
- malformed-input tests;
- fake providers;
- qualification scripts;
- documentation tables;
- package consumer examples.

### Claude-friendly tasks

- ABI and ownership review;
- resource lifecycle analysis;
- provider contract design;
- cross-layer compiler diagnosis;
- TLS threat model;
- error taxonomy;
- roadmap and gate closure;
- architectural consistency review.

### Tasks requiring owner decision

- TLS implementation/provider choice;
- system roots versus bundled roots;
- transfer-encoding ambiguity policy;
- redirect method policy;
- timeout budget policy;
- DNS ordering policy;
- provider capability dependency representation;
- release qualification threshold.

---

## 9. Milestones

## Milestone HTTP-A — Protocol Complete

Includes:

- `stark-url`;
- `stark-http-core`;
- request serializer;
- incremental response parser;
- fixed and chunked bodies;
- pure three-engine qualification.

Claim enabled:

> STARK has a complete, provider-independent HTTP/1.1 protocol core.

## Milestone HTTP-B — Plain Connected

Includes:

- DNS;
- TCP client;
- timeouts;
- plain HTTP client;
- local-server qualification.

Claim enabled:

> STARK can perform bounded synchronous HTTP calls by hostname.

## Milestone HTTP-C — Secure Connected

Includes:

- TLS;
- verified certificates;
- hostname verification;
- HTTPS client;
- cross-platform qualification.

Claim enabled:

> STARK can call ordinary HTTPS REST APIs.

## Milestone HTTP-D — REST Usable

Includes:

- JSON convenience API;
- stable errors;
- examples;
- package docs;
- manifest declarations;
- release evidence.

Claim enabled:

> STARK can build practical synchronous REST API clients.

---

## 10. Final definition of done

The roadmap is complete when this application works without raw ABI calls:

```stark
use stark_http::{Client, ClientConfig};
use stark_json::JsonValue;
use stark_time::Duration;

fn main() -> Result<(), ApiError> {
    let client = Client::new(ClientConfig {
        connect_timeout: Duration::seconds(10),
        read_timeout: Duration::seconds(30),
        write_timeout: Duration::seconds(30),
        max_response_bytes: 1u64 * 1024u64 * 1024u64,
        follow_redirects: false,
    });

    let request = Request::post(
        "https://api.example.com/v1/echo",
    )?
        .header("Content-Type", "application/json")?
        .json(
            JsonValue::object()
                .insert("message", JsonValue::string("hello")),
        )?
        .build()?;

    let response = client.send(request)?;

    if response.status() != 200u16 {
        return Err(ApiError::UnexpectedStatus(response.status()));
    }

    let value = response.json()?;
    println(value.to_string());

    return Ok(());
}
```

And the following properties are demonstrated:

1. the application manifest declares all required authority;
2. no undeclared network capability is reachable;
3. DNS resolves the hostname;
4. TCP connects with a finite timeout;
5. TLS validates the certificate and hostname;
6. the request is serialized exactly;
7. partial writes are handled;
8. fragmented responses are parsed;
9. fixed and chunked bodies are supported;
10. response size is bounded;
11. JSON is decoded safely;
12. every resource is cleaned up on success and failure;
13. Linux, macOS, and Windows qualification passes;
14. HIR, MIR, and native engines agree for all pure protocol behaviour;
15. exact provider identity and version are recorded;
16. every timeout and interruption terminates the current connection;
17. chunked trailers are consumed, bounded, validated, and discarded;
18. obs-fold is rejected;
19. the DNS provider ABI exactly matches WP-PKG-HOST-CAPABILITIES Part E;
20. TLS-over-TCP ownership is proven as one consumed resource transitioning to one output resource.

---

## 11. Immediate next work

Execute these first, in parallel:

### Work package 1

```text
HC0 — current-state inventory and contract freeze
```

### Work package 2

```text
HC1 — complete stark-url
```

### Work package 3

```text
HC5 + HC6 — HTTP core and request serializer
```

### Work package 4

```text
HC2 + HC3 — TCP public API and DNS provider design
```

Do not begin TLS implementation until:

- TCP ownership is frozen;
- provider-bound resource transfer is proven;
- time and secure-random dependencies are explicit;
- certificate-root policy has an owner decision.
---

## 12. General project law discovered by this roadmap

The following rule applies beyond HTTP:

> **Do not modify package semantics to hide a compiler defect.**

When a valid package design exposes a compiler, MIR, ABI, provider, or test-runner limitation:

1. preserve the intended package semantics;
2. reduce the limitation to a minimal reproducer;
3. classify the failing layer;
4. record any temporary workaround explicitly;
5. close the compiler/tooling defect with independent evidence;
6. remove the workaround when the defect closes.

A package must not silently weaken ownership, error behaviour, limits, capability declarations, or API shape merely to fit an implementation hole.

