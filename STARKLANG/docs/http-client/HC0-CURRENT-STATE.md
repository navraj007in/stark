# HC0-CURRENT-STATE — the networking and package surface, confirmed from source

**Stage:** HC0 of `WP-HTTP-CLIENT-ROADMAP.md`.
**Date:** 2026-08-01.
**Method:** every symbol below was read from source or produced by running the compiler. Nothing here
is quoted from a roadmap, a prior report, or memory. Where a roadmap statement disagrees with what
was measured, the measurement is recorded and the disagreement is called out.

---

## 1. TCP provider surface (`stark-net`)

**`stark-net` has no STARK package.** The directory contains `native/` only — a provider crate. There
is no `src/lib.stark`, so there is nothing importable from STARK today. HC2 therefore begins from
zero package surface, not from an incomplete one.

Seven exported provider symbols, with their ABI parameter kinds extracted from the declarations:

| Function | ABI parameters | `is_close_for` |
| --- | --- | --- |
| `stark_tcp_listener_bind` | `BufferIn`, `HandleOut` | none |
| `stark_tcp_listener_accept` | `HandleBorrowed`, `HandleOut` | none |
| `stark_tcp_stream_connect` | `BufferIn`, `HandleOut` | none |
| `stark_tcp_stream_read` | `HandleBorrowed`, `BufferInOut`, `ScalarOut` | none |
| `stark_tcp_stream_write` | `HandleBorrowed`, `BufferIn`, `ScalarOut` | none |
| `stark_tcp_listener_close` | `HandleConsumed` | `Some(listener)` |
| `stark_tcp_stream_close` | `HandleConsumed` | `Some(stream)` |

### Status of each item HC0 §2 asks about

| Item | Status |
| --- | --- |
| `TcpStream` type | exists as a provider RESOURCE, bound through `HandleOut`/`HandleBorrowed`/`HandleConsumed`. No STARK-side nominal in a package. |
| TCP connect | `stark_tcp_stream_connect`, address as `BufferIn`. **No separate host/port** — the address is one buffer. |
| read | `stark_tcp_stream_read`, `BufferInOut` + `ScalarOut` count. |
| write | `stark_tcp_stream_write`, `BufferIn` + `ScalarOut` count. |
| close / drop | `stark_tcp_stream_close`, a declared close function, so MIR manages it as resource destruction. |
| **shutdown** | **DOES NOT EXIST.** No `stark_tcp_stream_shutdown` symbol. Half-close is unavailable, which matters for HTTP request framing that signals end-of-body by shutdown. |
| provider synthesis | works — `stark test` synthesises `provider_api` (CD-300); before that every generated `*_raw` was E0200. |
| provider-bound package tests | work — `stark test` no longer panics on a package with dependencies (CD-302). |
| native qualification | the C7 P1 REST workload builds and passes 24 byte-exact HTTP exchanges on three platforms, so the TCP path is proven end to end at the workload level, though not as a package. |

## 2. Packages HC0 §3 asks about

### `stark-url` — origin-form only

```text
percent_encode_component(&str) -> String
percent_decode(&str) -> Result<String, UrlError>
parse_request_target(&str) -> Result<RequestTarget, UrlError>
parse_request_target_with_limits(&str, &UrlLimits) -> Result<RequestTarget, UrlError>
encode_query(&Vec<QueryParameter>) -> String
default_limits() -> UrlLimits
RequestTarget { path: String, query: Vec<QueryParameter> }
QueryParameter { name: String, value: String }
UrlLimits, UrlError, UrlErrorKind
```

**There is no absolute-URL parsing.** No scheme, host, port, userinfo or fragment. `RequestTarget`
is path + query, which is the origin-form of a request line. An HTTP client needs scheme/host/port to
decide where to connect, so HC1 is not a polish task — it is the missing half of the package.

### `stark-json` — parse and encode, no `to_string`

```text
parse(&str) -> Result<JsonValue, JsonError>
parse_with_limits(&str, JsonLimits) -> Result<JsonValue, JsonError>
encode(&JsonValue) -> String
number_raw(&JsonNumber) -> &str
member_key(&JsonMember) -> &str
member_value(&JsonMember) -> &JsonValue
default_limits() -> JsonLimits
JsonValue, JsonMember, JsonNumber, JsonLimits, JsonError, JsonErrorKind
```

**`JsonValue::to_string()` does not exist** — HC0 §8 asks specifically. The equivalent is the free
function `encode(&JsonValue) -> String`. Any roadmap fragment using `.to_string()` on a `JsonValue`
must be rewritten.

Qualified: 10/10 package tests, native consumer byte-exact, in CI on three platforms (CD-329).

### `stark-time` — `Duration` exists; the constructor is not what the roadmap names

```text
Duration { seconds: UInt64, nanoseconds: UInt32 }
  Duration::from_seconds(UInt64) -> Duration      <- CONSTRUCTOR
  Duration::from_millis(UInt64) -> Duration
  duration.seconds(&self) -> UInt64               <- ACCESSOR
Instant { ticks_nanos: UInt64 }
UnixTimestamp { seconds: Int64, nanoseconds: UInt32 }
TimeError
```

**`Duration::seconds` is an accessor, not a constructor.** The roadmap's fragment `Duration::seconds(30)`
does not compile; the admitted spelling is `Duration::from_seconds(30)`.

### `stark-random` — v0.1 landed

Four public items; package tests pass. Landed CD-297, which found four defects only execution could
surface — including that `next_u64` trapped on its second call, because STARK traps on integer
overflow in every build mode and a shift discarding set bits is an overflow.

## 3. Language surface, probed by compilation

Each probe below was compiled with `stark check`. These are results, not readings of the spec.

| Fragment | Result |
| --- | --- |
| `struct C { host: &String }` | **REFUSED** — `E0001 Core v1 does not permit declared reference fields` |
| `String::from("Bearer ") + token` | **REFUSED** — `E0500 type 'String' does not satisfy operator trait 'Num'` |
| builder method chain ending `.build()` | **ADMITTED** |
| associated fn returning `Result` consumed with `?` | **ADMITTED** |

## 4. Provider ABI: `HandleConsumed` + `HandleOut` in one call

HC0 §9 asks whether one provider call can both consume a handle and produce a new one — the shape a
TLS upgrade needs (`HandleConsumed(TcpStream)` → `HandleOut(TlsStream)`).

**Neither proven nor forbidden.** No existing provider function combines them: the only two
`HandleConsumed` uses are `stark_tcp_listener_close` and `stark_tcp_stream_close`, each a bare close
with one parameter. The "exactly one `HandleConsumed` and nothing else" rule in `provider_abi.rs`
§13.1 constrains functions declared `is_close_for: Some(..)`, and does not by its terms govern a
non-close function. `provider_bind.rs` builds an input plan for `HandleConsumed` generically and
`lower.rs` has independent arms for both kinds, so the combination looks structurally admissible.

It has never been exercised, so this is an untested path, not a supported one.

## 5. DNS

**No DNS provider exists.** No resolver symbol in `stark-net/native`, and no DNS specification found
in the host-capability roadmap's text under that name. `WP-PKG-HOST-CAPABILITIES.md` Part E is titled
"DNS" and is the intended authority; HC3 depends on it and cannot start before it is written.

Consequence for scope: without a resolver, a client can only connect to an address literal.
