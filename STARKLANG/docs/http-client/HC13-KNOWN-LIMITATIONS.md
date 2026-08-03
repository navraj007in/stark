# HC13 — known limitations

**Status:** current as of 2026-08-03
**Read this before the qualification report.** The report says what works. This says what does not,
and it is the shorter and more useful document.

A limitation here is one of three things, and the distinction matters more than the list:

| kind | meaning |
| --- | --- |
| **refused** | the client detects the situation and returns a named error. Safe; the caller can act. |
| **absent** | the feature is not implemented and not pretended. A caller who needs it must not use this client. |
| **unproven** | the code does something, and no test in this repository demonstrates it. **Treat as untested, not as working.** |

The third kind is the one that gets people hurt, so it is listed first.

---

## 1. Unproven

### 1.1 Timeout phases — what is proved, what is unproven, what is absent

`HttpTimeoutPhase` has five values. **Two are proved. One is unproven. Two are not implemented at
all**, and conflating the last two categories is the mistake this section previously made.

```text
ReadResponse    PROVEN        two stalling peers -- headers, and mid-body
TlsHandshake    PROVEN        a TCP peer that accepts and never speaks TLS
WriteRequest    UNPROVEN      the deadline IS installed on the socket; no peer fills a receive
                              window, so expiry has never been observed end to end
Connect         NOT IMPLEMENTED   see DEV-165 below
Resolve         ABSENT            no mechanism exists that could produce it
```

`ReadResponse` counts once, not twice. The three stalling routes prove **two distinct phases**;
`/slow-headers` and `/slow-body` both report `ReadResponse`, and describing them as three phases
overstated the evidence.

**DEV-165 — `ClientConfig.connect_timeout` is advertised but not enforced.** The client calls
`connect_no_timeout(target)` and installs read and write deadlines only *after* the connection
completes. `config.connect_timeout` is read by nothing. Underneath, `stark-net::connect` refuses
every non-zero timeout outright:

```stark
pub fn connect(address: SocketAddress, timeout: Duration) -> Result<TcpStream, NetworkError> {
    if !timeout.is_zero() {
        return Err(NetworkError::Unsupported);
    }
    connect_socket_address(&address)
}
```

So this is an **implementation gap, not an untested success**: a caller setting `connect_timeout`
gets no error and no effect. Deferred to the networking roadmap — enforcing it needs a non-blocking
connect plus a poll, which is a provider ABI change, not a client fix.

**`Resolve` is ABSENT, not merely unproven.** `stark-net::resolve` takes a host, a port and
size/count limits, and passes no duration or deadline to the DNS provider. There is no mechanism
that could produce `Timeout(Resolve)`. Implementing one likely needs either a different provider ABI
or a bounded resolver worker, because ordinary blocking system resolution exposes no portable
per-call timeout.

**Why the distinction is load-bearing.** DEV-163 was a phase whose deadline *worked* and whose
*report* was wrong, and it took a stalling peer to find. A phase with no deadline at all fails
differently and is not the same risk — but a document that files both under "unproven" invites a
reader to assume both merely lack a test.

### 1.2 Cross-compilation

`stark build --target <triple>` validates the triple and then refuses. Tier-1 binaries are built
natively on their own platform, in CI. Nothing here has been cross-compiled.

---

## 2. Absent

| what | note |
| --- | --- |
| HTTP/2, HTTP/3 | HTTP/1.1 only. There is no negotiation and no ALPN advertisement of `h2`. |
| Connection reuse / keep-alive | one exchange per connection, closed after. Correct but slow for repeated calls to one host. |
| Response decompression | `Content-Encoding: gzip` is returned as the bytes on the wire. The client does not advertise it. |
| Proxies | no `HTTP_PROXY`/`HTTPS_PROXY`, no `CONNECT`. |
| Cookie jar | `Cookie` may be set per request; nothing stores or replays one. |
| Client certificates | server authentication only. |
| Streaming bodies | a body is a buffered `Vec<UInt8>`, bounded by `max_response_bytes`. There is no incremental reader, so a response larger than the ceiling cannot be processed at all — only refused. |
| Dot-segment resolution | `.` and `..` in a redirect `Location` are not removed (HC12.1). Deferred to a bounded RFC 3986 resolver in `stark-url`, rather than a second URL implementation inside the client. |
| Connect and resolve deadlines | `connect_timeout` is accepted and ignored (DEV-165); no resolve deadline exists at all. See §1.1. |

---

## 3. Refused, by design

These are not gaps. Each is a decision with a reason, and the reason is usually that the safe
behaviour and the convenient behaviour differ.

| situation | what happens |
| --- | --- |
| `https://` → `http://` redirect | `InsecureRedirect`, before anything is dialled. |
| origin change while following | `Authorization` and `Cookie` are stripped. Opt-out exists and is named `preserve_authorization_same_origin_only`. |
| two `Location` headers | `AmbiguousLocation`. Following the first would be a silent choice between two things the server said. |
| foreign-scheme or fragment-only `Location` | `InvalidRedirectTarget` — neither names anything fetchable. |
| `Content-Length` with `Transfer-Encoding` | refused as ambiguous framing. This pair is the classic request-smuggling primitive. |
| obs-fold, bare LF | refused. Deprecated is not the reason; two hops disagreeing about where a value ends is the reason. |
| CR or LF in a header value | refused at the **serializer** boundary, not only at construction. `Header`'s fields are public, so constructor validation was bypassable — see §4. |
| a body shorter than `Content-Length` | `ClosedEarly`. Never returned as a short body. |

---

## 4. Known weaknesses in the public API

### 4.1 `Header` and `HeaderMap.entries` are public

A caller can construct a `Header` with an invalid name or a value containing CRLF, bypassing
`append_validated`. **The wire is safe** — HC12.1 moved validation to the serializer, and the
regression test for it is the working exploit that produced an injected header line before the fix.
But the *type* still permits an invalid value to exist in memory, and any future code that reads
`entries` directly inherits that.

Making them private is an API break, which is why it is its own packet rather than a quiet change
here.

### 4.2 A `String` is not guaranteed valid UTF-8 by the type

`body_text` decodes and reports `TextDecodeError`; that path is safe. The concern is that nothing in
the type system distinguishes a validated header value from an arbitrary one.

---

## 5. Compiler limitations this stack works around

These are STARK compiler defects, not HTTP defects. They are listed because they are visible in the
package source and a reader will otherwise wonder why the code is shaped oddly.

| id | effect on this code |
| --- | --- |
| **DEV-160b** | `send_once(builder.url.as_str(), builder.headers, builder.body)` does not build natively: the borrow reaches the call from an earlier block. The fields are bound to locals first. Refused by name since CD-374, deferred by owner ruling. |
| **DEV-160c** | a provider call with the same shape is refused by name. Not hit by this stack. |
| **DEV-160d** | a borrow outliving such a call is refused by name. Not hit by this stack. |
| **DEV-156** | `stark fmt` evicts a struct field's or enum variant's doc comment to after the type. Every field-level explanation here is folded into the type's own doc comment, which is why those are long. |
| **DEV-157** | `panic` in value position has no native representation. Match arms are nested instead. |
| **DEV-159** | a native build can race its own dependency build. Unreproduced. |
| **DEV-163** | a socket read deadline reported as a connection failure on Unix. **Fixed** in CD-375; listed because it explains why §1.1's untested phases are called *unproven* rather than *working*. |
| **DEV-164** | `stark-net`'s provider tests shared process-global state under `cargo test`'s parallelism. **Fixed** in CD-375 by serialising every test that opens a socket. |
| **DEV-165** | `ClientConfig.connect_timeout` is advertised and never enforced — see §1.1. **Open**, deferred to the networking roadmap. |

---

## 6. What this client is for

It fetches and posts to HTTP/1.1 and HTTPS endpoints, verifies certificates and hostnames by
default, bounds what it will read, follows redirects only when asked and never silently forwards
credentials across an origin. That is a real and useful envelope.

It is **not** a general-purpose replacement for a mature client. If an application needs connection
pooling, HTTP/2, streaming bodies or proxy support, this client will not do — and it says so here
rather than appearing to work and then not.
