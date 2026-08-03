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

### 1.1 The `Connect` and `Resolve` timeout phases

`HttpTimeoutPhase` has five values. Three are proved on the wire by a stalling peer
(`TlsHandshake`, and `ReadResponse` in both its header and body positions). Two are **not**:

```text
Resolve        no test stalls a DNS server
Connect        no test black-holes a SYN
WriteRequest   no test fills a peer's receive window
```

These are not proved because a loopback cannot produce them deterministically. Reaching a
black-holed address depends on the network the test runs on; a flaky negative test is worse than an
absent one, because it teaches people to re-run the suite until it passes.

`connect_timeout` and `write_timeout` **are** applied to the socket — that much is visible in
`stark-net`'s `set_read_timeout`/`set_write_timeout` calls and is covered by unit tests. What is
unproven is the end-to-end claim that expiry surfaces as `Timeout(Connect)` rather than as some
other error.

**This is not hypothetical.** DEV-163 was exactly that failure in the phase that *is* now tested:
a read timeout surfaced as `NetworkError::Interrupted` on Unix and `NetworkError::TimedOut` on
Windows, so the same peer produced "the connection failed" on macOS and "timed out reading the
response" on Windows. It was invisible to every test using a peer that answers, and it was found
the day a peer that stalls was written. **The two untested phases are the same shape of risk, and
nobody should assume they are right because the tested one now is.**

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

---

## 6. What this client is for

It fetches and posts to HTTP/1.1 and HTTPS endpoints, verifies certificates and hostnames by
default, bounds what it will read, follows redirects only when asked and never silently forwards
credentials across an origin. That is a real and useful envelope.

It is **not** a general-purpose replacement for a mature client. If an application needs connection
pooling, HTTP/2, streaming bodies or proxy support, this client will not do — and it says so here
rather than appearing to work and then not.
