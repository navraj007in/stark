# HC13 — threat model

**Status:** current as of 2026-08-03
**Scope:** `stark-http-client` and everything under it — `stark-tls`, `stark-net`,
`stark-http-parser`, `stark-http-serialize`, `stark-url`, `stark-json`.

This document names what the client defends against, what it does not, and — for each defence —
**the test that would fail if the defence were removed.** A threat model with no falsifier attached
is a list of intentions.

---

## 1. The adversary

A **hostile server**, or anyone who can act as one: a compromised host the application chose to
talk to, an operator of a redirect target, a machine-in-the-middle who can answer TCP but cannot
produce a certificate the client's trust anchors sign.

The adversary controls: response bytes, timing, framing, headers, `Location` targets, and how much
data they send and when they stop.

**Out of scope:** an attacker with the private key of a trusted CA; an attacker who controls the
application's own process, memory or trust store; denial of service against the host machine.

---

## 2. Defences, and what proves each

### T1 — Impersonating a server the caller named

*Present a certificate for a name the client did not ask for, or one signed by a root it does not
trust.*

| defence | falsifier |
| --- | --- |
| chain verification against explicit or system roots | `https: untrusted chain refused, reported as a TLS failure` — a peer whose root the client is not given |
| hostname verification, independent of chain validity | `https: hostname mismatch refused, though the chain itself is trusted` |

The second case matters more than the first, and is the one clients get wrong: the chain is
**valid**, and it is still the wrong server. A client checking only the chain passes T1's first test
and fails this one.

### T2 — Downgrading a secured request to cleartext

*Answer an `https://` request with a redirect to `http://`.*

| defence | falsifier |
| --- | --- |
| downgrade refused before DNS and before connect | `redirect: https to http refused as a downgrade` |
| scheme-relative `//host/path` inherits the **current** scheme | HC12 resolution tests |

The check runs on the parsed target, so a downgraded request never reaches the wire — not even a
SYN.

### T3 — Harvesting credentials through a redirect

*Redirect to a host you control and collect the `Authorization` header.*

| defence | falsifier |
| --- | --- |
| `Authorization` and `Cookie` stripped on any origin change | `redirect: Authorization stripped when the origin changed` |

**The falsifier reads the wire, not the policy flag.** A cleartext peer redirects to the TLS peer —
a genuinely different origin — and the `/echo` route reflects what it received. Asserting
`GET|-|-|` proves the header was absent on the second request, not merely that a boolean was set.

Origin comparison uses the **effective port**, so `https://h/` and `https://h:443/` are one origin.
Otherwise a redirect that only spelled the port differently would strip credentials for no reason,
and callers would learn to turn the stripping off — which is how a safety default dies.

### T4 — Request smuggling and response splitting

*Get two hops to disagree about where a message ends.*

This is the largest defended surface, because HTTP's framing has several ways to be ambiguous and
each has been used in a real attack.

| primitive | defence | falsifier |
| --- | --- | --- |
| `Content-Length` **and** `Transfer-Encoding` | refused as ambiguous framing | `/bad-length-and-te` |
| two `Content-Length` values | refused | `/bad-two-lengths` |
| bare LF as a line terminator | refused | `/bad-bare-lf` |
| obs-fold continuation lines | refused | `/bad-obs-fold` |
| space inside a header name | refused | `/bad-header-name` |
| unsupported transfer coding | refused | `/bad-transfer-encoding` |
| chunk data not terminated by CRLF | refused | `/bad-chunk-terminator` |
| non-numeric chunk size | refused | `/bad-chunk-size` |
| two `Location` headers | `AmbiguousLocation` | HC12.1 |
| CR/LF injected into an outbound header value | refused at the **serializer** | HC12.1's regression test, which is the working exploit |

**On the last row.** `Header`'s fields are public, so validating only in the constructor was
bypassable, and the serializer trusted its input. A value of `safe\r\nInjected: yes` produced a
genuine extra header line on the wire. The fix revalidates every header at the serializer boundary
— the last point before bytes leave — and the regression test is the exploit that worked.

**Every falsifier in this table is a live peer sending real bytes**, not a parser unit test.
`stark-http-parser`'s 34 unit tests assert that malformed input is *rejected*; they do not assert
*which* of its 23 error variants each input produces, and they hand the parser a literal rather than
delivering it over a socket in pieces after a plausible status line — the only form an attacker can
use. Both gaps matter: a parser that collapsed every fault to one error would pass those tests and
fail these.

### T5 — Resource exhaustion

*Send more than the client can hold.*

| limit | falsifier |
| --- | --- |
| `max_status_line_bytes` | `/big-status-line` |
| `max_header_line_bytes` | `/big-header-line` |
| `max_header_count` | `/big-header-count` |
| `max_response_bytes` | `/big-body` |
| `max_redirects` and loop detection | `/r-hopN` and `/r-loop`, separately |

`max_response_bytes` is enforced on **total bytes read**, not on the parsed body, so a peer cannot
evade it by lying in `Content-Length`. `/big-body` declares 12 MiB and actually sends it; the client
stops at its ceiling and closes. A header check alone would pass a peer that under-declares and
overspends.

The bound and the loop are proved **separately** — `/r-loop` revisits one target, `/r-hopN` walks an
ever-lengthening chain of distinct targets. One test could not distinguish the count bound from the
loop detector, and the two errors exist because a caller raising the limit should fix one and not
the other.

### T6 — Hanging the client

*Accept the connection and then say nothing.*

| phase | falsifier |
| --- | --- |
| TLS handshake | a TCP peer that accepts and never speaks TLS |
| reading headers | `/slow-headers` |
| reading the body | `/slow-body` |

`/slow-body` is the sharp one: a complete, plausible head arrives, and *then* the body stops. A
client that applies its read deadline only while waiting for headers hangs here for as long as the
peer cares to hold the socket.

**DEV-163 was found here.** A read timeout surfaced as `NetworkError::Interrupted` on Unix and
`NetworkError::TimedOut` on Windows, so the identical peer produced "the connection failed" on macOS
and "timed out reading the response" on Windows. The deadline worked; its *report* did not, and an
operator would have looked at the network instead of the peer. Invisible to every test that used a
peer which answers.

Two phases — `Connect` and `Resolve` — are **not** proved. See HC13-KNOWN-LIMITATIONS.md §1.1; they
are the same shape of risk that DEV-163 turned out to be.

### T7 — Truncation passed off as a complete response

*Promise a body and close early.*

| defence | falsifier |
| --- | --- |
| a short body is an error, never a quiet truncation | `close-early: reported as expected` |

A client that returns partial data as complete corrupts every caller downstream of it, silently,
and the corruption is indistinguishable from a legitimately short response.

### T8 — Reaching the host without permission

| defence | falsifier |
| --- | --- |
| a provider is unreachable unless the manifest declares its capability | `c78_capability_declaration` |
| application code cannot call raw ABI symbols | the declared-surface gate |
| a handle is consumed at call entry; no double close | `c788_resource_lifecycle`, and the live-stream count asserted around every consumer run |

---

## 3. Trust boundaries

```text
application code
   │  ← STARK types only; no raw pointers, no ABI symbols
stark-http-client
   │  ← Result<T, HttpError>; every failure named
stark-tls / stark-net  (STARK packages)
   │  ← Provider ABI v0.1: ProviderStatus, RawResourceHandle, BorrowedBuffer
native providers  (Rust)
   │  ← rustls 0.23 + aws-lc-rs 1.17, std::net
the network        ← hostile
```

The boundary that carries the most weight is the third: it is where `unsafe` lives, where handles
are transferred (CD-360 — ownership passes at call entry, unconditionally), and where DEV-163
turned out to be hiding.

---

## 4. What an application still has to do

The client does not know the application's policy. These remain the caller's:

- **choosing the URL.** Nothing here defends against fetching a URL an attacker supplied — SSRF is
  an application concern, and this client will faithfully dial a link-local address if asked.
- **deciding whether to follow redirects at all**, and whether to raise the bound.
- **deciding what a response means.** A verified TLS session to a hostile-but-genuine server is
  still a hostile server.
- **not turning off `preserve_authorization_same_origin_only`** without understanding that an open
  redirect on the first host then becomes credential exfiltration to the attacker's.
