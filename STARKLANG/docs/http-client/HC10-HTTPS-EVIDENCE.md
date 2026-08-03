# HC10 — the HTTPS client: closure record and evidence

**Status:** CLOSED 2026-08-03
**Depends on:** HC8 (plain client), HC9 (`stark-tls`), CD-360, CD-361
**Enables:** HC11 (JSON convenience), HC12 (redirects)

---

## 1. The claim

> `Client::send` selects HTTP or HTTPS **from the URL scheme alone**. A STARK program can call an
> ordinary hostname-based HTTPS endpoint with certificate and hostname verification, and no
> application-level distinction beyond the URL is required.

Read in both directions, which is the part that matters: there is **no** per-request TLS switch, no
"insecure" flag, and no way to reach `https://` without verification. The scheme is the security
decision and the caller makes it in the URL.

### What this does NOT claim

| | |
| --- | --- |
| redirects | HC12. `follow_redirects: true` is refused, not ignored. |
| JSON convenience | HC11. A body is `Vec<UInt8>` and a content type is a header. |
| bundled root store | still `Unsupported`. Vendoring a CA list is a distribution decision nobody has taken. |
| IPv6 literal hosts | refused by the URL parser, as in HC8. |
| Profile F (FIPS) | unchanged from HC9 — not qualified. |

---

## 2. What changed

```text
parse URL ─┬─ http://  ─→ TCP ───────────────────→ exchange_plain
           └─ https:// ─→ TCP ─→ TLS handshake ──→ exchange_secure
```

* **`parse_url`** replaces `parse_http_url`. It returns a `Scheme`, and the default port comes from
  it (80/443). That default also feeds `HostAuthority`, so `https://host/` sends `Host: host` and
  not `Host: host:443`.
* **`HttpError`** replaces `ClientError`, carrying each layer's own error — `Dns(DnsError)`,
  `Network(NetworkError)`, `Tls(TlsError)` — rather than flattening to a string. A caller that wants
  to retry on a connection reset but not on a certificate failure has to be able to *match*; a
  rendered message cannot be matched on. `error_text` exists for reporting only.
* **`HttpTimeoutPhase`** makes timeouts phase-specific, as the roadmap requires. `NetworkError::TimedOut`
  and `TlsError::HandshakeTimeout` become `Timeout(phase)`; everything else keeps its own error, so
  nothing is lost.
* **`SystemRoots`** is implemented in the TLS provider (`rustls-native-certs` 0.8.2) and is the
  default for `default_config()`. CD-361's point, delivered: system roots without handing the
  protocol to a platform TLS stack — certificate validation stays rustls-owned.

### The transport is deliberately not abstracted

`exchange_plain` and `exchange_secure` are written out twice. Core v1 has no trait objects and no
closures, and the remaining shape — `enum Transport { Plain(TcpStream), Secure(TlsStream) }` —
**cannot work**: a pattern binds an enum payload by value or by shared borrow, there is no
`ref mut`, so a `&mut self` method can never reach the stream inside. Probed and confirmed:

```text
E0400 mutable method receiver requires a mutable place
```

Mutable payload binding in patterns is a **language question**, not a defect, and is recorded for
the owner rather than worked around silently. The two functions are kept deliberately parallel —
same order, same names, same comments — so a change to one is visibly a change to both.

---

## 3. Evidence

| what | where | count |
| --- | --- | --- |
| pure surface, incl. scheme and error taxonomy | `stark-http-client/src/tests.stark` | 21 |
| TLS provider, incl. `SystemRoots` | `stark-tls/native/src/lib.rs` `mod tests` | 22 |
| `stark-tls` package surface | `stark-tls/src/tests.stark` | 11 |
| executed native lifecycle, cleartext AND TLS | `stark-http-client-consumer` under the gate | 11 cases |

### The executed lifecycle

One binary, one `fetch`/`send` call shape, eleven cases:

```text
  fixed: 200, Content-Length framing, body and headers intact
  chunked: 200, chunks decoded and joined
  fragmented: 200, reassembled across several socket reads
  close-early: reported as expected: the peer closed before the response completed
  refused port: connect failure reported, no stream acquired
  https: 200 over a verified TLS 1.3 session, headers and body intact
  https: chunked body decoded inside the TLS session
  https: untrusted chain refused, reported as a TLS failure
  https: hostname mismatch refused, though the chain itself is trusted
  https: a cleartext peer on the secure path is refused
  https: POST with a JSON body and a bearer token arrived intact
STARK_HTTP_CLIENT_RESOURCE_OK
```

Three of those are **refusals**, and that is the point. A gate that observed only the happy path
would pass just as well against a client that skipped verification entirely — the failure mode that
matters, and the one invisible from outside.

**The hostname case is the sharpest.** The peer on 39192 holds a valid, trusted certificate for
`localhost`. Reaching the *same peer* as `127.0.0.1` must fail. Same bytes, same anchor, different
name: the only variable is the name check.

**The POST case asserts the request, not the response.** The `/echo` route reflects the method, the
`Authorization` header, the `Content-Type` and the body it actually received. A route returning a
constant would pass for a client that sent nothing.

### Independent corroboration, outside the gate

A separate reviewer wrote their own client against the public API only and ran it against real
hosts: `GET https://api.github.com/rate_limit` returned 200 with JSON, over TLS validated against
the **system** trust store, with request headers reaching the server and response headers parsed
back. Two calls hit different backends (Varnish and fasthttp) and both framed correctly; an earlier
call returned GitHub's 403 rate-limit JSON, which is itself strong evidence — an application-layer
answer means the chain verified and the response was framed and decoded.

**This is corroboration, not gate evidence, and the distinction is deliberate.** HC13 forbids
qualification depending on public internet services, so nothing in the gate may rely on it. It is
recorded because it covers the one direction the offline tests cannot: `SystemRoots` is tested here
NEGATIVELY — the fixture CA is not in any machine's store — and only a real chain against a real
store shows the positive.

The same exercise confirmed the cleartext behaviours the gate already asserts (Content-Length,
chunked reassembly, 404 surfaced as a status rather than an error, POST echo, typed connect-refused
and DNS-failure errors) and the documented IPv6-literal refusal.

### Why the fixtures differ from HC9's

`stark-tls`'s own consumer dials 127.0.0.1 and presents `stark.test` separately, so it needs no
resolvable name. The HTTP client **resolves the URL's host**, so its fixture must use a name that
actually resolves — hence `localhost.cert.pem`. The first attempt reused `stark.test` and failed at
DNS, before the certificate check the case existed for.

---

## 4. Findings

| id | what | status |
| --- | --- | --- |
| **DEV-158** | Assigning over a struct field whose old value is a **drop unit**, then using the struct as a whole, aborts in native code with "mutable access to a dead slot: the slot is PARTIAL". **The interpreter accepts the same program** — a three-engine divergence. See `COMPILER-STATE.md` CD-366 for the reduced case, the real mechanism (a move-out to a temp in `lower_overwriting_assign`, not a drop-in-place), and the candidate fix. | OPEN |
| **CE-shaped** | Core v1 has no mutable binding of an enum payload in a pattern (`ref mut`). This is what makes a single `Transport` abstraction impossible; two parallel flows are the consequence. A **language** question for the owner, not a defect. | RAISED |
| — | `stark-http-client` now declares the `tls` capability and depends on `stark-tls`. Its resource set is `("TcpStream", "TlsStream")` in the gate. | done |
| **DEV-159** | A native build can FAIL once and succeed on retry: the generated crate raced its own `aws-lc-rs` dependency build. Reported by an independent reviewer building an HTTPS program at this HEAD. A user hitting this sees a confusing failure; at minimum the diagnostic should say to retry, and better, the build should not race. | OPEN |
| — | **An ergonomics gap two people hit independently:** with no `body_text`, both reached for a method that did not exist and then copied the same manual `Char::from_u32` byte loop out of an existing consumer. That loop is Latin-1, not UTF-8 — silently wrong for any non-ASCII body. Closed by HC11's `HttpResponse::body_text()` and its strict decoder. | closed by HC11 |

**HC10's workaround for DEV-158** is `config_with_explicit_roots` building one struct literal rather
than assigning over a field of `default_config()`. Same semantics, same API — an implementation
change, not a weakening — with the reason recorded inline. **Remove it when DEV-158 closes.**

---

## 5. Version additions

```text
rustls-native-certs =0.8.2    the platform trust store, read INTO rustls
```

Pinned exactly, as CD-361 requires of everything in this stack.
