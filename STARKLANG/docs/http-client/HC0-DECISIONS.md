# HC0-DECISIONS — frozen contracts for the HTTP client track

**Stage:** HC0 of `WP-HTTP-CLIENT-ROADMAP.md`. **Date:** 2026-08-01.
Every decision below is grounded in a measurement recorded in `HC0-CURRENT-STATE.md`.

---

## D1. Package names and dependency direction — FROZEN

```text
stark-ascii    (no deps)
stark-percent  -> stark-ascii
stark-url      -> stark-percent, stark-ascii
stark-mime     -> stark-ascii
stark-query    -> stark-percent
stark-form     -> stark-percent
stark-json     (no deps)
stark-net      -> provider: stark-net-native
stark-http-core   -> stark-url, stark-mime
stark-http-parser -> stark-http-core, stark-ascii
stark-http     -> stark-http-parser, stark-net, stark-url
stark-tls      -> provider: stark-tls-native
```

**Acyclic**, and the direction is one-way from data packages toward transport. `stark-http-core`
holds types only and must not depend on `stark-net`, so the protocol types stay testable without a
socket — the property that let the substrate packages be qualified before any of this existed.

## D2. Builder-held client references — FORBIDDEN

Core v1 refuses declared reference fields (`E0001`, probed). A builder therefore **cannot** hold
`&Client`, and HC0 §10's conditional resolves to its negative branch.

Consequence, frozen now so no later stage assumes otherwise: **the request builder owns its data and
the client is passed at send time.**

```text
client.send(request)       // client borrows request or takes it by value
NOT request.send()         // would require the request to hold a client reference
```

## D3. Request API shape — FROZEN as the roadmap requires

```text
let request = Request::post(target)
    .header(name, value)
    .body(bytes)
    .build()?;
let response = client.send(request)?;
```

Both halves are admitted: method chaining terminating in `.build()` compiles, and an associated
function returning `Result` consumed with `?` compiles. Frozen as `builder.build()?` then
`client.send(request)?`, per HC0's acceptance criterion.

## D4. String building uses `push_str`, not `+` — FROZEN

`String + String` is refused (`E0500`: `String` does not satisfy `Num`). The roadmap's own example
`"Bearer " + token` **does not compile**.

```text
let mut value = String::from("Bearer ");   // admitted
value.push_str(token.as_str());
```

Every header-construction fragment in the roadmap is rewritten to this form. This is a language
fact, not a package gap: arithmetic operators desugar to `Num`, which is compiler-known and
primitives-only, so no package can make `+` work on `String`. Concatenation via `+` would be a
language proposal, and is out of scope for this track.

## D5. JSON rendering uses `encode`, not `to_string` — FROZEN

`JsonValue::to_string()` does not exist. The admitted spelling is `stark_json::encode(&value)`,
returning `String`. HC11's convenience API is built on `encode`/`parse`, and no stage may assume a
method form.

## D6. Timeouts use `Duration::from_seconds` — FROZEN

`Duration::seconds` is an accessor. The constructor is `Duration::from_seconds` (or `from_millis`).
The roadmap's `Duration::seconds(30)` fragment is corrected wherever it appears.

## D7. Initial client scope — FROZEN

**In, for HC8 (plain HTTP):**
- HTTP/1.1 only
- `GET`, `POST`, `PUT`, `DELETE`, `HEAD`
- request: absolute target parsed into host/port/origin-form, headers, optional body
- response: status line, headers, body via `Content-Length` **and** chunked transfer-encoding
- connection close after each exchange — no pooling, no keep-alive reuse
- **address-literal connection only**, because no resolver exists (see D9)

**Out, and deferred to a named stage:** TLS (HC9/HC10), redirects (HC12), cookies, auth schemes,
proxies, compression, HTTP/2, connection pooling, request retry.

## D8. Qualification labels and evidence locations — FROZEN

Reusing the gate that exists rather than inventing one: every package added by this track joins
`starkc/scripts/qualify-first-party-packages.py`, which is the required CI check on three platforms
and runs check, test, `fmt --check`, consumer check, interpreter run with exact stdout, native build,
and native run with exact stdout.

```text
qualified      in CASES, green on linux-x64, macos-arm64, windows-x64
implemented    package tests pass locally, not yet in CASES
drafted        source exists, no tests
```

A package is **not** "done" at `implemented`. Three of the five substrate packages sat at `drafted`
with zero tests and unformatted source from the day they landed until they were put through this
gate — that is the precedent this labelling exists to prevent.

Evidence location: CI job output, plus the `stark-*-consumer` byte-exact stdout for each.

## D9. Connection targets before DNS — FROZEN

Until `WP-PKG-HOST-CAPABILITIES` Part E delivers a resolver, `Client::send` accepts a URL whose host
is an **address literal**. A hostname is refused with a distinct, named error rather than a generic
parse failure, so the gap is legible to a caller and cannot be mistaken for a malformed URL.

## D10. TLS transition protocol — DEFERRED with a fallback required

The `HandleConsumed` + `HandleOut` combination is neither proven nor forbidden (measured). HC0's
acceptance requires it be proven **or** a fallback designed before TLS work starts. It is neither
today, so:

- HC1 through HC8 proceed — none of them needs it;
- **HC9 may not begin** until either a probe proves a single provider call can consume a `TcpStream`
  and produce a `TlsStream`, or a two-call fallback is designed in which the stream is consumed and
  the TLS handle produced separately, with the intermediate state unrepresentable to STARK code.

Recorded as an entry blocker on HC9 rather than a blanket blocker, because treating it as the latter
would stall eight stages that do not depend on it.
