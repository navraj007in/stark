# HC11 — JSON convenience: closure record and evidence

**Status:** CLOSED 2026-08-03
**Depends on:** HC10 (HTTPS client), `stark-json`
**Enables:** HC12 (redirects), HC13 (release)

---

## 1. The claim

> Common JSON REST calls no longer require manual byte conversion or header construction, and HTTP
> core still knows nothing about JSON.

```stark
let request = post(url, empty_body).json(&value)?;
let response = client.send(request)?;
let parsed = response.json_checked()?;
```

### What this does NOT claim

| | |
| --- | --- |
| typed codecs | out of scope by roadmap §2. `JsonValue` in, `JsonValue` out; no derive, no schema. |
| streaming JSON | the body is fully buffered before parsing, bounded by `max_response_bytes`. |
| a JSON dependency in HTTP core | deliberately absent — see §2. |

---

## 2. Where the pieces live, and why they are split

```text
stark-http-core     TextDecodeError, decode_utf8, HttpResponse::body_text
stark-http-client   RequestBuilder::json, HttpResponse::json / json_checked, JsonBodyError
```

`body_text` is in **core** because `HttpResponse` is declared there — a method on your own type has
no coherence question, and text decoding is HTTP-adjacent rather than JSON-specific.

The JSON half **cannot** be in core: `stark-http-core` must not depend on `stark-json`, or every
program that parses a header pulls in a JSON parser. The composition package is the only place that
already depends on both, so that is where it goes.

> **A coherence hazard, noted:** STARK permits an inherent `impl` on a foreign type — verified, and
> that is how `HttpResponse::json` is declared from the client package. Rust forbids this. Nothing
> prevents two packages adding a `json` method to the same foreign type and colliding. It is not a
> problem today (one package does it) and it is what let the roadmap's frozen call shape be matched
> exactly, but it is a real gap in the orphan rule and is recorded rather than relied upon.

---

## 3. The UTF-8 decoder is the substantial part

There is no `String::from_utf8` in the core surface, so HC11 had to write one. Strict, with the
accepted set expressed as explicit ranges rather than "leading byte, then N continuations" — because
the short form accepts three things it must not:

| accepted by the naive form | why it matters |
| --- | --- |
| **overlong** `C0 80` | NUL written in two bytes; a checker scanning decoded text never sees it |
| **surrogates** `ED A0 80`–`ED BF BF` | `U+D800`–`U+DFFF` are not scalar values |
| **out of range** above `U+10FFFF` | not a character |

Each is a documented source of parser-differential bugs: two components disagreeing about what a
byte string means is how a filter gets bypassed.

**Strict also means no replacement characters.** An invalid sequence is an error carrying the byte
offset. A client that substitutes `U+FFFD` hands its caller a body that differs from what the server
sent, and nothing downstream can detect the difference.

### The gap this closed, found twice independently

Before HC11 there was no `body_text`. Two people — the author of the first consumer and an
independent reviewer writing their own client — each went looking for the obvious method, did not
find it, and copied the same manual loop out of an existing consumer:

```stark
match Char::from_u32(response.body[i] as UInt32) { .. }   // WRONG
```

That is Latin-1, not UTF-8. It treats each byte as a code point, so `é` comes back as two garbage
characters and `😀` as four — fine for ASCII, silently wrong for everything else, which is the worst
failure shape available. Two independent people reaching the same wrong idiom is the argument that
the helper had to exist.

---

## 4. The content-type policy has two settings, deliberately

| | behaviour |
| --- | --- |
| `json()` | parses whatever the body is, **regardless** of `Content-Type` |
| `json_checked()` | requires `application/json`, refuses otherwise and quotes what was seen |

Servers mislabel JSON often enough that refusing on the header alone would break working
integrations — and the parse is the real check. But a caller who would rather fail than parse
something an endpoint never claimed was JSON (an HTML error page from a proxy, which occasionally
parses as a JSON string and would be returned as if it were the answer) needs the strict form.

`application/json; charset=utf-8` is the same media type as `application/json`; the comparison takes
the part before `;`, trims space, and is ASCII case-insensitive. A client that refused the
parameterised form would be wrong about a large share of real servers.

---

## 5. Evidence

| what | where | count |
| --- | --- | --- |
| UTF-8 decoder: valid, boundaries, overlong, surrogate, out-of-range, truncated, offset | `stark-http-core/src/tests.stark` | 29 total (10 new) |
| JSON API: encode, parse, unicode, malformed, empty, content-type policy, non-UTF-8 | `stark-http-client/src/tests.stark` | 29 total (8 new) |
| executed native round trip over TLS | `stark-http-client-consumer` under the gate | 12th case |

The consumer's twelfth case encodes a value containing a four-byte scalar, POSTs it over a verified
TLS session, and re-encodes what comes back:

```text
  https: JSON encoded, sent, echoed and parsed back identically
```

Comparing **re-encoded** values rather than destructuring is the stronger assertion: it exercises
decode *and* encode, so a decoder and an encoder wrong in the same direction cannot agree their way
past it.

---

## 6. Findings

| id | what | status |
| --- | --- | --- |
| **DEV-158** | Hit a SECOND time, in `RequestBuilder::json` — `out.body = body` assigns over a `Vec<UInt8>`, a drop unit. Green under the interpreter, aborted natively. Fixed the same way: construct the builder as a literal from moved fields. | OPEN, worked around twice |
| **DEV-157** | Hit again: a helper with an `Err(_) => panic(..)` arm is a value-position `!`, which the native backend cannot represent. Nested instead. | OPEN |
| — | Payload binding through a reference in a `match` is refused ("bind by reference or match a Copy field instead"), and a NAMED wildcard (`_other`) binds where a bare `_` does not. Both shaped the test code; neither is a defect. | noted |

**DEV-158 has now cost two workarounds in one work package.** Both were caught only by a native run
— the interpreter accepted both programs. That is the argument for prioritising it: the divergence
means the cheap engine cannot be trusted to find it, and every future package that writes
`x.field = <owned>` is exposed.
