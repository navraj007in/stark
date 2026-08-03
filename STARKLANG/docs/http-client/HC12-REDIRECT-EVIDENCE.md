# HC12 — safe redirects: closure record and evidence

**Status:** CLOSED 2026-08-03
**Depends on:** HC10 (HTTPS client)
**Enables:** HC13 (release)

---

## 1. The claim

> Redirect support is **opt-in**, **bounded**, and cannot **silently** forward credentials to
> another origin.

All three words are load-bearing, and each is a separate mechanism:

| word | mechanism |
| --- | --- |
| opt-in | `follow_redirects` defaults false. Off, a `3xx` is returned as the response it IS — not an error, because a redirect is a valid answer and hiding it would misreport what the server said. |
| bounded | `max_redirects` (default 5) **and** loop detection over every visited URL. Two different faults, two different errors. |
| not silently | `Authorization` and `Cookie` are stripped on any origin change, always. Opting out is possible and is named `preserve_authorization_same_origin_only`. |

---

## 2. The rules, and why each is what it is

### The method-rewrite table

| status | method | body |
| --- | --- | --- |
| 301, 302 | `POST` → `GET`, others unchanged | dropped when rewritten |
| 303 | always `GET` | always dropped |
| 307, 308 | unchanged | replayed |

301/302 rewriting `POST` to `GET` **contradicts a literal reading of the RFCs** and matches what
every browser and `curl -L` do. Following the letter would send a POST body to a target the origin
server redirected a POST *away* from — both surprising and the more dangerous reading. 307 and 308
exist precisely to say "no, really, replay it", and that is honoured — safe only because a body is a
buffered `Vec<UInt8>`, so replay costs bytes and nothing else.

### Dropping a body drops the headers that describe it

`Content-Type`, `Content-Length`, `Content-Encoding`, `Content-Language` and `Transfer-Encoding` go
with the body. **Found by the 303 case observing the actual wire**: the method and the body were
already right, and `Content-Type: text/plain` was still riding along on a bodyless `GET` — a claim
about content that is not there. The pure test alone would not have caught it; the peer reflecting
what it received did.

### Location resolution — the forms handled, and the ones refused

```text
https://host/path   absolute            taken as-is
//host/path         scheme-relative     inherits the CURRENT scheme
/path               absolute path       inherits scheme, host and port
?q=1                query-only          keeps the whole path, replaces the query
path                relative path       replaces the last segment
ftp://host/f        foreign scheme      REFUSED (HC12.1)
#frag               fragment-only       REFUSED — nothing to fetch (HC12.1)
```

Scheme-relative inheriting the current scheme is what stops an `https://` page being walked onto
`http://` by omission. The result still goes through `parse_url`, so a redirect target faces exactly
the URL limits a caller-supplied one does — **a server cannot use a redirect to reach a URL this
client would have refused.**

### Origin comparison uses the EFFECTIVE port

`https://h/` and `https://h:443/` are one origin. Otherwise a redirect that only spelled the port
differently would strip credentials for no reason, and callers would learn to turn the stripping
off — which is how a safety default dies.

### A downgrade is refused before anything is dialled

The check runs on the parsed target, before DNS and before connect, so a downgraded request never
reaches the wire.

---

## 3. Evidence

| what | where | count |
| --- | --- | --- |
| policy defaults, status set, rewrite table, resolution, origin, errors | `stark-http-client/src/tests.stark` | 42 total (13 new, incl. HC12.1) |
| executed native lifecycle | `stark-http-client-consumer` under the gate | 22 cases (10 new) |

The consumer's ten redirect cases, all against live peers:

```text
  redirect: off by default, the 302 is returned as-is
  redirect: relative Location resolved and followed
  redirect: absolute Location followed
  redirect: a loop is detected and named
  redirect: the chain bound is enforced on distinct targets
  redirect: a 3xx with no Location is reported
  redirect: 303 converted POST to GET and dropped the body
  redirect: 307 preserved the method and replayed the body
  redirect: Authorization stripped when the origin changed
  redirect: https to http refused as a downgrade
```

**The credential case is the exit criterion, and it reads the wire rather than the policy flag.**
The cleartext peer redirects to the TLS peer — a genuinely different origin — and the `/echo` route
reflects what it received. Asserting `GET|-|-|` proves the `Authorization` header was absent on the
second request, not merely that a boolean was set.

**The bound and the loop are proved separately.** `/r-loop` revisits one target; `/r-hopN` walks an
ever-lengthening chain of *distinct* targets. A single test could not distinguish the count bound
from the loop detector, and the two errors exist because a caller raising the limit should fix one
and not the other.

**The 303/307 pair is what makes either assertion mean anything.** Same peer, same echo route,
opposite outcomes — one converts and drops, the other preserves and replays.

---

## 3a. HC12.1 — hardening from an external review (CD-369)

Three findings, all real. The first predates HC12 and was verified by exploit before being fixed.

| | |
| --- | --- |
| **P0 CRLF header injection** | `Header`'s fields and `HeaderMap.entries` are public, so the constructor's validation was bypassable and the serializer trusted it. A value of `safe\r\nInjected: yes` produced a genuine extra header line. **Fixed**: every header is revalidated at the serializer boundary, `SerializeError::InvalidHeader(name)` carries the name only. The regression test is the exploit. |
| **P1 query-only reference** | `/one/two?q=1` + `?page=2` resolved to `/one/?page=2` — a different resource. **Fixed.** |
| **P1 foreign absolute URI** | `ftp://other.test/f` fell through to the relative branch. **Fixed**: refused, along with fragment-only references. |
| **P1 duplicate `Location`** | was first-wins; now `get_singleton` and `AmbiguousLocation`. |

**Still open:** dot-segment (`.`/`..`) removal, and making `Header`/`HeaderMap.entries` private
behind validated accessors. Both belong in their own packets — the first as a bounded RFC 3986
resolver in `stark-url` rather than a second URL implementation inside the HTTP client, the second
because it is an API break.

## 4. Findings

| id | what | status |
| --- | --- | --- |
| **DEV-160** | STARK's borrow checker is place-granular (DEV-154) and accepts disjoint-field borrows in one call — `f(builder.url.as_str(), builder.headers)`. The generated projections take `&slot` / `&mut slot`, losing that granularity, so **rustc rejects the generated code** with `E0502`. A correct program refused by the backend. Worked around by moving fields into locals first. | OPEN |
| — | DEV-158 and DEV-157 were both already known and neither bit again here: the consumer's config builder was written as a literal and the redirect arms nested from the start. | avoided |

**DEV-160 is the same shape as DEV-158** — the slot abstraction is whole-value while STARK's
ownership model is place-granular — and it is the third defect in this family. Whatever fixes the
`Partial`/`Whole` transition should be scoped to look at projection granularity generally, not just
at field assignment.
