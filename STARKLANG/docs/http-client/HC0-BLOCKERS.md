# HC0-BLOCKERS — what stands between here and a working HTTP client

**Stage:** HC0 of `WP-HTTP-CLIENT-ROADMAP.md`. **Date:** 2026-08-01.

HC0's acceptance requires compiler work and package work to be **separately identified**, because
they are scheduled differently: package work can start immediately, compiler work needs a gate-owned
proposal. Each entry below says which it is, which stage it blocks, and what would clear it.

---

## Compiler / language blockers

### C-1 — No TCP half-close (`shutdown`) — BLOCKS HC7, HC8
**Kind:** provider surface (native crate + ABI declaration), not language.
`stark-net/native` exports connect, read, write, close — and no `shutdown`. A client that signals
end-of-body by half-closing the write side cannot express it. Workable without, since HTTP/1.1 with
`Content-Length` or chunked framing does not require half-close, but any request whose body length is
unknown in advance does.
**Clears when:** `stark_tcp_stream_shutdown(HandleBorrowed, direction)` is added and qualified.
**Severity:** medium. HC8 can ship without it if D7's framing rules hold.

### C-2 — `HandleConsumed` + `HandleOut` in one call is unproven — BLOCKS HC9 ENTRY
**Kind:** provider ABI capability, possibly already supported.
Measured as neither proven nor forbidden. See `HC0-DECISIONS.md` D10 for the resolution rule.
**Clears when:** a probe declares such a function and it lowers, verifies and executes — or a
two-call fallback is designed.
**Severity:** high for TLS, zero for everything before it.

### C-3 — No DNS resolver — BLOCKS HC3; CONSTRAINS HC8
**Kind:** host capability (provider), specified by `WP-PKG-HOST-CAPABILITIES` Part E.
No resolver symbol exists anywhere. Until one does, connection is address-literal only (D9).
**Clears when:** Part E is written and the provider lands.
**Severity:** high for usability, low for protocol correctness — the stack can be proven against a
literal address.

### C-4 — `String + String` is not admitted — NO STAGE BLOCKED
**Kind:** language, and deliberate.
Arithmetic operators desugar to `Num`, which is compiler-known and primitives-only, so no package can
make `+` work on `String`. Recorded because the roadmap's own examples use it and would mislead an
implementer, not because anything is broken. Resolved by D4 (`push_str`).
**Clears when:** never, within this track. A concatenation operator is a language proposal.

## Package blockers

### P-1 — `stark-net` has no STARK package — CLEARED BY HC2
The directory contains `native/` only. There is no `src/lib.stark`, so nothing is importable.
HC2 starts from zero, not from a partial surface — worth stating because the roadmap's title,
"Complete `stark-net` TCP Client Surface", reads as though a package exists to complete.

Cleared by `56a78b4 Implement HC2 stark-net package`. Remaining timeout and shutdown limitations are
recorded in `HC2-EVIDENCE.md`.

### P-2 — `stark-url` has no absolute-URL parsing — CLEARED BY HC1
`parse_request_target` handles origin-form (path + query) only. There is no scheme, host, port,
userinfo or fragment. A client cannot decide where to connect from what the package returns today.
HC1 is the missing half of the package, not polish.

Cleared by `e54833a Implement HC1 stark-url parsing`. Validation boundaries are recorded in
`HC1-EVIDENCE.md`.

### P-3 — `stark-mime`, `stark-query`, `stark-form` are newly tested — WATCH, NOT BLOCKING
All three had **zero tests** until 2026-08-01 and now have 10, 11 and 11. Writing those tests
immediately found a real defect (`stark-form` emitted a literal `+` unescaped, so
`serialize` → `parse` corrupted `"1+2"` into `"1 2"`). The packages are now qualified in CI, but
their test suites are one day old and were written by reading the implementations. HC5–HC7 depend on
them.
**Mitigation:** treat a failure in these as a package defect until proven otherwise, and add cases
when HC5–HC7 exercise paths the current tests do not.

## Non-blockers worth recording

- **Provider synthesis and package testing work.** Both were blockers within the last week —
  `stark test` did no provider synthesis (fixed CD-300) and panicked on any package with a
  dependency (fixed CD-302). Neither constrains this track.
- **Two lowering defects that would have blocked this track were fixed on 2026-08-01**: `&v[i].field`
  on a `Vec` of non-`Copy` elements (DEV-132) and `&[T; N]` → `&[T]` coercion (DEV-133). Both are the
  shape a parser writes constantly. Had HC5–HC7 started before the package qualification that found
  them, both would have surfaced as internal compiler errors mid-implementation.

## Open risk, not a blocker

`layer_audit` reports **six** further reachable lowering refusals — constructs the checker accepts
that lowering rejects — and DEV-132/DEV-133 were two more of that class found by a different route.
Nothing systematically enumerates the class (`WP-LOWERING-COVERAGE-MATRIX.md` is filed for exactly
this). An HTTP parser is written from indexing, slicing and borrowing, which is where these defects
live.

**Expect to hit at least one during HC5–HC7.** The mitigation is procedural, and it is the one this
repository already uses: when lowering refuses a program the checker accepted, register a DEV before
repairing, and do not rewrite the package to route around it — a workaround conceals a valid source
shape and leaves the next author to rediscover it.
