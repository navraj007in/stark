# stark-url v0.1 Evidence

## Audit

- Package: `stark-url`
- Package path: `/Users/nexper/Documents/GitHub/stark/stark-url`
- Repository head at latest local qualification: `97b67270947e229610e9e211a69d3d922578f9ee`
- Toolchain: repository `starkc/target/debug/stark`
- Status: `IMPLEMENTATION COMPLETE — ASCII QUALIFICATION PASS`

## Scope

Qualified locally:

- RFC 3986-style component percent encoding for ASCII and UTF-8 input bytes.
- Strict `%HH` decoding.
- UTF-8 validation after percent decoding.
- Exact malformed escape offsets.
- ASCII output construction, including control bytes.
- HTTP origin-form path/query request targets.
- Ordered and repeated query parameters.
- Query encoding canonicality.
- Explicit input and query-parameter limits.

Known boundary:

- Percent-decoded non-ASCII UTF-8 is validated, then returns `PercentDecodedNonAsciiBlocked`.
  Current package source still lacks a stable Core API for constructing `String` from validated
  runtime UTF-8 byte vectors or Unicode scalar values.

## Commands

```bash
# from /Users/nexper/Documents/GitHub/stark/stark-url
../starkc/target/debug/stark check
# result: stark-url: OK

../starkc/target/debug/stark test
# result: 13 passed; 0 failed; 0 ignored; 61ms total

# from /Users/nexper/Documents/GitHub/stark/stark-url-consumer
../starkc/target/debug/stark check
# result: stark-url-consumer: OK

../starkc/target/debug/stark run
# result stdout: q=stark%20url&tag=compiler&tag=language

../starkc/target/debug/stark build
# result: blocked: native build does not yet support this program: method bytes on Str
# (a later C4.5e sub-slice)
```

## Test Counts

- Package tests: 13
- Percent encode groups: 3
- Percent decode groups: 5
- Request-target groups: 4
- Limit groups: 1
- Query encode groups: 2
- Cross-package consumer: check pass; interpreter run pass; native build blocked

## Blockers

No blocker for ASCII request-target handling in package check/test or interpreter consumer run.

`URL-UNICODE-001`: full percent-decoded non-ASCII construction remains blocked by missing
byte-vector/scalar-to-`String` Core construction.

`URL-NATIVE-001`: native build of `stark-url-consumer` is blocked by native lowering for
`str.bytes()`, used by `percent_encode_component`, `percent_decode`, and request-target parsing.

## Final Status

`IMPLEMENTATION COMPLETE — ASCII INTERPRETER QUALIFICATION PASS; NATIVE BLOCKED`
