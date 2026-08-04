# stark-url v0.1 Evidence

## Audit

- Package: `stark-url`
- Package path: `/Users/nexper/Documents/GitHub/stark/stark-url`
- Repository head at latest local qualification: working tree after `5d3e76b`
- Toolchain: repository `starkc/target/debug/stark`
- Status: `IMPLEMENTATION COMPLETE — LOCAL NATIVE QUALIFICATION PASS`

## Scope

Qualified locally:

- RFC 3986-style component percent encoding for ASCII and UTF-8 input bytes.
- Strict `%HH` decoding.
- UTF-8 validation and Unicode scalar reconstruction after percent decoding.
- Exact malformed escape offsets.
- Output construction for ASCII controls, BMP scalars, and supplementary-plane scalars.
- HTTP origin-form path/query request targets.
- Ordered and repeated query parameters.
- Query encoding canonicality.
- Explicit input and query-parameter limits.

## Commands

```bash
# from /Users/nexper/Documents/GitHub/stark/stark-url
../starkc/target/debug/stark check
# result: stark-url: OK

../starkc/target/debug/stark test
# result: 14 passed; 0 failed; 0 ignored; 41ms total

# from /Users/nexper/Documents/GitHub/stark/stark-url-consumer
../starkc/target/debug/stark check
# result: stark-url-consumer: OK

../starkc/target/debug/stark run
# result stdout: q=stark%20url&tag=compiler&tag=language&emoji=%F0%9F%98%80

../starkc/target/debug/stark build --no-build-cache
# result: Built stark-url-consumer [debug] -> target/stark/debug/stark-url-consumer

./target/stark/debug/stark-url-consumer
# result stdout: q=stark%20url&tag=compiler&tag=language&emoji=%F0%9F%98%80
```

## Test Counts

- Package tests: 14
- Percent encode groups: 3
- Percent decode groups: 5
- Request-target groups: 4
- Limit groups: 1
- Query encode groups: 2
- Cross-package consumer: check pass; interpreter run pass; native build pass; native binary run pass

## Blockers

None remaining for the `stark-url` check/test/run/build path exercised here.

Retired blockers during this qualification:

- `URL-UNICODE-001`: percent-decoded non-ASCII construction now validates UTF-8 and appends
  Unicode scalar values through `Char::from_u32`.
- `URL-NATIVE-001`: native build of `stark-url-consumer` now passes on macOS arm64.

## Final Status

`IMPLEMENTATION COMPLETE — LOCAL NATIVE QUALIFICATION PASS`
