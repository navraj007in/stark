# stark-url

Bounded, deterministic URL component handling for HTTP origin-form request targets.

## Status

v0.1 source and tests are present, with the API frozen in
`../STARK-URL-v0.1-Codex-Implementation-Spec.md`. The package intentionally avoids the WHATWG
browser URL standard and supports only authority-free request targets such as `/health`,
`/users/123`, and `/search?q=stark&page=2`.

Local ASCII interpreter qualification passes: `stark check` and `stark test` pass with 13 package
tests covering component encoding, strict malformed `%HH`, ASCII control decoding, request-target
path/query splitting, ordered repeated query parameters, canonical query encoding, and exact error
offsets. The cross-package consumer checks and runs under `stark run`; native build is currently
blocked by native lowering for `str.bytes()`.

Known compiler/runtime blocker: percent-decoded non-ASCII UTF-8 is validated but currently returns
`PercentDecodedNonAsciiBlocked`, because package source cannot yet construct a `String` from a
validated runtime UTF-8 byte vector.

## Scope

Included:

- RFC 3986-style component percent encoding.
- Strict `%HH` decoding.
- UTF-8 validation after decoding.
- Path plus optional query parsing.
- Repeated query keys in deterministic order.
- Empty query values.
- Explicit input and query-parameter limits.
- Byte-offset errors.

Excluded:

- Absolute URLs, schemes, hosts, ports, DNS, IDNA, punycode, fragments, base resolution, routing,
  filesystem path conversion, HTTP parsing, and form encoding's `+`-as-space behavior.

## API

Primary entry points are:

- `percent_encode_component(input)`
- `percent_decode(input)`
- `parse_request_target(input)`
- `parse_request_target_with_limits(input, limits)`
- `encode_query(parameters)`

The query model is `Vec<QueryParameter>` rather than a map so repeated keys and input order are
preserved.

## Evidence

See `EVIDENCE.md` and `TEST-MATRIX.md`.
