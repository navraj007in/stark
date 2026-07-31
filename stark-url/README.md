# stark-url

Bounded, deterministic URL component handling for HTTP origin-form request targets.

## Status

v0.1 source and tests are present, with the API frozen in
`../STARKLANG/docs/packages/STARK-URL-v0.1-Codex-Implementation-Spec.md`. The package intentionally avoids the WHATWG
browser URL standard and supports only authority-free request targets such as `/health`,
`/users/123`, and `/search?q=stark&page=2`.

Local native qualification passes: `stark check` and `stark test` pass with 14 package tests
covering component encoding, strict malformed `%HH`, UTF-8 percent decoding, ASCII control
decoding, request-target path/query splitting, ordered repeated query parameters, canonical query
encoding, and exact error offsets. The cross-package consumer checks, runs under `stark run`,
builds natively, and the generated native binary runs.

## Scope

Included:

- RFC 3986-style component percent encoding.
- Strict `%HH` decoding.
- UTF-8 validation and Unicode scalar reconstruction after decoding.
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
