# ASCII / Percent / URL-Encoded Consolidation Evidence

Baseline branch: `develop`
Baseline SHA: `6e7fa959594691f56ff00ee9b1922e66d480cb93`
Working branch: `codex/urlencoded-consolidation-dev`

## Inventory

| Concern | Existing authorities/copies before | Intended/final authority |
| --- | --- | --- |
| ASCII classification/case | `stark-ascii`; private copies in `stark-http-core`, `stark-http-auth` | `stark-ascii` |
| Byte to ASCII `Char` | private ladders in `stark-percent`, `stark-query`, `stark-form`, `stark-url`, `stark-mime` | `stark-ascii::char_from_ascii` |
| Percent codec | `stark-percent`; independent `%HH` codec in `stark-url` | `stark-percent` |
| Query pairs | `stark-query`; independent URL query orchestration in `stark-url` | `stark-urlencoded` for generic query pairs; `stark-url` for URL-specific required-`=` query parameters and offsets |
| Form pairs | `stark-form` | `stark-urlencoded` form mode |
| URL structure | `stark-url` | `stark-url` |

## Package Counts

Before this packet, the develop baseline carried 61 package manifests, 31 toolchain-marked
libraries, and the `stark-get` application.

After this packet, the tree carries 59 package manifests, 30 toolchain-marked libraries, and the
`stark-get` application. Removed package identities: `stark-query`, `stark-query-consumer`,
`stark-form`, `stark-form-consumer`. Added package identities: `stark-urlencoded`,
`stark-urlencoded-consumer`.

## Surface Changes

`stark-ascii` adds `char_from_ascii(byte: UInt8) -> Option<Char>`.

`stark-percent` changes `decode` to take `&str` rather than `&String`, so a caller holding a
borrowed component no longer allocates an owned `String` to pass it. No new encode set was added:
`PathSegment` already encodes every byte outside the unreserved set, which is exactly the output
`stark-url::percent_encode_component` has always produced, so `stark-url` reuses it.

`stark-urlencoded` exposes shared `Pair`, `Limits`, `UrlEncodedError`, `parse_query`,
`serialize_query`, `parse_form`, `serialize_form`, and `default_limits`.

## Dependency Changes

New downward pure edges:

- `stark-percent -> stark-ascii::char_from_ascii`
- `stark-urlencoded -> stark-ascii, stark-percent`
- `stark-url -> stark-ascii, stark-percent`
- `stark-http-core -> stark-ascii`
- `stark-http-auth -> stark-ascii`

No affected package gains host capabilities. All packages in this packet remain pure.

## Characterization

Tests now cover:

- ASCII classification/case across all 256 byte values and byte-to-`Char` boundaries.
- Percent encode sets including the new component set, valid decoding, invalid hex, incomplete
  escapes, and literal plus preservation.
- Query empty input, missing `=`, empty names/values, repeated names, literal `+`, `%20`, `%2B`,
  invalid percent, and limits.
- Form empty input, `+` as space, `%2B` as plus, space-to-`+`, plus-to-`%2B`, multiple pairs, and
  limits.
- Both modes: only the first `=` separates, empty segments are preserved as pairs, each of the
  four limits is enforced separately, the limits measure decoded rather than escaped length, and
  serialize/parse round-trips.
- URL percent-escape error offsets where the offending hex digit is itself a `%` (`"%%41"`,
  `"%4%41"`, `"abc%%41"`).
- URL percent encoding/decoding, malformed offset preservation, UTF-8 validation, ordered query
  parameters, repeated parameters, and URL structure.

## Measured Results

Scoped package tests passed:

Counts are transitive: `stark test` runs the dependency packages' tests too.

- `stark-ascii`: 5 passed
- `stark-percent`: 8 passed
- `stark-urlencoded`: 23 passed (10 own test functions)
- `stark-url`: 33 passed
- `stark-http-core`: 35 passed
- `stark-http-auth`: 25 passed
- `stark-mime`: 16 passed

The first migration of `stark-query`/`stark-form` into `stark-urlencoded` carried the behaviour
but not all of the tests: 22 baseline test functions became 5. The properties that lost their
guard — first-`=`-wins, empty segments as pairs, per-limit enforcement, decoded-length
measurement, and round-tripping — are now pinned in `stark-urlencoded/src/tests.stark`.

Consumer execution passed:

- `stark-urlencoded-consumer`: `stark-urlencoded consumer ok`

First-party qualification passed with:

```bash
python3 starkc/scripts/qualify-first-party-packages.py \
  --stark starkc/target/debug/stark \
  --repo-root /private/tmp/stark-urlencoded-consolidation
```

The first attempt inside the sandbox failed at the `stark-net` echo-peer bind step with
`Operation not permitted`; the same command passed after escalation because the qualification gate
requires live local peers for resource lifecycle evidence.

External CI run IDs were not measured in this local packet.

## Residual Duplication

`stark-url` keeps URL-specific query orchestration because its query grammar requires `=` and maps
error offsets into `UrlError`; that is not the same semantic surface as generic query pairs.

`stark-url` keeps UTF-8 scalar validation after percent decoding because `stark-percent` owns only
generic byte decoding.

`stark-http-core` keeps HTTP header value, target, status, and reason validation because those are
HTTP protocol rules.

`stark-http-auth` keeps Basic/Bearer grammar and token policy because those are Authorization
syntax rules.

`stark-mime` keeps MIME whitespace and quoted-string policy because those are MIME grammar rules.

## Defect Found and Fixed Within the Packet

Delegating `stark-url`'s percent decoding to `stark-percent` moved the reported error offset from
the `%` that opened an escape to the offending hex digit. The first recovery scanned backwards for
the nearest `%`, which is wrong exactly when the invalid digit is itself a `%`: `"%%41"` reported
offset 1 and `"%4%41"` reported 2, where the baseline reported 0. The escape start is one or two
bytes behind the digit and is now computed as such. `stark-url`'s test for strict escapes fails
against the scanning version and passes against the replacement.

## Compiler Changes

None.
