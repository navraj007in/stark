# HC1-EVIDENCE — `stark-url` absolute URL parsing

**Stage:** HC1 of `WP-HTTP-CLIENT-ROADMAP.md`.
**Commit:** `e54833a Implement HC1 stark-url parsing`.

## Implemented surface

`stark-url` now exposes:

```text
UrlScheme::{Http,Https}
Authority { host, port }
Url { scheme, authority, path, query, fragment }
Url::parse(&String)
parse_url(&str, &UrlLimits)
Url::effective_port()
Url::origin_form_target()
Url::host_header_value()
```

The existing request-target and percent/query APIs remain in place.

## Contract boundaries

HC1 is URL decomposition for the HTTP client, not full DNS or address validation.

- Accepted schemes are exactly lowercase `http` and `https`.
- User-info is rejected.
- Host must be non-empty.
- Hostname text is preserved; it is not DNS-validated in HC1.
- IPv4 literal text is preserved as host text; octet syntax is not revalidated in HC1.
- Bracketed IPv6 authorities are recognized by bracket structure; IPv6 segment syntax is not
  validated in HC1.
- Explicit ports are decimal and must fit `UInt16`.
- Path and query text are preserved for request-target construction.
- Fragment text is parsed but excluded from origin-form request targets.
- Empty path emits `/`.
- Host header generation omits explicit default ports and includes explicit non-default ports.
- Internationalized domain handling is not implemented; callers must pass already-suitable host
  text. DNS/IDNA policy belongs to HC3 or a later URL-policy stage.
- Percent escapes in authority/path/query are preserved by absolute URL parsing. HC1 does not
  normalize or validate percent escapes outside the existing request-target parser.

## Evidence

Local evidence at implementation time:

```text
stark-url:          stark check
stark-url:          stark test                 19 passed
stark-url:          stark fmt --check
stark-url-consumer: stark check
stark-url-consumer: stark run                  byte-exact stdout
stark-url-consumer: stark build --no-build-cache
stark-url-consumer: native artifact run        byte-exact stdout
```

The full first-party qualification script also passed after HC2 added `stark-net` to the matrix:

```text
python3 starkc/scripts/qualify-first-party-packages.py --stark starkc/target/debug/stark --repo-root .
```

## Remaining non-goals

- WHATWG URL behavior.
- IDNA.
- DNS validation.
- IPv6 literal semantic validation.
- URL normalization.
- Percent-decoding absolute URL components.
