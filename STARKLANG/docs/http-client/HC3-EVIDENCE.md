# HC3-EVIDENCE — DNS resolver package/API

**Stage:** HC3 of `WP-HTTP-CLIENT-ROADMAP.md`.
**Status:** implementation in progress.

## Frozen ABI

HC3 uses `WP-PKG-HOST-CAPABILITIES.md` Part E as the authority. This pass froze the previously
open DNS decisions:

- 22-byte fixed resolver records;
- family tags `4` and `6`;
- address bytes in network byte order;
- zero `scope_id` in DNS v0.1;
- resolver order preserved;
- duplicates preserved;
- canonical names excluded;
- empty result maps to `NotFound`;
- package maximum of 32 records / 704 bytes;
- DNS provider status codes `101..107`.

The `101..107` status range is an implementation constraint of the current provider registry:
status vocabularies are provider-wide, while `stark-net-native` now owns both `tcp` and `dns`.

## Implemented surface

`stark-net` adds:

```text
ResolvedAddress { address, port }
ResolveLimits { max_host_bytes, max_results }
DnsError
default_resolve_limits()
resolve(&String, UInt16, ResolveLimits)
```

Provider bindings:

```text
stark_dns_resolve_len
stark_dns_resolve_fill
```

## Boundaries

- Uses the OS host resolver.
- No DNS wire protocol.
- No cache.
- No custom timeout; HC4 owns timeout work.
- No canonical-name output.
- No IDNA normalization.
- Provider output never returns native pointers.
