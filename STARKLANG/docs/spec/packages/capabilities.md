# Package capability vocabulary

**Status:** normative package-format contract, vocabulary version 1 (2026-08-09).

Package manifests that declare host authority record `"capability_vocabulary": 1`. The root
package's `capabilities` array is an upper-bound envelope, not a list of effects inferred by the
publisher. `stark.lock` records the vocabulary version so capability names are never interpreted
without their format.

Vocabulary v1 is:

```text
filesystem-read
filesystem-write
environment-read
network-client
network-listen
clock
randomness
process-execution
native-code
```

Pure means the empty set; it is not a capability. A future split may interpret an older capability
as the union of its successors. A capability is never silently renamed or removed within a
vocabulary version.

## Migration from the pre-v1 implementation names

| Pre-v1 name | Vocabulary-v1 successor |
| --- | --- |
| `filesystem` | `filesystem-read` and `filesystem-write` |
| `process.args`, `process.env` | `environment-read` |
| `tcp`, `dns`, `tls` | `network-client`; listener interfaces use `network-listen` |
| `clock` | `clock` |
| `random` | `randomness` |

The first-party `stark-io` surface references both filesystem roles. Its read/query interfaces map
to `filesystem-read`; create, write, flush/sync, resize, remove, rename, copy, and directory mutation
interfaces map to `filesystem-write`. `stark-net` stream and DNS interfaces map to
`network-client`; listener bind/accept/close map to `network-listen`. TLS interfaces map to
`network-client`.

Capability roles do not identify a provider. Several providers may implement disjoint interfaces
under one role (TCP and TLS both implement `network-client`). Interface symbols remain globally
unique in a selected provider set, and provider resolution uses the pair of authorized role and
referenced interface.
