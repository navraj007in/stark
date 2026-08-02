# WP-CRYPTO0 / HC9 — joint decision: TLS engine and native cryptographic foundation

**APPROVED 2026-08-03 (CD-361).** Joint HC9/CRYPTO0 decision. Backend selection is no longer part
of the HC9 estimate.

---

## 1. Ruling

> **Select `rustls` with `aws-lc-rs` as STARK's TLS and general native-cryptography foundation.
> Reject `native-tls` for the first-party TLS provider.**

The roadmap already required TLS and CRYPTO0 not to create independent cryptographic stacks. This
freezes that as one shared foundation:

```text
secure randomness · hashes · MACs · symmetric encryption · signatures · key exchange · TLS
```

Not every future crypto operation must be exposed directly from `aws-lc-rs`, but the architecture
does not accidentally commit STARK to unrelated crypto stacks.

## 2. Profiles

```text
Profile N — normal
  rustls -> aws-lc-rs -> aws-lc-sys

Profile F — FIPS-oriented
  rustls -> aws-lc-rs (fips feature) -> aws-lc-fips-sys -> validated AWS-LC-FIPS module
```

**Profile N and Profile F are qualified and reported SEPARATELY.**

## 3. Why not `native-tls`

`native-tls` is not one TLS implementation. It selects SChannel on Windows, Secure Transport on
macOS and OpenSSL on Linux. That reduces build size and integrates with platform certificate
stores, but for STARK it would mean **three** error-normalization surfaces, three sets of protocol
and certificate behaviour, three security-policy assumptions, three FIPS stories, weaker
reproducibility, harder cross-platform differential qualification, and no shared foundation with
CRYPTO0.

It is directly contrary to what the compiler track has spent its effort on — one rule every engine
satisfies by construction — and it multiplies the qualification burden CD-347/348 imposes, since
lifecycle evidence would be needed per platform TLS stack rather than once.

Permitted later as an **external, third-party alternative provider** under WP-EXTERNAL-PROVIDERS.
Not as the authoritative first-party implementation.

## 4. Root stores are a separate decision

```text
trust-anchor source  ≠  TLS implementation
```

Choosing rustls does not bundle one global CA store permanently. Root acquisition is policy:

```text
SystemRoots     BundledRoots     ExplicitRoots
```

* **HC9's controlled fixture uses `ExplicitRoots`** containing the test CA.
* **HC10** may implement `SystemRoots` through a root-loading dependency while certificate
  validation stays rustls-owned.

This is the point that defuses the strongest argument for `native-tls`: system roots can be used
without handing the TLS protocol itself to SChannel, Secure Transport or OpenSSL.

## 5. Versions — policy, not numbers

**Do not write "latest rustls" into any build.** Pin exact versions in the provider lockfile and in
qualification evidence.

Observed at decision time, and **verified against upstream documentation on 2026-08-03**:

| | observed |
| --- | --- |
| rustls | 0.23.43 |
| aws-lc-rs | 1.17.3 (released 2026-07-17) |
| FIPS certificate | **FIPS 140-3 certificate #4816**, AWS-LC-backed |

These are the versions *seen*, not the versions *pinned*. **You pin what you qualified**, so the
concrete pin comes from HC9's qualification output. The drift already observed — the ruling cited
0.23.42 and the documentation had moved to 0.23.43 within the same day — is itself the argument for
the policy.

Profile F's certificate applies only within the certificate's supported operating environments; the
module version, operating environment, build configuration and security policy must all match.

## 6. Build prerequisites — a named cost

Verified against `aws-lc-rs` 1.17.3 documentation:

| | Profile N | Profile F |
| --- | --- | --- |
| C/C++ compiler | **required** | required |
| CMake | *never* required | **required** |
| Go | *never* required | **required** |
| bindgen | *never* required (pre-generated bindings) | possibly, target-dependent |

**The consequence specific to STARK.** Providers are statically linked into the generated Cargo
workspace, so a C/C++ compiler becomes a requirement for **every user who builds a program that
uses TLS** — not merely for the provider's authors, as it would be in an ordinary Rust project. A
Windows user needs a working C toolchain to `stark build` an HTTPS client.

Accepted as a Profile N cost. The alternative — a lighter-weight crypto provider for TLS — forks
the foundation this decision exists to unify.

**Open, and recorded here so it is not rediscovered:** a provider manifest's `targets` field
declares triples the provider supports, but cannot currently express *toolchain* prerequisites. A
provider may therefore declare a target it cannot build on without extra tooling. That gap belongs
to WP-EXTERNAL-PROVIDERS.

## 7. Profile F is not a feature flag

Enabling the `fips` Cargo feature does **not** constitute a FIPS claim. Verified: two further steps
are required, and both are checkable, so they belong in Profile F's qualification criteria rather
than in prose.

```text
1. install the FIPS provider   default_fips_provider().install_default()
2. verify at runtime           ClientConfig::fips() / ServerConfig::fips()
```

The process-default provider assumes all uses of rustls go through it, so installation must be
deliberate rather than incidental.

## 8. Downstream consequence, recorded now

`stark-http-client::parse_http_url` currently refuses `https://` outright with
`UnsupportedScheme` — a deliberate choice, since silently downgrading to cleartext would be a
security defect. **HC10 turns that refusal into scheme dispatch.** Not a backend question, but it is
the visible edge of this decision in existing shipped code, and it is recorded here so it is not
rediscovered as a surprise.

## 9. Frozen

```text
TLS engine                    rustls
default crypto provider       aws-lc-rs, non-FIPS
Profile F provider            aws-lc-rs FIPS / aws-lc-fips-sys
native-tls                    REJECTED as first-party; permitted later as an external provider
root-store policy             separate from TLS-engine selection
versioning                    exact versions and checksums pinned at qualification
qualification                 Profile N and Profile F reported separately
```

## 10. Provenance

The version numbers, FIPS certificate number and build prerequisites in §5–§7 were fetched from
upstream documentation and checked on 2026-08-03 rather than carried over unverified. Two
corrections came out of that check: the rustls documentation had moved from 0.23.42 to 0.23.43, and
the FIPS activation requirement is a specific pair of steps rather than a general caution. Both are
recorded above.
